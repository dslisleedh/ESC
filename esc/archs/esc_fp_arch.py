import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from basicsr.utils.registry import ARCH_REGISTRY

from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional, Sequence, Literal


ATTN_TYPE = Literal['Naive', 'SDPA', 'Flex']
"""
Naive Self-Attention: 
    - Numerically stable
    - Choose this for train if you have enough time and GPUs
    - Training ESC with Naive Self-Attention: 33.46dB @Urban100x2

Flex Attention:
    - Fast and memory efficient
    - Choose this for train/test if you are using Linux OS
    - Training ESC with Flex Attention: 33.44dB @Urban100x2

SDPA with memory efficient kernel:
    - Memory efficient (not fast)
    - Choose this for train/test if you are using Windows OS
    - Training ESC with SDPA: 33.43dB @Urban100x2
"""


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]
    return bias_mod


def feat_to_win(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x, 'b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c',
        heads=heads, wh=window_size[0], ww=window_size[1], qkv=3
    )


def win_to_feat(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x, '(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)',
        h=h_div, w=w_div, wh=window_size[0], ww=window_size[1]
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if self.training:
                return F.layer_norm(x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
            else:
                return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class DecomposedConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int):
        super().__init__()
        self.pdim = pdim
        
        self.dynamic_kernel_size = 3
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 4, pdim * self.dynamic_kernel_size * self.dynamic_kernel_size, 1, 1, 0)
        )
    
    def forward(self, x: torch.Tensor, lk_channel: torch.Tensor, lk_spatial: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
        b, c = x1.shape[:2]
        
        # Dynamic Conv
        kernel = self.proj(x1)
        kernel = rearrange(
            kernel, 'b (c kh kw) 1 1 -> (b c) 1 kh kw', kh=3, kw=3
        )
        n_pad = (13 - self.dynamic_kernel_size) // 2
        kernel = F.pad(kernel, (n_pad, n_pad, n_pad, n_pad))
        
        x1 = F.conv2d(x1, lk_channel, padding=0)
        x1 = rearrange(x1, 'b c h w -> 1 (b c) h w')
        lk_spatial = lk_spatial.repeat(b, 1, 1, 1)
        x1 = F.conv2d(x1, kernel + lk_spatial, padding=13//2, groups=b*c)
        x1 = rearrange(x1, '1 (b c) h w -> b c h w', b=b, c=c)
        x = torch.cat([x1, x2], dim=1)
        return x
    
    def extra_repr(self):
        return f'pdim={self.pdim}'


class DecomposedConvolutionalAttentionWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int):
        super().__init__()
        self.pdim = pdim
        self.plk = DecomposedConvolutionalAttention(pdim)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_channel: torch.Tensor, lk_spatial: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_channel, lk_spatial)
        x = self.aggr(x)
        return x 


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim*exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(int(dim*exp_ratio), int(dim*exp_ratio), kernel_size, 1, kernel_size//2, groups=int(dim*exp_ratio))
        self.aggr = nn.Conv2d(int(dim*exp_ratio), dim, 1, 1, 0)

    def forward(self, x):
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
            self, dim: int, window_size: int, num_heads: int, 
            attn_func=None, attn_type: ATTN_TYPE = 'Flex'
        ):
        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        
        self.attn_type = attn_type
        self.attn_func = attn_func
        self.relative_position_bias = nn.Parameter(
            torch.randn(num_heads, (2*window_size[0]-1)*(2*window_size[1]-1)).to(torch.float32) * 0.001
        )
        if self.attn_type == 'Flex':
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)
        self.is_mobile = False 

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        # Transposed idxs of original Swin Transformer
        # But much easier to implement and the same relative position distance anyway
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        idxs = torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)
        return idxs
        
    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
    def to_mobile(self):
        bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
        self.rpe_bias = nn.Parameter(bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1]))
        
        del self.relative_position_bias
        del self.rpe_idxs
        
        self.is_mobile = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        h_div, w_div = x.shape[2] // self.window_size[0], x.shape[3] // self.window_size[1]
        
        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = feat_to_win(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.attn_type == 'Flex':
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        elif self.attn_type == 'SDPA':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                out = self.attn_func(q, k, v, attn_mask=bias, is_causal=False)
        elif self.attn_type == 'Naive':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            out = self.attn_func(q, k, v, bias)
        else:
            raise NotImplementedError(f'Attention type {self.attn_type} is not supported.')
        
        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out   

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class Block(nn.Module):
    def __init__(
            self, dim: int, pdim: int, conv_blocks: int, 
            window_size: int, num_heads: int, exp_ratio: int, 
            attn_func=None, attn_type: ATTN_TYPE = 'Flex'
        ):
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 1.5)
        self.ln_attn = LayerNorm(dim) 
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type)
        
        self.lns = nn.ModuleList([LayerNorm(dim) for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList([DecomposedConvolutionalAttentionWrapper(dim, pdim) for _ in range(conv_blocks)])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])
        
        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, lk_channel: torch.Tensor, lk_spatial: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        for ln, pconv, convffn in zip(self.lns, self.pconvs, self.convffns):
            x = x + pconv(convffn(ln(x)), lk_channel, lk_spatial)
        x = self.conv_out(self.ln_out(x))
        return x + skip


@ARCH_REGISTRY.register()
class ESCFP(nn.Module):
    def __init__(
        self, dim: int, pdim: int, kernel_size: int,
        n_blocks: int, conv_blocks: int, window_size: int, num_heads: int,
        upscaling_factor: int, exp_ratio: int = 2, attn_type: ATTN_TYPE = 'Flex',
    ):
        super().__init__()
        if attn_type == 'Naive':
            attn_func = attention
        elif attn_type == 'SDPA':
            attn_func = F.scaled_dot_product_attention
        elif attn_type == 'Flex':
            attn_func = torch.compile(flex_attention, dynamic=True)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported.')
        
        self.lk_channel = nn.Parameter(torch.randn(pdim, pdim, 1, 1))
        self.lk_spatial = nn.Parameter(torch.randn(pdim, 1, kernel_size, kernel_size))
        nn.init.orthogonal(self.lk_spatial)
        
        self.dim = dim
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        self.blocks = nn.ModuleList([
            Block(
                dim, pdim, conv_blocks, 
                window_size, num_heads, exp_ratio, 
                attn_func, attn_type
            ) for _ in range(n_blocks)
        ])
        self.ln_last = LayerNorm(dim)
        self.last = nn.Conv2d(dim, dim, 3, 1, 1)
        self.to_img = nn.Conv2d(dim, 3*upscaling_factor**2, 3, 1, 1)
        self.upscaling_factor = upscaling_factor

    @torch.no_grad()
    def load_state_dict(self, state_dict, strict = True, assign = False):
        to_img_k = state_dict.get('to_img.weight')
        to_img_b = state_dict.get('to_img.bias')
        sd_scale = int((to_img_k.shape[0] // 3)**0.5)
        if sd_scale != self.upscaling_factor:
            from basicsr.utils import get_root_logger
            from copy import deepcopy

            state_dict = deepcopy(state_dict)
            logger = get_root_logger()
            logger.info(
                f'Converting the SubPixelConvolution from {sd_scale}x to {self.upscaling_factor}x'
            )

            def interpolate_kernel(kernel, scale_in, scale_out):
                _, _, kh, kw = kernel.shape
                kernel = rearrange(kernel, '(rgb rh rw) cin kh kw -> (cin kh kw) rgb rh rw', rgb=3, rh=scale_in, rw=scale_in)
                kernel = F.interpolate(kernel, size=(scale_out, scale_out), mode='bilinear', align_corners=False)
                kernel = rearrange(kernel, '(cin kh kw) rgb rh rw -> (rgb rh rw) cin kh kw', kh=kh, kw=kw)
                return kernel

            def interpolate_bias(bias, scale_in, scale_out):
                bias = rearrange(bias, '(rgb rh rw) -> 1 rgb rh rw', rgb=3, rh=scale_in, rw=scale_in)
                bias = F.interpolate(bias, size=(scale_out, scale_out), mode='bilinear', align_corners=False)
                bias = rearrange(bias, '1 rgb rh rw -> (rgb rh rw)')
                return bias
            
            to_img_k = interpolate_kernel(to_img_k, sd_scale, self.upscaling_factor)
            to_img_b = interpolate_bias(to_img_b, sd_scale, self.upscaling_factor)
            state_dict['to_img.weight'] = to_img_k
            state_dict['to_img.bias'] = to_img_b

        return super().load_state_dict(state_dict, strict, assign)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        skip = feat
        for block in self.blocks:
            feat = block(feat, self.lk_channel, self.lk_spatial)
        feat = self.last(self.ln_last(feat)) + skip
        feat = self.to_img(feat)
        x = F.pixel_shuffle(feat, self.upscaling_factor) + F.interpolate(x, scale_factor=self.upscaling_factor, mode='bicubic')
        return x


if __name__== '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    import numpy as np
    from scripts.test_direct_metrics import test_direct_metrics
    
    test_size = 'HD'
    # test_size = 'FHD'
    # test_size = '4K'

    height = 720 if test_size == 'HD' else 1080 if test_size == 'FHD' else 2160
    width = 1280 if test_size == 'HD' else 1920 if test_size == 'FHD' else 3840
    upsampling_factor = 4
    batch_size = 1
    
    # FP
    model_kwargs = {
        'dim': 48,
        'pdim': 16,
        'kernel_size': 13, 
        'n_blocks': 5,
        'conv_blocks': 5,
        'window_size': 32,
        'num_heads': 3,
        'upscaling_factor': upsampling_factor,
        'exp_ratio': 1.25,
        'deployment': True,   # Comment this for FLOPs/Params calculation
    }
    shape = (batch_size, 3, height // upsampling_factor, width // upsampling_factor)
    model = ESCFP(**model_kwargs)
    # print(model)

    test_direct_metrics(model, shape, use_float16=False, n_repeat=100)
    
    # with torch.no_grad():
    #     x = torch.randn(shape)
    #     x = x.cuda()
    #     model = model.cuda()
    #     model = model.eval()
    #     flops = FlopCountAnalysis(model, x)
    #     print(f'FLOPs: {flops.total()/1e9:.2f} G')
    #     print(f'Params: {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000}K')
    