import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from basicsr.utils.registry import ARCH_REGISTRY

from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional, Sequence, Literal


ATTN_TYPE = Literal['Naive', 'SDPA', 'Flex', 'FlashBias']
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

FlashBias (Flash Attention with low-rank decomposed relative position bias):
    - Fast and memory efficient
    - Choose this for test if you can't use Flex Attention
    - Training from-scatch with FlashBias is not recommended !!! Use pre-trained weights from Flex Attention
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


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.pdim = pdim
        self.lk_size = kernel_size
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            
            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)
            
            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=self.lk_size // 2) + x1_
            
            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(self.pdim, 1, self.sk_size, self.sk_size) 
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=self.lk_size // 2) \
                + F.conv2d(x[:, :self.pdim], dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=self.pdim)
            
            # For Mobile Conversion, uncomment the following code
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(x_1, dynamic_kernel, stride=1, padding=1, groups=16)
            # x = torch.cat([x_1, x_2], dim=1)
        return x
    
    def extra_repr(self):
        return f'pdim={self.pdim}'
    

class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, kernel_size)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x 


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim*exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(int(dim*exp_ratio), int(dim*exp_ratio), kernel_size, 1, kernel_size//2, groups=int(dim*exp_ratio))
        self.aggr = nn.Conv2d(int(dim*exp_ratio), dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
            self, dim: int, window_size: int, num_heads: int,
            attn_func=None, attn_type: str = 'Flex', flashbias_rank: Optional[int] = None
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
        
        if attn_type != 'FlashBias':
            self.relative_position_bias = nn.Parameter(
                torch.randn(num_heads, (2*window_size[0]-1)*(2*window_size[1]-1)).to(torch.float32) * 0.001
            )

        if self.attn_type == 'Flex':
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)

        self.flashbias_rank: int = 256 - (dim // num_heads) if flashbias_rank is None else flashbias_rank
        if self.attn_type == 'FlashBias':
            self.flashbias_q = nn.Parameter(
                torch.zeros(num_heads, window_size[0]*window_size[1], self.flashbias_rank)
            )
            self.flashbias_k = nn.Parameter(
                torch.zeros(num_heads, window_size[0]*window_size[1], self.flashbias_rank)
            )
        else:
            self.flashbias_q = None
            self.flashbias_k = None

        self.is_mobile = False

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B*nwin, heads, N, head_dim)

        if self.attn_type == 'Flex':
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)

        elif self.attn_type == 'SDPA':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1, self.num_heads,
                self.window_size[0]*self.window_size[1],
                self.window_size[0]*self.window_size[1]
            )
            out = self.attn_func(q, k, v, attn_mask=bias, is_causal=False)

        elif self.attn_type == 'Naive':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1, self.num_heads,
                self.window_size[0]*self.window_size[1],
                self.window_size[0]*self.window_size[1]
            )
            out = self.attn_func(q, k, v, bias)

        elif self.attn_type == 'FlashBias':
            Bwin = q.shape[0]
            heads = q.shape[1]
            N = q.shape[2]
            head_dim = q.shape[-1]

            q_bias = self.flashbias_q.to(dtype=q.dtype, device=q.device).unsqueeze(0).expand(Bwin, -1, -1, -1)
            k_bias = self.flashbias_k.to(dtype=k.dtype, device=k.device).unsqueeze(0).expand(Bwin, -1, -1, -1)

            softmax_scale = head_dim ** -0.5
            q_cat = torch.cat([q * softmax_scale, q_bias], dim=-1)
            k_cat = torch.cat([k, k_bias], dim=-1)
            v_cat = torch.cat([v, torch.zeros((Bwin, heads, N, k_bias.shape[-1]), device=v.device, dtype=v.dtype)], dim=-1)

            # Is this necessary? Just in case?
            d_total = q_cat.shape[-1]
            pad = (8 - (d_total % 8)) % 8
            if pad:
                z = torch.zeros((Bwin, heads, N, pad), device=q_cat.device, dtype=q_cat.dtype)
                q_cat = torch.cat([q_cat, z], dim=-1)
                k_cat = torch.cat([k_cat, z], dim=-1)
                v_cat = torch.cat([v_cat, z], dim=-1)
                
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = self.attn_func(
                    # BF16 for Flash Attention Kernel; not F16 for not to use grad scaler
                    q_cat.to(torch.bfloat16).contiguous(), k_cat.to(torch.bfloat16).contiguous(), v_cat.to(torch.bfloat16).contiguous(),
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0,  # Since q_cat is already scaled
                )[:, :, :, :head_dim]

        else:
            raise NotImplementedError(f'Attention type {self.attn_type} is not supported.')

        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, attn_type={self.attn_type}'


class Block(nn.Module):
    def __init__(
            self, dim: int, pdim: int, conv_blocks: int, 
            kernel_size: int, window_size: int, num_heads: int, exp_ratio: int, 
            attn_func=None, attn_type: ATTN_TYPE = 'Flex', use_ln: bool = False,
            flashbias_rank: Optional[int] = None
        ):
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 2)

        self.ln_attn = LayerNorm(dim) 
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type, flashbias_rank=flashbias_rank)
        
        self.lns = nn.ModuleList([LayerNorm(dim) if use_ln else nn.Identity() for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList([ConvAttnWrapper(dim, pdim, kernel_size) for _ in range(conv_blocks)])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])
        
        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, plk_filter: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        for ln, pconv, convffn in zip(self.lns, self.pconvs, self.convffns):
            x = x + pconv(convffn(ln(x)), plk_filter)
        x = self.conv_out(self.ln_out(x))
        return x + skip


# To enhance LK's structural inductive bias, we use Feature-level Geometric Re-parameterization
#  as proposed in https://github.com/dslisleedh/IGConv
def _geo_ensemble(k):
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
    return k


@ARCH_REGISTRY.register()
class ESC(nn.Module):
    def __init__(
        self, dim: int, pdim: int, kernel_size: int,
        n_blocks: int, conv_blocks: int, window_size: int, num_heads: int,
        upscaling_factor: int, exp_ratio: int = 2, attn_type: ATTN_TYPE = 'Flex',
        use_ln: bool = False, flashbias_rank: Optional[int] = None
    ):
        super().__init__()
        if attn_type == 'Naive':
            attn_func = attention
        elif attn_type == 'SDPA' or attn_type == 'FlashBias':
            attn_func = F.scaled_dot_product_attention
        elif attn_type == 'Flex':
            attn_func = torch.compile(flex_attention, dynamic=True)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported.')
            
        self.plk_func = _geo_ensemble
            
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        # Initializing LK filters using orthogonal initialization is important for stabilizing early training phase.
        torch.nn.init.orthogonal_(self.plk_filter)  
        
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        self.blocks = nn.ModuleList([
            Block(
                dim, pdim, conv_blocks, 
                kernel_size, window_size, num_heads, exp_ratio,
                attn_func, attn_type, use_ln=use_ln, flashbias_rank=flashbias_rank
            ) for _ in range(n_blocks)
        ])
        self.last = nn.Conv2d(dim, dim, 3, 1, 1)
        self.to_img = nn.Conv2d(dim, 3*upscaling_factor**2, 3, 1, 1)
        self.upscaling_factor = upscaling_factor
        
    @torch.no_grad()
    def convert(self):
        self.plk_filter = nn.Parameter(self.plk_func(self.plk_filter))
        self.plk_func = nn.Identity()

    @torch.no_grad()
    def load_state_dict(self, state_dict, strict = True, assign = False):
        # For SubPixel Interpolation
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
        
        # For RelPos Decomposition
        if self.blocks[0].attn.attn_type == 'FlashBias' and 'blocks.0.attn.relative_position_bias' in state_dict:
            # Decompose RPE table into FlashBias factors when loading weights
            from basicsr.utils import get_root_logger
            logger = get_root_logger()
            logger.info('Decomposing RPE table into FlashBias factors when loading weights...')
            capture_str = 'attn.relative_position_bias'
            for block_idx in range(len(self.blocks)):
                rpe_key = f'blocks.{block_idx}.{capture_str}'
                rpe_table = state_dict[rpe_key]
                num_heads, table_size = rpe_table.shape
                window_size = int(((table_size)**0.5 + 1) // 2)
                
                # recreate idxs
                rpe_idxs = WindowAttention.create_table_idxs(window_size, num_heads)
                N = window_size * window_size
                bias_vec = rpe_table[rpe_idxs[:, 0], rpe_idxs[:, 1]] 
                bias = bias_vec.view(num_heads, N, N).to(torch.float32)  

                U, S, Vh = torch.linalg.svd(bias, full_matrices=False) 
                r = self.blocks[block_idx].attn.flashbias_rank
                sr = torch.sqrt(S[:, :r]).unsqueeze(1) 
                q_bias = U[:, :, :r] * sr 
                k_bias = Vh[:, :r, :].transpose(-2, -1) * sr 

                state_dict[f'blocks.{block_idx}.attn.flashbias_q'] = q_bias.to(rpe_table.dtype)
                state_dict[f'blocks.{block_idx}.attn.flashbias_k'] = k_bias.to(rpe_table.dtype)
                
                del state_dict[rpe_key]

        return super().load_state_dict(state_dict, strict, assign)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        skip = feat
        plk_filter = self.plk_func(self.plk_filter)
        for block in self.blocks:
            feat = block(feat, plk_filter)
        feat = self.last(feat) + skip
        x = self.to_img(feat) + torch.repeat_interleave(x, self.upscaling_factor**2, dim=1)
        x = F.pixel_shuffle(x, self.upscaling_factor)
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
    upsampling_factor = 2
    batch_size = 1
    
    # Base
    model_kwargs = {
        'dim': 64,
        'pdim': 16,
        'kernel_size': 13, 
        'n_blocks': 5,
        'conv_blocks': 5,
        'window_size': 32,
        'num_heads': 4,
        'upscaling_factor': upsampling_factor,
        'exp_ratio': 1.25,
        'attn_type': 'Flex',  # Naive, SDPA, Flex, and FlashBias / For FLOPs calculation, use Naive
    }
    # Light
    # model_kwargs = {
    #     'dim': 64,
    #     'pdim': 16,
    #     'kernel_size': 13, 
    #     'n_blocks': 3,
    #     'conv_blocks': 5,
    #     'window_size': 32,
    #     'num_heads': 4,
    #     'upscaling_factor': upsampling_factor,
    #     'exp_ratio': 1.25,
    #     'attn_type': 'Flex',  # Naive, SDPA, Flex / For FLOPs calculation, use Naive
    # }
    shape = (batch_size, 3, height // upsampling_factor, width // upsampling_factor)
    model = ESC(**model_kwargs)
    print(model)
    

    test_direct_metrics(model, shape, use_float16=False, n_repeat=100)

    # with torch.no_grad():
    #     x = torch.randn(shape)
    #     x = x.cuda()
    #     model = model.cuda()
    #     model = model.eval()
    #     flops = FlopCountAnalysis(model, x)
    #     print(f'FLOPs: {flops.total()/1e9:.2f} G')
    #     print(f'Params: {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000}K')
    