import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from torch.nn.attention.flex_attention import flex_attention

import numpy as np 
import tqdm


def attention(q, k, v, bias):
    score = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score, b, h, q_idx, kv_idx):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]
    return bias_mod


def feat_to_win(x, window_size, heads):
    return rearrange(
        x, 'b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c', heads=heads, wh=window_size[0], ww=window_size[1], qkv=3
    )
def win_to_feat(x, window_size, h_div, w_div):
    return rearrange(
        x, '(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)', h=h_div, w=w_div, wh=window_size[0], ww=window_size[1]
    )
    

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, attn_func=None, deployment=False):
        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        
        self.attn_func = attn_func
        self.is_deployment = deployment
        self.relative_position_bias = nn.Parameter(
            torch.randn(num_heads, (2*window_size[0]-1)*(2*window_size[1]-1)).to(torch.float32) * 0.001
        )
        if self.is_deployment:
            self.relative_position_bias = self.relative_position_bias.requires_grad_(False)
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)

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
        
    def pad_to_win(self, x, h, w):
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
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
        
        if self.is_deployment:
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        else:
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(1, self.num_heads, self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1])
            out = self.attn_func(q, k, v, bias)
        
        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out   

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--h', type=int, default=256)
    parser.add_argument('--w', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--attn_func', type=str, default='attention')
    args = parser.parse_args() 
    
    print('Testing ...')
    print(f'Arguments: {args}')
    
    if args.attn_func == 'attention':
        attn_func = attention
        is_deployment = False
    else:
        attn_func = torch.compile(flex_attention, dynamic=True)
        is_deployment = True
        
    model = WindowAttention(args.dim, args.window_size, args.heads, attn_func, deployment=is_deployment)
    model = model.cuda()
    model.eval()
    x = torch.randn(1, args.dim, args.h, args.w).cuda()
    n_repeat = 100
    
    with torch.inference_mode():
        print('warmup ...')
        for _ in tqdm.tqdm(range(100)):  
            model(x)  # Make sure CUDNN to find proper algorithms, especially for convolutions.
            torch.cuda.synchronize()

        print('testing ...')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((n_repeat, 1))
        
        for rep in tqdm.tqdm(range(n_repeat)):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    
    avg = np.sum(timings) / n_repeat
    med = np.median(timings)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('------------ Results ------------')
    print(f'Average time: {avg:.5f} ms')
    print(f'Median time: {med:.5f} ms') 
    print(f'Maximum GPU memory Occupancy: {torch.cuda.max_memory_allocated() / 1024**2:.5f} MB')
    print(f'Maximum GPU memory Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.5f} MB')
    print(f'Params: {params / 1000}K')  # For convenience and sanity check.
    print('---------------------------------')
