import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numbers
from einops import rearrange

class HybridFFN(nn.Module):
    """
    Hybrid-Scale Feedforward Network (HSFN)

    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(HybridFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
    def forward(self, x, mode=None):
            x = self.project_in(x)
            
            if mode == '3x3':
                res3 = self.relu3(self.dwconv3x3(x))
                x1_3, x2_3 = res3.chunk(2, dim=1)
                x1_5, x2_5 = torch.zeros_like(x1_3), torch.zeros_like(x2_3)
            elif mode == '5x5':
                res5 = self.relu5(self.dwconv5x5(x))
                x1_5, x2_5 = res5.chunk(2, dim=1)
                x1_3, x2_3 = torch.zeros_like(x1_5), torch.zeros_like(x2_5)
            else: 
                res3 = self.relu3(self.dwconv3x3(x))
                res5 = self.relu5(self.dwconv5x5(x))
                x1_3, x2_3 = res3.chunk(2, dim=1)
                x1_5, x2_5 = res5.chunk(2, dim=1)
            
            x1 = torch.cat([x1_3, x1_5], dim=1)
            x2 = torch.cat([x2_3, x2_5], dim=1)

            if mode == '3x3':
                x1 = self.relu3_1(self.dwconv3x3_1(x1))
                x2 = torch.zeros_like(x1) 
            elif mode == '5x5':
                x2 = self.relu5_1(self.dwconv5x5_1(x2))
                x1 = torch.zeros_like(x2) 
            else:
                x1 = self.relu3_1(self.dwconv3x3_1(x1))
                x2 = self.relu5_1(self.dwconv5x5_1(x2))

            x = torch.cat([x1, x2], dim=1)
            x = self.project_out(x)
            return x

class SparseAttention(nn.Module):
    """
    Implements Sparse Self-Attention (SSA)

    """
    def __init__(self, dim, num_heads, bias):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        # Top-k ratios used in ablation study 
        # self.k_rates = [1/4,1/3,1/2,2/3]
        # self.k_rates = [1/3,1/2,2/3,3/4]
        self.k_rates = [1/2,2/3,3/4,4/5]
        # self.k_rates = [2/3,3/4,4/5,5/6]
        # self.k_rates = [1]
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        weights = [self.attn1, self.attn2, self.attn3, self.attn4]
        # weights = [1]
        out = 0
        for i, rate in enumerate(self.k_rates):

            k_val = max(1, int(C * rate)) 
            index = torch.topk(attn, k=k_val, dim=-1, largest=True)[1]
            mask = torch.zeros_like(attn).scatter_(-1, index, 1.)
            
            
            branch_attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            branch_attn = branch_attn.softmax(dim=-1)
            
            
            out += (branch_attn @ v) * weights[i]

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class SFEM(nn.Module):
    """
    Sparse Feature Extraction Module (SFEM)
    Includes:
        - Sparse Self-Attention (SSA)
        - Hybrid-Scale Feedforward Network (HSFN)
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SFEM, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SparseAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = HybridFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        self.encoder_level1 = nn.Sequential(*[
            SFEM(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  
        self.encoder_level2 = nn.Sequential(*[
            SFEM(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  
        self.encoder_level3 = nn.Sequential(*[
            SFEM(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  
        self.latent = nn.Sequential(*[
            SFEM(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])   
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)  

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        return latent, out_enc_level3, out_enc_level2, out_enc_level1
