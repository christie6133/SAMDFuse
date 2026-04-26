"""
MSFS: Multi-Scale Fusion Strategy

Includes:
- SDFM: Shallow Detail Fusion Module
- DCFM: Deep Complementary Fusion Module (Self + Cross Attention)
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from layers import Block

class SDFM(nn.Module):
    """
    Shallow Detail Fusion Module (SDFM)
    Implementation of the channel-spatial recalibration strategy.
    """
    def __init__(self, channels=64, r=4):
        super(SDFM, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input 
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) 
        local_w = self.local_att(agg_input)  
        global_w = self.global_att(agg_input) 
        w = self.sigmoid(local_w * global_w) 
        xo = w * x1 + (1 - w) * x2 
        return xo

class Padding_tensor(nn.Module):
    def __init__(self, patch_size):
        super(Padding_tensor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        b, c, h, w = x.shape
        h_patches = int(np.ceil(h / self.patch_size))
        w_patches = int(np.ceil(w / self.patch_size))

        h_padding = np.abs(h - h_patches * self.patch_size)
        w_padding = np.abs(w - w_patches * self.patch_size)
        # # ======= 自动检查补零情况 ========
        # if h_padding > 0 or w_padding > 0:
        #     msg = f"\n 警告：图像尺寸不整除 patch_size={self.patch_size}，需要补零。\n" \
        #           f"→ 输入大小: ({h}, {w})\n" \
        #           f"→ 高度方向补偿: {h_padding} 像素\n" \
        #           f"→ 宽度方向补偿: {w_padding} 像素\n" \
        #           f" 请调整输入尺寸为 patch_size 的整数倍，避免特征错位或边缘误差。"
        #     raise RuntimeError(msg)
        reflection_padding = [0, w_padding, 0, h_padding]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x, [h_patches, w_patches, h_padding, w_padding]

class PatchEmbed_tensor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.padding_tensor = Padding_tensor(patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x, patches_paddings = self.padding_tensor(x)
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        # -------------------------------------------
        patch_matrix = None
        for i in range(h_patches):
            for j in range(w_patches):
                patch_one = x[:, :, i * self.patch_size: (i + 1) * self.patch_size,
                            j * self.patch_size: (j + 1) * self.patch_size]
                patch_one = patch_one.reshape(-1, c, 1, self.patch_size, self.patch_size)
                if i == 0 and j == 0:
                    patch_matrix = patch_one
                else:
                    patch_matrix = torch.cat((patch_matrix, patch_one), dim=2)
        
        return patch_matrix, patches_paddings
    
    
class Recons_tensor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, patches_tensor, patches_paddings):
        B, C, N, Ph, Pw = patches_tensor.shape
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        h_padding = patches_paddings[2]
        w_padding = patches_paddings[3]
        assert N == h_patches * w_patches, \
            f"The number of patches ({N}) doesn't match the Patched_embed operation ({h_patches}*{w_patches})."
        assert Ph == self.patch_size and Pw == self.patch_size, \
            f"The size of patch tensor ({Ph}*{Pw}) doesn't match the patched size ({self.patch_size}*{self.patch_size})."

        patches_tensor = patches_tensor.view(-1, C, N, self.patch_size, self.patch_size)
        # ----------------------------------------
        pic_all = None
        for i in range(h_patches):
            pic_c = None
            for j in range(w_patches):
                if j == 0:
                    pic_c = patches_tensor[:, :, i * w_patches + j, :, :]
                else:
                    pic_c = torch.cat((pic_c, patches_tensor[:, :, i * w_patches + j, :, :]), dim=3)
            if i == 0:
                pic_all = pic_c
            else:
                pic_all = torch.cat((pic_all, pic_c), dim=2)
        b, c, h, w = pic_all.shape
        pic_all = pic_all[:, :, 0:(h-h_padding), 0:(w-w_padding)]
        return pic_all

class self_atten_module(nn.Module):
    def __init__(self, embed_dim, num_p, depth, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p, cross=False)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x_in):
        x = x_in
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x_self = x

        return x_self

class cross_atten_module(nn.Module):
    def __init__(self, embed_dim, num_patches, depth, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.embed_dim,self.n_heads, self.mlp_ratio, self.qkv_bias, self.p, self.attn_p,self.depth =embed_dim,n_heads, mlp_ratio, qkv_bias, p, attn_p,depth
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                      cross=True)
                if i == 0 else
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                      cross=True)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x1_ori, x2_ori):
        x1 = x1_ori
        x2 = x2_ori
        x2 = self.pos_drop(x2)
        x = [x1, x2, x2]
        for block in self.blocks:
            x = block(x)
            x[2] = self.norm(x[2])
        x_self = x[2]

        return x_self

class self_atten(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches, depth_self, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_embed_tensor = PatchEmbed_tensor(patch_size)
        self.recons_tensor = Recons_tensor(patch_size)
        self.self_atten1 = self_atten_module(embed_dim, num_patches, depth_self,
                                              n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten2 = self_atten_module(embed_dim, num_patches, depth_self,
                                                   n_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, last=False):
        # patch
        x_patched1, patches_paddings = self.patch_embed_tensor(x1)
        # B, C, N, Ph, Pw = x_patched1.shape
        x_patched2, _ = self.patch_embed_tensor(x2)
        # B, C, N, Ph, Pw = x_patched1.shape
        b, c, n, h, w = x_patched1.shape
        # b, n, c*h*w
        x_patched1 = x_patched1.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x_patched2 = x_patched2.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x1_self_patch = self.self_atten1(x_patched1)
        x2_self_patch = self.self_atten2(x_patched2)
       
        # reconstruct
        if last is False:
            x1_self_patch = x1_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
            x_self1 = self.recons_tensor(x1_self_patch, patches_paddings)  # B, C, H, W
            x2_self_patch = x2_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
            x_self2 = self.recons_tensor(x2_self_patch, patches_paddings)  # B, C, H, W
        else:
            x_self1 = x1_self_patch
            x_self2 = x2_self_patch

        return x_self1, x_self2, patches_paddings


class cross_atten(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches, depth_self, depth_cross, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_embed_tensor = PatchEmbed_tensor(patch_size)
        self.recons_tensor = Recons_tensor(patch_size)
        self.embed_dim,self.depth_cross,self.n_heads, self.mlp_ratio, self.qkv_bias, self.p, self.attn_p =embed_dim,depth_cross,n_heads, mlp_ratio, qkv_bias, p, attn_p
        self.cross_atten1 = cross_atten_module(embed_dim, num_patches, depth_cross,
                                                     n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.cross_atten2 = cross_atten_module(embed_dim, num_patches, depth_cross,
                                                     n_heads, mlp_ratio, qkv_bias, p, attn_p)


    def forward(self, x1, x2, patches_paddings):
        # patch
        x_patched1, patches_paddings = self.patch_embed_tensor(x1)
        # B, C, N, Ph, Pw = x_patched1.shape
        x_patched2, _ = self.patch_embed_tensor(x2)
        # B, C, N, Ph, Pw = x_patched1.shape
        b, c, n, h, w = x_patched1.shape
        # b, n, c*h*w
        x1_self_patch = x_patched1.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x2_self_patch = x_patched2.transpose(2, 1).contiguous().view(b, n, c * h * w)
        
        x_in1 = x1_self_patch
        x_in2 = x2_self_patch
        cross1 = self.cross_atten1(x_in1, x_in2)
        cross2 = self.cross_atten2(x_in2, x_in1)
        out = cross1 + cross2
        
        # reconstruct
        x1_self_patch = x1_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        x_self1 = self.recons_tensor(x1_self_patch, patches_paddings)  # B, C, H, W
        x2_self_patch = x2_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        x_self2 = self.recons_tensor(x2_self_patch, patches_paddings)  # B, C, H, W
        
        cross1 = cross1.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        cross1_all = self.recons_tensor(cross1, patches_paddings)  # B, C, H, W
        
        cross2 = cross2.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        cross2_all = self.recons_tensor(cross2, patches_paddings)  # B, C, H, W
        
        out = out.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        out_all = self.recons_tensor(out, patches_paddings)  # B, C, H, W
        
        return out_all, x_self1, x_self2, cross1_all, cross2_all

class DCFM(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_patches, depth_self, depth_cross, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., use_self_attn=True): 
        super().__init__()
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.shift_size = int(img_size / 2)
        self.depth_cross = depth_cross
        self.use_self_attn = use_self_attn 

     
        self.self_atten_block1 = self_atten(self.patch_size, embed_dim, num_patches, depth_self,
                                              n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten_block2 = self_atten(self.patch_size, embed_dim, num_patches, depth_self,
                                                   n_heads, mlp_ratio, qkv_bias, p, attn_p)
        

        self.cross_atten_block = cross_atten(self.patch_size, embed_dim, self.num_patches, depth_self,
                                               depth_cross, n_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, shift_flag=True):

        paddings = None 

        # -------------------------------------
        # 1. Self-Attention 
        # -------------------------------------
        if self.use_self_attn:
            x1_atten, x2_atten, paddings = self.self_atten_block1(x1, x2)
            x1_a, x2_a = x1_atten, x2_atten 
            
            if shift_flag:
                shifted_x1 = torch.roll(x1_atten, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
                shifted_x2 = torch.roll(x2_atten, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
                x1_atten, x2_atten, _ = self.self_atten_block2(shifted_x1, shifted_x2)
                roll_x_self1 = torch.roll(x1_atten, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
                roll_x_self2 = torch.roll(x2_atten, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            else:
                x1_atten, x2_atten, _ = self.self_atten_block2(x1_atten, x2_atten)
                roll_x_self1, roll_x_self2 = x1_atten, x2_atten
        else:

            roll_x_self1, roll_x_self2 = x1, x2
            x1_a, x2_a = x1, x2

        # -------------------------------------
        # 2. Cross-Attention
        # -------------------------------------
        if self.depth_cross > 0:
            out, x_self1, x_self2, x_cross1, x_cross2 = self.cross_atten_block(roll_x_self1, roll_x_self2, paddings)
        else:
            out = roll_x_self1 + roll_x_self2
            x_self1, x_self2, x_cross1, x_cross2 = roll_x_self1, roll_x_self2, roll_x_self1, roll_x_self2
            
        return out, x1_a, x2_a, roll_x_self1, roll_x_self2, x_cross1, x_cross2   