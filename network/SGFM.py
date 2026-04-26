import torch
import torch.nn as nn
import torch.nn.functional as F
class SGFM(nn.Module):
    """
    Semantic-Guided Fusion Module (SGFM)

    This module modulates feature maps using semantic priors.

    Args:
        norm_nc (int): Number of channels of input feature map
        label_nc (int): Number of channels of segmap
        nhidden (int): Hidden dimension for MLP

    Input:
        x: feature map, shape [B, C, H, W]
        segmap: semantic map, shape [B, label_nc, H, W]

    Output:
        out: modulated feature map, shape [B, C, H, W]
    """
    def __init__(self, norm_nc, label_nc, nhidden=64):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)
        
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = self.bn(normalized * (1 + gamma)) + beta

        return out