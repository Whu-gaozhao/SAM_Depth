import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat,reduce

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from math import sqrt
from functools import partial
from torch import einsum
from model.swintransformerv2 import SwinTransformerV2

def nlc_to_nchw(x,hw_shape):
    H,W = hw_shape
    assert len(x.shape) == 3
    B,L,C = x.shape
    assert L == H*W
    return x.transpose(1,2).reshape(B,C,H,W)

def nchw_to_nlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1,2).contiguous()

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=96,stride_size=4, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class DepthDownSample(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dims=64,
                 num_stages=4,
                 num_heads=[1,2,4,8],
                 strides=[4,2,2,2],
                 overlap=True,
                 norm_cfg=nn.LayerNorm):
        super(DepthDownSample,self).__init__()
        assert(in_channels==1)

        pathch_sizes = [7,3,3,3] if overlap else [4,2,2,2]
        self.num_heads = num_heads
        self.layers = nn.ModuleList([])
        for i in range(num_stages):
            embed_dims_i = embed_dims * self.num_heads[i]
            self.layers.append(nn.ModuleList([
                PatchEmbed(
                in_chans=in_channels,
                embed_dim=embed_dims_i,
                patch_size=pathch_sizes[i],
                stride_size=strides[i],
                )])
            )
            in_channels = embed_dims_i


    def forward(self,x):
        outs = []
        for downsample in self.layers:
            x,hw_shape = downsample(x)
            x = nlc_to_nchw(x,hw_shape)
            outs.append(x)
        return outs