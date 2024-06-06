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
# from ..utils import nlc_to_nchw,nchw_to_nlc


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat,reduce

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import timm
# from math import sqrt
# from functools import partial
# from torch import einsum
# from model.swintransformerv2 import SwinTransformerV2

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

    def __init__(self,  patch_size=4, in_chans=1, embed_dim=96,stride_size=4,padding=0, norm_layer=None):
        super().__init__()

        patch_size = to_2tuple(patch_size)


        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=padding,stride=stride_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.proj(x)
        out_size = (x.shape[2],x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x,out_size

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
            self.layers.append(nn.Sequential(
                PatchEmbed(
                in_chans=in_channels,
                embed_dim=embed_dims_i,
                patch_size=pathch_sizes[i],
                stride_size=strides[i],
                padding=pathch_sizes[i]//2 if overlap else 0
                ))
            )
            in_channels = embed_dims_i


    def forward(self,x):
        outs = []
        for downsample in self.layers:
            x,hw_shape = downsample(x)
            x = nlc_to_nchw(x,hw_shape)
            outs.append(x)
        return outs

class DsConv2d(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_size,padding,stride=1,bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in,dim_in,kernel_size=kernel_size,padding=padding,groups=dim_in,stride=stride,bias=bias),
            nn.Conv2d(dim_in,dim_out,kernel_size=1,bias=bias)
        )

    def forward(self,x):
        return self.net(x)



class LayerNorm(nn.Module):
    def __init__(self,dim,eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1,dim,1,1))
        self.b = nn.Parameter(torch.zeros(1,dim,1,1))

    def forward(self,x):
        std = torch.var(x,dim=1,unbiased = False,keepdim = True).sqrt()
        mean = torch.mean(x,dim = 1, keepdim = True)
        return (x-mean) / (std +self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self,x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(self,*,dim,heads,reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim,dim,1,bias=False)
        self.to_kv = nn.Conv2d(dim,dim*2,reduction_ratio,stride=reduction_ratio,bias=False)
        self.to_out = nn.Conv2d(dim,dim,1,bias=False)


    def forward(self,x):
        h,w = x.shape[-2:]
        heads = self.heads

        q,k,v = (self.to_q(x),*self.to_kv(x).chunk(2,dim = 1))
        q,k,v = map(lambda t: rearrange(t,'b (h c) x y -> (b h) (x y) c', h = heads),(q,k,v))

        sim = einsum('b i d,b j d -> b i j',q,k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j,b j d -> b i j',attn,v)
        out = rearrange(out,'(b h) (x y) c -> b (h c) x y',h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(self,*,dim,expansion_factor):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,1),
            DsConv2d(hidden_dim,hidden_dim,3,padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim,dim,1)
        )

    def forward(self,x):
        return self.net(x)


class MIT(nn.Module):
    def __init__(self,*,channels,dims,heads,ff_expansion,reduction_ratio,num_layers):
        super().__init__()
        stage_kernel_stride_pad = ((7,4,3),(3,2,1),(3,2,1),(3,2,1))

        dims = (channels,*dims)
        dim_pairs = list(zip(dims[:-1],dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in,dim_out),(kernel,stride,padding),num_layers,ff_expansion,heads,reduction_ratio in zip(dim_pairs,stage_kernel_stride_pad,num_layers,ff_expansion,heads,reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel,stride = stride,padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2,dim_out,1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out,EfficientSelfAttention(dim = dim_out,heads = heads,reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out,MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))
            
            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))


    def forward(self,x,return_layer_outputs):
        h,w = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches,overlap_embed,layers) in self.stages:
            x = get_overlap_patches(x)
            num_patches = x.shape[-1]
            ratio = int(sqrt((h*w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h =h //ratio)

            x = overlap_embed(x)
            for (attn,ff) in layers:
                x = attn(x) + x
                x = ff(x) + x 

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret



def cast_tuple(val,depth):
    return val if isinstance(val,tuple) else (val,) *depth

# class MultiheadAttention(nn.Module):
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  attn_drop=0,
#                  proj_drop=0,
#                  dropout_layer=dict(type='Dropout',drop_prob=0.),
#                  batch_first = False,
#                  **kwargs):
#         super(MultiheadAttention,self).__init__()

#         self.embed_dims  = embed_dims
#         self.num_heads = num_heads
#         self.batch_first = batch_first
#         self.attn = nn.MultiheadAttention(embed_dims,num_heads,attn_drop,**kwargs)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.dropout_layer = build_dropout()


# class DepthFusionModule(MultiheadAttention):



class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x,
                    2, (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(
            x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        # self.spatial = ConvModule(
        #     2,
        #     1,
        #     kernel_size,
        #     stride=1,
        #     padding=(kernel_size - 1) // 2,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU'))

        self.spatial = nn.Sequential(nn.Conv2d(2,1,kernel_size,padding=(kernel_size-1) // 2,),
                                     nn.ReLU())

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

class Scale(nn.Module):
    def __init__(self,scale=1.0):
        super(Scale,self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale,dtype=torch.float))

    def forward(self,x):
        return x * self.scale

class CBAM(nn.Module):

    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max'],
                 no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
                                       pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class DepthFusionModule1(CBAM):
    def __init__(self,embed_dims):
        super(DepthFusionModule1,self).__init__(embed_dims * 2)
        self.embed_dims = embed_dims
        self.gamma = Scale(0)

    def forward(self,color,depth):
        x = torch.cat([color,depth],dim=1)
        out = super(DepthFusionModule1,self).forward(x)[:,:self.embed_dims]
        out = self.gamma(out) + color
        return color

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name1='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 channels = 1,
                 embed_dims=64,
                 dims = (64,128,256,512),
                 heads = (1,2,4,8),
                 ff_expansion = (8,8,4,4),
                 reduction_ratio = (8,4,2,1),
                 num_layers = 2,
                 num_stages = 4,
                 freeze_stages = -1
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        dims,heads,ff_expansion,reduction_ratio,num_layers = map(partial(cast_tuple,depth=4),(dims,heads,ff_expansion,reduction_ratio,num_layers))

        self.backbone1 = timm.create_model(backbone_name1, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        self.mit = MIT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )


        self.depth = DepthDownSample(
            1,
            embed_dims=embed_dims,
        )


        self.num_stages = num_stages
        self.dims= dims
        self.num_heads = heads


        self.dfms  = nn.ModuleList([])
        for i in range(self.num_stages):
            embed_dims_i = self.embed_dims * self.num_heads[i]
            self.dfms.append(
                DepthFusionModule1(embed_dims_i)
            )


        encoder_channels = self.backbone1.feature_info.channels()

        # self.backbone2 = SwinTransformerV2(in_chans=1, embed_dim=64, frozen_stages=freeze_stages)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)


    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m,nn.Linear):
    #             trunc_normal_init()

    def forward(self, x):
        color = x[:, :3]
        depth = x[:, 3:]
        h, w = color.size()[-2:]
        # h_d, w_d = depth.size()[-2:]
        # h, w = x.size()[-2:]
        # h,w = x.size()[-2:]
        # res1, res2, res3, res4 = self.backbone1(x)
        #彩色分支
        res1_c, res2_c, res3_c, res4_c = self.backbone1(color)
        # layer_outputs = self.mit(depth,return_layer_outputs = True)

        # 深度分支
        depth_outs = self.depth(depth)

        res1_d, res2_d, res3_d, res4_d = self.backbone2(depth)
        


        #特征融合
        temp1 = self.dfms[0](res1_c,depth_outs[0])
        temp2 = self.dfms[1](res2_c,depth_outs[1])
        temp3 = self.dfms[2](res3_c,depth_outs[2])
        temp4 = self.dfms[3](res4_c,depth_outs[3])
        x = self.decoder(temp1,temp2,temp3,temp4,h,w)


        # x = self.decoder(res1_c+layer_outputs[0], res2_c+layer_outputs[1], res3_c+layer_outputs[2], res4_c+layer_outputs[3], h, w)
        # x = self.decoder(res1_c, res2_c, res3_c, res4_c, h, w)
        # x = self.decoder(res1_c+res1_d, res2_c+res2_d, res3_c+res3_d, res4_c+res4_d, h, w)
        # x2 = self.decoder(layer_outputs[0],layer_outputs[1],layer_outputs[2],layer_outputs[3],h,w)#后期融合
        # x2 = self.decoder(layer_outputs[0], layer_outputs[1], layer_outputs[2], layer_outputs[3], h, w)
        # return x1+x2#后期融合
        
        return x