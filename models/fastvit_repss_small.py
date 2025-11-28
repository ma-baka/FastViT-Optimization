#
# FastViT with RepSS-Mixer [Small Patch Version]
# 适用: 13x13 或 9x9 微型切片
# 特性: 
# 1. 集成 RepSS-Mixer 空谱分离模块
# 2. Stride Surgery: Stem步长=1, 前两阶段不下采样
#

import os
import copy
from functools import partial
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from models.modules.mobileone import MobileOneBlock
from models.modules.replknet import ReparamLargeKernelConv

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }

default_cfgs = {
    "fastvit_m": _cfg(crop_pct=0.95),
}

# =========================================================================
# 核心模块: RepSpectralConv & RepSSMixer (保持一致)
# =========================================================================
class RepSpectralConv(nn.Module):
    def __init__(self, dim, inference_mode=False):
        super(RepSpectralConv, self).__init__()
        self.dim = dim
        self.inference_mode = inference_mode
        if inference_mode:
            self.reparam_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
        else:
            self.branch3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            self.branch5 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
            self.bn3 = nn.BatchNorm1d(1)
            self.bn5 = nn.BatchNorm1d(1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, C)
        if hasattr(self, "reparam_conv"):
            out = self.reparam_conv(x_reshaped)
        else:
            out_3 = self.bn3(self.branch3(x_reshaped))
            out_5 = self.bn5(self.branch5(x_reshaped))
            out = x_reshaped + out_3 + out_5
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

    def reparameterize(self):
        if self.inference_mode: return
        w3, b3 = self._fuse_bn(self.branch3, self.bn3)
        w5, b5 = self._fuse_bn(self.branch5, self.bn5)
        w3_pad = F.pad(w3, (1, 1)) 
        w_id = torch.zeros_like(w5)
        w_id[:, :, 2] = 1.0
        w_final = w5 + w3_pad + w_id
        b_final = b5 + b3 
        self.reparam_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final
        for para in self.parameters(): para.detach_()
        del self.branch3, self.branch5, self.bn3, self.bn5

    def _fuse_bn(self, conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        beta = bn.weight
        gamma = bn.bias
        if conv.bias is not None: b = conv.bias
        else: b = torch.zeros_like(mean)
        w = w * (beta / var_sqrt).reshape(-1, 1, 1)
        b = (b - mean) * (beta / var_sqrt) + gamma
        return w, b

class RepSSMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, use_layer_scale=True, layer_scale_init_value=1e-5, inference_mode=False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode
        self.norm = MobileOneBlock(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, use_act=False, use_scale_branch=False, num_conv_branches=0)
        self.spatial_mixer = MobileOneBlock(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, use_act=False)
        self.spectral_mixer = RepSpectralConv(dim, inference_mode=inference_mode)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.spatial_mixer(x)
        x = self.spectral_mixer(x)
        if self.use_layer_scale:
            x = shortcut + self.layer_scale * (x - self.norm(shortcut)) 
        else:
            x = shortcut + x - self.norm(shortcut)
        return x

    def reparameterize(self) -> None:
        if self.inference_mode: return
        self.norm.reparameterize()
        self.spatial_mixer.reparameterize()
        self.spectral_mixer.reparameterize()

# =========================================================================
# 基础组件 (关键修改: Stem 支持 stride 参数)
# =========================================================================
def convolutional_stem(in_channels, out_channels, stride=2, inference_mode=False):
    print(f"[DEBUG-Small] Building Stem: in={in_channels}, out={out_channels}, stride={stride}")
    return nn.Sequential(
        MobileOneBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
        MobileOneBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
        MobileOneBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
    )

class MHSA(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4: x = torch.flatten(x, start_dim=2).transpose(-2, -1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4: x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_channels, embed_dim, inference_mode=False):
        super().__init__()
        block = list()
        block.append(ReparamLargeKernelConv(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=stride, groups=in_channels, small_kernel=3, inference_mode=inference_mode))
        block.append(MobileOneBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1))
        self.proj = nn.Sequential(*block)
    def forward(self, x): return self.proj(x)

class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3, groups=in_channels, bias=False))
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RepCPE(nn.Module):
    def __init__(self, in_channels, embed_dim=768, spatial_shape=(7, 7), inference_mode=False):
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int): spatial_shape = tuple([spatial_shape] * 2)
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        if inference_mode:
            self.reparam_conv = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.spatial_shape, stride=1, padding=int(self.spatial_shape[0] // 2), groups=self.embed_dim, bias=True)
        else:
            self.pe = nn.Conv2d(in_channels, embed_dim, spatial_shape, 1, int(spatial_shape[0] // 2), bias=True, groups=embed_dim)
    def forward(self, x):
        if hasattr(self, "reparam_conv"): return self.reparam_conv(x)
        return self.pe(x) + x
    def reparameterize(self):
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros((self.in_channels, input_dim, self.spatial_shape[0], self.spatial_shape[1]), dtype=self.pe.weight.dtype, device=self.pe.weight.device)
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, self.spatial_shape[0] // 2, self.spatial_shape[1] // 2] = 1
        id_tensor = kernel_value
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias
        self.reparam_conv = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.spatial_shape, stride=1, padding=int(self.spatial_shape[0] // 2), groups=self.embed_dim, bias=True)
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final
        for para in self.parameters(): para.detach_()
        self.__delattr__("pe")

class RepMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, inference_mode=False):
        super().__init__()
        self.token_mixer = RepSSMixer(dim, kernel_size, use_layer_scale, layer_scale_init_value, inference_mode)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_channels=dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_channels=dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x

def basic_blocks(dim, block_index, num_blocks, token_mixer_type, kernel_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, inference_mode=False) -> nn.Sequential:
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = drop_path_rate * (block_idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
        if token_mixer_type == "repmixer":
            blocks.append(RepMixerBlock(dim, kernel_size, mlp_ratio, act_layer, drop_rate, block_dpr, use_layer_scale, layer_scale_init_value, inference_mode))
        elif token_mixer_type == "attention":
            blocks.append(AttentionBlock(dim, mlp_ratio, act_layer, norm_layer, drop_rate, block_dpr, use_layer_scale, layer_scale_init_value))
        else:
            raise ValueError("Token mixer type: {} not supported".format(token_mixer_type))
    return nn.Sequential(*blocks)

# =========================================================================
# FastViT 主类 (支持 stem_stride)
# =========================================================================
class FastViT(nn.Module):
    def __init__(self, layers, token_mixers, embed_dims=None, mlp_ratios=None, downsamples=None, repmixer_kernel_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU, num_classes=1000, pos_embs=None, down_patch_size=7, down_stride=2, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, fork_feat=False, init_cfg=None, pretrained=None, cls_ratio=2.0, 
                 in_chans=3, stem_stride=2, inference_mode=False, **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        if pos_embs is None:
            pos_embs = [None] * len(layers)
        
        # [修改] 使用 stem_stride
        self.patch_embed = convolutional_stem(in_chans, embed_dims[0], stride=stem_stride, inference_mode=inference_mode)
        
        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode))
            stage = basic_blocks(embed_dims[i], i, layers, token_mixer_type=token_mixers[i], kernel_size=repmixer_kernel_size, mlp_ratio=mlp_ratios[i], act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, inference_mode=inference_mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            # [核心修改] 根据 downsamples 列表控制是否下采样
            # 如果 downsamples[i] 为 False，我们强制 stride=1 (不下采样)
            current_stride = down_stride if downsamples[i] else 1
            network.append(PatchEmbed(patch_size=down_patch_size, stride=current_stride, in_channels=embed_dims[i], embed_dim=embed_dims[i + 1], inference_mode=inference_mode))

        self.network = nn.ModuleList(network)
        
        # Head (省略了 fork_feat 的部分逻辑，专注分类)
        if self.fork_feat:
            pass # 简化
        else:
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.conv_exp = MobileOneBlock(in_channels=embed_dims[-1], out_channels=int(embed_dims[-1] * cls_ratio), kernel_size=3, stride=1, padding=1, groups=embed_dims[-1], inference_mode=inference_mode, use_se=True, num_conv_branches=1)
            self.head = nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x): return self.patch_embed(x)
    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x
    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        return cls_out

# =========================================================================
# Day 7 专用: 注册小切片模型
# =========================================================================
@register_model
def fastvit_ma36_small_patch(pretrained=False, in_chans=3, **kwargs):
    """
    FastViT-MA36 for Small Patches (13x13).
    配置: Stem Stride=1, Stage 1 & 2 下采样移除
    """
    layers = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4, 4, 4, 4]
    
    # [重点] 控制下采样节奏: 前两个阶段不缩小
    downsamples = [False, False, True, True] 
    
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    
    # 强制覆盖 in_chans
    if 'in_chans' in kwargs:
        in_chans = kwargs['in_chans']
    
    print(f"[DEBUG-Small] Creating fastvit_ma36_small_patch, in_chans={in_chans}, stem_stride=1")
    
    model = FastViT(
        layers, 
        embed_dims=embed_dims, 
        token_mixers=token_mixers, 
        pos_embs=pos_embs, 
        mlp_ratios=mlp_ratios, 
        downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        in_chans=in_chans,
        stem_stride=1, # [重点] Stem 不缩小
        **kwargs
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    return model