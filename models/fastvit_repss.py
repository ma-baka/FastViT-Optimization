#
# FastViT with RepSS-Mixer (Reparameterizable Spatial-Spectral Mixer)
# Designed for Hyperspectral Image Classification
# [Pro Version] 标准化代码结构，集成 Day 4 架构创新
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

# 引入基础模块
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
# [核心创新 1] RepSpectralConv: 可重参数化的 1D 光谱卷积
# 物理意义: 在光谱维度上进行特征交互，捕捉光谱曲线的波峰波谷特征
# =========================================================================
class RepSpectralConv(nn.Module):
    """
    Reparameterizable Spectral Convolution (1D).
    
    [Training]: 多分支并行 (Identity + 3x1 Conv + 5x1 Conv)
    [Inference]: 单分支融合 (5x1 Conv)
    """
    def __init__(self, dim, inference_mode=False):
        super(RepSpectralConv, self).__init__()
        self.dim = dim
        self.inference_mode = inference_mode
        
        if inference_mode:
            # 推理态: 单一卷积核
            self.reparam_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
        else:
            # 训练态: 多分支增强特征提取能力
            # Branch 1: Kernel=3 (局部光谱特征)
            self.branch3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            # Branch 2: Kernel=5 (长程光谱特征)
            self.branch5 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
            
            # 1D BatchNorm (针对 Channel 维度)
            self.bn3 = nn.BatchNorm1d(1)
            self.bn5 = nn.BatchNorm1d(1)

    def forward(self, x):
        # Input x: (B, C, H, W) -> 这里的 C 是特征维度 (Feature Dim)
        B, C, H, W = x.shape
        
        # [Reshape]: 将每个空间像素视为一条独立的光谱序列
        # (B, C, H, W) -> (B, H, W, C) -> (N, 1, C), where N = B*H*W
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, C)
        
        if hasattr(self, "reparam_conv"):
            out = self.reparam_conv(x_reshaped)
        else:
            # Multi-branch forward
            out_3 = self.bn3(self.branch3(x_reshaped))
            out_5 = self.bn5(self.branch5(x_reshaped))
            # Identity + Branches
            out = x_reshaped + out_3 + out_5
            
        # [Reshape Back]: 还原回图像张量格式
        # (N, 1, C) -> (B, H, W, C) -> (B, C, H, W)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

    def reparameterize(self):
        """核心功能: 将多分支数学等价融合为一个卷积核"""
        if self.inference_mode: return

        # 1. 融合 BN 到 Conv
        w3, b3 = self._fuse_bn(self.branch3, self.bn3)
        w5, b5 = self._fuse_bn(self.branch5, self.bn5)
        
        # 2. Padding: 将 Kernel-3 (3) 填充为 Kernel-5 (5)
        w3_pad = F.pad(w3, (1, 1)) 
        
        # 3. 构造 Identity Kernel (中心为1，其余为0)
        w_id = torch.zeros_like(w5)
        w_id[:, :, 2] = 1.0
        
        # 4. 权重求和
        w_final = w5 + w3_pad + w_id
        b_final = b5 + b3 
        
        # 5. 创建推理层
        self.reparam_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True)
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final
        
        # 6. 删除旧分支
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


# =========================================================================
# [核心创新 2] RepSSMixer: 空谱分离混合器
# 替代原有的 RepMixer，实现 "Spatial -> Spectral" 串行解耦处理
# =========================================================================
class RepSSMixer(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        # 1. Pre-Norm
        self.norm = MobileOneBlock(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim,
            use_act=False, use_scale_branch=False, num_conv_branches=0,
        )
        
        # 2. Spatial Mixing (空间混合): 
        # 使用 2D Depthwise Conv，只在空间维度 (H, W) 混合，不跨通道
        self.spatial_mixer = MobileOneBlock(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim,
            use_act=False,
        )
        
        # 3. Spectral Mixing (光谱混合): 
        # 使用 RepSpectralConv，只在光谱维度 (C) 混合
        self.spectral_mixer = RepSpectralConv(dim, inference_mode=inference_mode)

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # 串行处理流程: Norm -> Spatial -> Spectral
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
# 基础组件 (PatchEmbed, Stem, MHSA, FFN, RepCPE)
# [注意] 这些是 FastViT 的基础设施，必须保留以防报错
# =========================================================================

def convolutional_stem(in_channels, out_channels, inference_mode=False):
    # [DEBUG] 确认输入通道数
    print(f"[DEBUG] Building Stem: in_channels={in_channels} -> out_channels={out_channels}")
    return nn.Sequential(
        MobileOneBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
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
        # [关键] 这里的 in_channels 必须是动态的
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

# =========================================================================
# [核心修改] RepMixerBlock: 调用新的 RepSSMixer
# =========================================================================
class RepMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, inference_mode=False):
        super().__init__()
        
        # [修改点] 使用 RepSSMixer 替代 RepMixer
        self.token_mixer = RepSSMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )
        
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
# FastViT 主架构
# =========================================================================
class FastViT(nn.Module):
    def __init__(self, layers, token_mixers, embed_dims=None, mlp_ratios=None, downsamples=None, repmixer_kernel_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU, num_classes=1000, pos_embs=None, down_patch_size=7, down_stride=2, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, fork_feat=False, init_cfg=None, pretrained=None, cls_ratio=2.0, in_chans=3, inference_mode=False, **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        if pos_embs is None:
            pos_embs = [None] * len(layers)
        
        # [DEBUG] 确认参数已传入
        print(f"[DEBUG] FastViT Init: in_chans={in_chans}")

        # [核心] 使用传入的 in_chans 初始化 Stem
        self.patch_embed = convolutional_stem(in_chans, embed_dims[0], inference_mode)
        
        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode))
            stage = basic_blocks(embed_dims[i], i, layers, token_mixer_type=token_mixers[i], kernel_size=repmixer_kernel_size, mlp_ratio=mlp_ratios[i], act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, inference_mode=inference_mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(PatchEmbed(patch_size=down_patch_size, stride=down_stride, in_channels=embed_dims[i], embed_dim=embed_dims[i + 1], inference_mode=inference_mode))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                self.add_module(f"norm{i_layer}", layer)
        else:
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.conv_exp = MobileOneBlock(in_channels=embed_dims[-1], out_channels=int(embed_dims[-1] * cls_ratio), kernel_size=3, stride=1, padding=1, groups=embed_dims[-1], inference_mode=inference_mode, use_se=True, num_conv_branches=1)
            self.head = nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _scrub_checkpoint(checkpoint, model):
        sterile_dict = {}
        for k1, v1 in checkpoint.items():
            if k1 not in model.state_dict():
                continue
            if v1.shape == model.state_dict()[k1].shape:
                sterile_dict[k1] = v1
        return sterile_dict

    def init_weights(self, pretrained=None):
        pass 

    def forward_embeddings(self, x):
        return self.patch_embed(x)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        return cls_out

@register_model
def fastvit_ma36(pretrained=False, in_chans=3, **kwargs):
    """
    Constructs a FastViT-MA36 model
    [修改] 显式接收 in_chans 参数，确保正确传递
    """
    layers = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    
    # [DEBUG] 打印确认调用参数
    print(f"[DEBUG] fastvit_ma36 called with in_chans={in_chans}")

    model = FastViT(
        layers, 
        embed_dims=embed_dims, 
        token_mixers=token_mixers, 
        pos_embs=pos_embs, 
        mlp_ratios=mlp_ratios, 
        downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        in_chans=in_chans, # 显式传递
        **kwargs
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    return model

