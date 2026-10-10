# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable

from torch import nn, Tensor

from .attention import Attention, CrossAttentionRope, PRopeFlashAttention
from ..dinov2.layers.layer_scale import LayerScale
from ..dinov2.layers.mlp import Mlp


class BlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool=False,
        rope=None
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()


    def forward(self, x: Tensor, xpos=None, attn_mask=None,
                past_key_values=None, use_cache: bool = False):
        """
        Args:
            x: (B, N, C) input tokens.
            xpos: RoPE position indices for the current tokens.
            attn_mask: optional additive attention mask (broadcasts over batch/heads).
            past_key_values: cached (k, v) tuple from previous streaming blocks
                (post-norm, post-RoPE); passed directly to FlashAttentionRope.
            use_cache: when True, return (output, new_kv) instead of just output.

        Returns:
            If use_cache is False: Tensor (B, N, C).
            If use_cache is True: (Tensor (B, N, C), (new_k, new_v)).
        """
        if use_cache:
            attn_out, new_kv = self.attn(
                self.norm1(x), xpos=xpos, attn_mask=attn_mask,
                past_key_values=past_key_values, use_cache=True,
            )
            x = x + self.ls1(attn_out)
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x, new_kv

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), xpos=xpos, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x


class PoseInjectBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = PRopeFlashAttention,
        ffn_layer: Callable[..., nn.Module] = None,
        qk_norm: bool = False,
        rope=None
    ) -> None:
        super().__init__()
        from ..dinov2.layers.mlp import Mlp as _Mlp
        if ffn_layer is None:
            ffn_layer = _Mlp
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, poses, H, W, patch_h, patch_w, K=None, connect=False, attn_mask=None) -> Tensor:
        from src.utils.geometry import se3_inverse
        extrinsics = se3_inverse(poses)

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), extrinsics, H, W, patch_h, patch_w, K=K, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if connect:
            return x + attn_residual_func(x) + ffn_residual_func(x)
        return attn_residual_func(x) + ffn_residual_func(x)


class CrossOnlyBlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        cross_attn_class: Callable[..., nn.Module] = CrossAttentionRope,
        ffn_layer: Callable[..., nn.Module] = None,
        init_values=None,
        qk_norm: bool = False,
        rope=None
    ) -> None:
        super().__init__()
        from ..dinov2.layers.mlp import Mlp as _Mlp
        if ffn_layer is None:
            ffn_layer = _Mlp
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_y = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            rope=rope,
            qk_norm=qk_norm
        )
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )

    def forward(self, x: Tensor, y: Tensor, xpos=None, ypos=None) -> Tensor:
        def cross_attn_residual_func(x: Tensor, y: Tensor) -> Tensor:
            return self.ls_y(self.cross_attn(self.norm2(x), y, y, qpos=xpos, kpos=ypos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm3(x)))

        y_ = self.norm_y(y)
        x = x + cross_attn_residual_func(x, y_)
        x = x + ffn_residual_func(x)
        return x
