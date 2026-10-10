# Pi3-derived portions: Copyright (c) 2025, the authors. All rights reserved.
# BSD 3-Clause; see licenses/Pi3-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md.
from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossOnlyBlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out


class SameCameraFrame0TransformerDecoder(nn.Module):
    """Five-layer decoder alternating view self-attn with same-camera frame0 cross-attn."""

    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()

        block_kwargs = dict(
            dim=dec_embed_dim,
            num_heads=dec_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            init_values=None,
            qk_norm=False,
            rope=rope,
        )
        self.layers = nn.ModuleList([
            BlockRope(
                **block_kwargs,
                ffn_layer=Mlp,
                attn_class=FlashAttentionRope,
            ),
            CrossOnlyBlockRope(
                **block_kwargs,
                ffn_layer=Mlp,
                cross_attn_class=FlashCrossAttentionRope,
            ),
            BlockRope(
                **block_kwargs,
                ffn_layer=Mlp,
                attn_class=FlashAttentionRope,
            ),
            CrossOnlyBlockRope(
                **block_kwargs,
                ffn_layer=Mlp,
                cross_attn_class=FlashCrossAttentionRope,
            ),
            BlockRope(
                **block_kwargs,
                ffn_layer=Mlp,
                attn_class=FlashAttentionRope,
            ),
        ])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def _same_camera_frame0(self, hidden, xpos, batch_size, num_frames, num_cameras):
        bsz = int(batch_size)
        num_frames = int(num_frames)
        num_cameras = int(num_cameras)
        if num_frames <= 0 or num_cameras <= 0:
            raise ValueError("num_frames and num_cameras must be positive")

        view_count = num_frames * num_cameras
        if hidden.shape[0] != bsz * view_count:
            raise ValueError(
                f"Expected hidden batch {bsz * view_count} for B={bsz}, "
                f"T={num_frames}, V={num_cameras}, got {hidden.shape[0]}"
            )

        _, token_count, dim = hidden.shape
        ref = hidden.reshape(bsz, num_frames, num_cameras, token_count, dim)[:, 0:1]
        ref = ref.expand(bsz, num_frames, num_cameras, token_count, dim)
        ref = ref.reshape(bsz * view_count, token_count, dim)

        if xpos is None:
            return ref, None

        pos_dim = xpos.shape[-1]
        ref_pos = xpos.reshape(bsz, num_frames, num_cameras, token_count, pos_dim)[:, 0:1]
        ref_pos = ref_pos.expand(bsz, num_frames, num_cameras, token_count, pos_dim)
        ref_pos = ref_pos.reshape(bsz * view_count, token_count, pos_dim)
        return ref, ref_pos

    def forward(self, hidden, xpos=None, batch_size=None, num_frames=None, num_cameras=None):
        if batch_size is None or num_frames is None or num_cameras is None:
            raise ValueError(
                "SameCameraFrame0TransformerDecoder requires batch_size, "
                "num_frames, and num_cameras."
            )

        hidden = self.projects(hidden)
        for layer in self.layers:
            if isinstance(layer, CrossOnlyBlockRope):
                ref, ref_pos = self._same_camera_frame0(
                    hidden, xpos, batch_size, num_frames, num_cameras
                )
                hidden = layer(hidden, ref, xpos=xpos, ypos=ref_pos)
            else:
                hidden = layer(hidden, xpos=xpos)

        out = self.linear_out(hidden)
        return out


class ContextOnlyTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        prenorm=False,
    ):
        super().__init__()
        if prenorm:
            self.pre_norm = nn.LayerNorm(in_dim)
        else:
            self.pre_norm = None

        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim)

        self.blocks = nn.ModuleList([
            CrossOnlyBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                init_values=None,
                qk_norm=False,
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, context, xpos=None, ypos=None):
        if self.pre_norm is not None:
            hidden = self.pre_norm(hidden)
            context = self.pre_norm(context)

        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden)
        return out
