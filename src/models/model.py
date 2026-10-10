"""Drive3R model.

A feed-forward, multi-camera 3D reconstruction network for driving scenes.

Key features
------------
- ``use_multimodal`` enables the ray (intrinsics) + ego_to_cam (extrinsics)
  conditioning branch on the decoder.
- ``enable_intrinsics_injection`` / ``enable_extrinsics_injection`` are
  deterministic switches that gate the two injection modalities.
- ``ckpt`` points to complete Drive3R inference weights (``.pt`` or
  ``.safetensors``). All parameters required by the selected configuration
  must be present with matching shapes. Constructing without ``ckpt``
  requires loading complete weights before inference.
"""

# Pi3-derived portions: Copyright (c) 2025, the authors. All rights reserved.
# BSD 3-Clause; see licenses/Pi3-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md.
import torch
import torch.nn as nn
from typing import Optional
from functools import partial

from .dinov2.layers import Mlp
from .dinov2.layers.patch_embed import PatchEmbed
from ..utils.checkpoint import load_model_weights
from ..utils.geometry import se3_inverse, get_pixel
from .layers.pos_embed import RoPE2D, PositionGetter, RotaryPositionEmbedding3D, PositionGetter3D
from .layers.block import BlockRope, PoseInjectBlock
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import (
    TransformerDecoder,
    ContextOnlyTransformerDecoder,
    SameCameraFrame0TransformerDecoder,
)
from .layers.camera_head import CameraHead, EgoDecoder, EgoHead
from .layers.conv_head import ConvHead
from .dinov2.hub.backbones import dinov2_vitl14_reg


class Drive3R(nn.Module):
    def __init__(
        self,
        use_multimodal=True,
        enable_intrinsics_injection: bool = True,
        enable_extrinsics_injection: bool = False,
        ckpt=None,
        cross_attn_mode='cross',
        use_ego_pose=False,
        temporal_dim_ratio: float = 0.25,
        temporal_spacing: float = 1.0,
        global_use_3d_rope: bool = False,
        block_size: int = 0,
        enable_ego_conf_head: bool = True,
        enable_cam_conf_head: Optional[bool] = None,
    ):
        """
        cross_attn_mode controls the decoder attention pattern:
          'global'  — interleaved self / global-cross: self → all-views → self → all-views → ...
                      (i%2==0: self-attn on each image; i%2==1: global cross-image attn)
          'cross'   — cross-shaped: self → cross-view → self → cross-frame → ...
                      (i%4==0,2: self; i%4==1: same-frame diff-camera; i%4==3: same-camera diff-frame)

        use_ego_pose: if True, adds EgoDecoder + EgoHead that predicts per-frame
          vehicle ego poses (B, T, 4, 4) via dedicated self/cross-view attention.

        temporal_dim_ratio: fraction of head_dim dedicated to the temporal axis in
          the 3D RoPE used for cross-frame attention layers (only when
          cross_attn_mode='cross').  Default 0.25.
        temporal_spacing: scale factor applied to frame indices before RoPE lookup
          (VideoRoPE ATS).  Default 1.0 (no scaling).
        global_use_3d_rope: whether to use 3D RoPE on odd global cross-image
          layers when cross_attn_mode='global'. Default False to keep the
          old global path on 2D RoPE.
        """
        super().__init__()

        self.use_multimodal = use_multimodal
        self.enable_intrinsics_injection = bool(enable_intrinsics_injection)
        self.enable_extrinsics_injection = bool(enable_extrinsics_injection)
        assert cross_attn_mode in ('global', 'cross'), \
            f"cross_attn_mode must be 'global' or 'cross', got '{cross_attn_mode}'"
        self.cross_attn_mode = cross_attn_mode
        self.use_ego_pose = use_ego_pose
        self.global_use_3d_rope = bool(global_use_3d_rope)
        self.block_size = int(block_size)
        if enable_cam_conf_head is None:
            conf_flag = bool(enable_ego_conf_head)
        else:
            conf_flag = bool(enable_cam_conf_head)
        # Backward-compatible alias: old configs/checkpoints may still use this key.
        self.enable_ego_conf_head = conf_flag
        self.enable_cam_conf_head = conf_flag

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #   Positional Encoding
        # ----------------------
        freq = 100
        self.rope = RoPE2D(freq=freq)
        self.position_getter = PositionGetter()

        # 3D spatio-temporal RoPE for cross-image / cross-frame attention layers.
        # align_spatial_freq=True: y/x frequencies use head_dim as denominator
        #   so they match the pretrained 2D RoPE.
        # Preserve the released inference setting: temporal_alpha=0.0.
        self.rope_3d = RotaryPositionEmbedding3D(
            frequency=freq,
            temporal_dim_ratio=temporal_dim_ratio,
            temporal_spacing=temporal_spacing,
            align_spatial_freq=True,
            temporal_alpha=0.0,
        )
        self.position_getter_3d = PositionGetter3D()

        # ----------------------
        #        Decoder
        # ----------------------
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36

        def _make_block(rope):
            return BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=rope,
            )

        # Layers that do cross-image/cross-frame attention use 3D RoPE (t,y,x);
        # all other layers (self-attention) keep the pretrained 2D RoPE (y,x).
        #   'global': i%2==1 (global cross-image)  -> rope_3d (optional)
        #   'cross':  i%4==3 (cross-frame)         -> rope_3d
        def _use_3d(i):
            if cross_attn_mode == 'global':
                return self.global_use_3d_rope and (i % 2 == 1)
            return i % 4 == 3

        self.decoder = nn.ModuleList([
            _make_block(self.rope_3d if _use_3d(i) else self.rope)
            for i in range(dec_depth)
        ])
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # -----------------------
        #       Multi-modal
        # -----------------------
        if use_multimodal:
            self.ray_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            nn.init.constant_(self.ray_embed.proj.weight, 0)
            nn.init.constant_(self.ray_embed.proj.bias, 0)

            self.pose_inject_blk = nn.ModuleList([PoseInjectBlock(
                dim=1024,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                init_values=0.01,
                qk_norm=True,
            ) for _ in range(5)])

        # ------------------------------
        #           Heads
        # ------------------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = ConvHead(
            num_features=4,
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=[2, 1],
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
        )
        self.camera_head = CameraHead(dim=512)

        if use_ego_pose:
            self.ego_decoder = EgoDecoder(
                in_dim=2 * self.dec_embed_dim,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=512,
                rope=self.rope,
            )
            self.ego_head = EgoHead(dim=512)

        self.metric_token = nn.Parameter(torch.randn(1, 1, 2 * self.dec_embed_dim))
        self.metric_decoder = ContextOnlyTransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=512,
            dec_num_heads=8,
            out_dim=512,
            rope=self.rope,
        )
        self.metric_head = nn.Linear(512, 1)
        nn.init.normal_(self.metric_token, std=1e-6)

        # ------------------------------
        #    Camera Block Confidence Head
        # ------------------------------
        # Predicts one scalar logit per temporal block (block_size frames) describing
        # whether that block has drifted relative to frame 0.
        # Only built when all of these are satisfied:
        #   - enable_{ego/cam}_conf_head=True
        #   - cross_attn_mode == 'global'
        #   - block_size > 0
        # (T>1 and V>1 are additionally checked at forward time.)
        self._ego_conf_enabled = (
            self.enable_cam_conf_head
            and self.cross_attn_mode == 'global'
            and self.block_size > 0
        )
        if self._ego_conf_enabled:
            self.ego_conf_decoder = SameCameraFrame0TransformerDecoder(
                in_dim=2 * self.dec_embed_dim,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.ego_conf_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1),
            )

        # ----------------------
        #   Load full checkpoint
        # ----------------------
        if ckpt is not None:
            tensor_count, result = load_model_weights(self, ckpt)
            print(f'[Drive3R] Loaded {tensor_count} tensors from {ckpt}.')
            if result.unexpected_keys:
                print(f'  ignored extra keys (first 5): {result.unexpected_keys[:5]}')

        print(
            '[Drive3R] Injection switches: '
            f'intrinsics={self.enable_intrinsics_injection}, '
            f'extrinsics={self.enable_extrinsics_injection}'
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        imgs,
        order_flow=None,
        intrinsics=None,
        cam_to_ego=None,
        num_frames=None,
        num_cameras=None,
        **kwargs
    ):
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        T, V = N, 1
        head_T, head_V = T, V
        if num_frames is not None and num_cameras is not None:
            _t = int(num_frames)
            _v = int(num_cameras)
            if _t > 0 and _v > 0 and _t * _v == N:
                T, V = _t, _v
                head_T, head_V = _t, _v

        hidden, decoder_cam_to_ego_poses, ego_decoder_cam_to_ego_poses = self.encode(
            imgs, intrinsics=intrinsics, cam_to_ego=cam_to_ego, **kwargs
        )
        hidden = hidden.reshape(B, N, -1, self.dec_embed_dim)

        hidden, pos, ref_idxs = self.decode(hidden, N, H, W, decoder_cam_to_ego_poses, T=T, V=V)

        outputs = self.forward_head(
            hidden,
            pos,
            order_flow,
            B,
            N,
            H,
            W,
            patch_h,
            patch_w,
            T=head_T,
            V=head_V,
            cam_to_ego_poses=ego_decoder_cam_to_ego_poses,
        )
        outputs.update(dict(
            order_flow=order_flow,
            ref_idxs=ref_idxs,
        ))
        return outputs

    @staticmethod
    def _build_global_block_causal_mask(
        num_images: int,
        hw: int,
        num_cameras: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build additive mask for global attention with shape (1, 1, L, L), L=num_images*hw.
        Rule:
          - Same block: bidirectional (unmasked)
          - Future block: masked (key block > query block)
        """
        if block_size <= 0:
            raise ValueError("block_size must be > 0 when building block mask")
        if num_images <= 0 or hw <= 0:
            raise ValueError("num_images and hw must be positive")
        if num_cameras <= 0:
            raise ValueError("num_cameras must be positive")

        total_tokens = num_images * hw
        token_idx = torch.arange(total_tokens, device=device, dtype=torch.long)
        image_id = token_idx // hw
        frame_id = image_id // num_cameras
        block_id = frame_id // block_size

        # mask[q, k] = -inf if key token is from a future temporal block.
        future_block = block_id.unsqueeze(0) > block_id.unsqueeze(1)
        mask = torch.zeros((total_tokens, total_tokens), device=device, dtype=dtype)
        mask = mask.masked_fill(future_block, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(self, imgs, intrinsics=None, cam_to_ego=None, **kwargs):
        B, N, _, H, W = imgs.shape
        device = imgs.device

        imgs_flat = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs_flat)["x_norm_patchtokens"]

        if not self.use_multimodal:
            return hidden, None, cam_to_ego

        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self.enable_intrinsics_injection and intrinsics is not None:
                pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(device).float()[None].repeat(B, 1, 1, 1)
                rays = torch.einsum('bnij, bhwj -> bnhwi', torch.inverse(intrinsics), pix)[..., :2]
            else:
                rays = torch.zeros((B, N, H, W, 2), device=device)

        ray_emb = self.ray_embed(rays.reshape(B * N, H, W, 2).permute(0, 3, 1, 2))
        hidden = hidden + ray_emb

        decoder_poses = cam_to_ego if self.enable_extrinsics_injection else None
        ego_decoder_poses = cam_to_ego
        return hidden, decoder_poses, ego_decoder_poses

    def decode(self, hidden, N, H, W, poses, T=None, V=None):
        """
        Decoder with two selectable attention patterns (self.cross_attn_mode):

        'global' — interleaved self / global-cross (i % 2):
          i % 2 == 0  →  self-attention   (B*N, hw, D)
          i % 2 == 1  →  global cross-image attn  (B, N*hw, D)

        'cross' — cross-shaped (i % 4):
          i % 4 == 0  →  self-attention          (B*N,   hw,   D)
          i % 4 == 1  →  cross-view attention    (B*T, V*hw,   D)  same frame, diff cameras
          i % 4 == 2  →  self-attention          (B*N,   hw,   D)
          i % 4 == 3  →  cross-frame attention   (B*V, T*hw,   D)  same camera, diff frames

        poses: cam_to_ego (B, N, 4, 4) for PoseInjectBlock. Internally inversed
               to ego_to_cam as viewmats for ProPE.
        """
        device = hidden.device
        if len(hidden.shape) == 4:
            B, N_, hw, _ = hidden.shape
        else:
            B_N, hw, _ = hidden.shape
            B = B_N // N

        if T is None:
            T = N
        if V is None:
            V = 1
        if T <= 0 or V <= 0 or T * V != N:
            T, V = N, 1

        hidden = hidden.reshape(B * N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        ref_idxs = None
        hw = hidden.shape[1]
        pose_inject_blk_idx = 0
        global_block_attn_mask = None
        if self.cross_attn_mode == 'global' and self.block_size > 0:
            global_block_attn_mask = self._build_global_block_causal_mask(
                num_images=N,
                hw=hw,
                num_cameras=V,
                block_size=self.block_size,
                device=device,
                dtype=hidden.dtype,
            )

        pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, device)
        if self.patch_start_idx > 0:
            pos_patch = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos_patch], dim=1)

        # Pre-build 3D positions (t, y, x) for layers that use rope_3d.
        pos_3d_cf = None
        pos_3d_global = None
        ph, pw = H // self.patch_size, W // self.patch_size

        if self.position_getter_3d is not None:
            if self.cross_attn_mode == 'cross':
                # pos_3d_cf: (B*V, T*hw, 3) for cross-frame blocks.
                pos_3d_single = self.position_getter_3d(
                    1, T, ph, pw, device, n_special=self.patch_start_idx
                )
                pos_3d_cf = pos_3d_single.expand(B * V, -1, -1)
            elif self.global_use_3d_rope:
                # pos_3d_global: (B, N*hw, 3) for global cross-image blocks.
                # N views are frame-major: [f0_c0, f0_c1, ..., f1_c0, ...].
                # Each token gets position (t, y, x); same-frame cameras share t.
                y_idx = torch.arange(ph, device=device)
                x_idx = torch.arange(pw, device=device)
                yx = torch.cartesian_prod(y_idx, x_idx)  # (ph*pw, 2)
                view_positions = []
                for t in range(T):
                    t_col = torch.full((ph * pw, 1), t, device=device, dtype=torch.long)
                    patch_pos = torch.cat([t_col, yx], dim=1)  # (ph*pw, 3)
                    if self.patch_start_idx > 0:
                        special = torch.zeros(self.patch_start_idx, 3, device=device, dtype=torch.long)
                        special[:, 0] = t
                        patch_pos = torch.cat([special, patch_pos], dim=0)  # (hw, 3)
                    # Same positions for all V cameras in this frame
                    for _ in range(V):
                        view_positions.append(patch_pos)
                pos_3d_global = torch.cat(view_positions, dim=0).unsqueeze(0).expand(B, -1, -1)

        pose_inject_mask = None

        for i, blk in enumerate(self.decoder):
            layer_attn_mask = None

            if self.cross_attn_mode == 'global':
                # Interleaved pattern: self (even) / global-cross (odd)
                if i % 2 == 0:
                    hidden = hidden.reshape(B * N, hw, -1)
                    pos    = pos.reshape(B * N, hw, -1)
                    cur_pos = pos
                else:
                    hidden = hidden.reshape(B, N * hw, -1)
                    if self.global_use_3d_rope and pos_3d_global is not None:
                        cur_pos = pos_3d_global
                    else:
                        cur_pos = pos.reshape(B, N * hw, -1)
                    layer_attn_mask = global_block_attn_mask
            else:
                # Cross-shaped pattern: self / cross-view / self / cross-frame
                layer_type = i % 4
                if layer_type == 0 or layer_type == 2:
                    hidden = hidden.reshape(B * N, hw, -1)
                    pos    = pos.reshape(B * N, hw, -1)
                    cur_pos = pos
                elif layer_type == 1:
                    # Cross-view: same frame, different cameras → (B*T, V*hw, D)
                    hidden  = hidden.reshape(B, T, V, hw, -1).reshape(B * T, V * hw, -1)
                    pos     = pos.reshape(B, T, V, hw, -1).reshape(B * T, V * hw, -1)
                    cur_pos = pos
                else:
                    # Cross-frame: same camera, different frames → (B*V, T*hw, D)
                    hidden = hidden.reshape(B, T, V, hw, -1).permute(0, 2, 1, 3, 4).reshape(B * V, T * hw, -1)
                    if pos_3d_cf is not None:
                        # pos_3d_cf: (B*V, T*hw, 3) with columns [t, y, x]
                        cur_pos = pos_3d_cf
                    else:
                        pos     = pos.reshape(B, T, V, hw, -1).permute(0, 2, 1, 3, 4).reshape(B * V, T * hw, -1)
                        cur_pos = pos

            hidden = blk(hidden, xpos=cur_pos, attn_mask=layer_attn_mask)

            # Cross-shaped only: restore (B*N, hw, D) after cross-view / cross-frame
            if self.cross_attn_mode == 'cross':
                layer_type = i % 4
                if layer_type == 1:
                    hidden = hidden.reshape(B, T, V, hw, -1).reshape(B * N, hw, -1)
                    pos    = pos.reshape(B, T, V, hw, -1).reshape(B * N, hw, -1)
                elif layer_type == 3:
                    hidden = hidden.reshape(B, V, T, hw, -1).permute(0, 2, 1, 3, 4).reshape(B * N, hw, -1)
                    # pos stays in 2D (B*N, hw, 2) — no reshape needed since we used pos_3d_flat separately

            # Pose injection after cross-attention layers, 5 times total.
            # poses is cam_to_ego; PoseInjectBlock internally computes
            # ego_to_cam = se3_inverse(cam_to_ego) as viewmats for ProPE.
            if self.use_multimodal and poses is not None and i in [1, 9, 17, 25, 33]:
                hidden = hidden.reshape(B, N, -1, 1024)
                poses_feat = self.pose_inject_blk[pose_inject_blk_idx](
                    hidden[..., self.patch_start_idx:, :].reshape(B, N * (hw - self.patch_start_idx), -1),
                    poses, H, W, H // 14, W // 14, attn_mask=pose_inject_mask
                ).reshape(B, N, -1, 1024)
                hidden[..., self.patch_start_idx:, :] += poses_feat
                hidden = hidden.reshape(B * N, hw, -1)
                pose_inject_blk_idx += 1

            if i == len(self.decoder) - 2:
                temp_features = hidden.clone().reshape(B * N, hw, -1)

        concatenated = torch.cat((temp_features, hidden.reshape(B * N, hw, -1)), dim=-1)
        return concatenated, pos.reshape(B * N, hw, -1), ref_idxs

    def decode_streaming(
        self,
        hidden: torch.Tensor,
        N_cur: int,
        H: int,
        W: int,
        V: int,
        past_kv: list,
        block_frame_offset: int,
        poses: Optional[torch.Tensor] = None,
    ):
        """Decode one streaming block with KV-cache accumulation (global mode only)."""
        if self.cross_attn_mode != "global":
            raise ValueError(
                f"decode_streaming() currently supports only 'global', got '{self.cross_attn_mode}'"
            )

        device = hidden.device
        B = hidden.shape[0] // N_cur

        register_token = self.register_token.repeat(B, N_cur, 1, 1).reshape(
            B * N_cur, *self.register_token.shape[-2:]
        )
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        pos = self.position_getter(B * N_cur, H // self.patch_size, W // self.patch_size, device)
        if self.patch_start_idx > 0:
            pos_patch = pos + 1
            pos_special = torch.zeros(
                B * N_cur, self.patch_start_idx, 2, device=device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos_patch], dim=1)

        if self.global_use_3d_rope and self.position_getter_3d is not None:
            single_img_pos = pos[0]
            frame_ids = torch.arange(N_cur, device=device) // V + int(block_frame_offset)
            t_col = frame_ids.view(N_cur, 1, 1).expand(N_cur, hw, 1).to(single_img_pos.dtype)
            yx_col = single_img_pos.unsqueeze(0).expand(N_cur, hw, 2)
            pos_3d_cur = (
                torch.cat([t_col, yx_col], dim=-1)
                .reshape(1, N_cur * hw, 3)
                .expand(B, -1, -1)
                .clone()
            )
        else:
            pos_3d_cur = None

        temp_features = None
        global_layer_idx = 0
        pose_inject_blk_idx = 0
        pose_inject_mask = None

        for i, blk in enumerate(self.decoder):
            if i % 2 == 0:
                hidden = hidden.reshape(B * N_cur, hw, -1)
                cur_pos = pos.reshape(B * N_cur, hw, -1)
                hidden = blk(hidden, xpos=cur_pos)
            else:
                hidden = hidden.reshape(B, N_cur * hw, -1)
                cur_pos = pos_3d_cur if pos_3d_cur is not None else pos.reshape(B, N_cur * hw, -1)
                hidden, new_kv = blk(
                    hidden,
                    xpos=cur_pos,
                    attn_mask=None,
                    past_key_values=past_kv[global_layer_idx],
                    use_cache=True,
                )
                past_kv[global_layer_idx] = new_kv
                global_layer_idx += 1

            if self.use_multimodal and poses is not None and i in [1, 9, 17, 25, 33]:
                hidden = hidden.reshape(B, N_cur, -1, self.dec_embed_dim)
                poses_feat = self.pose_inject_blk[pose_inject_blk_idx](
                    hidden[..., self.patch_start_idx:, :].reshape(
                        B, N_cur * (hw - self.patch_start_idx), -1
                    ),
                    poses,
                    H,
                    W,
                    H // self.patch_size,
                    W // self.patch_size,
                    attn_mask=pose_inject_mask,
                ).reshape(B, N_cur, -1, self.dec_embed_dim)
                hidden[..., self.patch_start_idx:, :] += poses_feat
                hidden = hidden.reshape(B * N_cur, hw, -1)
                pose_inject_blk_idx += 1

            if i == len(self.decoder) - 2:
                temp_features = hidden.clone().reshape(B * N_cur, hw, -1)

        concatenated = torch.cat([temp_features, hidden.reshape(B * N_cur, hw, -1)], dim=-1)
        return concatenated, pos.reshape(B * N_cur, hw, -1)

    @torch.no_grad()
    def inference_streaming(
        self,
        imgs: torch.Tensor,
        num_cameras: int = 1,
        block_size: int = None,
        intrinsics: Optional[torch.Tensor] = None,
        cam_to_ego: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Streaming inference using KV-cache (global mode)."""
        if self.cross_attn_mode != "global":
            raise ValueError(
                f"inference_streaming() currently supports only 'global', got '{self.cross_attn_mode}'"
            )

        B, total_N, _, H, W = imgs.shape
        V = int(num_cameras)
        if V <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")
        if total_N <= 0 or total_N % V:
            raise ValueError("The clip must contain a positive whole number of frames (N = T*V).")
        T = total_N // V
        bs = int(block_size) if block_size else (int(self.block_size) if self.block_size else 1)
        bs = max(bs, 1)

        num_global_layers = len(self.decoder) // 2
        past_kv = [None] * num_global_layers
        ego_conf_reference = None

        all_local_points = []
        all_xy = []
        all_camera_poses = []
        all_metric = []
        all_ego_poses = []
        all_ego_delta_t = []
        all_ego_delta_rotvec = []

        for blk_start in range(0, T, bs):
            blk_end = min(blk_start + bs, T)
            blk_T = blk_end - blk_start
            blk_N = blk_T * V
            start_idx = blk_start * V
            end_idx = blk_end * V
            block_imgs = imgs[:, start_idx:end_idx]

            block_intrinsics = None
            if isinstance(intrinsics, torch.Tensor) and intrinsics.dim() >= 2:
                block_intrinsics = intrinsics[:, start_idx:end_idx].contiguous()

            block_cam_to_ego = None
            if isinstance(cam_to_ego, torch.Tensor) and cam_to_ego.dim() >= 2:
                block_cam_to_ego = cam_to_ego[:, start_idx:end_idx].contiguous()

            hidden, decoder_cam_to_ego_poses, ego_decoder_cam_to_ego_poses = self.encode(
                block_imgs,
                intrinsics=block_intrinsics,
                cam_to_ego=block_cam_to_ego,
                **kwargs,
            )

            concatenated, pos_2d = self.decode_streaming(
                hidden,
                blk_N,
                H,
                W,
                V=V,
                past_kv=past_kv,
                block_frame_offset=blk_start,
                poses=decoder_cam_to_ego_poses,
            )

            patch_h, patch_w = H // self.patch_size, W // self.patch_size
            outputs = self.forward_head(
                concatenated,
                pos_2d,
                order_flow=None,
                B=B,
                N=blk_N,
                H=H,
                W=W,
                patch_h=patch_h,
                patch_w=patch_w,
                T=blk_T,
                V=V,
                cam_to_ego_poses=ego_decoder_cam_to_ego_poses,
                ego_conf_reference=ego_conf_reference,
            )
            if ego_conf_reference is None:
                ego_conf_reference = self._first_frame_reference(concatenated, pos_2d, B, blk_T, V)

            all_local_points.append(outputs["local_points"])
            all_xy.append(outputs["xy"])
            all_camera_poses.append(outputs["camera_poses"])
            all_metric.append(outputs["metric"])
            if outputs.get("ego_poses") is not None:
                all_ego_poses.append(outputs["ego_poses"])
            if outputs.get("ego_delta_t") is not None:
                all_ego_delta_t.append(outputs["ego_delta_t"])
            if outputs.get("ego_delta_rotvec") is not None:
                all_ego_delta_rotvec.append(outputs["ego_delta_rotvec"])

        result = dict(
            local_points=torch.cat(all_local_points, dim=1),
            xy=torch.cat(all_xy, dim=1),
            camera_poses=torch.cat(all_camera_poses, dim=1),
            metric=all_metric,
        )
        if all_ego_poses:
            result["ego_poses"] = torch.cat(all_ego_poses, dim=1)
        if all_ego_delta_t:
            result["ego_delta_t"] = torch.cat(all_ego_delta_t, dim=1)
        if all_ego_delta_rotvec:
            result["ego_delta_rotvec"] = torch.cat(all_ego_delta_rotvec, dim=1)
        return result

    @torch.no_grad()
    def forward_one_block(
        self,
        imgs: torch.Tensor,
        past_key_values=None,
        frame_offset: int = 0,
        block_size: int = None,
        num_frames: int = None,
        num_cameras: int = 1,
        intrinsics: Optional[torch.Tensor] = None,
        cam_to_ego: Optional[torch.Tensor] = None,
        ego_conf_reference=None,
        **kwargs,
    ):
        """Forward a single block and update KV cache (compat wrapper for eval scripts)."""
        _ = block_size  # kept for signature compatibility
        B, blk_N, _, H, W = imgs.shape
        V = int(num_cameras)
        if V <= 0 or blk_N <= 0 or blk_N % V:
            raise ValueError("A block must contain a positive whole number of frames (N = T*V).")
        blk_T = int(num_frames) if num_frames is not None else (blk_N // V)
        if blk_T <= 0 or blk_T * V != blk_N:
            raise ValueError(f"T*V mismatch: {blk_T}*{V} != {blk_N}")

        if past_key_values is None:
            past_key_values = [None] * (len(self.decoder) // 2)

        hidden, decoder_cam_to_ego_poses, ego_decoder_cam_to_ego_poses = self.encode(
            imgs,
            intrinsics=intrinsics,
            cam_to_ego=cam_to_ego,
            **kwargs,
        )
        concatenated, pos_2d = self.decode_streaming(
            hidden,
            blk_N,
            H,
            W,
            V=V,
            past_kv=past_key_values,
            block_frame_offset=int(frame_offset),
            poses=decoder_cam_to_ego_poses,
        )

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        pred = self.forward_head(
            concatenated,
            pos_2d,
            order_flow=None,
            B=B,
            N=blk_N,
            H=H,
            W=W,
            patch_h=patch_h,
            patch_w=patch_w,
            T=blk_T,
            V=V,
            cam_to_ego_poses=ego_decoder_cam_to_ego_poses,
            ego_conf_reference=ego_conf_reference,
        )
        return pred, past_key_values, concatenated, pos_2d

    @staticmethod
    def _first_frame_reference(hidden, pos, B, T, V):
        """Keep the segment's first frame as the confidence head's anchor."""
        first_hidden = hidden.reshape(B, T, V, *hidden.shape[1:])[:, 0]
        first_pos = pos.reshape(B, T, V, *pos.shape[1:])[:, 0]
        return (first_hidden.reshape(B * V, *hidden.shape[1:]).detach().clone(),
                first_pos.reshape(B * V, *pos.shape[1:]).detach().clone())

    def _predict_ego_conf(self, hidden, pos, B, T, V, reference=None):
        # Prepending the cached frame lets every decoder layer use exactly the
        # same same-camera reference as a full-clip forward, including the
        # reference's own intermediate features. Discard it before pooling.
        head_T = T
        if reference is not None:
            ref_hidden, ref_pos = reference
            hidden = torch.cat([
                ref_hidden.reshape(B, V, *hidden.shape[1:]),
                hidden.reshape(B, T * V, *hidden.shape[1:]),
            ], dim=1).flatten(0, 1)
            pos = torch.cat([
                ref_pos.reshape(B, V, *pos.shape[1:]),
                pos.reshape(B, T * V, *pos.shape[1:]),
            ], dim=1).flatten(0, 1)
            head_T += 1
        ret = self.ego_conf_decoder(
            hidden, xpos=pos, batch_size=B, num_frames=head_T, num_cameras=V,
        )
        feat = ret[:, self.patch_start_idx:].float()
        feat = feat.reshape(B, head_T, V, *feat.shape[1:]).mean(dim=(2, 3))
        if reference is not None:
            feat = feat[:, 1:]
        bs = int(self.block_size)
        block_feat = torch.stack([
            feat[:, start:min(start + bs, T)].mean(dim=1)
            for start in range(0, T, bs)
        ], dim=1)
        return self.ego_conf_head(block_feat).squeeze(-1)


    def forward_head(
        self,
        hidden,
        pos,
        order_flow,
        B,
        N,
        H,
        W,
        patch_h,
        patch_w,
        T=None,
        V=None,
        cam_to_ego_poses=None,
        ego_conf_reference=None,
    ):
        hw = patch_h * patch_w + self.patch_start_idx

        ret_point = self.point_decoder(hidden, xpos=pos)
        ret_camera = self.camera_decoder(hidden, xpos=pos)

        pos_hw = pos.reshape(B, N * hw, -1)
        ret_metric = self.metric_decoder(
            self.metric_token.repeat(B, 1, 1),
            hidden.reshape(B, N * hw, -1),
            xpos=pos_hw[:, 0:1],
            ypos=pos_hw
        )

        with torch.amp.autocast(device_type='cuda', enabled=False):
            xy, z = self.point_head(ret_point[:, self.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w)
            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = torch.exp(z.clamp(max=15.0))
            local_points = torch.cat([xy * z, z], dim=-1)

            camera_poses = self.camera_head(
                ret_camera[:, self.patch_start_idx:].float(), patch_h, patch_w
            ).reshape(B, N, 4, 4)

            metric = self.metric_head(ret_metric.float()).reshape(B).exp()

            ego_poses = None
            ego_delta_t = None
            ego_delta_rotvec = None
            if self.use_ego_pose and T is not None and V is not None and T >= 1 and V > 1:
                ego_feat = self.ego_decoder(
                    hidden, pos, T=T, V=V, H=H, W=W,
                    patch_h=patch_h, patch_w=patch_w,
                    cam_to_ego=cam_to_ego_poses,
                )
                ego_delta, ego_delta_t, ego_delta_rotvec = self.ego_head(
                    ego_feat[:, self.patch_start_idx:].float(),
                    patch_h, patch_w, T, V,
                )
                ego_delta = ego_delta.reshape(B, T, 4, 4)
                ego_delta_t = ego_delta_t.reshape(B, T, 3)
                ego_delta_rotvec = ego_delta_rotvec.reshape(B, T, 3)

                # Keep the cam0 pose tensor independent of camera_poses storage.
                cam0_poses = camera_poses.reshape(B, T, V, 4, 4)[:, :, 0].clone()

                # Convert predicted cam0 poses to ego frame using GT cam_to_ego.
                # cam_to_ego: cam(RDF) -> ego(RDF), so inv(cam_to_ego): ego -> cam.
                # pred_cam0 is cam->world; multiplying on the right by inv(cam_to_ego)
                # re-expresses it as ego->world, i.e. pred_ego_to_world = cam0 @ ego_to_cam.
                # Then ego_delta (learned residual in ego frame) is applied on the right.
                if cam_to_ego_poses is not None:
                    cam_to_ego_cam0 = cam_to_ego_poses.reshape(B, T, V, 4, 4)[:, :, 0]  # (B, T, 4, 4)
                    ego_to_cam0 = se3_inverse(cam_to_ego_cam0)  # inv: ego -> cam
                    ego_poses = torch.matmul(torch.matmul(cam0_poses, ego_to_cam0), ego_delta)
                else:
                    ego_poses = torch.matmul(cam0_poses, ego_delta)

            # Ego block confidence: one logit per temporal block.
            # Fires only when all conditions are satisfied at forward time.
            ego_conf_logits = None
            ego_conf_block_size = None
            if (
                getattr(self, '_ego_conf_enabled', False)
                and T is not None and V is not None
                and (T > 1 or ego_conf_reference is not None) and V > 1
            ):
                ego_conf_logits = self._predict_ego_conf(
                    hidden, pos, B, T, V, reference=ego_conf_reference,
                )
                ego_conf_block_size = int(self.block_size)

        return dict(
            local_points=local_points,
            xy=xy,
            camera_poses=camera_poses,
            metric=metric,
            ego_poses=ego_poses,
            ego_delta_t=ego_delta_t,
            ego_delta_rotvec=ego_delta_rotvec,
            ego_conf_logits=ego_conf_logits,
            ego_conf_block_size=ego_conf_block_size,
        )
