# Pi3-derived portions: Copyright (c) 2025, the authors. All rights reserved.
# BSD 3-Clause; see licenses/Pi3-BSD-3-Clause.txt.
# Additional source notices: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn as nn
from copy import deepcopy
from functools import partial
import torch.nn.functional as F

from .attention import FlashAttentionRope
from .block import BlockRope, PoseInjectBlock
from ..dinov2.layers import Mlp

# marepo-derived block: Copyright © Niantic, Inc. 2024. Patent Pending.
# Upstream terms: licenses/marepo-LICENSE.txt.
# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class CameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim))
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            out_r = self.fc_rot(feat.float())  # [B,9]
            pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r


class EgoDecoder(nn.Module):
    """4-layer decoder for ego pose: self-attn -> cross-view -> self-attn -> cross-view.

    Pose injection (PoseInjectBlock) is applied once after the first cross-view layer.
    """

    def __init__(
        self,
        in_dim,
        dec_embed_dim=1024,
        dec_num_heads=16,
        out_dim=512,
        mlp_ratio=4,
        rope=None,
    ):
        super().__init__()
        self.project = nn.Linear(in_dim, dec_embed_dim)

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
                attn_class=FlashAttentionRope,
                rope=rope,
            )
            for _ in range(4)
        ])

        self.pose_inject = PoseInjectBlock(
            dim=dec_embed_dim,
            num_heads=dec_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            init_values=0.01,
            qk_norm=True,
        )

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, pos, T, V, H, W, patch_h, patch_w, cam_to_ego=None):
        """
        Args:
            hidden:     (B*N, hw, in_dim) where N = T * V, hw includes register tokens.
            pos:        (B*N, hw, 2) RoPE positions.
            T:          number of frames.
            V:          number of cameras per frame.
            H, W:       original image height / width.
            patch_h, patch_w: patch grid size.
            cam_to_ego: (B, N, 4, 4) or None.
        Returns:
            out: (B*N, hw, out_dim)
        """
        BN, hw, _ = hidden.shape
        B = BN // (T * V)
        N = T * V
        D = self.blocks[0].attn.qkv.in_features

        hidden = self.project(hidden)

        patch_start_idx = hw - patch_h * patch_w

        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                # Self-attention per camera image
                cur = hidden.reshape(B * N, hw, D)
                cur_pos = pos.reshape(B * N, hw, -1)
            else:
                # Cross-view: same frame, different cameras -> (B*T, V*hw, D)
                cur = hidden.reshape(B, T, V, hw, D).reshape(B * T, V * hw, D)
                cur_pos = pos.reshape(B, T, V, hw, -1).reshape(B * T, V * hw, -1)

            cur = blk(cur, xpos=cur_pos)

            # Reshape back to (B*N, hw, D) after cross-view layers
            if i % 2 == 1:
                cur = cur.reshape(B, T, V, hw, D).reshape(B * N, hw, D)

            hidden = cur

            # Pose injection after the first cross-view layer (i == 1)
            if i == 1 and cam_to_ego is not None:
                hidden = hidden.reshape(B, N, hw, D)
                patch_tokens = hidden[:, :, patch_start_idx:, :].reshape(
                    B, N * (hw - patch_start_idx), D
                )
                pose_delta = self.pose_inject(
                    patch_tokens, cam_to_ego, H, W, patch_h, patch_w,
                ).reshape(B, N, hw - patch_start_idx, D)
                hidden[:, :, patch_start_idx:, :] = (
                    hidden[:, :, patch_start_idx:, :] + pose_delta
                )
                hidden = hidden.reshape(B * N, hw, D)

        return self.linear_out(hidden)


class EgoHead(nn.Module):
    """Predict a small per-frame ego delta from mean-over-cameras features.

    Input:  (B*N, patch_hw, dim) from EgoDecoder, N = T * V.
    Output: identity-initialized (B*T, 4, 4) SE(3) delta matrices plus
    their translation / rotation-vector parameters.
    """

    def __init__(self, dim=512):
        super().__init__()
        self.res_conv = nn.ModuleList([
            deepcopy(ResConvBlock(dim, dim)) for _ in range(2)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_rot = nn.Linear(dim, 3)
        self._init_identity_delta()

    def _init_identity_delta(self):
        nn.init.zeros_(self.fc_t.weight)
        nn.init.zeros_(self.fc_t.bias)
        nn.init.zeros_(self.fc_rot.weight)
        nn.init.zeros_(self.fc_rot.bias)

    @staticmethod
    def _skew(v):
        z = torch.zeros_like(v[..., 0])
        return torch.stack(
            [
                z, -v[..., 2], v[..., 1],
                v[..., 2], z, -v[..., 0],
                -v[..., 1], v[..., 0], z,
            ],
            dim=-1,
        ).reshape(*v.shape[:-1], 3, 3)

    @classmethod
    def _axis_angle_to_matrix(cls, rotvec):
        theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta2.clamp_min(1e-12))
        small = theta2 < 1e-8
        theta4 = theta2 * theta2
        A = torch.where(
            small,
            1.0 - theta2 / 6.0 + theta4 / 120.0,
            torch.sin(theta) / theta,
        )
        B = torch.where(
            small,
            0.5 - theta2 / 24.0 + theta4 / 720.0,
            (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
        )
        K = cls._skew(rotvec)
        eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).view(1, 3, 3)
        return eye + A[..., None] * K + B[..., None] * torch.matmul(K, K)

    def forward(self, feat, patch_h, patch_w, T, V):
        """
        Args:
            feat: (B*N, patch_hw, dim), N = T * V, no register tokens.
            patch_h, patch_w: spatial grid size.
            T: number of frames.
            V: number of cameras per frame.
        Returns:
            delta_pose: (B*T, 4, 4), delta_t: (B*T, 3), delta_rotvec: (B*T, 3)
        """
        BN, hw, c = feat.shape
        B = BN // (T * V)

        for block in self.res_conv:
            feat = block(feat)

        feat = self.avgpool(
            feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous()
        )
        feat = feat.view(BN, -1)  # (B*N, dim)

        # Mean over V cameras within each frame
        feat = feat.reshape(B, T, V, -1).mean(dim=2)  # (B, T, dim)
        feat = feat.reshape(B * T, -1)

        feat = self.more_mlps(feat)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            delta_t = self.fc_t(feat.float())
            delta_rotvec = self.fc_rot(feat.float())
            delta_r = self._axis_angle_to_matrix(delta_rotvec)
            delta_pose = torch.zeros((B * T, 4, 4), device=feat.device, dtype=torch.float32)
            delta_pose[:, :3, :3] = delta_r
            delta_pose[:, :3, 3] = delta_t
            delta_pose[:, 3, 3] = 1.0
        return delta_pose, delta_t, delta_rotvec
