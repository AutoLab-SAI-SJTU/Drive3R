# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

try:
    from models.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):

        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens

# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos


#----------------------------------------------------------
# RoPE3D: 3D spatio-temporal RoPE over (t, y, x)
#----------------------------------------------------------

class RotaryPositionEmbedding3D(nn.Module):
    """3D RoPE that encodes (t, y, x) positions.

    head_dim is split into three parts [y | x | t] according to
    temporal_dim_ratio.  Each part gets an independent 1-D RoPE.

    Spatial frequencies match the released 2D RoPE backbone.
    The temporal coefficient preserves the released inference setting.

    Args:
        frequency: base frequency for the sinusoidal embedding (default 100).
        temporal_dim_ratio: fraction of head_dim allocated to the temporal axis
            (default 0.25).  The remaining (1 - ratio) is split evenly between
            y and x.
        temporal_spacing: scale factor applied to frame indices before lookup,
            equivalent to VideoRoPE ATS (default 1.0 = no scaling).
        align_spatial_freq: if True, compute y/x frequencies using the full
            head_dim as the denominator so they match the pretrained 2D RoPE
            (default True).
        temporal_alpha: coefficient for t-axis rotation (default 0.0).
    """

    def __init__(
        self,
        frequency: float = 100.0,
        temporal_dim_ratio: float = 0.25,
        temporal_spacing: float = 1.0,
        align_spatial_freq: bool = True,
        temporal_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_frequency = frequency
        self.temporal_dim_ratio = temporal_dim_ratio
        self.temporal_spacing = temporal_spacing
        self.align_spatial_freq = align_spatial_freq
        self.temporal_alpha = temporal_alpha
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_dim_split(self, head_dim: int) -> Tuple[int, int, int]:
        """Return (dim_y, dim_x, dim_t) that sum to head_dim, all even."""
        dim_t = max(2, round(head_dim * self.temporal_dim_ratio / 2) * 2)
        dim_spatial = head_dim - dim_t
        dim_y = (dim_spatial // 2 // 2) * 2
        dim_x = head_dim - dim_t - dim_y
        assert dim_y > 0 and dim_x > 0 and dim_t > 0
        assert dim_y + dim_x + dim_t == head_dim
        return dim_y, dim_x, dim_t

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype,
        full_dim: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin lookup table of shape (seq_len, dim).

        Args:
            dim:      number of features this axis occupies (must be even).
            seq_len:  maximum position index + 1.
            full_dim: if > 0, use this as the denominator when computing
                      inv_freq instead of ``dim``.  Pass head_dim here for
                      the spatial axes to align frequencies with 2D RoPE.
        """
        denom = full_dim if full_dim > 0 else dim
        cache_key = (dim, denom, device, dtype)
        cached = self.frequency_cache.get(cache_key, None)
        if cached is None or cached[0].shape[0] < seq_len:
            alloc_len = max(seq_len, seq_len * 2)
            # Use only the first (dim/2) frequencies from the full-dim table
            exponents = torch.arange(0, dim, 2, device=device).float() / denom
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(alloc_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply 3D RoPE.

        Args:
            tokens:    (batch_size, n_heads, n_tokens, head_dim)
            positions: (batch_size, n_tokens, 3)  columns = [t, y, x]

        Returns:
            Tensor of same shape as tokens.
        """
        assert tokens.size(-1) % 4 == 0, "head_dim must be divisible by 4 for 3D RoPE"
        assert positions.ndim == 3 and positions.shape[-1] == 3, \
            "positions must have shape (batch_size, n_tokens, 3) with [t, y, x]"

        head_dim = tokens.size(-1)
        dim_y, dim_x, dim_t = self._get_dim_split(head_dim)

        y_feat = tokens[..., :dim_y]
        x_feat = tokens[..., dim_y:dim_y + dim_x]
        t_feat = tokens[..., dim_y + dim_x:]

        t_pos = positions[..., 0]
        y_pos = positions[..., 1]
        x_pos = positions[..., 2]

        if self.temporal_spacing != 1.0:
            t_pos = (t_pos.float() * self.temporal_spacing).long()

        max_spatial = int(max(y_pos.max(), x_pos.max())) + 1
        max_temporal = int(t_pos.max()) + 1

        # Spatial axes: optionally use full head_dim as denominator to align
        # frequencies with the pretrained 2D RoPE.
        spatial_full_dim = head_dim if self.align_spatial_freq else 0
        cos_s, sin_s = self._compute_frequency_components(
            dim_y, max_spatial, tokens.device, tokens.dtype, full_dim=spatial_full_dim
        )
        cos_t, sin_t = self._compute_frequency_components(
            dim_t, max_temporal, tokens.device, tokens.dtype
        )

        y_feat = self._apply_1d_rope(y_feat, y_pos, cos_s, sin_s)
        x_feat = self._apply_1d_rope(x_feat, x_pos, cos_s, sin_s)

        # Temporal coefficient: blend between unrotated and rotated.
        # alpha=0 → no temporal rotation (behaves like 2D RoPE on this segment)
        # alpha=1 → full 3D RoPE
        if self.temporal_alpha >= 1.0:
            t_feat = self._apply_1d_rope(t_feat, t_pos, cos_t, sin_t)
        elif self.temporal_alpha <= 0.0:
            pass  # leave t_feat unrotated
        else:
            t_feat_rotated = self._apply_1d_rope(t_feat, t_pos, cos_t, sin_t)
            t_feat = self.temporal_alpha * t_feat_rotated + (1.0 - self.temporal_alpha) * t_feat

        return torch.cat((y_feat, x_feat, t_feat), dim=-1)


class PositionGetter3D(object):
    """Return (t, y, x) positions for a sequence of T frames, each with h*w patches.

    The output shape is (b*n_per_frame, T*h*w, 3) where n_per_frame = b*N / (T * V)
    but callers are responsible for the final reshape; this helper returns
    (b_times_N, T*h*w, 3) given b_times_N = B*T*V and the per-call T, h, w.

    Concretely, for each token the position is:
        t  = frame index  in [0, T)
        y  = patch row    in [0, h)
        x  = patch column in [0, w)
    Register / special tokens at the front of each frame get position (t, 0, 0).
    """

    def __init__(self) -> None:
        self.cache_positions: Dict[Tuple, torch.Tensor] = {}

    def __call__(self, b_times_N: int, T: int, h: int, w: int, device: torch.device,
                 n_special: int = 0) -> torch.Tensor:
        """Return positions of shape (b_times_N, T*(n_special + h*w), 3).

        Args:
            b_times_N: batch dimension (B*N = B*T*V).
            T:         number of frames.
            h, w:      patch grid height and width.
            device:    target device.
            n_special: number of special (register) tokens prepended per frame.

        Returns:
            Tensor of shape (b_times_N, T*(n_special + h*w), 3) with dtype long.
        """
        key = (T, h, w, n_special, device)
        if key not in self.cache_positions:
            y = torch.arange(h, device=device)
            x = torch.arange(w, device=device)
            yx = torch.cartesian_prod(y, x)  # (h*w, 2)

            frame_tokens = []
            for t in range(T):
                t_tensor = torch.full((h * w, 1), t, device=device, dtype=yx.dtype)
                patch_pos = torch.cat([t_tensor, yx], dim=1)  # (h*w, 3)
                if n_special > 0:
                    special_pos = torch.zeros(n_special, 3, device=device, dtype=yx.dtype)
                    special_pos[:, 0] = t
                    patch_pos = torch.cat([special_pos, patch_pos], dim=0)
                frame_tokens.append(patch_pos)

            all_pos = torch.cat(frame_tokens, dim=0)  # (T*(n_special+h*w), 3)
            self.cache_positions[key] = all_pos

        all_pos = self.cache_positions[key]
        seq_len = all_pos.shape[0]
        return all_pos.view(1, seq_len, 3).expand(b_times_N, -1, 3).clone()
