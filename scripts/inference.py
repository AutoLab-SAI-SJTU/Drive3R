"""Drive3R streaming inference.

Run a Drive3R checkpoint on a multi-camera, multi-frame clip and dump
per-frame camera poses, ego poses and world point clouds.

Supports three input layouts:

  1. Per-camera sub-directories (default).
        --image-dir SCENE_DIR  --num-cameras V  [--num-frames T]
     SCENE_DIR/{CAM_0}/<frame>.jpg
              {CAM_1}/<frame>.jpg
              ...
     Cameras are sorted alphabetically; frames inside each camera are sorted
     alphabetically.  T defaults to ``min(len(frames_per_cam))``.

  2. JSON manifest.
        --manifest scene.json
     ``scene.json`` schema:
        {
          "num_cameras": V,
          "num_frames":  T,
          "frames": [               # length T, in temporal order
            { "cameras": [path_v0, path_v1, ...] },
            ...
          ],
          "intrinsics": [            # optional, length V or T*V
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            ...
          ],
          "cam_to_ego": [            # optional, length V or T*V
            [[r00, r01, r02, tx], [r10, r11, r12, ty],
             [r20, r21, r22, tz], [0, 0, 0, 1]],
            ...
          ]
        }

    ``cam_to_ego`` (4x4 rigid SE(3) per view, cam->ego) is **required** when the
    ego-pose head is enabled (``--no-ego-pose`` not set). It is used to compute
    ``ego_poses = cam0 @ inv(cam_to_ego_cam0) @ ego_delta`` in the actual
    vehicle ego frame, using the calibrated pose chain. Pass ``--no-ego-pose``
    to skip the ego head and run cam-only inference without ``cam_to_ego``.

  3. Monocular sequence.
        --image-dir CLIP_DIR  --num-cameras 1
     All images directly under CLIP_DIR are treated as a temporal sequence.

Output
------
``output_dir/`` contains
    pred.npz            local_points / camera_poses / ego_poses / conf / metric
    trajectory_cam0.txt TUM-style cam-0 trajectory (timestamp tx ty tz qx qy qz qw)
    trajectory_ego.txt  TUM-style ego trajectory   (only when ego_poses are produced)
    world_pointcloud.ply  fused world point cloud (sub-sampled if too large)
    stats.json          ego_conf logits, restart events, run config

Pass ``--viser`` to additionally launch an interactive viser scene (point cloud
+ camera frustums + ego trajectory) on ``http://<host>:8080`` after the run.

Example
-------
    python scripts/inference.py \
        --ckpt $DRIVE3R_CKPT_ROOT/drive3r.pt \
        --image-dir SCENE --num-cameras 6 \
        --resolution 518x336 --block-size 3 \
        --output-dir outputs/scene1 \
        --viser --viser-port 8080
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model import Drive3R
from src.utils.basic import write_ply
from src.utils.checkpoint import load_model_weights
from src.utils.scene_io import load_scene


# ---------------------------------------------------------------------------
# Image / intrinsics IO
# ---------------------------------------------------------------------------

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _list_images(folder: str) -> List[str]:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(_IMG_EXTS)
    )


def _resolve_target_size(
    src_w: int,
    src_h: int,
    explicit: Optional[Tuple[int, int]],
    pixel_limit: int,
    patch: int = 14,
) -> Tuple[int, int]:
    """Pick a (W, H) divisible by ``patch`` that respects an optional pixel budget."""
    if src_w <= 0 or src_h <= 0 or patch <= 0:
        raise ValueError("Source dimensions and patch size must be positive.")
    if explicit is not None:
        w, h = explicit
        if w <= 0 or h <= 0 or w % patch or h % patch:
            raise ValueError(f"--resolution {w}x{h} must both be multiples of {patch}.")
        return w, h
    if pixel_limit < patch * patch:
        raise ValueError(f"--pixel-limit must be at least {patch * patch}.")
    scale = math.sqrt(pixel_limit / max(src_w * src_h, 1))
    w_t = max(patch, src_w * scale)
    h_t = max(patch, src_h * scale)
    k = max(1, round(w_t / patch))
    m = max(1, round(h_t / patch))
    while (k * patch) * (m * patch) > pixel_limit:
        if k / m > w_t / h_t:
            k = max(1, k - 1)
        else:
            m = max(1, m - 1)
    return k * patch, m * patch


def _load_image_to_tensor(path: str, target_w: int, target_h: int) -> torch.Tensor:
    """Load `path`, resize to ``(target_w, target_h)``, return ``(3, H, W)`` in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0       # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _scale_intrinsics(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    K = K.astype(np.float64).copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def _read_intrinsics_for_path(
    K_spec: Optional[List], view_idx: int, num_cameras: int
) -> Optional[np.ndarray]:
    """Return a (3, 3) intrinsic for view ``view_idx`` from the spec list.

    The spec may be ``None``, length-V (per camera, broadcast across T),
    length-T*V (per view) or length-1 (broadcast across all).
    """
    if K_spec is None:
        return None
    if len(K_spec) == 1:
        K = K_spec[0]
    elif len(K_spec) == num_cameras:
        K = K_spec[view_idx % num_cameras]
    else:
        K = K_spec[view_idx]
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Each intrinsic must be 3x3; got {K.shape}.")
    return K


def _read_cam_to_ego_for_path(
    c2e_spec: Optional[List], view_idx: int, num_cameras: int
) -> Optional[np.ndarray]:
    """Return a (4, 4) cam->ego SE(3) for ``view_idx`` from the spec list.

    The spec may be ``None``, length-V (per camera, broadcast across T),
    length-T*V (per view) or length-1 (broadcast across all). cam_to_ego is
    a static rigid extrinsic, so length-V is the common case.
    """
    if c2e_spec is None:
        return None
    if len(c2e_spec) == 1:
        M = c2e_spec[0]
    elif len(c2e_spec) == num_cameras:
        M = c2e_spec[view_idx % num_cameras]
    else:
        M = c2e_spec[view_idx]
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"Each cam_to_ego must be 4x4; got {M.shape}.")
    return M


def _build_clip_inputs(
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int, int, List[str]]:
    """Resolve input paths -> ``(imgs, intrinsics, cam_to_ego, T, V, paths)``.

    ``imgs`` is ``(1, T*V, 3, H, W)``. ``intrinsics`` (optional) is
    ``(1, T*V, 3, 3)``. ``cam_to_ego`` (optional) is ``(1, T*V, 4, 4)`` and
    enables calibrated ego-frame ``ego_poses`` from the model.
    """
    K_spec: Optional[List] = None
    c2e_spec: Optional[List] = None
    paths: List[str] = []
    if getattr(args, "input_path", None) is not None:
        scene = load_scene(
            args.input_path, num_frames=args.num_frames, num_cameras=args.num_cameras,
            stride=args.stride, start_index=args.start_index,
        )
        T, V = scene["num_frames"], scene["num_cameras"]
        paths = [row["image"] for row in scene["samples"]]

        def scene_matrices(key):
            matrices = [row.get(key) for row in scene["samples"]]
            present = [matrix is not None for matrix in matrices]
            if any(present) and not all(present):
                raise ValueError(f"Scene {key} must be provided for every camera/frame or omitted entirely.")
            return matrices if all(present) else None

        K_spec = scene_matrices("intrinsic")
        c2e_spec = scene_matrices("cam_to_ego")
    elif args.manifest is not None:
        with open(args.manifest, "r") as f:
            mf = json.load(f)
        V = int(mf["num_cameras"])
        T = int(mf.get("num_frames", len(mf["frames"])))
        if T <= 0 or V <= 0:
            raise ValueError("Manifest num_frames and num_cameras must be positive.")
        if len(mf["frames"]) != T:
            raise ValueError(f"Manifest declares num_frames={T} but has {len(mf['frames'])} entries.")
        for frame in mf["frames"]:
            cams = frame["cameras"]
            if len(cams) != V:
                raise ValueError(f"Frame has {len(cams)} cams; expected {V}.")
            manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
            paths.extend(p if os.path.isabs(p) else os.path.join(manifest_dir, p) for p in cams)
        K_spec = mf.get("intrinsics")
        c2e_spec = mf.get("cam_to_ego")
    else:
        if args.image_dir is None:
            raise ValueError("Provide either --manifest or --image-dir.")
        if args.num_cameras is None:
            raise ValueError("--num-cameras is required when using --image-dir.")
        V = int(args.num_cameras)
        if V <= 0 or (args.num_frames is not None and args.num_frames <= 0):
            raise ValueError("--num-cameras and --num-frames must be positive.")
        if V == 1:
            cam_files = _list_images(args.image_dir)
            T = int(args.num_frames) if args.num_frames else len(cam_files)
            cam_files = cam_files[:T]
            if len(cam_files) < T:
                raise ValueError(f"--num-frames={T} but only {len(cam_files)} images found.")
            for p in cam_files:
                paths.append(p)
        else:
            cam_dirs = sorted(
                os.path.join(args.image_dir, d)
                for d in os.listdir(args.image_dir)
                if os.path.isdir(os.path.join(args.image_dir, d))
            )
            if len(cam_dirs) != V:
                raise ValueError(
                    f"--num-cameras={V} but found {len(cam_dirs)} sub-folders in {args.image_dir}: {cam_dirs}"
                )
            per_cam = [_list_images(d) for d in cam_dirs]
            min_T = min(len(x) for x in per_cam)
            T = int(args.num_frames) if args.num_frames else min_T
            if T > min_T:
                raise ValueError(f"--num-frames={T} but shortest camera has {min_T} frames.")
            for t in range(T):
                for v in range(V):
                    paths.append(per_cam[v][t])

        if args.intrinsics_json is not None:
            with open(args.intrinsics_json, "r") as f:
                K_spec = json.load(f)
            if isinstance(K_spec, dict):
                K_spec = K_spec.get("intrinsics", K_spec.get("K"))
        if args.cam_to_ego_json is not None:
            with open(args.cam_to_ego_json, "r") as f:
                c2e_spec = json.load(f)
            if isinstance(c2e_spec, dict):
                c2e_spec = c2e_spec.get("cam_to_ego", c2e_spec.get("c2e"))

    if not paths:
        raise RuntimeError("No images resolved for inference.")
    for label, spec in (("intrinsics", K_spec), ("cam_to_ego", c2e_spec)):
        if spec is not None and len(spec) not in (1, V, T * V):
            raise ValueError(f"{label} must contain 1, {V}, or {T * V} matrices.")

    # Probe the first image to pick a shared resolution.
    with Image.open(paths[0]) as im0:
        src_w, src_h = im0.size
    explicit = None
    if args.resolution:
        w_str, h_str = args.resolution.lower().split("x")
        explicit = (int(w_str), int(h_str))
    target_w, target_h = _resolve_target_size(
        src_w, src_h, explicit, pixel_limit=args.pixel_limit, patch=14
    )
    print(f"[infer] T={T}, V={V}, target_size=({target_w}, {target_h})")

    tensors = []
    for p in paths:
        tensors.append(_load_image_to_tensor(p, target_w, target_h))
    imgs = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, T*V, 3, H, W)

    intr_tensor = None
    if K_spec is not None:
        Ks = []
        for i, p in enumerate(paths):
            K = _read_intrinsics_for_path(K_spec, view_idx=i, num_cameras=V)
            if K is None:
                Ks = []
                break
            with Image.open(p) as im:
                w, h = im.size
            Ks.append(_scale_intrinsics(K, w, h, target_w, target_h))
        if Ks:
            intr_tensor = torch.from_numpy(np.stack(Ks, axis=0)).float().unsqueeze(0)  # (1, T*V, 3, 3)

    # cam_to_ego is a rigid extrinsic, independent of image resolution.
    c2e_tensor = None
    if c2e_spec is not None:
        Ms = []
        for i in range(len(paths)):
            M = _read_cam_to_ego_for_path(c2e_spec, view_idx=i, num_cameras=V)
            if M is None:
                Ms = []
                break
            Ms.append(M)
        if Ms:
            c2e_tensor = torch.from_numpy(np.stack(Ms, axis=0)).float().unsqueeze(0)  # (1, T*V, 4, 4)

    return imgs, intr_tensor, c2e_tensor, T, V, paths


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


class _NullCtx:
    """No-op context manager (used when autocast is disabled)."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    # SVD / det on CUDA do not support bf16/fp16. Run entirely in fp32
    # regardless of the input dtype (autocast may silently downcast matmul
    # outputs even when inputs were explicitly cast to fp32 beforehand).
    m = matrix.float()
    u, _s, vh = torch.linalg.svd(m)
    r = u @ vh
    if torch.det(r.float()) < 0:
        u_fix = u.clone()
        u_fix[:, -1] *= -1
        r = u_fix @ vh
    return r.float()


def _estimate_se3_from_pose_pairs(
    src_poses: torch.Tensor, dst_poses: torch.Tensor
) -> torch.Tensor:
    """Solve T s.t. ``dst[i] = T @ src[i]`` from (1, K, 4, 4) overlapping poses.

    All SE(3)/SVD math runs in fp32 to avoid ``torch.linalg.svd`` /
    ``torch.det`` / ``lu_factor_cusolver`` not supporting bf16/fp16 on CUDA.
    The result is cast back to the input dtype for the streaming loop.
    """
    in_dtype = src_poses.dtype
    device = src_poses.device
    # Use autocast(enabled=False) to ensure fp32 ops are not silently
    # downcast by an outer torch.autocast context.
    with torch.autocast(device_type=src_poses.device.type, enabled=False):
        src = src_poses[0].float()
        dst = dst_poses[0].float()
        if src.shape[0] == 0:
            return torch.eye(4, device=device, dtype=in_dtype)
        rel_r = dst[:, :3, :3].float() @ src[:, :3, :3].float().transpose(-1, -2)
        r = _project_to_so3(rel_r.mean(dim=0))
        src_t = src[:, :3, 3].float()
        dst_t = dst[:, :3, 3].float()
        t = (dst_t - (r @ src_t.T).T).mean(dim=0)
        out = torch.eye(4, device=device, dtype=torch.float32)
        out[:3, :3] = r
        out[:3, 3] = t
    return out.to(dtype=in_dtype)


def _apply_se3(poses: torch.Tensor, se3: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=poses.device.type, enabled=False):
        return se3.to(dtype=poses.dtype).view(1, 1, 4, 4) @ poses


def _scale_pose_translation(poses: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled = poses.clone()
    scaled[..., :3, 3] = scaled[..., :3, 3] * scale
    return scaled


def _collect_block_pose_subset(
    block_poses: torch.Tensor,
    block_start: int,
    frame_ids: List[int],
    num_cameras: int,
) -> torch.Tensor:
    chunks = []
    for frame_id in frame_ids:
        local = frame_id - block_start
        chunks.append(block_poses[:, local * num_cameras: (local + 1) * num_cameras])
    return torch.cat(chunks, dim=1)


def _fallback_ego_from_cam0(
    camera_poses: torch.Tensor,
    cam_to_ego: Optional[torch.Tensor],
    *,
    num_frames: int,
    num_cameras: int,
) -> Optional[torch.Tensor]:
    """Approximate ego poses from cam0 poses when a short block skips ego head."""
    if cam_to_ego is None or not isinstance(cam_to_ego, torch.Tensor):
        return None
    B = int(camera_poses.shape[0])
    T = int(num_frames)
    V = int(num_cameras)
    if T <= 0 or V <= 0:
        return None
    cam0_poses = camera_poses.reshape(B, T, V, 4, 4)[:, :, 0]
    cam_to_ego0 = cam_to_ego.reshape(B, T, V, 4, 4)[:, :, 0]
    ego_to_cam0 = torch.linalg.inv(cam_to_ego0.float()).to(dtype=cam0_poses.dtype)
    return torch.matmul(cam0_poses, ego_to_cam0)


def _aggregate_scene(
    frame_local_points: Dict[int, torch.Tensor],
    frame_camera_poses: Dict[int, torch.Tensor],
    frame_ego_poses: Optional[Dict[int, torch.Tensor]],
    total_frames: int,
    last_metric: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    frame_seg_id: Optional[Dict[int, int]] = None,
    seg_metric: Optional[Dict[int, torch.Tensor]] = None,
    segment_alignment_pairs: Optional[
        Dict[int, Tuple[torch.Tensor, torch.Tensor, List[int]]]
    ] = None,
) -> Dict[str, torch.Tensor]:
    """Align segments in meters, then package geometry with a shared metric.

    Each segment uses its last block's metric. Overlap poses are converted
    to meters before estimating SE(3), once those final scales are known.
    Returned points and pose translations use the final segment's metric
    as their common normalization, matching the output exporters.
    """
    final_metric = last_metric if last_metric is not None else torch.ones(1, device=device, dtype=dtype)
    use_per_seg = (
        frame_seg_id is not None
        and seg_metric is not None
        and len(seg_metric) > 0
    )
    final_metric_scalar = final_metric.reshape(-1)[0].to(dtype=torch.float32)

    def segment_metric(segment):
        metric = seg_metric.get(segment, final_metric) if use_per_seg else final_metric
        return metric.reshape(-1)[0].to(dtype=torch.float32)

    identity = torch.eye(4, device=device, dtype=torch.float32)
    segment_se3_meters = {}
    for segment, (old_poses, new_poses, old_segments) in sorted(
        (segment_alignment_pairs or {}).items()
    ):
        old_aligned = []
        for index, old_segment in enumerate(old_segments):
            old_meters = _scale_pose_translation(
                old_poses[:, index:index + 1].float(), segment_metric(old_segment),
            )
            old_aligned.append(_apply_se3(
                old_meters, segment_se3_meters.get(old_segment, identity),
            ))
        new_meters = _scale_pose_translation(new_poses.float(), segment_metric(segment))
        segment_se3_meters[segment] = _estimate_se3_from_pose_pairs(
            new_meters, torch.cat(old_aligned, dim=1),
        )

    lp_parts, cam_parts, ego_parts = [], [], []
    for t in range(total_frames):
        lp_t = frame_local_points[t]
        cp_t = frame_camera_poses[t]
        ep_t = frame_ego_poses.get(t) if frame_ego_poses is not None else None
        seg = frame_seg_id.get(t, 0) if frame_seg_id is not None else 0

        if use_per_seg:
            m_seg_scalar = segment_metric(seg)
            ratio = (m_seg_scalar / final_metric_scalar).to(dtype=lp_t.dtype)
            if not torch.equal(ratio, torch.ones((), device=ratio.device, dtype=ratio.dtype)):
                lp_t = lp_t * ratio
                cp_t = _scale_pose_translation(cp_t, ratio)
                if ep_t is not None:
                    ep_t = _scale_pose_translation(ep_t, ratio)

        if seg in segment_se3_meters:
            normalized_se3 = _scale_pose_translation(
                segment_se3_meters[seg], final_metric_scalar.reciprocal(),
            )
            cp_t = _apply_se3(cp_t, normalized_se3)
            if ep_t is not None:
                ep_t = _apply_se3(ep_t, normalized_se3)

        lp_parts.append(lp_t)
        cam_parts.append(cp_t)
        if ep_t is not None:
            ego_parts.append(ep_t)

    out = {
        "local_points": torch.cat(lp_parts, dim=1),
        "camera_poses": torch.cat(cam_parts, dim=1),
    }
    if len(ego_parts) == total_frames:
        out["ego_poses"] = torch.cat(ego_parts, dim=1)
    out["metric"] = final_metric
    return out


@torch.no_grad()
def run_streaming(
    model: nn.Module,
    imgs: torch.Tensor,
    *,
    num_frames: int,
    num_cameras: int,
    block_size: int,
    intrinsics: Optional[torch.Tensor] = None,
    cam_to_ego: Optional[torch.Tensor] = None,
    ego_conf_threshold: float = 0.9,
    disable_restart: bool = True,
    use_per_seg_metric: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Run Drive3R block-streaming inference.

    Set ``block_size <= 0`` for a single full-clip forward.
    Set ``disable_restart=False`` (CLI: ``--enable-restart``) to let the ego-
    confidence head reset the KV cache when the per-block confidence
    ``sigmoid(ego_conf_logit) <= ego_conf_threshold`` (default ``0.9``) and
    stitch segments via SE(3) alignment on the per-frame ego poses over the
    overlap window (these already absorb the learned ``ego_delta``, so the
    stitched trajectory is consistent with the ego head output). Segment
    scales are finalized before converting overlap poses to meters and
    estimating these alignments. When ego
    poses are unavailable for the overlap frames
    (e.g. a 1-frame block where the ego head is skipped), the segment falls
    back to cam-0 alignment.

    ``ego_conf_threshold`` is interpreted as a *probability* in ``[0, 1]``
    (confidence target is in ``[0, 1]`` -- a sigmoid maps the
    raw head logit back into that space). ``stats['ego_conf_logits']`` is
    kept for backward compatibility; ``stats['ego_conf_probs']`` mirrors it
    after the sigmoid for easy inspection.

    ``cam_to_ego`` (B, T*V, 4, 4) is the per-view camera->ego rigid extrinsic.
    When provided, ``pred["ego_poses"]`` is computed in the actual ego frame
    via ``cam0_pred @ inv(cam_to_ego_cam0) @ ego_delta`` (matches the
    calibrated pose definition). When None, the model falls back to
    ``cam0_pred @ ego_delta`` (ego frame == cam0 frame).
    """
    B, total_N = int(imgs.shape[0]), int(imgs.shape[1])
    assert B == 1, f"Streaming inference assumes B=1, got {B}."
    V = int(num_cameras)
    T = int(num_frames)
    if T <= 0 or V <= 0:
        raise ValueError("num_frames and num_cameras must be positive.")
    if not 0.0 <= ego_conf_threshold <= 1.0:
        raise ValueError("ego_conf_threshold must be in [0, 1].")
    if T * V != total_N:
        raise ValueError(f"T*V mismatch: {T}*{V} != {total_N}")

    if int(block_size) <= 0:
        pred = model(
            imgs,
            with_prior=False,
            num_frames=T,
            num_cameras=V,
            intrinsics=intrinsics,
            cam_to_ego=cam_to_ego,
        )
        eco = pred.get("ego_conf_logits")
        conf_logits = (
            [float(x) for x in eco.reshape(-1).detach().cpu().tolist()]
            if isinstance(eco, torch.Tensor) and eco.numel() > 0 else []
        )
        conf_probs = [float(torch.sigmoid(torch.tensor(x)).item()) for x in conf_logits]
        return pred, {
            "mode": "offline_full_forward",
            "ego_conf_logits": conf_logits,
            "ego_conf_probs": conf_probs,
            "restart_events": [],
            "restart_count": 0,
            "ego_conf_threshold_space": "prob",
        }

    bs = max(int(block_size), 1)
    cursor = 0
    past_kv = None
    ego_conf_reference = None
    frame_lp: Dict[int, torch.Tensor] = {}
    frame_cp: Dict[int, torch.Tensor] = {}
    frame_ep: Optional[Dict[int, torch.Tensor]] = {}
    frame_seg_id: Dict[int, int] = {}     # per-frame restart-segment index (0-based)
    seg_metric: Dict[int, torch.Tensor] = {}  # per-segment metric (overwritten each block,
                                              # so it ends up holding the segment's *last* block metric)
    cur_seg_id = 0
    segment_alignment_pairs: Dict[
        int, Tuple[torch.Tensor, torch.Tensor, List[int]]
    ] = {}
    conf_logits: List[float] = []
    restart_events: List[Dict[str, object]] = []
    last_metric = None
    pending_restart_start: Optional[int] = None

    while cursor < T:
        block_start = int(cursor)
        block_end = min(block_start + bs, T)
        blk_T = block_end - block_start
        view_start = block_start * V
        view_end = block_end * V
        block_imgs = imgs[:, view_start:view_end]
        block_intrinsics = intrinsics[:, view_start:view_end] if intrinsics is not None else None
        block_cam_to_ego = (
            cam_to_ego[:, view_start:view_end] if isinstance(cam_to_ego, torch.Tensor) else None
        )

        is_first_segment_block = past_kv is None
        pred, past_kv, _hidden, _pos = model.forward_one_block(
            block_imgs,
            past_key_values=past_kv,
            frame_offset=block_start,
            block_size=bs,
            num_frames=blk_T,
            num_cameras=V,
            intrinsics=block_intrinsics,
            cam_to_ego=block_cam_to_ego,
            ego_conf_reference=ego_conf_reference,
        )
        if ego_conf_reference is None and getattr(model, '_ego_conf_enabled', False):
            ego_conf_reference = model._first_frame_reference(_hidden, _pos, B, blk_T, V)
        last_metric = pred.get("metric", last_metric)
        # Track per-segment metric. Each block overwrites the current segment's
        # entry, so once the segment ends ``seg_metric[cur_seg_id]`` holds the
        # metric from the segment's *last* block. ``last_metric`` keeps the
        # global "metric from the most recent block ever forwarded".
        if isinstance(pred.get("metric"), torch.Tensor):
            seg_metric[cur_seg_id] = pred["metric"].detach()

        lp = pred["local_points"]              # (1, blk_T*V, H, W, 3)
        cp = pred["camera_poses"]              # (1, blk_T*V, 4, 4)
        ep = pred.get("ego_poses")             # (1, blk_T, 4, 4) or None
        if ep is None:
            ep = _fallback_ego_from_cam0(
                cp,
                block_cam_to_ego,
                num_frames=blk_T,
                num_cameras=V,
            )

        if pending_restart_start is not None and block_start == pending_restart_start:
            overlap_frames = list(range(block_start, min(block_start + max(bs - 1, 1), T)))
            valid = [f for f in overlap_frames if f in frame_cp]
            if valid:
                # Prefer aligning on per-frame ego poses (already contain the
                # learned ``ego_delta``); fall back to cam-0 alignment when ego
                # is unavailable for any overlap frame.
                use_ego_align = (
                    frame_ep is not None
                    and isinstance(ep, torch.Tensor)
                    and all(f in frame_ep for f in valid)
                )
                if use_ego_align:
                    old_align = torch.cat([frame_ep[f] for f in valid], dim=1)
                    new_align = _collect_block_pose_subset(ep, block_start, valid, 1)
                else:
                    old_overlap = torch.cat([frame_cp[f] for f in valid], dim=1)
                    new_overlap = _collect_block_pose_subset(cp, block_start, valid, V)
                    old_align = old_overlap[:, 0::V]
                    new_align = new_overlap[:, 0::V]
                segment_alignment_pairs[cur_seg_id] = (
                    old_align, new_align, [frame_seg_id[f] for f in valid],
                )
            pending_restart_start = None

        for t_local in range(blk_T):
            t_global = block_start + t_local
            s = t_local * V
            e = (t_local + 1) * V
            frame_lp[t_global] = lp[:, s:e].clone()
            frame_cp[t_global] = cp[:, s:e].clone()
            if frame_ep is not None and isinstance(ep, torch.Tensor):
                frame_ep[t_global] = ep[:, t_local: t_local + 1].clone()
            # Stamp segment id last: if a later block (same segment) re-emits
            # the same t_global (e.g. overlap after a restart), the segment id
            # is updated to the latest block's segment, which is what we want.
            frame_seg_id[t_global] = cur_seg_id

        eco = pred.get("ego_conf_logits")
        logit = None
        prob = None
        if isinstance(eco, torch.Tensor) and eco.numel() > 0:
            logit = float(eco.reshape(-1)[-1].item())
            # sigmoid in float to avoid bf16 overflow/precision loss.
            prob = float(torch.sigmoid(eco.reshape(-1)[-1].float()).item())
            conf_logits.append(logit)

        # ``ego_conf_threshold`` is in probability space (0-1); compare against
        # ``sigmoid(logit)`` so the threshold matches the BCE-with-logits target.
        should_restart = (
            (not disable_restart)
            and (not is_first_segment_block)  # the reference block establishes confidence context
            and (prob is not None)
            and (prob <= float(ego_conf_threshold))
        )
        if should_restart and block_end < T:
            restart_events.append(
                {
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                    "trigger_logit": float(logit),
                    "trigger_prob": float(prob),
                    "next_start": int(block_start + 1),
                }
            )
            past_kv = None
            ego_conf_reference = None
            pending_restart_start = int(block_start + 1)
            cursor = int(block_start + 1)
            cur_seg_id += 1   # next block belongs to a new restart segment
        else:
            cursor = block_end

    # Use the final segment's last-block metric as the global ``pred["metric"]``;
    # Align finalized segments in meters, then normalize the combined scene
    # so exporters recover meters by multiplying by pred["metric"].
    final_metric = seg_metric.get(cur_seg_id, last_metric)
    pred_full = _aggregate_scene(
        frame_lp, frame_cp, frame_ep,
        total_frames=T, last_metric=final_metric,
        device=imgs.device, dtype=imgs.dtype,
        frame_seg_id=frame_seg_id,
        seg_metric=seg_metric if use_per_seg_metric else None,
        segment_alignment_pairs=segment_alignment_pairs,
    )
    # Per-segment metric breakdown for stats / downstream inspection.
    seg_metric_values: List[float] = []
    for sid in sorted(seg_metric.keys()):
        try:
            seg_metric_values.append(float(seg_metric[sid].reshape(-1)[0].item()))
        except Exception:
            seg_metric_values.append(float("nan"))
    conf_probs = [float(torch.sigmoid(torch.tensor(x)).item()) for x in conf_logits]
    stats: Dict[str, object] = {
        "mode": "block_streaming",
        "block_size": bs,
        "ego_conf_logits": conf_logits,
        "ego_conf_probs": conf_probs,
        "restart_events": restart_events,
        "restart_count": int(sum(1 for ev in restart_events if ev.get("trigger_logit") is not None)),
        "ego_conf_threshold": float(ego_conf_threshold),
        "ego_conf_threshold_space": "prob",
        "disable_restart": bool(disable_restart),
        "num_segments": int(len(seg_metric)),
        "seg_metric": seg_metric_values,
    }
    return pred_full, stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (xyzw) quaternions."""
    m = R
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.zeros(R.shape[:-2] + (4,), dtype=np.float64)
    cond1 = trace > 0
    s = np.sqrt(np.clip(trace[cond1] + 1.0, a_min=1e-12, a_max=None)) * 2.0
    q[cond1, 3] = 0.25 * s
    q[cond1, 0] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s
    q[cond1, 1] = (m[cond1, 0, 2] - m[cond1, 2, 0]) / s
    q[cond1, 2] = (m[cond1, 1, 0] - m[cond1, 0, 1]) / s

    rest = ~cond1
    if np.any(rest):
        idx = np.argmax(np.stack(
            [m[rest, 0, 0], m[rest, 1, 1], m[rest, 2, 2]], axis=-1
        ), axis=-1)
        for k, mr in enumerate(np.where(rest)[0]):
            mat = m[mr]
            i = int(idx[k])
            j = (i + 1) % 3
            l = (i + 2) % 3
            s = np.sqrt(max(mat[i, i] - mat[j, j] - mat[l, l] + 1.0, 1e-12)) * 2.0
            qv = np.zeros(4)
            qv[3] = (mat[l, j] - mat[j, l]) / s
            qv[i] = 0.25 * s
            qv[j] = (mat[j, i] + mat[i, j]) / s
            qv[l] = (mat[l, i] + mat[i, l]) / s
            q[mr] = qv
    return q


def _save_tum(path: str, c2w: np.ndarray, *, metric_scale: float) -> None:
    """Write TUM poses with translations scaled to meters and rotations unchanged."""
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3] * float(metric_scale)
    q = _rotmat_to_quat_xyzw(R)
    with open(path, "w") as f:
        for i in range(c2w.shape[0]):
            f.write(
                f"{i:.6f} {t[i, 0]:.6f} {t[i, 1]:.6f} {t[i, 2]:.6f} "
                f"{q[i, 0]:.6f} {q[i, 1]:.6f} {q[i, 2]:.6f} {q[i, 3]:.6f}\n"
            )


def _local_to_world(local_points: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """``local_points`` (T*V, H, W, 3) + ``c2w`` (T*V, 4, 4)  ->  (T*V, H, W, 3) world points."""
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]
    world = torch.einsum("nij,nhwj->nhwi", R, local_points) + t[..., None, None, :]
    return world


def _build_world_pointcloud(
    imgs: torch.Tensor,
    local_points: torch.Tensor,
    c2w: torch.Tensor,
    metric: torch.Tensor,
    *,
    max_points: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, rgb)`` numpy arrays in world frame, optionally sub-sampled."""
    rgb = imgs[0].permute(0, 2, 3, 1).contiguous()                  # (N, H, W, 3) in [0, 1]
    lp = local_points[0]                                            # (N, H, W, 3)
    cp = c2w[0]                                                     # (N, 4, 4)
    metric_scale = float(metric.reshape(-1)[0].item()) if metric is not None else 1.0
    world = _local_to_world(lp, cp) * metric_scale                  # (N, H, W, 3)

    xyz = world.reshape(-1, 3).detach().cpu().numpy()
    col = rgb.reshape(-1, 3).detach().cpu().numpy()

    valid = np.isfinite(xyz).all(axis=1)
    xyz, col = xyz[valid], col[valid]

    if xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz, col = xyz[idx], col[idx]
    print(f"[infer] world pointcloud: {xyz.shape[0]:,} points "
          f"(metric_scale={metric_scale:.4f})")
    return xyz, col


# ---------------------------------------------------------------------------
# viser visualization
# ---------------------------------------------------------------------------


def _matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion (wxyz)."""
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[0] < 0:
        quat *= -1.0
    return quat.astype(np.float32)


def _derive_cam_c2w_via_ego(
    pred_ego_poses: torch.Tensor,   # (1, T, 4, 4) ego c2w from model (incl. ego_delta)
    cam_to_ego: torch.Tensor,       # (1, T*V, 4, 4) cam->ego rigid extrinsic (meters)
    metric_scale: float,
    T: int,
    V: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive per-camera c2w from the model's ego trajectory (with ``ego_delta``).

    The model computes
        ``pred["ego_poses"][t] = cam0_pred[t] @ inv(cam_to_ego[t, 0]) @ ego_delta[t]``
    so a single canonical ego pose per frame already exists. We just need to
    lift translations to meters and propagate to per-camera via the GT rig.

    Steps:

    1. ``ego_m[t] = scale_trans(pred_ego_poses[t], metric_scale)``    (T, 4, 4) meters
    2. ``cam_m[t, v] = ego_m[t] @ cam_to_ego[t, v]``                  (T, V, 4, 4) meters

    ``ego_delta`` is included implicitly via ``pred_ego_poses``.

    Returns:
        cam_c2w_m: (T, V, 4, 4) per-camera c2w in meters.
        ego_m   : (T, 4, 4) ego c2w track in meters.
    """
    eg = pred_ego_poses[0].float().reshape(T, 4, 4).cpu()
    c2e = cam_to_ego[0].float().reshape(T, V, 4, 4).cpu()

    ego_m = eg.clone()
    ego_m[..., :3, 3] = ego_m[..., :3, 3] * float(metric_scale)      # (T, 4, 4) in meters
    cam_c2w_m = torch.matmul(ego_m.unsqueeze(1), c2e)                # (T, V, 4, 4)
    return cam_c2w_m, ego_m


def _build_viser_point_cloud(
    imgs: torch.Tensor,
    local_points: torch.Tensor,
    cam_c2w_meter: torch.Tensor,   # (T, V, 4, 4) c2w in METERS (already scaled)
    metric_scale: float,
    T: int,
    V: int,
    *,
    stride: int,
    max_points: int,
    min_depth: float,
    max_depth: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, rgb_uint8)`` in metric world frame, stride-sub-sampled.

    ``cam_c2w_meter`` is expected in METERS (translations already × metric_scale
    by the caller's selected camera-to-world derivation.
    ``local_points`` is in the model's normalized scale; this function applies
    ``metric_scale`` to it internally.
    """
    rgb = imgs[0].permute(0, 2, 3, 1).contiguous()                     # (N, H, W, 3)
    lp = local_points[0]                                               # (N, H, W, 3)
    cp = cam_c2w_meter.reshape(T * V, 4, 4).to(device=lp.device, dtype=lp.dtype)
    lp_m = lp * float(metric_scale)                                    # (N, H, W, 3) in meters
    world = _local_to_world(lp_m, cp)                                  # (N, H, W, 3) in meters

    # Per-point depth in camera frame (z-component of local_points, in meters).
    depth = (lp[..., 2] * float(metric_scale)).detach().cpu().numpy()  # (N, H, W)

    if stride > 1:
        world_s = world[:, ::stride, ::stride, :]
        rgb_s = rgb[:, ::stride, ::stride, :]
        depth_s = depth[:, ::stride, ::stride]
    else:
        world_s = world
        rgb_s = rgb
        depth_s = depth

    xyz = world_s.reshape(-1, 3).detach().cpu().numpy()
    col = rgb_s.reshape(-1, 3).detach().cpu().numpy()
    dep = depth_s.reshape(-1)

    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(dep)
    if min_depth > 0.0:
        mask &= dep > float(min_depth)
    if max_depth > 0.0:
        mask &= dep < float(max_depth)

    xyz = xyz[mask].astype(np.float32)
    col = col[mask]
    col = (np.clip(col, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    if max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
        xyz = xyz[keep]
        col = col[keep]
    return xyz, col


def _resize_image_for_frustum(img_hw3: np.ndarray, target_w: int = 128) -> np.ndarray:
    """Down-sample (H, W, 3) uint8 image to <= ``target_w`` width, preserving aspect."""
    h, w = img_hw3.shape[:2]
    if w <= target_w:
        return img_hw3
    new_w = int(target_w)
    new_h = max(1, int(round(h * new_w / w)))
    pil = Image.fromarray(img_hw3)
    return np.asarray(pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR))


def _serve_viser(
    pred: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    cam_to_ego: torch.Tensor,    # (1, T*V, 4, 4) cam->ego in meters; required
    T: int,
    V: int,
    args,
) -> None:
    """Serve point cloud + camera frustums + ego trajectory via viser.

    Geometry derivation always uses ``ego_delta``: per-camera poses are
    reconstructed as ``cam[t, v] = ego[t] @ cam_to_ego[t, v]``
    where ``ego[t] = pred["ego_poses"][t]`` (which is itself
    ``cam0 @ inv(cam_to_ego_0) @ ego_delta`` produced by the model).
    ``pred["camera_poses"]`` for ``v > 0`` is NOT used here; the GT rig is
    folded in via ``cam_to_ego``.
    """
    try:
        import viser  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "viser is not installed in this environment. Install it with "
            "`pip install viser`, then rerun with --viser."
        ) from exc

    if cam_to_ego is None:
        raise ValueError(
            "--viser requires cam_to_ego (the ego->cam derivation needs the GT rig). "
            "Provide it via the manifest 'cam_to_ego' field or --cam-to-ego-json."
        )
    pred_ego_poses = pred.get("ego_poses")
    if not isinstance(pred_ego_poses, torch.Tensor):
        raise RuntimeError(
            "--viser requires pred['ego_poses'] (= cam0 @ inv(cam_to_ego_0) @ "
            "ego_delta). Enable the ego head and pass cam_to_ego through "
            "run_streaming so the in-model formula is exercised."
        )

    metric_scale = float(pred["metric"].reshape(-1)[0].item())

    # --- Derive per-camera c2w via the ego chain (uses ego_delta via pred['ego_poses']) ---
    cam_c2w_m_t, ego_m_t = _derive_cam_c2w_via_ego(
        pred_ego_poses.float(),
        cam_to_ego.float(),
        metric_scale,
        T,
        V,
    )

    # --- Point cloud (metric world frame, ego-chain cameras) ---
    xyz, rgb = _build_viser_point_cloud(
        imgs.float(),
        pred["local_points"].float(),
        cam_c2w_m_t,
        metric_scale,
        T,
        V,
        stride=int(args.viser_point_stride),
        max_points=int(args.viser_max_points),
        min_depth=float(args.viser_min_depth),
        max_depth=float(args.viser_max_depth),
        seed=int(args.seed),
    )
    print(
        f"[viser] point cloud: {xyz.shape[0]:,} points "
        f"(metric_scale={metric_scale:.4f}, "
        f"derivation=ego[t]@cam_to_ego[t,v] (incl. ego_delta))"
    )

    cam_c2w_m = cam_c2w_m_t.numpy()       # (T, V, 4, 4) meters
    ego_track_m = ego_m_t.numpy()         # (T, 4, 4) meters

    # --- Image tiles for frustums (resized down for transport) ---
    img_np = imgs[0].detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()  # (T*V, H, W, 3)
    img_np = (img_np * 255.0).round().astype(np.uint8).reshape(T, V, *img_np.shape[1:])
    H_img, W_img = img_np.shape[2:4]

    import viser  # re-import in local scope for type-hint friendliness
    server = viser.ViserServer(host=args.viser_host, port=int(args.viser_port))
    server.scene.world_axes.visible = True

    server.scene.add_point_cloud(
        "/drive3r/point_cloud",
        points=xyz,
        colors=rgb,
        point_size=float(args.viser_point_size),
    )

    # Camera frustums
    if args.viser_camera_mode != "none":
        if args.viser_camera_samples > 0 and args.viser_camera_samples < T:
            t_idxs = sorted({int(x) for x in np.linspace(0, T - 1, args.viser_camera_samples).round()})
        else:
            t_idxs = list(range(T))
        cam_idxs = [0] if args.viser_camera_mode == "cam0" else list(range(V))

        # Intrinsics: use predicted/loaded ones if available; otherwise infer fov from image size.
        # Default fy guess: focal ~= 0.85 * H for automotive cameras.
        default_fy = float(H_img) * 0.85
        for t in t_idxs:
            for v in cam_idxs:
                c2w = cam_c2w_m[t, v]
                name = f"/drive3r/cameras/t{t:03d}_cam{v}"
                fy = default_fy
                fov = 2.0 * float(np.arctan(float(H_img) / (2.0 * fy)))
                kwargs = {
                    "fov": fov,
                    "aspect": float(W_img) / float(H_img),
                    "scale": float(args.viser_frustum_scale),
                    "wxyz": _matrix_to_wxyz(c2w[:3, :3]),
                    "position": c2w[:3, 3].astype(np.float32),
                }
                if args.viser_show_images:
                    kwargs["image"] = _resize_image_for_frustum(img_np[t, v], target_w=128)
                try:
                    server.scene.add_camera_frustum(name, **kwargs)
                except TypeError:
                    kwargs.pop("image", None)
                    server.scene.add_camera_frustum(name, **kwargs)

    # Ego trajectory polyline + small frames
    if ego_track_m is not None:
        ego_xyz = ego_track_m[:, :3, 3].astype(np.float32)
        if ego_xyz.shape[0] >= 2:
            # add_spline_catmull_rom or simple line; viser exposes add_spline_catmull_rom
            try:
                server.scene.add_spline_catmull_rom(
                    "/drive3r/ego/trajectory",
                    positions=ego_xyz,
                    color=(255, 196, 0),
                    line_width=3.0,
                )
            except AttributeError:
                # Fallback: very thin point cloud as polyline
                server.scene.add_point_cloud(
                    "/drive3r/ego/trajectory",
                    points=ego_xyz,
                    colors=np.tile(np.array([[255, 196, 0]], dtype=np.uint8), (ego_xyz.shape[0], 1)),
                    point_size=max(float(args.viser_point_size) * 3.0, 0.05),
                )
        if args.viser_ego_frame_samples > 0:
            step = max(1, T // int(args.viser_ego_frame_samples))
            sample_ts = list(range(0, T, step))
            for t in sample_ts:
                pose = ego_track_m[t]
                server.scene.add_frame(
                    f"/drive3r/ego/frame_t{t:03d}",
                    wxyz=_matrix_to_wxyz(pose[:3, :3]),
                    position=pose[:3, 3].astype(np.float32),
                    axes_length=float(args.viser_frustum_scale) * 0.8,
                    axes_radius=max(float(args.viser_frustum_scale) * 0.04, 0.02),
                )

    print(
        f"[viser] serving at http://{args.viser_host}:{args.viser_port}  "
        f"(T={T}, V={V}, points={xyz.shape[0]:,}, "
        f"camera_mode={args.viser_camera_mode}, samples={args.viser_camera_samples})",
        flush=True,
    )
    print("[viser] press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[viser] stopped.", flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_model(
    ckpt_path: str,
    device: torch.device,
    *,
    block_size: int,
    cross_attn_mode: str,
    use_multimodal: bool,
    enable_intrinsics_injection: bool,
    enable_extrinsics_injection: bool,
    use_ego_pose: bool,
    enable_ego_conf_head: bool,
    global_use_3d_rope: bool,
    dtype: torch.dtype,
) -> Drive3R:
    model = Drive3R(
        ckpt=None,
        use_multimodal=use_multimodal,
        enable_intrinsics_injection=enable_intrinsics_injection,
        enable_extrinsics_injection=enable_extrinsics_injection,
        cross_attn_mode=cross_attn_mode,
        block_size=int(block_size if block_size > 0 else 0),
        use_ego_pose=use_ego_pose,
        enable_ego_conf_head=enable_ego_conf_head,
        enable_cam_conf_head=None,
        global_use_3d_rope=global_use_3d_rope,
    )
    tensor_count, result = load_model_weights(model, ckpt_path)
    print(f"[infer] loaded {tensor_count} tensors from {ckpt_path}")
    if result.unexpected_keys:
        print(f"[infer] ignored extra keys (first 5): {result.unexpected_keys[:5]}")
    model = model.to(device=device, dtype=dtype).eval()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Drive3R streaming inference.",
    )
    g_in = p.add_argument_group("inputs")
    source = g_in.add_mutually_exclusive_group()
    source.add_argument("--input-path", "--input_path", type=str, default=None,
                        help="Scene folder with images/<frame>_<camera>.* and camera/ calibration.")
    source.add_argument("--image-dir", type=str, default=None,
                      help="Scene directory; either with V camera sub-folders or a flat sequence (V=1).")
    source.add_argument("--manifest", type=str, default=None,
                      help="JSON manifest with explicit per-frame, per-camera image paths.")
    g_in.add_argument("--num-cameras", "--num_views", type=int, default=None)
    g_in.add_argument("--num-frames", "--num_frames", type=int, default=None)
    g_in.add_argument("--stride", type=int, default=1, help="Frame stride for --input-path.")
    g_in.add_argument("--start-index", type=int, default=0, help="First frame index for --input-path.")
    g_in.add_argument("--cam-to-ego-json", type=str, default=None,
                      help="JSON with cam->ego SE(3) extrinsics (length V or T*V list of 4x4 "
                           "matrices). REQUIRED when the ego-pose head is enabled. ego_poses "
                           "are computed in the actual vehicle ego frame via "
                           "cam0 @ inv(cam_to_ego_cam0) @ ego_delta.")
    g_in.add_argument("--intrinsics-json", type=str, default=None,
                      help="Optional JSON file with a list of 3x3 K matrices (length 1, V or T*V).")
    g_in.add_argument("--resolution", type=str, default=None,
                      help="WxH at which to run the model (must be multiples of 14). "
                           "Defaults to a pixel-budget driven resize.")
    g_in.add_argument("--pixel-limit", type=int, default=200_000,
                      help="Pixel budget (W*H) when --resolution is not given.")

    g_m = p.add_argument_group("model")
    g_m.add_argument("--ckpt", "--checkpoint", type=str, required=True, help="Drive3R checkpoint (.safetensors / .pt / .bin).")
    g_m.add_argument("--cross-attn-mode", type=str, default="global", choices=["cross", "global"])
    g_m.add_argument("--use-multimodal", action="store_true",
                     help="Enable intrinsics / extrinsics injection branches.")
    g_m.add_argument("--enable-intrinsics-injection", action="store_true")
    g_m.add_argument("--enable-extrinsics-injection", action="store_true")
    g_m.add_argument("--no-ego-pose", dest="use_ego_pose", action="store_false")
    g_m.add_argument("--no-ego-conf", dest="enable_ego_conf_head", action="store_false")
    g_m.add_argument("--global-use-3d-rope", action="store_true")
    g_m.set_defaults(use_ego_pose=True, enable_ego_conf_head=True)

    g_r = p.add_argument_group("runtime")
    g_r.add_argument("--block-size", type=int, default=3,
                     help="Streaming block size (0 = single full-clip forward).")
    g_r.add_argument("--ego-conf-threshold", type=float, default=0.9,
                     help="If --enable-restart, restart whenever the per-block "
                          "ego confidence (= sigmoid of the head logit, in [0, 1]) "
                          "is <= this value. Defaults to 0.9 (i.e. restart when "
                          "predicted block confidence drops below 90%%).")
    g_r.add_argument("--enable-restart", action="store_true",
                     help="Allow ego_conf-driven KV-cache restarts (off by default).")
    g_r.add_argument("--device", type=str, default="cuda")
    g_r.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    g_r.add_argument("--seed", type=int, default=0)

    g_o = p.add_argument_group("outputs")
    g_o.add_argument("--output-dir", "--output_dir", type=str, required=True)
    g_o.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false")
    g_o.add_argument("--max-points", type=int, default=2_000_000)
    g_o.set_defaults(save_pointcloud=True)

    g_v = p.add_argument_group("viser visualization")
    g_v.add_argument("--viser", action="store_true",
                     help="After inference, serve an interactive viser scene "
                          "(point cloud + camera frustums + ego trajectory).")
    g_v.add_argument("--viser-host", type=str, default="0.0.0.0")
    g_v.add_argument("--viser-port", type=int, default=8080)
    g_v.add_argument("--viser-point-stride", type=int, default=2,
                     help="Pixel-grid stride when fusing the world point cloud for viser.")
    g_v.add_argument("--viser-max-points", type=int, default=600_000,
                     help="Random-sample the world point cloud down to at most N points "
                          "before streaming to viser (0 = keep all).")
    g_v.add_argument("--viser-point-size", type=float, default=0.05)
    g_v.add_argument("--viser-min-depth", type=float, default=0.5,
                     help="Drop predicted points whose camera-frame depth is below this (meters).")
    g_v.add_argument("--viser-max-depth", type=float, default=80.0,
                     help="Drop predicted points whose camera-frame depth is above this (meters).")
    g_v.add_argument("--viser-camera-mode", type=str, default="cam0",
                     choices=["none", "cam0", "all"],
                     help="Which camera frustums to draw per frame.")
    g_v.add_argument("--viser-camera-samples", type=int, default=12,
                     help="Number of evenly-spaced frame indices to show frustums for "
                          "(0 = all frames).")
    g_v.add_argument("--viser-frustum-scale", type=float, default=0.6)
    g_v.add_argument("--viser-show-images", action="store_true",
                     help="Render input camera image inside each frustum (heavier transport).")
    g_v.add_argument("--viser-ego-frame-samples", type=int, default=8,
                     help="Number of evenly-spaced ego coordinate frames to draw along "
                          "the ego trajectory (0 = none).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.block_size > 0 and args.cross_attn_mode != "global":
        raise ValueError("Streaming (--block-size > 0) requires --cross-attn-mode global.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    # Keep weights in fp32 and run the
    # forward under torch.autocast(bf16/fp16). Casting the whole model to
    # bf16 breaks the autocast(enabled=False) block in Drive3R.encode
    # (``ray_embed`` would otherwise see bf16 weights but float-32 rays).
    autocast_dtype = {
        "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    }[args.dtype]
    use_autocast = (autocast_dtype != torch.float32) and torch.cuda.is_available()
    weight_dtype = torch.float32        # on-device weight / input dtype

    os.makedirs(args.output_dir, exist_ok=True)

    imgs, intrinsics, cam_to_ego, T, V, paths = _build_clip_inputs(args)
    imgs = imgs.to(device=device, dtype=weight_dtype)
    if intrinsics is not None:
        # Keep intrinsics in float32: torch.inverse / linalg.inv inside the model
        # encoder do not support bf16/fp16. The linear ray_embed layer happily
        # casts back to model dtype.
        intrinsics = intrinsics.to(device=device, dtype=torch.float32)

    # cam_to_ego is required when the ego-pose head is enabled: ego_poses are
    # only ever computed as cam0 @ inv(cam_to_ego_cam0) @ ego_delta (matches the
    # calibrated pose definition). Disable the ego head with --no-ego-pose if you do
    # not have cam_to_ego. ``--viser`` also requires cam_to_ego because the
    # visualization uses the ego-rig chain ``cam[t,v] = ego[t] @ c2e[t,v]``.
    if (args.use_ego_pose or args.viser) and cam_to_ego is None:
        raise ValueError(
            "Ego-pose head is enabled (or --viser is requested) but cam_to_ego "
            "was not provided. Supply it via the manifest 'cam_to_ego' field or "
            "--cam-to-ego-json (length V or T*V list of 4x4 cam->ego SE(3) "
            "matrices), or pass --no-ego-pose AND drop --viser to skip."
        )
    if cam_to_ego is not None:
        # Same float32 rationale as intrinsics: inv() in the model uses fp32.
        cam_to_ego = cam_to_ego.to(device=device, dtype=torch.float32)
    # Only forward cam_to_ego to the model when the ego head is enabled.
    cam_to_ego_for_model = cam_to_ego if args.use_ego_pose else None

    print(f"[infer] device={device}, dtype={args.dtype} (autocast={use_autocast}), "
          f"block_size={args.block_size}, "
          f"ego_restart={'on' if args.enable_restart else 'off'}, "
          f"ego_head={'on' if args.use_ego_pose else 'off'}")

    model = build_model(
        args.ckpt, device,
        block_size=args.block_size,
        cross_attn_mode=args.cross_attn_mode,
        use_multimodal=args.use_multimodal,
        enable_intrinsics_injection=args.enable_intrinsics_injection,
        enable_extrinsics_injection=args.enable_extrinsics_injection,
        use_ego_pose=args.use_ego_pose,
        enable_ego_conf_head=args.enable_ego_conf_head,
        global_use_3d_rope=args.global_use_3d_rope,
        dtype=weight_dtype,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if use_autocast else _NullCtx()
    )

    t0 = time.time()
    with autocast_ctx:
        pred, stats = run_streaming(
            model,
            imgs,
            num_frames=T,
            num_cameras=V,
            block_size=args.block_size,
            intrinsics=intrinsics,
            cam_to_ego=cam_to_ego_for_model,
            ego_conf_threshold=args.ego_conf_threshold,
            disable_restart=not args.enable_restart,
        )
    elapsed = time.time() - t0
    stats["elapsed_sec"] = float(elapsed)
    stats["num_frames"] = int(T)
    stats["num_cameras"] = int(V)
    stats["image_paths"] = paths
    print(f"[infer] forward: {elapsed:.2f}s ({T} frames, {V} cams, block={args.block_size})")

    # ---- Save tensors ----
    npz_path = os.path.join(args.output_dir, "pred.npz")
    npz_payload: Dict[str, np.ndarray] = {
        "local_points": pred["local_points"].float().cpu().numpy(),
        "camera_poses": pred["camera_poses"].float().cpu().numpy(),
        "metric": pred["metric"].float().cpu().numpy(),
        "num_frames": np.asarray([T], dtype=np.int64),
        "num_cameras": np.asarray([V], dtype=np.int64),
    }
    if "ego_poses" in pred and isinstance(pred["ego_poses"], torch.Tensor):
        npz_payload["ego_poses"] = pred["ego_poses"].float().cpu().numpy()
    np.savez_compressed(npz_path, **npz_payload)
    print(f"[infer] wrote {npz_path}")

    # ---- TUM trajectories ----
    metric_scale = float(pred["metric"].reshape(-1)[0].item())
    cam_poses = pred["camera_poses"][0].float().cpu().numpy().reshape(T, V, 4, 4)
    cam0_track = cam_poses[:, 0]
    _save_tum(os.path.join(args.output_dir, "trajectory_cam0.txt"), cam0_track,
              metric_scale=metric_scale)
    if "ego_poses" in pred and isinstance(pred["ego_poses"], torch.Tensor):
        ego_track = pred["ego_poses"][0].float().cpu().numpy()
        _save_tum(os.path.join(args.output_dir, "trajectory_ego.txt"), ego_track,
                  metric_scale=metric_scale)

    # ---- World point cloud ----
    if args.save_pointcloud:
        ply_path = os.path.join(args.output_dir, "world_pointcloud.ply")
        xyz, rgb = _build_world_pointcloud(
            imgs.float(),
            pred["local_points"].float(),
            pred["camera_poses"].float(),
            pred["metric"].float(),
            max_points=args.max_points,
        )
        write_ply(xyz, rgb, ply_path)
        print(f"[infer] wrote {ply_path}")

    # ---- Stats ----
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[infer] done -> {args.output_dir}")

    # ---- viser visualization ----
    if args.viser:
        if cam_to_ego is None:
            raise ValueError(
                "--viser uses the ego-chain camera derivation (cam[t,v] = "
                "ego[t] @ cam_to_ego[t,v]) and therefore requires cam_to_ego. "
                "Provide it via the manifest 'cam_to_ego' field or "
                "--cam-to-ego-json (length V or T*V list of 4x4 cam->ego SE(3) "
                "matrices)."
            )
        _serve_viser(pred, imgs.float(), cam_to_ego, int(T), int(V), args)


if __name__ == "__main__":
    main()
