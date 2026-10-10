"""Densify sparse depth with MoGe-2 and affine metric alignment.

Use --input-path for a scene folder with images/ and sparse_depths/, or
--manifest for RGB images plus sparse depth / calibrated XYZ point samples.
See README.md for input formats.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Optional

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.scene_io import load_scene


@dataclass
class Sample:
    sample_id: str
    image: Path
    sparse_depth: Optional[Path]
    lidar_points: Optional[Path]
    intrinsic: Optional[np.ndarray]
    lidar_to_camera: Optional[np.ndarray]
    metadata: dict


def _matrix(value, shape, name):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {shape} matrix.")
    if shape == (4, 4):
        rotation = matrix[:3, :3]
        if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)):
            raise ValueError(f"{name} must be a rigid 4x4 transform.")
    elif (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
          or not np.allclose(matrix[2], [0, 0, 1], atol=1e-5)):
        raise ValueError(f"{name} must be a pixel-space pinhole intrinsic matrix.")
    return matrix


def _parse_samples(rows, base_dir):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a non-empty 'samples' list.")
    samples, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Sample {index} must be an object.")
        sample_id = row.get("id", f"{index:06d}")
        if (not isinstance(sample_id, str) or not sample_id
                or not re.fullmatch(r"[A-Za-z0-9_./-]+", sample_id)
                or PurePosixPath(sample_id).is_absolute()
                or any(part in ("", ".", "..") for part in sample_id.split("/"))):
            raise ValueError(f"Invalid sample id: {sample_id!r}; use a relative output path.")
        if sample_id.casefold() in seen:
            raise ValueError(f"Duplicate sample id: {sample_id!r}")
        seen.add(sample_id.casefold())

        def input_path(key, required=False):
            value = row.get(key)
            if value is None and not required:
                return None
            if not isinstance(value, str) or not value:
                raise ValueError(f"Sample {sample_id}: '{key}' must be a file path.")
            result = (base_dir / value).resolve()
            if not result.is_file():
                raise ValueError(f"Sample {sample_id}: file not found: {result}")
            return result

        image = input_path("image", required=True)
        sparse, points = input_path("sparse_depth"), input_path("lidar_points")
        if (sparse is None) == (points is None):
            raise ValueError(f"Sample {sample_id}: provide exactly one of sparse_depth or lidar_points.")
        intrinsic = _matrix(row["intrinsic"], (3, 3), "intrinsic") if "intrinsic" in row else None
        transform = _matrix(row["lidar_to_camera"], (4, 4), "lidar_to_camera") if "lidar_to_camera" in row else None
        if points is not None and (intrinsic is None or transform is None):
            raise ValueError(f"Sample {sample_id}: LiDAR input requires intrinsic and lidar_to_camera.")
        metadata = {key: _matrix(row[key], (4, 4), key)
                    for key in ("cam_to_ego", "ego_to_global") if key in row}
        samples.append(Sample(sample_id, image, sparse, points, intrinsic, transform, metadata))
    return samples


def load_manifest(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("samples") if isinstance(document, dict) else document
    return _parse_samples(rows, path.parent)


def load_scene_samples(root):
    root = Path(root).resolve()
    rows = load_scene(root)["samples"]
    for row in rows:
        for suffix in (".npy", ".npz"):
            sparse = root / "sparse_depths" / (row["id"] + suffix)
            if sparse.is_file():
                row["sparse_depth"] = str(sparse)
                break
        else:
            raise ValueError(f"Missing sparse depth for {row['id']} in {root / 'sparse_depths'}.")
    return _parse_samples(rows, root)


def load_array(path, key):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            selected = key if key in archive else archive.files[0] if len(archive.files) == 1 else None
            if selected is None:
                raise ValueError(f"{path}: expected '{key}' or a single array.")
            return np.asarray(archive[selected], dtype=np.float32)
    raise ValueError(f"{path}: expected a .npy or .npz file.")


def project_lidar_to_camera(points, intrinsic, lidar_to_camera, height, width):
    """Project undistorted OpenCV camera-Z depth; nearest point wins per pixel."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("LiDAR points must have shape (N, 3), in meters.")
    camera = points @ lidar_to_camera[:3, :3].T + lidar_to_camera[:3, 3]
    camera = camera[np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.1)]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
             & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    u, v = np.floor(uv[valid]).astype(np.int64).T
    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), camera[valid, 2])
    depth[~np.isfinite(depth)] = 0
    return depth


def read_sparse_depth(sample, height, width):
    if sample.sparse_depth is not None:
        depth = load_array(sample.sparse_depth, "sparse_depth")
        if depth.shape != (height, width):
            raise ValueError(f"{sample.sample_id}: sparse depth shape {depth.shape} != image {(height, width)}.")
        return np.where(np.isfinite(depth) & (depth > 0.1), depth, 0).astype(np.float32)
    return project_lidar_to_camera(load_array(sample.lidar_points, "points"), sample.intrinsic,
                                   sample.lidar_to_camera, height, width)


@torch.no_grad()
def process_camera(model, image, sparse_depth, device, args, aligner):
    v, u = np.where(sparse_depth > 0.1)
    if len(u) < args.min_lidar_points:
        return None, "too_few_lidar_points"
    if np.std(np.stack([u, v], axis=1), axis=0).mean() < args.min_lidar_spatial_std:
        return None, "low_lidar_spatial_std"
    prediction = model.infer(torch.from_numpy(image).permute(2, 0, 1).to(device),
                             use_fp16=device.type == "cuda")
    depth = torch.as_tensor(prediction["depth"], device=device).float()
    mask = torch.as_tensor(prediction["mask"], device=device).bool()
    if tuple(depth.shape) != sparse_depth.shape or tuple(mask.shape) != sparse_depth.shape:
        raise ValueError("MoGe depth and mask must match the original image dimensions.")
    mask = mask & torch.isfinite(depth) & (depth > 0.1)
    uv_mask = mask[v, u]
    if uv_mask.float().mean().item() < args.min_valid_overlap:
        return None, "low_valid_overlap"
    source = depth[v, u][uv_mask]
    target = torch.from_numpy(sparse_depth[v, u]).to(device)[uv_mask]
    if source.numel() < args.min_lidar_points:
        return None, "too_few_valid_align_points"
    # MoGe's affine solver compares candidate anchors pairwise. Bound that
    # quadratic work, while keeping quality evaluation on all sparse pixels.
    fit_source, fit_target = source, target
    if source.numel() > args.max_alignment_points:
        indices = torch.linspace(0, source.numel() - 1, args.max_alignment_points,
                                 device=device).round().long()
        fit_source, fit_target = source[indices], target[indices]
    scale, shift = aligner(fit_source[None], fit_target[None], torch.ones_like(fit_source[None]),
                           trunc=args.alignment_trunc)
    scale, shift = float(scale.item()), float(shift.item())
    if not np.isfinite(scale) or not 1.0 / args.max_scale <= scale <= args.max_scale:
        return None, "bad_scale"
    if not np.isfinite(shift) or abs(shift) > args.max_shift:
        return None, "bad_shift"
    aligned = depth * scale + shift
    at_lidar = aligned[v, u][uv_mask]
    if not torch.isfinite(at_lidar).all() or (at_lidar <= 0.1).any():
        return None, "invalid_aligned_depth"
    abs_rel = ((at_lidar - target).abs() / target).mean().item()
    delta_125 = (torch.maximum(at_lidar / target, target / at_lidar) < 1.25).float().mean().item()
    if abs_rel > args.max_abs_rel:
        return None, "high_abs_rel"
    if delta_125 < args.min_delta_125:
        return None, "low_delta_125"
    valid = mask & torch.isfinite(aligned) & (aligned > 0.1)
    return {"dense_depth": torch.where(valid, aligned, 0).cpu().numpy().astype(np.float32),
            "valid_mask": valid.cpu().numpy(), "scale": np.float32(scale), "shift": np.float32(shift),
            "abs_rel": np.float32(abs_rel), "delta_125": np.float32(delta_125)}, None


def run_generation(samples, output_dir, model, aligner, device, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, reasons, filtered = Counter(), Counter(), []
    for sample in samples:
        destination = output_dir / f"{sample.sample_id}.npz"
        if destination.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
            continue
        with Image.open(sample.image) as pil_image:
            image = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        sparse = read_sparse_depth(sample, *image.shape[:2])
        result, reason = process_camera(model, image, sparse, device, args, aligner)
        if result is None:
            if args.overwrite:
                destination.unlink(missing_ok=True)
            counts["filtered"] += 1
            reasons[reason] += 1
            filtered.append({"id": sample.sample_id, "reason": reason})
            print(f"[dense] {sample.sample_id}: filtered ({reason})")
            continue
        result.update(sample.metadata)
        if sample.intrinsic is not None:
            result["intrinsic"] = sample.intrinsic
        result["image_path"] = np.asarray(str(sample.image))
        if args.save_sparse_depth:
            result["sparse_depth"] = sparse
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **result)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        counts["saved"] += 1
        print(f"[dense] {sample.sample_id}: saved")
    report = {"total": len(samples), "saved": counts["saved"], "filtered": counts["filtered"],
              "skipped_existing": counts["skipped_existing"], "filtered_reasons": dict(reasons),
              "filtered_samples": filtered, "moge_model": args.moge_model,
              "thresholds": {key: getattr(args, key) for key in (
                  "min_lidar_points", "min_lidar_spatial_std", "max_abs_rel", "min_delta_125",
                  "min_valid_overlap", "max_scale", "max_shift", "alignment_trunc", "max_alignment_points")}}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--input-path", "--input_path", type=Path)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--moge-model", "--moge_model", default="Ruicheng/moge-2-vitl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-alignment-points", type=int, default=2048,
                        help="Maximum sparse points used by the affine solver; quality uses all valid points.")
    parser.add_argument("--save-sparse-depth", "--save-lidar-sparse", "--save_lidar_sparse", action="store_true")
    for name, default, kind in (
        ("min_lidar_points", 50, int), ("min_lidar_spatial_std", 5.0, float),
        ("max_abs_rel", 0.3, float), ("min_delta_125", 0.6, float),
        ("min_valid_overlap", 0.7, float), ("max_scale", 3.0, float),
        ("max_shift", 10.0, float), ("alignment_trunc", 5.0, float),
    ):
        parser.add_argument("--" + name.replace("_", "-"), "--" + name, type=kind, default=default)
    args = parser.parse_args(argv)
    values = (args.min_lidar_spatial_std, args.max_abs_rel, args.min_delta_125,
              args.min_valid_overlap, args.max_scale, args.max_shift, args.alignment_trunc)
    if (not all(np.isfinite(values)) or args.min_lidar_points < 3 or args.max_alignment_points < 3
            or min(args.min_lidar_spatial_std, args.max_abs_rel, args.max_shift) < 0
            or not 0 <= args.min_delta_125 <= 1 or not 0 <= args.min_valid_overlap <= 1
            or args.max_scale < 1 or args.alignment_trunc <= 0):
        parser.error("Invalid alignment or quality-filter thresholds.")
    return args


def main():
    args = get_args()
    samples = load_manifest(args.manifest) if args.manifest else load_scene_samples(args.input_path)
    device = torch.device(args.device)
    model = aligner = None
    if any(args.overwrite or not (args.output_dir / f"{sample.sample_id}.npz").exists() for sample in samples):
        try:
            from moge.model.v2 import MoGeModel
            from moge.utils.alignment import align_depth_affine
        except ImportError as error:
            raise RuntimeError("Install MoGe-2 first: pip install git+https://github.com/microsoft/MoGe.git") from error
        model = MoGeModel.from_pretrained(args.moge_model).to(device).eval()
        aligner = align_depth_affine
    report = run_generation(samples, args.output_dir, model, aligner, device, args)
    print(json.dumps({key: report[key] for key in ("total", "saved", "filtered", "skipped_existing")}))


if __name__ == "__main__":
    main()
