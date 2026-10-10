"""Load scene folders for Drive3R inference."""

from pathlib import Path

import numpy as np


def load_matrix(path, shape):
    path = Path(path)
    matrix = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=np.float32)
    if shape == (3, 3) and matrix.size == 4:
        fx, fy, cx, cy = matrix.reshape(-1)
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    elif matrix.size == shape[0] * shape[1]:
        matrix = matrix.reshape(shape)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: expected a finite {shape} matrix.")
    return matrix


def _find_file(directory, stems, suffixes):
    for stem in stems:
        for suffix in suffixes:
            path = directory / (stem + suffix)
            if path.is_file():
                return path
    return None


def load_scene(root, *, num_frames=None, num_cameras=None, stride=1, start_index=0):
    """Read images/<frame>_<camera>.* and camera calibration.

    Camera IDs are non-negative integers. Calibrations may be static per camera
    or per image. Every selected frame must contain the same set of cameras.
    """
    root = Path(root).resolve()
    if stride < 1 or start_index < 0 or (num_frames is not None and num_frames <= 0):
        raise ValueError("stride and num_frames must be positive; start_index must be non-negative.")
    if num_cameras is not None and num_cameras <= 0:
        raise ValueError("num_cameras must be positive.")
    images = {}
    for image in sorted((root / "images").glob("*")):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        pieces = image.stem.rsplit("_", 1)
        if len(pieces) != 2 or not pieces[0] or not pieces[1].isdigit():
            raise ValueError(f"{image}: expected <frame>_<integer-camera-id>.<extension>.")
        key = (pieces[0], int(pieces[1]))
        if key in images:
            raise ValueError(f"Duplicate image for frame/camera: {key}")
        images[key] = image
    if not images:
        raise ValueError(f"No scene images found in {root / 'images'}.")
    cameras = sorted({camera for _, camera in images})
    if num_cameras is not None and len(cameras) != num_cameras:
        raise ValueError(f"Expected {num_cameras} cameras, found {len(cameras)}.")
    frames = sorted({frame for frame, _ in images}, key=lambda frame: (0, int(frame)) if frame.isdigit() else (1, frame))
    frames = frames[start_index::stride]
    if num_frames is not None:
        if len(frames) < num_frames:
            raise ValueError(f"Requested {num_frames} frames, only {len(frames)} available.")
        frames = frames[:num_frames]
    if not frames:
        raise ValueError("No frames selected.")
    samples = []
    for frame in frames:
        for camera in cameras:
            if (frame, camera) not in images:
                raise ValueError(f"Missing image for frame {frame}, camera {camera}.")
            image = images[(frame, camera)]
            stems = [image.stem, str(camera)]
            row = {"id": image.stem, "image": str(image)}
            for name, directory, shape in (
                ("intrinsic", "intrinsics", (3, 3)), ("cam_to_ego", "cam2ego", (4, 4)),
            ):
                path = _find_file(root / "camera" / directory, stems, [".txt", ".npy"])
                if path is not None:
                    row[name] = load_matrix(path, shape)
            samples.append(row)
    return {"num_frames": len(frames), "num_cameras": len(cameras), "samples": samples}
