"""Shared contract for local, user-authorized dataset conversion."""
import json
from pathlib import Path

import cv2
import numpy as np


FORMAT = "nidar-camera-cloud-v1"
# Map the pipeline's forward/left/up camera coordinates to optical right/down/forward.
OPTICAL_FROM_PIPELINE = np.array([
    [0., -1., 0., 0.], [0., 0., -1., 0.],
    [1., 0., 0., 0.], [0., 0., 0., 1.],
])


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("must be positive")
    return value


def matrix(value):
    result = np.asarray(value, dtype=np.float64).reshape(4, 4)
    rotation = result[:3, :3]
    if (not np.isfinite(result).all() or
            not np.allclose(result[3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError("Expected a finite rigid 4x4 transform")
    return result


def transform_points(xyz, transform):
    transform = matrix(transform)
    return np.asarray(xyz) @ transform[:3, :3].T + transform[:3, 3]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def start_export(output, source, dataset, **metadata):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Keep export output separate from the official input path")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new, empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(format=FORMAT, source_dataset=dataset, export_status="incomplete",
                    point_coordinates="world", reference_frame="LIDAR_TOP",
                    camera_coordinates="forward-left-up", frames=[], **metadata)
    write_json(output / "input_manifest.json", manifest)
    (output / "images").mkdir()
    (output / "pointclouds").mkdir()
    return output, manifest


def finish_export(output, manifest):
    if not manifest["frames"]:
        raise ValueError("No frames exported")
    manifest["export_status"] = "complete"
    manifest["frame_count"] = len(manifest["frames"])
    write_json(output / "input_manifest.json", manifest)


def write_cloud(output, index, xyz_world, intensity, world_from_lidar):
    xyz = np.asarray(xyz_world, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or
            len(xyz) != len(intensity) or not np.isfinite(xyz).all() or
            not np.isfinite(intensity).all()):
        raise ValueError("Expected nonempty finite, aligned XYZ and intensity arrays")
    stem = output / "pointclouds" / f"frame_{index:06d}"
    np.savez_compressed(str(stem) + "_raw_pc.npz",
                        points=np.column_stack([xyz, intensity]).astype(np.float32))
    # The reference frame is the LiDAR sensor, despite the legacy key 'ego2world'.
    write_json(str(stem) + "_lidar1.json", {
        "extrinsic": np.eye(4).ravel().tolist(),
        "ego2world": matrix(world_from_lidar).ravel().tolist(),
        "reference_frame": "LIDAR_TOP",
    })


def write_camera(output, index, camera_id, jpeg, intrinsic, reference_from_camera, **metadata):
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if image is None:
        raise ValueError("Camera image cannot be decoded")
    if intrinsic.shape != (4,) or not np.isfinite(intrinsic).all() or np.any(intrinsic[:2] <= 0):
        raise ValueError("Expected finite [fx, fy, cx, cy] with positive focal lengths")
    stem = output / "images" / f"frame_{index:06d}_cam{camera_id}"
    stem.with_suffix(".jpg").write_bytes(jpeg)
    write_json(stem.with_suffix(".json"), dict(
        intrinsic=intrinsic.tolist(), extrinsic=matrix(reference_from_camera).ravel().tolist(),
        width=image.shape[1], height=image.shape[0], **metadata))


def source_path(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Missing or out-of-root sensor file: {filename}")
    return path
