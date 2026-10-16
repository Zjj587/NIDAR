#!/usr/bin/env python3
"""Export nuScenes keyframes with camera-time poses into the NIDAR input contract."""
import argparse
from pathlib import Path

import numpy as np

from common import (OPTICAL_FROM_PIPELINE, finish_export, matrix, positive_int,
                    source_path, start_export, transform_points, write_camera, write_cloud)


CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def pose(record):
    from pyquaternion import Quaternion
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    transform[:3, 3] = record["translation"]
    return matrix(transform)


def camera_to_reference(world_from_lidar, world_from_camera):
    return np.linalg.inv(matrix(world_from_lidar)) @ matrix(world_from_camera) @ OPTICAL_FROM_PIPELINE


def select_samples(nusc, scene_name, count, start=0, step=1):
    scenes = sorted(nusc.scene, key=lambda scene: scene["name"])
    if not scenes:
        raise ValueError("No scenes found in the selected nuScenes split")
    scene = next((scene for scene in scenes if scene["name"] == scene_name), None) if scene_name else scenes[0]
    if scene is None:
        raise ValueError(f"Scene not found: {scene_name}")
    samples, token, index = [], scene["first_sample_token"], 0
    while token and len(samples) < count:
        sample = nusc.get("sample", token)
        if index >= start and (index - start) % step == 0:
            samples.append(sample)
        token, index = sample["next"], index + 1
    if len(samples) != count:
        raise ValueError(f"Scene {scene['name']} has too few keyframes for this selection")
    return scene, samples


def export_sample(nusc, sample, output, index, cloud_loader, pose_fn=pose):
    root = Path(nusc.dataroot)
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_calibration = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    world_from_lidar = (pose_fn(nusc.get("ego_pose", lidar["ego_pose_token"])) @
                        pose_fn(lidar_calibration))
    points = np.asarray(cloud_loader(str(source_path(root, lidar["filename"]))).points).T
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("Expected finite nuScenes SDK XYZ/intensity points")
    if np.any(points[:, 3] < 0) or np.any(points[:, 3] > 255):
        raise ValueError("Expected nuScenes intensity in [0, 255]")
    write_cloud(output, index, transform_points(points[:, :3], world_from_lidar),
                points[:, 3] / 255., world_from_lidar)
    camera_records = []
    for camera_id, channel in enumerate(CAMERAS, 1):
        camera = nusc.get("sample_data", sample["data"][channel])
        calibration = nusc.get("calibrated_sensor", camera["calibrated_sensor_token"])
        world_from_camera = (pose_fn(nusc.get("ego_pose", camera["ego_pose_token"])) @
                             pose_fn(calibration))
        k = np.asarray(calibration["camera_intrinsic"], dtype=float).reshape(3, 3)
        write_camera(output, index, camera_id, source_path(root, camera["filename"]).read_bytes(),
                     [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                     camera_to_reference(world_from_lidar, world_from_camera),
                     channel=channel, sample_data_token=camera["token"],
                     timestamp=int(camera["timestamp"]))
        camera_records.append(dict(camera_id=camera_id, channel=channel,
                                   sample_data_token=camera["token"], timestamp=int(camera["timestamp"])))
    return dict(frame=f"frame_{index:06d}", sample_token=sample["token"],
                lidar_sample_data_token=lidar["token"], timestamp=int(lidar["timestamp"]),
                cameras=camera_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"], default="v1.0-mini")
    parser.add_argument("--scene", help="Scene name; default is the first scene in sorted name order.")
    parser.add_argument("--frame-count", type=positive_int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--step", type=positive_int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if not (args.dataroot / args.version / "scene.json").is_file():
        parser.error("Expected dataroot/version/scene.json; keep official metadata and sensor files together")
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
    except ImportError as exc:
        parser.error(f"Install requirements-nuscenes-export.txt in a separate environment: {exc}")
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    scene, samples = select_samples(nusc, args.scene, args.frame_count, args.start_frame, args.step)
    output, manifest = start_export(args.output_root, args.dataroot, "nuscenes",
        data_version=args.version, scene_name=scene["name"], scene_token=scene["token"],
        intensity_encoding="native nuScenes / 255", sweep_count=1,
        limitations=["Ego motion is handled at each camera timestamp; object motion and per-point deskew are not."])
    for index, sample in enumerate(samples):
        manifest["frames"].append(export_sample(nusc, sample, output, index, LidarPointCloud.from_file))
    finish_export(output, manifest)
    print(f"Exported {len(samples)} frames from {scene['name']}: {output}")


if __name__ == "__main__":
    main()
