#!/usr/bin/env python3
"""Export selected frames from an authorized Waymo Perception Frame TFRecord."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import finish_export, matrix, start_export, transform_points, write_camera, write_cloud, write_json


def write_evaluation_calibration(output, index, calibration, range_image):
    height, width = range_image.shape.dims[:2]
    inclinations = np.asarray(calibration.beam_inclinations, dtype=float)
    if not inclinations.size:
        inclinations = (calibration.beam_inclination_min +
            (np.arange(height) + 0.5) / height *
            (calibration.beam_inclination_max - calibration.beam_inclination_min))
    if inclinations.shape != (height,) or not np.isfinite(inclinations).all():
        raise ValueError("Beam inclinations must match the TOP range-image height")
    # ego2world already transforms from LiDAR coordinates; do not apply the mount twice.
    write_json(output / "pointclouds" / f"frame_{index:06d}_eval_calib.json", dict(
        beam_inclinations=inclinations.tolist(), range_image_width=int(width),
        extrinsic_transform=np.eye(4).ravel().tolist()))


def rectify_image(jpeg, intrinsic):
    values = np.asarray(intrinsic, dtype=float)
    if values.size != 9 or not np.isfinite(values).all():
        raise ValueError("Expected nine Waymo camera intrinsic/distortion values")
    fx, fy, cx, cy = values[:4]
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if np.all(values[4:] == 0):
        return jpeg
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid Waymo camera JPEG")
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    image = cv2.undistort(image, k, values[4:], None, k)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode rectified camera image")
    return encoded.tobytes()


def export_frame(frame, frame_utils, output, index, source_index):
    ranges, projections, _, top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    # Official SDK handles inclination ordering, pixel poses and TOP motion compensation.
    clouds, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, ranges, projections, top_pose, ri_index=0, keep_polar_features=True)
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    top_index = next(i for i, c in enumerate(calibrations) if int(c.name) == 1)
    polar_xyz = np.asarray(clouds[top_index])
    if polar_xyz.ndim != 2 or polar_xyz.shape[1] != 6:
        raise ValueError("Waymo SDK must return [range, intensity, elongation, x, y, z]")
    world_from_vehicle = matrix(frame.pose.transform)
    world_from_lidar = world_from_vehicle @ matrix(calibrations[top_index].extrinsic.transform)
    write_cloud(output, index, transform_points(polar_xyz[:, 3:6], world_from_vehicle),
                polar_xyz[:, 1], world_from_lidar)
    write_evaluation_calibration(output, index, calibrations[top_index], ranges[1][0])
    camera_calibrations = {int(c.name): c for c in frame.context.camera_calibrations}
    cameras = []
    for camera in sorted(frame.images, key=lambda c: c.name):
        calibration = camera_calibrations[int(camera.name)]
        camera_pose = list(camera.pose.transform)
        world_from_camera_vehicle = matrix(camera_pose) if camera_pose else world_from_vehicle
        reference_from_camera = (np.linalg.inv(world_from_lidar) @ world_from_camera_vehicle @
                                 matrix(calibration.extrinsic.transform))
        write_camera(output, index, int(camera.name),
                     rectify_image(camera.image, calibration.intrinsic),
                     calibration.intrinsic[:4], reference_from_camera,
                     distortion_rectified=True,
                     camera_pose_source="camera.pose" if camera_pose else "frame.pose")
        cameras.append(int(camera.name))
    if not cameras:
        raise ValueError("Frame has no camera images")
    return dict(frame=f"frame_{index:06d}", source_frame_index=source_index,
                context_name=frame.context.name, timestamp_micros=int(frame.timestamp_micros),
                cameras=cameras)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", type=Path, required=True)
    parser.add_argument("--data-version", required=True, help="Provider release, e.g. v1.4.3.")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1], help="Zero-based record indices.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.tfrecord.is_file() or args.tfrecord.suffix not in (".tfrecord", ".tfrecords"):
        parser.error("Provide a Perception Frame TFRecord, not Motion data or modular Parquet")
    if min(args.frames) < 0 or len(set(args.frames)) != len(args.frames):
        parser.error("Frame indices must be distinct and nonnegative")
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.utils import frame_utils
    except ImportError as exc:
        parser.error(f"Install requirements-waymo-export.txt in a separate environment: {exc}")
    tf.config.set_visible_devices([], "GPU")
    output, manifest = start_export(args.output_root, args.tfrecord, "waymo",
        source_file=args.tfrecord.name, source_bytes=args.tfrecord.stat().st_size,
        data_version=args.data_version, return_index=0, intensity_encoding="native Waymo",
        evaluation_calibration="per-frame-lidar-reference",
        limitations=["Camera rolling shutter and object motion are not corrected."])
    requested = set(args.frames)
    for source_index, record in enumerate(tf.data.TFRecordDataset(str(args.tfrecord), compression_type="")):
        if source_index in requested:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(record.numpy()))
            manifest["frames"].append(export_frame(frame, frame_utils, output,
                                                    len(manifest["frames"]), source_index))
        if source_index >= max(requested):
            break
    if len(manifest["frames"]) != len(requested):
        raise ValueError("Requested frames exceed this TFRecord; export remains incomplete")
    finish_export(output, manifest)
    print(f"Exported {len(requested)} frames: {output}")


if __name__ == "__main__":
    main()
