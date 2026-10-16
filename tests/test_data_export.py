"""Synthetic contract/geometry tests, not official-dataset acceptance results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/data_export"))
sys.path.insert(0, str(ROOT / "inference"))
import common
import prepare_nuscenes as nuscenes
import prepare_waymo as waymo
import run_nidar_demo as demo
from src.datasets.waymo_dataset import WaymoDatasetWrapper
from src.projection.projection_utils import project_points_to_image


def transform(angle=0, translation=(0, 0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    result = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
    result[:3, 3] = translation
    return result


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "official"
        self.source.mkdir()
        self.jpeg = cv2.imencode(".jpg", np.full((8, 16, 3), 80, dtype=np.uint8))[1].tobytes()

    def start(self, dataset):
        return common.start_export(self.base / "prepared", self.source, dataset)

    def args(self, output):
        weights = self.base / "weights"
        for name in ["stn/47.pth", "irnet/epoch_010.pth"]:
            path = weights / name
            path.parent.mkdir(parents=True)
            path.touch()
        return argparse.Namespace(input_root=str(output), weights_root=str(weights),
            output_root=str(self.base / "results"), dataset="auto", quantile_model=None,
            frame_count=1, start_frame=0, step=1, device="cpu", no_eval=False)

    def dataset(self, output):
        return WaymoDatasetWrapper(output / "images", output / "pointclouds")

    def test_new_export_stays_incomplete_until_committed(self):
        output, manifest = self.start("waymo")
        self.assertEqual(json.loads((output / "input_manifest.json").read_text())["export_status"], "incomplete")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            demo.resolve_paths(self.args(output))
        with self.assertRaisesRegex(ValueError, "No frames"):
            common.finish_export(output, manifest)

    def test_export_refuses_existing_and_overlapping_inputs(self):
        self.start("waymo")
        with self.assertRaises(FileExistsError):
            self.start("waymo")
        with self.assertRaisesRegex(ValueError, "separate"):
            common.start_export(self.source / "export", self.source, "waymo")

    def test_source_path_rejects_traversal(self):
        outside = self.base / "secret.txt"
        outside.write_text("not sensor data")
        with self.assertRaises(FileNotFoundError):
            common.source_path(self.source, "../secret.txt")

    def test_transform_rejects_nonrigid_and_nonfinite(self):
        for value in [np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([2, 1, 1, 1])]:
            with self.assertRaises(ValueError):
                common.matrix(value)

    def test_cloud_rejects_nonfinite_and_misaligned(self):
        output, _ = self.start("waymo")
        for xyz, intensity in [(np.zeros((1, 3)), []), (np.full((1, 3), np.nan), [0]),
                               (np.empty((0, 3)), [])]:
            with self.assertRaises(ValueError):
                common.write_cloud(output, 0, xyz, intensity, np.eye(4))

    def test_waymo_zero_distortion_preserves_jpeg(self):
        self.assertEqual(waymo.rectify_image(self.jpeg, [10, 10, 8, 4, 0, 0, 0, 0, 0]), self.jpeg)

    def test_waymo_distortion_is_rectified_to_same_image_size(self):
        result = waymo.rectify_image(self.jpeg, [10, 10, 8, 4, .1, .01, 0, 0, 0])
        image = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (8, 16, 3))

    def test_waymo_sdk_motion_compensated_points_and_camera_time_pose(self):
        output, manifest = self.start("waymo")
        vehicle = transform(translation=(10, 20, 1))
        lidar = transform(translation=(1, 0, 2))
        camera_pose = transform(translation=(12, 20, 1))
        frame = NS(pose=NS(transform=vehicle.ravel()), timestamp_micros=1234,
            context=NS(name="synthetic-sdk-fixture",
                laser_calibrations=[NS(name=2, extrinsic=NS(transform=np.eye(4).ravel())),
                                    NS(name=1, extrinsic=NS(transform=lidar.ravel()),
                                       beam_inclinations=[-.2, .1])],
                camera_calibrations=[NS(name=1, intrinsic=[10, 10, 8, 4, 0, 0, 0, 0, 0],
                                        extrinsic=NS(transform=np.eye(4).ravel()))]),
            images=[NS(name=1, image=self.jpeg, pose=NS(transform=camera_pose.ravel()))])
        sdk = Mock()
        sdk.parse_range_image_and_camera_projection.return_value = (
            {1: [NS(shape=NS(dims=[2, 64, 4]))]}, {}, None, object())
        sdk.convert_range_image_to_point_cloud.return_value = ([np.array([[5, .3, .9, 5, 0, 0]]),
                                                               np.zeros((1, 6))], [])
        manifest["frames"].append(waymo.export_frame(frame, sdk, output, 0, 7))
        common.finish_export(output, manifest)
        self.assertTrue(sdk.convert_range_image_to_point_cloud.call_args.kwargs["keep_polar_features"])
        self.assertEqual(sdk.convert_range_image_to_point_cloud.call_args.kwargs["ri_index"], 0)
        dataset = self.dataset(output)
        points = dataset.load_pointcloud(0)
        np.testing.assert_allclose(points, [[15, 20, 1, .3]])
        camera = dataset.load_calibration(0, 1)
        _, uv, _ = project_points_to_image(points, camera["intrinsic"], camera["extrinsic"], 16, 8,
                                           dataset.load_lidar_calibration(0)["ego2world"])
        np.testing.assert_allclose(uv, [[8, 4]])
        self.assertEqual(manifest["frames"][0]["source_frame_index"], 7)
        calibration = json.loads((output / "pointclouds/frame_000000_eval_calib.json").read_text())
        np.testing.assert_array_equal(np.array(calibration["extrinsic_transform"]).reshape(4, 4), np.eye(4))
        self.assertEqual(calibration["range_image_width"], 64)
        self.assertEqual(calibration["beam_inclinations"], [-.2, .1])
        manifest["evaluation_calibration"] = "per-frame-lidar-reference"
        common.finish_export(output, manifest)
        args = self.args(output)
        paths = demo.resolve_paths(args)
        demo.validate_input(args, paths)
        demo.prepare_output(paths, False)
        import yaml
        config = yaml.safe_load(demo.write_generated_config(args, paths).read_text())
        self.assertIsNone(config["evaluation"]["waymo_calib_json"])
        self.assertEqual(config["evaluation"]["waymo_calib_dir"], str(output / "pointclouds"))

    def test_waymo_fallback_beams_use_sdk_pixel_centers(self):
        output, _ = self.start("waymo")
        waymo.write_evaluation_calibration(output, 0,
            NS(beam_inclinations=[], beam_inclination_min=-.2, beam_inclination_max=.2),
            NS(shape=NS(dims=[4, 32, 4])))
        calibration = json.loads((output / "pointclouds/frame_000000_eval_calib.json").read_text())
        np.testing.assert_allclose(calibration["beam_inclinations"], [-.15, -.05, .05, .15])

    def test_waymo_new_evaluator_does_not_apply_mount_twice(self):
        from src.evaluation.spherical_projection import project_to_range_image_waymo
        output, _ = self.start("waymo")
        waymo.write_evaluation_calibration(output, 0, NS(beam_inclinations=[0]),
                                         NS(shape=NS(dims=[1, 64, 4])))
        world_from_lidar = transform(.3, (9, 2, 3))
        world = common.transform_points([[5, 0, 0]], world_from_lidar)
        result = project_to_range_image_waymo(np.column_stack([world, [.25]]),
            str(output / "pointclouds/frame_000000_eval_calib.json"), ego2world=world_from_lidar)
        self.assertEqual(result.shape, (1, 64, 2))
        np.testing.assert_allclose(result[0, 32], [5, .25])

    def make_nuscenes(self):
        world_lidar = transform(.4, (10, -4, 1.7))
        world_camera = transform(-.3, (7, 3, 1))
        optical_xyz = np.array([[0, 0, 3], [.5, .25, 5], [0, 0, -1.]])
        world_xyz = common.transform_points(optical_xyz, world_camera)
        sensor_xyz = common.transform_points(world_xyz, np.linalg.inv(world_lidar))
        table = {("ego_pose", "lidar_pose"): {"matrix": world_lidar},
                 ("ego_pose", "camera_pose"): {"matrix": world_camera},
                 ("calibrated_sensor", "lidar_calib"): {"matrix": np.eye(4)},
                 ("calibrated_sensor", "camera_calib"): {
                     "matrix": np.eye(4), "camera_intrinsic": [[10, 0, 8], [0, 10, 4], [0, 0, 1]]}}
        sample = {"token": "synthetic-sample", "data": {}, "next": ""}
        for channel in ("LIDAR_TOP",) + nuscenes.CAMERAS:
            is_lidar = channel == "LIDAR_TOP"
            name = f"{channel}.bin" if is_lidar else f"{channel}.jpg"
            (self.source / name).write_bytes(b"fixture" if is_lidar else self.jpeg)
            sample["data"][channel] = channel
            table[("sample_data", channel)] = dict(token=channel, filename=name,
                calibrated_sensor_token="lidar_calib" if is_lidar else "camera_calib",
                ego_pose_token="lidar_pose" if is_lidar else "camera_pose", timestamp=100 if is_lidar else 120)
        nusc = NS(dataroot=str(self.source), get=lambda kind, token: table[(kind, token)])
        cloud = NS(points=np.column_stack([sensor_xyz, [0., 128., 255.]]).T)
        return nusc, sample, cloud

    def test_nuscenes_export_aligns_optical_axes_and_asynchronous_poses(self):
        output, manifest = self.start("nuscenes")
        nusc, sample, cloud = self.make_nuscenes()
        manifest["frames"].append(nuscenes.export_sample(nusc, sample, output, 0, lambda path: cloud,
                                                        pose_fn=lambda rec: rec["matrix"]))
        common.finish_export(output, manifest)
        dataset = self.dataset(output)
        self.assertEqual(len(dataset.get_frame_info(0)["cameras"]), 6)
        points = dataset.load_pointcloud(0)
        np.testing.assert_allclose(points[:, 3], [0, 128 / 255., 1])
        camera = dataset.load_calibration(0, 1)
        _, uv, indices = project_points_to_image(points, camera["intrinsic"], camera["extrinsic"],
            16, 8, dataset.load_lidar_calibration(0)["ego2world"])
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_allclose(uv, [[8, 4], [9, 4.5]], atol=1e-5)
        args = self.args(output)
        paths = demo.resolve_paths(args)
        self.assertEqual(paths["dataset_type"], "nuscenes")
        self.assertEqual(paths["quantile_model"].name, "nintensity_mapping_quantile_model.pkl")
        self.assertEqual(len(demo.validate_input(args, paths)["images"]), 6)
        demo.prepare_output(paths, False)
        import yaml
        config = yaml.safe_load(demo.write_generated_config(args, paths).read_text())
        self.assertEqual(config["dataset"]["type"], "nuscenes")
        self.assertIsNone(config["evaluation"]["waymo_calib_json"])
        self.assertFalse(config["stage2_deep"]["use_i_hat_as_r"])
        args.dataset = "waymo"
        with self.assertRaisesRegex(ValueError, "does not match"):
            demo.resolve_paths(args)

    def test_nuscenes_rejects_invalid_native_intensity(self):
        output, _ = self.start("nuscenes")
        nusc, sample, cloud = self.make_nuscenes()
        cloud.points[3, 0] = 256
        with self.assertRaisesRegex(ValueError, "255"):
            nuscenes.export_sample(nusc, sample, output, 0, lambda path: cloud, lambda rec: rec["matrix"])

    def test_nuscenes_requires_manifest(self):
        output, _ = self.start("nuscenes")
        (output / "input_manifest.json").unlink()
        args = self.args(output)
        args.dataset = "nuscenes"
        with self.assertRaisesRegex(ValueError, "requires input_manifest"):
            demo.resolve_paths(args)

    def test_nuscenes_empty_split_is_reported(self):
        with self.assertRaisesRegex(ValueError, "No scenes"):
            nuscenes.select_samples(NS(scene=[]), None, 1)

    def test_manifest_frame_count_and_cloud_validation(self):
        output, manifest = self.start("nuscenes")
        nusc, sample, cloud = self.make_nuscenes()
        manifest["frames"].append(nuscenes.export_sample(nusc, sample, output, 0, lambda path: cloud,
                                                        pose_fn=lambda rec: rec["matrix"]))
        common.finish_export(output, manifest)
        args = self.args(output)
        paths = demo.resolve_paths(args)
        manifest["frame_count"] = 2
        common.write_json(output / "input_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "frame list/count"):
            demo.validate_input(args, paths)
        manifest["frame_count"] = 1
        common.write_json(output / "input_manifest.json", manifest)
        np.savez(output / "pointclouds/frame_000000_raw_pc.npz", points=np.full((1, 4), np.nan))
        with self.assertRaisesRegex(ValueError, "finite point cloud"):
            demo.validate_input(args, paths)

    def test_check_input_does_not_run_pipeline_or_create_outputs(self):
        output, manifest = self.start("nuscenes")
        nusc, sample, cloud = self.make_nuscenes()
        manifest["frames"].append(nuscenes.export_sample(nusc, sample, output, 0, lambda path: cloud,
                                                        pose_fn=lambda rec: rec["matrix"]))
        common.finish_export(output, manifest)
        args = self.args(output)
        args.check_input = True
        with patch.object(demo, "parse_args", return_value=args), patch.object(demo, "validate_runtime"), \
                patch.object(demo, "run_internal_pipeline") as pipeline:
            self.assertEqual(demo.main(), 0)
            pipeline.assert_not_called()
        self.assertFalse(Path(args.output_root).exists())

    def test_removed_camera_pair_does_not_pass_as_complete_export(self):
        output, manifest = self.start("nuscenes")
        nusc, sample, cloud = self.make_nuscenes()
        manifest["frames"].append(nuscenes.export_sample(nusc, sample, output, 0, lambda path: cloud,
                                                        pose_fn=lambda rec: rec["matrix"]))
        common.finish_export(output, manifest)
        (output / "images/frame_000000_cam6.jpg").unlink()
        (output / "images/frame_000000_cam6.json").unlink()
        args = self.args(output)
        with self.assertRaisesRegex(ValueError, "Camera set"):
            demo.validate_input(args, demo.resolve_paths(args))

    def test_scene_selection_is_deterministic_and_respects_stride(self):
        samples = {str(i): {"token": str(i), "next": str(i + 1) if i < 5 else ""} for i in range(6)}
        nusc = NS(scene=[dict(name="scene-b", first_sample_token="0"),
                         dict(name="scene-a", first_sample_token="0")],
                  get=lambda kind, token: samples[token])
        scene, selected = nuscenes.select_samples(nusc, None, 2, start=1, step=2)
        self.assertEqual(scene["name"], "scene-a")
        self.assertEqual([sample["token"] for sample in selected], ["1", "3"])
        with self.assertRaisesRegex(ValueError, "too few"):
            nuscenes.select_samples(nusc, "scene-a", 7)
        with self.assertRaisesRegex(ValueError, "not found"):
            nuscenes.select_samples(nusc, "missing", 1)

    def test_converter_help_does_not_import_dataset_sdks(self):
        for name in ["prepare_waymo.py", "prepare_nuscenes.py"]:
            result = subprocess.run([sys.executable, str(ROOT / "tools/data_export" / name), "--help"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("--output-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
