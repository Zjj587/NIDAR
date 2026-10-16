"""Release and demo contract tests; no model weights or GPU required."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
import run_nidar_demo as demo

spec = importlib.util.spec_from_file_location("source_package", ROOT / "tools/release/make_source_package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.input = self.base / "raw"
        self.weights = self.base / "weights"
        (self.input / "images").mkdir(parents=True)
        (self.input / "pointclouds").mkdir()
        for name in ["stn/47.pth", "irnet/epoch_010.pth"]:
            path = self.weights / name
            path.parent.mkdir(parents=True)
            path.touch()
        self.args = argparse.Namespace(
            input_root=str(self.input), weights_root=str(self.weights),
            output_root=str(self.base / "output"), quantile_model=None,
            frame_count=2, start_frame=0, step=1, device="cpu", no_eval=False,
            preview_points=10,
        )
        self.paths = demo.resolve_paths(self.args)

    def make_frame(self, frame_id, cameras=(1, 2)):
        frame = f"frame_{frame_id:06d}"
        camera = {"intrinsic": [10, 10, 8, 4], "extrinsic": np.eye(4).ravel().tolist(),
                  "width": 16, "height": 8}
        for cam_id in cameras:
            path = self.input / "images" / f"{frame}_cam{cam_id}"
            path.with_suffix(".json").write_text(json.dumps(camera))
            cv2.imwrite(str(path.with_suffix(".jpg")), np.zeros((8, 16, 3), dtype=np.uint8))
        (self.input / "pointclouds" / f"{frame}_lidar1.json").write_text(
            json.dumps({"extrinsic": np.eye(4).ravel().tolist()}))
        np.savez(self.input / "pointclouds" / f"{frame}_raw_pc.npz",
                 points=np.array([[1, 2, 3, 0.5]], dtype=np.float32))

    def make_remapped(self, frame_id, points=None):
        directory = self.paths["work_dir"] / "stage3_5_remapped"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"dis_learned_frame_{frame_id:06d}_pseudo_intensity.npy"
        if points is None:
            points = np.array([[1, 2, 3, 0.5]], dtype=np.float32)
        np.save(path, points)
        demo.write_colored_ply(points, np.zeros((len(points), 3), dtype=np.uint8), path.with_suffix(".ply"))
        return path

    def test_default_architecture_and_output_route(self):
        model = yaml.safe_load(demo.DEFAULT_MODEL_CONFIG.read_text())["model"]
        config = yaml.safe_load(demo.DEFAULT_CONFIG.read_text())
        self.assertEqual(model["intrinsic_net_type"], "unet")
        self.assertEqual(model["intensity_head_type"], "deeplab")
        self.assertFalse(config["stage2_deep"]["use_i_hat_as_r"])
        self.assertNotIn("/home/", demo.DEFAULT_CONFIG.read_text())

    def test_manifest_includes_demo_and_model_dependencies(self):
        for name in ["LICENSE", "docs/DEMO.md", "docs/DATA_ACCESS.md", "docs/MODEL_ZOO.md",
                     "inference/run_nidar_demo.py", "inference/src/models/network_variants.py",
                     "inference/src/models/checkpoint.py", "requirements-viewer.txt",
                     "requirements-eval.txt", "tests/test_checkpoint.py",
                     "constraints-cpu-tested.txt",
                     "docs/RELEASE_ACCEPTANCE.md",
                     "tools/download_weights.py", "tests/test_download_weights.py"]:
            self.assertTrue(package.should_copy(Path(name), False), name)
            self.assertTrue((ROOT / name).is_file(), name)
        for name in [".git/config", "weights/stn/47.pth", "outputs/result.npy", "logs/COMMAND_LOG.md"]:
            self.assertFalse(package.should_copy(Path(name), False), name)

    def test_source_package_runs_demo_help(self):
        destination = self.base / "release"
        subprocess.run([sys.executable, str(ROOT / "tools/release/make_source_package.py"),
                        "--source-root", str(ROOT), "--out-dir", str(destination)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for name in ["LICENSE", "docs/MODEL_ZOO.md", "inference/src/models/network_variants.py"]:
            self.assertTrue((destination / name).is_file(), name)
        result = subprocess.run([sys.executable, str(destination / "inference/run_nidar_demo.py"), "--help"],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertIn("--input-root", result.stdout)
        result = subprocess.run([sys.executable, str(destination / "tools/download_weights.py"), "--help"],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertIn("--check-only", result.stdout)

    def test_invalid_integer_arguments(self):
        for flag, value in [("--frame-count", "0"), ("--step", "0"),
                            ("--preview-points", "0"), ("--start-frame", "-1")]:
            with self.subTest(flag=flag), patch.object(sys, "argv", ["demo", "--input-root", "raw",
                    "--weights-root", "weights", flag, value]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    demo.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_checkpoint_directory_is_not_a_file(self):
        path = self.weights / "stn/47.pth"
        path.unlink()
        path.mkdir()
        with self.assertRaises(FileNotFoundError):
            demo.resolve_paths(self.args)

    def test_missing_runtime_dependencies_fail_early(self):
        with patch("run_nidar_demo.importlib.util.find_spec", side_effect=lambda name: None if name == "skimage" else object()):
            with self.assertRaisesRegex(RuntimeError, "skimage.*requirements.txt"):
                demo.validate_runtime(self.args)
        self.assertFalse(self.paths["output_root"].exists())

    def test_no_eval_does_not_require_skimage(self):
        self.args.no_eval = True
        with patch("run_nidar_demo.importlib.util.find_spec", side_effect=lambda name: None if name == "skimage" else object()):
            demo.validate_runtime(self.args)

    def test_input_output_overlap_rejected(self):
        self.args.output_root = str(self.base)
        self.args.input_root = str(self.base / "results" / "raw")
        Path(self.args.input_root, "images").mkdir(parents=True)
        Path(self.args.input_root, "pointclouds").mkdir()
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            demo.resolve_paths(self.args)

    def test_symlinked_artifact_output_overlap_rejected(self):
        checkpoint = self.weights / "irnet/epoch_010.pth"
        checkpoint.unlink()
        target = self.base / "output/results/checkpoint.pth"
        target.parent.mkdir(parents=True)
        target.touch()
        checkpoint.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            demo.resolve_paths(self.args)

    def test_complete_input_manifest(self):
        self.make_frame(0)
        self.make_frame(1)
        manifest = demo.validate_input(self.args, self.paths)
        self.assertEqual(manifest["frames"], ["frame_000000", "frame_000001"])
        self.assertEqual(len(manifest["images"]), 4)
        self.assertFalse(self.paths["output_root"].exists())

    def test_missing_frame_fails_before_outputs(self):
        self.make_frame(0)
        with self.assertRaisesRegex(ValueError, "Requested 2 frames"):
            demo.validate_input(self.args, self.paths)
        self.assertFalse(self.paths["output_root"].exists())

    def test_selection_stride_uses_dataset_loader(self):
        for frame_id in [0, 1, 3, 7]:
            self.make_frame(frame_id)
        self.args.start_frame = 1
        self.args.step = 2
        self.assertEqual(demo.validate_input(self.args, self.paths)["frames"],
                         ["frame_000001", "frame_000007"])

    def test_missing_camera_image_rejected(self):
        self.make_frame(0)
        self.make_frame(1)
        (self.input / "images/frame_000000_cam2.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Input image missing"):
            demo.validate_input(self.args, self.paths)

    def test_missing_camera_calibration_rejected(self):
        self.make_frame(0)
        self.make_frame(1)
        (self.input / "images/frame_000000_cam2.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "camera calibration"):
            demo.validate_input(self.args, self.paths)

    def test_corrupt_image_rejected(self):
        self.make_frame(0)
        self.make_frame(1)
        (self.input / "images/frame_000000_cam1.jpg").write_text("not an image")
        with self.assertRaises(IOError):
            demo.validate_input(self.args, self.paths)

    def test_missing_intermediate_image_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "pseudo-NIR output"):
            demo.validate_intermediate_outputs({"images": ["frame_000000_cam1"]}, self.paths)

    def test_generated_config_preserves_R_route(self):
        demo.prepare_output(self.paths, False)
        path = demo.write_generated_config(self.args, self.paths)
        config = yaml.safe_load(path.read_text())
        self.assertFalse(config["stage2_deep"]["use_i_hat_as_r"])
        self.assertEqual(config["stage2_deep"]["model_config_path"], str(demo.DEFAULT_MODEL_CONFIG))
        self.assertEqual(config["stage3_5"]["quantile_model_path"], str(demo.DEFAULT_BUNDLED_QUANTILE))
        self.assertFalse(config["stage1"]["use_gpu"])

    def test_partial_clouds_rejected_even_without_evaluation(self):
        self.args.no_eval = True
        self.make_remapped(0)
        with self.assertRaisesRegex(RuntimeError, "Incomplete remapped outputs"):
            demo.collect_results(self.args, self.paths, self.base / "log", ["frame_000000", "frame_000001"])

    def test_nonfinite_cloud_rejected(self):
        self.args.no_eval = True
        self.make_remapped(0, np.array([[1, 2, 3, np.nan]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            demo.collect_results(self.args, self.paths, self.base / "log", ["frame_000000"])

    def test_successful_collection_writes_compact_outputs(self):
        self.args.no_eval = True
        self.args.frame_count = 1
        demo.prepare_output(self.paths, False)
        source = self.make_remapped(0, np.array([[0, 0, 0, 0.2], [1, 1, 0, 0.8]], dtype=np.float32))
        summary = demo.collect_results(self.args, self.paths, self.base / "log", ["frame_000000"])
        result = summary["frames"][0]
        np.testing.assert_array_equal(np.load(result["nidar_intensity_npy"]), np.load(source))
        self.assertTrue(Path(result["nidar_intensity_ply"]).is_file())
        self.assertTrue(Path(result["nidar_intensity_colored_ply"]).is_file())
        image = cv2.imread(result["preview_png"])
        self.assertIsNotNone(image)
        self.assertGreater(int(image.max()), int(image.min()))
        saved = json.loads(Path(summary["summary_json"]).read_text())
        self.assertEqual(saved["frame_count"], 1)
        self.assertIn("IRNet R", saved["route"])

    def test_missing_scalar_ply_rejected(self):
        self.args.no_eval = True
        demo.prepare_output(self.paths, False)
        self.make_remapped(0).with_suffix(".ply").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "scalar intensity PLY"):
            demo.collect_results(self.args, self.paths, self.base / "log", ["frame_000000"])

    def test_public_metrics_use_null_not_nonstandard_json_constants(self):
        source = self.base / "internal.json"
        destination = self.base / "public.json"
        source.write_text(json.dumps({"lpips": float("nan"), "average": {"mae": 0.1},
                                      "values": [float("inf"), float("-inf"), 2]}))
        demo.copy_json_result(source, destination)
        def reject_constant(value):
            raise ValueError(value)
        saved = json.loads(destination.read_text(), parse_constant=reject_constant)
        self.assertIsNone(saved["lpips"])
        self.assertEqual(saved["average"]["mae"], 0.1)
        self.assertEqual(saved["values"], [None, None, 2])

    def test_incomplete_evaluation_rejected(self):
        demo.prepare_output(self.paths, False)
        self.make_remapped(0)
        self.make_remapped(1)
        directory = self.paths["work_dir"] / "stage4_evaluation"
        directory.mkdir()
        (directory / "evaluation_summary.json").write_text(json.dumps({"num_samples": 1, "num_success": 1}))
        with self.assertRaisesRegex(RuntimeError, "Evaluation did not complete"):
            demo.collect_results(self.args, self.paths, self.base / "log", ["frame_000000", "frame_000001"])

    def test_forced_viewer_reports_missing_dependency(self):
        with patch.dict(sys.modules, {"open3d": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-viewer.txt"):
                demo.maybe_open_viewer("open3d", self.base / "cloud.ply")

    def test_viewer_detects_window_creation_failure(self):
        library = Mock()
        window = library.visualization.Visualizer.return_value
        window.create_window.return_value = False
        with patch.dict(sys.modules, {"open3d": library}):
            with self.assertRaisesRegex(RuntimeError, "Requested Open3D viewer failed"):
                demo.maybe_open_viewer("open3d", self.base / "cloud.ply")
        window.run.assert_not_called()
        window.destroy_window.assert_called_once()

    def test_linux_viewer_prefers_xwayland_and_restores_environment(self):
        library = Mock()
        window = library.visualization.Visualizer.return_value
        def check_display():
            self.assertEqual(os.environ["XDG_SESSION_TYPE"], "x11")
            self.assertNotIn("WAYLAND_DISPLAY", os.environ)
        window.run.side_effect = check_display
        with patch.dict(sys.modules, {"open3d": library}), patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"DISPLAY": ":0", "XDG_SESSION_TYPE": "wayland",
                                         "WAYLAND_DISPLAY": "wayland-0"}):
                demo.maybe_open_viewer("open3d", self.base / "cloud.ply")
                self.assertEqual(os.environ["XDG_SESSION_TYPE"], "wayland")
                self.assertEqual(os.environ["WAYLAND_DISPLAY"], "wayland-0")
        window.run.assert_called_once()
        window.destroy_window.assert_called_once()

    def test_auto_viewer_is_optional(self):
        with patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}):
            demo.maybe_open_viewer("auto", self.base / "cloud.ply")


if __name__ == "__main__":
    unittest.main()
