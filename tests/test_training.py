"""Training contract tests use generated fixtures, never redistributed sensor data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference"))
from training.data import IntensityDataset, epoch_batches, load_manifest, sha256, validate_records
from training.losses import DEFAULT_WEIGHTS, intrinsic_loss
from training.prepare import enumerate_sources, prepare, rasterize
from training.train import train
from tools.data_export.common import write_camera, write_cloud
from src.models.checkpoint import load_checkpoint
from src.models.deep_intrinsic_wrapper import DeepIntrinsicWrapper


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_zbuffer_and_valid_zero(self):
        points = np.array([[2, 0, 0, 0], [4, 0, 0, 0.9], [-2, 0, 0, 0.7]], np.float32)
        target, mask, clipped = rasterize(points, [2, 2, 2, 2], np.eye(4), np.eye(4),
                                          (4, 4), (4, 4), 1.0)
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(mask[2, 2])
        self.assertEqual(target[2, 2], 0)
        self.assertEqual(clipped, 0)

    def test_world_transform_and_clipping(self):
        transform = np.eye(4)
        transform[0, 3] = 10
        target, mask, clipped = rasterize(np.array([[12, 0, 0, 3.0]]), [2, 2, 2, 2],
                                          np.eye(4), transform, (4, 4), (2, 2), 2.0)
        self.assertTrue(mask[1, 1])
        self.assertEqual(target[1, 1], 1.0)
        self.assertEqual(clipped, 1)

    def test_no_points_is_not_dense_supervision(self):
        _, mask, _ = rasterize(np.array([[-1, 0, 0, 0.5]]), [2, 2, 2, 2],
                                np.eye(4), np.eye(4), (4, 4), (2, 2), 1.0)
        self.assertFalse(mask.any())

    def test_no_invalid_intensity_scale(self):
        for scale in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                rasterize(np.zeros((1, 4)), [1, 1, 1, 1], np.eye(4), np.eye(4),
                          (4, 4), (2, 2), scale)

    def test_split_leakage_and_duplicates(self):
        a = dict(scene_id="scene", sample_id="frame1", file="a.npz", split="train")
        for b in [dict(a, sample_id="frame2", file="b.npz", split="val"), dict(a)]:
            with self.assertRaises(ValueError):
                validate_records([a, b])

    def test_scene_identity_is_upstream_not_local_directory(self):
        manifest = dict(format="nidar-camera-cloud-v1", export_status="complete",
                        point_coordinates="world", camera_coordinates="forward-left-up",
                        source_dataset="nuscenes", intensity_encoding="native nuScenes / 255",
                        scene_token="same-original-scene", frames=[dict(frame="frame_000000",
                        sample_token="sample", cameras=[dict(camera_id=1)])])
        for name in ("a", "b"):
            (self.root / name).mkdir()
            (self.root / name / "input_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Scene leakage"):
            enumerate_sources({"sources": [dict(root="a", split="train"),
                                             dict(root="b", split="val")]}, self.root)

    def test_all_samples_including_final_singleton(self):
        batches = epoch_batches(5, 2, 0, 0)
        self.assertEqual([len(b) for b in batches], [2, 3])
        self.assertEqual(sorted(sum(batches, [])), list(range(5)))
        self.assertEqual(batches, epoch_batches(5, 2, 0, 0))
        with self.assertRaises(ValueError):
            epoch_batches(1, 2, 0, 0)

    def test_preparation_normalizes_uint8_once_and_keeps_zero_gt(self):
        for scene, split in [("a", "train"), ("b", "val")]:
            directory = self.root / scene
            (directory / "images").mkdir(parents=True)
            (directory / "pointclouds").mkdir()
            jpeg = cv2.imencode(".jpg", np.full((32, 32, 3), 80, np.uint8))[1].tobytes()
            write_camera(directory, 0, 1, jpeg, [10, 10, 16, 16], np.eye(4))
            write_cloud(directory, 0, [[2, 0, 0]], [0], np.eye(4))
            manifest = dict(format="nidar-camera-cloud-v1", export_status="complete",
                            point_coordinates="world", camera_coordinates="forward-left-up",
                            source_dataset="nuscenes", intensity_encoding="native nuScenes / 255",
                            scene_token=scene, frames=[dict(frame="frame_000000", sample_token=scene,
                            cameras=[dict(camera_id=1)])])
            (directory / "input_manifest.json").write_text(json.dumps(manifest))
        sources = [dict(root="a", split="train"), dict(root="b", split="val")]
        (self.root / "dummy.pth").write_bytes(b"test fixture, not a model")
        config = self.root / "sources.yaml"
        config.write_text(yaml.safe_dump(dict(sources=sources, image_shape=[32, 32],
                                               stn_checkpoint="dummy.pth")))
        with patch("training.prepare.STNWrapper") as stn:
            stn.return_value.process_image.return_value = np.full((32, 32), 128, np.uint8)
            prepare(config, self.root / "pairs")
            self.assertEqual(stn.return_value.process_image.call_count, 2)
        manifest = load_manifest(self.root / "pairs/manifest.json")
        sample = IntensityDataset(self.root / "pairs", manifest, "train")[0]
        self.assertAlmostEqual(float(sample["input"].mean()), 128 / 255, places=6)
        self.assertEqual(int(sample["mask"].sum()), 1)
        self.assertEqual(float(sample["target"].sum()), 0)
        with self.assertRaises(FileExistsError):
            prepare(config, self.root / "pairs")
        sources[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "train and val"):
            enumerate_sources(dict(sources=sources), self.root)

    def test_checksum_and_explicit_mask(self):
        target = np.zeros((32, 32), np.float32)
        mask = np.zeros_like(target, bool)
        mask[0, 0] = True
        path = self.root / "sample.npz"
        np.savez(path, input=target, target=target, valid_mask=mask,
                 rgb=np.zeros((32, 32, 3), np.uint8))
        record = dict(scene_id="a", sample_id="a:0", file=path.name, split="train", sha256=sha256(path))
        manifest = dict(format="nidar-training-v1", status="complete", image_shape=[32, 32], samples=[record])
        metadata = self.root / "manifest.json"
        metadata.write_text(json.dumps(manifest))
        loaded = load_manifest(metadata)
        sample = IntensityDataset(self.root, loaded, "train")[0]
        self.assertEqual(sample["mask"].sum().item(), 1)
        self.assertEqual(sample["target"].sum().item(), 0)
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_manifest(metadata)


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.full((2, 1, 4, 4), 0.5)
        self.r = torch.full_like(self.x, 0.4, requires_grad=True)
        self.s = torch.full_like(self.x, 0.8, requires_grad=True)
        self.pred = self.r * 0.5
        self.target = torch.zeros_like(self.x)
        self.mask = torch.zeros_like(self.x, dtype=torch.bool)
        self.mask[:, :, 0, 0] = True

    def test_reconstruction_targets_input_not_lidar(self):
        total, terms = intrinsic_loss(self.x, self.r, self.s, self.pred, self.target,
                                      self.mask, DEFAULT_WEIGHTS)
        self.assertAlmostEqual(terms["reconstruction"].item(), 0.18, places=6)
        self.assertAlmostEqual(terms["intensity"].item(), 0.2, places=6)
        total.backward()
        self.assertTrue(torch.isfinite(self.r.grad).all())
        self.assertTrue(torch.isfinite(self.s.grad).all())
        self.assertGreater(self.s.grad.abs().sum().item(), 0)

    def test_uncovered_pixels_do_not_supervise_intensity(self):
        self.pred.retain_grad()
        _, terms = intrinsic_loss(self.x, self.r, self.s, self.pred, self.target,
                                  self.mask, DEFAULT_WEIGHTS)
        terms["intensity"].backward()
        self.assertEqual(self.pred.grad[~self.mask].abs().sum().item(), 0)
        self.assertGreater(self.pred.grad[self.mask].abs().sum().item(), 0)

    def test_empty_mask_and_disabled_reconstruction_are_rejected(self):
        for mask, weights in [(torch.zeros_like(self.mask), DEFAULT_WEIGHTS),
                              (self.mask, dict(DEFAULT_WEIGHTS, reconstruction=0))]:
            with self.assertRaises(ValueError):
                intrinsic_loss(self.x, self.r, self.s, self.pred, self.target, mask, weights)


@unittest.skipUnless(hasattr(torch.serialization, "safe_globals"), "requires supported PyTorch")
class TrainingIntegrationTests(unittest.TestCase):
    def test_training_resume_and_inference(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            rng = np.random.default_rng(1)
            for i in range(3):
                path = root / f"{i}.npz"
                x = rng.random((32, 48), dtype=np.float32)
                np.savez(path, input=x, target=x * 0.4, valid_mask=np.ones_like(x, bool),
                         rgb=np.repeat((x[..., None] * 255).astype(np.uint8), 3, axis=2))
                records.append(dict(file=path.name, sha256=sha256(path), sample_id=str(i),
                                    scene_id="train" if i < 2 else "val", split="train" if i < 2 else "val"))
            metadata = root / "manifest.json"
            metadata.write_text(json.dumps(dict(format="nidar-training-v1", status="complete",
                                                  image_shape=[32, 48], samples=records)))
            model_config = dict(base_channels=4, deeplab_base_channels=4,
                                intrinsic_net_type="unet", intensity_head_type="deeplab")
            (root / "model.yaml").write_text(yaml.safe_dump({"model": model_config}))
            config = dict(model_config="model.yaml", seed=3,
                          training=dict(epochs=2, batch_size=2, lr=1e-4, weight_decay=1e-5),
                          loss=dict(weights=DEFAULT_WEIGHTS))
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with patch("training.train.save_preview"):
                continuous = train(config_path, metadata, root / "continuous", epochs=2, threads=2)
                train(config_path, metadata, root / "resumed", epochs=1, threads=2)
                resumed = train(config_path, metadata, root / "resumed", epochs=2,
                                resume=root / "resumed/last.pth", threads=2)
            for key, value in continuous.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            wrapper = DeepIntrinsicWrapper(str(root / "resumed/best_inference.pth"),
                                           image_shape=(32, 48), device="cpu", model_kwargs=model_config)
            outputs = wrapper.process_image(np.full((32, 48), 128, np.uint8))
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(np.isfinite(x).all() for x in outputs))
            with self.assertRaises(FileExistsError):
                train(config_path, metadata, root / "resumed", threads=2)
            with patch("training.train.save_preview"):
                train(config_path, metadata, root / "continuation", epochs=3,
                      resume=root / "resumed/last.pth", threads=2)
            self.assertTrue((root / "continuation/best_inference.pth").is_file())
            config["seed"] = 5
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train(config_path, metadata, root / "resumed", epochs=3,
                      resume=root / "resumed/last.pth", threads=2)
