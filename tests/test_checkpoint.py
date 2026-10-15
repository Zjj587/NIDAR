"""Checkpoint compatibility tests using generated fixtures, not paper results."""

import argparse
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from src.models.checkpoint import load_checkpoint
from src.models.deep_intrinsic_net import DeepIntrinsicIntensityNet
from src.models.deep_intrinsic_wrapper import DeepIntrinsicWrapper
from src.models.stn_wrapper import STNWrapper


class UnexpectedMetadata:
    pass


@unittest.skipUnless(hasattr(torch.serialization, "safe_globals"), "requires the supported PyTorch environment")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoint.pth"

    def test_legacy_metadata_is_scoped(self):
        torch.save({"stnet": {"value": torch.ones(1)},
                    "opt": argparse.Namespace(epoch=47), "vis": np.float64(1.0)}, self.path)
        before = list(torch.serialization.get_safe_globals())
        data = load_checkpoint(self.path)
        self.assertEqual(data["opt"].epoch, 47)
        self.assertEqual(float(data["vis"]), 1.0)
        self.assertEqual(list(torch.serialization.get_safe_globals()), before)

    def test_unknown_objects_are_not_unpickled(self):
        torch.save({"metadata": UnexpectedMetadata()}, self.path)
        with self.assertRaises(pickle.UnpicklingError):
            load_checkpoint(self.path)

    def test_numpy_1_scalar_module_path_loads_on_numpy_2(self):
        current = self.path.with_name("current.pth")
        torch.save({"scalar": np.float64(1.5)}, current)
        with zipfile.ZipFile(current) as source, zipfile.ZipFile(self.path, "w") as target:
            for entry in source.infolist():
                payload = source.read(entry.filename)
                if entry.filename.endswith("data.pkl"):
                    payload = payload.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")
                    self.assertIn(b"numpy.core.multiarray", payload)
                target.writestr(entry, payload)
        before = list(torch.serialization.get_safe_globals())
        self.assertEqual(float(load_checkpoint(self.path)["scalar"]), 1.5)
        self.assertEqual(list(torch.serialization.get_safe_globals()), before)

    def test_STN_does_not_accept_missing_parameters(self):
        torch.save({"stnet": {}}, self.path)
        with patch("src.models.stn_wrapper.STN", return_value=torch.nn.Linear(2, 1)):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                STNWrapper(str(self.path), device="cpu")

    def test_IRNet_different_head_is_rejected(self):
        kwargs = {"base_channels": 4, "deeplab_base_channels": 4, "intensity_head_type": "conv"}
        model = DeepIntrinsicIntensityNet(**kwargs)
        torch.save({"model_state_dict": model.state_dict()}, self.path)
        kwargs["intensity_head_type"] = "deeplab"
        with self.assertRaises(RuntimeError):
            DeepIntrinsicWrapper(str(self.path), image_shape=(32, 48), device="cpu", model_kwargs=kwargs)

    def test_matching_full_architecture_runs(self):
        kwargs = {"base_channels": 4, "deeplab_base_channels": 4, "intensity_head_type": "deeplab"}
        model = DeepIntrinsicIntensityNet(**kwargs)
        torch.save({"model_state_dict": model.state_dict()}, self.path)
        wrapper = DeepIntrinsicWrapper(str(self.path), image_shape=(32, 48), device="cpu", model_kwargs=kwargs)
        outputs = wrapper.process_image(np.zeros((32, 48), dtype=np.uint8))
        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(output.shape, (32, 48))
            self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
