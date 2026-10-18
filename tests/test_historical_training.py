"""Generated fixtures for the recovered dense recipe; no redistributed samples."""
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
sys.path.insert(0, str(ROOT / 'inference'))
from training.historical import RECIPE, WEIGHTS, DenseObjective, dense_target, intensity_weight, pair_transform
from training.data import IntensityDataset, epoch_batches, sha256, validate_records
from training.prepare import enumerate_sources, prepare
from training.train import train
from src.models.checkpoint import load_checkpoint
from tools.data_export.common import write_camera, write_cloud


class HistoricalContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_zero_based_warmup(self):
        self.assertEqual([intensity_weight(e) for e in range(4)], [0, 0, 1, 1])
        self.assertEqual(intensity_weight(0, warmup_epochs=0, rampup_epochs=0), 1)
        self.assertEqual(intensity_weight(2, warmup_epochs=1, rampup_epochs=2), 0.5)

    def test_dense_target_support_and_empty(self):
        dense, measured = dense_target(np.array([[2., 0, 0, 0.4]]), [8, 8, 16, 16],
                                        np.eye(4), np.eye(4), (32, 32))
        self.assertEqual(dense.dtype, np.uint8)
        self.assertEqual(int(measured.sum()), 1)
        self.assertGreater(int((dense > 0).sum()), 1)
        empty, support = dense_target(np.array([[-2., 0, 0, 0.4]]), [8, 8, 16, 16],
                                      np.eye(4), np.eye(4), (32, 32))
        self.assertFalse(empty.any())
        self.assertFalse(support.any())

    def test_target_row_crop_supervises_zeros(self):
        nir = np.tile(np.arange(32, dtype=np.uint8)[:, None], (1, 64))
        target = np.zeros((32, 64), np.uint8)
        target[20:24, 20:30] = 100
        x, y, mask, crop = pair_transform(nir, target, (32, 64))
        self.assertEqual(crop, [8, 24])
        self.assertTrue(mask.all())
        self.assertTrue((y == 0).any())
        self.assertEqual(x.dtype, np.float32)
        _, _, empty, crop = pair_transform(nir, np.zeros_like(target), (32, 64))
        self.assertFalse(empty.any())

    def test_loss_gradients_and_reconstruction(self):
        torch.manual_seed(2)
        x, target = torch.rand(2, 1, 32, 64), torch.rand(2, 1, 32, 64)
        r, s, pred = [torch.rand_like(x).requires_grad_() for _ in range(3)]
        mask = target > .2
        objective = DenseObjective(epoch=2)
        total, terms = objective(x, r, s, pred, target, mask)
        self.assertTrue(torch.allclose(terms['reconstruction'], (r * s - x).abs().mean()))
        total.backward()
        for value in [r, s, pred]:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)
        pred2 = pred.detach().requires_grad_()
        warmup, _ = DenseObjective(epoch=1)(x, r.detach(), s.detach(), pred2, target, mask)
        warmup.backward()
        self.assertEqual(float(pred2.grad.abs().sum()), 0)

    def test_historical_batch_and_split_policies(self):
        self.assertEqual([len(b) for b in epoch_batches(5, 2, 0, 0, drop_last=True)], [2, 2])
        with self.assertRaises(ValueError):
            epoch_batches(2, 8, 0, 0, drop_last=True)
        records = [dict(scene_id='a', sample_id='a1', split='train', file='a.npz'),
                   dict(scene_id='a', sample_id='a2', split='val', file='b.npz')]
        with self.assertRaises(ValueError):
            validate_records(records)
        validate_records(records, allow_scene_overlap=True)

    def test_legacy_split_is_seeded_and_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dict(format='nidar-camera-cloud-v1', export_status='complete',
                point_coordinates='world', camera_coordinates='forward-left-up',
                source_dataset='nuscenes', intensity_encoding='native nuScenes / 255',
                scene_token='same-scene', frames=[dict(frame=f'frame_{i:06d}',
                    sample_token=str(i), cameras=[dict(camera_id=1)]) for i in range(20)])
            (root / 'input_manifest.json').write_text(json.dumps(manifest))
            config = dict(split_policy='sample-random-legacy', split_seed=7, val_ratio=.1,
                          sources=[dict(root='.', split='pool')])
            first = enumerate_sources(config, root)
            self.assertEqual(first, enumerate_sources(config, root))
            expected = set(torch.randperm(20, generator=torch.Generator().manual_seed(7)).tolist()[:2])
            self.assertEqual({i for i, row in enumerate(first) if row['split'] == 'val'}, expected)
            self.assertEqual(sum(row['split'] == 'train' for row in first), 18)
            config['sources'][0]['split'] = 'train'
            with self.assertRaisesRegex(ValueError, 'pool'):
                enumerate_sources(config, root)

    def test_dense_prepare_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for scene, split in [('a', 'train'), ('b', 'val')]:
                directory = root / scene
                (directory / 'images').mkdir(parents=True)
                (directory / 'pointclouds').mkdir()
                jpeg = cv2.imencode('.jpg', np.full((32, 64, 3), 80, np.uint8))[1].tobytes()
                write_camera(directory, 0, 1, jpeg, [10, 10, 32, 16], np.eye(4))
                write_cloud(directory, 0, [[2, 0, 0]], [.4], np.eye(4))
                manifest = dict(format='nidar-camera-cloud-v1', export_status='complete',
                    point_coordinates='world', camera_coordinates='forward-left-up',
                    source_dataset='nuscenes', intensity_encoding='native nuScenes / 255',
                    scene_token=scene, frames=[dict(frame='frame_000000', sample_token=scene,
                                                   cameras=[dict(camera_id=1)])])
                (directory / 'input_manifest.json').write_text(json.dumps(manifest))
            checkpoint = root / 'stn.pth'
            checkpoint.write_bytes(b'fixture only, not a real checkpoint')
            config = dict(recipe=RECIPE, stn_checkpoint=str(checkpoint), image_shape=[32, 64],
                          sources=[dict(root=name, split=split) for name, split in [('a', 'train'), ('b', 'val')]])
            path = root / 'sources.yaml'
            path.write_text(yaml.safe_dump(config))
            with patch('training.prepare.STNWrapper') as stn:
                stn.return_value.process_image.return_value = np.full((32, 64), 100, np.uint8)
                manifest = prepare(path, root / 'prepared', threads=1)
            self.assertEqual(manifest['recipe'], RECIPE)
            with np.load(root / 'prepared' / manifest['samples'][0]['file']) as sample:
                self.assertTrue(sample['valid_mask'].all())
                self.assertLess(float(sample['input'].max()), .4)
            config.update(split_policy='sample-random-legacy', val_ratio=.5)
            for source in config['sources']:
                source.pop('split')
            with self.assertRaises(ValueError):
                enumerate_sources(config, root)


@unittest.skipIf(tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 6),
                 'Restricted checkpoint reload requires PyTorch >= 2.6')
class HistoricalResumeTests(unittest.TestCase):
    def test_three_epoch_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'samples').mkdir()
            records = []
            rng = np.random.RandomState(2)
            for i in range(6):
                relative = f'samples/{i}.npz'
                np.savez_compressed(root / relative,
                    input=rng.rand(32, 64).astype(np.float32), target=rng.rand(32, 64).astype(np.float32),
                    valid_mask=np.ones((32, 64), bool), rgb=np.zeros((32, 64, 3), np.uint8))
                records.append(dict(file=relative, sha256=sha256(root / relative), scene_id='val' if i > 3 else 'train',
                                    sample_id=str(i), split='val' if i > 3 else 'train'))
            manifest = dict(format='nidar-training-v1', status='complete', recipe=RECIPE,
                            image_shape=[32, 64], samples=records)
            data = root / 'manifest.json'
            data.write_text(json.dumps(manifest))
            model = yaml.safe_load((ROOT / 'inference/configs/deep_intrinsic_public_model.yaml').read_text())
            model['model'].update(base_channels=4, deeplab_base_channels=4)
            (root / 'model.yaml').write_text(yaml.safe_dump(model))
            config = yaml.safe_load((ROOT / 'training/config.yaml').read_text())
            config['model_config'] = 'model.yaml'
            config['training']['batch_size'] = 2
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            with patch('training.train.save_preview'):
                train(path, data, root / 'full', epochs=3, threads=1)
                train(path, data, root / 'resumed', epochs=1, threads=1)
                train(path, data, root / 'resumed', epochs=3, resume=root / 'resumed/last.pth', threads=1)
            a, b = [load_checkpoint(root / name / 'last.pth') for name in ['full', 'resumed']]
            for key, value in a['model_state_dict'].items():
                self.assertTrue(torch.equal(value, b['model_state_dict'][key]), key)
            self.assertEqual(a['scheduler_state_dict'], b['scheduler_state_dict'])
            self.assertEqual(a['numpy_rng_state'], b['numpy_rng_state'])
            self.assertEqual(b['val_metrics']['intensity_weight'], 1)
            self.assertLess(b['optimizer_state_dict']['param_groups'][0]['lr'], .0001)
            inference = load_checkpoint(root / 'resumed/last_inference.pth')
            self.assertEqual(inference['epoch'], 2)
            for key, value in b['model_state_dict'].items():
                self.assertTrue(torch.equal(value, inference['model_state_dict'][key]))


if __name__ == '__main__':
    unittest.main()
