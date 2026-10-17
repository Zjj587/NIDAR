"""Scene-disjoint manifests and sparse supervision; no dataset redistribution."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_records(records):
    scenes, samples, files = {}, set(), set()
    for record in records:
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        scene = record["scene_id"]
        if scene in scenes and scenes[scene] != split:
            raise ValueError(f"Scene leakage across splits: {scene}")
        scenes[scene] = split
        sample = record["sample_id"]
        if sample in samples or record["file"] in files:
            raise ValueError(f"Duplicate sample or file: {sample}")
        samples.add(sample)
        files.add(record["file"])
    if not records:
        raise ValueError("Empty training manifest")


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "nidar-training-v1" or manifest.get("status") != "complete":
        raise ValueError("Expected a complete nidar-training-v1 manifest")
    validate_records(manifest["samples"])
    for record in manifest["samples"]:
        target = (path.parent / record["file"]).resolve()
        if path.parent not in target.parents or not target.is_file():
            raise ValueError("Missing or out-of-root training sample")
        if sha256(target) != record["sha256"]:
            raise ValueError(f"Training sample checksum mismatch: {record['sample_id']}")
    return manifest


class IntensityDataset(Dataset):
    def __init__(self, root, manifest, split):
        self.root = Path(root)
        self.records = [r for r in manifest["samples"] if r["split"] == split]
        self.shape = tuple(manifest["image_shape"])
        if not self.records:
            raise ValueError(f"No samples in split {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(self.root / record["file"], allow_pickle=False) as sample:
            x, target, mask = (sample[k] for k in ("input", "target", "valid_mask"))
            rgb = sample["rgb"]
        if x.shape != self.shape or target.shape != self.shape or mask.shape != self.shape:
            raise ValueError("Sample shape does not match manifest; do not resize sparse targets")
        if mask.dtype != np.bool_ or not mask.any():
            raise ValueError("Expected a nonempty explicit boolean validity mask")
        if (not np.isfinite(x).all() or not np.isfinite(target).all() or
                x.min() < 0 or x.max() > 1 or target.min() < 0 or target.max() > 1):
            raise ValueError("Training arrays must be finite and in [0, 1]")
        return {"input": torch.from_numpy(x.astype(np.float32))[None],
                "target": torch.from_numpy(target.astype(np.float32))[None],
                "mask": torch.from_numpy(mask)[None], "rgb": rgb,
                "sample_id": record["sample_id"]}


def epoch_batches(count, batch_size, seed, epoch):
    if count < 2 or batch_size < 2:
        raise ValueError("DeepLab training BatchNorm requires at least two samples per batch")
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(count, generator=generator).tolist()
    batches = [indices[i:i + batch_size] for i in range(0, count, batch_size)]
    # Keep every sample; a final singleton joins the preceding batch for ASPP BN.
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches
