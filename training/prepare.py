"""Build pseudo-NIR / LiDAR pairs with an explicit target and split policy."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from src.models.stn_wrapper import STNWrapper
from src.projection.projection_utils import project_points_to_image
from tools.data_export.common import matrix, source_path, write_json
from training.data import sha256, validate_records
from training.historical import RECIPE, dense_target, pair_transform


def rasterize(points, intrinsic, extrinsic, world_from_reference, source_shape, shape, scale):
    """Nearest camera-depth return per output pixel; keep true zero returns valid."""
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("Expected finite [world X, Y, Z, intensity] points")
    if not np.isfinite(scale) or scale <= 0 or np.any(points[:, 3] < 0):
        raise ValueError("Expected nonnegative intensities and a positive intensity scale")
    h, w = shape
    sh, sw = source_shape
    _, uv, indices = project_points_to_image(points, intrinsic, extrinsic, sw, sh,
                                             world_from_reference)
    target = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), bool)
    if not len(indices):
        return target, mask, 0
    pixel = np.floor(uv * [w / sw, h / sh]).astype(np.int64)
    pixel = np.clip(pixel, [0, 0], [w - 1, h - 1])
    flat = pixel[:, 1] * w + pixel[:, 0]
    camera_from_world = np.linalg.inv(world_from_reference @ extrinsic)
    depth = points[indices, :3] @ camera_from_world[0, :3] + camera_from_world[0, 3]
    order = np.argsort(depth, kind="stable")
    _, first = np.unique(flat[order], return_index=True)
    chosen = order[first]
    values = points[indices[chosen], 3] / scale
    target.flat[flat[chosen]] = np.clip(values, 0, 1)
    mask.flat[flat[chosen]] = True
    return target, mask, int(np.count_nonzero(values > 1))


def enumerate_sources(config, config_dir):
    rows, scenes, seen = [], {}, set()
    policy = config.get("split_policy", "scene-disjoint")
    if policy not in {"scene-disjoint", "sample-random-legacy"}:
        raise ValueError("Unknown split policy")
    for source_index, source in enumerate(config["sources"]):
        root = (config_dir / source["root"]).resolve()
        path = root / "input_manifest.json"
        manifest = json.loads(path.read_text())
        if (manifest.get("format") != "nidar-camera-cloud-v1" or
                manifest.get("export_status") != "complete" or
                manifest.get("point_coordinates") != "world" or
                manifest.get("camera_coordinates") != "forward-left-up"):
            raise ValueError("Use a completed official-data export from tools/data_export")
        dataset = manifest["source_dataset"]
        if dataset not in {"waymo", "nuscenes"}:
            raise ValueError("This recipe accepts Waymo and nuScenes exports")
        expected_encoding = "native Waymo" if dataset == "waymo" else "native nuScenes / 255"
        if manifest.get("intensity_encoding") != expected_encoding:
            raise ValueError("Unrecognized source intensity encoding")
        manifest_sha256 = sha256(path)
        split = source.get("split", "pool")
        if policy == "sample-random-legacy" and split != "pool":
            raise ValueError("Legacy random splitting requires pool sources; do not mix explicit splits")
        if policy == "scene-disjoint" and split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        for frame in manifest["frames"]:
            scene = frame.get("context_name") if dataset == "waymo" else manifest.get("scene_token")
            identity = frame.get("timestamp_micros") if dataset == "waymo" else frame.get("sample_token")
            if not scene or identity is None:
                raise ValueError("Missing upstream scene/frame identity")
            scene_id = f"{dataset}:{scene}"
            if scene_id in scenes and scenes[scene_id] != split:
                raise ValueError(f"Scene leakage across splits: {scene_id}")
            scenes[scene_id] = split
            for camera in frame["cameras"]:
                camera_id = camera if isinstance(camera, int) else camera["camera_id"]
                if source.get("cameras") and camera_id not in source["cameras"]:
                    continue
                sample_id = f"{scene_id}:{identity}:cam{camera_id}"
                if sample_id in seen:
                    raise ValueError(f"Duplicate upstream sample: {sample_id}")
                seen.add(sample_id)
                rows.append(dict(root=root, frame=frame["frame"], camera=camera_id,
                                 scene_id=scene_id, sample_id=sample_id, split=split,
                                 intensity_scale=float(source.get("intensity_scale", 1.0)),
                                 source_index=source_index, manifest_sha256=manifest_sha256))
    if policy == "sample-random-legacy":
        ratio = float(config.get("val_ratio", 0.1))
        if not 0 < ratio < 1:
            raise ValueError("val_ratio must be between zero and one")
        val_count = int(len(rows) * ratio)
        if not val_count or len(rows) - val_count < 2:
            raise ValueError("Legacy split needs nonempty validation and at least two training samples")
        generator = torch.Generator().manual_seed(int(config.get("split_seed", 0)))
        val_indices = set(torch.randperm(len(rows), generator=generator).tolist()[:val_count])
        for index, row in enumerate(rows):
            row["split"] = "val" if index in val_indices else "train"
    if not {"train", "val"}.issubset({row["split"] for row in rows}):
        raise ValueError("Provide nonempty, scene-disjoint train and val exports")
    return rows


def prepare(config_path, output, device="cpu", threads=4):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = yaml.safe_load(config_path.read_text())
    recipe = config.get("recipe", "sparse-v1")
    if recipe not in {"sparse-v1", RECIPE}:
        raise ValueError("Unknown target recipe")
    rows = enumerate_sources(config, config_path.parent)
    exposure = float(config.get("exposure_ratio", 0.5))
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("exposure_ratio must be finite and positive")
    shape = tuple(config.get("image_shape", [384, 960] if recipe == RECIPE else [384, 1248]))
    if len(shape) != 2 or any(not isinstance(x, int) or x < 32 for x in shape):
        raise ValueError("image_shape must contain two integer dimensions >= 32")
    if output.exists():
        raise FileExistsError("Use a new output directory; partial runs are not reused")
    torch.set_num_threads(threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = (config_path.parent / config["stn_checkpoint"]).resolve()
    stn = STNWrapper(checkpoint, image_shape=(384, 1248), device=device)
    output.mkdir(parents=True)
    (output / "samples").mkdir()
    result = dict(format="nidar-training-v1", status="incomplete", image_shape=list(shape),
                  recipe=recipe, split_policy=config.get("split_policy", "scene-disjoint"),
                  stn_sha256=sha256(checkpoint), stn_image_shape=[384, 1248],
                  exposure_ratio=exposure, samples=[],
                  target_policy="nearest-depth per pixel; explicit mask; clip(raw / scale, 0, 1)")
    if recipe == RECIPE:
        result["target_policy"] = "input-X collision order; image p1/p99; dilate5; Gaussian2; uint8; target-row crop; full crop mask"
    if result["split_policy"] == "sample-random-legacy":
        result.update(split_seed=int(config.get("split_seed", 0)),
                      val_ratio=float(config.get("val_ratio", 0.1)),
                      split_warning="Sample-random validation is NOT held-out scene generalization")
    write_json(output / "manifest.json", result)
    for index, row in enumerate(rows):
        root, stem, cam = row["root"], row["frame"], row["camera"]
        rgb = cv2.imread(str(source_path(root, f"images/{stem}_cam{cam}.jpg")))
        if rgb is None:
            raise ValueError("Cannot decode camera image")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        calib = json.loads(source_path(root, f"images/{stem}_cam{cam}.json").read_text())
        lidar = json.loads(source_path(root, f"pointclouds/{stem}_lidar1.json").read_text())
        if rgb.shape[:2] != (calib["height"], calib["width"]):
            raise ValueError("Image/calibration dimensions differ")
        with np.load(source_path(root, f"pointclouds/{stem}_raw_pc.npz"), allow_pickle=False) as cloud:
            points = cloud["points"]
        nir = stn.process_image(rgb, exp_ratio=result["exposure_ratio"])
        if nir.dtype != np.uint8:
            raise ValueError("STN wrapper contract changed: expected uint8 pseudo-NIR")
        size = (shape[1], shape[0])
        extra = {}
        if recipe == RECIPE:
            dense, measured = dense_target(points, calib["intrinsic"], matrix(calib["extrinsic"]),
                                          matrix(lidar["ego2world"]), rgb.shape[:2])
            x, target, mask, crop = pair_transform(nir, dense, shape)
            from PIL import Image
            bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            preview = np.asarray(Image.fromarray(rgb).resize(size, bilinear))
            preview = np.asarray(Image.fromarray(preview[crop[0]:crop[1]]).resize(size, bilinear))
            clipped = None
            extra = dict(raw_projected_pixels=int(measured.sum()), crop_rows=crop,
                         target_nonzero_pixels=int((target > 0).sum()))
        else:
            target, mask, clipped = rasterize(points, calib["intrinsic"], matrix(calib["extrinsic"]),
                                              matrix(lidar["ego2world"]), rgb.shape[:2], shape,
                                              row["intensity_scale"])
            x = cv2.resize(nir.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_LINEAR)
            preview = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if not mask.any():
            raise ValueError(f"No usable supervision for {row['sample_id']}")
        relative = f"samples/{index:08d}.npz"
        np.savez_compressed(output / relative, input=x, target=target, valid_mask=mask,
                            rgb=preview)
        record = {k: row[k] for k in ("scene_id", "sample_id", "split", "intensity_scale",
                                      "source_index", "manifest_sha256")}
        record.update(file=relative, sha256=sha256(output / relative),
                      valid_pixels=int(mask.sum()), clipped_pixels=clipped, **extra)
        result["samples"].append(record)
        detail = (f"{extra['raw_projected_pixels']} raw projected pixels; {mask.sum()} training pixels"
                  if recipe == RECIPE else f"{mask.sum()} valid, {clipped} clipped")
        print(f"[{index + 1}/{len(rows)}] {row['split']}: {detail}", flush=True)
    validate_records(result["samples"], allow_scene_overlap=result["split_policy"] == "sample-random-legacy")
    result["status"] = "complete"
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    prepare(args.config, args.output, args.device, args.threads)


if __name__ == "__main__":
    main()
