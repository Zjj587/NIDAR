#!/usr/bin/env python3
"""Create a lightweight NIDAR source-release folder.

By default this creates the public Waymo inference package, not the full
development tree. Training, ROS, benchmark, and additional dataset routes still
need separate path and license cleanup before public packaging.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path


DEFAULT_EXCLUDES = (
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.o",
    "build",
    "dist",
    "*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".vscode",
    ".idea",
    "data",
    "datasets",
    "outputs",
    "work",
    "runs",
    "logs",
    "checkpoints",
    "*.bag",
    "*.pcd",
    "*.ply",
    "*.npy",
    "*.npz",
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.bak",
    "*.pre_release_*",
    "docs/*",
    "inference/configs/*_example.yaml",
    "inference/configs/my_*.yaml",
    "training/configs/*.yaml",
)

PUBLIC_WAYMO_INCLUDE = (
    ".gitignore",
    "README.md",
    "requirements.txt",
    "docs/OPEN_SOURCE_RELEASE_CHECKLIST.md",
    "tools/release/make_source_package.py",
    "inference/examples.py",
    "inference/run_waymo_pipeline.py",
    "inference/configs/waymo_public_template.yaml",
    "inference/configs/deep_intrinsic_public_model.yaml",
    "inference/configs/waymo_lidar_calib.json",
    "inference/quantile_model/*.pkl",
    "inference/quantile_model/new/*.pkl",
    "inference/src/__init__.py",
    "inference/src/datasets/__init__.py",
    "inference/src/datasets/waymo_dataset.py",
    "inference/src/evaluation/__init__.py",
    "inference/src/evaluation/intensity_evaluator.py",
    "inference/src/evaluation/metrics.py",
    "inference/src/evaluation/spherical_projection.py",
    "inference/src/models/__init__.py",
    "inference/src/models/deep_intrinsic_net.py",
    "inference/src/models/deep_intrinsic_wrapper.py",
    "inference/src/models/stn_net.py",
    "inference/src/models/stn_wrapper.py",
    "inference/src/projection/__init__.py",
    "inference/src/projection/projection_utils.py",
    "inference/src/stages/__init__.py",
    "inference/src/stages/stage1_rgb_to_pseudo_nir.py",
    "inference/src/stages/stage2_deep_intrinsic.py",
    "inference/src/stages/stage2_pseudo_nir_to_reflectance.py",
    "inference/src/stages/stage3_reflectance_to_pointcloud.py",
    "inference/src/stages/stage3_5_remap_intensity.py",
    "inference/src/stages/stage4_evaluate.py",
    "inference/src/utils/__init__.py",
    "inference/src/utils/config.py",
    "inference/src/utils/image_utils.py",
    "inference/src/utils/logger.py",
)

V0_ALLOWED = (
    "V0/*.md",
    "V0/configs/*.yaml",
    "V0/p0_visual_comparison/*.md",
    "V0/p0_visual_comparison/*.png",
    "V0/p0_qualitative_intensity_comparison/README.md",
    "V0/p0_qualitative_intensity_comparison/manifest.json",
)

PUBLIC_ALWAYS_INCLUDE = (
    "docs/OPEN_SOURCE_RELEASE_CHECKLIST.md",
    "inference/configs/*_public_template.yaml",
    "inference/configs/*_lidar_calib.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-v0", action="store_true")
    return parser.parse_args()


def matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    text = path.as_posix()
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def should_copy(rel: Path, include_v0: bool) -> bool:
    if matches_any(rel, PUBLIC_WAYMO_INCLUDE):
        return True
    if rel.parts and rel.parts[0] == "V0":
        return include_v0 and matches_any(rel, V0_ALLOWED)
    return False


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == source_root or source_root in out_dir.parents:
        raise ValueError("Choose an output directory outside the source tree.")
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    copied = 0
    for src in sorted(source_root.rglob("*")):
        rel = src.relative_to(source_root)
        if not should_copy(rel, args.include_v0):
            continue
        dst = out_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} files to {out_dir}")


if __name__ == "__main__":
    main()
