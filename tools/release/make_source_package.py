#!/usr/bin/env python3
"""Create a lightweight NIDAR source-release folder.

By default this creates the public inference, data-conversion and IRNet training
package, not the full development tree. Unreleased training variants, ROS,
benchmarks and additional routes are not included.
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
    "LICENSE",
    "LICENSES/CS_STEREO_MIT.txt",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    "requirements.txt",
    "constraints-cpu-tested.txt",
    "requirements-viewer.txt",
    "requirements-eval.txt",
    "requirements-waymo-export.txt",
    "requirements-nuscenes-export.txt",
    "docs/DEMO.md",
    "docs/DATA_ACCESS.md",
    "docs/MODEL_ZOO.md",
    "docs/RELEASE_ACCEPTANCE.md",
    "docs/TROUBLESHOOTING.md",
    "docs/TRAINING.md",
    "docs/TRAINING_DATA.md",
    "docs/assets/README.md",
    "docs/assets/nuscenes-gt.png",
    "docs/assets/nuscenes-nidar.png",
    "tests/test_public_release.py",
    "tests/test_release_docs.py",
    "tests/test_checkpoint.py",
    "tests/test_download_weights.py",
    "tests/test_data_export.py",
    "tests/test_training.py",
    "tests/test_historical_training.py",
    "training/__init__.py",
    "training/data.py",
    "training/losses.py",
    "training/historical.py",
    "training/prepare.py",
    "training/train.py",
    "training/config.yaml",
    "training/config_sparse.yaml",
    "training/sources.example.yaml",
    "training/sources.legacy.example.yaml",
    "tools/data_export/common.py",
    "tools/data_export/prepare_waymo.py",
    "tools/data_export/prepare_nuscenes.py",
    "tools/download_weights.py",
    "tools/release/make_source_package.py",
    "inference/examples.py",
    "inference/run_nidar_demo.py",
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
    "inference/src/models/checkpoint.py",
    "inference/src/models/deep_intrinsic_net.py",
    "inference/src/models/deep_intrinsic_wrapper.py",
    "inference/src/models/network_variants.py",
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
