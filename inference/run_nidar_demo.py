#!/usr/bin/env python3
"""End-to-end NIDAR demo entry point.

This wrapper hides the internal multi-stage pipeline behind one command. It
expects a Waymo-style raw export with `images/` and `pointclouds/`, writes final
results into a compact `results/` directory, and can open the synthesized
intensity point cloud in an Open3D viewer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_ROOT = PROJECT_ROOT / "inference"
DEFAULT_CONFIG = INFERENCE_ROOT / "configs" / "waymo_public_template.yaml"
DEFAULT_MODEL_CONFIG = INFERENCE_ROOT / "configs" / "deep_intrinsic_public_model.yaml"
DEFAULT_BUNDLED_QUANTILE = INFERENCE_ROOT / "quantile_model" / "wintensity_mapping_quantile_model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NIDAR end-to-end demo and visualize the synthesized-intensity point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Waymo-style raw directory containing images/ and pointclouds/.",
    )
    parser.add_argument(
        "--weights-root",
        required=True,
        help="Directory containing stn/47.pth and irnet/epoch_010.pth.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/nidar_demo",
        help="Directory for compact demo outputs.",
    )
    parser.add_argument("--frame-count", type=int, default=2, help="Number of frames to process.")
    parser.add_argument("--start-frame", type=int, default=0, help="Index into the discovered frame list.")
    parser.add_argument("--step", type=int, default=1, help="Stride in the discovered frame list.")
    parser.add_argument("--device", default="cuda", help="Torch device requested for STN/IRNet.")
    parser.add_argument(
        "--quantile-model",
        default=None,
        help="Optional quantile model path. Defaults to weights-root/quantile/... if present, otherwise the bundled model.",
    )
    parser.add_argument(
        "--viewer",
        choices=["auto", "open3d", "none"],
        default="auto",
        help="Open the first final colored point cloud after inference.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip GT-vs-pred range-image evaluation and only synthesize intensity point clouds.",
    )
    parser.add_argument(
        "--preview-points",
        type=int,
        default=80000,
        help="Maximum number of points used for the static preview PNG.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the internal pipeline work directory for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-root/results and output-root/_nidar_work if they already exist.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return path


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    input_root = Path(args.input_root).expanduser().resolve()
    weights_root = Path(args.weights_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    images_dir = require_dir(input_root / "images", "input images")
    pointcloud_dir = require_dir(input_root / "pointclouds", "input pointclouds")
    stn_ckpt = require_file(weights_root / "stn" / "47.pth", "STN checkpoint")
    irnet_ckpt = require_file(weights_root / "irnet" / "epoch_010.pth", "IRNet checkpoint")

    if args.quantile_model:
        quantile_model = require_file(Path(args.quantile_model).expanduser().resolve(), "quantile model")
    else:
        weights_quantile = weights_root / "quantile" / "wintensity_mapping_quantile_model.pkl"
        quantile_model = weights_quantile if weights_quantile.exists() else DEFAULT_BUNDLED_QUANTILE
        require_file(quantile_model, "quantile model")

    return {
        "input_root": input_root,
        "images_dir": images_dir,
        "pointcloud_dir": pointcloud_dir,
        "weights_root": weights_root,
        "stn_ckpt": stn_ckpt,
        "irnet_ckpt": irnet_ckpt,
        "quantile_model": quantile_model,
        "output_root": output_root,
        "work_dir": output_root / "_nidar_work",
        "results_dir": output_root / "results",
    }


def prepare_output(paths: Dict[str, Path], overwrite: bool) -> None:
    output_root = paths["output_root"]
    results_dir = paths["results_dir"]
    work_dir = paths["work_dir"]

    if overwrite:
        for path in [results_dir, work_dir]:
            if path.exists():
                shutil.rmtree(path)
    elif (results_dir.exists() and any(results_dir.iterdir())) or (work_dir.exists() and any(work_dir.iterdir())):
        raise FileExistsError(
            f"output-root already contains demo outputs: {output_root}. "
            "Use --overwrite or choose a new --output-root."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)


def write_generated_config(args: argparse.Namespace, paths: Dict[str, Path]) -> Path:
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["image_dir"] = str(paths["images_dir"])
    cfg["dataset"]["pointcloud_dir"] = str(paths["pointcloud_dir"])
    cfg["dataset"]["calib_dir"] = str(paths["images_dir"])
    cfg["dataset"]["work_dir"] = str(paths["work_dir"])
    cfg["dataset"]["start_frame"] = args.start_frame
    cfg["dataset"]["num_frames"] = args.frame_count
    cfg["dataset"]["step"] = args.step

    cfg["stage1"]["checkpoint"] = str(paths["stn_ckpt"])
    cfg["stage1"]["use_gpu"] = args.device != "cpu"

    cfg["stage2_deep"]["checkpoint_path"] = str(paths["irnet_ckpt"])
    cfg["stage2_deep"]["model_config_path"] = str(DEFAULT_MODEL_CONFIG)
    cfg["stage2_deep"]["device"] = args.device
    cfg["stage2_deep"]["use_i_hat_as_r"] = False
    cfg["stage2_deep"]["save_shading"] = False
    cfg["stage2_deep"]["save_reconstruction"] = False

    cfg["stage3"]["save_npy"] = True
    cfg["stage3"]["save_ply"] = True
    cfg["stage3"]["save_gt_pointcloud"] = not args.no_eval
    cfg["stage3"]["skip_existing"] = False

    cfg["stage3_5"]["quantile_model_dir"] = str(paths["quantile_model"].parent)
    cfg["stage3_5"]["quantile_model_path"] = str(paths["quantile_model"])
    cfg["stage3_5"]["save_ply"] = True
    cfg["stage3_5"]["skip_existing"] = False

    cfg["evaluation"]["compute_lpips"] = False
    cfg["evaluation"]["device"] = "cpu"
    cfg["evaluation"]["skip_existing"] = False
    cfg["evaluation"]["waymo_calib_json"] = str(INFERENCE_ROOT / "configs" / "waymo_lidar_calib.json")

    config_path = paths["output_root"] / "nidar_demo_generated_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


def run_internal_pipeline(args: argparse.Namespace, config_path: Path, paths: Dict[str, Path]) -> Path:
    stages = "1,2,3,3.5" if args.no_eval else "1,2,3,3.5,4"
    cmd = [
        sys.executable,
        str(INFERENCE_ROOT / "run_waymo_pipeline.py"),
        str(config_path),
        "--stages",
        stages,
        "--use-deep",
        "--keep-remap-with-deep",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(INFERENCE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = paths["output_root"] / "nidar_internal_pipeline.log"
    print("Running NIDAR end-to-end intensity synthesis...")
    print(f"Internal log: {log_path}")
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        print("\nInternal pipeline failed. Last log lines:", file=sys.stderr)
        print("\n".join(tail), file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    return log_path


def intensity_colors(intensity: np.ndarray, cmap_name: str = "cividis") -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    values = intensity.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros((len(values), 3), dtype=np.uint8)

    lo = float(np.percentile(values[finite], 1.0))
    hi = float(np.percentile(values[finite], 99.0))
    if hi <= lo:
        hi = float(values[finite].max())
        lo = float(values[finite].min())
    denom = max(hi - lo, 1e-6)
    norm = np.clip((values - lo) / denom, 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(norm)
    return (rgba[:, :3] * 255).astype(np.uint8)


def write_colored_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            x, y, z = point[:3]
            intensity = point[3] if point.shape[0] > 3 else 0.0
            r, g, b = color
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {float(intensity):.6f}\n")


def save_preview_png(points: np.ndarray, output_path: Path, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(points) == 0:
        return
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        pts = points[idx]
    else:
        pts = points

    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    intensity = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts), dtype=np.float32)
    ax.scatter(pts[:, 0], pts[:, 1], c=intensity, s=0.12, cmap="cividis", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def frame_name_from_remapped(path: Path) -> str:
    stem = path.stem
    if stem.startswith("dis_learned_"):
        stem = stem[len("dis_learned_") :]
    return stem.replace("_pseudo_intensity", "")


def collect_results(args: argparse.Namespace, paths: Dict[str, Path], internal_log: Path) -> Dict:
    stage3_5_dir = paths["work_dir"] / "stage3_5_remapped"
    eval_dir = paths["work_dir"] / "stage4_evaluation"
    results_dir = paths["results_dir"]

    remapped_files = sorted(stage3_5_dir.glob("dis_learned_*_pseudo_intensity.npy"))
    if not remapped_files:
        raise FileNotFoundError(f"No remapped point clouds found in {stage3_5_dir}")

    summary = {
        "input_root": str(paths["input_root"]),
        "output_root": str(paths["output_root"]),
        "internal_log": str(internal_log),
        "frames": [],
    }

    eval_summary = eval_dir / "evaluation_summary.json"
    if eval_summary.exists():
        shutil.copy2(eval_summary, results_dir / "evaluation_summary.json")
        summary["evaluation_summary"] = str(results_dir / "evaluation_summary.json")
        if not args.no_eval:
            with open(eval_summary, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            num_samples = int(eval_data.get("num_samples", 0))
            num_success = int(eval_data.get("num_success", 0))
            errors = {
                name: value.get("error")
                for name, value in eval_data.get("individual", {}).items()
                if isinstance(value, dict) and value.get("error")
            }
            if num_samples == 0 or num_success != num_samples or errors:
                raise RuntimeError(
                    "Evaluation did not complete for every frame. "
                    f"num_success={num_success}, num_samples={num_samples}, errors={errors}"
                )
    elif not args.no_eval:
        raise FileNotFoundError(f"Evaluation summary not found: {eval_summary}")

    for remapped_npy in remapped_files:
        frame = frame_name_from_remapped(remapped_npy)
        frame_dir = results_dir / frame
        frame_dir.mkdir(parents=True, exist_ok=True)

        points = np.load(str(remapped_npy))
        colors = intensity_colors(points[:, 3] if points.shape[1] > 3 else np.zeros(len(points)))

        final_npy = frame_dir / "nidar_intensity.npy"
        scalar_ply = frame_dir / "nidar_intensity.ply"
        colored_ply = frame_dir / "nidar_intensity_colored.ply"
        preview_png = frame_dir / "nidar_intensity_preview.png"

        shutil.copy2(remapped_npy, final_npy)
        source_scalar_ply = remapped_npy.with_suffix(".ply")
        if source_scalar_ply.exists():
            shutil.copy2(source_scalar_ply, scalar_ply)
        write_colored_ply(points, colors, colored_ply)
        save_preview_png(points, preview_png, args.preview_points)

        copied_eval = {}
        source_eval_dir = eval_dir / frame
        for src_name, dst_name in [
            (f"{frame}_comparison.png", "comparison.png"),
            (f"{frame}_comparison_masked.png", "comparison_masked.png"),
            (f"{frame}_comparison_masked_white.png", "comparison_masked_white.png"),
            (f"{frame}_gt_intensity.png", "gt_intensity.png"),
            (f"{frame}_pred_intensity.png", "pred_intensity.png"),
            (f"{frame}_mask.png", "valid_mask.png"),
            (f"{frame}_metrics.json", "metrics.json"),
        ]:
            src = source_eval_dir / src_name
            if src.exists():
                dst = frame_dir / dst_name
                shutil.copy2(src, dst)
                copied_eval[dst_name] = str(dst)

        summary["frames"].append(
            {
                "frame": frame,
                "nidar_intensity_npy": str(final_npy),
                "nidar_intensity_ply": str(scalar_ply) if scalar_ply.exists() else None,
                "nidar_intensity_colored_ply": str(colored_ply),
                "preview_png": str(preview_png),
                "evaluation_outputs": copied_eval,
            }
        )

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


def maybe_open_viewer(viewer: str, colored_ply: Path) -> None:
    should_open = viewer == "open3d" or (
        viewer == "auto" and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )
    if not should_open:
        print(f"Viewer not opened. Open this colored point cloud manually: {colored_ply}")
        return

    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(colored_ply))
        o3d.visualization.draw_geometries([pcd], window_name="NIDAR synthesized intensity")
    except Exception as exc:
        print(f"Open3D viewer failed: {exc}")
        print(f"Open this colored point cloud manually: {colored_ply}")


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    prepare_output(paths, args.overwrite)
    config_path = write_generated_config(args, paths)
    internal_log = run_internal_pipeline(args, config_path, paths)
    summary = collect_results(args, paths, internal_log)

    if not args.keep_work_dir and paths["work_dir"].exists():
        shutil.rmtree(paths["work_dir"])

    print("\nNIDAR demo complete.")
    print(f"Results: {paths['results_dir']}")
    print(f"Summary: {summary['summary_json']}")
    for frame in summary["frames"]:
        print(f"- {frame['frame']}:")
        print(f"  colored point cloud: {frame['nidar_intensity_colored_ply']}")
        print(f"  preview image: {frame['preview_png']}")
        if "comparison.png" in frame["evaluation_outputs"]:
            print(f"  GT/pred comparison: {frame['evaluation_outputs']['comparison.png']}")

    first_ply = Path(summary["frames"][0]["nidar_intensity_colored_ply"])
    maybe_open_viewer(args.viewer, first_ply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
