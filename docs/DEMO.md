# Minimal Waymo-Style Demo

The minimal demo is intended to show the complete paper route on a tiny
Waymo-style export:

```text
RGB image
  -> STN pseudo-NIR
  -> IRNet reflectance-like R
  -> projection to point cloud
  -> quantile remap
  -> range-image evaluation and visual comparison
```

## Inputs

Download or prepare the artifacts listed in `docs/MODEL_ZOO.md`, then set:

```bash
export PROJECT_ROOT=/path/to/NIDAR
export DATA_ROOT=/path/to/nidar_demo_data
export OUTPUT_ROOT=/path/to/nidar_demo_outputs
export WEIGHTS_ROOT=/path/to/nidar_weights
```

The demo sample should be placed under:

```text
$DATA_ROOT/waymo/demo_scene/raw/images
$DATA_ROOT/waymo/demo_scene/raw/pointclouds
```

## Command

```bash
cd "$PROJECT_ROOT"
cp inference/configs/waymo_public_template.yaml inference/configs/my_waymo_demo.yaml

python inference/run_waymo_pipeline.py \
  inference/configs/my_waymo_demo.yaml \
  --stages 1,2,3,3.5,4 \
  --use-deep \
  --keep-remap-with-deep
```

## Expected Outputs

The default config writes into `$OUTPUT_ROOT/waymo_demo`:

```text
$OUTPUT_ROOT/waymo_demo/
  logs/
    pipeline.log
  stage1_pseudo_nir/
    frame_000000_cam1_nir.png
    ...
  stage2_reflectance/
    frame_000000_cam1_r.png
    ...
  stage3_pointcloud/
    frame_000000_pseudo_intensity.npy
    frame_000000_pseudo_intensity.ply
    gt_pointcloud/
      frame_000000_gt_intensity.npz
      frame_000000_gt_intensity.npy
  stage3_5_remapped/
    dis_learned_frame_000000_pseudo_intensity.npy
    dis_learned_frame_000000_pseudo_intensity.ply
  stage4_evaluation/
    evaluation_summary.json
    frame_000000/
      frame_000000_metrics.json
      frame_000000_mask.png
      frame_000000_comparison.png
      frame_000000_comparison_masked.png
      frame_000000_comparison_masked_white.png
```

## What To Inspect

- `stage1_pseudo_nir/*.png`: grayscale pseudo-NIR images generated from RGB.
- `stage2_reflectance/*_r.png`: IRNet `R` output used by the paper route.
- `stage3_pointcloud/*.ply`: pre-remap point cloud with synthesized intensity.
- `stage3_5_remapped/*.ply`: final remapped point cloud with synthesized
  intensity.
- `stage4_evaluation/*/*_comparison.png`: vertical range-image comparison. The
  top half is the ground-truth LiDAR intensity range image; the bottom half is
  the NIDAR prediction rendered with the same colormap.
- `stage4_evaluation/*/*_comparison_masked.png`: the same comparison restricted
  to pixels where both GT and prediction are valid.
- `stage4_evaluation/evaluation_summary.json`: aggregate RMSE, MAE, MedAE,
  PSNR, SSIM, and optional LPIPS metrics.

For the quickest visual check, open one `*_comparison.png` together with the
matching remapped `.ply` file.
