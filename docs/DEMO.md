# End-To-End Waymo-Style Demo

The public demo is intended to feel like one operation: provide a Waymo-style
raw export, run one command, and inspect a synthesized-intensity point cloud.

```text
Waymo-style raw input
  -> NIDAR end-to-end synthesis
  -> colored intensity point cloud and optional viewer
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
$DATA_ROOT/waymo/demo_scene/raw/
  images/
  pointclouds/
```

## Command

```bash
cd "$PROJECT_ROOT"
python inference/run_nidar_demo.py \
  --input-root "$DATA_ROOT/waymo/demo_scene/raw" \
  --weights-root "$WEIGHTS_ROOT" \
  --output-root "$OUTPUT_ROOT/nidar_demo" \
  --viewer auto
```

Use `--viewer open3d` to force an Open3D point-cloud window, or
`--viewer none` on a headless server. The internal pipeline log is saved, but
the user-facing output is the compact `results/` directory.

## Expected Outputs

The command writes into `$OUTPUT_ROOT/nidar_demo`:

```text
$OUTPUT_ROOT/nidar_demo/
  nidar_demo_generated_config.yaml
  nidar_internal_pipeline.log
  results/
    summary.json
    evaluation_summary.json
    frame_000000/
      nidar_intensity.npy
      nidar_intensity.ply
      nidar_intensity_colored.ply
      nidar_intensity_preview.png
      comparison.png
      comparison_masked.png
      comparison_masked_white.png
      gt_intensity.png
      pred_intensity.png
      valid_mask.png
      metrics.json
```

## What To Inspect

- `nidar_intensity_colored.ply`: final point cloud colored by synthesized
  intensity. This is the primary demo output.
- `nidar_intensity_preview.png`: static top-down preview for quick inspection.
- `comparison.png`: vertical range-image comparison. The top half is the
  ground-truth LiDAR intensity range image; the bottom half is the NIDAR
  prediction rendered with the same colormap.
- `comparison_masked.png`: the same comparison restricted to pixels where both
  GT and prediction are valid.
- `summary.json` and `evaluation_summary.json`: machine-readable output paths
  and aggregate RMSE, MAE, MedAE, PSNR, and SSIM metrics.

For the quickest visual check, inspect `nidar_intensity_colored.ply` in the
viewer and open `comparison.png`.

## Advanced Pipeline Command

The single demo command internally uses the paper-aligned route
`RGB -> pseudo-NIR -> R -> remap -> point cloud`. Advanced users can run the
lower-level pipeline directly:

```bash
python inference/run_waymo_pipeline.py \
  inference/configs/waymo_public_template.yaml \
  --stages 1,2,3,3.5,4 \
  --use-deep \
  --keep-remap-with-deep
```
