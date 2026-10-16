# End-To-End Demo

Prepare Waymo or nuScenes input with the [official-data guide](DATA_ACCESS.md),
run one command, and inspect a synthesized-intensity point cloud.

```text
Prepared camera images, point cloud and calibration
  -> NIDAR end-to-end synthesis
  -> colored intensity point cloud and optional viewer
```

## Inputs

For a CPU-only Linux x86_64 machine with a Python 3.10 environment, use:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints-cpu-tested.txt
# Optional desktop window:
python -m pip install -r requirements-viewer.txt -c constraints-cpu-tested.txt
```

The constraints record the release-check environment; they are not a CUDA
environment specification. Add `--device cpu` to the demo command below.

Download or prepare the artifacts listed in `docs/MODEL_ZOO.md`, then set:

```bash
export PROJECT_ROOT=/path/to/NIDAR
export DATA_ROOT=/path/to/nidar_demo_data
export OUTPUT_ROOT=/path/to/nidar_demo_outputs
export WEIGHTS_ROOT=/path/to/nidar_weights
```

Prepare the public checkpoints and bundled Waymo calibration with:

```bash
python tools/download_weights.py --weights-root "$WEIGHTS_ROOT"
```

This pins the published model revision, verifies file sizes and SHA-256, and
creates the expected STN/IRNet/quantile subdirectories without a Hub login.
It does not download data. Waymo and nuScenes inputs must be obtained from
their official providers; see [download and conversion commands](DATA_ACCESS.md).
Fresh official-data conversion and real nuScenes inference still need release
acceptance. UE5 and Isaac Sim demos are deferred.

Large downloads print progress periodically. A file is ready only after the
downloader reports that its size and SHA-256 have been verified. An interrupted
download is removed and retried from the start on the next invocation; already
verified files are reused.

For example, the new converters produce:

```text
$DATA_ROOT/waymo_demo/
  input_manifest.json
  images/
  pointclouds/
```

## Command

```bash
cd "$PROJECT_ROOT"
python inference/run_nidar_demo.py \
  --input-root "$DATA_ROOT/waymo_demo" \
  --weights-root "$WEIGHTS_ROOT" \
  --output-root "$OUTPUT_ROOT/nidar_demo" \
  --viewer auto
```

Use `--viewer open3d` to force an Open3D point-cloud window, or
`--viewer none` on a headless server. The internal pipeline log is saved, but
the user-facing output is the compact `results/` directory.

For a converted nuScenes input, use `--input-root "$DATA_ROOT/nuscenes_demo"`.
The manifest selects the dataset and remap model; `--dataset waymo` or
`--dataset nuscenes` can assert that choice. Legacy Waymo exports without a
manifest remain supported. Add `--check-input` for validation without inference
or output writes; it checks artifact paths, not checkpoint hashes or loading.

Install `requirements-viewer.txt` for the window. Drag to rotate the cloud,
scroll to zoom, and close the window to finish the command. Static previews and
comparison images do not require Open3D. `--device cpu` selects CPU inference.

The default requests two frames. Use `--frame-count 1` for a one-frame input;
the command rejects missing frames, camera files, and partial pipeline outputs.
It also checks runtime dependencies before creating or overwriting outputs.
`--no-eval` skips the GT comparison but still produces the intensity point cloud.
The default IRNet checkpoint must be the U-Net + DeepLab model listed in the
model zoo. Its R branch is projected and remapped; I_hat is not used as the
final output.

When preparing your local export, keep the actual image, point-cloud, and
calibration files together. A directory containing symlinks into a development
machine is not portable. This project does not provide a mirrored Waymo sample.

## Expected Outputs

The command writes into `$OUTPUT_ROOT/nidar_demo`:

```text
$OUTPUT_ROOT/nidar_demo/
  nidar_demo_generated_config.yaml
  nidar_internal_pipeline.log
  results/
    input_manifest.json  # when provided by the converter
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

Unavailable metrics are `null` in the public `results/` JSON files. In
particular, default runs do not compute LPIPS; `null` does not mean zero error.

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
