# IRNet Training

This is a cleaned, runnable **U-Net + DeepLab** training recipe sharing the
public inference model. It retains intrinsic reconstruction and sparse LiDAR
supervision. It is **not an exact reproduction recipe for the hosted weights**:
the released IRNet checkpoint does not contain its complete training configuration.
The default optimizer, loss weights and preprocessing below are explicit baseline
choices, not a claim to recover the original paper's training run.

## Prepare Data

Use the inference environment from the [quick start](../README.md#quick-start).
Obtain data yourself under the provider's terms and follow [data access](DATA_ACCESS.md).
No raw or prepared Waymo/nuScenes training samples are included or redistributed.

For example, in the separate nuScenes export environment, export two **different
scenes** from the official mini split for a small engineering check:

```bash
python tools/data_export/prepare_nuscenes.py --dataroot ./data/nuscenes \
  --version v1.0-mini --scene scene-0061 --frame-count 2 --output-root ./data/scene_train
python tools/data_export/prepare_nuscenes.py --dataroot ./data/nuscenes \
  --version v1.0-mini --scene scene-0103 --frame-count 2 --output-root ./data/scene_val
```

For real training, choose adequate training/validation scenes from the official
split and keep a separate held-out test split. The mini example is not a benchmark.
Waymo exports use the same training preparation interface; assign separate
segments to separate splits. A two-frame, single-segment fixture cannot establish
scene-disjoint train/validation sets.

Edit [sources.example.yaml](../training/sources.example.yaml) to list your local
exports. Paths are relative to that YAML file. All frames in each export are used;
`cameras` optionally selects camera IDs. More exports may share the same split.
Split checks use upstream scene identities, not directory names: images or
neighboring frames from one scene cannot enter multiple splits.

Back in the NIDAR environment, run from the repository root:

```bash
python -m training.prepare --config training/sources.example.yaml \
  --output ./data/irnet_pairs --device cpu
```

The frozen STN runs at its released 384 x 1248 internal resolution and exposure
ratio 0.5. Its uint8 pseudo-NIR output is normalized **once** to [0, 1]. The
`image_shape` option controls the prepared IRNet resolution, not STN resolution.
Preparation saves float32 inputs/targets, RGB previews and a separate boolean
LiDAR validity mask. Sparse targets are projected directly at the target size,
not bilinearly resized or dilated. When returns share a pixel, the nearest in
camera depth is retained. This is a raster z-buffer, not a complete visibility
or moving-object correction.

Targets use `clip(exported_intensity / intensity_scale, 0, 1)`:

- nuScenes converter output is already divided by 255; use `intensity_scale: 1.0`.
- Waymo converter output is native intensity; scale 1.0 clips values above 1.
  Select and document a fixed sensor-appropriate scale using **training data only**.
  The manifest reports clipped-pixel counts so this choice is inspectable.
- Zero-intensity returns remain valid. Uncovered pixels do not supervise intensity.
- No per-image quantile mapping, GT min/max fit or test-set calibration occurs.

Preparation fails on missing supervision, scene leakage, duplicates, incomplete
exports or existing output directories. It records sample checksums and STN
identity in `manifest.json`. Locally prepared samples remain subject to their
upstream data terms; do not treat them as unrestricted redistribution artifacts.

## Train And Resume

```bash
python -m training.train --data ./data/irnet_pairs/manifest.json \
  --output ./runs/irnet --device cuda
```

The [default configuration](../training/config.yaml) uses Adam, constant learning
rate 1e-4, weight decay 1e-5, batch size 4, and 100 epochs from scratch. These are
starting settings, not a convergence guarantee. The model is the same full
U-Net + DeepLab architecture used by the released inference configuration.
The STN is **not trained** by this command. No pretrained torchvision weights,
perceptual-network downloads or additional training dependencies are required.

For a bounded CPU smoke, set `image_shape: [64, 192]` before preparation and use
`--device cpu --epochs 1`. This tests mechanics at reduced resolution, not quality.
Batch size must be at least 2 for the ASPP global-pooling BatchNorm branch. A final
singleton joins the preceding batch, so every training sample is retained and
the largest batch may be one larger than the configured size.

Optional initialization from the released IRNet (new optimizer, new run):

```bash
python -m training.train --data ./data/irnet_pairs/manifest.json \
  --output ./runs/irnet_finetune --device cuda \
  --init-checkpoint ./weights/irnet/epoch_010.pth
```

Resume at an **epoch boundary**, restoring model, Adam and PyTorch RNG state:

```bash
python -m training.train --data ./data/irnet_pairs/manifest.json \
  --output ./runs/irnet --device cuda --resume ./runs/irnet/last.pth --epochs 100
```

`--epochs` is the total target count, not extra epochs. Data/config changes are
rejected on resume. A failed partial epoch is rerun from the last complete
checkpoint; if history and checkpoint differ, use a new output directory.
When resuming into a new directory, best-checkpoint selection restarts over
that continuation's completed epochs; the previous run's best is not copied.
Seeded ordering and deterministic cuDNN settings improve reproducibility, but
bitwise CUDA or cross-platform equivalence is not asserted.

## Objective And Outputs

For input pseudo-NIR `X`, IRNet returns `R, S, I_hat, R*S`:

| Term | Definition | Weight |
| --- | --- | --- |
| Reconstruction | Mean absolute error of `R*S` against `X`, not LiDAR GT | 1.0 |
| Reflectance smoothness | Input-edge-weighted spatial gradients of `R` | 0.1 |
| Shading smoothness | L1 difference between `S` and its 5 x 5 local average | 0.05 |
| Reflectance mean | Absolute difference between batch mean of `R` and 0.5 | 0.1 |
| Sparse intensity | L1 of `I_hat` against GT at explicitly valid pixels | 1.0 |

The historical mean proxy is named **mean**, not median. Reconstruction and
intensity weights must be positive. There is no warmup, perceptual, region or
distribution loss in this baseline recipe. Intensity supervision backpropagates
through the DeepLab head into the U-Net. No direct reflectance ground truth is assumed.

Each epoch saves:

- `history.csv`: train/validation component losses, valid-pixel intensity MAE/RMSE,
  sample counts and elapsed seconds. Component losses are batch/sample-averaged;
  MAE/RMSE are aggregated over valid pixels. They need not be numerically identical.
- `last.pth`: resumable checkpoint; `best.pth`: minimum full validation objective.
- `best_inference.pth`: corresponding model-only checkpoint, no Adam state.
- `model_config.yaml` and `run_from_epoch_*.json`: model, loss, seed and data identity.
- `previews/epoch_*.png`: first validation sample, RGB/pseudo-NIR/R/S/reconstruction/
  sparse GT/intensity-head/error. Blue-gray indicates missing GT, not zero intensity.
  S uses its own display maximum; other grayscale panels use [0, 1].

These validation diagnostics evaluate the **intensity head**, not the final
point-cloud deployment route. Deployment remains **R + remap**; best intrinsic
validation loss is not proof of best deployment accuracy. Retraining can change
the R distribution: calibrate using training data only and validate the remap
separately before replacing released weights. Full convergence, GPU performance
and end-to-end quality after retraining require separate experiments.

For model-only inference, pass `best_inference.pth` and the saved model kwargs to
`DeepIntrinsicWrapper`, or configure `stage2_deep.checkpoint_path` and
`stage2_deep.model_config_path` in the YAML pipeline. The one-command demo still uses
the released weight filenames/configuration and must not be mistaken for automatic
checkpoint or calibration selection.

## Verification

```bash
python -m unittest discover -s tests -p 'test_training.py' -v
```

Tests cover projection, valid zeros, masks, split leakage, checksums, gradient
flow, full model train/resume and strict inference reload. See the
[acceptance record](RELEASE_ACCEPTANCE.md) for measured integration scope.
