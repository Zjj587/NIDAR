# IRNet Training

The default **`historical-dense-v1`** recipe restores the retained U-Net + DeepLab
training method: pseudo-NIR inputs, densified LiDAR-derived labels, intrinsic
reconstruction, multi-scale intensity supervision, warmup and AdamW/cosine
scheduling. The earlier sparse-label recipe remains an optional baseline.

Start with [training data and supervision](TRAINING_DATA.md). The
[pretrained demo](../README.md#quick-start) does not require training.

## Historical Training And Current Recipe

This is a **verified restoration of the retained method, not a bitwise replay
of the hosted checkpoint's entire training run**. The released checkpoint has
no embedded configuration. A separate companion checkpoint supplies saved
arguments, and the original 10-epoch TensorBoard history corroborates the
effective loss weights and cosine schedule. That companion checkpoint has
different model tensors and does not replace the published weights.

### Recovered Configuration Reference

| Setting | Default restored recipe |
| --- | --- |
| Architecture | U-Net + DeepLab, base channels 32, output stride 16; DeepLab receives R only |
| Image size | 384 x 960 for IRNet; frozen STN uses 384 x 1248 internally |
| Schedule | 10 epochs, batch 8, learning rate 1e-4, weight decay 1e-5 |
| Optimizer | AdamW; cosine decay to 1e-6 over the configured 10 epochs |
| Reconstruction | L1 between R*S and pseudo-NIR, weight 1.0 |
| Intrinsic regularization | R bilateral smoothness 0.01, color sigma 0.01; R mean proxy 0.01, target 0.5 |
| Intensity objective | Multi-scale L1 + 0.5 L2 + 0.2 Gaussian-window SSIM loss |
| L1 scales | 1.0, 0.5, 0.25, averaged; L2 and SSIM at full prepared resolution |
| Effective intensity weight | 0 in epochs 1 and 2; 1 in epochs 3 through 10 |
| Disabled terms | S smoothness/TV, VGG, region, color-matching and distribution losses |
| Training batches | Drop incomplete final batch; clip gradient norm to 1.0 |
| Augmentation | Paired horizontal flip; input-only contrast/brightness jitter |

The saved YAML's `lambda_lidar: 0.5` is for an alternative loss branch. The
active improved-loss branch uses `lambda_intensity: 1.0`. The original zero-based
warmup/ramp implementation gives weights **[0, 0, 1, ...]**, despite settings named
one warmup epoch and one ramp-up epoch. The restoration preserves this behavior.

**Checkpoint selection caveat:** the original companion `best` checkpoint is
at zero-based epoch 1, before intensity supervision is active. Minimum full
validation loss across different warmup phases is not a fair comparison of
intensity quality. The public hosted weight is the final, zero-based epoch-9
checkpoint. Both final and minimum-total model-only outputs are saved below;
evaluate their deployment behavior separately.

### Verification And Limits

- Compared against retained source: exact model-state layout and four forward
  outputs, three target-generator fixtures, three loader fixtures, and loss
  components/gradients across warmup phases (tolerance 1e-6).
- Six original Waymo image/target pairs across three scenes: restored loading
  agrees pixel-for-pixel with the original loader at 384 x 960.
- Six original raw-cloud target regenerations: portable and retained generators
  agree exactly. Compared with stored historical PNGs, four agree exactly;
  the other two differ at 136 and 10 of 2,457,600 pixels. The source of those
  small historical differences is not established; byte-identical replay of
  every historical label is not claimed.
- One RTX 3090 FP32 training step at batch 8 and 384 x 960 passed with finite
  gradients in both networks. Peak tensor allocation was 10.00 GiB; allocator
  reservation was 15.78 GiB. This used random initialization and six real pairs
  repeated to eight, not a convergence experiment or a memory minimum.
- Real nuScenes mini: four training and two validation views, full released
  architecture initialized from public IRNet, three epochs at 64 x 192, with
  save/resume across warmup and strict inference reload. See
  [acceptance scope](RELEASE_ACCEPTANCE.md#dense-training-restoration-2026-10-18).

The paper reports 17 Waymo scenarios, 10 epochs, batch 8 and one RTX 3090.
It calls the optimizer Adam; the retained trainer uses **AdamW**. We preserve
and disclose that discrepancy instead of treating the paper and source as
interchangeable records. Exact original split indices, per-epoch sample order,
training-time source revision and the byte-identical historical STN preparation
route remain unestablished. Current files or counts do not prove historical
dataset membership. No full retraining or new R+remap quality claim is made.

## Prepare Data

Use the [inference environment](../README.md#quick-start), including PyTorch
2.6 or newer for restricted checkpoint loading. Obtain data under its provider's
terms and use the [official converters](DATA_ACCESS.md). No raw or prepared
Waymo/nuScenes training samples are redistributed.

For a small engineering check, export two **different scenes** in the separate
nuScenes export environment:

```bash
python tools/data_export/prepare_nuscenes.py --dataroot ./data/nuscenes \
  --version v1.0-mini --scene scene-0061 --frame-count 2 --output-root ./data/scene_train
python tools/data_export/prepare_nuscenes.py --dataroot ./data/nuscenes \
  --version v1.0-mini --scene scene-0103 --frame-count 2 --output-root ./data/scene_val
```

Set local exports in [sources.example.yaml](../training/sources.example.yaml).
Paths are relative to that YAML. Each sample is one camera view; all frames in
an export are used and `cameras` selects views. The default scene-disjoint policy
rejects reuse of an upstream scene across train/val/test, even through aliases.
For real training, use adequate training scenes and independent validation/test
scenes. Two mini scenes are not a benchmark. Waymo segments use the same interface.

Back in the NIDAR environment, from the repository root:

```bash
python -m training.prepare --config training/sources.example.yaml \
  --output ./data/irnet_pairs --device cuda
```

The frozen STN produces uint8 pseudo-NIR, normalized once to [0, 1]. The dense
target is projected at the original camera resolution, per-image percentile
normalized, dilated and Gaussian-smoothed, then resized/cropped with the input
as in the retained loader. The crop mask is **training support, not measured
LiDAR coverage**: zero-valued pixels within the crop also supervise the head.
See [exact data semantics](TRAINING_DATA.md#historical-densified-targets).

The original trainer instead used a sample-random 90/10 split. To explicitly
replicate that policy, use
[sources.legacy.example.yaml](../training/sources.legacy.example.yaml):

```bash
python -m training.prepare --config training/sources.legacy.example.yaml \
  --output ./data/irnet_pairs_legacy --device cuda
```

`sample-random-legacy` requires pooled sources, fixes a new `split_seed`, and
records each sample's membership. It does **not** recover the unknown original
indices or measure scene-held-out generalization. Never use it as evidence of
generalization to unseen scenes. The scene-disjoint default is an intentional
evaluation-policy difference, not an undocumented claim of identical splitting.

## Train And Resume

```bash
python -m training.train --data ./data/irnet_pairs/manifest.json \
  --output ./runs/irnet --device cuda
```

[config.yaml](../training/config.yaml) contains the recovered settings above.
At least eight training views are needed for its complete batch. For a bounded
CPU smoke, set preparation `image_shape: [64, 192]`, set training `batch_size: 2`,
and run with `--device cpu --epochs 3`. Three epochs exercise the transition to
active intensity supervision; one epoch does not. Batch size must be at least
two for the DeepLab global-pooling BatchNorm branch. The dense recipe drops
incomplete final training batches; validation retains all samples.

Optional initialization from published IRNet, with a new optimizer:

```bash
python -m training.train --data ./data/irnet_pairs/manifest.json \
  --output ./runs/irnet_finetune --device cuda \
  --init-checkpoint ./weights/irnet/epoch_010.pth
```

Resume at an epoch boundary:

```bash
python -m training.train --data ./data/irnet_pairs/manifest.json \
  --output ./runs/irnet --device cuda --resume ./runs/irnet/last.pth --epochs 10
```

Resume restores model, optimizer, scheduler and Python/NumPy/PyTorch RNG states.
`--epochs` is the total stop count, not additional epochs. The cosine horizon
remains frozen to the configuration's `training.epochs`; resuming cannot extend
it. Config/data changes are rejected. A failed partial epoch is rerun from the
last completed checkpoint. Resume into a new directory if history/checkpoint
disagree; best selection then restarts within that continuation. Seeded batch
ordering is reproducible for a new run, not the unknown historical order.
Bitwise equivalence across platforms or CUDA environments is not asserted.

## Objective And Outputs

For pseudo-NIR input `X`, the model returns `R, S, I_hat, R*S`. Reconstruction
targets `X`, not intensity GT. Intensity supervision acts on `I_hat` and updates
both DeepLab and U-Net through `R`; there is no measured reflectance target.
The regularizer is a **mean** proxy, not a median. Frozen STN is not trained here.
The active recipe does not require torchvision or perceptual-weight downloads.

Each epoch saves:

- `history.csv`: component/total losses, effective intensity weight, learning
  rates, masked MAE/RMSE, sample counts and time. Dense losses are averaged over
  batches as in the retained trainer; MAE/RMSE are aggregated over training-support
  pixels. Multi-scale L1 and full-resolution MAE need not match.
- `last.pth`: resumable final epoch; `best.pth`: minimum full validation total.
- `last_inference.pth` and `best_inference.pth`: corresponding model-only weights.
- `model_config.yaml` and `run_from_epoch_*.json`: model, loss, seed and data identity.
- `previews/epoch_*.png`: RGB, pseudo-NIR, R, S, R*S, derived target, I_hat and error.
  S has a separate display maximum; other grayscale panels use [0, 1].

These diagnostics assess the **head and intrinsic objectives**, not the final
point-cloud route. Deployment remains **R + remap**. After retraining, separately
validate R, projected intensity and remap using held-out data; never fit remap to
the current evaluation frame's GT. Passing a training smoke does not establish
convergence or preservation of released quality.

Use a model-only output with `DeepIntrinsicWrapper` and the saved model kwargs,
or set `stage2_deep.checkpoint_path` and `stage2_deep.model_config_path` in the
pipeline YAML. The one-command demo still defaults to published weights; it
does not automatically select a new checkpoint or calibration.

## Optional Sparse Baseline

Set `recipe: sparse-v1` in a separate sources YAML, prepare into a new directory,
then explicitly select [config_sparse.yaml](../training/config_sparse.yaml):

```bash
python -m training.train --config training/config_sparse.yaml \
  --data ./data/irnet_pairs_sparse/manifest.json --output ./runs/irnet_sparse --device cuda
```

This preserves the earlier nearest-camera-depth sparse projection, explicit
measurement mask, fixed intensity scale, masked L1 and constant-rate Adam
baseline. It has no historical warmup/multi-scale objective. Data and training
recipe names must agree; sparse labels cannot silently enter dense training.
Old manifests/configurations without a recipe field retain sparse semantics.
