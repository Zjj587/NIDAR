# IRNet Training Data

[Training commands](TRAINING.md#prepare-data) |
[Historical recipe](TRAINING.md#historical-training-and-current-recipe) |
[Official data access](DATA_ACCESS.md)

IRNet takes a **single-channel pseudo-NIR image**. Its LiDAR intensity target
comes from the corresponding point cloud projected into that camera. A sample
is one camera view of one frame, not an entire scene or an individual point.
STN is frozen during this preparation and IRNet training.

## Inputs And Supervision

| Signal | Source | Role |
| --- | --- | --- |
| RGB | Calibrated camera image | Input to the frozen STN |
| Pseudo-NIR `X` | STN output | IRNet input and reconstruction target |
| XYZ and calibration | LiDAR and camera metadata | Align intensity measurements with the image |
| LiDAR intensity `Y` | Projected sensor returns | Supervise the auxiliary intensity head `I_hat` |
| Reflectance `R` and shading `S` | U-Net outputs | Intrinsic factors; their product reconstructs `X` |

For the released model configuration, DeepLab receives `R`, without `S`.
The intensity loss updates both DeepLab and the U-Net through `R`. There is
no measured reflectance target, and `R*S` is **not** supervised to reconstruct
the LiDAR intensity map. XYZ/depth is used for projection, not fed as an
additional image channel to this IRNet.

Training and deployment have different endpoints:

- **Training:** reconstruct pseudo-NIR with `R*S`, and predict the aligned
  LiDAR-derived target with `I_hat`, together with intrinsic regularization.
- **Deployment:** project `R` onto the point cloud, then apply the fixed
  dataset-specific quantile remapping. `I_hat` is not the default final output.

## Historical Densified Targets

The historical preparation pairs pseudo-NIR with grayscale intensity images:

1. Project LiDAR returns into each calibrated camera view.
2. Normalize covered intensity values using that image's 1st/99th percentiles.
3. Expand projected support using a 5 x 5 elliptical dilation.
4. Apply Gaussian smoothing with sigma 2.0 and normalize by smoothed support.
5. Save an 8-bit grayscale target and pair it with the matching pseudo-NIR view.

These are **densified targets derived from sparse measurements**, not dense
independent sensor ground truth or ground-truth material reflectance.
Uncovered image regions do not become measured ground truth through smoothing.
Historical data loading also resizes/crops paired images; that preprocessing
must be considered when comparing it with a direct sparse projection.

Per-image label normalization is a **training-data transformation**. It is not
the deployment quantile remapping, and it must not be used to fit a prediction
to an evaluation frame's ground truth. The current public preparation command
does not implement this historical densification path. See the
[recipe comparison](TRAINING.md#historical-training-and-current-recipe).

## Current Public Data Format

Use the [official converters](DATA_ACCESS.md) first. Each exported scene must
contain RGB images, calibration, world-coordinate XYZ/intensity arrays and a
completed `input_manifest.json`. List distinct scenes in
[sources.example.yaml](../training/sources.example.yaml), assigning each whole
scene to `train`, `val` or `test`.

```bash
python -m training.prepare --config training/sources.example.yaml \
  --output ./data/irnet_pairs --device cuda
```

The prepared directory contains `manifest.json` and `samples/*.npz`:

| Array | Shape / type | Meaning |
| --- | --- | --- |
| `input` | H x W, float32 | Pseudo-NIR in [0, 1] |
| `target` | H x W, float32 | Sparse projected intensity in [0, 1] |
| `valid_mask` | H x W, bool | Pixels with retained LiDAR returns, including valid zeros |
| `rgb` | H x W x 3, uint8 | RGB preview, not an extra IRNet input |

The manifest records upstream scene/frame/camera identity, split, sample and
source-manifest checksums, STN checksum, resolution, exposure ratio, scale,
valid-pixel counts and clipped-pixel counts. It is a local reproducibility
record, not permission to redistribute the underlying data.

Current preparation retains the nearest camera-depth return at each output
pixel and uses `clip(exported_intensity / intensity_scale, 0, 1)` without
dilation or per-image percentile normalization. Missing coverage is masked;
a valid zero return is not discarded. Use a fixed scale selected from
training data only. nuScenes exports are already divided by 255; Waymo
exports retain native intensity. See the [scale notes](TRAINING.md#prepare-data).

## Split And Quality Checks

- Keep adjacent frames and different cameras from one scene in the same split.
  The preparer rejects cross-split reuse of an upstream scene, even when its
  local directories have different names.
- Hold out a test split independently of training and checkpoint selection.
  More camera views are more samples, not more independent scenes.
- Inspect valid/clipped counts and the per-epoch RGB, pseudo-NIR, target,
  `R`, `S`, reconstruction, `I_hat` and error panels before long training.
  Missing sparse targets are shown in blue-gray rather than as black GT.
- A two-scene mini-data smoke checks mechanics, not generalization. Published
  weights, a new training run and its deployment remap need separate evaluation.

Download data under the provider's terms. NIDAR does not redistribute raw or
prepared Waymo/nuScenes training samples. Local preparation is supported;
STN retraining and an exact historical dense-training reproduction package
are not currently included.
