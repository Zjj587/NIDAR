# IRNet Training Data

[Training commands](TRAINING.md#prepare-data) |
[Historical recipe](TRAINING.md#historical-training-and-current-recipe) |
[Official data access](DATA_ACCESS.md)

IRNet takes a **single-channel pseudo-NIR image**. Its LiDAR intensity target
comes from the corresponding point cloud projected into that camera. A sample
is one camera view of one frame, not an entire scene or an individual point.
STN remains frozen during preparation and IRNet training.

## Inputs And Supervision

| Signal | Source | Role |
| --- | --- | --- |
| RGB | Calibrated camera image | Input to frozen STN |
| Pseudo-NIR X | STN output | IRNet input and reconstruction target |
| XYZ and calibration | LiDAR and camera metadata | Align intensity with images |
| Intensity Y | Projected, transformed sensor returns | Supervise auxiliary head I_hat |
| R and S | U-Net outputs | Intrinsic factors; R*S reconstructs X |

DeepLab receives `R`, without `S`. Its loss updates DeepLab and U-Net through
`R`. There is no measured reflectance target. `R*S` is **not** trained to
reconstruct LiDAR intensity. XYZ/depth is used for projection, not supplied as
an extra image channel. Deployment projects **R + remap**, not I_hat by default.

## Historical Densified Targets

The default public preparation now implements **`historical-dense-v1`**:

1. Project returns at the original camera image resolution.
2. Resolve collisions in the retained input-X ordering; see the caveat below.
3. Normalize occupied intensity values using each image's 1st/99th percentiles.
4. Apply 5 x 5 elliptical dilation and Gaussian smoothing with sigma 2.0;
   divide by the correspondingly smoothed support and convert to uint8.
5. Resize paired pseudo-NIR and target to 384 x 960 using PIL bilinear sampling.
6. Find rows with target > 0.02, crop the occupied row interval, expanding its
   top upward toward half the prepared image height when needed. Resize both
   cropped images back to the configured resolution, preserving the original
   uint8 round trip.
7. Use an **all-true** mask over that crop, including zeros. Inputs without any
   usable target are rejected, rather than silently trained without supervision.

These are **densified labels derived from sparse measurements**, not independently
measured dense GT or material reflectance. Original loading supervises even
zero-valued regions within the crop; `valid_mask` does not mean measurement
coverage for this recipe. Inspect the saved `target_nonzero_pixels` and preview
before running a large training job.

**Compatibility caveat:** the retained generator sorts `valid_points[:, 0]`,
but its projector returns input coordinates, not camera coordinates. Thus the
historical collision ordering is not generally a camera-depth z-buffer for
world-coordinate inputs. Compatibility mode preserves it explicitly. A corrected
visibility model would be a new recipe requiring separate evaluation.

Per-image percentile normalization is a **label transformation**, separate from
deployment-time quantile remapping. Never normalize predictions against an
evaluation frame's GT. The frozen public STN wrapper normalizes its uint8 output
once; exact reproduction of the historical pseudo-NIR image bytes is not claimed.

**Known label limitation:** zero-valued labels can arise from missing support,
percentile clipping, or small positive values lost in uint8 conversion/resizing.
`target > 0` is therefore not a reliable measurement-validity mask. The frozen
v0.1.0 recipe preserves the historical behavior; mask, precision and normalization
changes require separate controlled evaluation. Their contribution to R blur or
downstream task performance has not been established.

## Current Public Data Format

Use the [official converters](DATA_ACCESS.md). Each export must contain RGB,
calibration, world-coordinate XYZ/intensity arrays and a complete
`input_manifest.json`. Set scenes/splits in
[sources.example.yaml](../training/sources.example.yaml), then run:

```bash
python -m training.prepare --config training/sources.example.yaml \
  --output ./data/irnet_pairs --device cuda
```

The result contains `manifest.json` and `samples/*.npz`:

| Array | Shape / type | Meaning |
| --- | --- | --- |
| input | H x W, float32 | Transformed pseudo-NIR in [0, 1] |
| target | H x W, float32 | Transformed LiDAR-derived label in [0, 1] |
| valid_mask | H x W, bool | Recipe-specific supervision support |
| rgb | H x W x 3, uint8 | Aligned preview, not another IRNet input |

The manifest records recipe, split policy, upstream scene/frame/camera identity,
each sample's split/checksum, source manifest checksum, STN checksum, resolution
and exposure ratio. Dense records also give original projected-pixel count,
crop rows and nonzero target count. `intensity_scale` is unused and clipped count
is null in dense mode because the historical normalization is per image.
This is a local provenance record, not permission to redistribute the data.

For `sparse-v1`, the mask instead marks retained sensor returns, including true
zeros. The closest camera-depth point per output pixel is retained, no label
dilation is applied, and targets are `clip(exported_intensity / intensity_scale,
0, 1)`. nuScenes exports are already divided by 255; Waymo exports retain native
intensity. Choose a fixed scale from training data only and inspect clipped counts.
Missing sparse coverage appears blue-gray in previews, not as zero-valued GT.

## Split And Quality Checks

- The recommended `scene-disjoint` policy keeps adjacent frames and cameras from
  one upstream scene together and rejects cross-split aliases and duplicates.
- Explicit `sample-random-legacy` compatibility reproduces the old 90/10 splitting
  policy with a newly frozen seed, not its unknown historical membership. See
  [training commands](TRAINING.md#prepare-data); this is not held-out-scene evidence.
- Keep test scenes separate from training, model selection and remap calibration.
  More camera views are not more independent scenes.
- Check target coverage and per-epoch RGB/pseudo-NIR/target/R/S/R*S/I_hat/error
  panels. Cropping and dense-label zeros can strongly affect what is optimized.
- Two-scene mini smokes test mechanics, not generalization or convergence.

Data remain subject to provider terms. Raw/prepared Waymo and nuScenes samples,
complete historical splits and STN retraining are not distributed here.
