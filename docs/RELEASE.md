# Frozen Baseline v0.1.0

Released baseline date: 2026-10-18. Use the `v0.1.0` tag for a fixed source
snapshot; `main` may receive later development. This release freezes the verified
workflow, not a claim that the models or training labels need no improvement.

```bash
git clone --branch v0.1.0 https://github.com/Zjj587/NIDAR.git
cd NIDAR
```

Continue with the [quick start](../README.md#quick-start). Git may report a
detached HEAD when checking out a tag; this is expected for a fixed snapshot.

## What Is Frozen

- Inference, converters, training code, configuration and dependencies are
  unchanged from source commit `c59422faa0c34853559cec11de4d23f540211508`.
  This tag adds release documentation, instruction fixes and documentation tests.
- Inference: RGB -> frozen STN -> U-Net + DeepLab IRNet R -> projection ->
  dataset-specific quantile remapping. I_hat is not the default deployed output.
- Training: `historical-dense-v1`, with dense LiDAR-derived labels, R*S
  reconstruction, the recovered intensity objective and warmup/AdamW/cosine
  schedule. Scene-disjoint splitting is the public default; the legacy policy
  and sparse recipe are explicit alternatives.
- The downloader pins checkpoint revision
  `1390055c2c564d3276f29cc513b3ef3cdc355e40` on
  [Hugging Face](https://huggingface.co/zhangjunjie587/NIDAR/tree/1390055c2c564d3276f29cc513b3ef3cdc355e40).
  STN `47.pth`, full IRNet `epoch_010.pth`, and the bundled Waymo/nuScenes
  calibrations are unchanged. [Model identities](MODEL_ZOO.md#checkpoint-identity)
  give the verified hashes.
- The accompanying [model-card snapshot](https://huggingface.co/zhangjunjie587/NIDAR/tree/v0.1.0)
  contains artifact metadata and notices. Downloading files does not grant
  blanket commercial or redistribution rights.

## Verified Scope

Official Waymo two-frame input and two nuScenes mini keyframes passed conversion
and full CPU inference. Restored training passed source parity, bounded real-data
train/resume checks, and one full-size RTX 3090 update. See the
[acceptance record](RELEASE_ACCEPTANCE.md) for environments and limitations.

These checks are not a full-split benchmark, a complete retraining experiment,
or an exact replay of the hosted checkpoint's historical training run. Obtain
Waymo and nuScenes data [from their providers](DATA_ACCESS.md); no raw or prepared
training datasets are distributed here. Website simulator and BC videos are
research demonstrations, not packaged workflows in this release.

## Known Training Limitations

1. Intensity supervision is inactive during the first two epochs. Minimum total
   loss across warmup phases is not an intensity-quality selection criterion.
   The hosted IRNet is the final checkpoint, not the early minimum-total one.
2. Historical dense labels include zero-valued crop regions in supervision.
   Zeros can also result from normalization and quantization, not only missing
   measurements. See [label semantics](TRAINING_DATA.md#historical-densified-targets).

This release does not silently change either behavior. Future corrections must
use a new named recipe and separately evaluated weights. No improvement in R
quality or BC success is claimed from identifying these issues alone.

## Next Work

The [public progress list](../README.md#release-progress) distinguishes available
workflows from planned BC packaging, IRNet improvements and simulator demos.
The tag will not be moved to substitute later weights or behavior.
