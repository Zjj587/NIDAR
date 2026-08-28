# NIDAR: RGB-Conditioned LiDAR Intensity Synthesis

NIDAR synthesizes dense LiDAR intensity-like observations from RGB appearance
and simulator or dataset geometry. The pipeline combines spectral transfer,
intrinsic decomposition, point-cloud projection, and distribution calibration so
that simulated point clouds can carry an intensity channel without per-scene
network fitting in the evaluated settings.

This repository is the public source-code home for the IROS 2026 accepted NIDAR
project. The first public package focuses on Waymo-style inference and demo
reproduction. Pretrained weights, sample data, and third-party notices are
tracked separately from git.

## Pipeline

The paper-aligned inference route is:

```text
RGB image
  -> STN pseudo-NIR image
  -> IRNet reflectance-like output R
  -> LiDAR-camera projection
  -> quantile remapping
  -> point cloud with synthesized intensity
```

NIDAR outputs an intensity-like signal for simulation and evaluation. It should
not be described as sensor-independent reconstruction of absolute returned
power.

## Repository Layout

```text
inference/    Waymo-style inference pipeline and core models.
tools/        Release-package helper scripts.
docs/         Demo, artifact, and release notes.
```

Training code, ROS utilities, simulator-specific integrations, and diagnostic
experiment harnesses are not part of this first source package.

## Environment

Create a Python environment and install the core dependencies:

```bash
conda create -n nidar python=3.10
conda activate nidar
pip install -r requirements.txt
```

Optional routes need extra components:

- Creating your own Waymo export needs the Waymo Open Dataset package matching
  your TensorFlow version. The minimal demo uses already exported Waymo-style
  files.
- The legacy IIW-CRF route uses the Cython/C++ extension under
  `inference/krahenbuhl2013`.

## Model Zoo, Weights, And Data

Do not hardcode private machine paths in public configs. Use environment
variables and copy a template config:

```bash
export PROJECT_ROOT=/path/to/NIDAR
export DATA_ROOT=/path/to/datasets
export OUTPUT_ROOT=/path/to/nidar_outputs
export WEIGHTS_ROOT=/path/to/nidar_weights
```

Expected weight layout:

```text
$WEIGHTS_ROOT/
  stn/47.pth
  irnet/epoch_010.pth
  quantile/wintensity_mapping_quantile_model.pkl
```

Small quantile mapping files are present under `inference/quantile_model/`.
Large STN/IRNet checkpoints should be distributed separately after owner and
license review.

Artifact links and the demo-data layout are tracked in:

- `docs/MODEL_ZOO.md`
- `docs/DEMO.md`

## Quick Start: Minimal Waymo-Style Demo

Download or prepare the artifacts listed in `docs/MODEL_ZOO.md`, then copy and
edit the public template:

```bash
cd "$PROJECT_ROOT"
cp inference/configs/waymo_public_template.yaml inference/configs/my_waymo_demo.yaml
```

Run the paper-aligned `R+remap` route:

```bash
python inference/run_waymo_pipeline.py \
  inference/configs/my_waymo_demo.yaml \
  --stages 1,2,3,3.5,4 \
  --use-deep \
  --keep-remap-with-deep
```

The demo writes pseudo-NIR images, IRNet `R` images, pre-remap and remapped
point clouds, per-frame range-image comparison PNGs, and
`evaluation_summary.json` under `$OUTPUT_ROOT/waymo_demo`. See `docs/DEMO.md`
for the exact output tree and what to inspect visually.

For backward-compatible direct deep output without remapping:

```bash
python inference/run_waymo_pipeline.py \
  inference/configs/my_waymo_demo.yaml \
  --stages 1,2,3,4 \
  --use-deep
```

## Other Pipelines

nuScenes, Isaac Sim, UE5, ROS, and training utilities exist in the development
tree but are not included in this first public source package. They should be
released only after separate packaging, dependency, and license review.

## P0 Diagnostic Artifacts

The controlled pseudo-NIR-versus-RGB diagnostic materials used for the arXiv
revision are kept as research artifacts, not as part of this first public source
package. The P0 results support the main design conclusion: RGB and pseudo-NIR
can reach similar sparse intensity validation loss, while pseudo-NIR is stronger
through the reflectance-calibration `R+remap` route in the evaluated setup.

## Training

Training code is planned for a later release. This first package documents and
tests the inference route.

## Release Notes

Before publishing external artifacts, review:

- `docs/OPEN_SOURCE_RELEASE_CHECKLIST.md`
- third-party licenses for STN, IIW-CRF/DenseCRF, Waymo, nuScenes, ROS, and
  simulator integrations;
- whether generated artifacts such as `__pycache__`, `.pyc`, `.so`, large
  checkpoints, raw datasets, and local command logs are excluded from the public
  package.

## Citation

Please cite the IROS 2026 paper. Replace the placeholder fields below with the
final arXiv and proceedings metadata when available.

```bibtex
@inproceedings{nidar2026,
  title     = {NIDAR: RGB-Conditioned LiDAR Intensity Synthesis},
  author    = {Zhang, Junjie and collaborators},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year      = {2026}
}
```

## License

Source code in this repository is released under the Apache License 2.0. Model
weights, sample data, and third-party components may carry separate terms; check
their artifact pages before redistribution.
