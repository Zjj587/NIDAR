# NIDAR: RGB-Conditioned LiDAR Intensity Synthesis

NIDAR synthesizes dense LiDAR intensity-like observations from RGB appearance
and simulator or dataset geometry. The pipeline combines spectral transfer,
intrinsic decomposition, point-cloud projection, and distribution calibration so
that simulated point clouds can carry an intensity channel without per-scene
network fitting in the evaluated settings.

This repository is currently organized as a release candidate for the IROS 2026
accepted NIDAR project. Public upload still requires owner review of the
license, pretrained-weight distribution, citation metadata, and third-party
component notices.

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
inference/    Dataset and simulator inference pipelines.
training/     IRNet and STN training utilities.
ros/          ROS offline and online integration utilities.
benchmark/    Runtime and metric benchmarking scripts.
tools/        Data export, preparation, visualization, and evaluation helpers.
docs/         Dataset-specific guides and release notes.
V0/           P0 diagnostic reports, figures, and experiment harnesses.
```

## Environment

Create a Python environment and install the core dependencies:

```bash
conda create -n nidar python=3.10
conda activate nidar
pip install -r requirements.txt
```

Optional routes need extra system components:

- Waymo evaluation needs the Waymo Open Dataset package matching your TensorFlow
  version.
- ROS tools need a ROS 1 Python environment with `rospy`, `rosbag`,
  `sensor_msgs`, and `std_msgs`.
- The legacy IIW-CRF route uses the Cython/C++ extension under
  `inference/krahenbuhl2013`.

## Weights And Data

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

## Quick Start: Waymo-Style Export

Prepare a Waymo-style export with image, point-cloud, and calibration files, then
copy and edit the public template:

```bash
cd "$PROJECT_ROOT"
cp inference/configs/waymo_public_template.yaml inference/configs/my_waymo.yaml
```

Run the paper-aligned `R+remap` route:

```bash
python inference/run_waymo_pipeline.py \
  inference/configs/my_waymo.yaml \
  --stages 1,2,3,3.5,4 \
  --use-deep \
  --keep-remap-with-deep
```

For backward-compatible direct deep output without remapping:

```bash
python inference/run_waymo_pipeline.py \
  inference/configs/my_waymo.yaml \
  --stages 1,2,3,4 \
  --use-deep
```

## Other Pipelines

nuScenes, Isaac Sim, and UE5 entry points are available under `inference/`:

```bash
python inference/run_nuscenes_pipeline.py /path/to/clean_nuscenes_config.yaml --use-deep
python inference/run_isaacsim_pipeline.py /path/to/clean_isaacsim_config.yaml --use-deep
python inference/run_ue5_pipeline.py /path/to/clean_ue5_config.yaml --use-deep
```

The development tree contains local example configs for these routes. Before
public execution, copy them to cleaned configs and replace dataset, checkpoint,
output, and calibration paths with portable paths.

## P0 Diagnostic Artifacts

`V0/` contains the controlled pseudo-NIR-versus-RGB diagnostic materials used
for the arXiv revision:

- `V0/P0_FULL_INTRINSIC_CONVERGENCE_REPORT.md`
- `V0/P0_QUALITATIVE_INTENSITY_COMPARISON_REPORT.md`
- `V0/p0_qualitative_intensity_comparison/`
- `V0/p0_visual_comparison/`
- `V0/scripts/p0_matched_irnet_input_ablation.py`
- `V0/scripts/p0_branch_output_control.py`

The P0 results support the main design conclusion: RGB and pseudo-NIR can reach
similar sparse intensity validation loss, while pseudo-NIR is stronger through
the reflectance-calibration `R+remap` route.

## Training

IRNet training entry point:

```bash
python training/train_deep_intrinsic_v3.py \
  --config /path/to/clean_training_config.yaml
```

The provided training configs contain local evidence paths from the development
machine. Before public training, copy them to a new config and replace dataset,
checkpoint, output, and mask paths with portable paths under `DATA_ROOT`,
`OUTPUT_ROOT`, and `WEIGHTS_ROOT`.

## Release Notes

Before public upload, review:

- `docs/OPEN_SOURCE_RELEASE_CHECKLIST.md`
- third-party licenses for STN, IIW-CRF/DenseCRF, Waymo, nuScenes, ROS, and
  simulator integrations;
- whether generated artifacts such as `__pycache__`, `.pyc`, `.so`, large
  checkpoints, raw datasets, and local command logs are excluded from the public
  package.

## Citation

Citation metadata is pending final arXiv/publication information.

```bibtex
@inproceedings{nidar2026,
  title     = {NIDAR: RGB-Conditioned LiDAR Intensity Synthesis},
  author    = {Zhang Junjie and collaborators},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year      = {2026}
}
```

## License

License pending owner confirmation. Do not redistribute as a public release
until the license file and third-party notices are finalized.
