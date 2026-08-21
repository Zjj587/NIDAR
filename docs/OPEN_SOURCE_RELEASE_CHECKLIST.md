# NIDAR Open-Source Release Checklist

This checklist separates public release requirements from local evidence on the
development machine.

## Public Entrypoints

- [x] Top-level `README.md` describes the pipeline, quick start, weights, and
  release status without requiring private absolute paths.
- [x] `requirements.txt` lists core Python dependencies.
- [x] `inference/configs/waymo_public_template.yaml` uses environment-variable
  placeholders instead of private absolute paths.
- [x] `inference/src/utils/config.py` expands `${VAR}` and `~` in YAML strings.
- [x] Waymo and nuScenes CLIs provide `--keep-remap-with-deep` for the
  paper-aligned `R+remap` route while preserving the older default behavior.

## Items Requiring Owner Review Before GitHub Upload

- [ ] Final repository license.
- [ ] Third-party notices for STN/cs-stereo, IIW-CRF/DenseCRF,
  Waymo Open Dataset, nuScenes, ROS, Isaac Sim, UE5, and any external baseline
  code.
- [ ] Public checkpoint distribution policy for STN and IRNet weights.
- [ ] Public sample data policy and expected dataset layout.
- [ ] Final arXiv citation and project-page link.

## Exclude From Public Source Packages

- `__pycache__/`
- `*.pyc`
- generated native extensions such as `*.so` unless explicitly documented as
  release binaries;
- local command logs with private paths;
- raw datasets, bags, point clouds, and training outputs;
- large checkpoints unless released through a reviewed artifact channel.

## Local Evidence Kept Under V0

`V0/` contains P0 reports, qualitative figures, and diagnostic scripts from the
development machine. These files may contain local evidence paths; keep them out
of the public quick start unless they are clearly marked as local evidence or
moved to supplementary artifacts.

## Minimal Verification Before Public Upload

Run from the repository root:

```bash
python3 -m py_compile \
  inference/run_waymo_pipeline.py \
  inference/run_nuscenes_pipeline.py \
  inference/src/utils/config.py \
  training/train_deep_intrinsic_v3.py

python3 inference/run_waymo_pipeline.py --help

python tools/release/make_source_package.py --out-dir /path/outside/repo/nidar_source_check
```

Then scan the resulting source folder for private absolute paths and credentials
using your release-audit tool of choice.
