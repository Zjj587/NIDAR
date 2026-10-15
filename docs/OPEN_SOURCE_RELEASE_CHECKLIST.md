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
- [x] The included Waymo CLI provides `--keep-remap-with-deep` for the
  paper-aligned `R+remap` route while preserving the older default behavior.
- [x] `inference/run_nidar_demo.py` provides a single end-to-end demo command
  with compact final outputs and optional Open3D visualization.
- [x] `docs/DEMO.md` documents the end-to-end Waymo-style demo command, output
  tree, and qualitative comparison images.
- [x] `docs/MODEL_ZOO.md` documents the external checkpoint and sample-data
  artifact layout.
- [x] Default IRNet config selects U-Net + DeepLab; the model zoo distinguishes
  the full-paper checkpoint from the older small-head checkpoint.
- [x] The source-package manifest includes the demo entry point, DeepLab model
  implementation, license, artifact documentation, and release tests.
- [x] Open3D and LPIPS are separate optional dependencies; no dataset SDK is
  needed for the exported-data demo.
- [x] Public checkpoint metadata matches the recorded byte sizes and SHA-256.
- [x] Eight Hugging Face companion files were uploaded by the owner and
  anonymously read back with matching SHA-256.
- [x] README includes real research media, concise setup, and explicit demo
  status; citation, troubleshooting, and contribution files exist locally.
- [ ] Tag a coordinated source/model/sample release after public sample access
  and distribution terms are finalized.
- [x] A standard-library downloader pins the model upload revision, maps Hub
  root files into the runtime layout, and verifies downloaded bytes.
- [x] Complete real anonymous checkpoint downloads and the full CPU demo on ThinkPad.
- [x] Verify GT independence, strict public JSON, and the actual Open3D viewer
  entry; record results in `docs/RELEASE_ACCEPTANCE.md`.

## Items Requiring Owner Review Before Public Artifact Release

- [x] Source-code license file is present in the GitHub repository.
- [ ] Third-party notices for STN/cs-stereo, IIW-CRF/DenseCRF,
  Waymo Open Dataset, nuScenes, ROS, Isaac Sim, UE5, and any external baseline
  code.
- [ ] Public checkpoint distribution policy for STN and IRNet weights.
- [ ] Public sample data policy and expected dataset layout.
- [x] arXiv citation and project-page link aligned with the project website.
- [ ] Replace local owner-review paths in `docs/MODEL_ZOO.md` with public
  artifact links before broad external announcement, or keep them clearly
  marked as non-portable local evidence.

## Exclude From Public Source Packages

- `__pycache__/`
- `*.pyc`
- generated native extensions such as `*.so` unless explicitly documented as
  release binaries;
- local command logs with private paths;
- raw datasets, bags, point clouds, and training outputs;
- large checkpoints unless released through a reviewed artifact channel.

## Local Evidence Kept Under V0

P0 reports, qualitative figures, and diagnostic scripts are kept as research
artifacts outside this first source package. They may contain local evidence
paths; keep them out of the public quick start unless they are cleaned and moved
to supplementary artifacts.

## Minimal Verification Before Public Upload

Run from the repository root:

```bash
python3 -m py_compile \
  inference/run_nidar_demo.py \
  inference/run_waymo_pipeline.py \
  inference/src/utils/config.py \
  inference/src/models/deep_intrinsic_net.py \
  inference/src/models/network_variants.py \
  inference/src/models/stn_wrapper.py

python3 inference/run_waymo_pipeline.py --help
python3 inference/run_nidar_demo.py --help

python3 -m unittest discover -s tests -p 'test_public_release.py' -v
python3 -m unittest discover -s tests -p 'test_checkpoint.py' -v

python tools/release/make_source_package.py --out-dir /path/outside/repo/nidar_source_check
```

Then scan the resulting source folder for private absolute paths and credentials
using your release-audit tool of choice. Owner-only paths in `docs/MODEL_ZOO.md`
are intentional; executable configs must remain portable. Run the two-frame
demo using the checkpoint hashes above before calling an artifact release
verified. Unit tests alone do not validate the quality of synthesized intensity.
