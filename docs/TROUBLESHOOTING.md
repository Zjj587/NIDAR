# Troubleshooting

## Check The Checkpoint And Architecture Together

The public default needs the 226996282-byte `epoch_010.pth` with a U-Net
intrinsic network and a DeepLab intensity head. Historical checkpoints share
that filename. Verify the file, not only its name:

```bash
python tools/download_weights.py --weights-root ./weights --check-only
```

The matching `inference/configs/deep_intrinsic_public_model.yaml` must select
`intensity_head_type: deeplab`. Missing or unexpected state-dictionary keys
usually indicate a code/config/checkpoint mismatch. Keep strict loading
enabled; check the [model zoo](MODEL_ZOO.md) instead of ignoring unmatched keys.

## Images Or Calibration Are Missing

`--input-root` must point to the raw directory containing `images/` and
`pointclouds/`. The default asks for two frames and the configured cameras.
Each selected image needs its camera JSON and each frame needs point-cloud
and LiDAR calibration files. Use `--frame-count 1` for one complete frame.

The portable two-frame sample is being prepared. An example directory name
does not mean that data has been downloaded. See the
[required layout](MODEL_ZOO.md#expected-demo-data-layout).

## A Package Or Display Is Missing

Install `requirements.txt` in the same environment used for the demo.
The import `skimage` comes from `scikit-image` and is needed for default image
evaluation. `--no-eval` skips evaluation and its outputs. Open3D is optional
through `requirements-viewer.txt`; LPIPS is optional through `requirements-eval.txt`.

On a headless server, use `--viewer none` and inspect PNG/PLY outputs. Use
`--viewer open3d` in a desktop session for the interactive window; close the
window to allow the command to finish.

On Linux Wayland desktops, the legacy Open3D viewer uses XWayland when
`DISPLAY` is available. Run from the desktop terminal for automatic display
access. A plain SSH session usually has no display; use `--viewer none`
there. Failed window creation is reported explicitly with `--viewer open3d`,
and already saved point clouds remain available.

## A Legacy Checkpoint Fails To Load

The restricted loader needs PyTorch 2.6 or newer. Use the recorded hashes and
matching source. Do not enable unrestricted pickle loading just to suppress
an exception. Include the error and PyTorch version in an issue if a verified
file still fails.

An error mentioning `numpy.core.multiarray.scalar` with NumPy 2 can indicate
an older NIDAR loader. The current loader explicitly maps this legacy scalar
name while keeping restricted loading enabled. Update the source, rather
than setting `weights_only=False`. For CPU installation, see the pinned
[release-check environment](DEMO.md#inputs).

## Predictions Contain Empty Or Inconsistent Regions

Inspect image coverage, camera/LiDAR calibration, and selected frame IDs.
The full range-image metric includes points outside valid image coverage.
Report the masked comparison with its coverage, alongside the full view.
Quantile remapping changes predicted values; it does not fill missing coverage.

The calibration targets the Waymo domain. Different sensors, exposure,
materials, and ranges can shift intensity distributions. Inspect raw and
remapped outputs and range/visibility groups when diagnosing a new domain.

## Outputs Already Exist

Choose a fresh output directory for each run. Avoid putting outputs inside
input or weight directories. The demo checks incomplete outputs and invalid
values so a partial run is not presented as a successful reproduction.

## Report A Problem

Include the Git commit, OS, Python/PyTorch/CUDA versions, checkpoint hashes,
exact command, input layout, and relevant log excerpt. Attach
`results/summary.json` if generated. State whether the issue concerns
inference, projection, evaluation, or the viewer. Review logs for credentials
and personal information before sharing them publicly.
