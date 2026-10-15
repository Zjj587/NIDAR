# CPU Release Acceptance

Measured on 2026-10-15 using the published checkpoints and the accompanying
source package. This is an engineering smoke test, not a new generalization
benchmark or a reproduction of all paper experiments.

## Environment And Inputs

- ThinkPad T14p Gen 2, Intel Core Ultra 9 185H, approximately 32 GB RAM; CPU only.
- Linux x86_64, Python 3.10.12, PyTorch 2.8.0+cpu, NumPy 2.2.6.
- Four threads for OpenMP, MKL, and OpenBLAS. See `constraints-cpu-tested.txt`.
- Two exported Waymo frames, `scene_ws4/frame_000000` and `frame_000001`:
  10 RGB images, 10 camera JSONs, two point clouds, and two LiDAR JSONs.
- Portable raw input: 24 regular files, 5,690,362 bytes. Public sample access
  is still pending; these export labels alone are not an extraction recipe.
- Both full checkpoints were downloaded anonymously from pinned Hugging Face
  revision `1390055c2c564d3276f29cc513b3ef3cdc355e40`, with exact SHA256 checks.

## Verified Results

| Check | Result |
| --- | --- |
| Source, downloader, checkpoint and viewer regression tests | 49 passed |
| Strict loading and real image inference | STN and U-Net + DeepLab IRNet passed |
| Complete two-frame `R + remap` route | Passed; NPY, scalar/color PLY, previews and comparisons saved |
| Elapsed time | 27.36 s; earlier run 27.21 s |
| Peak process RSS from GNU time | 1,450,268 KiB, approximately 1.38 GiB |
| Output points | 162,996 and 163,058; finite intensity in [0, 1] |
| Geometry preservation | XYZ bitwise unchanged |
| Current-frame GT independence | Predictions bitwise unchanged after randomized GT / XYZ-only input |
| Public result JSON | Strict JSON; unavailable metrics are `null` |
| Open3D 0.19.0 desktop viewer | Window, nonblank rendering and programmed rotation passed through the demo entry |
| Identical image repeated twice | STN and all four IRNet output images bitwise equal |

Timing includes loading, neural inference, projection, remapping, evaluation,
and file/preview generation. It excludes installation, downloads, and the
interactive window. RSS is not total host memory usage. These two measurements
do not establish steady-state FPS or a GPU throughput claim.

The GT check used a separate input copy: frame 0 intensity values were replaced
with fixed-seed random values in [10, 100]; frame 1 supplied XYZ without an
intensity channel. Both were rerun with `--no-eval`. This tests inference
independence from current-frame GT, not independence from the training data
or the fixed quantile calibration.

Window creation and camera rotation were automated and captured, then the
window was closed. Human mouse-drag inspection is not claimed. Identical-input
repeatability does not establish temporal consistency across different frames.

## Reference Metrics

| Frame | MAE | RMSE | SSIM | Valid-pixel fraction |
| --- | ---: | ---: | ---: | ---: |
| 000000 | 0.097466 | 0.144186 | 0.576587 | 0.901651 |
| 000001 | 0.098425 | 0.145734 | 0.576224 | 0.901153 |

These are the current evaluator's outputs on two adjacent demo frames, not
evidence that this sample is representative or that NIDAR outperforms another
method. LPIPS was not computed. Inspect both full and masked comparisons.

## Remaining Release Work

- Provide a permitted public sample download or a complete extraction procedure.
- Finalize artifact-specific data/weight distribution notices.
- Validate additional platforms and CUDA independently; this test covers CPU only.
- Keep public code, model metadata and subsequent tagged releases aligned.
