# CPU Release Acceptance

Checks recorded from 2026-10-15 through 2026-10-17 using the published
checkpoints and matching source. The initial owner-export check is followed
by dated official-raw-data checks below. These are engineering smoke tests,
not new generalization benchmarks or a reproduction of all paper experiments.

## Environment And Inputs

- ThinkPad T14p Gen 2, Intel Core Ultra 9 185H, approximately 32 GB RAM; CPU only.
- Linux x86_64, Python 3.10.12, PyTorch 2.8.0+cpu, NumPy 2.2.6.
- Four threads for OpenMP, MKL, and OpenBLAS. See `constraints-cpu-tested.txt`.
- Two exported Waymo frames, `scene_ws4/frame_000000` and `frame_000001`:
  10 RGB images, 10 camera JSONs, two point clouds, and two LiDAR JSONs.
- Portable raw input: 24 regular files, 5,690,362 bytes. These owner-prepared
  exports are not distributed; use the [official-data guide](DATA_ACCESS.md)
  for reproducible inputs. These export labels alone are not an extraction recipe.
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

## Official-Data Integration Check, 2026-10-16

The new Waymo TFRecord and nuScenes keyframe converters share an explicit
camera/cloud contract, source manifests, and dataset-specific remap selection.
This follow-up used the same ThinkPad CPU inference environment:

| Check | Result |
| --- | --- |
| Complete supported-environment suite | 68 tests passed, none skipped |
| New exporter tests | 19 synthetic geometry/contract tests included in the suite |
| Existing owner Waymo sample input check | Two frames and 10 camera images passed |
| Existing two-frame Waymo full inference | Passed in 27.48 s; peak process RSS 1,410,900 KiB |
| Compatibility with previous public-source run | Both prediction arrays bitwise equal |
| Output checks | Finite [0, 1] intensities, complete point clouds and nonblank comparison images |

Synthetic tests cover nuScenes camera optical axes and asynchronous poses,
Waymo SDK-result handling and per-frame evaluation calibration, incomplete
exports, frame/camera manifests, and source-package entry points. They do not
exercise the real dataset SDKs on official raw input. No fresh official data
was downloaded, no converter environment was clean-installed, and no real
nuScenes inference was run for this check. The Waymo regression reused the
previous authorized export; it does not validate the new TFRecord converter.
The GUI was not retested in this headless follow-up.

## Official Waymo Raw Fixture, 2026-10-16

This later check used the official SDK's `two_frame.tfrecord`, not the
previous owner-prepared export. The [download guide](DATA_ACCESS.md#waymo)
records the upstream commit, URL, byte count and SHA-256. The source context is
`10485926982439064520_4980_000_5000_000`, with timestamps
`1543251275738460` and `1543251275838479` microseconds.

| Check | Result |
| --- | --- |
| Separate Waymo converter environment | Python 3.10, SDK 1.6.7, TensorFlow 2.13.0, NumPy 1.23.5, protobuf 3.20.3; `pip check` passed |
| Exporter suite in that environment | 20 passed, including native SDK decoding of a serialized range image |
| Raw conversion | Two frames, 10 rectified images, TOP first-return clouds; 3.83 s |
| Full CPU `R + remap` demo | Passed in 29.79 s; peak process RSS 1,448,564 KiB |
| Output point counts | 148,631 and 148,349; XYZ bitwise preserved |
| Intensity and images | Finite [0, 1] predictions; both comparisons nonblank; first comparison and cloud preview visually inspected |
| Evaluation MAE, frames 0 / 1 | 0.108672 / 0.109163 |
| Evaluation RMSE, frames 0 / 1 | 0.146674 / 0.147541 |
| Evaluation SSIM, frames 0 / 1 | 0.500093 / 0.508696 |

The clean install exposed two real compatibility issues: the SDK's pinned old
JAX wheel is now in JAX's official archive, and its protobuf parsing fails
with protobuf 4. Both fixes are in `requirements-waymo-export.txt`. A PyPI
mirror was used for slow downloads; the TensorFlow wheel's SHA-256 was checked
against PyPI's official metadata. No installed SDK source was patched.

The native test checks the SDK, not just mocked SDK outputs. These two raw
frames still do not establish full-segment support, broad generalization or
new paper results. Timings exclude downloads/installations and the GUI.

## Official nuScenes Mini Two-Frame Check, 2026-10-16

The [official tutorial's mini archive](https://www.nuscenes.org/data/v1.0-mini.tgz)
was downloaded directly from its provider, then converted with the released
`prepare_nuscenes.py`. The archive's 4,167,696,325 bytes, multipart ETag and
gzip integrity passed validation. Its measured SHA-256 is
`943037abbb3b26b3070dc76504a43eb440503b00baf9ac2f1538d9c03fc9298f`.
This checksum records the retrieved bytes; it is not a provider-published hash.
The [data guide](DATA_ACCESS.md#nuscenes) contains the reproduction commands.

The tested input is `v1.0-mini`, `scene-0061`, first two keyframes, with six
cameras and one LIDAR_TOP sweep per frame. Sample tokens are
`ca9a282c9e77460f8360f564131a8af5` and `39586f9d59004284a7114a68825e8eec`.
Automatic checks finished on 2026-10-16 at 22:04 China Standard Time;
saved-image inspection completed on 2026-10-17.

| Check | Result |
| --- | --- |
| Separate converter environment | Python 3.10, nuScenes devkit 1.2.0, NumPy 1.26.4, OpenCV 4.11.0; `pip check` and SDK imports passed |
| Official raw conversion and input preflight | Two keyframes and 12 camera images passed |
| Full CPU `R + remap` demo | Published STN and full U-Net + DeepLab IRNet, bundled nuScenes calibration; 28.31 s |
| Peak process RSS from GNU time | 1,365,772 KiB, approximately 1.30 GiB |
| Output point counts, frames 0 / 1 | 34,688 / 34,720; XYZ bitwise preserved |
| Predicted intensity | Finite in [0, 1]; positive-point fractions 0.580431 / 0.588191 |
| Independent native SDK projection check | Maximum discrepancy 0.087658 pixels over 12 camera/frame combinations, for tested points at camera depth greater than 1 m |
| Saved-image inspection | Both GT/prediction comparisons, both six-camera LiDAR overlays and frame 0 point-cloud preview inspected; nonblank, no obvious wholesale projection mismatch |

CPU hardware, inference environment and four-thread limit are the same as
above. Timing includes model loading, inference, projection, remapping,
evaluation and output generation, but excludes download, installation, raw
conversion and separate geometry QA. RSS is process memory, not total host RAM.
The SDK check verifies coordinate-transform compatibility, not physical
calibration accuracy or moving-object synchronization. The GUI was not retested.

| Frame | MAE | RMSE | SSIM | Valid-pixel fraction |
| --- | ---: | ---: | ---: | ---: |
| 000000 | 0.056886 | 0.088438 | 0.896455 | 0.276062 |
| 000001 | 0.057005 | 0.088483 | 0.893412 | 0.280350 |

These are reference outputs of the current evaluator, not accuracy acceptance
thresholds. The comparisons are sparse and mostly low intensity. Valid-pixel
fraction is an evaluator image statistic, not the positive-point fraction.
Do not compare these numbers with the Waymo checks as evidence of superior
model quality: sensor, scenes, intensity conventions and pixel coverage differ.
No LPIPS, full-split benchmark, CUDA result, temporal-consistency result or
robotic-task improvement is claimed. Provider data and the new output images
are not redistributed by NIDAR.

## Remaining Release Work

- Extend [official extraction acceptance](DATA_ACCESS.md) beyond the small
  Waymo fixture and two nuScenes mini keyframes to broader coverage.
- Finalize artifact-specific data/weight distribution notices.
- Validate additional platforms and CUDA independently; this test covers CPU only.
- Keep public code, model metadata and subsequent tagged releases aligned.
