# Official Data To Demo

Obtain Waymo and nuScenes data from their official providers under the
applicable terms. NIDAR does not mirror raw data or preprocessed copies, and
does not need your provider credentials. UE5 and Isaac Sim demos are deferred.

| Input | Local preparation | Inference route |
| --- | --- | --- |
| Waymo Perception Frame TFRecord | `prepare_waymo.py`, TOP LiDAR first return | STN -> IRNet R -> Waymo remap |
| nuScenes `v1.0-mini` | `prepare_nuscenes.py`, six cameras and LIDAR_TOP | STN -> IRNet R -> nuScenes remap |

**Validation boundary:** the official two-frame Waymo SDK TFRecord below has
passed raw conversion and complete CPU inference. Its converter environment
and 20 exporter tests, including native SDK/protobuf decoding, passed. This is
a small upstream fixture, not a full-segment or generalization benchmark.
Two keyframes from official nuScenes mini `scene-0061` also passed raw
conversion, full CPU inference, SDK projection checks and saved-image
inspection. This does not cover all mini scenes. See the
[Waymo record](RELEASE_ACCEPTANCE.md#official-waymo-raw-fixture-2026-10-16) and
[nuScenes record](RELEASE_ACCEPTANCE.md#official-nuscenes-mini-two-frame-check-2026-10-16).

Set up the inference environment and public weights using the
[quick start](../README.md#quick-start). Run each converter in its own
environment; do not install the Waymo TensorFlow SDK into the PyTorch inference
environment. Commands below assume the repository root as working directory.

## Waymo

### Download

For a small raw-format smoke test, the **official Waymo SDK repository**
provides a two-frame TFRecord. This is an upstream test fixture, not a complete
segment or a representative benchmark. Download it directly from its provider:

```bash
mkdir -p data/waymo_official
curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/v2/perception/compat_v1/testdata/two_frame.tfrecord \
  --output data/waymo_official/two_frame.tfrecord
```

Expected size: 9,773,046 bytes; SHA-256:
`e86c801470f1a25c5b6d57e832617ef4689bfc4d2cd9eff0979ae95a02665236`.
Use this filename in the conversion command below and set `--data-version`
to `official-sdk-99a4cb3`. We do not rehost this file.

For your own full Perception segment:

1. Open the [official portal](https://waymo.com/open/download/), sign in and
   accept the applicable [terms](https://waymo.com/open/terms/).
2. Select the **Perception Dataset in Frame TFRecord format** and download one
   segment. A full split is unnecessary for a two-frame demo. Record the release
   version and the original filename. Motion Dataset records and modular
   Parquet files are not supported by this converter.
3. Use the browser download or your authenticated Google Cloud CLI. Copy the
   exact authorized object URI from the provider; the placeholder below is not
   a public sample link:

```bash
mkdir -p data/waymo_official
export WAYMO_TFRECORD_URI='gs://YOUR_AUTHORIZED_BUCKET/PATH/SEGMENT.tfrecord'
gcloud storage cp "$WAYMO_TFRECORD_URI" data/waymo_official/
```

See the [official SDK](https://github.com/waymo-research/waymo-open-dataset) and
[Cloud Storage download instructions](https://docs.cloud.google.com/storage/docs/downloading-objects).
If your selected release only provides modular Parquet, do not pass it to this
converter or rename its extension.

### Prepare Two Frames

Use a separate Python 3.10 Linux x86_64 environment. This command uses the
official fixture downloaded above. For a full segment, replace its path and
set `--data-version` to your actual provider release, such as `v1.4.3`.

```bash
python3.10 -m venv .venv-waymo-export
.venv-waymo-export/bin/python -m pip install -r requirements-waymo-export.txt
.venv-waymo-export/bin/python tools/data_export/prepare_waymo.py \
  --tfrecord ./data/waymo_official/two_frame.tfrecord \
  --data-version official-sdk-99a4cb3 \
  --frames 0 1 \
  --output-root ./data/waymo_demo
```

The pinned SDK package is `waymo-open-dataset-tf-2-12-0==1.6.7`. Despite its
package name, that release declares TensorFlow 2.13 and NumPy 1.23.5. Let its
dependency metadata resolve the environment; do not force TensorFlow 2.12.
The requirements include JAX's official archive because old `jaxlib` wheels
are no longer on PyPI. They also pin protobuf 3.20.3: this legacy SDK's
`ParseFromString(bytearray(...))` fails with protobuf 4's parser. These fixes
were exercised on the raw fixture, not just an import check.
If `venv` reports missing `ensurepip`, use a separate
Conda Python 3.10 environment or install your OS's Python 3.10 venv support.

The exporter uses the official SDK's TOP-LiDAR motion compensation, preserves
native intensity for evaluation, rectifies camera distortion, and uses each
camera's recorded pose. Per-frame evaluation files retain the source beam
inclinations and range-image width; the legacy fixed mount is not reused.
Camera rolling shutter and moving-object motion are
not corrected. `input_manifest.json` records the context name, source frame
indices, timestamps and data version. Internal historical labels such as
`scene_ws4/frame_000000` are not official extraction identifiers.

### Run The Demo

Return to the NIDAR inference environment:

```bash
python inference/run_nidar_demo.py --dataset waymo \
  --input-root ./data/waymo_demo --weights-root ./weights \
  --device cpu --check-input
python inference/run_nidar_demo.py --dataset waymo \
  --input-root ./data/waymo_demo --weights-root ./weights \
  --output-root ./outputs/waymo_demo --device cpu --viewer auto
```

## nuScenes

### Download

1. Select **nuScenes `v1.0-mini`**, not nuImages. The
   [official tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)
   provides the direct mini download below. For other splits, use the
   [official download page](https://www.nuscenes.org/download) and its
   account/terms process. Follow the provider's applicable data terms.
   Allow at least 12 GB for the archive and extracted data:

```bash
mkdir -p data/nuscenes_official
curl --fail --location --retry 3 --continue-at - \
  https://www.nuscenes.org/data/v1.0-mini.tgz --output data/v1.0-mini.tgz
gzip -t data/v1.0-mini.tgz
tar -xzf data/v1.0-mini.tgz -C data/nuscenes_official
```

The archive retrieved for acceptance on 2026-10-16 is 4,167,696,325 bytes;
its measured SHA-256 is
`943037abbb3b26b3070dc76504a43eb440503b00baf9ac2f1538d9c03fc9298f`.
This is our download receipt, not a provider-published checksum. The provider's
multipart ETag and gzip integrity check also matched. Do not treat a partial
download as a complete dataset.

2. Follow the [official devkit layout](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup).
   Preserve all JSON metadata and the sensor directories; do not flatten or
   rename the files. The selected dataroot should look like:

```text
data/nuscenes_official/
  v1.0-mini/
    scene.json
    sample.json
    sample_data.json
    calibrated_sensor.json
    ego_pose.json
    ... (all other metadata tables)
  samples/
    CAM_FRONT/
    ... (all six cameras)
    LIDAR_TOP/
  sweeps/
  maps/
```

### Prepare Two Frames

```bash
python3.10 -m venv .venv-nuscenes-export
.venv-nuscenes-export/bin/python -m pip install -r requirements-nuscenes-export.txt
.venv-nuscenes-export/bin/python tools/data_export/prepare_nuscenes.py \
  --dataroot ./data/nuscenes_official --version v1.0-mini \
  --frame-count 2 --output-root ./data/nuscenes_demo
```

The default selects the first two keyframes of the alphabetically first scene.
Use `--scene scene-0061`, for example, only if that scene is in your split;
`--start-frame` and `--step` control selection within it. Scene/sample/sensor
tokens and timestamps are saved in `input_manifest.json`.

This is a single-sweep, six-camera export. It handles separate LiDAR/camera ego
poses and converts optical camera axes to the pipeline's coordinate system;
renaming raw nuScenes files is not an equivalent conversion. Native intensity
is divided by 255, matching the historical nuScenes evaluation convention.
Moving-object motion and per-point deskew are not corrected.

### Run The Demo

Return to the NIDAR inference environment:

```bash
python inference/run_nidar_demo.py --dataset nuscenes \
  --input-root ./data/nuscenes_demo --weights-root ./weights \
  --device cpu --check-input
python inference/run_nidar_demo.py --dataset nuscenes \
  --input-root ./data/nuscenes_demo --weights-root ./weights \
  --output-root ./outputs/nuscenes_demo --device cpu --viewer auto
```

The CLI selects the repository's existing
`inference/quantile_model/nintensity_mapping_quantile_model.pkl`, not the Waymo
calibration. This is not a new calibration fit or an accuracy guarantee for
the mini split. Only use trusted quantile files: their format is Python pickle.

## Outputs And Checks

Both routes produce `results/frame_*/nidar_intensity.npy`, scalar and colored
PLY files, a static preview, GT/prediction comparisons and JSON metrics.
The source manifest is copied into `results/`. See the [demo guide](DEMO.md).

- `--viewer auto`: open a window when the optional Open3D package and display
  are available. Use `--viewer none` for headless execution.
- `--device cuda`: use a supported GPU inference environment instead of CPU.
- `--frame-count 1`: infer one frame from the prepared export.
- `--no-eval`: skip GT comparisons and metrics, but keep point-cloud outputs.
- `--check-input`: check runtime dependencies, artifact paths, selected images,
  clouds and calibration without a model forward pass or output-directory
  writes. It does not replace checkpoint SHA verification by the downloader.

Exporters require a fresh output directory and never delete official input.
A failed conversion leaves an incomplete manifest that the demo rejects;
retry into a new empty directory. Conversion SDKs are unnecessary after export.

Acceptance records must identify the official source, converter environment,
complete inference result and visual inspection. Records are available for the
Waymo fixture and two real nuScenes mini frames. A full Waymo segment, broader
nuScenes coverage and CUDA remain separate checks; existing research figures
and synthetic tests alone do not satisfy them.
