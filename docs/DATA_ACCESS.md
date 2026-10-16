# Data Access And Preparation Status

NIDAR does not mirror Waymo or nuScenes raw data or preprocessed copies.
Obtain these datasets from their official providers under their respective
terms. Small UE5 and Isaac Sim exports are planned as the downloadable demo,
after asset-rights review and end-to-end validation.

**Status:** this page documents official acquisition and the release boundary.
It is not yet a validated raw-dataset-to-NIDAR conversion tutorial. The released
CLI has been tested on two already-exported Waymo frames, not on a fresh
official download. Simulator adapters and downloadable exports are not yet
available in this release.

## Waymo

1. Visit the [official download portal](https://waymo.com/open/download/) and
   complete the provider's sign-in and [terms](https://waymo.com/open/terms/).
   Keep account credentials and download authorization outside this repository.
2. Use the Perception Dataset for camera images, LiDAR measurements, and sensor
   calibration. Motion trajectories alone are not the input to this pipeline.
3. Preserve the dataset version, source segment identifier, frame timestamps,
   camera calibration, LiDAR calibration, and pose information. The historical
   exporter being reviewed expects Frame TFRecords; do not assume the newer
   modular format is interchangeable.
4. Use the [official Waymo code and format documentation](https://github.com/waymo-research/waymo-open-dataset)
   for the selected release. Keep dataset conversion dependencies separate from
   the CPU-tested NIDAR inference environment.

**Pending:** a portable exporter, pinned conversion environment, exact source
segment/frame selection, and validation from original data. Until those are
released, the demo requires an existing export matching the
[documented input layout](MODEL_ZOO.md#expected-demo-data-layout).
The internal names `scene_ws4/frame_000000` and `frame_000001` are export labels,
not official identifiers that independently reproduce the acceptance sample.

## nuScenes

1. Use the [official download page](https://www.nuscenes.org/download), complete
   the account/terms requirements, and select the nuScenes dataset rather than
   the separate nuImages dataset.
2. For a small setup check, start with the `v1.0-mini` split described in the
   [official tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html).
   Follow the [devkit setup instructions](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup)
   when unpacking the selected split; preserve sensor files and JSON metadata.
3. Keep camera intrinsics, calibrated-sensor transforms, ego poses, sample-data
   tokens, and timestamps. Camera/LiDAR alignment must account for the actual
   records rather than assuming equal filenames imply synchronized sensors.

**Pending:** a released nuScenes adapter, an exact tested devkit environment,
and an end-to-end run from official input. Do not rename nuScenes files to
Waymo filenames and expect the current demo to work. The shipped Waymo quantile
model is not a sensor-independent calibration.

## UE5 And Isaac Sim

The intended dataset repository is `zhangjunjie587/NIDAR-demo` on Hugging Face.
It is private during preparation; a public download link will be added only
after the package passes validation. No Waymo or nuScenes copies will be put
there as part of this release plan.

Each simulator package should contain exported RGB images, XYZ points,
calibration/poses, an input manifest, checksums, and reference outputs. The
offline demo should run without installing either simulator. Export scripts
and regeneration instructions will be documented separately.

Before release, check scene/material/asset redistribution terms, matching
projection conventions, the explicit IRNet output branch, and the source of
quantile calibration. Do not bundle engine binaries or third-party asset
source files. If there is no meaningful GT intensity, label the outputs as
predictions and omit GT accuracy claims.

## Acceptance Gates

- An official-data guide is complete only after its acquisition and conversion
  steps have been exercised on the recorded source version and frames.
- A simulator sample is ready only after its downloaded bytes reproduce a
  point cloud and nonblank visualization using the public source and weights.
- Existing research figures and the two-frame CPU acceptance remain separate
  evidence; they do not establish completion of these pending workflows.
