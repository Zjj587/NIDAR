# NIDAR Model Zoo And Demo Artifacts

This page tracks the external artifacts required by the public demo. Large
checkpoints and datasets are not stored in git.

## Download Table

| Artifact | Target path after download | Status |
| --- | --- | --- |
| STN pseudo-NIR checkpoint | `$WEIGHTS_ROOT/stn/47.pth` | Link pending |
| IRNet checkpoint | `$WEIGHTS_ROOT/irnet/epoch_010.pth` | Link pending |
| Quantile remap model | `$WEIGHTS_ROOT/quantile/wintensity_mapping_quantile_model.pkl` | Included in `inference/quantile_model/` and may also be copied under `$WEIGHTS_ROOT/quantile/` |
| Waymo-style demo sample | `$DATA_ROOT/waymo/demo_scene/raw/` | Link pending |

Replace the pending links with public download URLs after the artifacts are
uploaded to an approved storage location.

## Local Evidence Paths On zjj/li

The following paths are non-portable local evidence for the project owner. They
are intentionally included here so Zhang Junjie can find the current artifacts
before replacing the pending public links. They are not the public quick-start
paths.

```text
STN checkpoint:
  /home1/cs-stereo/ckpt/47.pth

IRNet checkpoint:
  /home1/pseudo-intensity-pipeline/checkpoints/deep_intrinsic_v3_mask/checkpoints/epoch_010.pth

Candidate demo sample source:
  /home1/ZJJ/lidarrt/outputs/export/default/scene_ws4/exports/raw

Last successful local demo output:
  /home1/ZJJ/nidar_public_demo_smoke_20260828_rsync_104219/output/waymo_demo
```

## Expected Weight Layout

```text
$WEIGHTS_ROOT/
  stn/
    47.pth
  irnet/
    epoch_010.pth
  quantile/
    wintensity_mapping_quantile_model.pkl
```

The public config template uses these paths through `${WEIGHTS_ROOT}`.

## Expected Demo Data Layout

```text
$DATA_ROOT/
  waymo/
    demo_scene/
      raw/
        images/
          frame_000000_cam1.jpg
          frame_000000_cam2.jpg
          ...
          frame_000001_cam5.jpg
          frame_000000_cam1.json
          ...
          frame_000001_cam5.json
        pointclouds/
          frame_000000_raw_pc.npz
          frame_000000_lidar1.json
          frame_000001_raw_pc.npz
          frame_000001_lidar1.json
```

The point cloud arrays must contain at least `x, y, z, intensity`. The camera
JSON files provide `intrinsic`, `extrinsic`, `width`, and `height`. The lidar
JSON files provide the lidar extrinsic and, when available, `ego2world`.
