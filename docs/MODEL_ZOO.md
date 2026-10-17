# NIDAR Model Zoo And Demo Artifacts

This page tracks the external artifacts required by the public demo. Large
checkpoints and datasets are not stored in git.

## Download Table

| Artifact | Target path after download | Download / status |
| --- | --- | --- |
| STN pseudo-NIR checkpoint | `$WEIGHTS_ROOT/stn/47.pth` | [47.pth](https://huggingface.co/zhangjunjie587/NIDAR/resolve/1390055c2c564d3276f29cc513b3ef3cdc355e40/47.pth) |
| IRNet U-Net + DeepLab checkpoint | `$WEIGHTS_ROOT/irnet/epoch_010.pth` | [epoch_010.pth](https://huggingface.co/zhangjunjie587/NIDAR/resolve/1390055c2c564d3276f29cc513b3ef3cdc355e40/epoch_010.pth) |
| Waymo remap model | `$WEIGHTS_ROOT/quantile/wintensity_mapping_quantile_model.pkl` | Bundled and copied by the downloader |
| nuScenes remap model | `inference/quantile_model/nintensity_mapping_quantile_model.pkl` | Bundled; automatically selected for converted nuScenes input |
| Waymo / nuScenes inputs | Obtain from the official provider | [Download and conversion commands](DATA_ACCESS.md); no project-hosted data copies |

Checkpoints, configuration and artifact metadata are hosted at
[zhangjunjie587/NIDAR](https://huggingface.co/zhangjunjie587/NIDAR).
The downloader pins checkpoint revision
`1390055c2c564d3276f29cc513b3ef3cdc355e40` and verifies the sizes and SHA-256
values below. Code, weights and datasets have distinct terms; read the
[artifact license notices](https://huggingface.co/zhangjunjie587/NIDAR/blob/main/LICENSE_NOTICES.md)
before use. Public availability does not grant unrestricted use.

The Hub stores both checkpoint files at its root. Prepare the local layout with:

```bash
python tools/download_weights.py --weights-root "$WEIGHTS_ROOT"
python tools/download_weights.py --weights-root "$WEIGHTS_ROOT" --check-only
```

This downloads the pinned revision, checks sizes and SHA-256, and copies the
bundled Waymo quantile model. A different existing file is rejected rather than
overwritten. Data is acquired separately; see [data access](DATA_ACCESS.md).

## Checkpoint Identity

Different historical checkpoints share the filename `epoch_010.pth`. The public
default requires the full U-Net + DeepLab model, not the older U-Net + small
convolution head.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| STN `47.pth` | 775894438 | `fcb676e06bc54d8e5b1589152244fcec8ef51b9e2c879e5465849184ad7f75f4` |
| IRNet U-Net + DeepLab `epoch_010.pth` | 226996282 | `91a6d10ec82295606fd7bb26f860703d000808a2e3f8714ad1ee9e6cdfa5f14f` |
| Bundled Waymo quantile model | 16262 | `0449bc5209c8bd1b4dc066b9b0f1e29a17e770f6ee5b0cab2cabf1b4650c81ca` |

These hashes identify the original checkpoints. Only load weights and remap
pickle files from trusted, verified sources. The inference loader uses
`weights_only=True` with a narrow allowlist for the legacy STN metadata, and
both networks must match their checkpoint state dictionaries strictly.

## STN Training Provenance

The released STN is a team-trained checkpoint based on
[cs-stereo](https://github.com/tiancheng-zhi/cs-stereo), frozen during NIDAR
inference. The STN architecture is not an original contribution of NIDAR.
Saved checkpoint metadata records:

| Setting | Value |
| --- | --- |
| Training data | PittsStereo-RGBNIR sequence identifiers |
| Selected epoch | 47, from 48 configured epochs |
| Batch size | 16 |
| Learning rate | `5e-5` |

This inference release does not include a complete STN training-reproduction
package. Checkpoint metadata alone does not establish a performance improvement
over upstream weights. See the
[model provenance](https://huggingface.co/zhangjunjie587/NIDAR/blob/main/PROVENANCE.md)
for artifact sources and verification scope.

The upstream code carries an
[MIT license](https://github.com/tiancheng-zhi/cs-stereo/blob/master/LICENSE);
retain its copyright and license notices. Review training-data and weight
distribution terms separately before public artifact release. Local retraining
does not automatically remove third-party attribution or data obligations.

## Verified Inference

Anonymous weight downloads, strict checkpoint loading and complete CPU
inference have been checked. Official Waymo and nuScenes two-frame checks,
the tested environment and their limits are documented in
[release acceptance](RELEASE_ACCEPTANCE.md). These checks do not establish
full-dataset quality, CUDA performance or training reproducibility.

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

The legacy layout above remains supported. New converters write `images/`,
`pointclouds/` and `input_manifest.json` directly under their `--output-root`;
nuScenes uses six cameras. Point-cloud NPZ files contain an Nx4 `points` array:
world XYZ and evaluation-only intensity. Camera JSON contains `[fx, fy, cx, cy]`,
`width`, `height`, and a camera-to-reference 4x4 `extrinsic`. Camera axes are
forward/left/up, not optical right/down/forward.

In the new exports, the reference is LIDAR_TOP. The legacy LiDAR JSON key
`ego2world` therefore stores the LiDAR-to-world transform, and its `extrinsic`
is identity. The manifest records this contract. Use the converters rather
than mixing these conventions with the native dataset coordinates.
