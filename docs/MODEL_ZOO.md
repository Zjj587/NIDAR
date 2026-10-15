# NIDAR Model Zoo And Demo Artifacts

This page tracks the external artifacts required by the public demo. Large
checkpoints and datasets are not stored in git.

## Download Table

| Artifact | Target path after download | Download / status |
| --- | --- | --- |
| STN pseudo-NIR checkpoint | `$WEIGHTS_ROOT/stn/47.pth` | [47.pth](https://huggingface.co/zhangjunjie587/NIDAR/resolve/1390055c2c564d3276f29cc513b3ef3cdc355e40/47.pth) |
| IRNet U-Net + DeepLab checkpoint | `$WEIGHTS_ROOT/irnet/epoch_010.pth` | [epoch_010.pth](https://huggingface.co/zhangjunjie587/NIDAR/resolve/1390055c2c564d3276f29cc513b3ef3cdc355e40/epoch_010.pth) |
| Quantile remap model | `$WEIGHTS_ROOT/quantile/wintensity_mapping_quantile_model.pkl` | Included in `inference/quantile_model/` and may also be copied under `$WEIGHTS_ROOT/quantile/` |
| Waymo-style demo sample | `$DATA_ROOT/waymo/demo_scene/raw/` | Link pending |

The owner published the checkpoints at
[zhangjunjie587/NIDAR](https://huggingface.co/zhangjunjie587/NIDAR). On 2026-10-14,
anonymous Hub API access confirmed revision
`1390055c2c564d3276f29cc513b3ef3cdc355e40`, both byte sizes, and the matching
LFS SHA-256 metadata below. This was metadata verification, not a new full
checkpoint download, security scan, or end-to-end inference run. The owner then
uploaded eight companion files at revision
`fc639b767ffbbc1d9063f2aa4a8ed9837d1009ca`: the model card, config, calibration,
manifest, checksums, provenance, and license notices. Anonymous read-back
matched all eight byte-for-byte. Training-data and weight distribution terms
still require final reconciliation; public visibility does not by itself
grant unrestricted use.

The Hub stores both checkpoint files at its root. Prepare the local layout with:

```bash
python tools/download_weights.py --weights-root "$WEIGHTS_ROOT"
python tools/download_weights.py --weights-root "$WEIGHTS_ROOT" --check-only
```

This downloads the pinned revision, checks sizes and SHA-256, and copies the
bundled Waymo quantile model. A different existing file is rejected rather than
overwritten. The demo sample link is still pending; do not substitute a directory
of development-machine symlinks for a portable sample archive.

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

The STN architecture and training approach derive from
[cs-stereo](https://github.com/tiancheng-zhi/cs-stereo). Architecture provenance
must not be confused with checkpoint provenance. The 2026-10-14 read-only audit
found strong evidence that the selected `ckpt/47.pth` is a team-local retraining
checkpoint, rather than assuming it is the unchanged upstream download:

- `li:/home1/cs-stereo/ckpt/` retains all 48 snapshots, `0.pth` through
  `47.pth`, dated 2025-09-17 through 2025-09-19.
- The selected checkpoint records epoch 47, 48 configured epochs, batch size
  16, learning rate `5e-5`, `resume=None`, and the PittsStereo-RGBNIR train/test
  sequence identifiers. Its remote SHA-256 matches the STN identity above.
- The local `train.py` updates the STN optimizer and saves these numbered
  snapshots; its non-resume branch initializes the networks.
- A separate file named `pretrained.pth` has SHA-256
  `4594e0da78b06e59deca30d8a8e3eda4d5e9df6c60a0b91b993499b3fef03879`,
  distinct from the selected checkpoint. This is a whole-file comparison, not
  proof of a particular tensor-level difference.

Describe this artifact as a locally trained STN based on cs-stereo, frozen for
NIDAR inference. Do not imply that the architecture is original to NIDAR or that
the `train_waymo` directory proves STN training on Waymo: that directory contains
pseudo-NIR inference outputs. Exact tensor-level comparison with the separate
pretrained file and copying the full training evidence remain pending after
intermittent SSH failures. The existing local backup allowed a restricted read
of checkpoint training metadata without loading model tensors.

The upstream code carries an
[MIT license](https://github.com/tiancheng-zhi/cs-stereo/blob/master/LICENSE);
retain its copyright and license notices. Review training-data and weight
distribution terms separately before public artifact release. Local retraining
does not automatically remove third-party attribution or data obligations.

## Owner-Only Artifact Locations

The following paths are non-portable local evidence for the project owner. They
are intentionally included here so Zhang Junjie can find the current artifacts
alongside the published checkpoint links. They are not the public quick-start
paths, and the sample location is still an unpublished candidate.

```text
STN checkpoint:
  /home1/cs-stereo/ckpt/47.pth
  /home/ubuntu/nidar_bc_20260920/nidar_release/weights/stn/47.pth

IRNet U-Net + DeepLab checkpoint on li:
  /home1/pseudo-intensity-pipeline/checkpoints/loss_exp/test/checkpoints/epoch_010.pth

Verified full-model backup on zjj:
  /media/zjj/EAGET1/CQU_ZJJ/NIDAR/epoch_010.pth

Verified full-model backup on 2204:
  /home/ubuntu/nidar_bc_20260920/paper_irnet_recovery_20260926/user_received_EAGET1_20260926/epoch_010.pth

Candidate demo sample source:
  /home1/ZJJ/lidarrt/outputs/export/default/scene_ws4/exports/raw

Historical small-head demo output (not a full-paper-model release check):
  /home1/ZJJ/nidar_public_demo_smoke_20260828_rsync_104219/output/waymo_demo
```

The 2026-10-10 cleanup verified strict loading of the original STN and full
IRNet checkpoints on 2204. It did not complete a new two-frame end-to-end run:
the sample transfer from li was interrupted, and the existing 2204 BC environment
does not contain the required `scikit-image` and `plyfile` packages. The
historical small-head output above must not be used as evidence for that check.

The 2026-10-15 ThinkPad check subsequently completed anonymous public weight
downloads, full two-frame CPU inference, GT-independence checks, and Open3D
window validation. See [release acceptance](RELEASE_ACCEPTANCE.md).
The portable owner sample is now at
`/media/zjj/Newsmy/NIDAR/datasets/waymo_demo_2frames_20261010/raw`;
ThinkPad outputs are at `/home/ubuntu/NIDAR_acceptance_20261015/outputs/demo/results`.
These remain owner review paths, not public download links.

Do not use the old checkpoint under
`/home1/pseudo-intensity-pipeline/checkpoints/deep_intrinsic_v3_mask/` with the
default public config. It is a different architecture and is retained only as
historical evidence.

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
