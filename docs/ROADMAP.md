# Release Roadmap

Checkboxes record completed work, not promised delivery dates.

## First Reproducible Inference Release

- [x] Public STN and full U-Net + DeepLab IRNet checkpoints with exact hashes.
- [x] Hugging Face model card, architecture config, provenance, and notices.
- [x] End-to-end source CLI, static previews, and optional Open3D viewer.
- [x] Pinned downloader and focused source/demo checks.
- [x] Validate the source package with the correct full-model configuration.
- [ ] Provide a portable two-frame sample or a precise extraction procedure
  for authorized dataset users, with file IDs and calibration.
- [x] Reproduce installation, public downloads, inference, and visualization
  on a clean machine; record environment, peak memory, and elapsed time.
- [x] Confirm predictions are unchanged when evaluation-only current-frame
  GT intensity is removed or replaced in the input.
- [x] Save reference output images, hashes, and a run manifest for the tested package.
- [ ] Finalize artifact-specific terms and tag the matched code/model release.

## Reuse And Maintenance

- [ ] Document the custom-data coordinate, image, and calibration contract
  with a worked example and an input validation command.
- [ ] Add continuous integration for the existing CPU source/demo tests.
- [ ] Release one simulator or additional-dataset adapter at a time, each with
  a sample, supported environment, reference outputs, and a test.
- [x] Record the CPU compatibility and runtime report in [release acceptance](RELEASE_ACCEPTANCE.md).
- [ ] Release training entry points, splits, losses, and model-selection rules
  as a separate verified training package.

## Presentation And Optional Hosting

- [x] Reuse real GT/NIDAR research media in the repository overview.
- [ ] Add a concise prerecorded video after the reproducible demo is ready.
- [ ] Consider a hosted interactive demo based on demand and compute needs.

The project page contains broader research demonstrations. A research result
does not mark an adapter or training workflow as publicly reproducible.
