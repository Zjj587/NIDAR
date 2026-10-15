# Contributing To NIDAR

Focused bug reports, reproducibility fixes, and tested input adapters are
welcome. Check [demo setup](docs/DEMO.md) and
[troubleshooting](docs/TROUBLESHOOTING.md) before opening an issue.

Include the source revision, environment, checkpoint hashes, exact command,
and relevant log excerpt in bug reports. Describe expected and observed
behavior. For a new adapter, describe the geometry/calibration contract,
test sample, and expected output before proposing a large change.

Keep pull requests scoped to one behavior. Preserve the default
`RGB -> pseudo-NIR -> R -> remap` route, strict checkpoint matching, and the
separation between prediction and evaluation. Document intentional changes
to those contracts and add regression coverage for executable behavior.

Run these focused checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_public_release.py' -v
python -m unittest discover -s tests -p 'test_download_weights.py' -v
```

Checkpoint-loader changes also need `test_checkpoint.py` in a supported
PyTorch environment. Describe checks actually run in the pull request;
CPU fixtures do not establish full-model inference quality.

Keep large weights and raw datasets outside Git. Use the documented artifact
channels, retain attribution for reused components, and remove credentials
from public issue logs.
