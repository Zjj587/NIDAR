#!/usr/bin/env python3
"""Download the public NIDAR checkpoints and verify their artifact identities."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "zhangjunjie587/NIDAR"
REVISION = "1390055c2c564d3276f29cc513b3ef3cdc355e40"
CHUNK_SIZE = 1024 * 1024
ARTIFACTS = (
    {
        "filename": "47.pth",
        "target": "stn/47.pth",
        "bytes": 775894438,
        "sha256": "fcb676e06bc54d8e5b1589152244fcec8ef51b9e2c879e5465849184ad7f75f4",
    },
    {
        "filename": "epoch_010.pth",
        "target": "irnet/epoch_010.pth",
        "bytes": 226996282,
        "sha256": "91a6d10ec82295606fd7bb26f860703d000808a2e3f8714ad1ee9e6cdfa5f14f",
    },
    {
        "target": "quantile/wintensity_mapping_quantile_model.pkl",
        "source": "inference/quantile_model/wintensity_mapping_quantile_model.pkl",
        "bytes": 16262,
        "sha256": "0449bc5209c8bd1b4dc066b9b0f1e29a17e770f6ee5b0cab2cabf1b4650c81ca",
    },
)


def artifact_url(artifact):
    return f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{artifact['filename']}"


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_artifact(weights_root, artifact, check_only=False, timeout=60):
    destination = weights_root / artifact["target"]
    if destination.parent.is_symlink() or destination.is_symlink():
        raise ValueError(f"Refusing a symlink destination: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Expected a regular file: {destination}")
        if (destination.stat().st_size != artifact["bytes"]
                or file_digest(destination) != artifact["sha256"]):
            raise ValueError(
                f"Existing artifact does not match its size/SHA-256: {destination}. "
                "Use a different weights directory or review this file manually."
            )
        print(f"Verified existing: {destination}", flush=True)
        return destination
    if check_only:
        raise FileNotFoundError(f"Missing artifact: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if "source" in artifact:
        source = (ROOT / artifact["source"]).open("rb")
    else:
        print(f"Downloading {artifact['filename']} ({artifact['bytes']} bytes)", flush=True)
        request = Request(artifact_url(artifact), headers={"User-Agent": "NIDAR-weight-downloader/1.0"})
        source = urlopen(request, timeout=timeout)

    temporary = None
    try:
        with source, tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            started = last_report = time.monotonic()
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > artifact["bytes"]:
                    raise ValueError(f"Unexpected size for {artifact['target']}")
                digest.update(chunk)
                output.write(chunk)
                now = time.monotonic()
                if "filename" in artifact and (now - last_report >= 10 or size == artifact["bytes"]):
                    elapsed = max(now - started, 1e-6)
                    print(
                        f"  {artifact['filename']}: {size / artifact['bytes']:.1%} "
                        f"({size / 1024**2:.1f}/{artifact['bytes'] / 1024**2:.1f} MiB, "
                        f"{size / 1024**2 / elapsed:.2f} MiB/s)",
                        flush=True,
                    )
                    last_report = now
        if size != artifact["bytes"]:
            raise ValueError(f"Unexpected size for {artifact['target']}")
        if digest.hexdigest() != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {artifact['target']}")
        # Do not silently replace another file created while downloading.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Destination appeared during download: {destination}")
        temporary.replace(destination)
        print(f"Installed and verified: {destination}", flush=True)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify local files without network access.")
    parser.add_argument("--timeout", type=int, default=60, help="Network operation timeout in seconds.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    root = args.weights_root.expanduser().resolve()
    try:
        for artifact in ARTIFACTS:
            ensure_artifact(root, artifact, args.check_only, args.timeout)
    except (OSError, URLError, ValueError) as error:
        parser.exit(1, f"Weight preparation failed: {error}\n")
    print(f"All NIDAR artifacts verified under {root}", flush=True)


if __name__ == "__main__":
    main()
