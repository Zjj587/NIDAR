"""Downloader contracts with small fixture bytes; no real network or models."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_weights", ROOT / "tools/download_weights.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadWeightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "weights"
        self.payload = b"small checkpoint fixture"
        self.artifact = {
            "filename": "47.pth", "target": "stn/47.pth", "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.target = self.root / self.artifact["target"]

    def write_existing(self, data):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(data)

    def assert_no_partial(self):
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_pinned_public_urls_and_local_layout(self):
        for artifact in downloader.ARTIFACTS[:2]:
            url = downloader.artifact_url(artifact)
            self.assertIn("/zhangjunjie587/NIDAR/resolve/" + downloader.REVISION + "/", url)
            self.assertNotIn("/main/", url)
        self.assertEqual(downloader.ARTIFACTS[0]["target"], "stn/47.pth")
        self.assertEqual(downloader.ARTIFACTS[1]["target"], "irnet/epoch_010.pth")

    def test_download_installs_verified_bytes(self):
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.payload)) as opened:
            self.assertEqual(downloader.ensure_artifact(self.root, self.artifact), self.target)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(opened.call_args.args[0].full_url, downloader.artifact_url(self.artifact))
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_existing_valid_file_skips_network(self):
        self.write_existing(self.payload)
        with patch.object(downloader, "urlopen") as opened:
            downloader.ensure_artifact(self.root, self.artifact)
        opened.assert_not_called()

    def test_download_reports_progress(self):
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.payload)):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                downloader.ensure_artifact(self.root, self.artifact)
        self.assertIn("47.pth: 100.0%", output.getvalue())
        self.assertIn("MiB/s", output.getvalue())

    def test_existing_bad_hash_is_not_overwritten(self):
        original = b"x" * len(self.payload)
        self.write_existing(original)
        with patch.object(downloader, "urlopen") as opened:
            with self.assertRaisesRegex(ValueError, "Existing artifact"):
                downloader.ensure_artifact(self.root, self.artifact)
        self.assertEqual(self.target.read_bytes(), original)
        opened.assert_not_called()

    def test_existing_bad_size_is_not_overwritten(self):
        self.write_existing(b"old")
        with self.assertRaisesRegex(ValueError, "Existing artifact"):
            downloader.ensure_artifact(self.root, self.artifact)
        self.assertEqual(self.target.read_bytes(), b"old")

    def test_download_hash_mismatch_removes_partial(self):
        wrong = b"x" * len(self.payload)
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(wrong)):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                downloader.ensure_artifact(self.root, self.artifact)
        self.assert_no_partial()

    def test_download_wrong_size_removes_partial(self):
        for payload in (b"short", self.payload + b"extra"):
            with self.subTest(size=len(payload)):
                with patch.object(downloader, "urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaisesRegex(ValueError, "size"):
                        downloader.ensure_artifact(self.root, self.artifact)
                self.assert_no_partial()

    def test_network_error_does_not_install_file(self):
        with patch.object(downloader, "urlopen", side_effect=URLError("offline")):
            with self.assertRaises(URLError):
                downloader.ensure_artifact(self.root, self.artifact)
        self.assert_no_partial()

    def test_interrupted_read_removes_partial(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("interrupted")
                return super().read(size)

        with patch.object(downloader, "urlopen", return_value=Interrupted(self.payload)):
            with self.assertRaisesRegex(OSError, "interrupted"):
                downloader.ensure_artifact(self.root, self.artifact)
        self.assert_no_partial()

    def test_check_only_has_no_network_or_directory_creation(self):
        with patch.object(downloader, "urlopen") as opened:
            with self.assertRaises(FileNotFoundError):
                downloader.ensure_artifact(self.root, self.artifact, check_only=True)
        self.assertFalse(self.root.exists())
        opened.assert_not_called()

    def test_check_only_accepts_verified_file(self):
        self.write_existing(self.payload)
        with patch.object(downloader, "urlopen") as opened:
            downloader.ensure_artifact(self.root, self.artifact, check_only=True)
        opened.assert_not_called()

    def test_directory_destination_is_rejected(self):
        self.target.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "regular file"):
            downloader.ensure_artifact(self.root, self.artifact)

    def test_dangling_symlink_destination_is_rejected(self):
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            downloader.ensure_artifact(self.root, self.artifact)
        self.assertTrue(self.target.is_symlink())

    def test_symlinked_subdirectory_is_rejected(self):
        self.root.mkdir(parents=True)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.target.parent.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            downloader.ensure_artifact(self.root, self.artifact)
        self.assertFalse(list(elsewhere.iterdir()))

    def test_bundled_quantile_is_copied_and_verified(self):
        artifact = downloader.ARTIFACTS[2]
        with patch.object(downloader, "urlopen") as opened:
            target = downloader.ensure_artifact(self.root, artifact)
        self.assertEqual(downloader.file_digest(target), artifact["sha256"])
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
