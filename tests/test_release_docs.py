"""Public documentation and source-package privacy boundaries; standard library only."""

import importlib.util
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_package_docs", ROOT / "tools/release/make_source_package.py"
)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ReleaseDocumentationTests(unittest.TestCase):
    def setUp(self):
        self.documents = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))

    def test_public_docs_have_no_private_artifact_paths(self):
        private_path = re.compile(r"/(?:media|home\d?)/[^\s`]+")
        for path in self.documents:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text()
                self.assertIsNone(private_path.search(text))
                self.assertNotIn("Owner-Only Artifact Locations", text)

    def test_internal_planning_is_not_published_or_packaged(self):
        for name in ("docs/ROADMAP.md", "docs/OPEN_SOURCE_RELEASE_CHECKLIST.md"):
            with self.subTest(name=name):
                self.assertFalse((ROOT / name).exists())
                self.assertFalse(package.should_copy(Path(name), False))
                self.assertFalse(package.should_copy(Path(name), True))
        self.assertTrue(package.should_copy(Path("tests/test_release_docs.py"), False))

    def test_readme_does_not_lead_with_internal_acceptance_narrative(self):
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn("Release preview:", readme)
        self.assertNotIn("docs/ROADMAP.md", readme)
        self.assertNotIn("docs/OPEN_SOURCE_RELEASE_CHECKLIST.md", readme)
        self.assertIn("docs/RELEASE_ACCEPTANCE.md", readme)

    def test_third_party_stn_notice_is_shipped(self):
        for name in ("LICENSES/CS_STEREO_MIT.txt", "THIRD_PARTY_NOTICES.md"):
            self.assertTrue((ROOT / name).is_file())
            self.assertTrue(package.should_copy(Path(name), False))
        notice = (ROOT / "LICENSES/CS_STEREO_MIT.txt").read_text()
        self.assertIn("Copyright (c) 2019 Tiancheng Zhi", notice)
        self.assertIn("The above copyright notice and this permission notice", notice)

    def test_training_release_is_explicit_and_portable(self):
        for name in ("training/__init__.py", "training/data.py", "training/losses.py",
                     "training/prepare.py", "training/train.py", "training/config.yaml",
                     "training/sources.example.yaml", "tests/test_training.py", "docs/TRAINING.md",
                     "docs/TRAINING_DATA.md", "training/historical.py", "training/config_sparse.yaml",
                     "training/sources.legacy.example.yaml", "tests/test_historical_training.py"):
            self.assertTrue((ROOT / name).is_file())
            self.assertTrue(package.should_copy(Path(name), False))
            self.assertIsNone(re.search(r"/(?:media|home\d?)/", (ROOT / name).read_text()))
        self.assertFalse(package.should_copy(Path("training/private_experiment.yaml"), False))
        self.assertFalse(package.should_copy(Path("training/samples/frame.npz"), False))

    def test_local_document_links_and_anchors_resolve(self):
        for path in self.documents:
            text = path.read_text()
            links = re.findall(r"\[[^\]\n]*\]\(([^)\s]+)\)", text)
            links += re.findall(r'(?:href|src)="([^"]+)"', text)
            for link in links:
                url = urlsplit(link)
                if url.scheme or url.netloc:
                    continue
                target = path.parent / unquote(url.path) if url.path else path
                with self.subTest(path=path.relative_to(ROOT), link=link):
                    self.assertTrue(target.exists(), link)
                    if target.is_file() and target.suffix == ".md" and url.fragment:
                        headings = re.findall(r"^#{1,6}\s+(.+)$", target.read_text(), re.M)
                        anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
                                   for h in headings}
                        self.assertIn(unquote(url.fragment), anchors)

    def test_training_data_and_historical_scope_are_explicit(self):
        readme = (ROOT / "README.md").read_text()
        guide = (ROOT / "docs/TRAINING.md").read_text()
        data = (ROOT / "docs/TRAINING_DATA.md").read_text()
        self.assertIn("docs/TRAINING_DATA.md", readme)
        self.assertIn("Recovered Configuration Reference", guide)
        self.assertIn("lambda_intensity: 1.0", guide)
        self.assertIn("lambda_lidar: 0.5", guide)
        self.assertIn("valid_mask", data)
        self.assertIn("R*S", data)
        self.assertIn("I_hat", data)
        self.assertIn("historical-dense-v1", data)
        self.assertIn("sample-random-legacy", guide)
        self.assertIn("all-true", data)
        self.assertIn("last_inference.pth", guide)
        self.assertNotIn("does not implement this historical densification path", data)

    def test_frozen_release_and_progress_have_explicit_boundaries(self):
        readme = (ROOT / "README.md").read_text()
        release = (ROOT / "docs/RELEASE.md").read_text()
        self.assertTrue(package.should_copy(Path("docs/RELEASE.md"), False))
        self.assertIn("git clone --branch v0.1.0", readme)
        self.assertIn("git clone --branch v0.1.0", release)
        for label in ("Waymo:", "nuScenes:", "Public STN", "IRNet training:"):
            self.assertIn("- [x] " + label, readme)
        for label in ("BC training/evaluation code", "IRNet improvements:", "Packaged UE5"):
            self.assertIn("- [ ] " + label, readme)
        self.assertNotIn("Frozen baseline:", readme)
        self.assertNotIn("Unchecked items are planned work", readme)
        self.assertIn("research demonstrations, not packaged workflows", release)
        self.assertIn("No improvement in R", release)
        self.assertIn("1390055c2c564d3276f29cc513b3ef3cdc355e40", release)

    def test_training_example_matches_download_guide(self):
        guide = (ROOT / "docs/TRAINING.md").read_text()
        self.assertEqual(guide.count("--dataroot ./data/nuscenes_official"), 2)
        self.assertEqual(guide.count(".venv-nuscenes-export/bin/python"), 2)
        data = (ROOT / "docs/TRAINING_DATA.md").read_text()
        self.assertIn("uint8 conversion/resizing", data)
        self.assertIn("not a reliable measurement-validity mask", data)


if __name__ == "__main__":
    unittest.main()
