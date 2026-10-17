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


if __name__ == "__main__":
    unittest.main()
