"""
test_corpus.py
--------------
Automated test suite verifying the integrity and reproducibility
of the research paper corpus and its metadata manifest.
"""

import json
import unittest
from pathlib import Path

from scripts.validate_corpus import (
    ALLOWED_CATEGORIES,
    MIN_VALID_PDF_SIZE_BYTES,
    REQUIRED_FIELDS,
    validate_corpus,
)

MANIFEST_PATH = Path("data/metadata/papers.json")


class TestCorpus(unittest.TestCase):
    def test_manifest_file_exists(self):
        self.assertTrue(MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}")

    def test_manifest_validation(self):
        is_valid, errors = validate_corpus(MANIFEST_PATH, check_files=True)
        self.assertTrue(is_valid, f"Corpus validation failed with errors: {errors}")

    def test_corpus_size_target(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)
        self.assertTrue(10 <= len(papers) <= 20, f"Expected 10-20 papers, got {len(papers)}")

    def test_all_categories_covered(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)
        categories_present = {p["category"] for p in papers}
        self.assertEqual(
            categories_present,
            ALLOWED_CATEGORIES,
            f"Missing categories: {ALLOWED_CATEGORIES - categories_present}",
        )

    def test_raw_files_integrity(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)

        for paper in papers:
            local_path = Path(paper["local_path"])
            self.assertTrue(
                local_path.is_file(),
                f"Missing file for {paper['document_id']}: {local_path}",
            )
            self.assertGreaterEqual(
                local_path.stat().st_size,
                MIN_VALID_PDF_SIZE_BYTES,
                f"File too small for {paper['document_id']}",
            )
            with open(local_path, "rb") as f_pdf:
                self.assertEqual(
                    f_pdf.read(5),
                    b"%PDF-",
                    f"Corrupt PDF header for {paper['document_id']}",
                )


if __name__ == "__main__":
    unittest.main()

