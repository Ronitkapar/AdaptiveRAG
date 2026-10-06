#!/usr/bin/env python3
"""
validate_corpus.py
------------------
Validates the research paper corpus manifest and local files.

Checks performed:
1. Manifest JSON exists and is syntactically valid JSON.
2. Manifest contains an expected number of papers (target: 10-20).
3. Document IDs are unique and follow valid identifier naming.
4. All required fields are present with correct data types.
5. All categories belong to the recognized taxonomy.
6. URLs are well-formed http/https links.
7. Local files exist, are valid PDFs (%PDF-), and match SHA256 checksums (if present).
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_corpus")

ALLOWED_CATEGORIES = {
    "foundational_rag",
    "dense_retrieval",
    "sparse_retrieval",
    "hybrid_retrieval",
    "reranking",
    "advanced_rag",
    "adaptive_retrieval",
}

REQUIRED_FIELDS = {
    "document_id": str,
    "title": str,
    "authors": list,
    "year": int,
    "category": str,
    "source_url": str,
    "download_url": str,
    "local_path": str,
    "description": str,
    "concepts": list,
}

URL_REGEX = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
DOC_ID_REGEX = re.compile(r"^[a-z0-9_]+$")
MIN_VALID_PDF_SIZE_BYTES = 50_000


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_corpus(manifest_path: Path, check_files: bool = True) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not manifest_path.is_file():
        return False, [f"Manifest file not found: {manifest_path}"]

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except Exception as exc:
        return False, [f"Failed to parse manifest JSON: {exc}"]

    if not isinstance(papers, list):
        return False, ["Manifest top-level entity must be a JSON list"]

    if len(papers) < 10 or len(papers) > 20:
        errors.append(f"Corpus size {len(papers)} outside target experimental range (10-20 papers)")

    seen_ids = set()
    category_counts = {cat: 0 for cat in ALLOWED_CATEGORIES}

    for idx, paper in enumerate(papers, 1):
        if not isinstance(paper, dict):
            errors.append(f"Entry {idx} is not a dictionary: {paper}")
            continue

        doc_id = paper.get("document_id")

        # 1. Required fields and types
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in paper:
                errors.append(f"Entry {idx} ('{doc_id}'): Missing required field '{field}'")
            elif not isinstance(paper[field], expected_type):
                errors.append(
                    f"Entry {idx} ('{doc_id}'): Field '{field}' expected {expected_type.__name__}, got {type(paper[field]).__name__}"
                )

        if not doc_id:
            continue

        # 2. Document ID uniqueness and format
        if not DOC_ID_REGEX.match(doc_id):
            errors.append(f"Document ID '{doc_id}' invalid format. Must be lowercase alphanumeric with underscores.")

        if doc_id in seen_ids:
            errors.append(f"Duplicate document_id detected: '{doc_id}'")
        seen_ids.add(doc_id)

        # 3. Authors non-empty string list
        authors = paper.get("authors")
        if isinstance(authors, list):
            if len(authors) == 0:
                errors.append(f"'{doc_id}': authors list cannot be empty")
            for a in authors:
                if not isinstance(a, str) or not a.strip():
                    errors.append(f"'{doc_id}': authors list contains non-string or blank item: {a}")

        # 4. Concepts non-empty string list
        concepts = paper.get("concepts")
        if isinstance(concepts, list):
            if len(concepts) < 3:
                errors.append(f"'{doc_id}': expected at least 3 concepts, found {len(concepts)}")
            for c in concepts:
                if not isinstance(c, str) or not c.strip():
                    errors.append(f"'{doc_id}': concepts list contains non-string or blank item: {c}")

        # 5. Category validation
        category = paper.get("category")
        if category:
            if category not in ALLOWED_CATEGORIES:
                errors.append(f"'{doc_id}': Unknown category '{category}'. Must be one of {sorted(ALLOWED_CATEGORIES)}")
            else:
                category_counts[category] += 1

        # 6. Year sanity check
        year = paper.get("year")
        if isinstance(year, int) and not (1990 <= year <= 2026):
            errors.append(f"'{doc_id}': Publication year {year} is out of expected bounds (1990-2026)")

        # 7. URL checks
        for url_field in ("source_url", "download_url"):
            url_val = paper.get(url_field)
            if isinstance(url_val, str) and not URL_REGEX.match(url_val):
                errors.append(f"'{doc_id}': Invalid URL format in '{url_field}': {url_val}")

        # 8. Local file verification (if requested)
        if check_files:
            local_path_str = paper.get("local_path")
            if local_path_str:
                path = Path(local_path_str)
                if not path.is_file():
                    errors.append(f"'{doc_id}': Local file does not exist: {path}")
                else:
                    size = path.stat().st_size
                    if size < MIN_VALID_PDF_SIZE_BYTES:
                        errors.append(f"'{doc_id}': File size {size} bytes is under minimum {MIN_VALID_PDF_SIZE_BYTES}")
                    try:
                        with open(path, "rb") as f:
                            header = f.read(5)
                            if header != b"%PDF-":
                                errors.append(f"'{doc_id}': File header is not '%PDF-': {header!r}")
                    except OSError as exc:
                        errors.append(f"'{doc_id}': Failed reading file {path}: {exc}")

                    expected_sha = paper.get("sha256")
                    if expected_sha:
                        actual_sha = compute_sha256(path)
                        if actual_sha != expected_sha:
                            errors.append(f"'{doc_id}': SHA-256 hash mismatch! Expected {expected_sha}, got {actual_sha}")

    # Check category distribution: every category should have at least 1 paper
    for cat, count in category_counts.items():
        if count == 0:
            errors.append(f"Category '{cat}' has no papers assigned in the corpus")

    is_valid = len(errors) == 0
    return is_valid, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate corpus manifest and local files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/papers.json"),
        help="Path to papers.json manifest",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only validate manifest metadata structure without checking local PDFs",
    )
    args = parser.parse_args()

    check_files = not args.manifest_only
    logger.info("Validating manifest: %s (check_files=%s)", args.manifest, check_files)

    is_valid, errors = validate_corpus(args.manifest, check_files=check_files)

    if is_valid:
        logger.info("PASSED: Corpus and manifest validation succeeded with zero errors.")
        return 0
    else:
        logger.error("FAILED: Found %d validation error(s):", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
