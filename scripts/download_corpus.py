#!/usr/bin/env python3
"""
download_corpus.py
------------------
Reproducibly downloads research paper PDFs defined in data/metadata/papers.json.
Validates downloaded files by size and PDF magic bytes (%PDF-).
Avoids re-downloading existing valid documents unless --force is specified.
"""

import argparse
import json
import logging
import sys
import time
import hashlib

import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_corpus")

USER_AGENT = "AdaptiveRAG-Research-Corpus/1.0 (academic research; mailto:research@example.com)"
DEFAULT_TIMEOUT_SEC = 30
MIN_VALID_PDF_SIZE_BYTES = 50_000  # valid papers are at least 50 KB


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hexadecimal digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_valid_pdf_file(path: Path, expected_sha256: str | None = None) -> bool:
    """Check if file exists, meets minimum size, and starts with %PDF- header."""
    if not path.is_file():
        return False
    if path.stat().st_size < MIN_VALID_PDF_SIZE_BYTES:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                return False
    except OSError:
        return False

    if expected_sha256:
        actual_hash = compute_sha256(path)
        if actual_hash != expected_sha256:
            logger.warning(
                "SHA256 mismatch for %s: expected %s, got %s",
                path,
                expected_sha256,
                actual_hash,
            )
            return False

    return True


def download_paper(
    download_url: str,
    dest_path: Path,
    expected_sha256: str | None = None,
    max_retries: int = 3,
) -> bool:
    """Download a paper PDF with exponential backoff retries."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(download_url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("  Connecting (attempt %d/%d): %s", attempt, max_retries, download_url)
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP status {resp.status}")

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)

            # Validate temporary file
            if not is_valid_pdf_file(temp_path, expected_sha256=expected_sha256):
                temp_path.unlink(missing_ok=True)
                raise ValueError("Downloaded file failed validation (header/size/sha256)")

            # Atomic rename on success
            temp_path.replace(dest_path)
            size_kb = dest_path.stat().st_size / 1024
            logger.info("  Downloaded successfully: %.1f KB -> %s", size_kb, dest_path)
            return True

        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            logger.warning("  Attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                sleep_time = 2 ** attempt
                logger.info("  Retrying in %d seconds...", sleep_time)
                time.sleep(sleep_time)

    logger.error("  Failed to download %s after %d attempts", download_url, max_retries)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download research corpus PDFs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/papers.json"),
        help="Path to corpus metadata JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local PDF already exists and is valid",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        logger.error("Manifest file not found: %s", args.manifest)
        return 1

    with open(args.manifest, "r", encoding="utf-8") as f:
        try:
            papers = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON manifest: %s", exc)
            return 1

    logger.info("Loaded %d paper entries from %s", len(papers), args.manifest)

    downloaded = 0
    skipped = 0
    failed = 0

    for idx, paper in enumerate(papers, 1):
        doc_id = paper.get("document_id")
        title = paper.get("title")
        download_url = paper.get("download_url")
        local_path_str = paper.get("local_path")
        expected_sha = paper.get("sha256")

        if not doc_id or not download_url or not local_path_str:
            logger.error("[%d/%d] Incomplete record: %s", idx, len(papers), paper)
            failed += 1
            continue

        dest_path = Path(local_path_str)

        logger.info("[%d/%d] Processing %s: '%s'", idx, len(papers), doc_id, title)

        if not args.force and is_valid_pdf_file(dest_path, expected_sha256=expected_sha):
            logger.info("  Already exists and verified valid (%.1f KB). Skipping.", dest_path.stat().st_size / 1024)
            skipped += 1
            continue

        success = download_paper(download_url, dest_path, expected_sha256=expected_sha)
        if success:
            downloaded += 1
            # Polite pause to avoid hitting rate limits on public repositories
            time.sleep(1.0)
        else:
            failed += 1

    logger.info("=" * 60)
    logger.info("Download Summary: %d succeeded, %d skipped (valid), %d failed", downloaded, skipped, failed)
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
