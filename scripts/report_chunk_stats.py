#!/usr/bin/env python3
"""
report_chunk_stats.py
---------------------
Analyzes generated chunk artifacts and prints token distribution calibration metrics.
"""

import json
import sys
from pathlib import Path

from adaptive_rag.config.paths import CHUNKS_DIR, STATS_DIR


def main() -> int:
    # The chunking pipeline writes the stats file beside the chunks it
    # describes (`out_dir.parent / "chunk_stats.json"`), so read it from the
    # same location. The pre-Phase-8 fixed `STATS_DIR` copy is kept only as a
    # fallback for older layouts.
    stats_file = CHUNKS_DIR.parent / "chunk_stats.json"
    if not stats_file.is_file():
        stats_file = STATS_DIR / "chunk_stats.json"
    if not stats_file.is_file():
        print("No chunk stats found. Run scripts/build_chunks.py first.", file=sys.stderr)
        return 1

    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    print("=" * 60)
    print("AdaptiveRAG Chunking Calibration Report")
    print("=" * 60)
    print(f"Total Chunks:       {stats.get('count', 0)}")
    print(f"Total Tokens:       {stats.get('total_tokens', 0):,}")
    print(f"Min Tokens/Chunk:   {stats.get('min', 0)}")
    print(f"Median Tokens:      {stats.get('median', 0)}")
    print(f"p95 Tokens:         {stats.get('p95', 0)}")
    print(f"Max Tokens/Chunk:   {stats.get('max', 0)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
