"""Frozen-result and on-disk experiment-artifact loading.

Two sources, deliberately kept separate:

* **The committed snapshot** (``frontend/data/headline.json``) is a
  curated projection of already-committed research results. It is
  what makes the Benchmarks page render on a fresh clone, where the
  gitignored experiment artifacts do not exist.
* **The on-disk artifacts** (``experiments/...``) are the richer,
  working-tree-only record: per-query rows, run manifests and the
  project's own figures. They are read when present and reported as
  absent otherwise, never faked.

Every loader records whether its data came from the snapshot or the
working tree, so the UI can label a number ``FROZEN (committed
snapshot)`` or ``WORKING TREE (research machine)`` honestly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_rag.config.paths import EXPERIMENTS_DIR, REPO_ROOT

FRONTEND_DIR = REPO_ROOT / "frontend"
SNAPSHOT_PATH = FRONTEND_DIR / "data" / "headline.json"
PHASES_PATH = FRONTEND_DIR / "data" / "phases.json"

SNAPSHOT = "snapshot"
WORKING_TREE = "working_tree"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_row_system(row: dict[str, Any]) -> str:
    """Return the row's strategy/system name across the project's schema variants.

    Older working-tree exports occasionally use ``strategy`` instead of
    ``system``; some legacy tables also use ``arm``/``policy``. The UI should
    accept any of these names and keep rendering the same benchmark charts.
    """
    for key in ("system", "strategy", "arm", "policy", "System", "Strategy"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _to_float(value: Any) -> float | None:
    """Parse a metric/latency value across snapshot and artifact variants.

    Snapshot rows carry floats; working-tree tables carry display strings
    (e.g. ``"472.84"``) with ``""`` for not-applicable stages. Returns
    ``None`` when the value is absent or non-numeric so charts can skip
    the point instead of raising ``KeyError``/``ValueError``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("na", "n/a", "none", "null", "-"):
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def normalize_e1_table(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the E1 table with canonical numeric row keys.

    Accepts both the committed snapshot shape (``system``/``recall_at_5``/
    ``mrr``/``hit_at_5``/``median_latency_ms`` as floats) and the
    working-tree artifact shape (``System``/``Recall@5``/``MRR``/
    ``Median latency (ms)`` as display strings, no hit column).
    Idempotent: normalizing an already-canonical table is a no-op.
    """
    if data is None:
        return None
    rows = data.get("rows", []) if isinstance(data, dict) else []
    normalised: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalised.append(
            {
                "system": resolve_row_system(row),
                "recall_at_5": _to_float(row.get("recall_at_5", row.get("Recall@5"))),
                "mrr": _to_float(row.get("mrr", row.get("MRR"))),
                "hit_at_5": _to_float(row.get("hit_at_5", row.get("Hit@5"))),
                "median_latency_ms": _to_float(
                    row.get("median_latency_ms", row.get("Median latency (ms)"))
                ),
                "p95_latency_ms": _to_float(
                    row.get("p95_latency_ms", row.get("P95 latency (ms)"))
                ),
            }
        )
    out = dict(data)
    out["rows"] = normalised
    return out


def load_headline_snapshot() -> dict[str, Any]:
    """Load the committed headline snapshot (frozen results)."""
    data = _load_json(SNAPSHOT_PATH)
    if data is None:
        raise FileNotFoundError(
            f"Headline snapshot not found at {SNAPSHOT_PATH}. "
            "The Benchmarks page needs frontend/data/headline.json."
        )
    return data


def load_phase_timeline() -> list[dict[str, Any]]:
    """Load the committed phase timeline (Research Journey)."""
    data = _load_json(PHASES_PATH)
    if data is None:
        raise FileNotFoundError(f"Phase timeline not found at {PHASES_PATH}.")
    return data["phases"]


# --- On-disk experiment artifacts (working tree only) ----------------------


def _phase7_arm_dir(arm: str) -> Path | None:
    """Locate a Phase 7 E1 arm's run directory by strategy name."""
    phase7 = EXPERIMENTS_DIR / "phase7"
    if not phase7.is_dir():
        return None
    # E1 arm directories are named e1r107[_bc]__E1_baseline_comparison__<arm>.
    candidates = sorted(phase7.glob(f"e1r107*__E1_baseline_comparison__{arm}"))
    return candidates[0] if candidates else None


def load_arm_rows(arm: str) -> tuple[list[dict[str, Any]], str]:
    """Load per-query result rows for a Phase 7 E1 arm.

    Returns ``(rows, source)`` where ``source`` is ``working_tree``
    when the artifact was read, or ``snapshot`` with an empty list
    when it is not present (a fresh clone).
    """
    arm_dir = _phase7_arm_dir(arm)
    if arm_dir is None:
        return [], SNAPSHOT
    rows_path = arm_dir / "rows.jsonl"
    if not rows_path.is_file():
        return [], SNAPSHOT
    rows: list[dict[str, Any]] = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, WORKING_TREE


def load_e1_table() -> tuple[dict[str, Any] | None, str]:
    """Load the frozen E1 five-arm comparison table.

    Prefers the on-disk artifact (richer, with the full column set)
    and falls back to the committed snapshot. The source is returned
    so the UI can label it.
    """
    artifact = (
        EXPERIMENTS_DIR
        / "phase7"
        / "analysis"
        / "e1_e6_e7_e8"
        / "tables"
        / "e1_main_comparison.json"
    )
    data = _load_json(artifact)
    if data is not None:
        return normalize_e1_table(data), WORKING_TREE
    snapshot = load_headline_snapshot()
    return normalize_e1_table(snapshot.get("phase7_e1")), SNAPSHOT


def load_figures() -> dict[str, Path]:
    """Locate the project's own matplotlib figures (working tree only)."""
    figures_dir = EXPERIMENTS_DIR / "phase7" / "figures"
    found: dict[str, Path] = {}
    if figures_dir.is_dir():
        for path in sorted(figures_dir.glob("*.png")):
            found[path.stem] = path
    return found


def experiment_tree() -> list[dict[str, Any]]:
    """A lightweight listing of the experiment directories present.

    Used by the Experiment Explorer to show what is available on this
    machine. Empty on a fresh clone, which the UI reports honestly.
    """
    if not EXPERIMENTS_DIR.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(EXPERIMENTS_DIR.iterdir()):
        if path.name == ".gitkeep" or path.name == "README.md":
            continue
        if path.is_dir():
            entries.append(
                {"name": path.name, "path": str(path), "kind": "directory"}
            )
        elif path.suffix == ".json":
            entries.append({"name": path.name, "path": str(path), "kind": "file"})
    return entries
