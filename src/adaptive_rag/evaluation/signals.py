"""
evaluation.signals
------------------
Phase 9 per-query **signal table**: one row per query joining the registered
candidate signals to their dispersion targets.

This is the artifact Gate 9.2 reads. It is deliberately a *table*, not an
analysis: every statistical decision happens in the gate script, so the numbers a
result rests on can be inspected, diffed and recomputed without re-running.

**The leakage boundary is structural, not documentary.** `SignalRow` holds no
ground truth beyond the two dispersion targets it is paired with, and
`build_signal_table` accepts only `StrategyOutcome` maps -- which cannot carry
`relevant_documents` even in principle (`evaluation.dispersion`). The features
and the labels are assembled side by side here, at the one place where that is
legitimate, and nowhere else.

**Nothing here is a deployable feature.** Signals requiring two arms (`jaccard_*`)
are computed for Gate 9.2's *mechanism* question only. Gate 9.3 separately asks
whether a single-arm or query-only proxy carries the same information.
Conflating the two would produce a router feature that cannot be computed before
the router must decide -- the exact error this phase exists to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.evaluation.agreement import (
    DEFAULT_K,
    distinct_doc_ratio,
    jaccard,
    overlap_at_k,
    rank_correlation,
    registered_pairs,
    score_decay_slope,
    score_gap_top1_top2,
    top1_agreement,
    union_concentration,
)
from adaptive_rag.evaluation.dispersion import (
    SOUND_STRATEGIES,
    StrategyOutcome,
    dispersion,
)

SIGNALS_VERSION = "phase9_signals_v1"

#: Arms carrying a score-geometry signal, per §2.4 (bm25 and dense only).
#: `hybrid` is excluded because its RRF scores are rank-derived, not comparable
#: to a raw retrieval score, and no hybrid score-geometry signal is registered.
SCORE_ARMS: tuple[str, ...] = ("bm25", "dense")


def _opt(value: float | None) -> float | None:
    """Round a computed signal, preserving `None` as `None`.

    A signal that could not be computed must not become 0.0. Rounding has to
    happen *after* the None check, or the sentinel itself would be corrupted.
    """
    return None if value is None else round(value, 6)


class Candidate(BaseModel):
    """One Gate 9.2 candidate signal: its column name, family, and direction.

    The `expected_direction` field is the machine-readable form of §2.4's
    direction column. It exists because that column was a single blanket
    annotation across a whole family, which was wrong for `union_concentration`
    and `score_decay_slope`: both run *opposite* to the jaccard-style signals.
    Encoding direction per candidate makes that explicit and lets
    `tests/test_signals.py` assert the convention rather than leave it to prose.

    Direction semantics, stated once:
      * `negative` -- higher signal value associates with LOWER dispersion.
      * `positive` -- higher signal value associates with HIGHER dispersion.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    family: Literal["disagreement", "score_geometry", "query_intrinsic"]
    expected_direction: Literal["negative", "positive", "either"]
    definition: str


#: **The** Gate 9.2 candidate family, transcribed from `docs/phases/phase-9.md`
#: §2.4. This constant -- not the set of quantities the code happens to be able
#: to compute -- defines what may be tested.
#:
#: §2.4 states: "A signal not listed here may not enter the analysis as a
#: candidate; adding one after seeing results requires a new, later document."
#: `overlap_at_k`, the non-registered `top1_agreement` pairs,
#: `distinct_doc_ratio@hybrid`, and `rank_correlation@bm25|hybrid` are all
#: computable and are all *absent* here. They are computed into the row for
#: inspection but are excluded by `signal_columns`.
#:
#: The query-intrinsic representatives are **deliberately empty**: §2.4's own
#: wording is self-contradictory (it names "the 37 Phase 6 QueryFeatures columns"
#: in one place and "4 query-intrinsic representatives" in the next), and the
#: repository does not determine which four were intended. See
#: `docs/phases/phase-9.md` §8 Issue C. Until that is resolved the family is
#: incomplete and Gate 9.2 must not run.
REGISTERED_CANDIDATES: tuple[Candidate, ...] = (
    # --- disagreement (H1): 8 ---
    Candidate(
        name="jaccard@bm25|dense",
        family="disagreement",
        expected_direction="negative",
        definition="|A n B| / |A u B| over top-5 document IDs; higher = more agreement",
    ),
    Candidate(
        name="jaccard@bm25|hybrid",
        family="disagreement",
        expected_direction="negative",
        definition="|A n B| / |A u B| over top-5 document IDs; higher = more agreement",
    ),
    Candidate(
        name="jaccard@dense|hybrid",
        family="disagreement",
        expected_direction="negative",
        definition="|A n B| / |A u B| over top-5 document IDs; higher = more agreement",
    ),
    Candidate(
        name="union_concentration",
        family="disagreement",
        expected_direction="positive",
        definition="|A u B| / (|A| + |B|) over top-5; HIGHER = MORE DIFFERENT",
    ),
    Candidate(
        name="distinct_doc_ratio@bm25",
        family="disagreement",
        expected_direction="either",
        definition="distinct documents in bm25 top-5, over 5; single-arm, deployable",
    ),
    Candidate(
        name="distinct_doc_ratio@dense",
        family="disagreement",
        expected_direction="either",
        definition="distinct documents in dense top-5, over 5; single-arm, deployable",
    ),
    Candidate(
        name="top1_agreement@bm25|dense",
        family="disagreement",
        expected_direction="negative",
        definition="True when bm25 and dense share the rank-1 document; higher = more agreement",
    ),
    Candidate(
        name="rank_correlation@bm25|dense",
        family="disagreement",
        expected_direction="negative",
        definition="mid-rank Spearman over shared top-5 documents; higher = more agreement",
    ),
    # --- score geometry (H4): 4 ---
    Candidate(
        name="score_gap_top1_top2@bm25",
        family="score_geometry",
        expected_direction="negative",
        definition="(s1 - s2) / |s1| on bm25 top-5; larger gap = more confident = less dispersion",
    ),
    Candidate(
        name="score_gap_top1_top2@dense",
        family="score_geometry",
        expected_direction="negative",
        definition="(s1 - s2) / |s1| on dense top-5; larger gap = more confident = less dispersion",
    ),
    Candidate(
        name="score_decay_slope@bm25",
        family="score_geometry",
        expected_direction="positive",
        definition="OLS slope of bm25 score on rank, top-5; FLAT (near 0) = more dispersion",
    ),
    Candidate(
        name="score_decay_slope@dense",
        family="score_geometry",
        expected_direction="positive",
        definition="OLS slope of dense score on rank, top-5; FLAT (near 0) = more dispersion",
    ),
    # --- query-intrinsic (H5): UNRESOLVED, see §8 Issue C ---
)

#: Gate 9.2 hypothesis family size. **Amended 16 -> 12** (docs/phases/phase-9.md
#: §8.7): H5 (query-intrinsic) is withdrawn as an unidentifiable family, because
#: no document, commit, plan, ADR or comment ever named the four representatives
#: §2.4 declared. The family is 8 disagreement + 4 score geometry.
#:
#: Note this is a correction to the *multiplicity family*, not a tightening: a
#: smaller family yields smaller Holm-adjusted p-values. It is adopted because a
#: family must be composed of hypotheses that actually exist.
DECLARED_FAMILY_SIZE: int = 12


class SignalRow(BaseModel):
    """One query: its dispersion targets and every registered signal for it.

    `extra="forbid"` is the leakage guard. The Phase 6 `QueryFeatures` columns are
    not fields here -- they are carried as an opaque mapping so this model cannot
    smuggle in an arbitrary column, and so the Gate 3 control set stays visibly
    separate from the Phase 9 signals rather than blending into one block.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str

    # --- targets (never features) ---
    dispersion: float
    dispersion_mrr: float

    # --- disagreement family: two-arm signals, mechanism-only ---
    jaccard: dict[str, float] = Field(default_factory=dict)
    overlap_at_k: dict[str, float] = Field(default_factory=dict)
    top1_agreement: dict[str, bool] = Field(default_factory=dict)
    rank_correlation: dict[str, float | None] = Field(default_factory=dict)
    union_concentration: float | None = None

    # --- single-arm signals: the only deployable kind ---
    distinct_doc_ratio: dict[str, float] = Field(default_factory=dict)

    # --- score geometry (H4): single-arm, read from traces.jsonl ---
    score_gap_top1_top2: dict[str, float | None] = Field(default_factory=dict)
    score_decay_slope: dict[str, float | None] = Field(default_factory=dict)

    # --- Phase 6 query features, carried opaquely for the Gate 3 control ---
    query_features: dict[str, float] = Field(default_factory=dict)


def _numeric_features(raw: Mapping[str, Any] | None) -> dict[str, float]:
    """Keep only the numeric Phase 6 features.

    `content_terms` and the one-hot categoricals are dropped: they are lists and
    strings, not quantities, and coercing them would invent numbers. The one-hot
    columns Gate 3 used are recovered by the gate script from the raw trace, not
    from here -- a signal table that encoded categoricals would quietly change
    the Gate 3 control's feature set.
    """
    if not raw:
        return {}
    out: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def build_signal_table(
    per_query: Mapping[str, Mapping[str, StrategyOutcome]],
    strategies: Sequence[str] = SOUND_STRATEGIES,
    k: int = DEFAULT_K,
    query_features: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[SignalRow]:
    """One `SignalRow` per query complete under `strategies`, in id order.

    Only queries present for **every** strategy are emitted, so the table has one
    fixed denominator. A row missing an arm would have to leave that arm's signals
    null, and a correlation over a varying column set is not a correlation.

    `query_features` is optional and defaults to empty: the Phase 6 columns are a
    *control* for Gate 9.3, and Gate 9.2 must be computable without them so a
    missing feature source cannot silently shrink the disagreement family.
    """
    features = query_features or {}
    pairs = registered_pairs(strategies)
    rows: list[SignalRow] = []
    for query_id in sorted(per_query):
        outcomes = per_query[query_id]
        if not all(strategy in outcomes for strategy in strategies):
            continue
        target = dispersion(query_id, outcomes, strategies)

        jaccards: dict[str, float] = {}
        overlaps: dict[str, float] = {}
        tops: dict[str, bool] = {}
        rank_corrs: dict[str, float | None] = {}
        for system_a, system_b in pairs:
            pair = f"{system_a}|{system_b}"
            docs_a = outcomes[system_a].retrieved_document_ids
            docs_b = outcomes[system_b].retrieved_document_ids
            jaccards[pair] = round(jaccard(docs_a, docs_b, k), 6)
            overlaps[pair] = round(overlap_at_k(docs_a, docs_b, k), 6)
            tops[pair] = top1_agreement(docs_a, docs_b)
            rank_corrs[pair] = rank_correlation(docs_a, docs_b, k)

        rows.append(
            SignalRow(
                query_id=query_id,
                dispersion=target.dispersion,
                dispersion_mrr=target.dispersion_mrr,
                jaccard=jaccards,
                overlap_at_k=overlaps,
                top1_agreement=tops,
                rank_correlation=rank_corrs,
                union_concentration=union_concentration(
                    query_id,
                    outcomes[strategies[0]].retrieved_document_ids,
                    outcomes[strategies[1]].retrieved_document_ids,
                    k,
                ).union_concentration,
                distinct_doc_ratio={
                    strategy: round(
                        distinct_doc_ratio(outcomes[strategy].retrieved_document_ids, k),
                        6,
                    )
                    for strategy in strategies
                },
                score_gap_top1_top2={
                    strategy: _opt(score_gap_top1_top2(outcomes[strategy].scores, k))
                    for strategy in SCORE_ARMS
                    if strategy in outcomes
                },
                score_decay_slope={
                    strategy: _opt(score_decay_slope(outcomes[strategy].scores, k))
                    for strategy in SCORE_ARMS
                    if strategy in outcomes
                },
                query_features=_numeric_features(features.get(query_id)),
            )
        )
    return rows


def signal_columns(
    row: SignalRow, *, registered_only: bool = True
) -> dict[str, float]:
    """Flatten a row into the numeric columns Gate 9.2 correlates.

    **Restricted to `REGISTERED_CANDIDATES` by default.** §2.4 states that a
    signal not listed there "may not enter the analysis as a candidate", and the
    implementation computes several quantities the preregistration never named --
    `overlap_at_k`, the non-registered `top1_agreement` pairs,
    `distinct_doc_ratio@hybrid`, `rank_correlation@bm25|hybrid`. Those remain
    computed on the row for inspection, but they are excluded here, and
    `tests/test_signals.py` asserts the exclusion.

    Set `registered_only=False` only to inspect the full row. A caller that wants
    the family must not have to remember an opt-out.

    `None` values are **dropped**, not coerced. A signal that could not be
    computed is missing data, and a 0.0 would be indistinguishable from a
    measured absence of relationship -- which is how `rank_correlation@bm25|dense`,
    undefined on 84 of 107 queries, would silently enter the correlation as a
    real number.
    """
    columns: dict[str, float] = {}

    for pair, value in row.jaccard.items():
        _put(columns, f"jaccard@{pair}", value)
    for pair, value in row.overlap_at_k.items():
        _put(columns, f"overlap_at_k@{pair}", value)
    for pair, value in row.top1_agreement.items():
        _put(columns, f"top1_agreement@{pair}", float(value))
    for pair, value in row.rank_correlation.items():
        _put(columns, f"rank_correlation@{pair}", value)
    for strategy, value in row.distinct_doc_ratio.items():
        _put(columns, f"distinct_doc_ratio@{strategy}", value)
    for strategy, value in row.score_gap_top1_top2.items():
        _put(columns, f"score_gap_top1_top2@{strategy}", value)
    for strategy, value in row.score_decay_slope.items():
        _put(columns, f"score_decay_slope@{strategy}", value)
    _put(columns, "union_concentration", row.union_concentration)

    for name, value in row.query_features.items():
        _put(columns, f"query_feature:{name}", value)

    if not registered_only:
        return columns
    allowed = {candidate.name for candidate in REGISTERED_CANDIDATES}
    return {name: value for name, value in columns.items() if name in allowed}


def defined_observations(
    rows: Sequence[SignalRow], signal_name: str
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Query IDs and values where `signal_name` is defined, in table order.

    **This is the inferential population for one signal** (docs/phases/phase-9.md
    §8.9). Inference runs over the observations on which that signal is defined;
    undefined observations are excluded *pairwise, for that signal only*. There is
    no complete-case restriction across the family, so a signal with fewer defined
    observations is analysed on its own subset and does not shrink any other.

    No imputation, no NULL-to-zero, and no silent redefinition: `rank_correlation`
    is undefined wherever bm25 and dense share fewer than two top-5 documents, and
    those queries simply are not observations of *this* signal. The same tuple
    feeds the correlation, the bootstrap resample and the permutation shuffle, so
    all three see exactly the same queries and the effective n is reported
    explicitly rather than inferred.
    """
    query_ids: list[str] = []
    values: list[float] = []
    for row in rows:
        columns = signal_columns(row)
        if signal_name in columns:
            query_ids.append(row.query_id)
            values.append(float(columns[signal_name]))
    return tuple(query_ids), tuple(values)


def _put(columns: dict[str, float], name: str, value: float | None) -> None:
    """Add a column unless the value is undefined."""
    if value is not None:
        columns[name] = float(value)


def to_record(row: SignalRow) -> dict[str, Any]:
    """A row as a JSON-ready mapping for the `signal_table.jsonl` export.

    Nested maps stay nested: the export is meant to be read, and `signal_columns`
    is the flattening analysis uses. The targets are emitted under `target_` so no
    consumer can pick one up as a feature by accident.
    """
    return {
        "query_id": row.query_id,
        "target_dispersion": row.dispersion,
        "target_dispersion_mrr": row.dispersion_mrr,
        "jaccard": row.jaccard,
        "overlap_at_k": row.overlap_at_k,
        "top1_agreement": row.top1_agreement,
        "rank_correlation": row.rank_correlation,
        "union_concentration": row.union_concentration,
        "distinct_doc_ratio": row.distinct_doc_ratio,
        "score_gap_top1_top2": row.score_gap_top1_top2,
        "score_decay_slope": row.score_decay_slope,
        "query_features": row.query_features,
    }


def rows_to_jsonl(rows: Sequence[SignalRow]) -> str:
    """Signal rows as a JSONL string with stable key order.

    `sort_keys=True` so the export is byte-identical across runs and a diff
    between two executions shows only real changes.
    """
    return "".join(json.dumps(to_record(row), sort_keys=True) + "\n" for row in rows)


def write_signal_table(rows: Sequence[SignalRow], path: Path | str) -> int:
    """Write `signal_table.jsonl` to `path`; return the number of rows written."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rows_to_jsonl(rows))
    return len(rows)


def read_signal_table(path: Path | str) -> list[dict[str, Any]]:
    """Read a `signal_table.jsonl` back as raw records.

    Returns dicts rather than `SignalRow` on purpose: the gate script needs the
    original nested shape to build its own family, and re-validating here would
    quietly drop any column a newer writer added -- a reader that silently
    discards fields is how an artifact stops being auditable.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
