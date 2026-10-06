"""
evaluation.agreement
---------------------
Phase 9 cross-strategy **agreement** signals: how similarly two retrieval
strategies answer the same query.

Registered family in `docs/phases/phase-9.md` §2.4. Every function here is a
pure function of two ranked identifier lists, so the whole module is testable
without an index, a corpus, or a provider.

**These signals are informative, not usable.** Jaccard between `bm25` and
`dense` is the strongest registered candidate, and it is also the one a router
cannot consult before choosing between them: computing it *requires having run
both*. That is not a defect in the signal, it is a fact about where it sits in
the pipeline, and conflating the two is precisely how a mechanism finding turns
into a router that cannot be built. Gate 9.2 asks whether these signals explain
dispersion; Gate 9.3 separately asks whether any *deployable* proxy carries the
same information. Nothing in this module decides that question.

**`k` is frozen at 5**, per §2.5, and is a parameter only so the sensitivity
analysis can vary it explicitly. It is not tuned here, and no function picks a
"best" k for itself.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.errors import EvaluationError

AGREEMENT_VERSION = "phase9_agreement_v1"

#: Frozen cutoff, pre-registered in `docs/phases/phase-9.md` §2.5. Matches the
#: `top_k` the E1 arms were run at, so it is the depth the benchmark measures.
#: Note it is not the value that maximised the pilot correlation (k=3 was
#: marginally higher); freezing it anyway is the point, since selecting the
#: argmax of a sweep on the data being tested is selection.
DEFAULT_K: int = 5

#: Sensitivity values reported alongside the headline, never in place of it.
SENSITIVITY_K: tuple[int, ...] = (3, 10)


class PairAgreement(BaseModel):
    """Agreement between two strategies on one query, at one cutoff.

    Every field is a named quantity rather than a matrix of pairwise overlaps, so
    a reported correlation can always be traced to the exact function that
    produced it.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    system_a: str
    system_b: str
    k: int
    #: |A n B| / |A u B| over the top-k document sets.
    jaccard: float
    #: |A n B| / k -- the fraction of one arm's budget the other also filled.
    overlap_at_k: float
    #: Whether rank 1 is the same document on both arms.
    top1_agreement: bool
    #: Spearman correlation of the two arms' rank positions, restricted to
    #: documents either arm retrieved (rank correlation over a *shared*
    #: vocabulary; documents only one arm saw have no counterpart to correlate).
    rank_correlation: float | None = None
    #: Distinct documents in each arm's top-k, over k -- a per-arm concentration
    #: measure that needs no partner arm and is therefore deployable from a
    #: single strategy's output.
    distinct_doc_ratio_a: float = 0.0
    distinct_doc_ratio_b: float = 0.0


class UnionConcentration(BaseModel):
    """How *spread out* the two arms are jointly: `|union| / (|A| + |B|)`.

    Direction matters and is easy to state backwards, so it is pinned by a test.
    **Higher means the pair is more different.** Two arms returning disjoint
    documents score 1.0; two arms returning the same document score 0.5. The
    minimum is 0.5 rather than 0.0 because the denominator counts each arm's
    slots separately, so a shared document can never collapse the ratio past
    half.

    Reported alongside Jaccard rather than instead of it because the two
    disagree usefully: Jaccard's union normalisation already folds coverage into
    agreement, so a pair that is broad *and* agreed reads as less agreeing than a
    pair that is narrow and agreed. This ratio keeps the coverage effect visible
    instead of folding it away.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    k: int
    union_size: int
    #: Sum of per-arm distinct-document counts -- the denominator that makes this
    #: a concentration ratio rather than a bare count.
    total_slots: int
    union_concentration: float


def _prefix(ids: Sequence[str], k: int) -> list[str]:
    """The top-`k` identifiers, de-duplicated but order-preserving.

    De-duplication happens here rather than after truncation, so a strategy that
    returns the same document twice at two different ranks occupies one slot.
    Rank-based measures treat a repeated document as one item of evidence;
    counting it twice would let a single redundant arm look like broad coverage.
    """
    seen: set[str] = set()
    out: list[str] = []
    for identifier in ids[:k]:
        if identifier not in seen:
            seen.add(identifier)
            out.append(identifier)
    return out


def jaccard(a: Sequence[str], b: Sequence[str], k: int = DEFAULT_K) -> float:
    """Jaccard similarity of two arms' top-k document sets.

    Both empty scores **1.0**, not 0.0: two strategies that retrieved nothing
    agreed completely about there being nothing, and scoring that as maximal
    disagreement would make "both arms failed identically" look like the strongest
    possible evidence of strategy sensitivity -- exactly backwards. It is also
    unreachable in practice here, since every E1 arm returns 10 results on every
    query; the convention is stated so the edge case is deliberate if that
    changes.
    """
    set_a, set_b = set(_prefix(a, k)), set(_prefix(b, k))
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int = DEFAULT_K) -> float:
    """Fraction of `a`'s top-k documents that also appear in `b`'s.

    Asymmetric by construction, unlike Jaccard, and kept because it is the
    measure that matches the retrieval question: "how much of what this arm
    returned did the other arm also find?" Jaccard's normalisation by the union
    penalises a pair where one arm is narrow and the other broad even when the
    narrow arm's every result is confirmed.
    """
    prefix_a = _prefix(a, k)
    if not prefix_a:
        return 1.0
    return len(set(prefix_a) & set(_prefix(b, k))) / len(prefix_a)


def distinct_doc_ratio(ids: Sequence[str], k: int = DEFAULT_K) -> float:
    """Distinct documents in the top-k, over `k`.

    The one agreement-family signal computable from a **single** arm, and
    therefore the only kind that could ever be a pre-routing feature. Returns
    1.0 for an empty result: an arm that retrieved nothing is maximally
    undiversified, and `0/0` would raise.
    """
    prefix = _prefix(ids, k)
    if not prefix:
        return 1.0
    return len(prefix) / float(k)


def top1_agreement(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether both arms put the same document at rank 1.

    A deliberately coarse signal, retained because its coarseness is
    informative: it is the sharpest statement of "the two arms disagree about
    what the answer is", with no tolerance for anything below the top.
    """
    if not a or not b:
        return False
    return a[0] == b[0]


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rho with **mid-ranks**, or None if either side is constant.

    Mid-ranks are mandatory rather than stylistic. `recall_at_5` takes only four
    distinct values on this benchmark (92 of 107 queries carry a single relevant
    document), so ties are pervasive, and an average-rank-free implementation
    would misreport the correlation on exactly the data this phase studies.

    Returns None on a constant input because Pearson correlation against a
    zero-variance vector is undefined, and 0.0 would be a fabricated number that
    reads as "no relationship" rather than "cannot be computed".
    """
    n = len(x)
    if n != len(y) or n < 2:
        return None
    rank_x, rank_y = _midranks(x), _midranks(y)
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    den_x = sum((a - mean_x) ** 2 for a in rank_x)
    den_y = sum((b - mean_y) ** 2 for b in rank_y)
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / (den_x**0.5 * den_y**0.5)


def rank_correlation(
    a: Sequence[str], b: Sequence[str], k: int = DEFAULT_K
) -> float | None:
    """Rank correlation of two arms over the documents they share.

    Both arms' rank positions are correlated over their *intersection* only: a
    document one arm returned and the other did not has no counterpart rank to
    correlate, and padding the missing side with a synthetic rank would encode
    recall depth into what is supposed to be an ordering-agreement measure.
    Returns None below two shared documents, since a correlation over one point
    is not defined.
    """
    rank_a = {identifier: i for i, identifier in enumerate(_prefix(a, k), start=1)}
    rank_b = {identifier: i for i, identifier in enumerate(_prefix(b, k), start=1)}
    shared = sorted(set(rank_a) & set(rank_b))
    if len(shared) < 2:
        return None
    return spearman(
        [float(rank_a[c]) for c in shared], [float(rank_b[c]) for c in shared]
    )


def _midranks(values: Sequence[float]) -> list[float]:
    """Ranks 1..n with ties averaged.

    Sorting by value with a stable sort and assigning each tied run its mean
    position, so a value's rank depends only on the multiset of values -- not on
    the order they arrived in.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for position in range(i, j + 1):
            ranks[order[position]] = shared
        i = j + 1
    return ranks


def pair_agreement(
    query_id: str,
    system_a: str,
    ids_a: Sequence[str],
    system_b: str,
    ids_b: Sequence[str],
    k: int = DEFAULT_K,
) -> PairAgreement:
    """Every pairwise agreement signal for one query and one arm pair."""
    return PairAgreement(
        query_id=query_id,
        system_a=system_a,
        system_b=system_b,
        k=k,
        jaccard=round(jaccard(ids_a, ids_b, k), 6),
        overlap_at_k=round(overlap_at_k(ids_a, ids_b, k), 6),
        top1_agreement=top1_agreement(ids_a, ids_b),
        rank_correlation=rank_correlation(ids_a, ids_b, k),
        distinct_doc_ratio_a=round(distinct_doc_ratio(ids_a, k), 6),
        distinct_doc_ratio_b=round(distinct_doc_ratio(ids_b, k), 6),
    )


def union_concentration(
    query_id: str,
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    k: int = DEFAULT_K,
) -> UnionConcentration:
    """Joint spread of two arms: `|union| / (|A| + |B|)`.

    **Higher means more different**, which is the opposite of Jaccard and is
    pinned by tests because the two are easy to conflate. Disjoint arms score
    1.0; identical arms score 0.5. Raising rather than lowering on disagreement
    is what makes the column a diversity measure, so it must not be reported with
    an "agreement" label attached.
    """
    set_a, set_b = set(_prefix(ids_a, k)), set(_prefix(ids_b, k))
    union = set_a | set_b
    total = len(set_a) + len(set_b)
    if total == 0:
        raise EvaluationError("union_concentration requires at least one result")
    return UnionConcentration(
        query_id=query_id,
        k=k,
        union_size=len(union),
        total_slots=total,
        union_concentration=round(len(union) / total, 6),
    )


def registered_pairs(
    strategies: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Every unordered arm pair, enumerated once, in the given order.

    Exposed for internal use and diagnostics. Note this is *not* the Gate 9.2
    candidate family: `docs/phases/phase-9.md` §2.4 registers specific signals,
    and `evaluation.signals.REGISTERED_CANDIDATES` is the authority on which of
    these computations may enter the family. A helper being computable is not a
    licence to test it.
    """
    return tuple(
        (strategies[i], strategies[j])
        for i in range(len(strategies))
        for j in range(i + 1, len(strategies))
    )


# --------------------------------------------------------------------------
# Score geometry (H4), registered in docs/phases/phase-9.md 2.4.
#
# The preregistration names these two signals but does not give closed-form
# definitions, so the definitions below are supplied here as a documented
# amendment (phase-9.md section 8) and are frozen before Gate 9.2 runs. They are
# the standard forms; neither was chosen by inspecting a correlation.
#
# Both return `None` rather than a number when undefined. A fabricated 0.0 would
# be indistinguishable from a measured absence of relationship, which is exactly
# the distinction §2.6 asks to preserve.
# --------------------------------------------------------------------------


def _finite_scores(scores: Sequence[float | None], k: int) -> list[float]:
    """The top-`k` finite scores, in rank order.

    Non-finite entries are dropped rather than coerced. A `None` score means the
    arm did not produce one at that rank, which is missing data; substituting 0.0
    would invent a sharp decay out of an absence.
    """
    out: list[float] = []
    for value in list(scores)[:k]:
        if value is None:
            continue
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            continue
        out.append(numeric)
    return out


def score_gap_top1_top2(
    scores: Sequence[float | None], k: int = DEFAULT_K
) -> float | None:
    """Relative margin between rank 1 and rank 2: `(s1 - s2) / |s1|`.

    Normalising by `|s1|` rather than taking the raw difference is required here:
    BM25 scores run to ~27 while cosine similarities top out near 1.0, so a raw
    gap would put the two arms on incomparable scales and make the BM25 column
    dominate any pooled correlation purely through units.

    Returns None when fewer than two finite scores exist or when `s1 == 0` (the
    ratio is undefined, not zero).

    Direction: a **larger** gap means the top result is clearly separated, i.e.
    confident evidence, i.e. *less* strategy sensitivity. Expected association
    with dispersion is therefore **negative**.
    """
    finite = _finite_scores(scores, k)
    if len(finite) < 2:
        return None
    first = finite[0]
    if first == 0.0:
        return None
    return (first - finite[1]) / abs(first)


def score_decay_slope(scores: Sequence[float | None], k: int = DEFAULT_K) -> float | None:
    """Ordinary-least-squares slope of score on rank over the top-`k` results.

    Fitted as `score = a + b * rank`, so `b` is negative for any decaying score
    sequence and steeply negative for a sharp drop-off.

    Direction: a **flat** decay has `b` near 0, and the preregistration's stated
    rationale is "flat decay => more dispersion". Since flat (`b ≈ 0`) is a
    *higher* value than steep (`b ≪ 0`), a flat-decaying query associates with
    *more* dispersion, so the expected association is **positive**. This is the
    opposite of `score_gap_top1_top2` and of the jaccard family; the blanket
    "negative" annotation the preregistration carried for this row was wrong for
    this signal specifically and is corrected in section 8.

    Returns None below two finite scores, where a slope is not defined.
    """
    finite = _finite_scores(scores, k)
    n = len(finite)
    if n < 2:
        return None
    mean_rank = (n + 1) / 2.0
    mean_score = sum(finite) / n
    numerator = sum(
        (rank - mean_rank) * (score - mean_score)
        for rank, score in enumerate(finite, start=1)
    )
    denominator = sum((rank - mean_rank) ** 2 for rank in range(1, n + 1))
    if denominator == 0.0:
        return None
    return numerator / denominator