"""
reranking.base
--------------
Second-stage scoring protocol.

A `Reranker` turns one query and an ordered list of passages into one relevance
score per passage. It is deliberately isolated: it holds no index, no retriever,
and no corpus access, and it never decides *which* passages it is asked about —
that selection is the first stage's job.
"""

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Protocol for second-stage passage scorers."""

    version: str
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage, in input order.

        The returned list must have exactly the same length as `passages`, and
        `scores[i]` must correspond to `passages[i]`. A length mismatch is a
        contract violation, never a silent truncation.

        Scores are model-specific ranking signals (for a cross-encoder, a raw
        logit). They are not calibrated probabilities and must not be averaged,
        normalized, or blended with first-stage scores.
        """
        ...