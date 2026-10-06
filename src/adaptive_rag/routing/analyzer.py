"""
routing.analyzer
----------------
Phase 6 query analyzer: a lightweight, deterministic feature extractor.

Design constraints, all deliberate:

* **No model, no network, no index, no corpus.** Analysis is a pure function of the
  query string, so it costs microseconds and works offline. Deciding *which
  retriever to run* must never require an LLM call or an embedding round trip --
  that would defeat the efficiency goal the whole system exists to pursue.
* **Interpretable.** Every feature is a named, inspectable quantity, so a routing
  decision can be explained after the fact instead of being a black box.
* **Reusable.** The features are the router's only input, which is what lets a
  learned router later consume exactly the same signals.

This is deliberately *not* linguistic analysis: no stemming, no part-of-speech
tagging, no dependency parsing. It counts surface cues with frozen lexicons. The
analyzer version is recorded on every `QueryFeatures`, so a feature change is
always visible in a run's trace.
"""

import re
import unicodedata

from adaptive_rag.errors import InvalidQueryError
from adaptive_rag.schemas import QueryFeatures

ANALYZER_VERSION = "analyzer_v1"

# Frozen stopword list. Kept small and explicit: this analyzer measures how much
# *content* a query carries, so a large auto-generated stoplist would add
# maintenance burden without changing routing behaviour materially.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am do does did doing done
    have has had having of in on at to for from by with without about into over under
    it its as so such not no nor too very can will just
    what which who whom whose when where why how
    i me my we our you your he she they them their
    """.split()
)

# Surface forms that mark a technical, domain-specific lookup. Matched as whole
# phrases against the normalized (casefolded) query text.
TECHNICAL_TERMS = (
    "bm25", "okapi", "tf-idf", "tfidf", "idf", "dpr", "realm", "retro", "splade",
    "colbert", "contriever", "pyserini", "faiss", "qdrant", "hnsw", "ann",
    "bi-encoder", "cross-encoder", "dense retrieval", "dense passage retrieval",
    "sparse retrieval", "lexical retrieval", "hybrid retrieval", "query expansion",
    "reciprocal rank fusion", "rank fusion", "rerank", "reranking", "reranker",
    "inverted index", "tokenizer", "embedding", "embeddings", "latent variable",
    "masked language model", "masked language modeling", "k1", "b parameter",
    "latency", "throughput",
)

# Phrases indicating an abstract, explanatory, conceptual question -- the kind of
# query a dense encoder is expected to serve better than lexical overlap.
SEMANTIC_CUES = (
    "why", "rationale", "intuition", "motivation", "purpose", "benefit",
    "advantage", "role", "impact", "effect", "explain", "concept", "conceptual",
    "approach", "mechanism", "principle", "implication", "significance",
    "underlying", "reason", "how does", "how do", "how can", "what is the role",
)

# Tokens that mark an explicit comparison between two things.
COMPARISON_CUES = (
    "compare", "comparison", "versus", "vs", "vs.", "difference", "differences",
    "differ", "better", "worse", "outperform", "outperforms", "relative", "than",
    "advantage", "disadvantage", "tradeoff", "trade-off", "contrast", "stronger",
    "cheaper",
)

# Connectives that signal more than one concept in a single query.
CONJUNCTION_CUES = (
    " and ", " both ", " or ", " as well as ", " along with ", " versus ", " vs ",
)
_QUOTED_RE = re.compile(r"[\"'\u201c\u2018][^\"'\u201c\u201d\u2018\u2019]{1,60}[\"'\u201d\u2019]")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9\-]{1,}$")
_CAMEL_RE = re.compile(r"^[a-z][A-Z]")
_IDENTIFIER_RE = re.compile(r"^(?=.*\d)[A-Za-z0-9_]|[A-Za-z0-9_]*[_.\-][A-Za-z0-9_\-]*$")
_QUESTION_WORDS = ("what", "how", "why", "which", "who", "when", "where")
_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(what is|what are|what was|what does|what do|how do|how does|how is|how are"
    r"|why is|why are|why does|why do|which|who|when|where)\b",
    re.IGNORECASE,
)
_DEFINE_RE = re.compile(
    r"^\s*(define|definition of|what is meant by|what does .* mean|what is .* called"
    r"|what are .* called)\b",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|differences|differ|tradeoff|trade-off)\b",
    re.IGNORECASE,
)
# Short cues must match a whole token, so `vs` never fires inside `adversarial`
# and `than` never fires inside `thanksgiving`.
_WORD_BOUNDARY_CUES = frozenset(
    {"vs", "vs.", "than", "relative", "differ", "better", "worse"}
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.]*")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Casefold and collapse whitespace, keeping punctuation for shape tests."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).strip().lower())


def _words(text: str) -> list[str]:
    """Surface word tokens, order preserved."""
    return _WORD_RE.findall(text)


def _cap(value: float) -> float:
    """Clamp a value to [0, 1]."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _cue_present(normalized: str, cue: str) -> bool:
    """Whole-token match for short cues, substring match for multi-word phrases."""
    if cue in _WORD_BOUNDARY_CUES:
        return re.search(rf"\b{re.escape(cue)}\b", normalized) is not None
    return cue in normalized


def _classify_question(query: str, normalized: str) -> str:
    """Classify the interrogative frame with fixed, ordered rules.

    First match wins, so the order encodes precedence rather than frequency: an
    explicit comparison beats a leading interrogative, and an explicit definition
    frame beats a bare `what`.
    """
    if _COMPARE_RE.search(normalized):
        return "compare"
    if _DEFINE_RE.match(query.strip()):
        return "define"
    prefix = _QUESTION_PREFIX_RE.match(query.strip())
    if prefix:
        word = prefix.group(1).lower().split()[0]
        if word in _QUESTION_WORDS:
            return word
    first = _words(normalized)[:1]
    if first and first[0] in _QUESTION_WORDS:
        return first[0]
    return "other"


class QueryFeatureAnalyzer:
    """Extracts routing features from a query using fixed, inspectable cues.

    The analyzer is stateless and thread-safe: it holds only frozen constants, so
    repeated analysis of the same query is byte-identical.
    """

    version = ANALYZER_VERSION

    def analyze(self, query: str) -> QueryFeatures:
        """Compute `QueryFeatures` for a raw query string.

        Raises `InvalidQueryError` for an empty or whitespace-only query, matching
        every retriever's precondition, so the adaptive layer rejects bad input
        before any index is touched.
        """
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")

        normalized = _normalize(query)
        words = _words(normalized)
        n_words = len(words)
        if n_words == 0:
            raise InvalidQueryError("Query contains no analyzable terms")

        # Lexical shape --------------------------------------------------------
        content_terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
        # A degenerate query ("q") still has a content term, so an exact-token
        # lookup is never scored as an empty query.
        if not content_terms:
            content_terms = words[:1]
        n_content = len(content_terms)
        lexical_density = n_content / n_words

        # Exact-term / entity indicators ---------------------------------------
        # Counted over ORIGINAL-case words, because normalization would erase the
        # acronym and CamelCase shapes this signal depends on.
        entity_count = len(_QUOTED_RE.findall(query))
        for word in _words(query):
            if (
                _ACRONYM_RE.match(word)
                or _CAMEL_RE.match(word)
                or _IDENTIFIER_RE.match(word)
            ):
                entity_count += 1
        entity_ratio = entity_count / n_words

        # Domain terminology and conceptual phrasing ---------------------------
        technical_count = sum(1 for term in TECHNICAL_TERMS if term in normalized)
        technical_ratio = technical_count / n_content if n_content else 0.0

        semantic_count = sum(1 for cue in SEMANTIC_CUES if cue in normalized)
        semantic_ratio = semantic_count / n_content if n_content else 0.0

        comparison_count = sum(
            1 for cue in COMPARISON_CUES if _cue_present(normalized, cue)
        )
        comparison_ratio = comparison_count / n_content if n_content else 0.0

        # Question type --------------------------------------------------------
        question_type = _classify_question(query, normalized)

        # Multi-concept structure ---------------------------------------------
        padded = f" {normalized} "
        concept_count = 1 + sum(
            1 for cue in CONJUNCTION_CUES if cue in padded
        )
        multi_concept = concept_count > 1

        # Complexity ------------------------------------------------------------
        # A bounded [0, 1] blend of the surface cues that make a query hard to
        # satisfy from a single cheap pass. Each term is capped at 1 so no single
        # cue can dominate, and the weights sum to 1.
        complexity_score = _cap(
            0.30 * _cap(n_content / 12.0)
            + 0.20 * _cap((concept_count - 1) / 2.0)
            + 0.20 * _cap(comparison_ratio)
            + 0.15 * _cap(entity_ratio)
            + 0.15 * _cap(semantic_ratio)
        )

        return QueryFeatures(
            query_length_words=n_words,
            query_length_chars=len(query),
            content_terms=content_terms,
            content_term_count=n_content,
            lexical_density=round(lexical_density, 6),
            entity_indicator_count=entity_count,
            entity_ratio=round(entity_ratio, 6),
            technical_term_count=technical_count,
            technical_ratio=round(technical_ratio, 6),
            semantic_indicator_count=semantic_count,
            semantic_ratio=round(semantic_ratio, 6),
            question_type=question_type,
            concept_count=concept_count,
            multi_concept=multi_concept,
            comparison_indicator_count=comparison_count,
            complexity_score=round(complexity_score, 6),
            analyzer_version=self.version,
        )