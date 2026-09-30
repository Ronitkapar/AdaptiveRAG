"""
reranking.onnx_backend
----------------------
ONNX Runtime backend for a pre-exported cross-encoder.

This is the only module that performs model inference. It holds no index and no
retriever: it is handed a query and passages and returns scores. The artifact is
fetched on first use (not at import time) so that importing this module never
touches the network, and is then reused from the local cache directory.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from adaptive_rag.errors import RerankerModelError, RerankerUnavailableError

logger = logging.getLogger(__name__)

MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"

# Files the artifact must provide for the session to be constructible.
REQUIRED_FILES = ("onnx/model.onnx", "tokenizer.json", "config.json")

# Preferred execution providers per requested device, most specific first.
_PROVIDERS = {
    "cpu": ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
}


class OnnxCrossEncoderReranker:
    """Batched cross-encoder scorer running on ONNX Runtime."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        model_revision: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        model_dir: Path | None = None,
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.version = "onnx_cross_encoder_v1"
        self.device = "cpu"
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._num_labels = 1
        self._relevant_label_index = 0
        self._resolved_revision: str | None = None

    @property
    def is_loaded(self) -> bool:
        """True once the session exists; loading is deferred to first use."""
        return self._session is not None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        session, tokenizer, _config = self._load_artifacts()
        self._session = session
        self._tokenizer = tokenizer

    def _resolve_model_path(self) -> Path:
        """Fetch the artifact into the local cache directory if it is absent."""
        if self.model_dir is None:
            raise RerankerModelError(
                f"no local directory configured for reranker artifact {self.model_id!r}"
            )
        target = self.model_dir / self.model_id.replace("/", "--")
        missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
        if missing:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RerankerModelError(
                    "huggingface-hub is required to download the reranker artifact"
                ) from exc
            logger.info("Downloading reranker artifact %s to %s", self.model_id, target)
            try:
                snapshot_download(
                    repo_id=self.model_id,
                    revision=self.model_revision,
                    local_dir=str(target),
                )
            except RerankerModelError:
                raise
            except Exception as exc:
                # A missing repo, a 401/404, or a network fault must surface as the
                # project's own typed error rather than leaking a transport-specific
                # exception through the Reranker contract.
                raise RerankerModelError(
                    f"failed to download reranker artifact {self.model_id!r} "
                    f"into {target}: {type(exc).__name__}: {exc}"
                ) from exc
            missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
            if missing:
                raise RerankerModelError(
                    f"reranker artifact {self.model_id!r} is missing: "
                    + ", ".join(str(target / name) for name in missing)
                )
        self._resolved_revision = self.model_revision
        return target

    def _load_artifacts(self) -> tuple[Any, Any, dict[str, Any]]:
        import onnxruntime
        from tokenizers import Tokenizer

        model_path = self._resolve_model_path()

        tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding()

        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        self._configure_output(config)

        session = onnxruntime.InferenceSession(
            str(model_path / "onnx" / "model.onnx"),
            providers=list(self._resolve_providers(onnxruntime)),
        )
        self._input_names = tuple(i.name for i in session.get_inputs())
        self._output_names = tuple(o.name for o in session.get_outputs())
        return session, tokenizer, config

    def _resolve_providers(self, onnxruntime_module: Any) -> tuple[str, ...]:
        available = set(onnxruntime_module.get_available_providers())
        requested = self.requested_device
        if requested == "auto":
            requested = "cuda" if "CUDAExecutionProvider" in available else "cpu"
            logger.info("Reranker device 'auto' resolved to '%s'", requested)
        providers = _PROVIDERS.get(requested, ())
        if providers[0] not in available:
            raise RerankerUnavailableError(
                f"reranker device {requested!r} unavailable; "
                f"available providers: {sorted(available)}"
            )
        self.device = requested
        return providers

    def _configure_output(self, config: dict[str, Any]) -> None:
        """Read the head shape from the model config and pick a scoring rule."""
        id2label = config.get("id2label") or {}
        self._num_labels = max(len(id2label), 1)
        # For a two-logit head, the positive class is conventionally label_1 or
        # "relevant"; fall back to the highest numeric label index.
        index = 1 if self._num_labels == 2 else 0
        for key, label in id2label.items():
            if self._num_labels == 2 and str(label).lower() in ("label_1", "relevant"):
                try:
                    index = int(key)
                except ValueError:
                    pass
        self._relevant_label_index = index

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each (query, passage) pair independently, in `batch_size` chunks.

        Batching cannot change the result: no chunk attends to another chunk, so
        output is batch-size invariant.
        """
        self._ensure_loaded()
        scores: list[float] = []
        total = len(passages)
        for start in range(0, total, self.batch_size):
            chunk = list(passages[start : start + self.batch_size])
            scores.extend(self._score_batch(query, chunk))
        if len(scores) != total:
            raise RerankerModelError(
                f"reranker produced {len(scores)} scores for {total} passages"
            )
        return scores

    def _score_batch(self, query: str, passages: Sequence[str]) -> list[float]:
        encodings = self._tokenizer.encode_batch([(query, p) for p in passages])
        input_ids = [enc.ids for enc in encodings]
        attention_mask = [enc.attention_mask for enc in encodings]
        feed = self._build_feed(input_ids, attention_mask)
        outputs = self._session.run(list(self._output_names), feed)
        logits = outputs[0]
        if self._num_labels == 1:
            return [float(row[0]) for row in logits]
        relevant = self._relevant_label_index
        return [self._softmax(row)[relevant] for row in logits]

    def _build_feed(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> dict[str, Any]:
        feed: dict[str, Any] = {}
        mapping = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": [[0] * len(ids) for ids in input_ids],
        }
        for name in self._input_names:
            if name in mapping:
                feed[name] = mapping[name]
        return feed

    @staticmethod
    def _softmax(row: Sequence[float]) -> list[float]:
        highest = max(row)
        exps = [math.exp(value - highest) for value in row]
        total = sum(exps)
        return [value / total for value in exps] if total else [0.0 for _ in row]