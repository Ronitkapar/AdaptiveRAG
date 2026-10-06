"""
test_embedding_retry.py
----------------------
Regression tests for the AICredits embedding transport.

Two defects motivated these. First, the SDK's implicit `max_retries=2` retried
timed-out requests *inside* a single call, so one stalled request cost
3 x 45s + backoff ~= 136s of silence on top of the project's own retry loop.
Second, timeouts were classified with a `"timeout" in str(exc)` test that never
matched the SDK's "Request timed out.", so they were neither mapped to
`EmbeddingTimeoutError` nor retried. Everything here runs offline against real
`openai` exception objects; no request is ever issued.
"""

import httpx2
import openai
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from adaptive_rag.config.settings import Settings
from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel
from adaptive_rag.errors import EmbeddingAPIError, EmbeddingTimeoutError
from adaptive_rag.schemas import EmbeddingConfig

_REQUEST = httpx2.Request("POST", "https://api.aicredits.in/v1/embeddings")


def _status_error(status: int, message: str) -> APIStatusError:
    response = httpx2.Response(
        status, request=_REQUEST, headers={"x-request-id": "req_test"}, json={}
    )
    return APIStatusError(message, response=response, body={})


def _rate_limit_error() -> APIStatusError:
    """A 429 whose text does *not* say "rate limit", isolating the status path."""
    return _status_error(429, "Error code: 429")


def _model(max_retries: int = 4) -> AICreditsEmbeddingModel:
    # `normalize=False` keeps these assertions on the transport layer: the
    # vectors below come back exactly as the fake client returned them.
    settings = Settings(AICREDITS_API_KEY="test-key-not-real")
    config = EmbeddingConfig(max_retries=max_retries, normalize=False)
    return AICreditsEmbeddingModel(config=config, settings=settings)


def _response(vectors: list[list[float]]):
    class _Item:
        def __init__(self, embedding: list[float]):
            self.embedding = embedding

    class _Data:
        def __init__(self, embeddings: list[list[float]]):
            self.data = [_Item(v) for v in embeddings]

    return _Data(vectors)


def _install_fake_client(
    model: AICreditsEmbeddingModel, outcomes: list
) -> list[dict]:
    """Attach a recording fake client; returns the list of captured call kwargs.

    `outcomes` is consumed in order: an `Exception` is raised, anything else is
    treated as the list of vectors to return.
    """
    calls: list[dict] = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return _response(outcome)

    class _Client:
        def __init__(self):
            self.embeddings = _Embeddings()

    model._client = _Client()
    return calls


@pytest.fixture
def no_sleep(monkeypatch):
    """Keep the retry loop's backoff from actually sleeping."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


# --- Defect 1: the SDK must not retry behind our back ----------------------


def test_client_disables_sdk_retries_and_pins_timeout(monkeypatch):
    """`max_retries=0` makes the project's own loop the single retry authority."""
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    model = _model()
    model._client = None
    model._get_client()

    assert captured["timeout"] == 45.0
    assert captured["max_retries"] == 0


def test_sdk_retry_default_would_have_multiplied_the_timeout():
    """Pins the multiplier the fix removes, so the client's comment stays true."""
    from openai._constants import DEFAULT_MAX_RETRIES

    assert DEFAULT_MAX_RETRIES == 2
    worst_case_seconds = (DEFAULT_MAX_RETRIES + 1) * 45.0 + 0.5 + 1.0
    assert worst_case_seconds > 130.0


# --- Defect 2: classify by type, not by message text -----------------------


def test_sdk_timeout_text_does_not_contain_the_word_timeout():
    """The root cause: "Request timed out." never satisfied `"timeout" in exc`."""
    message = str(APITimeoutError(request=_REQUEST))
    assert message == "Request timed out."
    assert "timeout" not in message.lower()


def test_timeout_is_retried_and_then_succeeds(no_sleep):
    model = _model()
    calls = _install_fake_client(
        model,
        [
            APITimeoutError(request=_REQUEST),
            APITimeoutError(request=_REQUEST),
            [[0.1, 0.2, 0.3]],
        ],
    )

    assert model.embed_query("q") == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
    assert len(calls) == 3


def test_timeout_exhausting_retries_raises_embedding_timeout_error(no_sleep):
    model = _model(max_retries=2)
    calls = _install_fake_client(
        model, [APITimeoutError(request=_REQUEST), APITimeoutError(request=_REQUEST)]
    )

    with pytest.raises(EmbeddingTimeoutError) as excinfo:
        model.embed_query("q")

    assert not isinstance(excinfo.value, EmbeddingAPIError)
    assert "timed out" in str(excinfo.value)
    assert len(calls) == 2


def test_connection_error_keeps_its_own_handling(no_sleep):
    """`APIConnectionError` is not a timeout: it stays an `EmbeddingAPIError`."""
    model = _model()
    calls = _install_fake_client(model, [APIConnectionError(request=_REQUEST)])

    with pytest.raises(EmbeddingAPIError) as excinfo:
        model.embed_query("q")

    assert not isinstance(excinfo.value, EmbeddingTimeoutError)
    assert excinfo.value.status_code is None
    assert "Connection error." in str(excinfo.value)
    assert len(calls) == 1


def test_rate_limit_status_is_retried_then_raises_api_error(no_sleep):
    """A 429 keeps its `status_code` classification: retried, then API error."""
    model = _model(max_retries=3)
    calls = _install_fake_client(model, [_rate_limit_error()] * 3)

    with pytest.raises(EmbeddingAPIError) as excinfo:
        model.embed_query("q")

    assert not isinstance(excinfo.value, EmbeddingTimeoutError)
    assert excinfo.value.status_code == 429
    assert len(calls) == 3


def test_rate_limit_recovers_on_retry(no_sleep):
    model = _model()
    calls = _install_fake_client(model, [_rate_limit_error(), [[0.0, 1.0]]])

    assert model.embed_query("q") == pytest.approx([0.0, 1.0], rel=1e-5)
    assert len(calls) == 2


def test_server_error_is_retried(no_sleep):
    model = _model()
    calls = _install_fake_client(
        model, [_status_error(503, "Error code: 503"), [[0.5, 0.5]]]
    )

    assert model.embed_query("q") == pytest.approx([0.5, 0.5], rel=1e-5)
    assert len(calls) == 2


def test_non_retryable_status_fails_on_first_attempt(no_sleep):
    model = _model()
    calls = _install_fake_client(model, [_status_error(401, "Error code: 401")])

    with pytest.raises(EmbeddingAPIError) as excinfo:
        model.embed_query("q")

    assert excinfo.value.status_code == 401
    assert len(calls) == 1


def test_oversized_batch_still_splits_before_the_retry_decision(no_sleep):
    """The 413 split is unaffected: it still precedes retry classification."""
    model = _model()
    calls = _install_fake_client(
        model, [_status_error(413, "Error code: 413"), [[0.1]], [[0.2]]]
    )

    assert model.embed(["a", "b"]) == [[0.1], [0.2]]
    assert len(calls) == 3
