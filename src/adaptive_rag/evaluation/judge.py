"""
evaluation.judge
----------------
Groq-based LLM-as-judge for answer correctness, relevance, and faithfulness.
Deliberately independent of the generation module and cached to avoid repeat calls.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from adaptive_rag.config.hashing import compute_sha256
from adaptive_rag.config.paths import EVAL_CACHE_PATH
from adaptive_rag.config.settings import Settings, settings as default_settings
from adaptive_rag.errors import MissingCredentialError
from adaptive_rag.generation.prompts import JUDGE_PROMPT_V1

JUDGE_METRICS = ("correctness", "answer_relevance", "faithfulness")


class JudgeCache:
    """SQLite cache keyed by judge inputs so repeated runs make no extra calls."""

    def __init__(self, db_path: Path = EVAL_CACHE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT response FROM judge_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def put(self, cache_key: str, response: dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO judge_cache (cache_key, response) VALUES (?, ?)",
                (cache_key, json.dumps(response, sort_keys=True)),
            )
            conn.commit()

    def count(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT count(*) FROM judge_cache").fetchone()[0]


def judge_cache_key(
    prompt_version: str,
    model: str,
    query: str,
    generated_answer: str,
    reference_answer: str,
) -> str:
    """Deterministic cache key for a judge invocation."""
    payload = "|".join([prompt_version, model, query, generated_answer, reference_answer])
    return compute_sha256(payload)


class GroqLLMJudge:
    """Scores generated answers with a versioned rubric over Groq."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        prompt_version: str = "judge_prompt_v1",
        settings: Settings | None = None,
        cache: JudgeCache | None = None,
        use_cache: bool = True,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.settings = settings or default_settings
        self.cache = cache if cache is not None else (JudgeCache() if use_cache else None)
        self._client = None
        self.usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "calls": 0,
            "cache_hits": 0,
        }

    def _get_client(self):
        if self._client is None:
            api_key = self.settings.require_groq_key()
            from groq import Groq

            self._client = Groq(api_key=api_key, timeout=60.0)
        return self._client

    def judge(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
        retrieved_context: str = "",
    ) -> dict[str, Any]:
        """Return {correctness, answer_relevance, faithfulness, rationale} scores."""
        key = judge_cache_key(
            self.prompt_version, self.model, query, generated_answer, reference_answer
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.usage["cache_hits"] += 1
                return cached

        if not self.settings.GROQ_API_KEY:
            raise MissingCredentialError(
                "GROQ_API_KEY is not set; required for LLM-as-judge evaluation."
            )

        prompt = JUDGE_PROMPT_V1.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            retrieved_context=retrieved_context[:4000] if retrieved_context else "(not provided)",
        )

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict, objective evaluator. Respond with JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            # Reasoning models (gpt-oss) consume part of the completion budget on
            # internal reasoning before emitting the JSON body; at 512 harder
            # examples returned empty content and a provider-side 400. 4096 gives
            # ample headroom for reasoning spikes while the JSON body is small.
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        self.usage["calls"] += 1
        if response.usage:
            self.usage["input_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0

        parsed = self._parse_scores(response.choices[0].message.content or "")
        if self.cache is not None:
            self.cache.put(key, parsed)
        return parsed

    @staticmethod
    def _parse_scores(raw: str) -> dict[str, Any]:
        """Parse judge JSON, tolerating fenced blocks and partial output."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {
                "correctness": None,
                "answer_relevance": None,
                "faithfulness": None,
                "rationale": "judge_output_unparseable",
            }

        scores: dict[str, Any] = {}
        for name in JUDGE_METRICS:
            value = data.get(name)
            scores[name] = float(value) if isinstance(value, (int, float)) else None
        scores["rationale"] = str(data.get("rationale", ""))[:500]
        return scores
