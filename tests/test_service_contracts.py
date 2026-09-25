"""Model-free contracts for the service and HTTP boundary.

The live-model suite verifies quality.  These tests use a recording engine so
the request plumbing, cache routing, and response shape are covered without
downloading model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi import HTTPException

from src.truetype import api
from src.truetype.letters import LetterReadout, distribution_from_letter_logits
from src.truetype.render import render_question
from src.truetype.service import TypeSafeReplica


def _readout(**preferred: float) -> LetterReadout:
    logits = {chr(ord("A") + index): float(index) for index in range(26)}
    logits.update(preferred)
    return LetterReadout(
        logits=logits,
        top=distribution_from_letter_logits(logits, top_k=2),
        letter_mass=0.9,
    )


@dataclass
class _Config:
    prefix_cache: bool = True


class RecordingEngine:
    """Small stand-in that exposes only the engine surface the service uses."""

    def __init__(self, *, prefix_cache: bool = True) -> None:
        self.config = _Config(prefix_cache=prefix_cache)
        self.capacity_requests: list[int] = []
        self.batch_calls: list[tuple[list[str], float]] = []
        self.prefix_calls: list[tuple[str, list[str], float | None]] = []

    def ensure_prefix_capacity(self, count: int) -> int:
        self.capacity_requests.append(count)
        return count

    def score_batch(self, prompts: list[str], *, temperature: float) -> list[LetterReadout]:
        self.batch_calls.append((prompts, temperature))
        return [_readout(A=4.0, B=1.0, Y=3.0, N=0.0) for _ in prompts]

    def score_with_prefix(
        self, prefix: str, suffixes: list[str], *, temperature: float | None = None
    ) -> list[LetterReadout]:
        self.prefix_calls.append((prefix, suffixes, temperature))
        return [_readout(A=4.0, B=1.0, Y=3.0, N=0.0) for _ in suffixes]


QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "charges", "shipping": "delivery"},
    },
}


def test_system_one_uses_cached_prefixes_and_preserves_question_order():
    engine = RecordingEngine(prefix_cache=True)
    result = TypeSafeReplica(engine).system_one(
        state="duplicate charge", questions=QUESTIONS, model="requested-model", temperature=0.5,
        include_debug=True,
    )

    assert engine.capacity_requests == [2]
    assert engine.batch_calls == []
    assert len(engine.prefix_calls) == 2
    assert [suffixes for _, suffixes, _ in engine.prefix_calls] == [
        ["Text: duplicate charge\nAnswer:"],
        ["Text: duplicate charge\nAnswer:"],
    ]
    assert all(temperature == 0.5 for _, _, temperature in engine.prefix_calls)
    assert result.model == "requested-model"
    assert list(result.answers) == ["refund", "team"]
    assert result.answers["refund"] == {"type": "noul", "noul": pytest.approx(0.997527, abs=1e-6)}
    assert result.answers["team"]["choice"] == "billing"
    assert set(result.debug) == set(QUESTIONS)
    assert result.usage["output_tokens"] == 2


def test_system_one_without_prefix_cache_scores_complete_prompts():
    engine = RecordingEngine(prefix_cache=False)
    state = {"order": 104, "reason": "duplicate charge"}
    result = TypeSafeReplica(engine).system_one(state=state, questions=QUESTIONS, temperature=0.25)

    assert engine.capacity_requests == []
    assert engine.prefix_calls == []
    prompts, temperature = engine.batch_calls[0]
    assert temperature == 0.25
    assert len(prompts) == 2
    assert '"order": 104' in prompts[0]
    assert prompts[0].endswith("Answer:")
    assert result.debug == {}


def test_warm_populates_each_prefix_and_empty_requests_are_free():
    engine = RecordingEngine(prefix_cache=True)
    service = TypeSafeReplica(engine)

    assert service.warm({}) == 0.0
    assert engine.prefix_calls == []

    elapsed = service.warm(QUESTIONS)
    assert elapsed >= 0.0
    assert engine.capacity_requests == [2]
    assert [suffixes for _, suffixes, _ in engine.prefix_calls] == [
        ["Text: warmup\nAnswer:"],
        ["Text: warmup\nAnswer:"],
    ]


def test_service_rejects_an_empty_question_map_before_scoring():
    engine = RecordingEngine()
    with pytest.raises(ValueError, match="at least one question"):
        TypeSafeReplica(engine).system_one(state="anything", questions={})
    assert engine.prefix_calls == []
    assert engine.batch_calls == []


def test_service_prompts_match_the_public_renderer_when_cache_is_disabled():
    engine = RecordingEngine(prefix_cache=False)
    service = TypeSafeReplica(engine)
    spec = {"only": QUESTIONS["refund"]}
    service.system_one(state="refund please", questions=spec)

    from src.truetype.questions import build_question

    expected = render_question(build_question("only", spec["only"]), "refund please").text
    assert engine.batch_calls[0][0] == [expected]


def test_health_reports_starting_without_a_lifespan_service(monkeypatch):
    monkeypatch.setattr(api, "_state", {})
    assert api.health() == {"status": "starting", "model_loaded": False}


def test_system_one_returns_503_until_the_lifespan_registers_a_service(monkeypatch):
    monkeypatch.setattr(api, "_state", {})
    request = api.SystemOneRequest(state="hello", questions=QUESTIONS)

    with pytest.raises(HTTPException) as raised:
        api.system_one(request)
    assert raised.value.status_code == 503


def test_api_handler_translates_service_value_errors_to_422(monkeypatch):
    class RejectingService:
        def system_one(self, **_kwargs):
            raise ValueError("bad question")

    monkeypatch.setattr(api, "_state", {"service": RejectingService()})
    request = api.SystemOneRequest(state="hello", questions=QUESTIONS)

    with pytest.raises(HTTPException) as raised:
        api.system_one(request)
    assert raised.value.status_code == 422
    assert raised.value.detail == "bad question"
