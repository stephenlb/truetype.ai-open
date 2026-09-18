"""Service layer: TypeSafe-shaped requests in, TypeSafe-shaped answers out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .engine import DEFAULT_TEMPERATURE, EngineConfig, GemmaLetterEngine
from .questions import Question, QuestionAnswer, build_question, score_question
from .render import render_batch, render_example_prefix, render_target_block

SUPPORTED_MODELS = {
    "gemma-4-12b": "google/gemma-4-12B",
    "gemma-4-12b-base": "google/gemma-4-12B",
    "google/gemma-4-12B": "google/gemma-4-12B",
}


@dataclass
class BatchResult:
    model: str
    answers: dict[str, dict] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


class TypeSafeReplica:
    """Evaluates a state against typed questions using a single Gemma forward pass."""

    def __init__(self, engine: GemmaLetterEngine | None = None, model_name: str = "gemma-4-12b") -> None:
        self.engine = engine or GemmaLetterEngine(EngineConfig())
        self.model_name = model_name

    def system_one(
        self,
        *,
        state: str | dict | list,
        questions: dict[str, dict],
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        include_debug: bool = False,
    ) -> BatchResult:
        if not questions:
            raise ValueError("at least one question is required")

        state_text = state if isinstance(state, str) else json.dumps(state, indent=2)
        parsed: list[Question] = [
            build_question(qid, spec) for qid, spec in questions.items()
        ]

        rendered = render_batch(parsed, state_text)
        if self.engine.config.prefix_cache:
            # Each question has its own static few-shot prefix, so score them one at
            # a time against their own cached prefix. Sequential cached calls beat a
            # single uncached batch: the prefill saved per question is far larger
            # than anything batching buys on this hardware.
            distributions = []
            for question in parsed:
                distributions.extend(
                    self.engine.score_with_prefix(
                        render_example_prefix(question),
                        [render_target_block(question, state_text)],
                        temperature=temperature,
                    )
                )
        else:
            distributions = self.engine.score_batch(
                [item.text for item in rendered], temperature=temperature
            )

        answers: dict[str, dict] = {}
        debug: dict[str, dict] = {}
        for question, readout in zip(parsed, distributions):
            # Score over the *full* 26-letter readout so a question with more options
            # than top_k is never truncated; score_question renormalizes over the
            # question's own legal letters.
            answer: QuestionAnswer = score_question(
                question, readout.logits, temperature=temperature
            )
            answers[question.id] = answer.to_dict()
            if include_debug:
                debug[question.id] = {
                    "prompt": next(r.text for r in rendered if r.question_id == question.id),
                    "letter_logits": readout.logits,
                    "top_k": readout.top.top_k,
                    "top_k_letters": [letter for letter, _ in readout.top.ranked],
                }

        input_tokens = sum(len(r.text) for r in rendered) // 4
        result = BatchResult(
            model=model or self.model_name,
            answers=answers,
            usage={"input_tokens": input_tokens, "output_tokens": len(parsed)},
            debug=debug,
        )
        return result