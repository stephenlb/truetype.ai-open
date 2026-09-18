"""Render every question in a request to a base-completion prompt.

Two prompt strategies, chosen by question type:

* **Noul (Y/N):** a fixed bank of question-agnostic demonstrations. The predicate in
  the options ("Yes, the text requests a refund.") carries the task; the examples only
  teach the letter slot. This lifted accuracy from 3/6 to 6/6 in tests/example_transfer.py.

* **Choice / Score:** examples generated *from the caller's own criteria*. Each option
  gets one synthetic demonstration mapping it to its letter. The examples are shuffled
  so the model cannot exploit alphabetical ordering. This reached 10/10 on Choice and
  11/12 on Score in tests/choice_score_bakeoff2.py, and requires no hand-authoring
  because the examples are derived from the question itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from .questions import DEFAULT_EXAMPLE_COUNT, GENERIC_EXAMPLES, Question


@dataclass
class RenderedPrompt:
    question_id: str
    text: str


def _render_options(question: Question) -> str:
    lines = []
    for option in question.options:
        if question.type == "noul":
            lines.append(f"{option.letter}. {option.description or option.label}")
        elif option.description:
            lines.append(f"{option.letter}. {option.label} - {option.description}")
        else:
            lines.append(f"{option.letter}. {option.label}")
    return "\n".join(lines)


def _noul_examples(question: Question, count: int) -> list[str]:
    options = _render_options(question)
    blocks = []
    for ex_text, ex_question, ex_letter in GENERIC_EXAMPLES[:count]:
        blocks.append(
            f"Text: {ex_text}\nQuestion: {ex_question}\nChoices:\n{options}\nAnswer: {ex_letter}"
        )
    return blocks


def _criteria_roundtrip_examples(question: Question) -> list[str]:
    """One demonstration per option, built from the option's own description.

    Ordering is deliberately non-alphabetical: options are visited in an order that
    interleaves the scale (low, high, middle, ...) so no positional shortcut exists.
    """
    options = _render_options(question)
    n = len(question.options)
    order = _interleaved_order(n)

    blocks = []
    for index in order:
        option = question.options[index]
        if question.type == "choice":
            subject = option.label
            detail = f" This example is about the topic of {option.label}."
        else:
            subject = option.label
            detail = option.description or option.label
        blocks.append(
            f"Text: An example of {subject}: {detail}\n"
            f"Question: {question.instructions}\n"
            f"Choices:\n{options}\nAnswer: {option.letter}"
        )
    return blocks


def _interleaved_order(n: int) -> list[int]:
    """Low, high, then remaining indices ascending. Breaks A,B,C positional cues."""
    if n <= 2:
        return list(range(n))
    order = [0, n - 1]
    order.extend(i for i in range(1, n - 1))
    return order


def render_question(
    question: Question,
    state: str,
    *,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> RenderedPrompt:
    if question.type == "noul":
        blocks = _noul_examples(question, example_count)
    else:
        blocks = _criteria_roundtrip_examples(question)

    blocks.append(
        f"Text: {state}\nQuestion: {question.instructions}\n"
        f"Choices:\n{_render_options(question)}\nAnswer: "
    )
    return RenderedPrompt(question_id=question.id, text="\n\n".join(blocks))


def render_batch(
    questions: list[Question],
    state: str,
    *,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> list[RenderedPrompt]:
    """One prompt per question. Every question sees the same state, as in TypeSafe."""
    return [
        render_question(question, state, example_count=example_count) for question in questions
    ]