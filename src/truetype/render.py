"""Render every question in a request to a base-completion prompt.

Prompt layout
-------------
Every block puts the answer-relevant scaffolding *first* and the text being
judged *last*, so a prompt is::

    Choices:
    Y. Yes, ...
    N. No, ...
    Question: <question>
    Text: <example text>
    Answer: Y

    ... more examples ...

    Choices:
    ...
    Question: <target question>
    Text: <the caller's state>
    Answer:

Putting the state last makes the engine's prefix KV cache effective:
everything above ``Text:`` depends only on the question, so it is byte-identical
across calls and can be cached once. That took a warm Doom decision from 357ms to
178ms (mean prompt tail 68 -> 20 tokens) with no accuracy change: 63/63 on the
50-case suite plus the Doom probe and 8 held-out noul cases.

Two example strategies, chosen by question type:

- Noul (Y/N) uses a fixed bank of demonstrations with their own questions.
  The predicate in the options ("Yes, the text requests a refund.") carries the
  task; the examples teach the letter slot. Their questions
  differ from the target question, so their Y/N labels stay truthful rather than
  being relabelled under a question they were not written for. This lifted
  accuracy from 3/6 to 6/6 in tests/example_transfer.py.

- Choice and Score use examples generated from the caller's criteria. Each
  option gets one synthetic demonstration mapping it to its letter, visited in a
  non-alphabetical order so the model cannot exploit A,B,C positional cues. This
  reached 10/10 on Choice and 11/12 on Score in tests/choice_score_bakeoff2.py and
  needs no hand-authoring.
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


def _header(question: Question, instructions: str | None = None) -> str:
    """Choices + question. Identical for every call on the same question."""
    return (
        f"Choices:\n{_render_options(question)}\n"
        f"Question: {instructions or question.instructions}\n"
    )


def _noul_examples(question: Question, count: int) -> list[str]:
    blocks = []
    for ex_text, ex_question, ex_letter in GENERIC_EXAMPLES[:count]:
        blocks.append(f"{_header(question, ex_question)}Text: {ex_text}\nAnswer: {ex_letter}")
    return blocks


def _criteria_roundtrip_examples(question: Question) -> list[str]:
    """One demonstration per option, built from the option's own description.

    Options use a non-alphabetical order that
    interleaves the scale (low, high, middle, ...) so no positional shortcut exists.
    """
    header = _header(question)
    blocks = []
    for index in _interleaved_order(len(question.options)):
        option = question.options[index]
        if question.type == "choice":
            detail = f" This example is about the topic of {option.label}."
        else:
            detail = " " + (option.description or option.label)
        blocks.append(
            f"{header}Text: An example of {option.label}:{detail}\nAnswer: {option.letter}"
        )
    return blocks


def _interleaved_order(n: int) -> list[int]:
    """Low, high, then remaining indices ascending. Breaks A,B,C positional cues."""
    if n <= 2:
        return list(range(n))
    order = [0, n - 1]
    order.extend(i for i in range(1, n - 1))
    return order


def render_example_prefix(
    question: Question,
    *,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> str:
    """The part of the prompt that depends only on the question, never on the state.

    This is what the engine keeps in a KV cache across calls: the few-shot examples
    plus the target question's own Choices/Question header. For a fixed question it
    is byte-identical every time, and it is ~90% of the prompt.
    """
    if question.type == "noul":
        blocks = _noul_examples(question, example_count)
    else:
        blocks = _criteria_roundtrip_examples(question)
    body = "\n\n".join(blocks) + "\n\n" if blocks else ""
    return body + _header(question)


def render_target_block(question: Question, state: str) -> str:
    """The state-dependent tail, ending at the ``Answer:`` slot.

    There is no trailing space after ``Answer:``. The letter tokens
    the engine reads are space-prefixed (``"▁Y"``), which is how the examples above
    tokenize (``['Answer', ':', '▁Y']``). Adding a trailing space here would emit a
    standalone ``'▁'`` token and strand the space, so the model would want a bare
    ``"Y"`` while the engine read ``"▁Y"``. This measured 0.0000 probability mass on
    the letters being scored, versus 0.9871 without the space.
    """
    return f"Text: {state}\nAnswer:"


def render_question(
    question: Question,
    state: str,
    *,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> RenderedPrompt:
    # Composed from the two halves so the cached-prefix path and the plain path
    # always produce the exact same prompt string.
    text = render_example_prefix(question, example_count=example_count) + render_target_block(
        question, state
    )
    return RenderedPrompt(question_id=question.id, text=text)


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
