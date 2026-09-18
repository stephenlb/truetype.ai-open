"""Prompt construction for the letter-logit classifier.

Base completions models need the answer to be the continuation of *text*, so every
question is rendered as a short transcript: a few worked examples, then the real
question, then a deliberate ``Answer:`` prefix whose next token is the letter.

Few-shot examples are not decoration here. A base model with zero examples will
happily answer "Y" to "is the sky blue?" because a bare ``Answer:`` slot carries no
evidence about what a yes/no letter means. A handful of consistent demonstrations
collapses that ambiguity and the letter distribution becomes usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .letters import LETTERS, LETTER_INDEX

DEFAULT_SYSTEM = "You are a careful classifier. Answer every question with exactly one letter."


@dataclass(frozen=True)
class LetterOption:
    letter: str
    label: str
    description: str | None = None

    def __post_init__(self) -> None:
        if self.letter not in LETTER_INDEX:
            raise ValueError(f"option letter must be A-Z, got {self.letter!r}")


@dataclass(frozen=True)
class FewShotExample:
    question: str
    letter: str
    state: str | None = None

    def __post_init__(self) -> None:
        if self.letter not in LETTER_INDEX:
            raise ValueError(f"example letter must be A-Z, got {self.letter!r}")


@dataclass
class PromptSpec:
    """Everything needed to render one question into a completion prompt."""

    instructions: str
    options: list[LetterOption]
    state: str = ""
    examples: list[FewShotExample] = field(default_factory=list)
    preamble: str = ""
    answer_prefix: str = "Answer:"


def _render_options(options: list[LetterOption]) -> str:
    lines = []
    for option in options:
        if option.description:
            lines.append(f"{option.letter}. {option.label} - {option.description}")
        else:
            lines.append(f"{option.letter}. {option.label}")
    return "\n".join(lines)


def _render_example(example: FewShotExample, options: list[LetterOption]) -> str:
    block = []
    if example.state:
        block.append(f"Text: {example.state}")
    block.append(f"Question: {example.question}")
    block.append("Choices:")
    block.append(_render_options(options))
    block.append("Answer: " + example.letter)
    return "\n".join(block)


def build_prompt(spec: PromptSpec) -> str:
    """Render a PromptSpec into a plain-text completion prompt ending at ``Answer:``."""
    if not spec.options:
        raise ValueError("at least one option is required")
    letters = [option.letter for option in spec.options]
    if len(set(letters)) != len(letters):
        raise ValueError(f"duplicate option letters: {letters}")

    blocks: list[str] = []
    if spec.preamble:
        blocks.append(spec.preamble)

    for example in spec.examples:
        blocks.append(_render_example(example, spec.options))

    question_block = []
    if spec.state:
        question_block.append(f"Text: {spec.state}")
    question_block.append(f"Question: {spec.instructions}")
    question_block.append("Choices:")
    question_block.append(_render_options(spec.options))
    blocks.append("\n".join(question_block))

    body = "\n\n".join(blocks)
    return f"{body}\n{spec.answer_prefix} "


def yes_no_options() -> list[LetterOption]:
    return [
        LetterOption("Y", "yes", "The statement is true of the text."),
        LetterOption("N", "no", "The statement is false or not supported by the text."),
    ]


def choice_options(criteria: dict[str, str | None]) -> list[LetterOption]:
    return [
        LetterOption(chr(ord("A") + i), name, description)
        for i, (name, description) in enumerate(criteria.items())
    ]


def score_options(criteria: list[str]) -> list[LetterOption]:
    if len(criteria) < 2:
        raise ValueError("a score needs at least two levels")
    if len(criteria) > len(LETTERS):
        raise ValueError(f"a score supports at most {len(LETTERS)} levels")
    return [
        LetterOption(chr(ord("A") + i), f"level {i}", description)
        for i, description in enumerate(criteria)
    ]