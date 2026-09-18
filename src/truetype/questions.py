"""Map Gemma letter logits onto TypeSafe's three question primitives.

The letter distribution is the model's raw output. Each primitive renormalizes it
over the letters that are legal answers for that question, so probabilities always
sum to 1 across exactly the options the caller defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .letters import LETTERS, LETTER_INDEX

# Fixed bank of question-agnostic demonstrations. Their predicates differ from the
# target question on purpose: they teach the letter-slot format, not the task.
GENERIC_EXAMPLES: tuple[tuple[str, str, str], ...] = (
    ("I need this fixed immediately, it is blocking everything.", "Does the text express urgency?", "Y"),
    ("Here is the documentation link for reference.", "Does the text express urgency?", "N"),
    ("The app crashes when I open the settings page.", "Does the text contain a bug report?", "Y"),
    ("Just wanted to share the meeting notes from today.", "Does the text contain a bug report?", "N"),
    ("Please send this to a human agent right now.", "Does the text ask for a human agent?", "Y"),
    ("No rush on this, whenever you get a chance.", "Does the text ask for a human agent?", "N"),
    ("URGENT: the payment system is failing right now.", "Does the text express urgency?", "Y"),
    ("The sky is blue and the grass is green.", "Does the text express urgency?", "N"),
)

DEFAULT_EXAMPLE_COUNT = 6


@dataclass(frozen=True)
class QuestionOption:
    letter: str
    label: str
    description: str | None = None


@dataclass
class Question:
    id: str
    type: str  # noul | choice | score
    instructions: str
    options: list[QuestionOption] = field(default_factory=list)
    criteria: object = None


@dataclass(frozen=True)
class QuestionAnswer:
    id: str
    type: str
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    noul: float | None = None
    score: float | None = None
    legend: dict[str, str] | None = None
    letters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload: dict = {"type": self.type}
        if self.type == "noul":
            payload["noul"] = round(self.noul or 0.0, 6)
        elif self.type == "choice":
            payload["choice"] = self.choice
            payload["probabilities"] = {k: round(v, 6) for k, v in self.probabilities.items()}
            payload["confidence"] = round(self.confidence or 0.0, 6)
        elif self.type == "score":
            payload["score"] = round(self.score or 0.0, 6)
            payload["legend"] = self.legend or {}
            payload["probabilities"] = {k: round(v, 6) for k, v in self.probabilities.items()}
            payload["confidence"] = round(self.confidence or 0.0, 6)
        return payload


def _softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def confidence_from_probabilities(probabilities: list[float]) -> float:
    """1.0 when all mass is on one option, 0.0 when uniform."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, 1.0 - entropy / math.log(n))


def _predicate_forms(instructions: str) -> tuple[str, str]:
    """Turn an instruction into affirmative/negative option text.

    "Does the text request a refund?" -> ("Yes, the text requests a refund.",
                                          "No, the text does not request a refund.")
    """
    text = instructions.strip().rstrip(".").strip()
    lowered = text.lower()

    if lowered.startswith("does the text "):
        body = text[len("Does the text ") :]
        affirmative = f"the text {body}"
    elif lowered.startswith("is the text "):
        body = text[len("Is the text ") :]
        affirmative = f"the text {body}"
    elif lowered.startswith("is this "):
        body = text[len("Is this ") :]
        affirmative = f"this {body}"
    elif lowered.startswith("does this "):
        body = text[len("Does this ") :]
        affirmative = f"this {body}"
    elif lowered.startswith("does the "):
        body = text[len("Does the ") :]
        affirmative = f"the {body}"
    elif lowered.startswith("is the "):
        body = text[len("Is the ") :]
        affirmative = f"the {body}"
    elif lowered.startswith("did the "):
        body = text[len("Did the ") :]
        affirmative = f"the {body}"
    elif lowered.startswith("can the "):
        body = text[len("Can the ") :]
        affirmative = f"the {body}"
    else:
        affirmative = text

    affirmative = affirmative.rstrip("?").strip()
    if affirmative.lower().startswith("the "):
        negative = "the " + affirmative[4:].split(" ", 1)[0] + " does not" + (
            " " + affirmative[4:].split(" ", 1)[1] if " " in affirmative[4:] else ""
        )
    else:
        negative = f"it is not the case that {affirmative}"
    return f"Yes, {affirmative}.", f"No, {negative}."


def build_question(qid: str, spec: dict) -> Question:
    """Parse one TypeSafe question spec into a letter-addressable Question."""
    qtype = spec.get("type")
    if qtype not in {"noul", "choice", "score"}:
        raise ValueError(f"question {qid!r} has unsupported type {qtype!r}")

    instructions = spec.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"question {qid!r} requires non-empty string instructions")

    criteria = spec.get("criteria")

    if qtype == "noul":
        options = []
        if isinstance(criteria, dict):
            yes_desc = criteria.get("true")
            no_desc = criteria.get("false")
        else:
            yes_desc = no_desc = None
        yes_text, no_text = _predicate_forms(instructions)
        options = [
            QuestionOption("Y", "yes", yes_desc or yes_text),
            QuestionOption("N", "no", no_desc or no_text),
        ]
        return Question(id=qid, type="noul", instructions=instructions, options=options, criteria=criteria)

    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"choice question {qid!r} requires a non-empty criteria map")
        if len(criteria) > len(LETTERS):
            raise ValueError(f"choice question {qid!r} supports at most 26 options")
        options = [
            QuestionOption(LETTERS[i], name, description)
            for i, (name, description) in enumerate(criteria.items())
        ]
        return Question(id=qid, type="choice", instructions=instructions, options=options, criteria=criteria)

    if not isinstance(criteria, list) or len(criteria) < 2:
        raise ValueError(f"score question {qid!r} requires an ordered list of at least two levels")
    if len(criteria) > len(LETTERS):
        raise ValueError(f"score question {qid!r} supports at most 26 levels")
    options = [
        QuestionOption(LETTERS[i], f"level {i}", level if isinstance(level, str) else str(level))
        for i, level in enumerate(criteria)
    ]
    return Question(id=qid, type="score", instructions=instructions, options=options, criteria=criteria)


def score_question(question: Question, letter_logits: dict[str, float], temperature: float = 1.0) -> QuestionAnswer:
    """Turn raw letter logits into a typed answer, renormalized over legal options."""
    option_letters = [option.letter for option in question.options]
    missing = [letter for letter in option_letters if letter not in letter_logits]
    if missing:
        raise ValueError(f"question {question.id!r} is missing logits for {missing}")

    logits = [letter_logits[letter] for letter in option_letters]
    probs = _softmax(logits, temperature)

    if question.type == "noul":
        by_label = {option.label: p for option, p in zip(question.options, probs)}
        noul = by_label.get("yes", 0.0)
        return QuestionAnswer(
            id=question.id,
            type="noul",
            noul=noul,
            probabilities={option.label: p for option, p in zip(question.options, probs)},
            letters={letter: p for letter, p in zip(option_letters, probs)},
        )

    if question.type == "choice":
        pairs = sorted(zip(question.options, probs), key=lambda op: op[1], reverse=True)
        top = pairs[0]
        return QuestionAnswer(
            id=question.id,
            type="choice",
            choice=top[0].label,
            probabilities={option.label: p for option, p in pairs},
            confidence=confidence_from_probabilities(probs),
            letters={letter: p for letter, p in zip(option_letters, probs)},
        )

    score = sum(i * p for i, p in enumerate(probs))
    return QuestionAnswer(
        id=question.id,
        type="score",
        score=score,
        probabilities={str(i): p for i, p in enumerate(probs)},
        legend={str(i): option.description or "" for i, option in enumerate(question.options)},
        confidence=confidence_from_probabilities(probs),
        letters={letter: p for letter, p in zip(option_letters, probs)},
    )