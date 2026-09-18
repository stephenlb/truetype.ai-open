"""Format bake-off: find the prompt shape that gives Gemma 4 a clean letter readout."""

from __future__ import annotations

import itertools

from src.truetype.engine import EngineConfig, GemmaLetterEngine

POSITIVE = [
    "Please help ASAP, we are losing sales.",
    "Our production is down, please respond right away.",
    "I need this fixed immediately, it is blocking everything.",
    "URGENT: the payment system is failing right now.",
]
NEGATIVE = [
    "Just wanted to share the meeting notes from today.",
    "Here is the documentation link for reference.",
    "No rush on this, whenever you get a chance.",
    "The sky is blue and the grass is green.",
]

EXAMPLES_POS = EXAMPLES_POS = [
    ("I need this fixed immediately, it is blocking everything.", "Y"),
    ("Here is the documentation link for reference.", "N"),
    ("Our production is down, please respond right away.", "Y"),
    ("No rush on this, whenever you get a chance.", "N"),
]

OPTIONS_WITH_DESC = "Y. yes - The statement is true of the text.\nN. no - The statement is false or not supported by the text."
OPTIONS_BARE = "Y. yes\nN. no"
OPTIONS_LETTER_ONLY = "Y. Yes, the text expresses urgency.\nN. No, the text does not express urgency."


def fmt_a(text: str, question: str, options: str, examples: list, prefix: str = "Answer:") -> str:
    """Choices block repeated in every example."""
    blocks = []
    for ex_text, ex_letter in examples:
        blocks.append(
            f"Text: {ex_text}\nQuestion: {question}\nChoices:\n{options}\nAnswer: {ex_letter}"
        )
    blocks.append(f"Text: {text}\nQuestion: {question}\nChoices:\n{options}\n{prefix} ")
    return "\n\n".join(blocks)


def fmt_b(text: str, question: str, options: str, examples: list, prefix: str = "Answer:") -> str:
    """Choices listed once up front."""
    head = f"Answer each question with one letter.\n\nChoices:\n{options}\n"
    blocks = [head]
    for ex_text, ex_letter in examples:
        blocks.append(f"Text: {ex_text}\nQuestion: {question}\nAnswer: {ex_letter}")
    blocks.append(f"Text: {text}\nQuestion: {question}\n{prefix} ")
    return "\n\n".join(blocks)


def fmt_c(text: str, question: str, options: str, examples: list, prefix: str = "Answer:") -> str:
    """Compact, no 'Question:' label."""
    blocks = []
    for ex_text, ex_letter in examples:
        blocks.append(f"{ex_text}\n{question} {ex_letter}")
    blocks.append(f"{text}\n{question} {prefix} ")
    return "\n\n".join(blocks)


def fmt_d(text: str, question: str, options: str, examples: list, prefix: str = "Answer:") -> str:
    """Statement framing with explicit instruction line."""
    blocks = []
    for ex_text, ex_letter in examples:
        blocks.append(
            f"Text: {ex_text}\nIs it true that {question} Answer with Y or N.\nAnswer: {ex_letter}"
        )
    blocks.append(
        f"Text: {text}\nIs it true that {question} Answer with Y or N.\n{prefix} "
    )
    return "\n\n".join(blocks)


FORMATS = {"A": fmt_a, "B": fmt_b, "C": fmt_c, "D": fmt_d}


def evaluate(engine: GemmaLetterEngine, name: str, fn, options: str, n_examples: int) -> dict:
    examples = EXAMPLES_POS[:n_examples]
    q = "Does the text express urgency?"
    prompts, labels = [], []
    for text in POSITIVE:
        prompts.append(fn(text, q, options, examples))
        labels.append("Y")
    for text in NEGATIVE:
        prompts.append(fn(text, q, options, examples))
        labels.append("N")

    dists = engine.score_batch(prompts, top_k=2)
    correct = 0
    margins = []
    for d, label in zip(dists, labels):
        if d.top_letter == label:
            correct += 1
        probs = d.probabilities
        top = list(probs.items())
        y = probs.get("Y", 0.0)
        n = probs.get("N", 0.0)
        margins.append(y - n if label == "Y" else n - y)
    return {
        "format": name,
        "examples": n_examples,
        "accuracy": correct / len(labels),
        "mean_margin": sum(margins) / len(margins),
    }


def main() -> None:
    engine = GemmaLetterEngine(EngineConfig(top_k=2))
    engine.load()

    rows = []
    for name, options, n_examples in itertools.product(
        FORMATS, [OPTIONS_BARE, OPTIONS_LETTER_ONLY], [1, 2, 4]
    ):
        rows.append(evaluate(engine, format_name(name, options, n_examples), FORMATS[name], options, n_examples))

    rows.sort(key=lambda r: (r["accuracy"], r["mean_margin"]), reverse=True)
    print(f"{'accuracy':>9} {'margin':>8}  format")
    for r in rows:
        print(f"{r['accuracy']:>9.2f} {r['mean_margin']:>+8.3f}  {r['format']}")


def format_name(name: str, options: str, n_examples: int) -> str:
    tag = "bare" if options == OPTIONS_BARE else "verb"
    return f"{name}/{tag}/k={n_examples}"


if __name__ == "__main__":
    main()