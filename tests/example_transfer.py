"""Can generic, question-agnostic few-shot examples calibrate an arbitrary question?

The TypeSafe API accepts arbitrary `instructions`, so we cannot hand-author examples
per question. This probes whether a fixed bank of examples, whose questions differ
from the target question, still produces a usable letter distribution.
"""

from __future__ import annotations

from src.truetype.engine import EngineConfig, GemmaLetterEngine

# --- target task: refund detection, examples below are about a DIFFERENT question
TARGET_QUESTION = "Does the text request a refund?"

POS = [
    "I was charged twice for order A-104. Please refund the duplicate.",
    "Can I get my money back for the cancelled flight?",
    "I want a full refund, this product is not what I ordered.",
]
NEG = [
    "Can I swap these shoes for a size 10 instead?",
    "When will my package arrive? It has been two weeks.",
    "The app crashes every time I open the settings page.",
]

# Generic bank: same Y/N shape, unrelated questions.
GENERIC_EXAMPLES = [
    ("The customer says: I need this fixed immediately, it is blocking everything.", "Does the text express urgency?", "Y"),
    ("The customer says: Here is the documentation link for reference.", "Does the text express urgency?", "N"),
    ("The customer says: Our production is down, please respond right away.", "Does the text contain a bug report?", "Y"),
    ("The customer says: Just wanted to share the meeting notes from today.", "Does the text contain a bug report?", "N"),
    ("The customer says: Please send this to a human agent right now.", "Does the text ask for a human agent?", "Y"),
    ("The customer says: No rush on this, whenever you get a chance.", "Does the text ask for a human agent?", "N"),
]

MATCHED_EXAMPLES = [
    ("The customer says: I was charged twice, please refund the duplicate.", "Does the text request a refund?", "Y"),
    ("The customer says: Can I swap these shoes for a size 10?", "Does the text request a refund?", "N"),
    ("The customer says: I want my money back for the cancelled flight.", "Does the text request a refund?", "Y"),
    ("The customer says: When will my package arrive?", "Does the text request a refund?", "N"),
]

NO_EXAMPLES: list = []

OPTIONS_VERB = (
    "Y. Yes, the text requests a refund.\n"
    "N. No, the text does not request a refund."
)
OPTIONS_GENERIC = (
    "Y. Yes, the statement is true of the text.\n"
    "N. No, the statement is false or not supported by the text."
)


def render(text: str, question: str, options: str, examples: list, option_style: str) -> str:
    blocks = []
    for ex_text, ex_q, ex_letter in examples:
        if option_style == "verb":
            opt = f"Y. Yes, {ex_q[4:-1].rstrip('?').lower() if ex_q.startswith('Does ') else ex_q}\nN. No."
        else:
            opt = options
        blocks.append(
            f"Text: {ex_text}\nQuestion: {ex_q}\nChoices:\n{opt}\nAnswer: {ex_letter}"
        )
    blocks.append(f"Text: {text}\nQuestion: {question}\nChoices:\n{options}\nAnswer: ")
    return "\n\n".join(blocks)


def run(engine, name, examples, option_style, options):
    prompts, labels = [], []
    for t in POS:
        prompts.append(render(t, TARGET_QUESTION, options, examples, option_style))
        labels.append("Y")
    for t in NEG:
        prompts.append(render(t, TARGET_QUESTION, options, examples, option_style))
        labels.append("N")
    dists = engine.score_batch(prompts, top_k=2)
    correct = sum(d.top_letter == l for d, l in zip(dists, labels))
    margins = []
    for d, l in zip(dists, labels):
        y, n = d.probabilities.get("Y", 0.0), d.probabilities.get("N", 0.0)
        margins.append(y - n if l == "Y" else n - y)
    print(f"{name:<34} acc={correct}/{len(labels)}  margin={sum(margins)/len(margins):+.3f}")
    for d, l, t in zip(dists, labels, POS + NEG):
        flag = "ok " if d.top_letter == l else "XX "
        print(f"    {flag}{d.top_letter} (want {l})  {list(d.probabilities.items())}  {t[:46]}")


def main():
    engine = GemmaLetterEngine(EngineConfig(top_k=2))
    engine.load()
    run(engine, "no examples", NO_EXAMPLES, "generic", OPTIONS_GENERIC)
    run(engine, "generic bank, generic opts", GENERIC_EXAMPLES, "generic", OPTIONS_GENERIC)
    run(engine, "generic bank, verb opts", GENERIC_EXAMPLES, "verb", OPTIONS_VERB)
    run(engine, "matched bank, verb opts", MATCHED_EXAMPLES, "verb", OPTIONS_VERB)


if __name__ == "__main__":
    main()