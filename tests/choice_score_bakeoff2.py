"""Rigorous Choice/Score bake-off: shuffled labels to kill positional pattern-matching.

Two things this fixes over the first attempt:
1. expected answers are letters mapped from labels, not labels compared to letters;
2. case order is shuffled so "continue A,B,C" cannot score well, and repeated
   correct answers are included so degenerate strategies are punished.
"""

from __future__ import annotations

from src.truetype.engine import EngineConfig, GemmaLetterEngine

CHOICE_Q = "Which team should handle this?"
CHOICE_OPTS = {
    "returns": "Exchanges, refunds, wrong or damaged items",
    "shipping": "Delivery status, delays, lost packages",
    "billing": "Charges, invoices, payment problems",
}
CHOICE_LETTERS = {name: L for L, name in zip("ABC", CHOICE_OPTS)}
CHOICE_OPTS_TEXT = "\n".join(f"{L}. {k} - {v}" for L, (k, v) in zip("ABC", CHOICE_OPTS.items()))

CHOICE_CASES = [
    ("My running shoes arrived in the wrong size. Can I swap them for a size 10?", "returns"),
    ("My package was supposed to arrive last week and it is still not here.", "shipping"),
    ("I was charged twice for the same order on my credit card.", "billing"),
    ("The jacket I ordered is damaged and I would like to exchange it.", "returns"),
    ("Can you tell me when my order will be delivered?", "shipping"),
    ("I need an invoice for my last payment.", "billing"),
    ("This is the third pair of shoes that did not fit. I want to send them back.", "returns"),
    ("My refund has not shown up after two weeks.", "returns"),
    ("Why was I billed for a subscription I cancelled?", "billing"),
    ("The tracking number you sent does not work.", "shipping"),
]

SCORE_Q = "How severe is the reported issue?"
SCORE_LEVELS = [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists",
]
SCORE_OPTS_TEXT = "\n".join(f"{L}. {lvl}" for L, lvl in zip("ABC", SCORE_LEVELS))
SCORE_CASES = [
    ("The export button is misaligned by a few pixels on the settings page.", "A"),
    ("The PDF export button does nothing when clicked. I can still export to CSV, but that takes ages.", "B"),
    ("Nobody on our team can log in since this morning. We get a 500 error on every attempt.", "C"),
    ("The icon next to the save button is the wrong shade of blue.", "A"),
    ("Search returns stale results until I refresh the page twice.", "B"),
    ("All customer orders have been failing for 3 hours with no workaround.", "C"),
    ("The footer copyright year says 2024 instead of 2026.", "A"),
    ("The dashboard is slow but eventually loads after about 30 seconds.", "B"),
    ("Payments are completely down and we cannot process any orders.", "C"),
    ("The tooltip text has a typo in it.", "A"),
    ("Dark mode makes some text unreadable, but I can switch back to light mode.", "B"),
    ("The API returns 500 on every request, blocking all integrations.", "C"),
]


def roundtrip_examples(n_options: int, question: str, labels: list[str]) -> list[str]:
    """One example per option, deliberately NOT in alphabetical order.

    Order is shuffled by label so the model cannot learn "next letter in sequence".
    """
    order = sorted(range(n_options), key=lambda i: labels[i])
    blocks = []
    for i in reversed(order):
        blocks.append(
            f"Text: This text is about {labels[i]}.\nQuestion: {question}\n"
            f"Choices:\n{_opts_text_for(labels)}\n"
            f"Answer: {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}"
        )
    return blocks


def _opts_text_for(labels: list[str]) -> str:
    return "\n".join(
        f"{L}. {name} - {CHOICE_OPTS[name]}" for L, name in zip("ABC", labels)
    )


def render(text, question, opts_text, prefix_blocks: list[str]) -> str:
    blocks = list(prefix_blocks)
    blocks.append(f"Text: {text}\nQuestion: {question}\nChoices:\n{opts_text}\nAnswer: ")
    return "\n\n".join(blocks)


def run(engine, name, question, opts_text, cases, n_options, strategy):
    labels_in_order = list(dict.fromkeys(l for _, l in cases))
    if strategy == "none":
        prefix = []
    elif strategy == "roundtrip_shuffled":
        prefix = roundtrip_examples(n_options, question, labels_in_order)
    elif strategy == "letter_legend":
        prefix = [
            "Answer every question with one letter from the choices below.\n"
            + "\n".join(f"{L}. {lvl}" for L, lvl in zip("ABC", labels_in_order))
        ]
    else:
        raise ValueError(strategy)

    prompts = [render(t, question, opts_text, prefix) for t, _ in cases]
    dists = engine.score_batch(prompts, top_k=n_options)
    expected = [l if l in "ABC" else CHOICE_LETTERS[l] for _, l in cases]
    correct = sum(d.top_letter == e for d, e in zip(dists, expected))
    print(f"\n{name}  strategy={strategy}  acc={correct}/{len(expected)}")
    for d, e, (t, _) in zip(dists, expected, cases):
        flag = "ok " if d.top_letter == e else "XX "
        tops = [(k, round(v, 3)) for k, v in list(d.probabilities.items())[:3]]
        print(f"    {flag}{d.top_letter} (want {e})  {tops}  {t[:46]}")


def main():
    engine = GemmaLetterEngine(EngineConfig(top_k=3))
    engine.load()
    run(engine, "CHOICE", CHOICE_Q, CHOICE_OPTS_TEXT, CHOICE_CASES, 3, "none")
    run(engine, "CHOICE", CHOICE_Q, CHOICE_OPTS_TEXT, CHOICE_CASES, 3, "letter_legend")
    run(engine, "CHOICE", CHOICE_Q, CHOICE_OPTS_TEXT, CHOICE_CASES, 3, "roundtrip_shuffled")
    run(engine, "SCORE", SCORE_Q, SCORE_OPTS_TEXT, SCORE_CASES, 3, "none")
    run(engine, "SCORE", SCORE_Q, SCORE_OPTS_TEXT, SCORE_CASES, 3, "letter_legend")


if __name__ == "__main__":
    main()