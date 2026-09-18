"""Format bake-off for Choice and Score questions (arbitrary option labels/letters)."""

from __future__ import annotations

from src.truetype.engine import EngineConfig, GemmaLetterEngine

# ---------------- Choice task: support triage ----------------
CHOICE_Q = "Which team should handle this?"
CHOICE_OPTS = {
    "returns": "Exchanges, refunds, wrong or damaged items",
    "shipping": "Delivery status, delays, lost packages",
    "billing": "Charges, invoices, payment problems",
}
CHOICE_OPTS_TEXT = "\n".join(f"{L}. {k} - {v}" for L, (k, v) in zip("ABC", CHOICE_OPTS.items()))

CHOICE_CASES = [
    ("My running shoes arrived in the wrong size. Can I swap them for a size 10?", "returns"),
    ("My package was supposed to arrive last week and it is still not here.", "shipping"),
    ("I was charged twice for the same order on my credit card.", "billing"),
    ("The jacket I ordered is damaged and I would like to exchange it.", "returns"),
    ("Can you tell me when my order will be delivered?", "shipping"),
    ("I need an invoice for my last payment.", "billing"),
]

# ---------------- Score task: bug severity ----------------
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
]


def roundtrip_examples(n_options: int, opts_text: str, question: str) -> list[str]:
    """One self-referential example per option: teaches letter<->option mapping only."""
    blocks = []
    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]):
        label = opts_text.split("\n")[i].split(". ", 1)[1].split(" - ")[0]
        blocks.append(
            f"Text: This text is about {label}.\nQuestion: {question}\n"
            f"Choices:\n{opts_text}\nAnswer: {letter}"
        )
    return blocks


def render(text, question, opts_text, prefix_blocks: list[str]) -> str:
    blocks = list(prefix_blocks)
    blocks.append(f"Text: {text}\nQuestion: {question}\nChoices:\n{opts_text}\nAnswer: ")
    return "\n\n".join(blocks)


def run(engine, name, question, opts_text, cases, n_options, strategy):
    if strategy == "none":
        prefix = []
    elif strategy == "roundtrip":
        prefix = roundtrip_examples(n_options, opts_text, question)
    elif strategy == "letter_legend":
        legend = "\n".join(
            f"{L}. {opts_text.splitlines()[i].split('. ', 1)[1]}" for i, L in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options])
        )
        prefix = [f"Answer every question with one letter:\n{legend}"]
    else:
        raise ValueError(strategy)

    prompts = [render(t, question, opts_text, prefix) for t, _ in cases]
    dists = engine.score_batch(prompts, top_k=n_options)
    labels = [l for _, l in cases]
    correct = sum(d.top_letter == l for d, l in zip(dists, labels))
    print(f"\n{name}  strategy={strategy}  acc={correct}/{len(labels)}")
    for d, l, (t, _) in zip(dists, labels, cases):
        flag = "ok " if d.top_letter == l else "XX "
        tops = list(d.probabilities.items())[:3]
        print(f"    {flag}{d.top_letter} (want {l})  {[(k, round(v,3)) for k,v in tops]}  {t[:44]}")


def main():
    engine = GemmaLetterEngine(EngineConfig(top_k=3))
    engine.load()
    for strategy in ["none", "roundtrip", "letter_legend"]:
        run(engine, "CHOICE", CHOICE_Q, CHOICE_OPTS_TEXT, CHOICE_CASES, 3, strategy)
    for strategy in ["none", "roundtrip", "letter_legend"]:
        run(engine, "SCORE", SCORE_Q, SCORE_OPTS_TEXT, SCORE_CASES, 3, strategy)


if __name__ == "__main__":
    main()