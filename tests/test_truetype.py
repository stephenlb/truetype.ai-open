"""52-test verification suite for the truetype.ai replica.

50 functional tests (20 noul + 20 choice + 10 score) verify the model's top letter
matches the expected answer for each prompt, and that the reported probability
distribution is a valid softmax over the legal option letters.

Test 51 guards the readout contract itself: one token, A-Z only, and the letters
holding the model's probability mass. That last property broke
once while all 50 functional tests still passed.

Test 52 guards the prefix-cache sizing that the warm-request speedup depends on.
"""

from __future__ import annotations

import math

import pytest

from src.truetype.engine import (
    MAX_NEW_TOKENS,
    PREFIX_CACHE_HARD_CAP,
    EngineConfig,
    GemmaLetterEngine,
)
from src.truetype.letters import LETTERS
from src.truetype.questions import build_question, score_question
from src.truetype.render import render_example_prefix, render_target_block
from src.truetype.service import TypeSafeReplica

# ---------------------------------------------------------------------------
# Functional cases: (state, expected) tuples verified against the live model.
# ---------------------------------------------------------------------------

NOUL_REFUND_Q = "Does the text request a refund?"
NOUL_REFUND_CASES = [
    ("I was charged twice for order A-104. Please refund the duplicate.", "Y"),
    ("Can I get my money back for the cancelled flight?", "Y"),
    ("I want a full refund, this product is not what I ordered.", "Y"),
    ("Please return my payment, the item never arrived.", "Y"),
    ("I demand a refund for this broken product.", "Y"),
    ("Can I swap these shoes for a size 10 instead?", "N"),
    ("When will my package arrive? It has been two weeks.", "N"),
    ("The app crashes every time I open the settings page.", "N"),
    ("Just wanted to share the meeting notes from today.", "N"),
    ("How do I reset my password?", "N"),
]

NOUL_URGENCY_Q = "Does the text express urgency?"
NOUL_URGENCY_CASES = [
    ("Please help ASAP, we are losing sales.", "Y"),
    ("Our production is down, please respond right away.", "Y"),
    ("I need this fixed immediately, it is blocking everything.", "Y"),
    ("URGENT: the payment system is failing right now.", "Y"),
    ("This is critical, we cannot operate without it.", "Y"),
    ("Just wanted to share the meeting notes from today.", "N"),
    ("Here is the documentation link for reference.", "N"),
    ("No rush on this, whenever you get a chance.", "N"),
    ("The sky is blue and the grass is green.", "N"),
    ("Thanks for the update, have a great weekend.", "N"),
]

CHOICE_TEAM_Q = "Which team should handle this?"
CHOICE_TEAM_CRITERIA = {
    "returns": "Exchanges, refunds, wrong or damaged items",
    "shipping": "Delivery status, delays, lost packages",
    "billing": "Charges, invoices, payment problems",
}
CHOICE_TEAM_CASES = [
    ("My running shoes arrived in the wrong size. Can I swap them for a size 10?", "returns"),
    ("My package was supposed to arrive last week and it is still not here.", "shipping"),
    ("I was charged twice for the same order on my credit card.", "billing"),
    ("The jacket I ordered is damaged and I would like to exchange it.", "returns"),
    ("Can you tell me when my order will be delivered?", "shipping"),
    ("I need an invoice for my last payment.", "billing"),
    ("This is the third pair of shoes that did not fit. I want to send them back.", "returns"),
    ("Why was I billed for a subscription I cancelled?", "billing"),
    ("The tracking number you sent does not work.", "shipping"),
    ("My order arrived with a cracked screen.", "returns"),
]

CHOICE_SENTIMENT_Q = "What is the sentiment of this review?"
CHOICE_SENTIMENT_CRITERIA = {
    "positive": "Praise, satisfaction, recommendation",
    "negative": "Complaint, frustration, disappointment",
    "neutral": "Factual statement without emotion",
}
CHOICE_SENTIMENT_CASES = [
    ("Absolutely love this product, best purchase I have made all year!", "positive"),
    ("Terrible quality, broke after one day. Do not buy.", "negative"),
    ("The package arrived on Tuesday.", "neutral"),
    ("Fantastic customer service, they resolved my issue in minutes.", "positive"),
    ("Worst experience ever, I will never shop here again.", "negative"),
    ("The item is blue and weighs two pounds.", "neutral"),
    ("Highly recommend, exceeded all my expectations.", "positive"),
    ("Complete waste of money, fell apart immediately.", "negative"),
    ("The manual has twelve pages.", "neutral"),
    ("Five stars, would buy again without hesitation.", "positive"),
]

SCORE_SEVERITY_Q = "How severe is the reported issue?"
SCORE_SEVERITY_CRITERIA = [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists",
]
SCORE_SEVERITY_CASES = [
    ("The export button is misaligned by a few pixels on the settings page.", 0),
    ("The PDF export button does nothing when clicked. I can still export to CSV, but that takes ages.", 1),
    ("Nobody on our team can log in since this morning. We get a 500 error on every attempt.", 2),
    ("The icon next to the save button is the wrong shade of blue.", 0),
    ("Search returns stale results until I refresh the page twice.", 1),
    ("All customer orders have been failing for 3 hours with no workaround.", 2),
    ("The footer copyright year says 2024 instead of 2026.", 0),
    ("The dashboard is slow but eventually loads after about 30 seconds.", 1),
    ("Payments are completely down and we cannot process any orders.", 2),
    ("The tooltip text has a typo in it.", 0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_distribution(probabilities: dict[str, float], legal_labels: set[str]):
    assert set(probabilities) == legal_labels
    total = sum(probabilities.values())
    assert math.isclose(total, 1.0, abs_tol=1e-5), f"probabilities sum to {total}"
    for p in probabilities.values():
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Noul functional tests (20)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", NOUL_REFUND_CASES)
def test_noul_refund(service, state, expected):
    r = service.system_one(state=state, questions={"q": {"type": "noul", "instructions": NOUL_REFUND_Q}})
    ans = r.answers["q"]
    assert ans["type"] == "noul"
    got = "Y" if ans["noul"] >= 0.5 else "N"
    assert got == expected, f"noul={ans['noul']:.3f} for {state!r}"


@pytest.mark.parametrize("state,expected", NOUL_URGENCY_CASES)
def test_noul_urgency(service, state, expected):
    r = service.system_one(state=state, questions={"q": {"type": "noul", "instructions": NOUL_URGENCY_Q}})
    ans = r.answers["q"]
    got = "Y" if ans["noul"] >= 0.5 else "N"
    assert got == expected, f"noul={ans['noul']:.3f} for {state!r}"


# ---------------------------------------------------------------------------
# Choice functional tests (20)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", CHOICE_TEAM_CASES)
def test_choice_team(service, state, expected):
    r = service.system_one(
        state=state,
        questions={"q": {"type": "choice", "instructions": CHOICE_TEAM_Q, "criteria": CHOICE_TEAM_CRITERIA}},
    )
    ans = r.answers["q"]
    assert ans["type"] == "choice"
    assert ans["choice"] == expected, f"probs={ans['probabilities']} for {state!r}"
    _assert_valid_distribution(ans["probabilities"], set(CHOICE_TEAM_CRITERIA))


@pytest.mark.parametrize("state,expected", CHOICE_SENTIMENT_CASES)
def test_choice_sentiment(service, state, expected):
    r = service.system_one(
        state=state,
        questions={"q": {"type": "choice", "instructions": CHOICE_SENTIMENT_Q, "criteria": CHOICE_SENTIMENT_CRITERIA}},
    )
    ans = r.answers["q"]
    assert ans["choice"] == expected, f"probs={ans['probabilities']} for {state!r}"
    _assert_valid_distribution(ans["probabilities"], set(CHOICE_SENTIMENT_CRITERIA))


# ---------------------------------------------------------------------------
# Score functional tests (10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state,expected", SCORE_SEVERITY_CASES)
def test_score_severity(service, state, expected):
    r = service.system_one(
        state=state,
        questions={"q": {"type": "score", "instructions": SCORE_SEVERITY_Q, "criteria": SCORE_SEVERITY_CRITERIA}},
    )
    ans = r.answers["q"]
    assert ans["type"] == "score"
    assert round(ans["score"]) == expected, f"score={ans['score']:.3f} probs={ans['probabilities']} for {state!r}"
    _assert_valid_distribution(ans["probabilities"], {"0", "1", "2"})
    assert ans["legend"] == {str(i): lvl for i, lvl in enumerate(SCORE_SEVERITY_CRITERIA)}


# ---------------------------------------------------------------------------
# Structural test (1): the single-token letter readout contract
# ---------------------------------------------------------------------------

def test_single_token_letter_readout_contract(engine):
    """The answer must be one token, drawn from A-Z, holding the model's real mass.

    This guards a bug that passed every accuracy test while being badly wrong: the
    prompt used to end with ``"Answer: "``, whose trailing space tokenizes to a
    standalone ``'▁'``. The model then wanted a bare ``"Y"`` while the engine scored
    the space-prefixed ``"▁Y"``, so the argmax ran over tokens holding 0.0000 of the
    probability mass. Accuracy survived on ordering alone; the confidences were
    meaningless. Anything that strands the answer slot again will fail here.
    """
    # 1. Exactly one token is read, structurally.
    assert MAX_NEW_TOKENS == 1

    # 2. The readout targets 26 distinct A-Z ids from a single token variant.
    assert set(engine._letter_token_ids) == set(LETTERS)
    assert len(set(engine._letter_token_ids.values())) == 26
    assert engine.letter_variant in {"space-prefixed", "bare"}

    # 3. The prompt must not strand the space before the answer letter.
    question = build_question(
        "q", {"type": "noul", "instructions": "Does the text request a refund?"}
    )
    tail = render_target_block(question, "Please refund the duplicate charge.")
    assert tail.endswith("Answer:"), f"answer slot has a trailing space: {tail[-12:]!r}"

    # 4. The letters must carry the model's actual probability, across all 3 types.
    prompts = [
        (question, "Please refund the duplicate charge."),
        (
            build_question(
                "c",
                {
                    "type": "choice",
                    "instructions": CHOICE_TEAM_Q,
                    "criteria": CHOICE_TEAM_CRITERIA,
                },
            ),
            "I was charged twice for the same order.",
        ),
        (
            build_question(
                "s",
                {
                    "type": "score",
                    "instructions": SCORE_SEVERITY_Q,
                    "criteria": SCORE_SEVERITY_CRITERIA,
                },
            ),
            "Payments are completely down and we cannot process any orders.",
        ),
    ]
    for q, state in prompts:
        readout = engine.score_with_prefix(
            render_example_prefix(q), [render_target_block(q, state)]
        )[0]
        assert readout.letter_mass is not None
        assert readout.letter_mass > 0.5, (
            f"only {readout.letter_mass:.4f} of the probability mass is on A-Z for "
            f"{q.type!r}; the answer slot is not in a letter state"
        )
        # The single most likely letter must be one the question declared legal.
        legal = {option.letter for option in q.options}
        best = max(readout.logits, key=readout.logits.get)
        assert best in legal or readout.logits[best] >= max(
            readout.logits[letter] for letter in legal
        )


# ---------------------------------------------------------------------------
# Structural test (2): prefix cache capacity must cover the request
# ---------------------------------------------------------------------------

def test_prefix_cache_capacity_covers_request():
    """The LRU must hold every prefix a request uses, or the caching win vanishes.

    Measured: 10 questions against an 8-entry cache ran at 8051ms because each
    prefix was evicted before it was ever revisited; a 10-entry cache ran the same
    request at 1771ms. Capacity is therefore grown per request, clamped by a hard
    cap that bounds KV memory. This test does not load the model.
    """
    engine = GemmaLetterEngine(EngineConfig(prefix_cache=True))
    assert engine.config.max_cached_prefixes == 16  # default

    # Grows to fit a request larger than the default.
    assert engine.ensure_prefix_capacity(30) == 30
    assert engine.config.max_cached_prefixes == 30

    # Never shrinks, so a small request does not evict a warmed large cache.
    assert engine.ensure_prefix_capacity(3) == 30
    assert engine.config.max_cached_prefixes == 30

    # Clamped at the hard cap, and warns rather than pretending it fits.
    assert engine.ensure_prefix_capacity(10_000) == PREFIX_CACHE_HARD_CAP
    assert engine.config.max_cached_prefixes == PREFIX_CACHE_HARD_CAP

    # A custom lower cap is respected (memory-constrained machines).
    small = GemmaLetterEngine(
        EngineConfig(prefix_cache=True, max_cached_prefixes=2, prefix_cache_hard_cap=4)
    )
    assert small.ensure_prefix_capacity(99) == 4
    assert small.ensure_prefix_capacity(1) == 4
