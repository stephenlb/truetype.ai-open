"""Latency benchmark: prefix KV cache on vs off, plus a correctness check.

The engine caches the KV of a prompt's static prefix (few-shot examples + the
question's Choices/Question header) and only prefills the state-dependent tail.
This script measures what that buys and confirms it does not change decisions.

Run:
    python demo/latency_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from truetype.engine import EngineConfig, GemmaLetterEngine
from truetype.service import TypeSafeReplica
from common import print_header

DOOM_CRITERIA = {
    "turn_left": "Rotate left to aim - correct when the nearest monster is to the left of the crosshair",
    "turn_right": "Rotate right to aim - correct when the nearest monster is to the right of the crosshair",
    "attack": "Fire the weapon - correct when a monster is already dead ahead in the crosshair",
}
DOOM_Q = (
    "The player must aim at the nearest monster and shoot it. "
    "Which action should the player take right now?"
)
BEARINGS = ["to the left", "dead ahead", "to the right", "far to the left", "far to the right"]

NOUL_Q = {"q": {"type": "noul", "instructions": "Does the text request a refund?"}}
NOUL_CASES = [
    "I was charged twice for order A-104. Please refund the duplicate.",
    "When will my package arrive? It has been two weeks.",
    "I want a full refund, this product is not what I ordered.",
]

BATCH_Q = {
    "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
    "urgent": {"type": "noul", "instructions": "Does the text express urgency?"},
    "bug": {"type": "noul", "instructions": "Does the text contain a bug report?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": "Charges, invoices, payment problems",
            "engineering": "App crashes, bugs, technical issues",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the reported issue?",
        "criteria": [
            "Cosmetic; no impact to functionality",
            "Broken or degraded feature, but workaround exists",
            "Blocking issue; no workaround exists",
        ],
    },
}
BATCH_STATE = (
    "Hi, I was charged twice for order A-104 last week and the duplicate charge is "
    "still on my card. I need this refunded immediately. Also, the app crashes."
)


def doom_state(bearing: str) -> str:
    return (
        "Doom. Health: 100. Ammo: 26.\n"
        "Nearest monster: Demon at medium range.\n"
        f"Monster bearing relative to crosshair: {bearing}.\n"
        f"Monster lined up in crosshair: {'yes' if bearing == 'dead ahead' else 'no'}."
    )


def mean_ms(fn, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    print_header("Latency: prefix KV cache on vs off")

    cached = GemmaLetterEngine(EngineConfig(top_k=5, prefix_cache=True))
    cached.load()
    # Second engine shares the loaded weights so only the cache policy differs.
    plain = GemmaLetterEngine(EngineConfig(top_k=5, prefix_cache=False))
    for attr in ("_model", "_tokenizer", "_letter_token_ids", "_pad_token_id", "load_seconds"):
        setattr(plain, attr, getattr(cached, attr))

    svc_cached, svc_plain = TypeSafeReplica(cached), TypeSafeReplica(plain)
    doom_q = {"act": {"type": "choice", "instructions": DOOM_Q, "criteria": DOOM_CRITERIA}}

    print("Correctness: same decision with and without the cache?")
    mismatches = 0
    for bearing in BEARINGS:
        a = svc_cached.system_one(state=doom_state(bearing), questions=doom_q).answers["act"]
        b = svc_plain.system_one(state=doom_state(bearing), questions=doom_q).answers["act"]
        same = a["choice"] == b["choice"]
        mismatches += not same
        drift = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
        print(f"  {'ok ' if same else 'XX '}{bearing:<16} {a['choice']:<10} max|dp|={drift:.4f}")
    for text in NOUL_CASES:
        a = svc_cached.system_one(state=text, questions=NOUL_Q).answers["q"]["noul"]
        b = svc_plain.system_one(state=text, questions=NOUL_Q).answers["q"]["noul"]
        same = (a >= 0.5) == (b >= 0.5)
        mismatches += not same
        print(f"  {'ok ' if same else 'XX '}noul {a:.3f} vs {b:.3f}   {text[:44]}")
    print(f"  decision mismatches: {mismatches}")

    print("\nSingle question, warm cache (mean of 5):")
    rows = [
        ("Doom 3-option choice", lambda s: s.system_one(state=doom_state("to the left"), questions=doom_q)),
        ("noul refund", lambda s: s.system_one(state=NOUL_CASES[0], questions=NOUL_Q)),
    ]
    for label, call in rows:
        on = mean_ms(lambda: call(svc_cached), 5)
        off = mean_ms(lambda: call(svc_plain), 5)
        print(f"  {label:<22} cache on {on:7.1f}ms   off {off:7.1f}ms   {off/on:.2f}x")

    print("\n5-question request:")
    cached._prefix_caches.clear()
    cold = mean_ms(lambda: svc_cached.system_one(state=BATCH_STATE, questions=BATCH_Q), 1)
    warm = mean_ms(lambda: svc_cached.system_one(state=BATCH_STATE, questions=BATCH_Q), 2)
    off = mean_ms(lambda: svc_plain.system_one(state=BATCH_STATE, questions=BATCH_Q), 2)
    print(f"  cache off (batched)        {off:7.1f}ms")
    print(f"  cache on, cold (all miss)  {cold:7.1f}ms   {cold/off:.2f}x vs off")
    print(f"  cache on, warm (all hit)   {warm:7.1f}ms   {off/warm:.2f}x vs off")

    print(f"\ncache hits={cached.prefix_cache_hits} misses={cached.prefix_cache_misses}")


if __name__ == "__main__":
    main()
