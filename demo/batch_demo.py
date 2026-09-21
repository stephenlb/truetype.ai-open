"""Batch-throughput demo: many questions over one state in a single call.

Submits five questions about one customer message and measures total wall time.

Run:
    python demo/batch_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from truetype.service import TypeSafeReplica
from common import print_header

STATE = (
    "Hi, I was charged twice for order A-104 last week and the duplicate charge "
    "is still on my card. I need this refunded immediately; rent is due tomorrow. "
    "Also, the app crashes every time I try to open the receipt. Please help."
)

QUESTIONS = {
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


def main() -> None:
    print_header("Batch Throughput: 5 questions, 1 state, 1 forward pass each")
    print("Loading model (one-time cost)...")

    service = TypeSafeReplica()
    service.engine.load()
    print(f"Model loaded in {service.engine.load_seconds:.1f}s\n")

    started = time.perf_counter()
    result = service.system_one(state=STATE, questions=QUESTIONS)
    total_ms = (time.perf_counter() - started) * 1000

    print(f"State: {STATE}\n")
    for qid, answer in result.answers.items():
        if answer["type"] == "noul":
            print(f"  {qid:<10} noul={answer['noul']:.3f}")
        elif answer["type"] == "choice":
            print(f"  {qid:<10} choice={answer['choice']:<12} conf={answer['confidence']:.2f}")
        else:
            print(f"  {qid:<10} score={answer['score']:.2f}  conf={answer['confidence']:.2f}")

    print(f"\nTotal wall time for 5 decisions: {total_ms:.1f}ms")
    print(f"Average per decision: {total_ms / len(QUESTIONS):.1f}ms")
    print(f"Usage: {result.usage}")


if __name__ == "__main__":
    main()
