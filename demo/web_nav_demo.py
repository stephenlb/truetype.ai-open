"""Web-page navigation demo: Gemma 4 as a fast decision layer over page state.

Simulates a support portal with three pages (home, articles, ticket). The model
reads the current page text and decides the next action — click a link, fill a
field, or submit — via a single-letter readout. Every decision is timed.

Run:
    python demo/web_nav_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from truetype.service import TypeSafeReplica
from common import LatencyLog, print_header

PAGES = {
    "home": {
        "text": (
            "Welcome to Acme Support. Find answers in our knowledge base or open a ticket. "
            "Popular topics: billing, password reset, shipping status."
        ),
        "actions": {
            "search": "Click the search box and type a query",
            "browse": "Browse the knowledge base categories",
            "ticket": "Open a support ticket",
        },
    },
    "articles": {
        "text": (
            "Search results for 'refund':\n"
            "1. How to request a refund\n"
            "2. Refund processing time\n"
            "3. Refund policy for digital goods"
        ),
        "actions": {
            "read_1": "Open article 1: How to request a refund",
            "read_2": "Open article 2: Refund processing time",
            "ticket": "Still need help? Open a ticket",
        },
    },
    "article_1": {
        "text": (
            "Article: How to request a refund. "
            "To request a refund, open a support ticket and include your order number. "
            "Refunds are processed within 5 business days."
        ),
        "actions": {
            "ticket": "Open a support ticket as instructed",
            "back": "Back to search results",
        },
    },
    "ticket": {
        "text": (
            "Open a support ticket. Subject: ______  Description: ______  "
            "Priority: [low|medium|high]. Our team responds within 24 hours."
        ),
        "actions": {
            "fill": "Fill in the subject and description fields",
            "submit": "Submit the ticket",
            "cancel": "Cancel and return home",
        },
    },
    "ticket_filled": {
        "text": (
            "Open a support ticket. Subject: Duplicate charge on order A-104  "
            "Description: I was charged twice and need a refund.  "
            "Priority: high. Our team responds within 24 hours."
        ),
        "actions": {
            "submit": "Submit the completed ticket",
            "cancel": "Cancel and return home",
        },
    },
}

QUESTION = "What is the next best action to help the user get a refund for a duplicate charge?"


def main() -> None:
    print_header("Web-Page Navigation — Gemma 4 as a decision layer over page state")
    print("Goal: user wants a refund for a duplicate charge.")
    print("Loading model (one-time cost)...")

    service = TypeSafeReplica()
    service.engine.load()
    print(f"Model loaded in {service.engine.load_seconds:.1f}s\n")

    latency = LatencyLog("decision")
    current = "home"
    history = [current]

    for step in range(1, 8):
        page = PAGES[current]
        state = f"Current page: {current}\n\n{page['text']}"

        started = time.perf_counter()
        result = service.system_one(
            state=state,
            questions={"action": {"type": "choice", "instructions": QUESTION, "criteria": page["actions"]}},
        )
        ms = latency.record(started)

        answer = result.answers["action"]
        action = answer["choice"]
        confidence = answer["confidence"]

        print(
            f"step {step}  page={current:<9} action={action:<8} "
            f"conf={confidence:.2f}  latency={ms:6.1f}ms"
        )

        # State transitions
        if current == "home":
            current = "articles" if action in {"search", "browse"} else "ticket"
        elif current == "articles":
            if action == "read_1":
                current = "article_1"
            elif action == "read_2":
                current = "article_1"
            else:
                current = "ticket"
        elif current == "article_1":
            current = "ticket" if action == "ticket" else "articles"
        elif current == "ticket":
            if action == "fill":
                current = "ticket_filled"
            elif action == "submit":
                print("\nTicket submitted. Navigation complete.")
                break
            if action == "cancel":
                current = "home"
        elif current == "ticket_filled":
            if action == "submit":
                print("\nTicket submitted. Navigation complete.")
                break
            if action == "cancel":
                current = "home"
        history.append(current)

    print("\n" + latency.summary())
    print(f"Pages visited: {' -> '.join(history)}")


if __name__ == "__main__":
    main()
