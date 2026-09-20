"""Text-adventure demo: Gemma 4 plays a game in real time.

A linear 4-room text adventure. The model reads the room description and picks
the next action via a single-letter readout. Every decision is timed so you can
see the per-move latency of a 12B-parameter model used as a game controller.

Run:
    python demo/game_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from truetype.service import TypeSafeReplica
from common import LatencyLog, print_header

ROOMS = {
    "cave_mouth": {
        "text": (
            "You stand at the mouth of a dark cave. A cold wind blows from within. "
            "A narrow tunnel to the north descends into darkness. "
            "A rocky ledge to the east overlooks a bottomless chasm."
        ),
        "actions": {
            "tunnel": "Enter the narrow tunnel to the north",
            "ledge": "Climb the rocky ledge to the east",
        },
    },
    "ledge": {
        "text": (
            "You are on a narrow ledge above a bottomless chasm. The wind howls. "
            "There is nothing here but loose rock and vertigo. "
            "The only way is back to the cave mouth."
        ),
        "actions": {
            "cave_mouth": "Return to the cave mouth",
        },
    },
    "tunnel": {
        "text": (
            "You are in a narrow tunnel. The walls are damp and the air smells of earth. "
            "Ahead, the tunnel splits: a wide passage to the left, and a low crawlway to the right."
        ),
        "actions": {
            "wide": "Take the wide passage to the left",
            "crawl": "Take the low crawlway to the right",
        },
    },
    "crawl": {
        "text": (
            "You are in a low crawlway. The ceiling scrapes your back. "
            "After a few feet, it dead-ends at a wall of solid rock. "
            "The only way is back to the tunnel junction."
        ),
        "actions": {
            "tunnel": "Crawl back to the tunnel junction",
        },
    },
    "wide": {
        "text": (
            "You are in a wide, high-ceilinged chamber. Stalactites hang like stone teeth. "
            "In the center, a stone pedestal holds a small locked chest. "
            "A key glints on a nearby rock."
        ),
        "actions": {
            "key": "Take the key from the rock",
            "chest": "Unlock the chest with the key",
        },
    },
}

QUESTION = "What is the best next action to find the treasure in the locked chest?"
MAX_STEPS = 10


def main() -> None:
    print_header("Text Adventure: Gemma 4 as a real-time game controller")
    print("Goal: open the locked chest. Start: cave mouth.")
    print("Loading model (one-time cost)...")

    service = TypeSafeReplica()
    service.engine.load()
    print(f"Model loaded in {service.engine.load_seconds:.1f}s\n")

    latency = LatencyLog("decision")
    current = "cave_mouth"
    visited = [current]
    has_key = False

    for step in range(1, MAX_STEPS + 1):
        room = ROOMS[current]
        if current == "wide" and has_key:
            state = (
                f"Current location: {current}\n\n"
                "You are in a wide, high-ceilinged chamber. Stalactites hang like stone teeth. "
                "In the center, a stone pedestal holds a small locked chest. "
                "You have the key.\n\n"
                "Inventory: key\n"
                "Goal: open the locked chest."
            )
        else:
            state = (
                f"Current location: {current}\n\n{room['text']}\n\n"
                f"Inventory: {'key' if has_key else 'empty'}\n"
                f"Goal: open the locked chest."
            )

        started = time.perf_counter()
        result = service.system_one(
            state=state,
            questions={"action": {"type": "choice", "instructions": QUESTION, "criteria": room["actions"]}},
        )
        ms = latency.record(started)

        answer = result.answers["action"]
        action = answer["choice"]
        confidence = answer["confidence"]

        print(
            f"step {step:2d}  room={current:<11} action={action:<11} "
            f"conf={confidence:.2f}  latency={ms:6.1f}ms"
        )

        # State transitions
        if current == "cave_mouth":
            current = action
        elif current == "ledge":
            current = "cave_mouth"
        elif current == "tunnel":
            current = action
        elif current == "crawl":
            current = "tunnel"
        elif current == "wide":
            if action == "key":
                if not has_key:
                    has_key = True
                    print("        (key acquired)")
                else:
                    print("        (already have the key)")
            elif action == "chest":
                if has_key:
                    print("\nYou unlock the chest with the key. Inside is the treasure. You win!")
                    break
                else:
                    print("        (chest is locked; you need a key)")
        visited.append(current)
    else:
        print(f"\nDid not open the chest in {MAX_STEPS} steps.")

    print("\n" + latency.summary())
    print(f"Rooms visited: {' -> '.join(visited)}")


if __name__ == "__main__":
    main()
