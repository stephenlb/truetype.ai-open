"""Real Doom demo: Gemma 4 plays Doom through ViZDoom.

This runs the actual Doom engine. ViZDoom is a ZDoom fork built for AI agents and
ships the Freedoom IWADs, so there is no screen capture and no OS permission
needed: object labels, health, ammo and kill count come straight out of the
engine each tick.

Each tick the engine state is reduced to a short text prompt whose final line is
the decisive fact (where the target sits relative to the crosshair). Gemma 4
answers with one letter, that letter maps to a Doom button, and the engine
advances. Every decision is timed.

Run:
    python demo/doom_demo.py                    # defend_the_center (default)
    python demo/doom_demo.py health_gathering   # walk onto medkits to survive
    python demo/doom_demo.py deadly_corridor    # fight down a corridor

Add ``--watch`` to any of the above to open ViZDoom's window and see the gameplay
as it happens (uses the engine's own renderer, so no screen-capture permission is
involved). Rendering happens inside ``make_action``, so per-decision latency is
unchanged; wall-clock throughput drops by roughly 20%.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import vizdoom as vzd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from truetype.service import TypeSafeReplica
from common import LatencyLog, print_header

# Option descriptions spell out *when* each action is right. The probe in this
# directory showed plain descriptions ("rotate left") collapse to always-A,
# while rule-bearing descriptions track the target correctly.
SCENARIOS = {
    "defend_the_center": {
        "question": (
            "The player must aim at the nearest monster and shoot it. "
            "Which action should the player take right now?"
        ),
        "target_kind": "monster",
        "targets": {"Demon", "MarineChainsawVzd", "Zombieman", "ShotgunGuy", "ChaingunGuy"},
        "actions": {
            "turn_left": (
                "TURN_LEFT",
                "Rotate left to aim - correct when the nearest monster is to the left of the crosshair",
            ),
            "turn_right": (
                "TURN_RIGHT",
                "Rotate right to aim - correct when the nearest monster is to the right "
                "of the crosshair, and also correct when no monster is visible at all "
                "(sweep the view around to find the next one)",
            ),
            "attack": (
                "ATTACK",
                "Fire the weapon - correct when a monster is already dead ahead in the crosshair",
            ),
        },
    },
    "health_gathering": {
        "question": (
            "The floor is poisoned and the player must walk onto a medkit to heal. "
            "Which action should the player take right now?"
        ),
        "target_kind": "medkit",
        "targets": {"Medikit"},
        "actions": {
            "turn_left": (
                "TURN_LEFT",
                "Rotate left to line up - correct when the nearest medkit is to the left",
            ),
            "turn_right": (
                "TURN_RIGHT",
                "Rotate right to line up - correct when the nearest medkit is to the right, "
                "and also correct when no medkit is visible at all (sweep to find one)",
            ),
            "forward": (
                "MOVE_FORWARD",
                "Walk forward onto the medkit - correct when the medkit is already dead ahead",
            ),
        },
    },
    "deadly_corridor": {
        "question": (
            "The player must fight down the corridor to the green armour at the end. "
            "Which action should the player take right now?"
        ),
        "target_kind": "monster",
        "targets": {"Zombieman", "ShotgunGuy", "ChaingunGuy", "Demon"},
        "actions": {
            "turn_left": (
                "TURN_LEFT",
                "Rotate left to aim - correct when the nearest monster is to the left of the crosshair",
            ),
            "turn_right": (
                "TURN_RIGHT",
                "Rotate right to aim - correct when the nearest monster is to the right of the crosshair",
            ),
            "attack": (
                "ATTACK",
                "Fire the weapon - correct when a monster is already dead ahead in the crosshair",
            ),
            "forward": (
                "MOVE_FORWARD",
                "Advance down the corridor - correct when no monster is visible",
            ),
        },
    },
}

FRAME_SKIP = 4  # engine tics advanced per decision (Doom runs at 35 tics/sec)
MAX_DECISIONS = 40


def build_game(scenario: str, watch: bool = False) -> tuple[vzd.DoomGame, list[str]]:
    """Load the scenario, restricted to the buttons this demo exposes.

    With ``watch`` the engine opens its own SDL window and draws the game as it
    plays. This is ViZDoom's native renderer, not an OS screen grab, so it needs no
    macOS Screen Recording permission. The render cost lands inside ``make_action``,
    outside the timed decision, so reported latency is unchanged while wall-clock
    throughput drops (measured ~5.5 -> ~4.5 decisions/sec).
    """
    cfg = Path(vzd.__file__).parent / "scenarios" / f"{scenario}.cfg"
    game = vzd.DoomGame()
    game.load_config(str(cfg))
    game.set_window_visible(watch)
    if watch:
        # Bigger window to actually watch, and draw every skipped tic so motion is
        # continuous instead of jumping to the end state of each decision. Safe for
        # the agent: bearings are normalised by screen_width/2, and the model reads
        # engine labels, never pixels.
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_render_all_frames(True)
        # The scenario .cfg files ship with render_hud = false; turn the HUD and
        # crosshair on so a human can follow health, ammo and aim.
        game.set_render_hud(True)
        game.set_render_crosshair(True)
    else:
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_labels_buffer_enabled(True)
    game.set_available_game_variables(
        [vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT]
    )

    names = list(SCENARIOS[scenario]["actions"])
    game.set_available_buttons(
        [getattr(vzd.Button, SCENARIOS[scenario]["actions"][n][0]) for n in names]
    )
    game.init()
    return game, names


def nearest_target(state, scenario: str, screen_width: int, sticky: str | None = None):
    """Closest relevant object, as (name, bearing, centred, proximity).

    ``sticky`` is the object id the agent was tracking last tick. With two
    monsters flanking the player, raw "tallest label wins" flips target every
    tick and the agent oscillates instead of shooting, so we keep the previous
    target unless something is clearly closer.
    """
    wanted = SCENARIOS[scenario]["targets"]
    candidates = [
        label
        for label in state.labels
        if label.object_name in wanted or (wanted == {"Medikit"} and "medikit" in label.object_name.lower())
    ]
    if not candidates:
        return None

    # On-screen height is a serviceable proximity proxy; tallest == closest.
    label = max(candidates, key=lambda l: (l.height, -abs(l.x + l.width / 2 - screen_width / 2)))
    if sticky is not None:
        held = [c for c in candidates if str(c.object_id) == sticky]
        if held and label.height < held[0].height * 1.3:
            label = held[0]
    centre = screen_width / 2
    offset = (label.x + label.width / 2 - centre) / centre  # -1 left .. +1 right

    if offset < -0.5:
        bearing = "far to the left"
    elif offset < -0.12:
        bearing = "to the left"
    elif offset <= 0.12:
        bearing = "dead ahead"
    elif offset <= 0.5:
        bearing = "to the right"
    else:
        bearing = "far to the right"

    if label.height >= 60:
        proximity = "very close"
    elif label.height >= 30:
        proximity = "close"
    elif label.height >= 15:
        proximity = "medium range"
    else:
        proximity = "far away"

    return label.object_name, bearing, bearing == "dead ahead", proximity, str(label.object_id)


def describe_state(state, scenario: str, screen_width: int, sticky: str | None = None) -> tuple[str, str, str | None]:
    """Build the prompt.

    Two deliberate choices, both measured (see demo/README.md):

    * **Bearing is the decisive fact**, so it is the second-to-last line and the
      crosshair line is last. The state omits health, ammo and proximity: none of
      them change which action is correct, and including them cost ~28ms per
      decision (36-token tail ran 163ms vs 134ms for a 28-token tail) while
      scoring identically. Health and kills are still printed to the console.
    * **Wording mirrors the criteria** ("bearing relative to crosshair",
      "lined up in crosshair"). Paraphrasing it scored 17/18 where the exact
      wording scored 18/18 on the same states.
    """
    kind = SCENARIOS[scenario]["target_kind"]
    target = nearest_target(state, scenario, screen_width, sticky)
    subject = kind.capitalize()

    if target is None:
        text = f"No {kind} is visible on screen.\n{subject} lined up in crosshair: no."
        return text, f"no {kind} visible", None

    name, bearing, centred, proximity, object_id = target
    text = (
        f"{subject} bearing relative to crosshair: {bearing}.\n"
        f"{subject} lined up in crosshair: {'yes' if centred else 'no'}."
    )
    return text, f"{name} {bearing}, {proximity}", object_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 plays real Doom through ViZDoom, one letter per decision.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="defend_the_center",
        choices=list(SCENARIOS),
        help="which scenario to play (default: defend_the_center)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="open ViZDoom's window and render the gameplay while it plays "
        "(per-decision latency is unchanged; throughput drops ~20%%)",
    )
    args = parser.parse_args()
    scenario = args.scenario

    print_header(f"Real Doom via ViZDoom — scenario: {scenario}")
    print("Loading model (one-time cost)...")

    service = TypeSafeReplica()
    service.engine.load()
    print(f"Model loaded in {service.engine.load_seconds:.1f}s")

    game, action_names = build_game(scenario, watch=args.watch)
    screen_width = game.get_screen_width()
    criteria = {name: SCENARIOS[scenario]["actions"][name][1] for name in action_names}
    question = SCENARIOS[scenario]["question"]

    print(f"Engine: ViZDoom {vzd.__version__}   actions: {', '.join(action_names)}")
    print(f"Frame skip: {FRAME_SKIP} tics per decision")
    if args.watch:
        print(
            f"Watching: {game.get_screen_width()}x{game.get_screen_height()} window, "
            "HUD + crosshair on, all frames rendered."
        )
        print(
            "Note: rendering runs inside make_action, so per-decision latency is "
            "unchanged; wall-clock throughput drops ~20%."
        )

    # Warm the question's KV prefix so decision 1 is not the only cold call. The
    # prefix depends on the question and criteria only, never on the game state,
    # and this loop reuses it ~40 times, so the one-time prefill pays for itself.
    warm_ms = service.warm({"act": {"type": "choice", "instructions": question, "criteria": criteria}})
    print(f"Prefix warmed in {warm_ms:.0f}ms\n")

    game.new_episode()
    latency = LatencyLog("decision")
    chosen = {name: 0 for name in action_names}
    total_reward = 0.0
    decisions = 0
    sticky: str | None = None
    wall_started = time.perf_counter()

    while not game.is_episode_finished() and decisions < MAX_DECISIONS:
        state = game.get_state()
        if state is None:
            break
        state_text, digest, sticky = describe_state(state, scenario, screen_width, sticky)

        started = time.perf_counter()
        result = service.system_one(
            state=state_text,
            questions={"act": {"type": "choice", "instructions": question, "criteria": criteria}},
        )
        ms = latency.record(started)
        decisions += 1

        answer = result.answers["act"]
        action = answer["choice"]
        chosen[action] += 1

        total_reward += game.make_action(
            [1 if name == action else 0 for name in action_names], FRAME_SKIP
        )

        health = int(game.get_game_variable(vzd.GameVariable.HEALTH))
        kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        print(
            f"decision {decisions:2d}  action={action:<10} conf={answer['confidence']:.2f}  "
            f"latency={ms:6.1f}ms  health={health:3d} kills={kills}  | {digest[:40]}"
        )

    wall_s = time.perf_counter() - wall_started
    print(
        f"\nEpisode finished: {game.is_episode_finished()}   "
        f"total reward: {total_reward:.1f}   "
        f"kills: {int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))}"
    )
    print(latency.summary())
    print(
        f"throughput: {decisions / wall_s:.2f} decisions/sec over {wall_s:.1f}s wall, "
        f"covering {decisions * FRAME_SKIP / 35.0:.1f}s of game time"
    )
    print("action mix: " + ", ".join(f"{n}={c}" for n, c in chosen.items()))
    game.close()


if __name__ == "__main__":
    main()
