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
"""

from __future__ import annotations

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


def build_game(scenario: str) -> tuple[vzd.DoomGame, list[str]]:
    """Load the scenario, restricted to the buttons this demo exposes."""
    cfg = Path(vzd.__file__).parent / "scenarios" / f"{scenario}.cfg"
    game = vzd.DoomGame()
    game.load_config(str(cfg))
    game.set_window_visible(False)
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
    """Build the prompt. The decisive fact is deliberately the last line."""
    kind = SCENARIOS[scenario]["target_kind"]
    health, ammo, _kills = (int(v) for v in state.game_variables)
    target = nearest_target(state, scenario, screen_width, sticky)

    head = f"Doom. Health: {health}. Ammo: {ammo}."
    if target is None:
        text = (
            f"{head}\n"
            f"No {kind} is visible on screen.\n"
            f"Nearest {kind} bearing relative to crosshair: none visible.\n"
            f"{kind.capitalize()} lined up in crosshair: no."
        )
        return text, f"no {kind} visible", None

    name, bearing, centred, proximity, object_id = target
    text = (
        f"{head}\n"
        f"Nearest {kind}: {name} at {proximity}.\n"
        f"{kind.capitalize()} bearing relative to crosshair: {bearing}.\n"
        f"{kind.capitalize()} lined up in crosshair: {'yes' if centred else 'no'}."
    )
    return text, f"{name} {bearing}, {proximity}", object_id


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "defend_the_center"
    if scenario not in SCENARIOS:
        print(f"Unknown scenario {scenario!r}. Options: {', '.join(SCENARIOS)}")
        return

    print_header(f"Real Doom via ViZDoom — scenario: {scenario}")
    print("Loading model (one-time cost)...")

    service = TypeSafeReplica()
    service.engine.load()
    print(f"Model loaded in {service.engine.load_seconds:.1f}s")

    game, action_names = build_game(scenario)
    screen_width = game.get_screen_width()
    criteria = {name: SCENARIOS[scenario]["actions"][name][1] for name in action_names}
    question = SCENARIOS[scenario]["question"]

    print(f"Engine: ViZDoom {vzd.__version__}   actions: {', '.join(action_names)}")
    print(f"Frame skip: {FRAME_SKIP} tics per decision")

    # Warm the question's KV prefix so decision 1 is not the only cold call. The
    # prefix depends on the question and criteria only, never on the game state.
    from truetype.questions import build_question
    from truetype.render import render_example_prefix, render_target_block

    warm_q = build_question("act", {"type": "choice", "instructions": question, "criteria": criteria})
    t0 = time.perf_counter()
    service.engine.score_with_prefix(
        render_example_prefix(warm_q), [render_target_block(warm_q, "warmup")]
    )
    print(f"Prefix warmed in {(time.perf_counter()-t0)*1000:.0f}ms\n")

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
