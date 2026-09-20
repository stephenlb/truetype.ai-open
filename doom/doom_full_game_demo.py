"""Gemma 4 plays a full Doom level through ViZDoom.

The reference demo (``demo/doom_demo.py``) plays three bundled arena scenarios
with a fixed action set. This demo plays Freedoom MAP01 by default, or another
IWAD and map supplied by the user. It builds the state report from engine data.

Implementation
--------------
The demo builds each decision from engine objects rather than pixels. The report
includes monster species, bearing, distance, unseen damage, and clearance. It
does not require screen capture or OS permission.

The code selects a phase from the state, then asks the model one question for
that phase:

  ==============  ==================================================================
  ``combat``      monsters in view: shoot / advance / turn left / turn right / back off
  ``stuck``       a forward move was blocked: strafe left / strafe right / back away
  ``navigate``    nothing to fight and not stuck: forward / turn left / turn right
  ==============  ==================================================================

The engine supplies the level's blocking lines. The demo builds a walkability
grid, routes with BFS, and points the model toward the next waypoint instead of
the distant item. This avoids repeated direction changes at walls.

The forward band matches the state wording. "dead ahead" and "slightly
left/right" are inside the 25-degree forward band; "far left/right" requires a
turn. When the words and the band disagreed, a -22 degree waypoint read as
forward and sent the agent into the wall beside a doorway for 55 decisions.

A closed door does not appear in the object list, so the model cannot be asked
about one. Forward movement holds USE. A blocked move retries several times
before the state calls the player stuck because a door needs about four presses.
Treating the first failure as "stuck" made the agent strafe away from every door.

After two failed sideways attempts, the report records the failures, which makes
``back_up`` the expected answer. After four, the demo presses the engine's
TURN180 button and walks out the way it came. The button costs no inference.

When the selected weapon is out of ammo, the demo switches to the best owned
weapon with ammunition before asking the model.

``--self-test`` checks all three questions against synthetic states built by the
same functions as the live loop. It catches wording regressions without running
the engine.

Run (from the repo root):
    python doom/doom_full_game_demo.py                    # Freedoom 2 MAP01
    python doom/doom_full_game_demo.py --watch            # ...and watch it play
    python doom/doom_full_game_demo.py --map E1M1        # the cramped first level
    python doom/doom_full_game_demo.py --decisions 400 --episodes 2
    python doom/doom_full_game_demo.py --iwad doom2.wad --map MAP01 --skill 2

``--watch`` opens ViZDoom's own SDL window with HUD and crosshair. Rendering runs
inside ``make_action``, outside the timed decision, so per-decision latency is
unchanged while wall-clock throughput drops.

MAP01 is the default because it is open. On E1M1 a 300-decision run manages around
2 kills and 2 items; on MAP01 the same budget reaches 5 kills and 8 items with a
third of the stuck decisions, because routing has room to work and there are fewer
doorways to negotiate.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import vizdoom as vzd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from truetype.service import TypeSafeReplica
from common import LatencyLog, print_header

# --------------------------------------------------------------------------- names
# Species names are presented to the model as readable nouns. Anything not listed
# is reported by its engine name.
MONSTERS = {
    "Zombieman": "rifle zombie",
    "ShotgunGuy": "shotgun zombie",
    "ChaingunGuy": "chaingun zombie",
    "DoomImp": "imp",
    "Demon": "pinky demon",
    "Spectre": "partly invisible spectre",
    "Cacodemon": "cacodemon",
    "HellKnight": "hell knight",
    "BaronOfHell": "baron of hell",
    "LostSoul": "lost soul",
    "Revenant": "revenant",
    "Arachnotron": "arachnotron",
    "Mancubus": "mancubus",
    "PainElemental": "pain elemental",
    "Archvile": "archvile",
    "Cyberdemon": "cyberdemon",
    "SpiderMastermind": "spider mastermind",
    "WolfensteinSS": "SS guard",
}
PICKUPS = {
    "Medikit": "medkit",
    "Stimpack": "stimpack",
    "HealthBonus": "health bonus",
    "GreenArmor": "green armor",
    "BlueArmor": "blue armor",
    "ArmorBonus": "armor bonus",
    "Soulsphere": "soulsphere",
    "Megasphere": "megasphere",
    "Berserk": "berserk pack",
    "Chainsaw": "chainsaw",
    "Shotgun": "shotgun",
    "SuperShotgun": "super shotgun",
    "Chaingun": "chaingun",
    "RocketLauncher": "rocket launcher",
    "PlasmaRifle": "plasma rifle",
    "BFG9000": "BFG9000",
    "Clip": "ammo clip",
    "ClipBox": "box of bullets",
    "Shell": "shells",
    "ShellBox": "box of shells",
    "RocketAmmo": "rockets",
    "RocketBox": "box of rockets",
    "Cell": "energy cell",
    "CellPack": "cell pack",
    "Backpack": "backpack",
    "BlueCard": "blue keycard",
    "RedCard": "red keycard",
    "YellowCard": "yellow keycard",
    "BlueSkull": "blue skull key",
    "RedSkull": "red skull key",
    "YellowSkull": "yellow skull key",
}
HEALTH_PICKUPS = {"Medikit", "Stimpack", "HealthBonus", "Soulsphere", "Megasphere"}
AMMO_PICKUPS = {"Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack"}
GEAR_PICKUPS = {"GreenArmor", "BlueArmor", "ArmorBonus", "Berserk", "Chainsaw", "Shotgun",
                "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG9000", "Backpack"}
KEY_PICKUPS = {"BlueCard", "RedCard", "YellowCard", "BlueSkull", "RedSkull", "YellowSkull"}

# Dead things, gibs and decorations produce labels too; they are scenery.
SCENERY_PREFIXES = (
    "Dead", "Gibbed", "Gib", "Nonsolid", "Meat", "Blood", "Pool", "Stain",
    "BigTree", "TorchTree", "LiveStick", "Tree", "Bush",
)

# Weapon slot -> (display name, ammo game variable). ViZDoom reports ammo per
# weapon slot: AMMO{n} is the ammo pool weapon n draws from, so the pistol and
# chaingun both read AMMO2/AMMO4 against the same bullet pool. Measured on E1M1:
# the starting pistol reports 50 in AMMO2 and firing it decrements AMMO2 and
# AMMO4 together. Guessing a different variable read 0 and made the demo switch
# to fists while fully loaded.
WEAPONS = {
    1: ("fists", None),
    2: ("pistol", "AMMO2"),
    3: ("shotgun", "AMMO3"),
    4: ("chaingun", "AMMO4"),
    5: ("rocket launcher", "AMMO5"),
    6: ("plasma rifle", "AMMO6"),
    7: ("BFG9000", "AMMO7"),
}
# Preference order for the automatic switch. Hitscan close-range weapons first;
# rockets are deliberately low because the level is close quarters.
WEAPON_PREFERENCE = [3, 4, 2, 6, 5, 7, 1]

# ------------------------------------------------------------------------- actions
# Button order is fixed for the whole run; action vectors are built against it.
BUTTONS = [
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,
    vzd.Button.SELECT_WEAPON1,
    vzd.Button.SELECT_WEAPON2,
    vzd.Button.SELECT_WEAPON3,
    vzd.Button.SELECT_WEAPON4,
    vzd.Button.SELECT_WEAPON5,
    vzd.Button.SELECT_WEAPON6,
    vzd.Button.SELECT_WEAPON7,
    vzd.Button.TURN180,
]

# Each macro is (buttons, engine tics). TURN moves the view ~17 degrees (3.5
# degrees per tic measured in E1M1), a forward macro covers ~40 units, and ATTACK
# needs ~20 tics to complete the weapon raise and fire once (measured: 16 tics is
# still the raise, 32 tics is two shots), so ``shoot`` is deliberately long.
TURN_TICS = 5
MOVE_TICS = 8
COMBAT_MOVE_TICS = 6
SHOOT_TICS = 20
TURN_DEG_PER_TIC = 3.516  # from a 40-tic calibration sweep in E1M1

COMBAT_ACTIONS: dict[str, tuple[dict, int]] = {
    "shoot": ({vzd.Button.ATTACK: 1}, SHOOT_TICS),
    "turn_left": ({vzd.Button.TURN_LEFT: 1}, TURN_TICS),
    "turn_right": ({vzd.Button.TURN_RIGHT: 1}, TURN_TICS),
    "advance": ({vzd.Button.MOVE_FORWARD: 1}, COMBAT_MOVE_TICS),
    "back_up": ({vzd.Button.MOVE_BACKWARD: 1}, COMBAT_MOVE_TICS),
}

STUCK_ACTIONS: dict[str, tuple[dict, int]] = {
    "strafe_left": ({vzd.Button.MOVE_LEFT: 1}, MOVE_TICS),
    "strafe_right": ({vzd.Button.MOVE_RIGHT: 1}, MOVE_TICS),
    "back_up": ({vzd.Button.MOVE_BACKWARD: 1}, MOVE_TICS),
}

NAVIGATION_ACTIONS: dict[str, tuple[dict, int]] = {
    # USE rides along with forward movement. A closed door is not in the object
    # list, so the model cannot see one; holding USE while walking means doors open
    # as the player reaches them. Measured on E1M1: plain forward stopped dead at
    # the first door for 19 macro steps, forward+USE pushed through it in 4.
    "forward": ({vzd.Button.MOVE_FORWARD: 1, vzd.Button.USE: 1}, MOVE_TICS),
    "turn_left": ({vzd.Button.TURN_LEFT: 1}, TURN_TICS),
    "turn_right": ({vzd.Button.TURN_RIGHT: 1}, TURN_TICS),
}

# ------------------------------------------------------------------------- prompts
COMBAT_CRITERIA = {
    # 350 units, not 700: the pistol's spread means a distant monster is a 3-pixel
    # sprite and shots miss it. A 700-unit "shoot" rule produced 229 shots and no
    # kills; at 350 the agent closes to a range where its hitscan lands.
    # "close range" / "far away" match the words in the state report exactly.
    "shoot": "Fire - monster at close range, under 350 units, health 30 or more",
    "advance": "Walk forward - monster far away, over 350 units, health 30 or more",
    "turn_left": "Turn left to aim - monster far left of the crosshair, health 30 or more",
    "turn_right": "Turn right to aim - monster far right of the crosshair, health 30 or more",
    "back_up": "Back away, do not fight - health below 30",
}

STUCK_CRITERIA = {
    "strafe_left": (
        "Step left - the way past the obstacle continues to the left, and the "
        "player has not been stuck long"
    ),
    "strafe_right": (
        "Step right - the way past the obstacle continues to the right, and the "
        "player has not been stuck long"
    ),
    "back_up": (
        "Back away and turn - the player has been stuck here for several decisions "
        "and stepping sideways has failed, or neither side is open"
    ),
}

NAVIGATION_CRITERIA = {
    # The forward band is 25 degrees and the state's words are chosen to match it
    # exactly: "dead ahead" and "slightly left/right" are forward, "far left/right"
    # is a turn. An earlier 30-degree band disagreed with the words - a bearing of
    # -22 degrees was reported as "slightly right", read as forward, and the agent
    # pushed into the wall beside a doorway for 55 decisions. Numbers and words now
    # cannot conflict. The confidence jump (0.06 to 0.6+) confirms the model reads
    # the two consistently.
    "forward": (
        "Walk forward - the objective or next waypoint is dead ahead or slightly "
        "left or slightly right, bearing between -25 and +25 degrees"
    ),
    "turn_left": (
        "Turn left - the objective or next waypoint is far left of the crosshair, "
        "bearing above +25 degrees"
    ),
    "turn_right": (
        "Turn right - the objective or next waypoint is far right of the crosshair, "
        "bearing below -25 degrees"
    ),
}

CONTRACT = {
    "combat": {
        "type": "choice",
        "instructions": "A monster is visible. Which action should the player take?",
        "criteria": COMBAT_CRITERIA,
    },
    "stuck": {
        "type": "choice",
        "instructions": (
            "The player walked into something directly ahead and cannot get through. "
            "Which action should the player take?"
        ),
        "criteria": STUCK_CRITERIA,
    },
    "navigate": {
        "type": "choice",
        "instructions": (
            "No monster is visible and the player is free to move. "
            "Which action should the player take?"
        ),
        "criteria": NAVIGATION_CRITERIA,
    },
}

# ----------------------------------------------------------------------- geometry
# The engine reports angles counterclockwise from +X (verified on E1M1: TURN_LEFT
# increases ANGLE, TURN_RIGHT decreases it), matching atan2(dy, dx) in Doom's
# coordinate space. A bearing is that angle minus the player's facing, so a
# positive bearing means the object sits to the player's **left**.
MAX_RAY = 640.0  # world units; long enough to compare side openings


def _bearing_to(px: float, py: float, facing: float, x: float, y: float) -> float:
    """Signed bearing in degrees from the player's facing to a world point."""
    dx, dy = x - px, y - py
    return (math.degrees(math.atan2(dy, dx)) - facing + 180.0) % 360.0 - 180.0


def ray_clearance(segments, px: float, py: float, direction: float, limit: float = MAX_RAY) -> float:
    """Distance from (px, py) to the nearest blocking line along ``direction``.

    ``segments`` is an iterable of (x1, y1, x2, y2) world-space blocking lines. A
    plain 2D segment/ray intersection is enough for clearance comparisons; no
    attempt is made to model heights, and a miss reads as ``limit``.
    """
    dx = math.cos(math.radians(direction))
    dy = math.sin(math.radians(direction))
    best = limit
    for x1, y1, x2, y2 in segments:
        ex, ey = x2 - x1, y2 - y1
        denominator = dx * ey - dy * ex
        if abs(denominator) < 1e-9:
            continue
        t = ((x1 - px) * ey - (y1 - py) * ex) / denominator
        s = ((x1 - px) * dy - (y1 - py) * dx) / denominator
        if 0.0 < t < best and 0.0 <= s <= 1.0:
            best = t
    return best


def has_line_of_sight(segments, px: float, py: float, x: float, y: float) -> bool:
    """True when no blocking line crosses the straight segment to (x, y).

    The ray cast is reused by aiming it along the bearing to the target and
    comparing the hit distance with the target distance. Without this the demo
    walks at pickups through walls: the object list reports everything within
    range regardless of geometry, and the model has no way to know better.
    """
    distance = math.hypot(x - px, y - py)
    if distance < 1.0:
        return True
    direction = math.degrees(math.atan2(y - py, x - px))
    return ray_clearance(segments, px, py, direction, limit=distance + 1.0) > distance


def blocking_segments(state) -> list[tuple[float, float, float, float]]:
    """Unique blocking lines from the sector geometry, for the ray cast."""
    seen: set[tuple[float, float, float, float]] = set()
    segments: list[tuple[float, float, float, float]] = []
    for sector in state.sectors or []:
        for line in sector.lines:
            if not line.is_blocking:
                continue
            key = (round(line.x1, 1), round(line.y1, 1), round(line.x2, 1), round(line.y2, 1))
            if key in seen:
                continue
            seen.add(key)
            segments.append((line.x1, line.y1, line.x2, line.y2))
    return segments


# ------------------------------------------------------------------- pathfinding
# The engine hands over every blocking line in the level, so a real route can be
# planned instead of reacted to. The reactive version - "turn toward the widest
# nearby gap" - is what made earlier runs shuffle against walls: it re-decides a
# local direction every tick and has no memory of where it was going. A grid plus
# BFS over the static geometry produces an actual path, and the state then reports
# the next waypoint on that path, which is a far easier thing for the model to
# follow ("head that way") than a wall-avoidance judgement.

GRID_CELL = 32.0  # world units per grid cell
PLAYER_RADIUS = 20.0  # clearance kept from walls when marking cells walkable

# Forward moves that may fail before the state calls the player stuck. A closed
# door needs about four presses of forward+USE before it opens, so treating the
# first failure as "stuck" made the agent strafe away from every door in the level
# instead of opening it.
DOOR_RETRIES = 4


def _point_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


class NavGrid:
    """A walkability grid over the level's blocking lines, with BFS routing.

    Built once per level (the geometry never changes), then queried per decision.
    A cell is walkable when its centre is at least ``PLAYER_RADIUS`` from every
    blocking line.
    """

    def __init__(
        self,
        segments: list[tuple[float, float, float, float]],
        *,
        cell: float = GRID_CELL,
        radius: float = PLAYER_RADIUS,
    ) -> None:
        self.cell = cell
        self.radius = radius
        if not segments:
            self.min_x = self.min_y = 0.0
            self.width = self.height = 1
            self.walkable = [[True]]
            return
        xs = [x for seg in segments for x in (seg[0], seg[2])]
        ys = [y for seg in segments for y in (seg[1], seg[3])]
        pad = cell * 2.0
        self.min_x, self.min_y = min(xs) - pad, min(ys) - pad
        self.width = int((max(xs) - min(xs) + 2 * pad) / cell) + 1
        self.height = int((max(ys) - min(ys) + 2 * pad) / cell) + 1

        self.walkable = [[True] * self.width for _ in range(self.height)]
        limit = radius + cell * 0.71
        for x1, y1, x2, y2 in segments:
            lo_x = int(max(0, (min(x1, x2) - limit - self.min_x) / cell))
            hi_x = int(min(self.width - 1, (max(x1, x2) + limit - self.min_x) / cell))
            lo_y = int(max(0, (min(y1, y2) - limit - self.min_y) / cell))
            hi_y = int(min(self.height - 1, (max(y1, y2) + limit - self.min_y) / cell))
            for gy in range(lo_y, hi_y + 1):
                cy = self.min_y + (gy + 0.5) * cell
                row = self.walkable[gy]
                for gx in range(lo_x, hi_x + 1):
                    if not row[gx]:
                        continue
                    cx = self.min_x + (gx + 0.5) * cell
                    if _point_segment_distance(cx, cy, x1, y1, x2, y2) < radius:
                        row[gx] = False

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.min_x) / self.cell)
        gy = int((y - self.min_y) / self.cell)
        return (
            max(0, min(self.width - 1, gx)),
            max(0, min(self.height - 1, gy)),
        )

    def _is_walkable_xy(self, x: float, y: float) -> bool:
        gx, gy = self.cell_of(x, y)
        return self.walkable[gy][gx]

    def _nearest_free(self, gx: int, gy: int) -> tuple[int, int] | None:
        if self.walkable[gy][gx]:
            return gx, gy
        for ring in range(1, 6):
            for dy in range(-ring, ring + 1):
                for dx in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and self.walkable[ny][nx]:
                        return nx, ny
        return None

    def route(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        """A list of world-space waypoints from start to goal, or [] if unreachable.

        The raw BFS path is string-pulled: consecutive cells are replaced by the
        furthest cell still visible from the current one, so the result is a handful
        of corner waypoints rather than hundreds of cell centres. Line of sight is
        checked against the grid with a coarse ray march.
        """
        from collections import deque

        start_cell = self._nearest_free(*self.cell_of(*start))
        goal_cell = self._nearest_free(*self.cell_of(*goal))
        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return [goal]

        predecessor: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        queue: deque[tuple[int, int]] = deque([start_cell])
        found = False
        while queue and not found:
            current = queue.popleft()
            cx, cy = current
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cx + dx, cy + dy)
                if nxt in predecessor:
                    continue
                nx, ny = nxt
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if not self.walkable[ny][nx]:
                    continue
                predecessor[nxt] = current
                if nxt == goal_cell:
                    found = True
                    break
                queue.append(nxt)

        if not found:
            return []
        cells: list[tuple[int, int]] = []
        node: tuple[int, int] | None = goal_cell
        while node is not None:
            cells.append(node)
            node = predecessor[node]
        cells.reverse()

        points = [
            (self.min_x + (gx + 0.5) * self.cell, self.min_y + (gy + 0.5) * self.cell)
            for gx, gy in cells
        ]
        points[-1] = goal
        return self._string_pull(points)

    def _grid_line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Coarse walkability check along the segment a-b, at half-cell steps."""
        distance = math.dist(a, b)
        steps = max(1, int(distance / (self.cell * 0.5)))
        for i in range(1, steps):
            t = i / steps
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if not self._is_walkable_xy(x, y):
                return False
        return True

    def _string_pull(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points
        pulled = [points[0]]
        i = 0
        while i < len(points) - 1:
            furthest = i + 1
            for j in range(len(points) - 1, i, -1):
                if self._grid_line_of_sight(points[i], points[j]):
                    furthest = j
                    break
            pulled.append(points[furthest])
            i = furthest
        return pulled[1:]

    def next_waypoint(
        self, start: tuple[float, float], goal: tuple[float, float], *, min_distance: float = 90.0
    ) -> tuple[float, float] | None:
        """The furthest route point still within ``min_distance`` of the player.

        Reporting a point a few cells ahead rather than the final goal keeps the
        model's instruction local and actionable; reporting the far goal makes it
        walk into the wall between.
        """
        route = self.route(start, goal)
        if not route:
            return None
        chosen = route[-1]
        for point in route:
            if math.dist(start, point) <= min_distance:
                chosen = point
            else:
                break
        return chosen


# ----------------------------------------------------------------------- snapshot
@dataclass
class Contact:
    """One visible object reduced to the facts the model reasons over."""

    object_id: int
    name: str
    kind: str  # monster | health | ammo | gear | key
    bearing: float  # signed degrees relative to facing, positive = left
    distance: float
    category: str = ""
    reachable: bool = True  # no wall between the player and it right now
    # World position, kept for pathfinding. Monsters and pickups both carry it so
    # routes can be planned without a second lookup in the object list.
    pickup_x: float = 0.0
    pickup_y: float = 0.0

    @property
    def label(self) -> str:
        return MONSTERS.get(self.name) or PICKUPS.get(self.name) or self.name


@dataclass
class Snapshot:
    """Everything the prompt builders need, sampled once per decision."""

    tick: int
    health: int
    armor: int
    ammo: int
    weapon: int
    kills: int
    items: int
    dead: bool
    bearing: float  # facing in degrees
    x: float
    y: float
    damage_now: int
    monsters: list[Contact] = field(default_factory=list)
    pickups: list[Contact] = field(default_factory=list)
    left_clearance: float = MAX_RAY
    right_clearance: float = MAX_RAY
    back_clearance: float = MAX_RAY
    # The pickup being walked toward, chosen with stickiness so it does not flip
    # between two similar items every tick (armor bonus <-> berserk pack churn was
    # the largest single source of wasted decisions in the E1M1 runs).
    objective: Contact | None = None
    # The next waypoint on the BFS route to the objective, in world space. This is
    # what the model is pointed at, rather than the objective itself, so its
    # instruction is a local "walk that way" instead of a distant bearing that
    # would send it into the wall between.
    waypoint: tuple[float, float] | None = None


@dataclass
class NavigationState:
    """Sticky choices that survive across decisions.

    ``objective_id`` keeps the agent committed to one pickup instead of re-picking
    the nearest each tick, which used to make it alternate between two similar
    items and walk at the midpoint, reaching neither.
    """

    objective_id: int | None = None


def build_snapshot(
    game: vzd.DoomGame,
    state,
    *,
    px: float,
    py: float,
    facing: float,
    sticky_id: int | None,
    segments,
    abandoned: set[int] | None = None,
    navigation: NavigationState | None = None,
    grid: NavGrid | None = None,
) -> Snapshot:
    """Reduce one engine frame to the contacts, status and clearance report.

    Monsters come from the *labels* buffer, which only holds objects the engine
    drew this frame, meaning what the player can see. Pickups come from the
    full object list so an objective behind the player still steers the next turn.
    """
    health = int(game.get_game_variable(vzd.GameVariable.HEALTH))
    armor = int(game.get_game_variable(vzd.GameVariable.ARMOR))
    ammo = int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO))
    weapon = int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON))
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    items = int(game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
    dead = bool(game.get_game_variable(vzd.GameVariable.DEAD))
    damage_now = int(game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN))

    objects_by_id = {obj.id: obj for obj in (state.objects or [])}

    monsters: list[Contact] = []
    for label in state.labels:
        if label.object_category != "Monster" or label.object_id not in objects_by_id:
            continue
        obj = objects_by_id[label.object_id]
        monsters.append(
            Contact(
                object_id=obj.id,
                name=obj.name,
                kind="monster",
                category=label.object_category,
                bearing=_bearing_to(px, py, facing, obj.position_x, obj.position_y),
                distance=math.dist((px, py), (obj.position_x, obj.position_y)),
            )
        )
    # Several labels can be emitted for one object; keep one contact each.
    monsters = list({c.object_id: c for c in monsters}.values())

    if sticky_id is not None:
        # Keep the previous target unless another monster is clearly closer. With
        # two monsters flanking the player, raw "nearest wins" flips every tick and
        # the agent oscillates instead of committing to one.
        held = next((c for c in monsters if c.object_id == sticky_id), None)
        best = min(monsters, key=lambda c: c.distance, default=None)
        if held is not None and best is not None and best.object_id != held.object_id:
            if best.distance > held.distance * 0.75:
                monsters.sort(key=lambda c: (c.object_id != sticky_id, c.distance))

    pickups: list[Contact] = []
    for obj in objects_by_id.values():
        if obj.name in MONSTERS:
            continue
        if abandoned and obj.id in abandoned:
            continue
        if any(obj.name.startswith(prefix) for prefix in SCENERY_PREFIXES):
            continue
        kind = None
        if obj.name in HEALTH_PICKUPS:
            kind = "health"
        elif obj.name in AMMO_PICKUPS:
            kind = "ammo"
        elif obj.name in GEAR_PICKUPS:
            kind = "gear"
        elif obj.name in KEY_PICKUPS:
            kind = "key"
        if kind is None:
            continue
        distance = math.dist((px, py), (obj.position_x, obj.position_y))
        if distance > 1600:
            continue
        pickups.append(
            Contact(
                object_id=obj.id,
                name=obj.name,
                kind=kind,
                bearing=_bearing_to(px, py, facing, obj.position_x, obj.position_y),
                distance=distance,
                reachable=has_line_of_sight(segments, px, py, obj.position_x, obj.position_y),
                pickup_x=obj.position_x,
                pickup_y=obj.position_y,
            )
        )
    pickups = list({c.object_id: c for c in pickups}.values())
    pickups.sort(key=lambda c: c.distance)

    # Clearance uses the engine's angle convention: +90 degrees is the player's
    # left, -90 the right. Swapping these hands the stuck question a mirrored
    # world, and the model then walks into the wall it was told was open.
    left = ray_clearance(segments, px, py, facing + 90.0)
    right = ray_clearance(segments, px, py, facing - 90.0)
    back = ray_clearance(segments, px, py, facing + 180.0)

    snap = Snapshot(
        tick=state.tic,
        health=health,
        armor=armor,
        ammo=ammo,
        weapon=weapon,
        kills=kills,
        items=items,
        dead=dead,
        bearing=facing,
        x=px,
        y=py,
        damage_now=damage_now,
        monsters=monsters,
        pickups=pickups,
        left_clearance=left,
        right_clearance=right,
        back_clearance=back,
    )
    if navigation is not None and grid is not None:
        snap.objective = choose_objective(snap, navigation)
        if snap.objective is None:
            snap.objective = explore_objective(snap, grid)
        # Point the model at the objective directly when it is visible, and at the
        # next BFS waypoint when it is not. One route plan per tick, on the single
        # chosen objective - planning every candidate was what made navigation
        # ~30ms slower.
        if snap.objective is not None:
            target = (snap.objective.pickup_x, snap.objective.pickup_y)
            if snap.objective.reachable:
                snap.waypoint = target
            else:
                snap.waypoint = grid.next_waypoint((px, py), target)
                if snap.waypoint is None:
                    # The route is sealed off, so drop it instead of walking at it.
                    navigation.objective_id = None
                    snap.objective = explore_objective(snap, grid)
                    if snap.objective is not None:
                        snap.waypoint = (snap.objective.pickup_x, snap.objective.pickup_y)
    return snap


def choose_objective(snap: Snapshot, navigation: NavigationState) -> Contact | None:
    """The pickup to walk toward: the nearest one with a clear line of sight.

    Ranked by straight-line distance because it is cheap. Planning a BFS route for
    every visible pickup on every tick pushed navigation latency from ~130ms to
    ~159ms, and the extra accuracy is wasted because the winner is re-planned next
    tick. Pathfinding is reserved for the single chosen objective, and only when it
    is *not* directly visible (see ``build_snapshot``).

    Line-of-sight ``reachable`` items are preferred over anything else: the model
    follows bearings, and pointing it at an item through a wall is what produced
    wall-hugging loops. If nothing is visible the nearest overall is still tried,
    routed properly, and dropped on the first unreachable verdict.

    The choice is sticky: the previous objective keeps the lead while it is within
    1.4x of the best candidate's distance, so the agent does not oscillate between
    two similar items and walk at the midpoint.
    """
    def candidates(kind: str) -> list[Contact]:
        return [c for c in snap.pickups if c.kind == kind]

    pool: list[Contact] = []
    if snap.health < 60:
        pool = candidates("health")
    if not pool and snap.ammo < 25:
        pool = candidates("ammo")
    if not pool:
        pool = candidates("gear") + candidates("key")
    if not pool:
        return None

    visible = [c for c in pool if c.reachable]
    ranked = visible or pool
    best = min(ranked, key=lambda c: c.distance)
    held = next((c for c in ranked if c.object_id == navigation.objective_id), None)
    if held is not None and best.distance > held.distance * 0.7:
        return held
    return best


def explore_objective(snap: Snapshot, grid: NavGrid) -> Contact | None:
    """A reachable floor cell to walk to when no pickup is worth going for.

    Standing still is the worst outcome, so the agent keeps moving. The target is
    chosen from the grid rather than from nearby geometry: repeatedly pick the
    furthest reachable cell that is *not* almost straight back the way the player
    came, which walks the level forward instead of shuffling at a wall. Because it
    is a grid cell, routes to it are guaranteed to exist.
    """
    start = (snap.x, snap.y)
    candidates: list[tuple[float, tuple[float, float]]] = []
    for gy in range(2, grid.height - 2, 6):
        for gx in range(2, grid.width - 2, 6):
            if not grid.walkable[gy][gx]:
                continue
            x = grid.min_x + (gx + 0.5) * grid.cell
            y = grid.min_y + (gy + 0.5) * grid.cell
            distance = math.dist(start, (x, y))
            if distance < 300.0:
                continue
            bearing = _bearing_to(snap.x, snap.y, snap.bearing, x, y)
            if abs(bearing) > 120.0:
                continue  # do not walk back over ground already covered
            candidates.append((distance, (x, y)))
    if not candidates:
        return None
    target = max(candidates, key=lambda item: item[0])[1]
    waypoint = grid.next_waypoint(start, target)
    if waypoint is None:
        return None
    return Contact(
        object_id=-1,
        name="Exploration",
        kind="explore",
        bearing=_bearing_to(snap.x, snap.y, snap.bearing, *waypoint),
        distance=math.dist(start, waypoint),
        reachable=True,
        # The waypoint is the target for an exploration leg; without this the
        # Contact's default (0, 0) would send the agent to the map origin.
        pickup_x=waypoint[0],
        pickup_y=waypoint[1],
    )


def bearing_words(bearing: float) -> str:
    """Human words for a bearing from ``_bearing_to``.

    The engine's facing angle increases counterclockwise (turning left), so a
    bearing computed as ``object_angle - facing`` is **positive when the object is
    to the player's left** and negative when it is to the right. Getting this
    backwards is not a cosmetic bug: the model would turn right on a left-side
    target, which pushes the bearing further positive, and it orbits forever.
    """
    magnitude = abs(bearing)
    if magnitude < 8:
        return "dead ahead"
    side = "left" if bearing > 0 else "right"
    # The 25-degree cut matches NAVIGATION_CRITERIA exactly: "slightly" is inside the
    # forward band, "far" is outside it. Everything past 90 degrees is still "far
    # left/right" rather than "behind", because a "behind the player on the left"
    # phrase tested badly - the model answered ``advance`` on a monster at +130
    # degrees when the wording said "behind", while "far left (+130 degrees)"
    # answers ``turn_left``.
    return f"slightly {side}" if magnitude < 25 else f"far {side}"


# ------------------------------------------------------------------ state reports
def range_words(distance: float) -> str:
    """Word for the decision-relevant range band.

    350 units is where the pistol's spread stops landing hits on a distant sprite,
    so it is the shoot/advance split. Naming the band in the state ("close range,
    under 350 units") scored 5/5 where the raw distance alone scored 4/5, and the
    gap widened with range: a 1900-unit monster reported as "1900 units away"
    still drew ``shoot`` at 0.52 confidence.
    """
    return "close range, under 350 units" if distance < 350 else "far away, over 350 units"


def combat_state(snap: Snapshot) -> str:
    """The nearest monster is the decisive fact, so it is the last line.

    The health/weapon line is omitted on purpose. It is not just dead weight: with
    "The selected weapon is pistol with 30 shots." present, the model answered
    ``shoot`` on a monster 1900 units away, and without it the same state answered
    ``advance``. Nothing in the criteria depends on the weapon, so the line only
    biases the decision.
    """
    nearest = snap.monsters[0]
    lines = [
        f"Health {snap.health}.",
        f"Nearest monster: {nearest.label}, {range_words(nearest.distance)}, "
        f"{bearing_words(nearest.bearing)} ({nearest.bearing:+.0f} degrees).",
    ]
    if len(snap.monsters) > 1:
        second = snap.monsters[1]
        lines.append(
            f"Also visible: {second.label}, {range_words(second.distance)}, "
            f"{bearing_words(second.bearing)} ({second.bearing:+.0f} degrees)."
        )
    return "\n".join(lines)


def stuck_state(snap: Snapshot, stuck_for: int) -> str:
    """The physical situation, and nothing else.

    The objective is deliberately absent: the stuck question is "which way is
    physically open", and naming a pickup pulled the answer toward whichever
    direction the objective happened to lie (the self-test's strafe-right case
    regressed to ``back_up`` the moment the line was added). Clearance and how long
    the player has been stuck are the only facts that bear on the choice.
    """
    lines = []
    if snap.left_clearance > snap.right_clearance * 1.2:
        lines.append("The path continues to the left.")
    elif snap.right_clearance > snap.left_clearance * 1.2:
        lines.append("The path continues to the right.")
    else:
        lines.append("Both sides are wall.")
    if snap.back_clearance > 160:
        lines.append("The way back is open.")
    if stuck_for >= 2:
        # On the first two ticks the model is allowed to strafe; after that the
        # state says plainly that sideways has failed, which is what unlocks
        # ``back up``. Without this the agent strafed into the same wall forever.
        lines.append(
            f"The player has been stuck here for {stuck_for} decisions and "
            "stepping sideways has not freed them."
        )
    else:
        lines.append("The player just walked into something directly ahead and stopped.")
    return "\n".join(lines)


def navigation_state(snap: Snapshot, damage_this_tick: int = 0) -> str:
    """Short enough to stay fast, with the thing to walk at as the only bearing.

    An earlier version reported both the total distance to the objective and the
    waypoint distance. That is a longer tail (~160ms vs ~130ms warm) and the extra
    number is not decisive: the waypoint is the only thing the player can act on
    this tick. The objective is named so the instruction has a purpose, and the
    waypoint supplies the single bearing.
    """
    objective = snap.objective
    if objective is None:
        lines = ["Objective: nothing visible to walk to."]
    elif snap.waypoint is not None and not objective.reachable:
        wx, wy = snap.waypoint
        bearing = _bearing_to(snap.x, snap.y, snap.bearing, wx, wy)
        lines = [
            f"The {objective.label} is behind a wall. Next waypoint: "
            f"{bearing_words(bearing)} ({bearing:+.0f} degrees), "
            f"{int(round(math.dist((snap.x, snap.y), (wx, wy))))} units."
        ]
    elif objective.kind == "explore":
        lines = [
            f"Objective: explore onward, {bearing_words(objective.bearing)} "
            f"({objective.bearing:+.0f} degrees)."
        ]
    else:
        lines = [
            f"Objective: {objective.label}, {bearing_words(objective.bearing)} "
            f"({objective.bearing:+.0f} degrees), "
            f"{int(round(objective.distance))} units."
        ]
    if damage_this_tick > 0:
        # Being shot at from off-screen is the one fact the object list cannot
        # show, and it is decisive: with no monster to face, the right move is to
        # keep moving rather than stand and scan.
        lines.append("The player is taking damage from an attacker that is not in view.")
    return "\n".join(lines)


# --------------------------------------------------------------- weapon handling
def ammo_for(game: vzd.DoomGame, slot: int) -> int:
    variable = WEAPONS.get(slot, (None, None))[1]
    if variable is None:
        return 999  # fists never run out
    return int(game.get_game_variable(getattr(vzd.GameVariable, variable)))


def ensure_usable_weapon(game: vzd.DoomGame) -> str | None:
    """Select the best owned weapon with ammo when the current one is dry.

    Fists are the fallback rather than a valid choice: picking up a Berserk pack
    makes the engine switch to fists on its own, and treating "fists never run
    out" as "fists are fine" left the player punching monsters while 50 bullets
    sat in the clip. Returns a note when a switch happened, else None.
    """
    current = int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON))
    if current != 1 and ammo_for(game, current) > 0:
        return None
    best = next(
        (
            slot
            for slot in WEAPON_PREFERENCE
            if game.get_game_variable(getattr(vzd.GameVariable, f"WEAPON{slot}")) > 0
            and ammo_for(game, slot) > 0
        ),
        None,
    )
    if best is None or best == current:
        return None
    values = [0] * len(BUTTONS)
    values[BUTTONS.index(getattr(vzd.Button, f"SELECT_WEAPON{best}"))] = 1
    game.make_action(values, 6)
    return f"switched to {WEAPONS[best][0]}"


def fine_aim(
    game: vzd.DoomGame, bearing: float, *, max_ticks: int = 4
) -> None:
    """Nudge the crosshair onto the target before a shot.

    Doom auto-aims vertically but not horizontally, so a monster at +6 degrees and
    700 units is ~75 units off the axis and every shot passes it. The model still
    decides *when* to shoot; this only removes the residual aim error, the way a
    human nudges the mouse before pulling the trigger. Bearings are positive to the
    player's left, matching the engine's angle direction.

    Engine notes: ``TURN_LEFT_RIGHT_DELTA`` is reported at max value 0 in this
    ViZDoom build and does nothing, so the correction is a short press of the
    discrete turn button. Turning while ATTACK is held also works and fires, so the
    nudge and the first tics of the shot are one action.
    """
    ticks = min(max_ticks, max(0, int(round(abs(bearing) / TURN_DEG_PER_TIC))))
    if ticks == 0:
        return
    button = vzd.Button.TURN_LEFT if bearing > 0 else vzd.Button.TURN_RIGHT
    values = [0] * len(BUTTONS)
    values[BUTTONS.index(button)] = 1
    values[BUTTONS.index(vzd.Button.ATTACK)] = 1
    game.make_action(values, ticks)


def to_button_values(action: str, actions: dict[str, tuple[dict, int]]) -> tuple[list[int], int]:
    buttons, tics = actions[action]
    values = [0] * len(BUTTONS)
    for button, value in buttons.items():
        values[BUTTONS.index(button)] = value
    return values, tics


# ------------------------------------------------------------------ question flow
def next_phase(snap: Snapshot, stuck: bool) -> str:
    if snap.monsters:
        return "combat"
    if stuck:
        return "stuck"
    return "navigate"


def action_table(phase: str) -> dict[str, tuple[dict, int]]:
    return {"combat": COMBAT_ACTIONS, "stuck": STUCK_ACTIONS, "navigate": NAVIGATION_ACTIONS}[phase]


# ---------------------------------------------------------------------- self test
# Synthetic situations run through the same builders the live loop uses, so the
# strings under test are the strings sent to the model. A hand-written copy
# of the wording drifts the moment range bands or bearing words change, and a test
# that checks a paraphrase proves nothing.
def _combat_case(health: int, distance: float, bearing: float) -> str:
    contact = Contact(object_id=1, name="Zombieman", kind="monster",
                      bearing=bearing, distance=distance)
    snap = Snapshot(
        tick=0, health=health, armor=0, ammo=30, weapon=2, kills=0, items=0,
        dead=False, bearing=0.0, x=0.0, y=0.0, damage_now=0, monsters=[contact],
    )
    return combat_state(snap)


def _stuck_case(left: float, right: float, stuck_for: int) -> str:
    objective = Contact(object_id=1, name="Medikit", kind="health",
                        bearing=0.0, distance=300.0, reachable=True)
    snap = Snapshot(
        tick=0, health=100, armor=0, ammo=30, weapon=2, kills=0, items=0,
        dead=False, bearing=0.0, x=0.0, y=0.0, damage_now=0,
        pickups=[objective], objective=objective,
        left_clearance=left, right_clearance=right, back_clearance=300.0,
    )
    return stuck_state(snap, stuck_for)


def _navigation_case(units: float, bearing: float, *, waypoint: bool = False) -> str:
    # Health 50 (< 60) makes the medkit the chosen objective, matching how the live
    # loop picks health pickups first when hurt.
    objective = Contact(object_id=1, name="Medikit", kind="health",
                        bearing=bearing, distance=units, reachable=not waypoint)
    snap = Snapshot(
        tick=0, health=50, armor=0, ammo=30, weapon=2, kills=0, items=0,
        dead=False, bearing=0.0, x=0.0, y=0.0, damage_now=0,
        pickups=[objective], objective=objective,
    )
    if waypoint:
        snap.waypoint = (units * math.cos(math.radians(bearing)),
                         units * math.sin(math.radians(bearing)))
    return navigation_state(snap)


SELF_TEST_CASES: list[tuple[str, str, str]] = [
    ("combat", _combat_case(100, 340.0, 2.0), "shoot"),
    ("combat", _combat_case(100, 340.0, -5.0), "shoot"),
    ("combat", _combat_case(100, 615.0, 3.0), "advance"),
    ("combat", _combat_case(100, 1900.0, -3.0), "advance"),
    ("combat", _combat_case(100, 900.0, 55.0), "turn_left"),
    ("combat", _combat_case(100, 500.0, -40.0), "turn_right"),
    ("combat", _combat_case(100, 1000.0, 130.0), "turn_left"),
    ("combat", _combat_case(100, 1000.0, -150.0), "turn_right"),
    ("combat", _combat_case(22, 90.0, 0.0), "back_up"),
    ("combat", _combat_case(18, 300.0, 30.0), "back_up"),
    ("stuck", _stuck_case(400.0, 40.0, 0), "strafe_left"),
    ("stuck", _stuck_case(40.0, 400.0, 0), "strafe_right"),
    ("stuck", _stuck_case(400.0, 40.0, 3), "back_up"),
    ("stuck", _stuck_case(40.0, 40.0, 0), "back_up"),
    ("navigate", _navigation_case(400.0, 45.0), "turn_left"),
    ("navigate", _navigation_case(300.0, 1.0), "forward"),
    ("navigate", _navigation_case(650.0, -55.0), "turn_right"),
    ("navigate", _navigation_case(1000.0, -154.0), "turn_right"),
    ("navigate", _navigation_case(170.0, 22.0, waypoint=True), "forward"),
    ("navigate", _navigation_case(170.0, -22.0, waypoint=True), "forward"),
    ("navigate", _navigation_case(192.0, 88.0, waypoint=True), "turn_left"),
    ("navigate", _navigation_case(192.0, -88.0, waypoint=True), "turn_right"),
]


def self_test(service: TypeSafeReplica) -> bool:
    print("Self-test: checking each question against synthetic states")
    service.warm(CONTRACT)
    passed = 0
    for phase, state, expected in SELF_TEST_CASES:
        answer = service.system_one(
            state=state, questions={phase: CONTRACT[phase]}
        ).answers[phase]
        hit = answer["choice"] == expected
        passed += hit
        print(
            f"  {'ok  ' if hit else 'FAIL'} {phase:<8} expected={expected:<12} "
            f"got={answer['choice']:<12} conf={answer['confidence']:.2f}"
        )
    print(f"  {passed}/{len(SELF_TEST_CASES)}")
    return passed == len(SELF_TEST_CASES)


# --------------------------------------------------------------------------- main
def build_game(args: argparse.Namespace, iwad: Path) -> vzd.DoomGame:
    vizdoom_dir = Path(vzd.__file__).parent
    game = vzd.DoomGame()
    # Use the scenario config that matches the IWAD's format. Freedoom 1 / Doom 1
    # use ExMy names and freedoom1.cfg; Freedoom 2 / Doom 2 use MAPxx names and
    # doom2.cfg. Loading the wrong pair does not error - the engine never finishes
    # starting the episode and spins forever inside new_episode().
    map_name = args.map.upper()
    doom2_format = iwad.name.lower().startswith(("doom2", "freedoom2")) or map_name.startswith("MAP")
    config_name = "doom2.cfg" if doom2_format else "freedoom1.cfg"
    if doom2_format and not map_name.startswith("MAP"):
        map_name = f"MAP{int(map_name.lstrip('E').split('M')[-1]):02d}" if map_name.startswith("E") else map_name
    game.load_config(str(vizdoom_dir / "scenarios" / config_name))
    game.set_doom_game_path(str(iwad))
    game.set_doom_map(map_name)
    game.set_doom_skill(args.skill)
    if args.seed is not None:
        game.set_seed(args.seed)
    # Movement macros are capped by --tics, but the attack macro always runs its
    # own fixed time (a shot needs ~20 tics), so the timeout must allow the worse
    # of the two per decision or the episode would cut off mid-fight.
    game.set_episode_timeout(args.decisions * max(args.tics, SHOOT_TICS, MOVE_TICS) + 400)
    game.set_window_visible(args.watch)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_available_buttons(BUTTONS)
    game.set_available_game_variables([
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.ANGLE,
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.ARMOR,
        vzd.GameVariable.SELECTED_WEAPON,
        vzd.GameVariable.SELECTED_WEAPON_AMMO,
        vzd.GameVariable.AMMO1,
        vzd.GameVariable.AMMO2,
        vzd.GameVariable.AMMO3,
        vzd.GameVariable.AMMO4,
        *[getattr(vzd.GameVariable, f"WEAPON{slot}") for slot in range(1, 8)],
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.ITEMCOUNT,
        vzd.GameVariable.DEAD,
        vzd.GameVariable.DAMAGE_TAKEN,
    ])
    if args.watch:
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_render_all_frames(True)
        game.set_render_hud(True)
        game.set_render_crosshair(True)
    else:
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.init()
    return game


def run_episode(
    game: vzd.DoomGame,
    service: TypeSafeReplica,
    segments,
    grid: NavGrid,
    *,
    decisions: int,
    tics: int,
) -> dict:
    """Play one episode and return its summary.

    Two counters steer movement:

    * ``blocked_now`` selects the stuck question. It is set when a forward attempt
      fails and cleared as soon as any move succeeds, so a successful sidestep
      sends the agent back to navigation to retry the doorway instead of standing
      there strafing forever.
    * ``blocked_count`` counts failed escapes for escalation. It only resets on a
      successful forward move, so repeated sidestep-then-block cycles eventually
      reach four and the demo turns the player around.
    * ``abandoned`` holds pickups the agent has failed to reach repeatedly. Without
      it a walled-off item pins the agent: one E1M1 run spent 70 decisions walking
      at a Berserk pack it could never touch because every failure was forgiven the
      moment the agent drifted away from it.
    """
    game.new_episode()
    latency = {phase: LatencyLog(phase) for phase in CONTRACT}
    action_mix: dict[str, int] = {}
    sticky: int | None = None
    damage_before = 0
    blocked_now = False
    blocked_count = 0
    forward_retries = 0
    abandoned: set[int] = set()
    reach_progress: dict[int, dict[str, float]] = {}  # objective id -> closest/stale
    navigation = NavigationState()
    total_reward = 0.0

    for step in range(decisions):
        if game.is_episode_finished():
            break
        state = game.get_state()
        if state is None:
            break

        swap_note = ensure_usable_weapon(game)
        if swap_note:
            print(f"      auto: {swap_note}")

        px = game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        facing = game.get_game_variable(vzd.GameVariable.ANGLE)

        snap = build_snapshot(
            game, state, px=px, py=py, facing=facing,
            sticky_id=sticky, segments=segments, abandoned=abandoned,
            navigation=navigation, grid=grid,
        )
        damage_tick = snap.damage_now - damage_before
        damage_before = snap.damage_now

        phase = next_phase(snap, stuck=blocked_now)

        # Give up on a real pickup the agent cannot make progress toward. Pathfinding
        # filters out unreachable items, but a route can still be long and
        # the agent can wedge on geometry the grid's coarse cells treat as open; if
        # the closest approach has not improved for 20 navigation decisions, drop it.
        if (
            phase == "navigate"
            and snap.objective is not None
            and snap.objective.object_id > 0
        ):
            objective = snap.objective
            state_of = reach_progress.setdefault(
                objective.object_id, {"closest": objective.distance, "stale": 0.0}
            )
            if objective.distance < state_of["closest"] - 10.0:
                state_of["closest"] = objective.distance
                state_of["stale"] = 0
            else:
                state_of["stale"] += 1
                if state_of["stale"] >= 20:
                    abandoned.add(objective.object_id)
                    reach_progress.pop(objective.object_id, None)
                    navigation.objective_id = None
                    blocked_count = 0
                    blocked_now = False

        if phase == "combat":
            state_text = combat_state(snap)
            sticky = snap.monsters[0].object_id
        elif phase == "stuck":
            state_text = stuck_state(snap, blocked_count)
            sticky = None
        else:
            state_text = navigation_state(snap, damage_tick)
            sticky = None
            if snap.objective is not None:
                navigation.objective_id = snap.objective.object_id

        started = time.perf_counter()
        answer = service.system_one(
            state=state_text, questions={phase: CONTRACT[phase]}
        ).answers[phase]
        ms = latency[phase].record(started)
        action = answer["choice"]

        escalated = phase == "stuck" and action == "back_up" and blocked_count >= 4
        if escalated:
            # Sidesteps have failed four times, so turn around and walk out the way
            # the player came. TURN180 *with* MOVE_FORWARD is the escape: TURN180
            # plus MOVE_BACKWARD cancels itself out, because the 180 fixes facing
            # and then walking backward heads straight back into the wall.
            values = [0] * len(BUTTONS)
            values[BUTTONS.index(vzd.Button.TURN180)] = 1
            values[BUTTONS.index(vzd.Button.MOVE_FORWARD)] = 1
            total_reward += game.make_action(values, MOVE_TICS)
            action = "turn_and_leave"
        else:
            if phase == "combat" and action == "shoot":
                fine_aim(game, snap.monsters[0].bearing)
            values, action_tics = to_button_values(action, action_table(phase))
            # Each macro has a calibrated length (a turn sweeps ~17 degrees, a walk
            # covers ~40 units). ``--tics`` can cap movement for experimentation, but
            # the default 0 means "use the calibrated value": capping every macro at
            # 4 made turns 14 degrees and walks 20 units, which was itself a source
            # of thrash because each decision changed the world less than the state
            # report suggested. The attack macro is never capped: a shot needs ~20
            # tics (16 is still the weapon raise), so a cap of 4 would mean the
            # model presses fire and nothing leaves the barrel.
            budget = action_tics if action == "shoot" or tics <= 0 else min(tics, action_tics)
            total_reward += game.make_action(values, budget)

        moved = math.dist(
            (px, py),
            (game.get_game_variable(vzd.GameVariable.POSITION_X),
             game.get_game_variable(vzd.GameVariable.POSITION_Y)),
        )
        pressed_forward = action in {"forward", "advance"}
        if not game.is_player_dead():
            if pressed_forward:
                if moved < 1.5:
                    blocked_count += 1
                    forward_retries += 1
                    # A forward move with USE held is how doors open. A door needs
                    # several presses before it gives (measured: four macro steps of
                    # ~0.5 units each, then it opens), so a single blocked step is
                    # not evidence of being stuck - retrying forward is. Only after
                    # DOOR_RETRIES failures does this become the stuck question, and
                    # the cost for a real wall is just those retries.
                    blocked_now = forward_retries >= DOOR_RETRIES
                else:
                    blocked_count = 0
                    forward_retries = 0
                    blocked_now = False
            elif action in {"strafe_left", "strafe_right", "back_up", "turn_and_leave"}:
                if moved < 1.5:
                    blocked_count += 1
                    blocked_now = True
                else:
                    # A successful sidestep clears the stuck phase so the next
                    # decision re-tries the doorway; the escape count survives.
                    blocked_now = False
            # Turning and firing say nothing about being blocked, so they leave
            # both counters alone.
        action_mix[action] = action_mix.get(action, 0) + 1

        health = int(game.get_game_variable(vzd.GameVariable.HEALTH))
        ammo = int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO))
        kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        print(
            f"{step + 1:4d}  {phase:<8} {action:<12} conf={answer['confidence']:.2f}  "
            f"{ms:6.1f}ms  hp={health:3d} ammo={ammo:3d} kills={kills}  "
            f"| {state_text.splitlines()[-1][:56]}"
        )

    return {
        "decisions": sum(log.count for log in latency.values()),
        "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
        "items": int(game.get_game_variable(vzd.GameVariable.ITEMCOUNT)),
        "health": int(game.get_game_variable(vzd.GameVariable.HEALTH)),
        "dead": game.is_player_dead(),
        "reward": total_reward,
        "action_mix": action_mix,
        "latency": latency,
        "tic": game.get_episode_time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 plays a full Doom level through ViZDoom.",
    )
    parser.add_argument("--iwad", default=None,
                        help="path to an IWAD (default: Freedoom 1 for ExMy maps, "
                             "Freedoom 2 for MAPxx maps)")
    # MAP01 is the default because E1M1 is cramped and doors, which the model
    # handles far worse than open rooms: on E1M1 a 300-decision run manages ~2
    # kills and ~2 items, while MAP01 reaches ~5 kills and ~8 items with a third of
    # the stuck decisions (the level is open, so routing has room to work).
    parser.add_argument("--map", default="MAP01",
                        help="map to play (default: MAP01; use E1M1 for the "
                             "Freedoom 1 opening level)")
    parser.add_argument("--skill", type=int, default=3, help="Doom skill 1-5 (default: 3)")
    parser.add_argument("--decisions", type=int, default=150,
                        help="model decisions per episode (default: 150)")
    parser.add_argument("--tics", type=int, default=0,
                        help="optional cap on engine tics per movement macro "
                             "(default 0 = use each macro's calibrated length)")
    parser.add_argument("--episodes", type=int, default=1, help="episodes to play")
    parser.add_argument("--watch", action="store_true",
                        help="open the engine window and render the gameplay")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--self-test", action="store_true",
                        help="check the questions against synthetic states and exit")
    args = parser.parse_args()

    if args.self_test:
        print_header("Question self-test")
        service = TypeSafeReplica()
        service.engine.load()
        ok = self_test(service)
        sys.exit(0 if ok else 1)

    print_header(f"Full Doom - {args.map}, skill {args.skill}, "
                 f"{args.episodes} episode(s), {args.decisions} decisions each")

    vizdoom_dir = Path(vzd.__file__).parent
    if args.iwad:
        iwad = Path(args.iwad)
    else:
        # Pick the IWAD that matches the map name's format. Freedoom 1 is the Doom 1
        # IWAD (ExMy); Freedoom 2 is the Doom 2 IWAD (MAPxx). Loading MAP01 against
        # Freedoom 1 makes the engine hang inside new_episode rather than error.
        iwad = vizdoom_dir / (
            "freedoom2.wad" if args.map.upper().startswith("MAP") else "freedoom1.wad"
        )
    game = build_game(args, iwad)
    # Guard against a map that does not exist in the IWAD: without this the engine
    # spins at 100% CPU inside new_episode() and the process never exits.
    game.new_episode()
    if game.get_state() is None:
        game.close()
        raise SystemExit(
            f"map {args.map!r} did not start from {iwad.name}; check that the map "
            "name matches the IWAD (ExMy for Freedoom 1, MAPxx for Freedoom 2)"
        )
    segments = blocking_segments(game.get_state())
    grid = NavGrid(segments)
    print(f"IWAD: {iwad}   blocking lines: {len(segments)}   "
          f"navigation grid: {grid.width}x{grid.height} cells")
    if args.watch:
        print("Watching: 640x480 window, HUD and crosshair on, every tic rendered.")

    print("Loading model (one-time cost)...")
    service = TypeSafeReplica()
    service.engine.load()
    print(f"Model loaded in {service.engine.load_seconds:.1f}s")
    warm_ms = service.warm(CONTRACT)
    print(f"Question prefixes warmed in {warm_ms:.0f}ms\n")

    results = []
    started = time.perf_counter()
    for episode in range(1, args.episodes + 1):
        print(f"--- episode {episode} ---")
        result = run_episode(
            game, service, segments, grid, decisions=args.decisions, tics=args.tics
        )
        results.append(result)
        outcome = "died" if result["dead"] else "survived"
        print(f"    {outcome}: kills={result['kills']} items={result['items']} "
              f"health={result['health']} reward={result['reward']:+.1f} "
              f"game_time={result['tic'] / 35.0:.1f}s\n")

    wall_s = time.perf_counter() - started
    print(f"Total: {len(results)} episode(s) in {wall_s:.1f}s wall clock")
    print(f"kills={sum(r['kills'] for r in results)}  "
          f"items={sum(r['items'] for r in results)}  "
          f"deaths={sum(1 for r in results if r['dead'])}")
    for phase in CONTRACT:
        samples = [ms for r in results for ms in r["latency"][phase].samples_ms]
        if samples:
            print("  " + LatencyLog(phase, samples).summary())
    mix: dict[str, int] = {}
    for result in results:
        for action, count in result["action_mix"].items():
            mix[action] = mix.get(action, 0) + count
    print("action mix: " + ", ".join(f"{k}={v}" for k, v in sorted(mix.items())))
    game.close()


if __name__ == "__main__":
    main()
