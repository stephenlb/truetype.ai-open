"""Gemma 4 plays Flappy Bird in real time.

The Doom demo (``doom/doom_full_game_demo.py``) drives an engine that waits for
the model: ViZDoom advances only when ``make_action`` is called, so a slow
decision costs wall-clock time but never game time. Flappy Bird is the harder
test of the same claim, because the bird falls whether or not anyone has decided
anything. This demo makes the model's latency the clock.

Real time
---------
The simulation runs at a fixed 30 frames per second and takes one decision every
``FRAMES_PER_DECISION`` (4) frames, so one decision is worth 133ms of game time.
A warm decision on this replica measures ~130ms. The bird therefore falls at
roughly the rate the model thinks, and the run reports the ratio: game seconds
simulated over wall seconds spent deciding. A ratio near 1.0 means the model kept
up with a bird flying at normal speed; below 1.0 means the game was played in
slow motion and the number says by how much.

The chosen action applies to the whole 4-frame block: ``flap`` presses once on
the first frame and coasts for the rest, exactly like a human whose finger is off
the screen for those 100ms.

The decision
------------
One question per decision, in one of two phases selected from the world state:

  ============  ==================================================================
  ``approach``  a pipe is within 160 units: flap / coast, judged on projections
  ``line_up``   no pipe close: flap / coast, judged against the target line
  ============  ==================================================================

The demo does the physics and the model does the judgement, which is the same
split as the Doom demo's BFS waypoints. Each decision rolls one future forward -
the bird coasting - and reports where it ends up: which side of the opening it
passes for ``approach``, how far from the target line it lands for ``line_up``.
The model reads an outcome rather than estimating a trajectory from position and
velocity, which it cannot do in one token.

Projecting the *whole* crossing, not just the arrival at the near edge, is what
makes "passes through the opening" safe to act on. The bird is horizontally
inside a pipe for two to three decisions, so a path that enters the opening
cleanly can still clip the bottom lip on the way out, and a projection that
stopped at the near edge reported a clear crossing right up to the frame the bird
clipped the lip.

Only the coast future is projected, because only it is decisive. If coasting
passes below the opening then flapping can only reduce how low the bird is, and
if coasting passes above it then flapping can only make that worse; there is
never a frame where the flap projection changes the answer. An earlier version
reported both futures and asked the model to compare them, which produced a rule
that coasted into the pipe: a bird well below a high opening has *both* futures
blocked, because one flap climbs 13 units and the opening is 40 above, and
"neither line is clear, so hold" was the wrong reading of a frame that needed
four flaps in a row. All five baseline seeds died on the first pipe.

Wording
-------
Two measurements shaped the prompts, both from bake-off runs over the synthetic
cases now in ``SELF_TEST_CASES``:

*Option order is not neutral.* With ``flap`` listed first it takes letter A, and
every ``coast`` case failed (3/5 on approach, 2/4 on line_up) while the ``flap``
cases answered at 0.8 confidence. Listing ``coast`` first - the same criteria,
the same states, the same instructions, only the dict order changed - scored 5/5
and 4/4. The no-op action goes first for that reason.

*Numbers in the state get read instead of the verdict.* An earlier approach state
reported where the path passes relative to the middle of the opening ("passes 18
units below the middle of the opening, inside the opening") and scored 4/8 at
0.51 confidence, answering ``flap`` whenever the numbers looked alarming. Cutting
the offset took the same cases to 5/5. The offsets are still computed and still
logged, they are just not in the prompt.

The verdict itself then has to name the mistake rather than the geometry. "The
bird passes below the opening" against "Flap - the bird passes below the opening"
scored 3/5, and the below cases came back at 0.00 confidence - a dead tie between
the two letters, decided by whichever held letter A. "Without a flap the bird will
be too low" against "Flap - without a flap the bird is too low" scored 5/5 at 0.79
mean confidence on the same five cases, and is stable under both option orders
(5/5 coast-first, 4/5 flap-first). Both phrasings match the state word for word;
only the second says what is wrong with the position.

Tuning
------
The physics is tuned for a 7.5Hz controller rather than a 60Hz one. A flap moves
the bird about 13 units up over one decision and a sustained fall moves it 14
down, against an opening 130 units tall, so a decision is worth roughly a tenth
of the opening and a missed one is recoverable. The classic 60Hz values (0.25
gravity against a 100-unit gap) leave a controller at this rate no way to hold an
altitude at all: the bird crosses the whole opening between decisions.

Run (from the repo root):
    python flappy/flappy_bird_demo.py                   # 150 decisions, 1 episode
    python flappy/flappy_bird_demo.py --ascii           # ...and watch in the terminal
    python flappy/flappy_bird_demo.py --watch           # ...in a pygame window
    python flappy/flappy_bird_demo.py --episodes 3 --decisions 400
    python flappy/flappy_bird_demo.py --baseline        # reference player, no model
    python flappy/flappy_bird_demo.py --self-test       # check the questions, no game

``--baseline`` replaces the model with the rule its criteria describe and runs the
same loop. It is not a scoreboard to beat; it is there to show the physics is
winnable at this control rate (39 pipes in 600 decisions, no crashes), so that a
crash in a model run is evidence about the decision and not about the tuning.

``--watch`` and ``--ascii`` draw every simulated frame, from inside the stepping
loop that runs after the timed decision, so rendering never lands inside a
measured latency. The window shows the decision as well as the game: the dashed
line is the line the current question is judged against (the middle of the next
opening), and its colour, the label on the ground strip and the bar next to it
are the phase, the chosen action and the model's confidence in it.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from truetype.service import TypeSafeReplica
from common import LatencyLog, print_header

# -------------------------------------------------------------------------- world
# Units are pixels of a 640x512 field; y grows downward, so a positive velocity is
# a fall and a negative offset is "above".
FIELD_WIDTH = 640.0
FIELD_HEIGHT = 512.0

BIRD_X = 140.0
BIRD_RADIUS = 12.0

# Frame rate and control rate. 30Hz with a decision every 4 frames puts one
# decision at 133ms of game time, which is what a warm decision costs here.
FRAME_HZ = 30.0
FRAMES_PER_DECISION = 4

# A flap is worth ~13 units up over one decision and a sustained fall ~14 down,
# against a 130-unit opening. See the module docstring on why the 60Hz constants
# do not transfer.
GRAVITY = 0.5
FLAP_IMPULSE = -4.5
MAX_FALL = 3.5

PIPE_WIDTH = 40.0
GAP_HEIGHT = 130.0
PIPE_SPACING = 300.0
SCROLL = 5.0  # 20 units per decision

# Openings stay inside this band so neither lip leaves the field, and consecutive
# openings move by at most GAP_DELTA so the next one is always reachable: 90 units
# is about seven decisions of climb, and there are fifteen between pipes.
GAP_CENTER_MIN = 150.0
GAP_CENTER_MAX = 362.0
GAP_DELTA = 90.0

# Inside this range the opening is close enough to be worth projecting through;
# beyond it the bird is lining up. The gap between consecutive pipe bodies is
# PIPE_SPACING - PIPE_WIDTH = 260, so at 160 both phases get real airtime: a
# measured baseline run split 433 approach decisions to 167 line_up.
APPROACH_RANGE = 160.0

# Offsets under this are reported as "level" rather than above or below.
LEVEL_BAND = 4.0


@dataclass
class Pipe:
    x: float  # left edge
    gap_center: float
    passed: bool = False

    @property
    def gap_top(self) -> float:
        return self.gap_center - GAP_HEIGHT / 2

    @property
    def gap_bottom(self) -> float:
        return self.gap_center + GAP_HEIGHT / 2


@dataclass
class World:
    """One episode of Flappy Bird, stepped a frame at a time."""

    bird_y: float = FIELD_HEIGHT / 2
    velocity: float = 0.0
    frame: int = 0
    score: int = 0
    crashed: str = ""  # "" while alive, else what was hit
    pipes: list[Pipe] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def new(cls, seed: int | None = None) -> World:
        world = cls(rng=random.Random(seed))
        # The first pipe starts well ahead of the bird so a run opens in line_up
        # rather than mid-crossing.
        world.pipes.append(Pipe(x=520.0, gap_center=FIELD_HEIGHT / 2))
        world._extend_pipes()
        return world

    @property
    def alive(self) -> bool:
        return not self.crashed

    @property
    def game_seconds(self) -> float:
        return self.frame / FRAME_HZ

    def _extend_pipes(self) -> None:
        while self.pipes[-1].x < FIELD_WIDTH + PIPE_SPACING:
            previous = self.pipes[-1]
            center = previous.gap_center + self.rng.uniform(-GAP_DELTA, GAP_DELTA)
            center = min(GAP_CENTER_MAX, max(GAP_CENTER_MIN, center))
            self.pipes.append(Pipe(x=previous.x + PIPE_SPACING, gap_center=center))

    def next_pipe(self) -> Pipe | None:
        """The first pipe the bird has not cleared, including one it is inside."""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X - BIRD_RADIUS:
                return pipe
        return None

    def step(self, flap: bool) -> None:
        """Advance one frame. ``flap`` presses the button on this frame only."""
        if self.crashed:
            return
        self.bird_y, self.velocity = step_physics(self.bird_y, self.velocity, flap)
        for pipe in self.pipes:
            pipe.x -= SCROLL
        self.frame += 1

        if self.bird_y + BIRD_RADIUS >= FIELD_HEIGHT:
            self.crashed = "the floor"
            return
        if self.bird_y - BIRD_RADIUS <= 0.0:
            # The original game lets the bird leave the top of the screen. Crashing
            # instead makes over-flapping a real mistake, which is what the coast
            # side of the question is for.
            self.crashed = "the ceiling"
            return
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS:
                if not pipe.passed:
                    pipe.passed = True
                    self.score += 1
                continue
            if pipe.x > BIRD_X + BIRD_RADIUS:
                break
            if self.bird_y - BIRD_RADIUS < pipe.gap_top:
                self.crashed = "the top pipe"
                return
            if self.bird_y + BIRD_RADIUS > pipe.gap_bottom:
                self.crashed = "the bottom pipe"
                return

        self.pipes = [p for p in self.pipes if p.x + PIPE_WIDTH > -PIPE_SPACING]
        self._extend_pipes()


def step_physics(y: float, velocity: float, flap: bool) -> tuple[float, float]:
    """One frame of bird kinematics, shared by the world and the projections.

    A flap *replaces* the velocity rather than adding to it, as in the original
    game, which is what makes one press worth a fixed amount of climb no matter
    how fast the bird was falling.
    """
    if flap:
        velocity = FLAP_IMPULSE
    velocity = min(MAX_FALL, velocity + GRAVITY)
    return y + velocity, velocity


# --------------------------------------------------------------------- projections
@dataclass
class Projection:
    """Where coasting leaves the bird as it crosses the next pipe.

    ``blocked`` names what the path hits, or is "" when the path is clear.
    ``offset`` is signed from the middle of the opening, positive downward, taken
    at the worst point of the crossing rather than at its start. The number is
    logged but deliberately kept out of the prompt, which reports only its side:
    see the module docstring on wording.
    """

    offset: float
    blocked: str = ""

    @property
    def clear(self) -> bool:
        return not self.blocked

    @property
    def low(self) -> bool:
        """True when the blocked path passes below the opening rather than above."""
        return self.offset > 0


def project(world: World, pipe: Pipe) -> Projection:
    """Roll the bird forward, coasting, to the far edge of ``pipe``.

    The far edge and not the near one: the bird is horizontally inside a pipe for
    two to three decisions, so a path that enters the opening cleanly can still
    clip the bottom lip on the way out, and a near-edge projection reported a clear
    crossing right up to the frame the bird clipped that lip.
    """
    y, velocity = world.bird_y, world.velocity
    # Frames until the pipe's near edge reaches the bird, and until its far edge
    # leaves the bird. Both can be negative while the bird is inside the pipe.
    entry_frames = (pipe.x - (BIRD_X + BIRD_RADIUS)) / SCROLL
    exit_frames = (pipe.x + PIPE_WIDTH - (BIRD_X - BIRD_RADIUS)) / SCROLL
    total = max(1, math.ceil(exit_frames))

    worst_offset = 0.0
    blocked = ""
    for frame in range(1, total + 1):
        y, velocity = step_physics(y, velocity, flap=False)
        if y + BIRD_RADIUS >= FIELD_HEIGHT:
            return Projection(y - pipe.gap_center, "the floor")
        if y - BIRD_RADIUS <= 0.0:
            return Projection(y - pipe.gap_center, "the ceiling")
        if frame < entry_frames:
            continue  # not over the pipe yet, so its lips cannot be hit
        offset = y - pipe.gap_center
        if abs(offset) > abs(worst_offset):
            worst_offset = offset
        if not blocked:
            if y - BIRD_RADIUS < pipe.gap_top:
                blocked = "the top pipe"
            elif y + BIRD_RADIUS > pipe.gap_bottom:
                blocked = "the bottom pipe"
    return Projection(worst_offset, blocked)


def coast_offset(world: World, target: float) -> float:
    """Where one decision of coasting leaves the bird, relative to ``target``.

    One decision, not the whole approach: between pipes the bird only needs to
    stop drifting away from the line it will want, and a longer horizon would
    always read as "far below" once the fall rate saturates.
    """
    y, velocity = world.bird_y, world.velocity
    for _ in range(FRAMES_PER_DECISION):
        y, velocity = step_physics(y, velocity, False)
    return y - target


# ------------------------------------------------------------------------- prompts
# ``coast`` is listed first in both questions so it takes letter A. That is not
# cosmetic: with ``flap`` first, every coast case in SELF_TEST_CASES failed while
# the flap cases answered at 0.8 confidence. See the module docstring.
APPROACH_CRITERIA = {
    # "too low", "too high" and "lined up with the opening" are the state's words
    # verbatim, and nothing here refers to a fact the report omits. Two earlier
    # vocabularies for the same three verdicts did worse on the same cases:
    # "the safer line", which the state never names, sat at 0.1 confidence, and
    # "passes below / through / above the opening" left the below cases an exact
    # 0.00-confidence tie that fell to whichever option held letter A. Geometry
    # words describe the bird's position; "too low" describes the mistake, and
    # only the second one implies what to do about it.
    "coast": "Do not flap - without a flap the bird is lined up with the opening, or too high",
    "flap": "Flap - without a flap the bird is too low",
}

LINE_UP_CRITERIA = {
    "coast": "Do not flap - not flapping leaves the bird level with or above the target line",
    "flap": "Flap - not flapping leaves the bird below the target line",
}

CONTRACT = {
    "approach": {
        "type": "choice",
        "instructions": (
            "A pipe is coming up and the bird must fly through the opening. "
            "Should the player flap now?"
        ),
        "criteria": APPROACH_CRITERIA,
    },
    "line_up": {
        "type": "choice",
        "instructions": (
            "No pipe is close and the bird should get level with the line it will "
            "need. Should the player flap now?"
        ),
        "criteria": LINE_UP_CRITERIA,
    },
}


# -------------------------------------------------------------------- state report
def offset_words(offset: float, noun: str) -> str:
    """A signed offset as words plus the number, positive being below."""
    if abs(offset) < LEVEL_BAND:
        return f"level with {noun}"
    return f"{abs(offset):.0f} units {'below' if offset > 0 else 'above'} {noun}"


def approach_state(coast: Projection) -> str:
    """One line: which side of the opening the bird ends up on if it coasts.

    Only the coast projection is reported, because only it is decisive. If
    coasting passes below the opening then a flap can only reduce how low the bird
    is, and if coasting passes above it then a flap can only make that worse, so a
    second projection of flapping adds nothing to choose between.

    Distance to the pipe is not here, and neither is bird height or velocity. All
    three are already inside the projection, and reporting the offset in units
    alongside the verdict scored 4/8 against 5/5 for the verdict alone: the model
    answered from the numbers whenever they looked alarming.

    ``line_up`` keeps its number because there the offset *is* the verdict - there
    is no opening to be inside of yet - so the two phases word their states
    differently on purpose. Each was tuned against its own cases.
    """
    if coast.clear:
        verdict = "lined up with the opening"
    else:
        verdict = "too low" if coast.low else "too high"
    return f"Without a flap the bird will be {verdict}."


def line_up_state(offset: float) -> str:
    """One line: where coasting leaves the bird relative to the target line."""
    return f"Not flapping leaves the bird {offset_words(offset, 'the target line')}."


def next_phase(world: World) -> tuple[str, Pipe | None]:
    pipe = world.next_pipe()
    if pipe is None:
        return "line_up", None
    if pipe.x - (BIRD_X + BIRD_RADIUS) <= APPROACH_RANGE:
        return "approach", pipe
    return "line_up", pipe


def target_line(pipe: Pipe | None) -> float:
    """Where to sit while no pipe is close: the next opening, or mid-field."""
    return pipe.gap_center if pipe is not None else FIELD_HEIGHT / 2


# ------------------------------------------------------------------------ baseline
def baseline_action(world: World) -> str:
    """The criteria, applied directly, with no model in the loop.

    This is the rule the model is shown, not a better one, so the comparison is
    about reading the state rather than about strategy. Running it shows the
    physics is winnable at 7.5 decisions per second - 39 pipes in 600 decisions
    with no crash, on each of five seeds - so a crash in a model run is evidence
    about the decision.
    """
    phase, pipe = next_phase(world)
    if phase == "approach" and pipe is not None:
        coast = project(world, pipe)
        return "flap" if coast.blocked and coast.low else "coast"
    return "flap" if coast_offset(world, target_line(pipe)) > LEVEL_BAND else "coast"


# ----------------------------------------------------------------------- rendering
# Both views are built so that nothing expensive happens per frame. The window
# renders the sky, the hills and the pipe and bird sprites once at startup and
# blits them afterwards; the terminal view builds one string. Drawing has to stay
# far cheaper than a decision, because the real-time ratio the demo reports is
# only honest if the renderer is not part of what is being measured.
ASCII_ROWS = 24
ASCII_COLS = 64

# 256-colour codes, close to the palette the window uses. Terminals that ignore
# them still show the right characters.
ANSI = {
    "sky": "\033[38;5;153m",
    "pipe": "\033[38;5;71m",
    "lip": "\033[38;5;79m",
    "bird": "\033[38;5;220m",
    "dead": "\033[38;5;203m",
    "frame": "\033[38;5;244m",
    "off": "\033[0m",
}


def ascii_frame(world: World) -> str:
    """A fixed-size text view of the field, cheap enough to draw every frame.

    Cells are tagged with what they are and coloured in runs, rather than one
    escape sequence per character, so a frame is a few dozen writes.
    """
    grid = [[(" ", "sky")] * ASCII_COLS for _ in range(ASCII_ROWS)]
    row_height = FIELD_HEIGHT / ASCII_ROWS
    for pipe in world.pipes:
        lo = int(pipe.x / FIELD_WIDTH * ASCII_COLS)
        hi = int((pipe.x + PIPE_WIDTH) / FIELD_WIDTH * ASCII_COLS)
        for col in range(max(0, lo), min(ASCII_COLS, hi + 1)):
            for row in range(ASCII_ROWS):
                top, bottom = row * row_height, (row + 1) * row_height
                # The row that holds a lip is drawn as a half block, which places
                # the edge of the opening within half a row of where it really is.
                if bottom <= pipe.gap_top:
                    last = pipe.gap_top - bottom < row_height
                    grid[row][col] = ("▀", "lip") if last else ("█", "pipe")
                elif top >= pipe.gap_bottom:
                    first = top - pipe.gap_bottom < row_height
                    grid[row][col] = ("▄", "lip") if first else ("█", "pipe")
    row = min(ASCII_ROWS - 1, max(0, int(world.bird_y / FIELD_HEIGHT * ASCII_ROWS)))
    col = min(ASCII_COLS - 1, int(BIRD_X / FIELD_WIDTH * ASCII_COLS))
    grid[row][col] = ("✗", "dead") if world.crashed else ("◗", "bird")

    lines = []
    for cells in grid:
        out, kind, run = [], cells[0][1], ""
        for char, cell_kind in cells:
            if cell_kind != kind:
                out.append(f"{ANSI[kind]}{run}")
                kind, run = cell_kind, ""
            run += char
        out.append(f"{ANSI[kind]}{run}{ANSI['off']}")
        edge = f"{ANSI['frame']}│{ANSI['off']}"
        lines.append(edge + "".join(out) + edge)
    top = f"{ANSI['frame']}╭{'─' * ASCII_COLS}╮{ANSI['off']}"
    bottom = f"{ANSI['frame']}╰{'─' * ASCII_COLS}╯{ANSI['off']}"
    return "\n".join([top, *lines, bottom])


# The window is FIELD_HEIGHT plus a ground strip, so the floor the bird crashes
# into is a visible edge rather than the bottom of the screen.
GROUND_HEIGHT = 86
WINDOW_SIZE = (int(FIELD_WIDTH), int(FIELD_HEIGHT) + GROUND_HEIGHT)

SKY_TOP = (74, 160, 208)
SKY_BOTTOM = (186, 226, 235)
HILL_FAR = (132, 194, 164)
HILL_NEAR = (98, 172, 133)
PIPE_LIGHT = (156, 220, 118)
PIPE_BODY = (96, 178, 74)
PIPE_DARK = (52, 118, 48)
PIPE_EDGE = (34, 78, 38)
GRASS = (128, 202, 98)
GRASS_DARK = (92, 166, 76)
DIRT = (206, 178, 120)
DIRT_DARK = (184, 154, 98)
BIRD_BODY = (250, 206, 66)
BIRD_BELLY = (253, 236, 164)
BIRD_WING = (234, 160, 50)
BIRD_BEAK = (240, 122, 56)
BIRD_DEAD = (216, 88, 76)
INK = (24, 34, 44)
PAPER = (244, 249, 252)
PHASE_COLOR = {"approach": (250, 172, 72), "line_up": (122, 198, 242)}


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))  # type: ignore[return-value]


class PygameView:
    """Optional window. Imported lazily so the demo runs without pygame."""

    def __init__(self) -> None:
        import pygame  # noqa: PLC0415 - optional dependency

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Gemma 4 plays Flappy Bird")
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 14)
        self.font_bold = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15, bold=True)
        self.font_score = pygame.font.SysFont("menlo,dejavusansmono,monospace", 44, bold=True)
        self.clock = pygame.time.Clock()
        # Parallax layers, each one field wide and seamless across that width, so
        # scrolling is two blits at an offset rather than anything recomputed.
        self.sky = self._render_sky()
        self.clouds = self._render_clouds()
        self.hills = self._render_hills()
        self.stripes = self._render_stripes()
        self.pipe_column = self._render_pipe_column()
        self.pipe_cap = self._render_pipe_cap()
        # Three wing positions, cycled while the bird climbs and held flat while
        # it falls, so a flap is visible as a beat and not only as a change of
        # direction.
        self.birds = [self._render_bird(phase) for phase in (-1, 0, 1)]
        self.bird_dead = self._render_bird(0, dead=True)
        self.trail: list[list[float]] = []
        self.held_crash = False

    # ---------------------------------------------------------------- sprites
    def _render_sky(self):
        """The gradient and the ground, the two things that do not scroll."""
        pygame = self.pygame
        width, height = WINDOW_SIZE
        surface = pygame.Surface((width, height))
        for y in range(int(FIELD_HEIGHT)):
            shade = _lerp(SKY_TOP, SKY_BOTTOM, (y / FIELD_HEIGHT) ** 0.8)
            pygame.draw.line(surface, shade, (0, y), (width, y))
        floor = int(FIELD_HEIGHT)
        pygame.draw.rect(surface, DIRT, pygame.Rect(0, floor, width, GROUND_HEIGHT))
        pygame.draw.rect(surface, GRASS, pygame.Rect(0, floor, width, 14))
        pygame.draw.rect(surface, GRASS_DARK, pygame.Rect(0, floor + 14, width, 4))
        return surface

    def _render_hills(self):
        """Two ranges of hills on a transparent layer.

        The outline is a sum of sines with whole numbers of periods across the
        field width, so the layer tiles seamlessly and the parallax scroll is two
        blits of the same surface.
        """
        pygame = self.pygame
        width = WINDOW_SIZE[0]
        surface = pygame.Surface((width, int(FIELD_HEIGHT)), self.pygame.SRCALPHA)
        for color, base, amp, periods in (
            (HILL_FAR, FIELD_HEIGHT - 104, 34, 2),
            (HILL_NEAR, FIELD_HEIGHT - 56, 22, 3),
        ):
            points = [
                (x, base + amp * math.sin(2 * math.pi * periods * x / width)
                 + amp * 0.4 * math.sin(2 * math.pi * periods * 2 * x / width))
                for x in range(0, width + 8, 8)
            ]
            pygame.draw.polygon(
                surface, color, [*points, (width, FIELD_HEIGHT), (0, FIELD_HEIGHT)]
            )
        return surface

    def _render_clouds(self):
        """A band of soft clouds, each drawn three times so the layer wraps."""
        pygame = self.pygame
        width = WINDOW_SIZE[0]
        surface = pygame.Surface((width, int(FIELD_HEIGHT)), pygame.SRCALPHA)
        rng = random.Random(7)
        for _ in range(12):
            cx = rng.uniform(0, width)
            cy = rng.uniform(36, 210)
            scale = rng.uniform(0.7, 1.5)
            alpha = int(130 + 80 * rng.random())
            for wrap in (-width, 0, width):
                for dx, dy, r in ((-22, 6, 16), (0, 0, 24), (24, 8, 18), (6, 11, 20)):
                    pygame.draw.circle(
                        surface, (255, 255, 255, alpha),
                        (int(cx + wrap + dx * scale), int(cy + dy * scale)),
                        int(r * scale),
                    )
        return surface

    def _render_stripes(self):
        """Dirt marks on the ground, at a pitch that divides the field width.

        They are the cue that the world is scrolling at all: between pipes there
        is nothing else on screen that moves horizontally.
        """
        pygame = self.pygame
        width = WINDOW_SIZE[0]
        surface = pygame.Surface((width, GROUND_HEIGHT - 18), pygame.SRCALPHA)
        for x in range(0, width, 32):
            pygame.draw.rect(surface, DIRT_DARK, pygame.Rect(x, 8, 14, 8))
            pygame.draw.rect(surface, DIRT_DARK, pygame.Rect(x + 18, 22, 8, 6))
        return surface

    def _render_pipe_column(self):
        """One full-height pipe body, shaded across its width.

        Shading the width and not the height means the same sprite can be cut to
        any length, which is what the top and bottom halves of every pipe need.
        """
        pygame = self.pygame
        width = int(PIPE_WIDTH)
        surface = pygame.Surface((width, int(FIELD_HEIGHT) + GROUND_HEIGHT))
        for x in range(width):
            t = x / (width - 1)
            if t < 0.3:
                color = _lerp(PIPE_DARK, PIPE_LIGHT, t / 0.3)
            else:
                color = _lerp(PIPE_LIGHT, PIPE_DARK, (t - 0.3) / 0.7)
            pygame.draw.line(surface, color, (x, 0), (x, surface.get_height()))
        pygame.draw.line(surface, PIPE_EDGE, (0, 0), (0, surface.get_height()))
        pygame.draw.line(
            surface, PIPE_EDGE, (width - 1, 0), (width - 1, surface.get_height())
        )
        return surface

    def _render_pipe_cap(self):
        """The wider lip that marks the edge of the opening."""
        pygame = self.pygame
        width, height = int(PIPE_WIDTH) + 12, 24
        surface = pygame.Surface((width, height))
        for x in range(width):
            t = x / (width - 1)
            if t < 0.28:
                color = _lerp(PIPE_DARK, PIPE_LIGHT, t / 0.28)
            else:
                color = _lerp(PIPE_LIGHT, PIPE_BODY, (t - 0.28) / 0.72)
            pygame.draw.line(surface, color, (x, 0), (x, height))
        pygame.draw.rect(surface, PIPE_EDGE, surface.get_rect(), 2)
        return surface

    def _render_bird(self, wing: int, *, dead: bool = False):
        """The bird, nose to the right, with the wing at one of three heights."""
        pygame = self.pygame
        r = int(BIRD_RADIUS)
        surface = pygame.Surface((r * 4, r * 3), pygame.SRCALPHA)
        cx, cy = r * 3 // 2, r * 3 // 2
        body = BIRD_DEAD if dead else BIRD_BODY
        pygame.draw.circle(surface, body, (cx, cy), r)
        pygame.draw.circle(surface, _lerp(body, BIRD_BELLY, 0.8), (cx - 2, cy + 4), r - 5)
        # Tail, then wing, then the head details on top.
        pygame.draw.polygon(
            surface, _lerp(body, PIPE_EDGE, 0.15),
            [(cx - r + 2, cy - 4), (cx - r - 6, cy - 9), (cx - r - 4, cy + 4)],
        )
        wing_rect = pygame.Rect(0, 0, r + 4, r - 2)
        wing_rect.center = (cx - 3, cy + 2 + wing * 5)
        pygame.draw.ellipse(surface, BIRD_WING if not dead else _lerp(body, INK, 0.3), wing_rect)
        pygame.draw.ellipse(surface, _lerp(BIRD_WING, INK, 0.35), wing_rect, 1)
        pygame.draw.polygon(
            surface, BIRD_BEAK,
            [(cx + r - 3, cy - 3), (cx + r + 7, cy + 1), (cx + r - 3, cy + 5)],
        )
        pygame.draw.circle(surface, PAPER, (cx + 4, cy - 4), 4)
        if dead:
            pygame.draw.line(surface, INK, (cx + 1, cy - 7), (cx + 7, cy - 1), 2)
            pygame.draw.line(surface, INK, (cx + 7, cy - 7), (cx + 1, cy - 1), 2)
        else:
            pygame.draw.circle(surface, INK, (cx + 5, cy - 4), 2)
        return surface

    # ------------------------------------------------------------------ frame
    def draw(
        self,
        world: World,
        phase: str,
        action: str,
        confidence: float = 1.0,
        ms: float = 0.0,
    ) -> None:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        self.screen.blit(self.sky, (0, 0))
        travelled = world.frame * SCROLL
        # Slower layers for things that are further away, at the pitch the real
        # game uses: the pipes move at 1.0, so anything behind them moves less.
        self._parallax(self.clouds, travelled * 0.10, 0)
        self._parallax(self.hills, travelled * 0.30, 0)
        self._parallax(self.stripes, travelled, int(FIELD_HEIGHT) + 18)
        self._draw_pipes(world, phase)
        self._draw_trail(world)
        self._draw_bird(world)
        if world.crashed:
            # Before the HUD, so the tint covers the field and not the readouts.
            self._draw_crash(world)
        self._draw_hud(world, phase, action, confidence, ms)
        pygame.display.flip()

        if world.crashed and not self.held_crash:
            # Hold the last frame long enough to see what was hit. This runs
            # after the episode's decisions, so it costs no measured latency.
            self.held_crash = True
            pygame.time.wait(1400)
        # Frames are drawn as fast as they are simulated; the model's latency, not
        # this clock, paces the run. The tick only keeps the window responsive.
        self.clock.tick(240)

    def _parallax(self, layer, travelled: float, y: int) -> None:
        offset = -(travelled % WINDOW_SIZE[0])
        self.screen.blit(layer, (offset, y))
        self.screen.blit(layer, (offset + WINDOW_SIZE[0], y))

    def _draw_pipes(self, world: World, phase: str) -> None:
        pygame = self.pygame
        cap = self.pipe_cap
        for pipe in world.pipes:
            x = int(pipe.x)
            if x > FIELD_WIDTH or x + PIPE_WIDTH < 0:
                continue
            top, bottom = int(pipe.gap_top), int(pipe.gap_bottom)
            self.screen.blit(self.pipe_column, (x, 0), (0, 0, PIPE_WIDTH, top - 10))
            self.screen.blit(
                self.pipe_column, (x, bottom + 10),
                (0, 0, PIPE_WIDTH, int(FIELD_HEIGHT) - bottom - 10),
            )
            self.screen.blit(cap, (x - 6, top - cap.get_height()))
            self.screen.blit(cap, (x - 6, bottom))

        target = world.next_pipe()
        if target is not None:
            # The line the decision is being judged against: the opening's middle
            # while a pipe is close, the same line as the target while lining up.
            color = PHASE_COLOR[phase]
            shadow = _lerp(color, INK, 0.55)
            y = int(target.gap_center)
            for x in range(int(BIRD_X), int(target.x + PIPE_WIDTH), 14):
                pygame.draw.line(self.screen, shadow, (x, y + 2), (x + 7, y + 2), 3)
                pygame.draw.line(self.screen, color, (x, y), (x + 7, y), 3)
            marker = (int(target.x + PIPE_WIDTH / 2), y)
            pygame.draw.circle(self.screen, shadow, marker, 6)
            pygame.draw.circle(self.screen, color, marker, 4)

    def _draw_trail(self, world: World) -> None:
        """A short fading wake, which is the only cue to the bird's speed."""
        pygame = self.pygame
        for point in self.trail:
            point[0] -= SCROLL
        self.trail = [p for p in self.trail if p[0] > 0][-26:]
        self.trail.append([BIRD_X, world.bird_y])
        for index, (x, y) in enumerate(self.trail[:-1]):
            fade = (index + 1) / len(self.trail)
            radius = max(1, int(BIRD_RADIUS * 0.45 * fade))
            dot = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(dot, (255, 255, 255, int(120 * fade)), (radius, radius), radius)
            self.screen.blit(dot, (x - radius, y - radius))

    def _draw_bird(self, world: World) -> None:
        """Pitch the sprite with the velocity: nose up on a flap, down on a fall."""
        pygame = self.pygame
        if world.crashed:
            sprite = self.bird_dead
            angle = -70.0
        else:
            climbing = world.velocity < 0
            sprite = self.birds[world.frame // 2 % 3 if climbing else 2]
            angle = max(-52.0, min(28.0, -world.velocity * 11.0))
        rotated = pygame.transform.rotozoom(sprite, angle, 1.0)
        self.screen.blit(rotated, rotated.get_rect(center=(int(BIRD_X), int(world.bird_y))))

    def _draw_hud(
        self, world: World, phase: str, action: str, confidence: float, ms: float
    ) -> None:
        pygame = self.pygame
        floor = int(FIELD_HEIGHT)
        self._shadowed(self.font_score, str(world.score), (24, 18))

        # The decision itself, on the ground strip so it never covers the field.
        color = PHASE_COLOR[phase]
        pygame.draw.rect(self.screen, _lerp(INK, color, 0.1),
                         pygame.Rect(0, floor + 40, WINDOW_SIZE[0], GROUND_HEIGHT - 40))
        pygame.draw.line(self.screen, color, (0, floor + 40), (WINDOW_SIZE[0], floor + 40), 2)
        label = self.font_bold.render(f"{phase}  →  {action}", True, color)
        self.screen.blit(label, (14, floor + 50))
        meta = self.font.render(f"{ms:5.0f} ms/decision", True, _lerp(PAPER, color, 0.3))
        self.screen.blit(meta, (WINDOW_SIZE[0] - meta.get_width() - 14, floor + 52))

        # Confidence as a bar, because its size is the point and its digits are not.
        bar = pygame.Rect(14, floor + 70, 200, 8)
        pygame.draw.rect(self.screen, _lerp(INK, PAPER, 0.18), bar, border_radius=4)
        filled = bar.copy()
        filled.width = max(2, int(bar.width * max(0.0, min(1.0, confidence))))
        pygame.draw.rect(self.screen, color, filled, border_radius=4)
        conf = self.font.render(f"conf {confidence:.2f}", True, _lerp(PAPER, color, 0.3))
        self.screen.blit(conf, (bar.right + 12, floor + 66))

    def _draw_crash(self, world: World) -> None:
        pygame = self.pygame
        tint = pygame.Surface((WINDOW_SIZE[0], int(FIELD_HEIGHT)), pygame.SRCALPHA)
        tint.fill((200, 60, 50, 60))
        self.screen.blit(tint, (0, 0))
        text = f"crashed into {world.crashed}"
        surface = self.font_bold.render(text, True, PAPER)
        box = surface.get_rect(center=(WINDOW_SIZE[0] // 2, int(FIELD_HEIGHT) // 2))
        panel = box.inflate(28, 20)
        backing = pygame.Surface(panel.size, pygame.SRCALPHA)
        backing.fill((28, 34, 44, 215))
        self.screen.blit(backing, panel)
        pygame.draw.rect(self.screen, BIRD_DEAD, panel, 2, border_radius=6)
        self.screen.blit(surface, box)

    def _shadowed(self, font, text: str, position: tuple[int, int]) -> None:
        """Outlined text, so the score stays readable over sky and pipe alike."""
        x, y = position
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            self.screen.blit(font.render(text, True, INK), (x + dx, y + dy))
        self.screen.blit(font.render(text, True, PAPER), (x, y))

    def close(self) -> None:
        self.pygame.quit()


# ---------------------------------------------------------------------- self test
# Synthetic situations run through the same builders the live loop uses, so the
# strings under test are the strings sent to the model.
def _approach_case(offset: float, blocked: str) -> str:
    return approach_state(Projection(offset, blocked))


SELF_TEST_CASES: list[tuple[str, str, str]] = [
    # Coasting goes through: hold, whatever the offset within the opening.
    ("approach", _approach_case(0.0, ""), "coast"),
    ("approach", _approach_case(40.0, ""), "coast"),
    ("approach", _approach_case(-40.0, ""), "coast"),
    # Coasting passes below the opening: climb.
    ("approach", _approach_case(90.0, "the bottom pipe"), "flap"),
    ("approach", _approach_case(240.0, "the floor"), "flap"),
    ("approach", _approach_case(66.0, "the bottom pipe"), "flap"),
    # Coasting passes above it: another flap would only make that worse.
    ("approach", _approach_case(-90.0, "the top pipe"), "coast"),
    ("approach", _approach_case(-240.0, "the ceiling"), "coast"),
    # line_up is a single comparison against the target line.
    ("line_up", line_up_state(74.0), "flap"),
    ("line_up", line_up_state(10.0), "flap"),
    ("line_up", line_up_state(140.0), "flap"),
    ("line_up", line_up_state(-60.0), "coast"),
    ("line_up", line_up_state(-9.0), "coast"),
    ("line_up", line_up_state(-2.0), "coast"),
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
            f"  {'ok  ' if hit else 'FAIL'} {phase:<9} expected={expected:<6} "
            f"got={answer['choice']:<6} conf={answer['confidence']:.2f}"
        )
    print(f"  {passed}/{len(SELF_TEST_CASES)}")
    return passed == len(SELF_TEST_CASES)


# --------------------------------------------------------------------------- loop
def run_episode(
    service: TypeSafeReplica | None,
    *,
    decisions: int,
    seed: int | None,
    view: PygameView | None,
    show_ascii: bool,
    quiet: bool,
) -> dict:
    """Play one episode and return its summary.

    ``decision_wall`` accumulates only the decisions, so the real-time ratio is
    reported against deciding rather than against drawing: watching the game must
    not flatter or penalise the number being claimed.
    """
    world = World.new(seed)
    latency = {phase: LatencyLog(phase) for phase in CONTRACT}
    action_mix: dict[str, int] = {}
    decision_wall = 0.0
    step = 0

    while step < decisions and world.alive:
        step += 1
        phase, pipe = next_phase(world)

        if phase == "approach" and pipe is not None:
            coast = project(world, pipe)
            state_text = approach_state(coast)
            note = (f"pipe {max(0.0, pipe.x - BIRD_X - BIRD_RADIUS):3.0f} ahead, "
                    f"coast {coast.offset:+4.0f} {coast.blocked or 'clear'}")
        else:
            state_text = line_up_state(coast_offset(world, target_line(pipe)))
            note = state_text

        started = time.perf_counter()
        if service is None:
            action = baseline_action(world)
            confidence = 1.0
        else:
            answer = service.system_one(
                state=state_text, questions={phase: CONTRACT[phase]}
            ).answers[phase]
            action = answer["choice"]
            confidence = answer["confidence"]
        decision_wall += time.perf_counter() - started
        ms = latency[phase].record(started)
        action_mix[action] = action_mix.get(action, 0) + 1

        # The action covers the whole block: one press on the first frame, then
        # coast. Rendering happens here, after the decision has been timed.
        for frame in range(FRAMES_PER_DECISION):
            world.step(flap=action == "flap" and frame == 0)
            if view is not None:
                view.draw(world, phase, action, confidence, ms)
            if show_ascii:
                print(f"\033[H\033[J{ascii_frame(world)}", end="")
            if not world.alive:
                break

        if show_ascii:
            print(
                f"\nscore={world.score}  step {step}  {phase} -> {action}  "
                f"conf={confidence:.2f}  {ms:.0f}ms"
            )
        elif not quiet:
            print(
                f"{step:4d}  {phase:<9} {action:<5} conf={confidence:.2f}  "
                f"{ms:6.1f}ms  score={world.score:2d}  | {note[:58]}"
            )

    return {
        "decisions": step,
        "score": world.score,
        "crashed": world.crashed,
        "game_seconds": world.game_seconds,
        "decision_wall": decision_wall,
        "action_mix": action_mix,
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 plays Flappy Bird in real time.",
    )
    parser.add_argument("--decisions", type=int, default=150,
                        help="model decisions per episode (default: 150)")
    parser.add_argument("--episodes", type=int, default=1, help="episodes to play")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the pipes")
    parser.add_argument("--watch", action="store_true",
                        help="draw the game in a pygame window (needs pygame)")
    parser.add_argument("--ascii", dest="show_ascii", action="store_true",
                        help="draw the game in the terminal")
    parser.add_argument("--baseline", action="store_true",
                        help="play with the criteria rule instead of the model")
    parser.add_argument("--quiet", action="store_true",
                        help="summary only, no per-decision log")
    parser.add_argument("--self-test", action="store_true",
                        help="check the questions against synthetic states and exit")
    args = parser.parse_args()

    if args.self_test:
        print_header("Question self-test")
        service = TypeSafeReplica()
        service.engine.load()
        sys.exit(0 if self_test(service) else 1)

    who = "criteria rule" if args.baseline else "Gemma 4"
    print_header(
        f"Flappy Bird - {who}, {args.episodes} episode(s), "
        f"{args.decisions} decisions each"
    )
    print(
        f"{FRAME_HZ:.0f} fps, one decision every {FRAMES_PER_DECISION} frames = "
        f"{FRAMES_PER_DECISION / FRAME_HZ * 1000:.0f}ms of game time per decision"
    )

    service: TypeSafeReplica | None = None
    if not args.baseline:
        print("Loading model (one-time cost)...")
        service = TypeSafeReplica()
        service.engine.load()
        print(f"Model loaded in {service.engine.load_seconds:.1f}s")
        warm_ms = service.warm(CONTRACT)
        print(f"Question prefixes warmed in {warm_ms:.0f}ms")

    view: PygameView | None = None
    if args.watch:
        try:
            view = PygameView()
        except ImportError:
            print("pygame is not installed; falling back to --ascii "
                  "(pip install pygame for the window)")
            args.show_ascii = True
    print()

    results = []
    started = time.perf_counter()
    try:
        for episode in range(1, args.episodes + 1):
            if not args.show_ascii:
                print(f"--- episode {episode} ---")
            seed = None if args.seed is None else args.seed + episode
            result = run_episode(
                service,
                decisions=args.decisions,
                seed=seed,
                view=view,
                show_ascii=args.show_ascii,
                quiet=args.quiet,
            )
            results.append(result)
            outcome = f"crashed into {result['crashed']}" if result["crashed"] else "still flying"
            print(
                f"    {outcome}: pipes={result['score']} "
                f"decisions={result['decisions']} "
                f"game_time={result['game_seconds']:.1f}s\n"
            )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if view is not None:
            view.close()

    if not results:
        return
    wall_s = time.perf_counter() - started
    game_s = sum(r["game_seconds"] for r in results)
    decision_s = sum(r["decision_wall"] for r in results)
    print(f"Total: {len(results)} episode(s) in {wall_s:.1f}s wall clock")
    print(
        f"pipes={sum(r['score'] for r in results)}  "
        f"crashes={sum(1 for r in results if r['crashed'])}  "
        f"decisions={sum(r['decisions'] for r in results)}"
    )
    # The ratio that matters: game time simulated over time spent deciding it. At
    # 1.0 the bird flew at normal speed; below that, the run was slow motion.
    if decision_s > 0:
        print(
            f"real-time ratio: {game_s / decision_s:.2f}x "
            f"({game_s:.1f}s of game in {decision_s:.1f}s of decisions)"
        )
    for phase in CONTRACT:
        samples = [ms for r in results for ms in r["latency"][phase].samples_ms]
        if samples:
            print("  " + LatencyLog(phase, samples).summary())
    mix: dict[str, int] = {}
    for result in results:
        for action, count in result["action_mix"].items():
            mix[action] = mix.get(action, 0) + count
    print("action mix: " + ", ".join(f"{k}={v}" for k, v in sorted(mix.items())))


if __name__ == "__main__":
    main()
