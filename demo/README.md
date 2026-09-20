# Demonstrations

Five demos showing how fast a 12B-parameter base model can make decisions when the
output is restricted to a single letter (A-Z) and a low-temperature softmax.

All demos load `google/gemma-4-12B` once (~5s), then time every decision.

## The decision contract

Every answer is **exactly one token**, and that is a property of how the model is
called rather than a limit we ask it to respect:

- one `forward()` pass, `logits_to_keep=MAX_NEW_TOKENS` (=1), reading `logits[:, -1, :]`
- no `generate()`, no sampling, no decode loop anywhere in `src/`
- logits gathered at **only the 26 A-Z token ids**, all from one token variant
  (space-prefixed `"▁A"`), verified distinct and single-token at load
- softmaxed at **temperature 0.7** over just the letters a question declares legal

`noul` and `choice` answers are the argmax of that distribution. `score` is
deliberately *not* an argmax — it is the expected value `Σ i·pᵢ` over the ordinal
levels, so 1.0 can mean "confidently level 1" or "split between 0 and 2"; read its
`confidence` alongside it.

## Run

```bash
# from the repo root
python demo/doom_demo.py       # real Doom, 3 arena scenarios (add --watch to see it)
python doom/doom_full_game_demo.py  # real Doom, a full level (see below)
python demo/game_demo.py       # text-adventure game
python demo/web_nav_demo.py    # web-page navigation
python demo/batch_demo.py      # 5 questions in 1 request
python demo/latency_bench.py   # prefix-cache benchmark + correctness check
```

## Measured latency

| Demo | Task | Decision type | Latency (mean) |
|------|------|---------------|----------------|
| `doom_demo.py` | real Doom engine, 3 scenarios | choice (3-4 actions) | **~133ms** (7.2 decisions/sec) |
| `game_demo.py` | 4-room text adventure | choice | ~500ms (every step a new prefix) |
| `web_nav_demo.py` | support-portal refund flow | choice | ~680ms (every step a new prefix) |
| `batch_demo.py` | 5 questions over 1 message | noul + choice + score | ~815ms warm / ~4.3s cold |

## The 150ms warm budget

Every question in the suite, measured warm (`demo/latency_bench.py` asserts this):

```
Warm latency budget (<150ms per question, all 50 suite states):
  noul_refund        p50= 121.4ms  p95= 130.1ms  ok
  noul_urgency       p50= 106.2ms  p95= 126.7ms  ok
  choice_team        p50= 128.8ms  p95= 138.9ms  ok
  choice_sentiment   p50= 103.0ms  p95= 129.8ms  ok
  score_severity     p50= 121.5ms  p95= 134.7ms  ok
  doom_choice        p50= 130.4ms  p95= 131.8ms  ok

  overall warm: n=55  p50=113.2ms  p95=134.2ms  max=138.9ms
  budget check: PASS (all questions < 150ms at p95)
```

The assertion fails the benchmark if any question exceeds the budget, so a future
prompt change cannot quietly regress it.

### Where the time goes

Profiling the warm path (Text `prefix=552 tok, tail=51 tok`):

| Phase | Cost |
|---|---|
| tokenize full prompt | 1.5ms |
| **forward (cached prefix + tail)** | **166ms** |
| full-vocab logits `.float()` | 0.06ms |
| letter slice (26 ids) + mass | 4.9ms |
| `cache.crop` | 0.09ms |

The forward pass is ~97% of the call. Two properties of it matter:

- **The floor is ~60ms and barely depends on prefix length.** A 1-token tail against
  a cached prefix measured 59ms at 128 tokens and 64ms at 552 tokens. Shrinking the
  prefix is not a lever.
- **Tail tokens cost ~2ms each.** Doom's 51-token state was 166ms; a 28-token state is
  ~134ms. The tail is the only real lever, and it is what got this under budget.

float16 was also tried and is no faster than bfloat16 (999ms vs 1000ms) — this
workload is memory-bandwidth bound on weights, not compute bound.

### Getting Doom under 150ms

The Doom state tail was 51 tokens at 196ms. Cutting it to 28 tokens at ~134ms
required two changes, each validated separately:

1. **Drop health, ammo and proximity from the state.** None of them change which
   action is correct (the action depends only on bearing), and including them cost a
   disproportionate ~28ms: a 36-token tail ran 163ms while a 28-token tail ran 134ms.
   Health and kills are still printed to the console for the human watching.
2. **Keep the exact wording the criteria reference.** Paraphrasing scored 17/18 while
   the literal `"bearing relative to crosshair"` / `"lined up in crosshair"` phrasing
   scored 18/18 on the same states.

Both were validated on a 252-case matrix (3 scenarios × 6 bearings × 4 proximity
bands × 4 health/ammo states, expectations derived from the criteria text itself so
the test could not encode a wrong answer): **252/252 before and after**, at 133-140ms
warm. Live gameplay confirmed unchanged afterwards — 5 kills / reward 160 / 2 kills
across the three scenarios.

## Why not batch the questions into one forward pass?

Because on this hardware batching buys almost nothing, and the cache buys a lot.
Measured on 10 questions over one state (`demo/latency_bench.py`):

```
cache off (1 batched forward)    7687.3ms    <- all 10 prompts, one forward
cache on, hard cap 8 (THRASH)    7909.7ms    hits+0 evictions+20
cache on, auto-grown to 10       1599.6ms    4.81x vs off   hits+20 evictions+0
```

A batched forward still processes every prompt's full ~350-token prefix — batching
parallelises the same tokens, it does not eliminate them. Forward time here scales
with *total tokens processed* (~2ms/token), so one batched pass over 10 full prompts
is ~7.7s no matter how many rows you stack. The 4.8x win comes entirely from the
prefix KV cache: a warm call encodes only the ~20-token state tail instead of the
~370-token full prompt. Different questions have different prefixes, so there is
nothing shared to batch across them.

This is why `system_one` scores questions **sequentially** against per-question
cached prefixes rather than in one batched forward.

### The capacity requirement (and the bug it caused)

The LRU must hold **every** prefix a request uses. At 10 distinct questions with an
8-entry capacity, the cache evicts a prefix before it is ever revisited: `hits+0,
evictions+20`, warm latency 7909ms — *worse* than not caching at all (7687ms), and
no error was raised.

`ensure_prefix_capacity()` now grows the cache to the request's distinct-question
count before scoring, clamped by `prefix_cache_hard_cap` (default 64, ~8GB of KV at
the measured ~127MB/prefix) with a warning if a request exceeds the cap. Default
capacity is 16. Test 52 covers grow / never-shrink / clamp.

### Pre-warming

`TypeSafeReplica.warm(questions)` pre-fills prefixes for known upcoming questions.
`doom_demo.py` uses it so decision 1 is not the only cold call: before warming it
spiked to ~1.2-1.5s; after, decision 1 is ~190ms.

Warming does **not** make a first-seen question faster than one forward pass — a
cache miss already costs exactly that. It relocates the unavoidable prefill to
before the timed loop. That is a net win only when a prefix is reused many times
(Doom reuses one prefix ~40 times, so 1.2s up front saves ~40s) and a net loss when
it is used once (warming `game_demo`'s four one-shot room prefixes costs ~4.8s to
save ~1.6s). `game_demo`, `web_nav_demo` and `batch_demo` are therefore left
unwarmed on purpose.

The API can warm at startup via `TYPESAFE_REPLICA_WARM_QUESTIONS` (a JSON object of
question specs); it is off by default since unknown workloads gain nothing.

## The bug that passed every test

The prompt used to end with `"Answer: "` — with a trailing space. That tokenizes to
a **standalone `'▁'` token**, stranding the space that belongs to the letter:

```
example line : ['Answer', ':', '▁Y']   <- space is part of the letter token
target tail  : ['Answer', ':', '▁']    <- space stranded on its own
```

The engine scored space-prefixed ids (`▁Y`) while the model, sitting after a dangling
space, wanted a bare `"Y"`. Measured at the answer position:

| | before (`"Answer: "`) | after (`"Answer:"`) |
|---|---|---|
| model's actual top-1 token | `'1'` / `'\n\n'` | `'▁Y'` / `'▁N'` |
| probability mass on the 26 scored letters | **0.0000** | **0.9872** (min 0.9424) |
| accuracy, 50-case suite | 50/50 | 50/50 |
| mean top1−top2 margin | 0.792 | 0.914 |

Accuracy was 50/50 either way, because the *relative ordering* of the letters
survived — the argmax was being taken over the noise floor of the distribution and
still happened to rank correctly. What was genuinely broken was calibration: every
`confidence` and `score` renormalised a subset holding ~0.01% of the probability.
Two visible consequences, both now fixed:

- a duplicate-charge + app-crash ticket scored severity **0.77 (cosmetic)**; it now
  scores **1.82 (blocking)**
- Doom `attack` confidence hovered near 0.5 even when correct; it is now ~0.9

**Test 51** (`test_single_token_letter_readout_contract`) guards this. It asserts the
one-token contract, the distinct 26-id A-Z set, that the answer slot has no trailing
space, and that letter mass exceeds 0.5 for all three question types. Verified to
fail on both the string check and the mass check when the bug is reintroduced.

### It also changed a demo's behaviour

With the readout fixed, `defend_the_center` started answering `attack` 40/40 times
and stalled at 1 kill. The cause was not the fix but an **underspecified prompt**: no
option said what to do when *no monster is visible*, so that state was near-uniform
(confidence 0.15) and the winner was arbitrary — pre-fix it happened to fall on
`turn_left` and sweep, post-fix on `attack`. Giving `turn_right` an explicit "also
correct when no monster is visible (sweep to find one)" rule took that state to
confidence 0.85 and the scenario back to 4 kills. Low confidence was the signal that
the prompt, not the model, was at fault.

## Latency work

These changes took a Doom decision from **1017ms to ~133ms (7.6x)**, the test suite
from **39.8s to 15.6s**, and warm per-question latency under the 150ms budget above.
Zero decision changes throughout.

### 1. Prefix KV caching

Profiling showed the forward pass was 99% of a call (tokenising 1.5ms, letter readout
5ms) and that 71-85% of every prompt was a *static* few-shot prefix being recomputed
each time. The engine now caches that prefix's KV (`EngineConfig.prefix_cache`, on by
default) and prefills only the tail.

Safety properties, because a wrong cache is worse than a slow one:

- The full prompt is always tokenised as one string and the cached prefix tokens are
  compared against its head. On mismatch the call falls back to a full forward pass.
- A cache **miss** costs exactly one forward pass over the whole prompt — the same
  work the uncached path does — and keeps the prefix half of that pass's KV. An
  earlier version prefilled the prefix separately, making misses ~13% *slower* than
  no caching; that surfaced as `game_demo` regressing from 550ms to 660ms.
- `max_cached_prefixes` (default 8) must be >= the distinct questions in a request.
  At 4, a 5-question request thrashed the LRU and re-paid every prefill: 5067ms vs
  3900ms uncached.

Each cached prefix costs ~127MB of KV at 369 tokens.

### 2. Putting the state last in the prompt

Layout changed from `Text / Question / Choices / Answer:` to
`Choices / Question / Text / Answer:`. Everything except the caller's state is now
static, so the cacheable prefix grows and the tail shrinks from 68 to 20 tokens —
warm Doom decisions went 357ms → 178ms.

Adopted only after accuracy held at **63/63**: the 50-case suite, the 5-case Doom
bearing probe, and 8 held-out noul cases written specifically to test a layout not
used during tuning. An intermediate variant that also hoisted the target question
above every example was **rejected despite matching accuracy**, because it relabelled
the generic noul demos under a question they were not written for ("I need this fixed
immediately" → `Y` under "does the text request a refund?").

### Results

```
Single question, warm cache (mean of 5):
  Doom 3-option choice   cache on   173.1ms   off  1007.4ms   5.82x
  noul refund            cache on   126.8ms   off   788.2ms   6.22x

5-question request:
  cache off (batched)         3881.0ms
  cache on, cold (all miss)   4143.8ms   1.07x vs off
  cache on, warm (all hit)     794.4ms   4.89x vs off
```

Reproduce with `python demo/latency_bench.py`, which also asserts the cached and
uncached paths reach the same decisions.

### Honest limits

- **Cold multi-question requests are ~7% slower than uncached.** Distinct questions
  have distinct prefixes, so a first-ever request pays a miss per question and runs
  them sequentially rather than as one batch. Every later request is ~5x faster.
- `game_demo` and `web_nav_demo` benefit little: every step presents different
  criteria, so nearly every call is a cold miss.
- float16 was tried and is not faster than bfloat16 (999ms vs 1000ms) — this is
  memory-bandwidth bound on weights, not arithmetic bound.
- Temperature cannot change a `noul` or `choice` decision (it is a monotonic
  transform of the logits); it only sharpens reported probabilities, `confidence`,
  and `score` expected values. All of 1.0/0.8/0.7/0.6/0.5/0.3 score 50/50; 0.7 was
  chosen because 0.3 pins confidence near 1.0 and discards the calibration signal.

## Doom demo

Runs the **actual Doom engine**. [ViZDoom](https://vizdoom.farama.org/) is a ZDoom
fork built for AI agents and ships the Freedoom IWADs, so object labels, health,
ammo and kill count come straight out of the engine — no screen capture and no
macOS Screen Recording permission.

```bash
python demo/doom_demo.py                    # defend_the_center (default)
python demo/doom_demo.py health_gathering   # walk onto medkits to survive
python demo/doom_demo.py deadly_corridor    # fight down a corridor

python demo/doom_demo.py --watch            # ...and actually watch it play
```

### Watching it play

`--watch` opens ViZDoom's own window (640x480, HUD and crosshair on, every skipped
tic drawn so motion is continuous). This is the engine's native SDL renderer, not an
OS screen grab, so it does not need macOS Screen Recording permission — the approach
that failed when this demo was first attempted against `chocolate-doom`.

The render cost lands inside `make_action`, outside the timed decision, so
**per-decision latency is unchanged** watched vs headless (~133ms both). What drops
is wall-clock throughput: **~5.5 decisions/sec watched vs ~7.2 headless**. Headless
remains the default so the benchmark figures above stay reproducible.

Resolution is safe to change: bearings are normalised by `screen_width / 2`, and the
model reads engine labels rather than pixels, so the agent behaves identically at
either resolution.

Results from 40-decision runs:

| Scenario | Outcome | Latency (p50) | Throughput |
|----------|---------|---------------|------------|
| `defend_the_center` | 5 kills, health 100 | 133ms | 7.2 dec/sec |
| `health_gathering` | health held at 100, reward 160 | 132ms | 7.5 dec/sec |
| `deadly_corridor` | 2-3 kills, health 70 | 134ms | 7.2 dec/sec |

The demo warms the question's KV prefix before the loop, so decision 1 is not the
only cold call (it used to spike to ~1.3s).

### Three things were required to make it play rather than flail

1. **The decisive fact must be the last line of the prompt.** A verbose state block
   (health, ammo, a list of visible objects) made the model answer `turn_left` on
   every tick — 0 kills. Moving the target bearing to the final line took a
   synthetic 5-case probe from 2/5 to 5/5.
2. **Option descriptions must encode when the action is correct,** including the
   fallback case. "Rotate left" is not enough; "Rotate left to aim — correct when the
   nearest monster is to the left of the crosshair" is. Any game state no option
   describes becomes a coin flip.
3. **A sticky target.** With two monsters flanking the player, picking the tallest
   label each tick flips the target every frame, so the agent oscillates and never
   fires. Holding the previous target unless something is clearly closer fixed
   `deadly_corridor` from death-without-firing to 3 kills.

### Doom limits

- At ~7.2 decisions/sec against an engine running 35 tics/sec, this is still **not
  real-time Doom**. 40 decisions at 4 tics each covers ~4.6s of game time.
- The state is a text reduction of the engine's labels, not pixels. The model reasons
  over a symbolic summary, not the frame.

## Full-game Doom demo (`doom/doom_full_game_demo.py`)

The arena demo answers one question with three actions. This one plays a **whole
level** — Freedoom MAP01 by default, or any IWAD/map — and puts the model in charge
of navigation as well as combat.

```bash
python doom/doom_full_game_demo.py                  # Freedoom 2 MAP01, 150 decisions
python doom/doom_full_game_demo.py --watch          # ...and watch it play
python doom/doom_full_game_demo.py --decisions 300
python doom/doom_full_game_demo.py --map E1M1       # the cramped opening level
python doom/doom_full_game_demo.py --iwad doom2.wad --map MAP01 --skill 2
python doom/doom_full_game_demo.py --self-test      # check the questions, no engine
```

### What the model actually decides

The phase is chosen deterministically from engine data, and the model answers only
the ambiguous question for that phase:

| Phase | When | Options |
|---|---|---|
| `combat` | monster labels on screen | shoot / advance / turn left / turn right / back away |
| `stuck` | forward has failed several times | strafe left / strafe right / back away |
| `navigate` | nothing to fight, not stuck | forward / turn left / turn right |

Everything else is engine bookkeeping, because it is mechanical rather than
ambiguous: a BFS route on the level's blocking lines picks the waypoint, `USE` rides
along with forward so doors open on contact, the best loaded weapon is selected when
the current one runs dry, and a short corrective turn (with the trigger held)
squares the crosshair before a shot. Measured warm latency: combat ~175ms, stuck
~160ms, navigate ~130ms — three cached prefixes.

### Why MAP01 is the default

E1M1 is cramped and door-heavy, and this agent is much worse at doors than at open
rooms. Same code, same 300-decision budget, two maps:

| Map | Kills | Items | Stuck decisions |
|---|---|---|---|
| `E1M1` (Freedoom 1) | 2 | 2 | ~7 |
| `MAP01` (Freedoom 2) | 5 | 8 | ~3 |

The model itself is identical; the level geometry is the variable. `--map E1M1`
still works, and pairs with the Freedoom 1 IWAD automatically.

### The biggest effectiveness change: real pathfinding

The first version reacted to nearby geometry — "the widest gap is to the left,
turn that way" — and re-decided the direction every tick. That shuffles at walls,
because each tick's instruction is local and there is no memory of where the agent
was going. The engine hands over every blocking line in the level, so the demo now
builds a walkability grid (32-unit cells, 20-unit wall clearance) and runs BFS to
the objective. The state reports the next waypoint on that route rather than the
distant item, turning "walk around this wall" into a local "head that way" the
model handles well. Grid build is 2-3ms per level; routing is under 1ms.

This did not change latency (the prompt stayed a single line) but it did change how
much of the map the agent crosses. On E1M1 it cut stuck decisions from ~33 to ~3;
the earlier numbers were dominated by wall-shuffling, not by combat.

### Turning points that were measured, not guessed

1. **Bearing sign.** The engine's angle increases counterclockwise, so a bearing
   computed as `object_angle - facing` is *positive to the player's left*. With the
   sign inverted the model turned away from its target, which pushes the bearing
   further out, and it orbited forever.
2. **`TURN180` + forward, not + backward.** Turning 180 and walking *backward*
   cancels itself out: the 180 fixes facing and then backward heads back into the
   wall. Forward walks out.
3. **`shoot` needs ~20 engine tics.** At 4 tics the model pressed fire and nothing
   left the barrel; 16 tics is still the weapon raise. 20 is one shot, 32 is two.
4. **Shoot at ≤350 units.** A 700-unit "shoot" rule produced 229 shots and zero
   kills — a distant monster is a 3-pixel sprite and the pistol's spread misses it.
5. **The weapon line biases the answer.** "The selected weapon is pistol with 30
   shots." made the model answer `shoot` on a monster 1900 units away; the same
   state without it answered `advance`. Removed.
6. **The forward band and the state's words must agree.** `bearing_words` says
   "slightly left/right" below 25 degrees and "far left/right" above it, matching
   the 25-degree forward band in the criteria exactly. When the words said
   "slightly right" at -22 degrees but the band was ±30, the model walked forward
   into the wall beside a doorway for 55 decisions; with the bands aligned, the
   same decision has confidence 0.6 instead of 0.06.
7. **A blocked forward is not "stuck" — a door needs about four presses.** The
   first failure used to switch to the stuck question, so the agent strafed away
   from every closed door. Forward now retries several times before the state calls
   the player stuck.
8. **Do not cap movement macros at 4 tics.** `--tics` used to default to 4, which
   truncated every calibrated macro (turns 14° instead of 17.5°, walks 20 units
   instead of 40), so each decision changed the world less than the report implied.
   The default is now the calibrated length.

### A worked example of a prompt bug

The first version described bearings in words only. The state said "dead ahead" for
a monster 1900 units away, and the model answered `shoot` at confidence 0.52. The
word and the 350-unit rule disagreed. Naming the band in the state — "close range,
under 350 units" / "far away, over 350 units" — matched the criteria text and scored
5/5. The self-test uses the demo's own state builders, so a wording change cannot
silently keep passing against hand-copied strings.

### Results (300-decision runs, headless, MAP01)

| Seed | Kills | Items | Health | Deaths |
|---|---|---|---|---|
| default | 5 | 8 | 60 | 0 |
| 3 | 5 | 8 | 68 | 0 |

`--self-test` runs 22 synthetic states through the same prompt builders and fails
loudly on a regression; all 22 pass.
