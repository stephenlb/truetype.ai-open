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
python demo/doom_demo.py       # real Doom via ViZDoom
python demo/game_demo.py       # text-adventure game
python demo/web_nav_demo.py    # web-page navigation
python demo/batch_demo.py      # 5 questions in 1 request
python demo/latency_bench.py   # prefix-cache benchmark + correctness check
```

## Measured latency

| Demo | Task | Decision type | Latency (mean) |
|------|------|---------------|----------------|
| `doom_demo.py` | real Doom engine, 3 scenarios | choice (3-4 actions) | **~180ms** (5.4 decisions/sec) |
| `game_demo.py` | 4-room text adventure | choice | ~500ms (every step a new prefix) |
| `web_nav_demo.py` | support-portal refund flow | choice | ~700ms (every step a new prefix) |
| `batch_demo.py` | 5 questions over 1 message | noul + choice + score | ~790ms warm / ~4.5s cold |

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

Two changes took a Doom decision from **1017ms to ~180ms (5.8x)** and the test suite
from **39.8s to 15.6s**, with zero decision changes.

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
```

Results from 40-decision runs:

| Scenario | Outcome | Latency (p50) | Throughput |
|----------|---------|---------------|------------|
| `defend_the_center` | 4 kills, health 100 | 179ms | 5.4 dec/sec |
| `health_gathering` | health held at 100, reward 160 | 175ms | 5.5 dec/sec |
| `deadly_corridor` | 2-3 kills, health 70 | 180ms | 5.4 dec/sec |

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

- At ~5.4 decisions/sec against an engine running 35 tics/sec, this is still **not
  real-time Doom**. 40 decisions at 4 tics each covers ~4.6s of game time.
- The state is a text reduction of the engine's labels, not pixels. The model reasons
  over a symbolic summary, not the frame.
