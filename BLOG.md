# We Rebuilt Jev's API on an Open Model and Used It to Play Doom

LLM calls often dominate agent-loop latency. TypeSafe AI's Jev replaces chat completions with a typed decision API. Its hosted "System One" models return decisions and calibrated probabilities that software can use directly. At Blocks.ai, we rebuilt that interface on an open model and used it to play Doom within Jev's published latency range.

TypeSafe AI released Jev on September 15, 2026. The company describes it as a System One Model that returns typed decisions with calibrated probabilities in 70-500ms at a cost of $42 per billion input tokens. The API accepts unstructured state and returns a typed probabilistic decision without generating prose. One launch demo used it to play Doom.

We wanted to find out how much of Jev's utility comes from the interface itself.

**[github.com/stephenlb/truetype.ai-open](https://github.com/stephenlb/truetype.ai-open)**

Our replica runs on local hardware, implements the same `POST /v1/systemone` API, supports the same three question types, and plays Doom. The core implementation is about 1,200 lines of Python.

## TypeSafe's launch

The launch tweet drew nearly 30 million views. TypeSafe presented RLCD, reinforcement learning for calibrated decisions, as an alternative to RLHF. It claimed up to 200x lower latency, 400x lower cost, free output tokens, and zero hallucinations. A benchmark slide placed Jev near frontier chat models on decision tasks. Its tagline was "we're building prod not God."

The demos put Doom inside the game loop. A five-hop Wikiracing run finished in about half a second; three comparison models each took four to five seconds. In another demo, 50 simulated townspeople reacted to a broadcast in under a second: 39 carried on, 6 investigated, 4 joined in, and 1 warned the others. TypeSafe also showed Jev sorting 150,000 Skittles, controlling a driving loop, and playing Super Smash Bros. Melee.

TypeSafe named chess, coding from scratch, and open-ended chat as poor fits for Jev. In one chess match, its opponent led by 16 points of material at move 29 and promoted a second queen. Jev won on time because its opponent spent 6 to 15 seconds per move while Jev answered in 2.6.

We could not verify TypeSafe's training claims, so we tested a narrower question: could an open base model behind the same one-token interface make useful decisions within the published latency range? It did on our test suite and Doom demos.

## What we replicated

We copied the API contract: unstructured state and typed questions go in; typed answers, probabilities, and confidence come out. We did not retrain a model or reproduce TypeSafe's architecture, parallel sampler, or RLCD method.

The engine loads the base Gemma 4 12B model, runs one `forward()` pass, and reads the next-token logits. It keeps the 26 token IDs for A through Z and applies a temperature-0.7 softmax over the letters allowed by the question. The code in `src/` has no `generate()`, sampling, or decode loop. Each answer uses one output token.

Because the model does not write text, it cannot invent a label. If a `choice` question lists `returns`, `shipping`, and `billing`, the response must contain one of them. The caller can act on the result without checking a generated string.

FastAPI validates and authorizes each request. It rejects a temperature of zero at parse time, returns 422 for an empty `questions` map, and returns 503 while the weights load. The service serializes non-string state as JSON, converts each question into a typed specification with legal letters, renders one prompt per question, and scores it. Every prompt uses the same four fields: `Choices:`, `Question:`, `Text:`, and `Answer:`. The caller's state comes last, leaving a byte-identical prefix for the engine to cache. One forward pass returns logits for all 26 letters; the service renormalizes the legal subset and builds the typed answer. A single process owns the engine and loads the 12B weights once.

Two values check the readout on every call. `letter_mass` measures how much of the model's full-vocabulary probability sits on A through Z: `exp(logsumexp(letters) - logsumexp(vocab))`. A low value means the engine is ranking letters from the tail of the distribution, where a plausible argmax may be noise. For `choice` and `score`, `confidence` is normalized entropy, `1 - H / log n`; a value of 1.0 means the model assigned all probability to one option. `usage` reports one output token per question and estimates input tokens from character count.

The `include_debug` flag returns the prompt, raw letter logits, and ranked top-k letters for each question. We found most of the bugs below with this output.

Few-shot examples tune the readout. For `choice` and `score`, the renderer creates one synthetic round trip per option from the caller's criteria. It orders them low, high, then ascending to avoid teaching an A, B, C positional shortcut. For `noul`, it uses six fixed demonstrations with predicates unrelated to the target question. An urgency example can teach the answer format for a refund question because the target options define the task and the example labels remain truthful. The renderer also turns the caller's instruction into affirmative and negative options. "Does the text request a refund?" becomes "Yes, the text requests a refund." and "No, the text does not request a refund."

## Results

The replica answers all 50 cases in our main suite: 20 `noul`, 20 `choice`, and 10 `score`. Two structural tests cover the one-token contract and prefix cache. It also passes a 63-case set containing the main suite, a 5-case Doom bearing probe, and 8 `noul` cases that use a prompt layout excluded from tuning.

Latency, measured on an Apple Silicon Mac in bfloat16, across 55 warm states:

| Metric | Value |
|---|---|
| p50 | 113ms |
| p95 | 134ms |
| max | 139ms |

These results fall within TypeSafe's published 70-500ms range. Our benchmark fails when p95 exceeds 150ms. TypeSafe's home page compares an LLM at 8.566 seconds and $0.013880 per call with Jev at 0.114 seconds and $0.000081. We did not verify those API prices. On a similar task, our cold path takes about one second. The warm path is about two orders of magnitude faster than TypeSafe's LLM figure. Cached and uncached paths produce identical decisions. Caching cut the Python test suite from 39.8 seconds to 15.6 seconds and reduced a Doom decision from 1017ms to 133ms, a 7.6x improvement.

## Doom on ViZDoom

The demo runs Doom through [ViZDoom](https://vizdoom.farama.org/), which ships the Freedoom IWADs. It reads object labels, health, ammo, and kill count from the engine, so it does not need screen capture or macOS Screen Recording permission. The model receives a short text summary, chooses one of three or four actions, and advances the game.

```bash
pip install vizdoom
python demo/doom_demo.py                  # defend_the_center
python demo/doom_demo.py health_gathering # walk onto medkits to survive
python demo/doom_demo.py deadly_corridor  # fight down a corridor
python demo/doom_demo.py --watch          # ...and watch it play
```

`--watch` opens ViZDoom's 640x480 SDL window with the HUD and crosshair visible. It draws every skipped tic to keep motion continuous. Rendering sits outside the timed decision, so watched and headless runs have the same per-decision latency. Wall-clock throughput falls to about 5.5 decisions per second.

| Scenario | Outcome | Latency (p50) | Throughput |
|---|---|---|---|
| `defend_the_center` | 5 kills, health 100 | 133ms | 7.2 dec/sec |
| `health_gathering` | health held at 100, reward 160 | 132ms | 7.5 dec/sec |
| `deadly_corridor` | 2-3 kills, health 70 | 134ms | 7.2 dec/sec |

The repository includes a full-level agent for navigation and combat:

```bash
python doom/doom_full_game_demo.py            # Freedoom 2 MAP01, 150 decisions
python doom/doom_full_game_demo.py --watch
python doom/doom_full_game_demo.py --decisions 300
```

In a 300-decision MAP01 run, the agent gets 5 kills and 8 items without dying. Engine data selects a phase (`combat`, `stuck`, or `navigate`), and the model answers one question for that phase. A BFS route over the level's blocking lines supplies the next waypoint, then the model chooses a local direction. With the same code and decision budget, it gets 2 kills and 2 items on Freedoom 1's cramped, door-heavy E1M1.

Doom simulates at 35 tics per second. At 7.2 decisions per second and 4 tics per decision, 40 decisions cover about 4.6 seconds of game time. The live demo runs slower than real time. You can see the delay as the agent sweeps a room for a monster and fires. TypeSafe noted the same limitation in its demo. Model inference remains the bottleneck.

The full-game run exposed bugs in prompt wording and world modeling. An inverted bearing sign made the agent orbit its target. A shooting threshold of 700 units produced 229 shots and no kills because a monster at that distance is a three-pixel sprite. Switching between two flanking monsters on every tick made the agent oscillate instead of firing. We checked each fix against a 252-case matrix. All three fixes changed the prompt or surrounding code; the model stayed the same.

Ordinary code handles deterministic work. An early version asked the model which direction looked open on every tick. It chose a new heading every frame and shuffled against walls. The engine now builds a walkability grid from blocking lines and runs BFS to the objective. On E1M1, that change cut stuck decisions from about 33 to about 3. The surrounding code picks the phase, holds a target, handles doors and reloading, and squares the crosshair. The model handles the remaining local choice.

## Prefix caching

The first working version took 1017ms per Doom decision. The forward pass consumed 99% of the call, and a static few-shot prefix accounted for 71-85% of each prompt. The engine now caches the prefix KV and prefills only the state tail. Warm calls encode about 20 tokens instead of 370.

The engine tokenizes the full prompt as one string, compares its head with the cached prefix tokens, and falls back to a full forward pass on mismatch. On a hit, it reuses the prefix slice from a previous forward pass instead of replaying or re-tokenizing text. After use, `cache.crop(-tail_len)` restores the cache to the prefix length. A miss costs about the same as running without a cache.

At 369 tokens, each cached prefix uses about 127MB of KV. The cache defaults to 16 entries and has a hard ceiling of 64. Before scoring, the service expands it to cover the request's distinct questions. Without enough entries, an 8-entry cache handled a 10-question request in 7909ms; disabling the cache took 7687ms.

Three results from the benchmark:

```
Single question, warm cache (mean of 5):
  Doom 3-option choice   cache on   173.1ms   off  1007.4ms   5.82x
  noul refund            cache on   126.8ms   off   788.2ms   6.22x

5-question request:
  cache off (batched)         3881.0ms
  cache on, cold (all miss)   4143.8ms   1.07x vs off
  cache on, warm (all hit)     794.4ms   4.89x vs off
```

Batching questions into one forward pass saves little on this hardware because it still processes every prompt's full prefix. Forward time scales with the total tokens processed at about 2ms per token. The service therefore scores questions in sequence against per-question cached prefixes. A cold multi-question request is about 7% slower than an uncached batch; later requests are about 5x faster.

A profile of the warm path measured 1.5ms to tokenize the full prompt, about 166ms for the forward pass over the cached prefix and tail, 4.9ms to extract the 26 letters and mass, and 0.09ms to crop the cache. The forward pass consumes about 97% of the call. Its floor sits near 60ms and changes little with prefix length; each tail token adds about 2ms. Shortening the state tail offers the best route to further gains. Float16 and bfloat16 perform the same here, 999ms against 1000ms, because weight memory bandwidth limits the workload.

Pre-filling pays off when a prefix is reused. Doom uses the same prefix 40 times, so warming helps. In the text-adventure demo, warming four one-shot prefixes costs 4.8 seconds and saves 1.6 seconds during play, so that demo starts cold.

## The trailing-space bug

The prompt used to end with `"Answer: "`. The tokenizer treated the trailing space as its own token. At that position the model preferred a bare `"Y"`, while the engine scored space-prefixed IDs such as `"▁Y"`. The model's top token was `'1'` or a newline. Accuracy stayed at 50/50 because the relative letter ranking survived, but the 26 scored letters held almost none of the probability mass: 0.0000 with the space and 0.9871 without it. The mean margin between the top two letters rose from 0.792 to 0.914 after the fix. Before the fix, every `confidence` and `score` value came from renormalizing a subset that contained about 0.01% of the probability.

A ticket describing a duplicate charge and an app crash exposed the problem. It scored 0.77, or cosmetic, when it should have scored 1.82, or blocking. Doom's `attack` confidence also hovered near 0.5 even when the action was correct. Low confidence can indicate an underspecified prompt. Here it revealed a broken readout. After the fix, `defend_the_center` chose `attack` 40 out of 40 times but stalled at one kill because no option covered a state with no visible monster. Adding that rule raised confidence to 0.85 and brought the scenario back to 4 kills.

The structural test now checks that the engine reads one token, all 26 letter IDs are distinct, the answer slot has no trailing space, and letter mass exceeds 0.5 for all three question types. Reintroducing the space fails both the string and mass checks. A label-only test suite misses this failure because accuracy remains perfect.

## Scope and limits

The replica supports `noul`, `choice`, and `score` with up to 26 options. It returns probabilities normalized across legal answers and entropy-based confidence. It also handles multi-question requests, supports an optional bearer key, and ships with a Docker image and demos. A `score` is the expected value across ordinal levels. An expected value of 1.0 can mean confident support for level 1 or an even split between 0 and 2, so read it alongside confidence.

The model produces a full 26-letter readout on each call. The service scores all 26 letters and renormalizes over the question's legal set, so `top_k` cannot cut off valid options. The setting controls only the debug output.

We did not reproduce TypeSafe's parallel sampler, training, the workflow evaluations behind the 193.6x and 444.6x figures, Wikiracing, or pixel input for Doom. The replica uses a base 12B model and inherits its judgment.

The replica does not guarantee identical probabilities on every run. Test it with inputs whose wording changes but meaning does not. Temperature cannot change the winning `noul` or `choice` option because it applies a monotonic transform to the logits; it changes only the reported probabilities. Tune it for calibration. The model can choose the wrong answer, so use probability thresholds and escalate low-confidence cases.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" vizdoom
truetype-api
```

The first start downloads `google/gemma-4-12B` (~22GB) and needs about 24GB of RAM. An NVIDIA GPU or Apple Silicon helps. Then:

```bash
curl -s localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice for order A-104. Please refund the duplicate.",
    "questions": {
      "refund": {"type": "noul", "instructions": "Does the text request a refund?"},
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "returns": "Exchanges, refunds, wrong or damaged items",
          "shipping": "Delivery status, delays, lost packages",
          "billing": "Charges, invoices, payment problems"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the reported issue?",
        "criteria": [
          "Cosmetic; no impact to functionality",
          "Broken or degraded feature, but workaround exists",
          "Blocking issue; no workaround exists"
        ]
      }
    }
  }'
```

The response contains typed answers, probabilities, and confidence, using one output token per question. Your code can read `answers.team.choice` and `answers.team.confidence`, apply a threshold, and continue without parsing prose or retrying malformed JSON.

At this latency, local inference can handle small, frequent decisions inside an agent loop. If you build something with the replica or find a failure case, open an issue.

**[github.com/stephenlb/truetype.ai-open](https://github.com/stephenlb/truetype.ai-open)**
