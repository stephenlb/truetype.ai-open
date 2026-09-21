# We Rebuilt Jev's API on an Open Model and Used It to Play Doom

At Blocks.ai, we wanted agents to make routine decisions faster. LLM calls dominate the latency of many agent loops. TypeSafe AI's Jev packages a familiar idea, ML classification, as a typed decision API instead of a chat completion. Its hosted "System One" models return decisions and calibrated probabilities directly to software. We built an open replica of that interface, then used it to play Doom at comparable latency.

TypeSafe AI released Jev on September 15, 2026. The company describes it as a System One Model that returns typed decisions with calibrated probabilities in 70-500ms and costs $42 per billion input tokens. Its demos included Doom. The API accepts unstructured state and returns a typed probabilistic decision, without generating prose.

We wanted to separate the model's contribution from the interface's. We built an open-source replica to test the interface.

**[github.com/stephenlb/truetype.ai-open](https://github.com/stephenlb/truetype.ai-open)**

It runs locally, implements the same `POST /v1/systemone` API, answers the same three question types, and plays Doom. The implementation is about 1,200 lines of Python plus demos.

## TypeSafe's launch

The launch tweet drew nearly 30 million views. TypeSafe presented RLCD, reinforcement learning for calibrated decisions, as an alternative to RLHF. It claimed up to 200x lower latency, up to 400x lower cost, free output tokens, and zero hallucinations. Its benchmark slide placed Jev near frontier chat models on decision tasks. The tagline was "we're building prod not God."

The demos showed Doom running inside the game loop and a five-hop Wikiracing run finishing in about half a second, compared with four to five seconds for each of three other models. A town of 50 characters reacted to a broadcast in under a second: 39 carried on, 6 investigated, 4 joined in, and 1 warned the others. Other demos sorted 150,000 Skittles, built a driving loop in an hour, and played Super Smash Bros. Melee.

TypeSafe also named poor fits for Jev: chess, coding from scratch, and open-ended chat. In one chess match, its opponent led by 16 points of material at move 29 and promoted a second queen. Jev won on time because the opponent spent 6 to 15 seconds per move while Jev answered in 2.6.

We cannot verify TypeSafe's training claims. We can test whether an open base model, behind the same one-token interface, produces useful decisions in the published latency range. On our suite and in Doom, it does.

## What we replicated

We copied the API contract: unstructured state and typed questions in, typed answers with probabilities and confidence out. We did not retrain a model or reproduce TypeSafe's architecture, parallel sampler, or RLCD method.

The implementation behind that contract is small. The engine loads the base Gemma 4 12B model, runs one `forward()` pass, and reads the next-token logits. It keeps the 26 token IDs for A through Z and applies a temperature-0.7 softmax over the letters that the question declares legal. There is no `generate()`, sampling, or decode loop in `src/`. The code permits exactly one output token per answer.

Because the model never writes text, it cannot invent a label. If a `choice` question lists `returns`, `shipping`, and `billing`, the response must contain one of those options. That constraint becomes useful when other software acts on the answer.

FastAPI validates and authorizes each request. It rejects a temperature of zero at parse time, returns 422 for an empty `questions` map, and returns 503 if the weights are still loading. The service serializes non-string state as JSON, converts each question into a typed specification with legal letters, renders one prompt per question, and scores it. Every prompt uses the same layout: `Choices:`, `Question:`, `Text:`, `Answer:`. The caller's state comes last, leaving a byte-identical prefix that the engine can cache. After one forward pass returns logits for all 26 letters, the service renormalizes the legal letters and builds the typed answer. One process owns one engine, so the 12B weights load once.

Two values check the readout on every call. `letter_mass` measures how much of the model's full-vocabulary probability sits on A through Z: `exp(logsumexp(letters) - logsumexp(vocab))`. A low value means the engine is ranking letters from the tail of the distribution, where a plausible argmax may still be noise. For `choice` and `score`, `confidence` is normalized entropy, `1 - H / log n`; 1.0 puts all probability on one option. `usage` reports one output token per question and estimates input tokens from the rendered characters.

The `include_debug` flag returns the prompt, raw letter logits, and ranked top-k letters for each question. We found most of the bugs below with this output.

Few-shot examples tune the readout. For `choice` and `score`, the renderer creates one synthetic round trip per option from the caller's criteria. It orders them low, high, then ascending to avoid teaching an A, B, C positional shortcut. For `noul`, it uses six fixed demonstrations with predicates that differ from the target question. An urgency example can therefore teach the answer format while the caller asks about refunds. The target options carry the task, and the example labels remain truthful. The renderer also rewrites the caller's instruction into affirmative and negative options. "Does the text request a refund?" becomes "Yes, the text requests a refund." and "No, the text does not request a refund."

## The results

The replica answers all 50 cases in our main suite: 20 `noul`, 20 `choice`, and 10 `score`. Two structural tests cover the one-token contract and prefix cache. It also passes a 63-case held-out set containing the main suite, a 5-case Doom bearing probe, and 8 `noul` cases for a prompt layout excluded from tuning.

Latency, measured on an Apple Silicon Mac in bfloat16, across 55 warm states:

| Metric | Value |
|---|---|
| p50 | 113ms |
| p95 | 134ms |
| max | 139ms |

These results fall within TypeSafe's published 70-500ms range. Our benchmark fails when p95 exceeds 150ms. TypeSafe's home page compares an LLM at 8.566 seconds and $0.013880 per call with Jev at 0.114 seconds and $0.000081. We did not verify those API prices. For the same kind of task, our cold path takes about one second and the warm path is roughly two orders of magnitude faster than the LLM figure. Cached and uncached paths produce identical decisions. Caching also cut the Python test suite from 39.8 seconds to 15.6 seconds.

Caching cut a Doom decision from 1017ms to 133ms, a 7.6x improvement, without changing any decisions.

## Doom, on the actual engine

The demo runs Doom through [ViZDoom](https://vizdoom.farama.org/), which ships the Freedoom IWADs. It reads object labels, health, ammo, and kill count from the engine, avoiding screen capture and macOS Screen Recording permission. The model receives a short text summary, chooses one of three or four actions, and advances the game.

```bash
pip install vizdoom
python demo/doom_demo.py                  # defend_the_center
python demo/doom_demo.py health_gathering # walk onto medkits to survive
python demo/doom_demo.py deadly_corridor  # fight down a corridor
python demo/doom_demo.py --watch          # ...and watch it play
```

`--watch` opens ViZDoom's 640x480 SDL window with the HUD and crosshair visible. It draws every skipped tic to keep motion continuous. Rendering sits outside the timed decision, so watched and headless runs have the same per-decision latency, though wall-clock throughput falls to about 5.5 decisions per second.

| Scenario | Outcome | Latency (p50) | Throughput |
|---|---|---|---|
| `defend_the_center` | 5 kills, health 100 | 133ms | 7.2 dec/sec |
| `health_gathering` | health held at 100, reward 160 | 132ms | 7.5 dec/sec |
| `deadly_corridor` | 2-3 kills, health 70 | 134ms | 7.2 dec/sec |

There is also a full-level agent that plays a whole map with both navigation and combat decisions:

```bash
python doom/doom_full_game_demo.py            # Freedoom 2 MAP01, 150 decisions
python doom/doom_full_game_demo.py --watch
python doom/doom_full_game_demo.py --decisions 300
```

A 300-decision MAP01 run gets 5 kills and 8 items without dying. Engine data selects a phase (`combat`, `stuck`, or `navigate`), and the model answers one ambiguous question for that phase. A BFS route over the level's blocking lines supplies the next waypoint, leaving the model to choose a local direction. With the same code and decision budget, the agent gets 5 kills and 8 items on Freedoom 2's MAP01, then 2 kills and 2 items on Freedoom 1's cramped, door-heavy E1M1.

Doom simulates at 35 tics per second. At 7.2 decisions per second and 4 tics per decision, 40 decisions cover about 4.6 seconds of game time. The agent is live and watchable, but it does not yet play in real time. You can see the delay when it sweeps a room for a monster and fires. TypeSafe noted the same limitation in its demo. The bottleneck is the time required to turn structured state into a decision.

The full-game run exposed bugs in prompt wording and world modeling. An inverted bearing sign made the agent orbit its target. A shoot rule set at 700 units produced 229 shots and no kills because a monster at that distance is a three-pixel sprite. Switching targets between two flanking monsters every tick made the agent oscillate instead of firing. We checked each fix against a 252-case matrix. The model stayed the same; we changed the prompt and surrounding code.

The demo reserves deterministic work for ordinary code. An early version asked the model which direction looked open on every tick. It chose a new heading every frame and shuffled against walls. The engine now builds a walkability grid from blocking lines and runs BFS to the objective. On E1M1, that change cut stuck decisions from about 33 to about 3. The surrounding code picks the phase, holds a target, opens doors, reloads, and squares the crosshair. The model chooses the remaining local action.

## How it gets fast

The first working version took 1017ms per Doom decision. The forward pass consumed 99% of the call, and a static few-shot prefix made up 71-85% of each prompt. The engine now caches the prefix KV and prefills only the state tail. Warm calls encode about 20 tokens instead of 370.

The full prompt is tokenized as one string. The engine compares cached prefix tokens with the head of that string and falls back to a full forward pass on mismatch. On a hit, the cached prefix is the prefix slice of a real forward pass, not replayed or re-tokenized text. After use, `cache.crop(-tail_len)` restores it to the prefix length. A miss costs the same as running without a cache.

At 369 tokens, each cached prefix uses about 127MB of KV. The cache defaults to 16 entries and has a hard ceiling of 64. Before scoring, the service expands it to cover the request's distinct questions. Without enough entries, an 8-entry cache handled a 10-question request in 7909ms, compared with 7687ms with caching disabled.

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

Batching questions into one forward pass buys almost nothing on this hardware. A batched forward still processes every prompt's full prefix. On this machine, forward time scales with the total tokens processed at roughly 2ms per token. The service therefore scores questions in sequence against per-question cached prefixes. A cold multi-question request is about 7% slower than an uncached batch, while later requests are roughly 5x faster.

A profile of the warm path measured 1.5ms to tokenize the full prompt, about 166ms for the forward pass over the cached prefix and tail, 4.9ms to extract the 26 letters and mass, and 0.09ms to crop the cache. The forward pass consumes about 97% of the call. Its floor sits near 60ms and changes little with prefix length; each tail token adds roughly 2ms. Shortening the state tail is the remaining lever. Float16 and bfloat16 perform the same here, 999ms against 1000ms, because weight memory bandwidth limits the workload.

Pre-filling pays off only when a prefix is reused. Doom uses the same prefix 40 times, so warming helps. Four one-shot prefixes in the text-adventure demo cost 4.8 seconds to save 1.6, so that demo starts cold.

## The bug that passed every test

The prompt used to end with `"Answer: "` and a trailing space. The tokenizer treated that space as its own token. At that position the model preferred a bare `"Y"`, while the engine scored space-prefixed IDs such as `"▁Y"`. Its actual top token was `'1'` or a newline. Accuracy stayed at 50/50 because the relative letter ranking survived, but the 26 scored letters held almost none of the probability mass: 0.0000 with the space and 0.9871 without it. The mean margin between the top two letters rose from 0.792 to 0.914 after the fix. Before the fix, every `confidence` and `score` value renormalized a subset containing about 0.01% of the probability.

A duplicate-charge plus app-crash ticket exposed the problem by scoring 0.77, or cosmetic, when it should have scored 1.82, or blocking. Doom's `attack` confidence also hovered near 0.5 even when the action was correct. Low confidence can indicate an underspecified prompt; in this case it revealed a broken readout. After the fix, `defend_the_center` chose `attack` 40 out of 40 times but stalled at one kill because no option covered a state with no visible monster. Adding that rule raised confidence to 0.85 and brought the scenario back to 4 kills.

The structural test now checks that the engine reads one token, all 26 letter IDs are distinct, the answer slot has no trailing space, and letter mass exceeds 0.5 for all three question types. Reintroducing the space fails both the string and mass checks. A label-only test suite misses this failure because accuracy remains perfect.

## Scope and limits

The replica supports `noul`, `choice`, and `score` with up to 26 options; probabilities normalized across legal answers; entropy-based confidence; multi-question requests; an optional bearer key; a Docker image; and the demos above. A `score` is the expected value across ordinal levels. An expected value of 1.0 can mean either confident support for level 1 or an even split between 0 and 2, so read it alongside confidence.

The model always gets the full 26-letter readout. The service scores all 26 letters and renormalizes over the question's legal set, so `top_k` never cuts off valid options. It only controls the debug output.

We did not reproduce TypeSafe's parallel sampler, training, workflow evaluations behind the 193.6x and 444.6x figures, Wikiracing, or pixel input for Doom. This readout replica uses a base 12B model and inherits that model's judgment.

The replica does not guarantee identical probabilities for every run. More useful is consistency across inputs whose wording changes but meaning does not. Temperature cannot change the winning `noul` or `choice` option because it applies a monotonic transform to the logits; it changes only the reported probabilities. Tune it for calibration. The model can still choose the wrong answer, so use probability thresholds and escalate low-confidence cases.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" vizdoom
truetype-api
```

The first start downloads `google/gemma-4-12B` (~22GB) and needs about 24GB of RAM; an NVIDIA GPU or Apple Silicon helps a lot. Then:

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

The response contains typed answers, probabilities, and confidence, with one output token per question. Your code can read `answers.team.choice` and `answers.team.confidence`, apply a threshold, and continue without parsing prose or retrying malformed JSON.

Lower latency and local inference make small, frequent decisions practical to automate. If you build something with the replica or find a failure case, open an issue.

**[github.com/stephenlb/truetype.ai-open](https://github.com/stephenlb/truetype.ai-open)**
