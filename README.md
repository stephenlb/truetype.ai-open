# Truetype.ai Jev Replica

A local replica of the Jev TypeSafe AI System One API, tuned to match its latency
while reading letter logits from Gemma 4 12B. `POST /v1/systemone` accepts and
returns the original formats for all three supported question types.

![Jev Replica System One Model](media/jev-replica-system-one-model.jpg)

Each answer uses one token. The engine runs one `forward()` pass, reads
next-token logits at the answer position, keeps the 26 A-Z token IDs, and
applies a temperature-0.7 softmax to the letters allowed by the question. At
load time, it assigns one token variant to each letter and checks that all 26
are distinct. The code in `src/` does not call `generate()`, sample tokens, or
run a decode loop. See `demo/README.md` for measurements and implementation
notes.

https://github.com/user-attachments/assets/c24ad3fd-044c-46b9-8862-46b70dd8e201

https://github.com/user-attachments/assets/50a48f64-c483-4b1d-8956-7fb430837a60

## Why we rebuilt Jev

We wanted a fast, self-hosted decision model for
[Blocks.ai](https://blocks.ai). Blocks.ai agents interact with external systems
from wherever they run, including a laptop, but general-purpose LLM reasoning
adds latency and gives agents more room to make mistakes. A local Jev-style
system gives them faster, more reliable decisions without sending every request
to a hosted model.

## Requirements

- Python 3.11 or newer
- About 24GB of RAM for the 12B weights; prefix caching needs more
- Optional: an NVIDIA GPU for CUDA, or an Apple Silicon Mac for MPS
- Optional: ViZDoom (`pip install vizdoom`) for the Doom demos
- Optional: pygame (`pip install pygame`) to watch the Flappy Bird demo in a window

<img width="1536" height="1024" alt="truetype-jev-replicat-architecture" src="https://github.com/user-attachments/assets/3d2b9dfb-beca-4e1d-aba0-98f4a517e7e8" />

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first load downloads `google/gemma-4-12B` (about 22GB) from Hugging
Face and caches it in `~/.cache/huggingface`. Set `HF_TOKEN` if the repository
requires authentication.

For the test suite and Doom demos:

```bash
pip install -e ".[test]" vizdoom
```

## Run the API

```bash
truetype-api                 # or: python -m truetype.api
```

The server listens on `127.0.0.1:8000` by default. While the model loads,
`/health` reports `status: starting`.

```bash
curl -s localhost:8000/health
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

Response:

```json
{
  "model": "gemma-4-12b",
  "answers": {
    "refund": {"type": "noul", "noul": 0.9872},
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.981, "returns": 0.014, "shipping": 0.005},
      "confidence": 0.842
    },
    "severity": {
      "type": "score",
      "score": 1.82,
      "legend": {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists"
      },
      "probabilities": {"0": 0.045, "1": 0.09, "2": 0.865},
      "confidence": 0.61
    }
  },
  "usage": {"input_tokens": 431, "output_tokens": 3}
}
```

A `score` answer is the expected value `sum(i * p_i)` across the ordinal levels.
For example, 1.82 reflects a probability split between levels 1 and 2. Use
`confidence` to distinguish a concentrated result from a split distribution. A
`choice` answer is the argmax over the criteria letters. A `noul` answer is the
probability of "yes."

## Tests

```bash
python -m pytest
```

The suite has 102 tests: 50 fast deterministic contracts plus 50 functional
cases against the live model (20 `noul`, 20 `choice`, and 10 `score`) and two
live structural checks. The readout check requires the A-Z tokens to carry
meaningful probability mass, while the cache check verifies capacity sizing.

Run the suite from the repository root. The first run loads the 12B model and
takes about a minute. Subsequent tests reuse the session model and take seconds.

## Demos

Each demo timestamps its decisions and prints latency statistics. The model
takes about five seconds to load before startup warming. Run these commands from
the repository root:

```bash
python demo/batch_demo.py       # 5 questions over 1 state in one request
python demo/game_demo.py        # 4-room text adventure
python demo/web_nav_demo.py     # support-portal refund flow
python demo/latency_bench.py    # prefix cache on/off, warm budget assertions
python flappy/flappy_bird_demo.py --ascii   # Flappy Bird against a running game clock
```

Measured on an Apple Silicon Mac (bf16), from `demo/README.md`:

| Demo | Task | Decision type | Warm latency |
|---|---|---|---|
| `doom_demo.py` | real Doom, 3 arena scenarios | choice (3-4 actions) | ~133ms (7.2 decisions/sec) |
| `flappy_bird_demo.py` | Flappy Bird, game clock running | choice (flap / coast) | ~112ms (8.9 decisions/sec) |
| `game_demo.py` | text adventure | choice | ~500ms (new prefix every step) |
| `web_nav_demo.py` | page navigation | choice | ~680ms (new prefix every step) |
| `batch_demo.py` | 5 questions, 1 request | noul + choice + score | ~815ms warm, ~4.3s cold |

`latency_bench.py` requires every measured question to stay under a 150ms p95
warm budget. The script fails if a prompt change pushes latency over that limit.

## Doom demos

Both Doom demos use [ViZDoom](https://vizdoom.farama.org/), a ZDoom fork for AI
agents that includes the Freedoom IWADs. ViZDoom supplies object labels, health,
ammo, and kill count. The model reads a short text summary of the game state
instead of pixels. The demos do not need screen capture or macOS Screen
Recording permission.

```bash
pip install vizdoom
python demo/doom_demo.py                    # defend_the_center (default)
python demo/doom_demo.py health_gathering   # walk onto medkits to survive
python demo/doom_demo.py deadly_corridor    # fight down a corridor
python demo/doom_demo.py --watch            # ...and watch it play
```

Each scenario runs for 40 decisions with three or four available buttons. The
demo warms the question's KV prefix before the loop. Warm results include the
first measured decision.

| Scenario | Outcome | Latency (p50) | Throughput |
|---|---|---|---|
| `defend_the_center` | 5 kills, health 100 | 133ms | 7.2 dec/sec |
| `health_gathering` | health held at 100, reward 160 | 132ms | 7.5 dec/sec |
| `deadly_corridor` | 2-3 kills, health 70 | 134ms | 7.2 dec/sec |

`--watch` opens ViZDoom's 640x480 SDL window with the HUD and crosshair enabled,
then draws every skipped tic. Rendering happens inside `make_action`, outside
the timed decision. Per-decision latency stays the same, while throughput drops
to about 5.5 decisions/sec. Headless mode is the default.

In the full-game demo, the model handles navigation and combat for an entire
level:

[Watch the full-game Doom demo](media/typesafe-replica-doom-game-only.mp4)

```bash
python doom/doom_full_game_demo.py                  # Freedoom 2 MAP01, 150 decisions
python doom/doom_full_game_demo.py --watch          # ...and watch it play
python doom/doom_full_game_demo.py --decisions 300
python doom/doom_full_game_demo.py --map E1M1       # the cramped opening level
python doom/doom_full_game_demo.py --iwad doom2.wad --map MAP01 --skill 2
python doom/doom_full_game_demo.py --self-test      # check the questions, no engine
```

Engine data selects a phase (`combat`, `stuck`, or `navigate`), then the model
answers one question for that phase. BFS finds the next waypoint on a
walkability grid built from the level's blocking lines, and the model chooses
the local direction. A 300-decision MAP01 run reaches 5 kills and 8 items.
`--self-test` sends 22 synthetic states through the same prompt builders and
fails if a wording change causes a regression.

The demo runs slower than Doom's clock. At about 7 decisions/sec against an
engine running 35 tics/sec, 40 decisions at 4 tics each cover roughly 4.6 seconds
of game time.

## Flappy Bird

ViZDoom waits for the model: the engine advances only when the demo calls
`make_action`, so a slow decision costs wall-clock time but never game time. The
Flappy Bird demo removes that concession. The simulation runs at a fixed 30 fps
and takes one decision every 4 frames, so each decision is worth 133ms of game
time whether or not it has arrived, and the bird falls at the rate the model
thinks.

[Watch the Flappy Bird demo](media/jev-system-one-replica-flappy-bird.mp4)

```bash
python flappy/flappy_bird_demo.py                  # 150 decisions, 1 episode
python flappy/flappy_bird_demo.py --ascii          # ...and watch it in the terminal
python flappy/flappy_bird_demo.py --watch          # ...in a pygame window
python flappy/flappy_bird_demo.py --decisions 300 --episodes 3
python flappy/flappy_bird_demo.py --baseline       # reference player, no model
python flappy/flappy_bird_demo.py --self-test      # check the questions, no game
```

`--watch` draws the decision alongside the game: a dashed line through the middle
of the next opening is the line the current question is judged against, and its
colour, the label on the ground strip and the bar beside it are the phase, the
chosen action and the model's confidence in it.

Three 250-decision episodes finished without a crash, deciding in 110ms against
the 133ms budget:

```
pipes=48  crashes=0  decisions=750
real-time ratio: 1.19x (100.0s of game in 84.2s of decisions)
  approach: n=528  mean=110.4ms  p50=115.7ms  p95=121.0ms
  line_up:  n=222  mean=116.8ms  p50=117.7ms  p95=122.3ms
```

Crashes are the metric, not pipes. A surviving bird passes one pipe every 15
decisions whatever it does well, so the pipe count only measures how long the run
was; 16 per episode is what 250 decisions buys.

The demo projects the bird's coasting path through the whole pipe crossing. It
then asks whether the bird will end up too low, lined up, or too high.
`--self-test` runs 14 synthetic states through the same prompt builders.
`--baseline` follows the rule described by the criteria and clears 39 pipes in
600 decisions on every seed. This separates model errors from problems in the
game tuning.

`--watch` needs `pip install pygame`; without it, the demo falls back to
`--ascii`. The module docstring records two prompt findings. An A-position bias
caused every `coast` case to fail until the no-op action was listed first, and
the model initially read the offsets instead of the verdict.

## Docker

The image installs the package, API dependencies, and PyTorch, but excludes
ViZDoom, the demos, and the test suite. `TORCH_INDEX_URL` defaults to the CPU
wheel index. Set it to a CUDA index for GPU builds.

```bash
docker build -t truetype-jev-replica .
```

CPU, model on a mounted Hugging Face cache:

```bash
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/home/app/.cache/huggingface \
  truetype-jev-replica
```

NVIDIA GPU (needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/app/.cache/huggingface \
  -e TYPESAFE_REPLICA_DEVICE=cuda \
  truetype-jev-replica
```

The image binds `0.0.0.0` inside the container. The 12B model needs about 24GB of
RAM in CPU mode. Mounting the Hugging Face cache keeps the 22GB model download
out of the image.
