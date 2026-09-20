# Truetype.ai Jev Replica

A local latency-match replica of the Jev TypeSafe AI System One API that reads letter logits from
Gemma 4 12B. `POST /v1/systemone` matches the original request and response
formats for all three supported question types.

Each answer uses one token. The engine runs one `forward()` pass, reads the
next-token logits at the answer position, keeps the 26 A-Z token IDs, and
applies a temperature-0.7 softmax to the letters allowed by the question. At
load time, it assigns one token variant to each letter and checks that all 26
are distinct. The code in `src/` does not call `generate()`, sample tokens, or
run a decode loop. See `demo/README.md` for measurements and implementation
notes.

## Requirements

- Python 3.11 or newer
- About 24GB of RAM for the 12B weights; prefix caching needs more
- Optional: an NVIDIA GPU for CUDA, or an Apple Silicon Mac for MPS
- Optional: ViZDoom (`pip install vizdoom`) for the Doom demos

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first load downloads the model (`google/gemma-4-12B`, ~22GB) from Hugging
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
For example, 1.82 reflects probability split between levels 1 and 2. Read the
score with `confidence`. A `choice` answer is the argmax over the criteria
letters. A `noul` answer is the probability of "yes."

### Configuration

All variables are read at startup.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | bind address |
| `PORT` | `8000` | bind port |
| `TYPESAFE_REPLICA_API_KEY` | unset | if set, require `Authorization: Bearer <key>` |
| `TYPESAFE_REPLICA_MODEL_ID` | `google/gemma-4-12B` | Hugging Face model id |
| `TYPESAFE_REPLICA_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` |
| `TYPESAFE_REPLICA_DTYPE` | `auto` | `bfloat16`, `float16`, `float32`, or `auto` (float32 on CPU, bfloat16 elsewhere) |
| `TYPESAFE_REPLICA_TOP_K` | `5` | ranked letters reported per question |
| `TYPESAFE_REPLICA_PREFIX_CACHE` | `1` | cache the static few-shot prefix KV (~5x faster repeats) |
| `TYPESAFE_REPLICA_MAX_CACHED_PREFIXES` | `16` | LRU size; must cover distinct questions per request |
| `TYPESAFE_REPLICA_PREFIX_CACHE_HARD_CAP` | `64` | ceiling for per-request cache growth |
| `TYPESAFE_REPLICA_WARM_QUESTIONS` | unset | JSON object of question specs to prefill at startup |

Each cached prefix adds roughly 127MB of KV data on top of the weights. On
memory-constrained machines, lower
`TYPESAFE_REPLICA_PREFIX_CACHE_HARD_CAP` or set
`TYPESAFE_REPLICA_PREFIX_CACHE=0`.

Warming is off by default. It helps when requests reuse a prefix: a Doom run
uses one prefix about 40 times, while the text-adventure demo supplies new
criteria at each step.

## Tests

```bash
python -m pytest
```

The suite runs 50 functional cases against the live model: 20 `noul`, 20
`choice`, and 10 `score`. Two structural tests check the single-token letter
readout contract and prefix-cache sizing. The readout test requires one A-Z
token with nonzero probability mass among the scored letters.

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
```

Measured on an Apple Silicon Mac (bf16), from `demo/README.md`:

| Demo | Task | Decision type | Warm latency |
|---|---|---|---|
| `doom_demo.py` | real Doom, 3 arena scenarios | choice (3-4 actions) | ~133ms (7.2 decisions/sec) |
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

The full-game demo plays a whole level, with the model making both navigation
and combat decisions:

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

The demo does not run Doom in real time. At about 7 decisions/sec against an
engine running 35 tics/sec, 40 decisions at 4 tics each cover roughly 4.6 seconds
of game time.

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
If the port is published beyond localhost, set `TYPESAFE_REPLICA_API_KEY` as well.

## Layout

```
src/truetype/       the replica library and HTTP API
  engine.py         model load, letter readout, prefix KV cache
  letters.py        A-Z logit extraction and softmax
  questions.py      noul / choice / score primitives
  render.py         prompt construction (prefix and state tail)
  service.py        request orchestration, sequential cached scoring
  api.py            FastAPI app and entry point
demo/               five demos (see demo/README.md for measurements)
doom/               full-level Doom demo
tests/              the 52-test verification suite
```

`demo/README.md` contains the measurements behind the cache design and Doom
prompt changes. It covers an undersized cache that erased the speedup and a
prompt bug that left the scored letters with 0.0000 probability mass while all
50 accuracy tests still passed.
