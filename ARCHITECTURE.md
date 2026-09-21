# Architecture

A plain-text summary of the diagram in `architecture/index.html`, focused on how
input becomes a prompt and how the model's output becomes a typed answer.
The inference core is `src/truetype/engine.py`.

## The one-sentence version

A request arrives as state plus a set of typed questions; each question is
rendered into a base-completion prompt that ends in `Answer:`; the engine runs
**one forward pass** and reads the next-token logits for the 26 letters `A-Z`;
those letters are softmaxed over the letters the question declared legal and
returned as a typed answer. No sampling, no decode loop, no chat template.

```
state + questions -> prompts -> 1 forward pass -> logits[A..Z] -> typed answers
```

## Input processing

### 1. Request (`api.py`)

`POST /v1/systemone` takes:

- `state` — a string, dict, or list describing what is being judged.
- `questions` — a map of question id to `{type, instructions, criteria}`.
- `model`, `temperature`, `include_debug` — optional knobs.

Pydantic rejects a non-positive `temperature` and an empty `questions` map at
parse time (422). A request that lands before weights finish
loading is 503. The process holds exactly one engine, so the model loads
once per process.

### 2. Orchestration (`service.py`)

Non-string `state` is serialized with `json.dumps(indent=2)`. Each question spec
becomes a typed `Question` object of one of three kinds:

| type | shape | letters used |
| --- | --- | --- |
| `noul` | yes/no | 2 |
| `choice` | named options | up to 26 |
| `score` | ordered levels | up to 26 |

### 3. Prompt rendering (`render.py`, `questions.py`)

Every prompt is plain base-model completion text built from repeated blocks:

```
Choices: ...
Question: ...
Text: ...
Answer: X
```

Two properties of this layout matter:

- **The state goes last.** All answer-relevant scaffolding (choices, question
  text, few-shot examples) sits above `Text:`, so for a fixed question the head
  of the prompt is byte-identical across calls. That head is the cacheable
  *prefix*; the per-call state is the *tail*.
- **No trailing space after `Answer:`.** The natural continuation is then the
  space-prefixed letter token (`"▁Y"`), matching how the examples tokenize.
  Adding a space strands it and collapses the probability mass on scored
  letters to ~0.

Few-shot examples teach the letter slot rather than the task: `noul` uses a
fixed bank of 6 demonstrations with unrelated predicates, while `choice` and
`score` synthesize one roundtrip per option from the caller's criteria. Options
are visited low, high, then ascending, so no positional A/B/C shortcut exists.

### 4. Tokenization and the forward pass (`engine.py`)

`load()` (`engine.py:110`) is idempotent and double-checked under a lock. Device
resolution is cuda → mps → cpu; dtype is `float32` on cpu and `bfloat16`
elsewhere. Padding is left-side so the last position is always the real end of
the prompt.

At load time `_build_letter_token_ids()` (`engine.py:160`) maps `A-Z` to token
ids, preferring the space-prefixed `" A"` form and falling back to bare `"A"`.
All 26 must be single tokens **from the same variant** and mutually distinct —
mixing `"▁Q"` with a bare `"Q"` would make the logits incomparable — otherwise
the engine refuses to run.

Two scoring paths:

- `score_batch()` (`engine.py:210`) — chunks prompts at `max_batch_size=16`,
  tokenizes with padding, and runs one `forward` per chunk with
  `logits_to_keep=1` and `use_cache=False`, then takes `logits[:, -1, :]`.
- `score_with_prefix()` (`engine.py:297`) — reuses the KV cache of the static
  prefix. On a **miss**, one full forward runs (identical cost to the uncached
  path) and the prefix slice of its KV is kept via `cache.crop(-tail_len)`. On a
  **hit**, only the tail is prefilled, passing `past_key_values` and
  `cache_position=arange(prefix_len, total)`, and the entry is cropped back to
  prefix-only afterward.

The full prompt is always tokenized as one string and the cached prefix ids are
compared against its head. If they disagree (a tokenizer merge across the
boundary, say), the prompt silently falls back to a full pass — correctness
never depends on the cache. The cache is an `OrderedDict` LRU under its own
lock, with `ensure_prefix_capacity()` (`engine.py:273`) growing it one-way to
cover every distinct prefix in a request, clamped to a hard cap.

## Output processing

The model emits `logits[batch, vocab]` for a single position. Turning that into
an answer is three steps.

### 1. Gather the letters (`letters.py`)

`batch_letter_logits()` indexes the 26 `A-Z` token ids out of the vocab row into
one logit dict per prompt. This is the raw vote, before any renormalization.

### 2. Sanity-check the mass

`batch_letter_mass()` reports what fraction of the *full-vocabulary*
probability sits on `A-Z`, computed as
`exp(logsumexp(letters) − logsumexp(vocab))`. A healthy prompt puts nearly all
its mass there; a value near zero means the argmax is noise from the
distribution's tail and the prompt is malformed.

### 3. Renormalize over legal letters

Scoring receives the **full 26-letter dict**, never a truncated top-k, so a
question with more options than `top_k` is not cut off. Softmax runs at
`temperature=0.7` over only the letters that question declared legal, so the
reported probabilities sum to 1 across real options. Temperature cannot change
the argmax — it is monotonic — it only sharpens `score` expected values and
makes `confidence` meaningful.

Then each type reads the distribution its own way:

| type | returned value | confidence |
| --- | --- | --- |
| `noul` | P(yes) over the two legal letters | — |
| `choice` | argmax label + full probability map | `1 − H / log n` |
| `score` | expected level `Σ i·pᵢ` over ordered criteria | `1 − H / log n` |

A `LetterLogitError` is raised on non-letter keys, an empty readout, or a
non-finite softmax denominator.

### 4. Response envelope

```
{ model, answers, usage }
```

`answers` is keyed by question id. `usage` is approximate by construction:
`input_tokens = chars // 4` across rendered prompts, and `output_tokens` equals
the number of questions, since each answer is exactly one token. With
`include_debug`, the response also carries the exact prompt text, the raw letter
logits, and the ranked top-k letters per question.

## Deliberately absent

- `generate()` — no HF generation loop is called anywhere in `src/`.
- Sampling — the distribution is reported, not drawn from.
- A decode loop — `MAX_NEW_TOKENS = 1` is a property of the call, not a cap.
- A chat template — the base checkpoint predicts the letter directly after the
  prompt, so nothing stands between the prompt and the first predicted token.
