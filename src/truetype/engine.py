"""Gemma 4 base-completion engine: one forward pass, one token, letter logits.

Design notes
------------
* We load the *base* model (``google/gemma-4-12B``), not the instruct variant.
  Base models are pure next-token predictors: no chat template, no thinking
  channel, no multi-token preamble to fight. The letter that follows our prompt
  *is* the answer, and its logit is directly comparable across questions.
* ``MAX_NEW_TOKENS = 1`` is enforced *structurally*, not by a generation limit: we
  run a single ``forward`` and read the logits at the last position. Nothing is
  sampled and there is no decode loop, so the model cannot emit a second token
  even in principle.
* The readout is restricted to the 26 uppercase letters A-Z, all drawn from one
  token variant so their logits are comparable, then softmaxed at a low
  temperature over just the letters a given question declares legal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

from .letters import (
    LETTERS,
    LetterReadout,
    SoftmaxDistribution,
    batch_letter_logits,
    batch_letter_mass,
    distribution_from_letter_logits,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-4-12B"

# The answer is exactly one token. This is a property of how we call the model (a
# single forward, logits at one position) rather than a cap we ask it to respect.
MAX_NEW_TOKENS = 1

# Softmax temperature for turning letter logits into a distribution. Moderately
# low: it sharpens the readout toward the model's preferred letter and keeps the
# reported probabilities/scores close to discrete levels. It cannot change the
# argmax for noul/choice (a monotonic transform), but it does tighten `score`
# expected values and make `confidence` meaningful.
DEFAULT_TEMPERATURE = 0.7

# Upper bound on cached prefixes. At the measured ~127MB of KV per 369-token
# prefix, 64 entries is ~8GB on top of the ~24GB of weights. The count is
# configurable so smaller machines can lower it.
PREFIX_CACHE_HARD_CAP = 64


@dataclass
class EngineConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    top_k: int = 5
    temperature: float = DEFAULT_TEMPERATURE
    max_batch_size: int = 16
    trust_remote_code: bool = False
    letters_with_leading_space: bool = True
    # Reuse the KV cache of the static few-shot prefix across calls. The prefix is
    # 70-85% of a prompt and identical for every call on the same question, so
    # caching it removes most of the prefill work (~2.9x faster per decision).
    prefix_cache: bool = True
    # Each cached prefix costs ~127MB of KV at 369 tokens. This MUST be >= the
    # number of distinct questions in a request or the LRU thrashes and every call
    # re-pays the full prefill — which silently erases the entire speedup. The
    # service auto-grows this per request (see ensure_prefix_capacity), bounded by
    # prefix_cache_hard_cap, so callers do not have to get it right by hand.
    max_cached_prefixes: int = 16
    # Ceiling for auto-growth. Lower this on memory-constrained machines.
    prefix_cache_hard_cap: int = PREFIX_CACHE_HARD_CAP
    extra: dict = field(default_factory=dict)


class GemmaLetterEngine:
    """Lazily-loaded, thread-safe wrapper that turns prompts into letter distributions."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._letter_token_ids: dict[str, int] = {}
        self._pad_token_id: int | None = None
        self.load_seconds: float | None = None
        # prefix string -> (KV cache, prefix token ids)
        self._prefix_caches: "OrderedDict[str, tuple[object, object]]" = OrderedDict()
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_evictions = 0

    # ------------------------------------------------------------------ loading

    def _resolve_device(self) -> str:
        import torch

        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        """Load tokenizer + model. Idempotent and safe to call from any thread."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForMultimodalLM, AutoTokenizer

            started = time.time()
            device = self._resolve_device()

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
            )

            dtype = self.config.dtype
            if dtype == "auto":
                # fp32 on CPU (MPS fp32 is fine); bf16 where the hardware likes it.
                dtype = "float32" if device == "cpu" else "bfloat16"
            torch_dtype = getattr(torch, dtype)

            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.config.model_id,
                dtype=torch_dtype,
                device_map=device if device != "mps" else None,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            if device == "mps":
                self._model = self._model.to("mps")
            self._model.eval()

            self._letter_token_ids = self._build_letter_token_ids()
            self._pad_token_id = self._tokenizer.pad_token_id
            if self._pad_token_id is None:
                self._pad_token_id = self._tokenizer.eos_token_id
            self._tokenizer.padding_side = "left"

            self.load_seconds = time.time() - started
            logger.info(
                "loaded %s on %s in %.1fs (bf16/fp32=%s)",
                self.config.model_id,
                device,
                self.load_seconds,
                dtype,
            )

    def _build_letter_token_ids(self) -> dict[str, int]:
        """Map each A-Z to the single token id read at the answer position.

        The prompt ends at ``Answer:`` with no trailing space, so the natural
        continuation is the space-prefixed letter (``"▁Y"``), matching how the
        few-shot examples tokenize. We therefore prefer the ``" A"`` forms.

        The whole 26-letter set must come from *one* variant. Falling back
        per-letter could mix ``"▁Q"`` with a bare ``"Q"``, and logits for tokens of
        different shapes are not comparable — that would silently corrupt the
        argmax. So we require all 26 to be single tokens in the same variant and
        fail loudly otherwise.
        """
        assert self._tokenizer is not None

        variants: list[tuple[str, dict[str, int]]] = []
        candidate_forms = [("space-prefixed", " {}"), ("bare", "{}")]
        if not self.config.letters_with_leading_space:
            candidate_forms.reverse()

        for name, template in candidate_forms:
            mapping: dict[str, int] = {}
            for letter in LETTERS:
                ids = self._tokenizer.encode(template.format(letter), add_special_tokens=False)
                if len(ids) != 1:
                    mapping = {}
                    break
                mapping[letter] = ids[0]
            if len(mapping) == len(LETTERS):
                variants.append((name, mapping))

        if not variants:
            raise RuntimeError(
                f"no single-token A-Z variant found in {self.config.model_id}; "
                "the letter readout cannot be trusted"
            )

        name, mapping = variants[0]
        if len(set(mapping.values())) != len(LETTERS):
            raise RuntimeError(f"{name} A-Z token ids are not distinct in {self.config.model_id}")

        self.letter_variant = name
        logger.info("letter readout uses %s A-Z tokens", name)
        return mapping

    # ----------------------------------------------------------------- inference

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score_batch(
        self,
        prompts: Sequence[str],
        *,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> list[LetterReadout]:
        """One forward pass over the batch, one full letter readout per prompt."""
        self.load()
        if not prompts:
            return []

        k = self.config.top_k if top_k is None else top_k
        temp = self.config.temperature if temperature is None else temperature

        results: list[LetterReadout] = []
        batch_size = max(1, self.config.max_batch_size)
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            results.extend(self._score_chunk(chunk, k, temp))
        return results

    def _score_chunk(
        self, prompts: list[str], top_k: int, temperature: float
    ) -> list[LetterReadout]:
        import torch

        assert self._tokenizer is not None and self._model is not None

        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self._model(
                **encoded,
                logits_to_keep=MAX_NEW_TOKENS,
                use_cache=False,
            )

        next_token_logits = outputs.logits[:, -1, :].float()
        letter_logits = batch_letter_logits(next_token_logits, self._letter_token_ids)
        masses = batch_letter_mass(next_token_logits, self._letter_token_ids)
        return [
            LetterReadout(
                logits=row,
                top=distribution_from_letter_logits(row, top_k=top_k, temperature=temperature),
                letter_mass=mass,
            )
            for row, mass in zip(letter_logits, masses)
        ]

    def score_one(
        self, prompt: str, *, top_k: int | None = None, temperature: float | None = None
    ) -> LetterReadout:
        return self.score_batch([prompt], top_k=top_k, temperature=temperature)[0]

    # ------------------------------------------------------------ prefix caching

    def ensure_prefix_capacity(self, distinct_prefixes: int) -> int:
        """Grow the prefix cache so a request with this many distinct prefixes fits.

        The LRU must hold every prefix a request will use, otherwise it evicts one
        before that prefix is ever revisited and every call re-pays a full prefill —
        silently costing the entire caching win. Growth is one-way (never shrinks)
        and clamped to ``prefix_cache_hard_cap``; if the request exceeds the cap we
        warn, because the caller's latency will be worse than they probably expect.
        """
        target = max(1, int(distinct_prefixes))
        cap = max(1, int(self.config.prefix_cache_hard_cap))
        if target > cap:
            logger.warning(
                "request has %d distinct question prefixes, above the cache cap of %d; "
                "prefixes will thrash and warm-request latency will be no better than "
                "a cold run. Raise EngineConfig.prefix_cache_hard_cap (each entry is "
                "~127MB of KV) or split the request.",
                target,
                cap,
            )
        grown = min(max(target, self.config.max_cached_prefixes), cap)
        self.config.max_cached_prefixes = grown
        return grown

    def score_with_prefix(
        self,
        prefix: str,
        suffixes: Sequence[str],
        *,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> list[LetterReadout]:
        """Score ``prefix + suffix`` for each suffix, reusing the prefix's KV cache.

        The full prompt is always tokenized as one string and the cached prefix ids
        are checked against its head, so the tokens fed to the model are exactly the
        tokens the uncached path would use. If a prompt does not start with the
        cached prefix tokens (a BPE merge across the boundary, say), that prompt
        silently falls back to a full forward pass.
        """
        self.load()
        if not suffixes:
            return []

        k = self.config.top_k if top_k is None else top_k
        temp = self.config.temperature if temperature is None else temperature

        if not self.config.prefix_cache or not prefix:
            return self.score_batch(
                [prefix + s for s in suffixes], top_k=top_k, temperature=temperature
            )

        results: list[LetterReadout] = []
        for suffix in suffixes:
            scored = self._letter_logits_cached(prefix, suffix)
            if scored is None:
                results.append(self.score_batch([prefix + suffix], top_k=k, temperature=temp)[0])
            else:
                logits, mass = scored
                results.append(
                    LetterReadout(
                        logits=logits,
                        top=distribution_from_letter_logits(logits, top_k=k, temperature=temp),
                        letter_mass=mass,
                    )
                )
        return results

    def _letter_logits_cached(
        self, prefix: str, suffix: str
    ) -> tuple[dict[str, float], float] | None:
        """Letter logits for ``prefix + suffix``, reusing or populating the prefix cache.

        A miss costs exactly one forward pass over the whole prompt — the same work
        the uncached path does — and the prefix half of that pass's KV is kept for
        next time. A hit only prefills the tail.
        """
        import torch

        with self._cache_lock:
            device = self._model.device
            full_ids = self._tokenizer(
                prefix + suffix, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)
            total = full_ids.shape[1]

            entry = self._prefix_caches.get(prefix)
            if entry is not None:
                cache, prefix_ids = entry
                prefix_len = prefix_ids.shape[1]
                # Only safe if this prompt really begins with the cached tokens.
                if total <= prefix_len or not torch.equal(full_ids[:, :prefix_len], prefix_ids):
                    return None
                self._prefix_caches.move_to_end(prefix)
                self.prefix_cache_hits += 1

                tail = full_ids[:, prefix_len:]
                with torch.inference_mode():
                    out = self._model(
                        input_ids=tail,
                        attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
                        past_key_values=cache,
                        cache_position=torch.arange(prefix_len, total, device=device),
                        logits_to_keep=MAX_NEW_TOKENS,
                        use_cache=True,
                    )
                next_token_logits = out.logits[:, -1, :].float()
                # Drop the tail so the cache holds only the prefix again.
                cache.crop(-(total - prefix_len))
                return (
                    batch_letter_logits(next_token_logits, self._letter_token_ids)[0],
                    batch_letter_mass(next_token_logits, self._letter_token_ids)[0],
                )

            # Miss: one full forward, then keep the prefix slice of its KV cache.
            self.prefix_cache_misses += 1
            prefix_ids = self._tokenizer(
                prefix, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)
            prefix_len = prefix_ids.shape[1]

            with torch.inference_mode():
                out = self._model(
                    input_ids=full_ids,
                    attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
                    logits_to_keep=MAX_NEW_TOKENS,
                    use_cache=True,
                )
            next_token_logits = out.logits[:, -1, :].float()

            if total > prefix_len and torch.equal(full_ids[:, :prefix_len], prefix_ids):
                cache = out.past_key_values
                cache.crop(-(total - prefix_len))
                self._prefix_caches[prefix] = (cache, prefix_ids)
                while len(self._prefix_caches) > max(1, self.config.max_cached_prefixes):
                    self._prefix_caches.popitem(last=False)
                    self.prefix_cache_evictions += 1

            return (
                batch_letter_logits(next_token_logits, self._letter_token_ids)[0],
                batch_letter_mass(next_token_logits, self._letter_token_ids)[0],
            )