"""Gemma 4 base-completion engine using one forward pass and one-token answers.

Design notes
------------
* The code loads the base model (``google/gemma-4-12B``). Base models predict the
  next token without a chat template or multi-token preamble. The letter after
  the prompt is the answer.
* The code enforces ``MAX_NEW_TOKENS = 1`` by running one ``forward`` call and
  reading the logits at the last position. It does not sample or decode text.
* The readout keeps the 26 uppercase letters A-Z from one token variant, which
  makes their logits comparable, then applies softmax at a low
  temperature over just the letters a given question declares legal.
"""

from __future__ import annotations

import logging
import threading
import time
from copy import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

from .letters import (
    LETTERS,
    LetterReadout,
    batch_letter_logits,
    batch_letter_mass,
    distribution_from_letter_logits,
    gather_letter_logits,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-4-12B"

# The answer is one token because the engine runs a single forward pass and reads
# logits at one position.
MAX_NEW_TOKENS = 1

# Softmax temperature for turning letter logits into a distribution. A low value
# sharpens the readout toward the model's preferred letter and keeps the
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
    # Length bucketing can reduce padding for large, heterogeneous CPU/CUDA
    # batches. On the tested MPS workload, its extra tokenization pass costs
    # more than the saved work, so leave it opt-in.
    bucket_by_length: bool = False
    trust_remote_code: bool = False
    letters_with_leading_space: bool = True
    # Full-vocabulary logsumexp is useful validation telemetry, but is an
    # expensive reduction for large vocabularies. Keep it on by default for the
    # existing API contract; high-throughput callers may opt out.
    report_letter_mass: bool = True
    # Reuse the KV cache of the static few-shot prefix across calls. The prefix
    # accounts for 70-85% of a prompt and is identical for the same question, so
    # caching it removes most of the prefill work (~2.9x faster per decision).
    prefix_cache: bool = True
    # Each cached prefix costs ~127MB of KV at 369 tokens. This MUST be >= the
    # number of distinct questions in a request or the LRU thrashes and every call
    # repeats the full prefill, erasing the speedup. The
    # service auto-grows this per request (see ensure_prefix_capacity), bounded by
    # prefix_cache_hard_cap, so callers do not have to get it right by hand.
    max_cached_prefixes: int = 16
    # Ceiling for auto-growth. Lower this on memory-constrained machines.
    prefix_cache_hard_cap: int = PREFIX_CACHE_HARD_CAP
    extra: dict = field(default_factory=dict)


@dataclass
class PrefixCacheEntry:
    """A reusable prefix KV cache with its own extension lock."""

    cache: object
    prefix_ids: object
    lock: threading.Lock = field(default_factory=threading.Lock)


def _fork_cache_for_extension(cache: object) -> object:
    """Copy cache metadata without duplicating immutable prefix KV tensors.

    Transformers' dynamic cache appends by assigning new key/value tensors to
    its layer objects.  A shallow copy of the cache and each layer therefore
    gives an extension its own mutable metadata while safely sharing the
    read-only prefix tensors.  This avoids copying roughly 127MB of MPS KV
    tensors before every cached request; the model still materializes the
    required prefix-plus-tail tensors during its normal attention work.
    """
    fork = copy(cache)
    fork.layers = [copy(layer) for layer in cache.layers]
    return fork


class GemmaLetterEngine:
    """Thread-safe wrapper that loads on first use and returns letter distributions."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._letter_token_ids: dict[str, int] = {}
        self._letter_token_id_tensor = None
        self._pad_token_id: int | None = None
        self.load_seconds: float | None = None
        # The LRU lock only protects map operations. Inference forks the cache's
        # lightweight metadata under an entry-local lock, allowing independent
        # prefixes to run without copying the static MPS KV tensors.
        self._prefix_caches: "OrderedDict[str, PrefixCacheEntry]" = OrderedDict()
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
            self._letter_token_id_tensor = torch.tensor(
                list(self._letter_token_ids.values()), dtype=torch.long, device=self._model.device
            )
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

        The 26-letter set must come from one variant. Falling back
        per-letter could mix ``"▁Q"`` with a bare ``"Q"``, and logits for tokens of
        different shapes are not comparable. All 26 must be single tokens in the
        same variant.
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

        # Bucketing is opt-in: on the measured MPS workload the extra
        # tokenization pass outweighed padding savings. Reassemble by index so
        # this remains an order-preserving public API.
        ordered = list(enumerate(prompts))
        if self.config.bucket_by_length:
            tokenized = self._tokenizer(list(prompts), add_special_tokens=True)
            lengths = [len(ids) for ids in tokenized.input_ids]
            ordered.sort(key=lambda item: lengths[item[0]])
        results: list[LetterReadout | None] = [None] * len(prompts)
        batch_size = max(1, self.config.max_batch_size)
        for start in range(0, len(ordered), batch_size):
            indexed_chunk = ordered[start : start + batch_size]
            chunk_results = self._score_chunk([prompt for _, prompt in indexed_chunk], k, temp)
            for (index, _), result in zip(indexed_chunk, chunk_results):
                results[index] = result
        return [result for result in results if result is not None]

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
        letter_values = gather_letter_logits(next_token_logits, self._letter_token_id_tensor)
        letter_logits = batch_letter_logits(letter_values)
        masses = (
            batch_letter_mass(next_token_logits, letter_values)
            if self.config.report_letter_mass
            else [None] * len(letter_logits)
        )
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

    def _letter_readout_values(self, next_token_logits):
        """Gather the A-Z slice once and optionally calculate mass from it."""
        letter_values = gather_letter_logits(next_token_logits, self._letter_token_id_tensor)
        logits = batch_letter_logits(letter_values)
        mass = (
            batch_letter_mass(next_token_logits, letter_values)
            if self.config.report_letter_mass
            else [None] * len(logits)
        )
        return logits, mass

    # ------------------------------------------------------------ prefix caching

    def ensure_prefix_capacity(self, distinct_prefixes: int) -> int:
        """Grow the prefix cache so a request with this many distinct prefixes fits.

        The LRU must hold every prefix a request will use, otherwise it evicts one
        before that prefix is revisited, and each call repeats a full prefill. This
        removes the caching benefit. Growth is one-way (never shrinks)
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
        for suffix, scored in zip(suffixes, self._letter_logits_cached_many(prefix, suffixes)):
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

    def _letter_logits_cached_many(
        self, prefix: str, suffixes: Sequence[str]
    ) -> list[tuple[dict[str, float], float | None] | None]:
        """Score compatible cached tails in batches grouped by token length.

        A cache miss seeds the prefix from one full prompt (preserving the
        tokenizer-boundary validation); remaining equal-length tails share one
        repeated KV cache and one forward pass.
        """
        import torch

        with self._cache_lock:
            entry = self._prefix_caches.get(prefix)

        if entry is None:
            first = self._letter_logits_cached(prefix, suffixes[0])
            if len(suffixes) == 1:
                return [first]
            rest = self._letter_logits_cached_many(prefix, suffixes[1:])
            return [first, *rest]

        device = self._model.device
        encoded = [
            self._tokenizer(prefix + suffix, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            for suffix in suffixes
        ]
        prefix_len = entry.prefix_ids.shape[1]
        grouped: dict[int, list[tuple[int, object]]] = {}
        results: list[tuple[dict[str, float], float | None] | None] = [None] * len(suffixes)
        for index, full_ids in enumerate(encoded):
            total = full_ids.shape[1]
            if total <= prefix_len or not torch.equal(full_ids[:, :prefix_len], entry.prefix_ids):
                continue
            grouped.setdefault(total, []).append((index, full_ids[:, prefix_len:]))

        with self._cache_lock:
            # LRU bookkeeping is deliberately short; inference below is guarded
            # only by this prefix's lock.
            # A different request may have evicted this entry after our lookup.
            # The local entry is still safe to score, but it is no longer an LRU
            # hit and must not be moved by key.
            if grouped and self._prefix_caches.get(prefix) is entry:
                self._prefix_caches.move_to_end(prefix)
                self.prefix_cache_hits += sum(map(len, grouped.values()))

        for total, group in grouped.items():
            tails = torch.cat([tail for _, tail in group], dim=0)
            with entry.lock:
                cache = _fork_cache_for_extension(entry.cache)
                cache.batch_repeat_interleave(len(group))
                with torch.inference_mode():
                    out = self._model(
                        input_ids=tails,
                        attention_mask=torch.ones((len(group), total), dtype=torch.long, device=device),
                        past_key_values=cache,
                        cache_position=torch.arange(prefix_len, total, device=device),
                        logits_to_keep=MAX_NEW_TOKENS,
                        use_cache=False,
                    )
            logits, masses = self._letter_readout_values(out.logits[:, -1, :].float())
            for (index, _), row, mass in zip(group, logits, masses):
                results[index] = row, mass
        return results

    def _letter_logits_cached(
        self, prefix: str, suffix: str
    ) -> tuple[dict[str, float], float] | None:
        """Letter logits for ``prefix + suffix``, reusing or populating the prefix cache.

        A miss costs one forward pass over the whole prompt, matching the uncached
        path. The prefix half of that pass's KV is kept for
        next time. A hit only prefills the tail.
        """
        import torch

        device = self._model.device
        full_ids = self._tokenizer(
            prefix + suffix, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(device)
        with self._cache_lock:
            total = full_ids.shape[1]

            entry = self._prefix_caches.get(prefix)
            if entry is not None:
                prefix_len = entry.prefix_ids.shape[1]
                # Only safe if this prompt really begins with the cached tokens.
                if total <= prefix_len or not torch.equal(full_ids[:, :prefix_len], entry.prefix_ids):
                    return None
                # The entry can be evicted after the lookup above. Its tensors
                # remain valid for this request, but avoid moving a missing (or
                # replacement) mapping entry.
                if self._prefix_caches.get(prefix) is entry:
                    self._prefix_caches.move_to_end(prefix)
                    self.prefix_cache_hits += 1

                tail = full_ids[:, prefix_len:]
            else:
                self.prefix_cache_misses += 1

        if entry is not None:
            # Do not mutate the reusable cache while serving a tail. This keeps
            # callers sharing this prefix safe while other prefixes proceed too.
            with entry.lock:
                cache = _fork_cache_for_extension(entry.cache)
                with torch.inference_mode():
                    out = self._model(
                        input_ids=tail,
                        attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
                        past_key_values=cache,
                        cache_position=torch.arange(prefix_len, total, device=device),
                        logits_to_keep=MAX_NEW_TOKENS,
                        use_cache=False,
                    )
                next_token_logits = out.logits[:, -1, :].float()
            logits, masses = self._letter_readout_values(next_token_logits)
            return logits[0], masses[0]

        # Miss: one full forward, then keep the prefix slice of its KV cache.
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
            with self._cache_lock:
                self._prefix_caches[prefix] = PrefixCacheEntry(cache, prefix_ids)
                while len(self._prefix_caches) > max(1, self.config.max_cached_prefixes):
                    self._prefix_caches.popitem(last=False)
                    self.prefix_cache_evictions += 1

        logits, masses = self._letter_readout_values(next_token_logits)
        return logits[0], masses[0]
