"""Gemma 4 base-completion engine: one forward pass, one token, letter logits.

Design notes
------------
* We load the *base* model (``google/gemma-4-12B``), not the instruct variant.
  Base models are pure next-token predictors: no chat template, no thinking
  channel, no multi-token preamble to fight. The letter that follows our prompt
  *is* the answer, and its logit is directly comparable across questions.
* ``max_new_tokens=1`` is enforced structurally, not by generation limits: we run
  a single ``forward`` and read the last position of the logits. Nothing is
  sampled, so there is no way for the model to run away and emit prose.
* All 26 uppercase letters (and their space-prefixed variants) are single tokens
  in this vocabulary, so the letter distribution is a clean 26-way readout.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

from .letters import (
    LETTERS,
    LetterReadout,
    SoftmaxDistribution,
    batch_letter_logits,
    distribution_from_letter_logits,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-4-12B"


@dataclass
class EngineConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    top_k: int = 5
    temperature: float = 1.0
    max_batch_size: int = 16
    trust_remote_code: bool = False
    letters_with_leading_space: bool = True
    extra: dict = field(default_factory=dict)


class GemmaLetterEngine:
    """Lazily-loaded, thread-safe wrapper that turns prompts into letter distributions."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._letter_token_ids: dict[str, int] = {}
        self._pad_token_id: int | None = None
        self.load_seconds: float | None = None

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
        """Map each A-Z to the single token id used for the answer position.

        Base models are trained on natural text where an answer letter follows a
        space, so ``" A"`` is the canonical continuation. We verify single-token
        status at load time and fail loudly rather than silently producing apples
        -to-oranges logits.
        """
        assert self._tokenizer is not None
        mapping: dict[str, int] = {}
        for letter in LETTERS:
            candidates = [f" {letter}", letter] if self.config.letters_with_leading_space else [letter]
            token_id = None
            for candidate in candidates:
                ids = self._tokenizer.encode(candidate, add_special_tokens=False)
                if len(ids) == 1:
                    token_id = ids[0]
                    break
            if token_id is None:
                raise RuntimeError(
                    f"letter {letter!r} is not a single token in {self.config.model_id}"
                )
            mapping[letter] = token_id
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
                logits_to_keep=1,
                use_cache=False,
            )

        next_token_logits = outputs.logits[:, -1, :].float()
        letter_logits = batch_letter_logits(next_token_logits, self._letter_token_ids)
        return [
            LetterReadout(
                logits=row,
                top=distribution_from_letter_logits(row, top_k=top_k, temperature=temperature),
            )
            for row in letter_logits
        ]

    def score_one(
        self, prompt: str, *, top_k: int | None = None, temperature: float | None = None
    ) -> LetterReadout:
        return self.score_batch([prompt], top_k=top_k, temperature=temperature)[0]