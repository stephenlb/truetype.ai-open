"""Extract single-letter logits and compute top-k softmax scores.

The engine reads one next-token position, keeps logits for the 26 uppercase
ASCII letters, and converts the top-k letter logits into a probability
distribution. It does not sample or decode text.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass

LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase)
LETTER_INDEX: dict[str, int] = {c: i for i, c in enumerate(LETTERS)}

# Rank at which we stop tracking for the reported top-k. k <= 26 by construction.
MAX_TOP_K = len(LETTERS)


class LetterLogitError(ValueError):
    """Raised when the model's next-token distribution cannot be read as letters."""


@dataclass(frozen=True)
class LetterReadout:
    """The full 26-letter logit readout for one prompt, plus a top-k view.

    ``logits`` keeps every letter when a question has more options than ``top_k``.
    ``top`` is the reported top-k softmax distribution.
    ``letter_mass`` is how much of the model's *full-vocabulary* probability landed
    on the 26 letters we scored. This checks whether the prompt puts the
    model in a "next token is a letter" state. If it approaches zero, the
    readout is measuring noise in the tail of the distribution, even though an
    argmax over it may still look plausible.
    """

    logits: dict[str, float]
    top: "SoftmaxDistribution"
    letter_mass: float | None = None

    @property
    def top_letter(self) -> str:
        return self.top.top_letter

    @property
    def top_probability(self) -> float:
        return self.top.top_probability

    @property
    def probabilities(self) -> dict[str, float]:
        return self.top.probabilities


@dataclass(frozen=True)
class SoftmaxDistribution:
    """A probability distribution over a chosen slice of letters."""

    ranked: tuple[tuple[str, float], ...]
    probabilities: dict[str, float]
    top_k: int
    temperature: float

    @property
    def top_letter(self) -> str:
        return self.ranked[0][0]

    @property
    def top_probability(self) -> float:
        return self.probabilities[self.top_letter]


def _softmax(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        raise LetterLogitError("softmax denominator is not finite")
    return [e / total for e in exps]


def distribution_from_letter_logits(
    letter_logits: dict[str, float],
    *,
    top_k: int = 5,
    temperature: float = 1.0,
) -> SoftmaxDistribution:
    """Softmax the top-k letter logits into a probability distribution.

    `top_k` is clamped to 2..26. The softmax is taken *only* over the k letters we
    keep, which is what TypeSafe-compatible consumers expect: the reported
    probabilities sum to 1 across the surfaced options.
    """
    unknown = set(letter_logits) - set(LETTER_INDEX)
    if unknown:
        raise LetterLogitError(f"non-letter logit keys: {sorted(unknown)}")
    if not letter_logits:
        raise LetterLogitError("no letter logits supplied")

    k = max(1, min(int(top_k), len(letter_logits), MAX_TOP_K))
    ranked = sorted(letter_logits.items(), key=lambda kv: kv[1], reverse=True)[:k]
    probs = _softmax([v for _, v in ranked], temperature)

    distribution = {letter: p for (letter, _), p in zip(ranked, probs)}
    # Keep the distribution in descending-probability order for stable output.
    ordered = dict(sorted(distribution.items(), key=lambda kv: kv[1], reverse=True))
    return SoftmaxDistribution(
        ranked=tuple((letter, logit) for letter, logit in ranked),
        probabilities=ordered,
        top_k=k,
        temperature=temperature,
    )


def batch_letter_logits(
    next_token_logits,  # torch.Tensor [batch, vocab]
    letter_token_ids: dict[str, int],
) -> list[dict[str, float]]:
    """Gather per-letter logits for every row of a batched next-token logits tensor."""
    if next_token_logits.dim() != 2:
        raise LetterLogitError("expected next-token logits of shape [batch, vocab]")

    ids = list(letter_token_ids.values())
    gathered = next_token_logits[:, ids]
    return [
        {letter: float(value) for letter, value in zip(letter_token_ids, row)}
        for row in gathered
    ]


def batch_letter_mass(
    next_token_logits,  # torch.Tensor [batch, vocab]
    letter_token_ids: dict[str, int],
) -> list[float]:
    """Fraction of full-vocabulary probability sitting on the A-Z tokens, per row.

    Uses log-sum-exp over the vocabulary to avoid overflow and replace a full
    softmax with one reduction.
    """
    import torch

    if next_token_logits.dim() != 2:
        raise LetterLogitError("expected next-token logits of shape [batch, vocab]")

    ids = list(letter_token_ids.values())
    total = torch.logsumexp(next_token_logits, dim=-1)
    letters = torch.logsumexp(next_token_logits[:, ids], dim=-1)
    return [float(value) for value in torch.exp(letters - total)]
