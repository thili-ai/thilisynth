"""Seeded sampling — the layer that produces the ground truth.

Everything here is driven by an explicit `random.Random(seed)`, because the corpus
has to be reproducible from its seed alone. A grader who cannot regenerate your
dataset cannot check your results.

This module is deliberately dumb: weighted picks, power-law ranks, bounded numbers.
The realism comes from the *reference tables* a pack feeds it, not from cleverness
here.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


class Sampler:
    """A seeded source of records. One Sampler per corpus run."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    # --- primitives -------------------------------------------------------

    def pick(self, items: Sequence[T]) -> T:
        return self.rng.choice(list(items))

    def weighted(self, items: Sequence[T], weights: Sequence[float]) -> T:
        return self.rng.choices(list(items), weights=list(weights), k=1)[0]

    def zipf(self, items: Sequence[T], exponent: float = 1.1) -> T:
        """Pick with a power-law bias toward the front of the list.

        Real freight lanes, carriers and commodities are all heavy-tailed: a few
        appear constantly and most appear rarely. Sampling uniformly produces a
        corpus that looks nothing like the business, and any model trained on it
        learns a world where every lane is equally likely.
        """
        n = len(items)
        weights = [1.0 / ((i + 1) ** exponent) for i in range(n)]
        return self.weighted(items, weights)

    def integer(self, low: int, high: int, step: int = 1) -> int:
        return self.rng.randrange(low, high + 1, step)

    def money(self, low: int, high: int, round_to: int = 25) -> int:
        raw = self.rng.randint(low, high)
        return max(round_to, int(round(raw / round_to)) * round_to)

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def subset(self, items: Sequence[T], k: int) -> List[T]:
        return self.rng.sample(list(items), min(k, len(items)))

    # --- reporting --------------------------------------------------------

    def distribution(self, draws: List[T]) -> Dict[str, int]:
        """Count what actually came out — for the corpus card, not for the model."""
        counts: Dict[str, int] = {}
        for d in draws:
            counts[str(d)] = counts.get(str(d), 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def split_by(records: Sequence[dict], key: str, holdout: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Split a corpus by an entity, not at random.

    Splitting rows randomly leaks a shipper's house formatting across train and
    test, and formatting is most of what an extractor learns — so a random split
    reports a score the model will never repeat on a new customer.
    """
    rng = random.Random(seed)
    values = sorted({str(r[key]) for r in records})
    rng.shuffle(values)
    n_held = max(1, int(len(values) * holdout))
    held = set(values[:n_held])
    test = [r for r in records if str(r[key]) in held]
    train = [r for r in records if str(r[key]) not in held]
    return train, test
