"""Measure the corpus. "It looks varied" is not a measurement.

Two failure modes these catch, both of which are invisible by eye at n=5 and fatal
at n=5000:

  mode collapse   — the model writes the same twenty sentences forever. `distinct_n`
                    falls; the corpus teaches a model one email.
  flat entities   — every shipper appears equally often. Real freight is heavy-
                    tailed; `zipf_slope` near 0 means your corpus is a world where
                    every customer is the same size.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Sequence

_WORD = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def distinct_n(texts: Sequence[str], n: int = 3) -> float:
    """Unique n-grams / total n-grams. 1.0 = never repeats, 0.0 = one sentence."""
    seen, total = set(), 0
    for t in texts:
        tok = tokens(t)
        for i in range(len(tok) - n + 1):
            seen.add(tuple(tok[i:i + n]))
            total += 1
    return round(len(seen) / total, 4) if total else 0.0


def type_token_ratio(texts: Sequence[str]) -> float:
    all_tok: List[str] = []
    for t in texts:
        all_tok.extend(tokens(t))
    return round(len(set(all_tok)) / len(all_tok), 4) if all_tok else 0.0


def entity_counts(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def zipf_slope(values: Iterable[str]) -> float:
    """Slope of log(frequency) against log(rank).

    Real entity distributions land near -1. Near 0 means uniform sampling — a
    corpus where every shipper is the same size, which no freight desk has seen.
    """
    counts = list(entity_counts(values).values())
    if len(counts) < 3:
        return 0.0
    xs = [math.log(i + 1) for i in range(len(counts))]
    ys = [math.log(c) for c in counts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom, 4)


def field_coverage(texts: Sequence[str], records: Sequence[Dict], fields: Sequence[str]) -> Dict[str, float]:
    """How often each record field is actually findable in its rendered text.

    Low coverage is not automatically a bug — real emails omit things, and defects
    remove things on purpose. It is a number you must be able to explain.
    """
    out: Dict[str, float] = {}
    for f in fields:
        hits = 0
        for text, rec in zip(texts, records):
            v = rec.get(f)
            if v is None:
                continue
            forms = [f"{int(v):,}", str(v)] if isinstance(v, (int, float)) else [str(v)]
            if any(form.lower() in text.lower() for form in forms):
                hits += 1
        out[f] = round(hits / len(texts), 4) if texts else 0.0
    return out


def summary(texts: Sequence[str], records: Sequence[Dict], entity_field: str) -> Dict[str, float]:
    return {
        "n": len(texts),
        "distinct_1": distinct_n(texts, 1),
        "distinct_3": distinct_n(texts, 3),
        "type_token_ratio": type_token_ratio(texts),
        "zipf_slope_" + entity_field: zipf_slope(r[entity_field] for r in records),
        "mean_chars": round(sum(len(t) for t in texts) / len(texts), 1) if texts else 0.0,
    }
