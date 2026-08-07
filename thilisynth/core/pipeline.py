"""The generation pipeline, and the order of its steps is the whole argument.

    sample  ->  render  ->  VERIFY  ->  damage  ->  manifest
      |           |           |           |
      |           |           |           `-- defects applied in code, so the flag
      |           |           |               is a label and not the model's claim
      |           |           `-- reject renders that contradict the record, BEFORE
      |           |               any deliberate damage. Otherwise you cannot tell
      |           |               an injected defect from a hallucination.
      |           `-- the model writes only the surface
      `-- the record is the ground truth, and it already exists

Verification runs on the *clean* render. Damage after. Swap those two and the
corpus can no longer distinguish "we planted this" from "the model made this up",
which is the only distinction that makes the labels worth anything.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .defects import apply_defects, observed_rates
from .sample import Sampler
from .verify import VerifyReport, verify


@dataclass
class Row:
    record: Dict[str, Any]
    text: str
    defects: List[str] = field(default_factory=list)
    render_attempts: int = 1


@dataclass
class Corpus:
    name: str
    seed: int
    renderer: str
    rows: List[Row] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def defect_rates(self) -> Dict[str, float]:
        return observed_rates([r.defects for r in self.rows])

    @property
    def reject_rate(self) -> float:
        total = len(self.rows) + len(self.rejected)
        return round(len(self.rejected) / total, 4) if total else 0.0

    def to_jsonl(self) -> List[Dict[str, Any]]:
        return [
            {"record": r.record, "text": r.text, "defects": r.defects,
             "render_attempts": r.render_attempts}
            for r in self.rows
        ]


def generate(
    name: str,
    n: int,
    sample_fn: Callable[[Sampler, int], Any],
    render_fn: Callable[[Dict[str, Any]], str],
    to_dict: Callable[[Any], Dict[str, Any]],
    verify_kwargs: Dict[str, Any],
    defect_rates: Optional[Dict[str, float]] = None,
    seed: int = 0,
    renderer_name: str = "unknown",
    max_render_attempts: int = 3,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Corpus:
    """Build a corpus. Reproducible from `seed` alone."""
    sampler = Sampler(seed)
    rng = random.Random(seed ^ 0x5EED)
    corpus = Corpus(name=name, seed=seed, renderer=renderer_name)

    for i in range(n):
        record = to_dict(sample_fn(sampler, i))

        text: Optional[str] = None
        report: Optional[VerifyReport] = None
        attempts = 0
        for attempts in range(1, max_render_attempts + 1):
            candidate = render_fn(record)
            report = verify(candidate, record, **verify_kwargs)
            if report.ok or not report.contradictions:
                text = candidate
                break

        if text is None:
            # Every attempt contradicted the record. Drop the row and say so —
            # a corpus that silently keeps these is a corpus with wrong labels.
            corpus.rejected.append({
                "record": record,
                "reason": str(report) if report else "no render",
                "attempts": attempts,
            })
            continue

        damaged = apply_defects(text, record, rng, defect_rates or {})
        corpus.rows.append(Row(record=record, text=damaged.text,
                               defects=damaged.flags, render_attempts=attempts))

        if on_progress:
            on_progress(i + 1, n)

    return corpus
