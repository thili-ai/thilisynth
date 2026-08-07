"""Check a rendered document against the record it was supposed to render.

This module exists because of a measurement. Handed a record with
`rate_offered_usd: 1850`, qwen2.5:1.5b wrote **$18,500** into the email — a 10x
error on the one field that decides money — and also turned "53ft reefer" into
"53ft Rer efer".

We only caught it because the record was sampled *before* the text was written. If
the model had generated the ground truth too, that row would carry a silently wrong
label, and every model trained on the corpus would inherit the error with no way to
detect it.

So: **every render is diffed against its source record, and contradictions are
rejected.** A generator without this loop produces label noise it cannot see.

Two different checks, and the second is the one that matters:

    missing       — the value never appears. Often fine: real emails omit things,
                    and a defect may have removed it on purpose.
    contradicted  — a value of the right *kind* appears with the wrong content.
                    A money token that is not the rate. This is always a defect
                    in the render, never realism.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)")
_WEIGHT = re.compile(r"([0-9][0-9,]*)\s?(?:lb|lbs|pounds|kg|kgs)\b", re.I)


def money_forms(value: float) -> List[str]:
    """Every way a dollar amount is plausibly written."""
    i = int(value)
    return [f"{i}", f"{i:,}", f"${i}", f"${i:,}", f"{i}.00", f"${i:,}.00"]


def number_forms(value: float) -> List[str]:
    i = int(value)
    return [f"{i}", f"{i:,}"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


@dataclass
class Violation:
    field: str
    kind: str            # "missing" | "contradicted"
    expected: Any
    found: Optional[str] = None

    def __str__(self) -> str:
        if self.kind == "contradicted":
            return f"{self.field}: expected {self.expected!r}, text says {self.found!r}"
        return f"{self.field}: {self.expected!r} does not appear"


@dataclass
class VerifyReport:
    ok: bool
    violations: List[Violation] = field(default_factory=list)

    @property
    def contradictions(self) -> List[Violation]:
        return [v for v in self.violations if v.kind == "contradicted"]

    @property
    def missing(self) -> List[Violation]:
        return [v for v in self.violations if v.kind == "missing"]

    def __str__(self) -> str:
        if self.ok:
            return "ok"
        return "; ".join(str(v) for v in self.violations)


def verify(
    text: str,
    record: Dict[str, Any],
    must_appear: Sequence[str] = (),
    money_fields: Sequence[str] = (),
    weight_fields: Sequence[str] = (),
    may_be_absent: Sequence[str] = (),
) -> VerifyReport:
    """Diff a rendered document against its source record.

    Args:
        must_appear:   fields whose value must be findable in the text verbatim
                       (identifiers, place names, equipment).
        money_fields:  fields checked against every `$N` token in the text. Any
                       money token that matches no money field is a contradiction.
        weight_fields: same idea for `N lb` / `N kg` tokens.
        may_be_absent: fields exempt from the "missing" check — because a defect
                       is expected to have removed them.
    """
    hay = _norm(text)
    violations: List[Violation] = []
    exempt = set(may_be_absent)

    for name in must_appear:
        if name in exempt:
            continue
        value = record.get(name)
        if value is None:
            continue
        forms = number_forms(value) if isinstance(value, (int, float)) else [str(value)]
        if not any(_norm(f) in hay for f in forms):
            violations.append(Violation(name, "missing", value))

    # Money: collect the legal amounts, then look for any amount that is not one.
    legal_money = {int(record[f]) for f in money_fields if record.get(f) is not None}
    if legal_money:
        for raw in _MONEY.findall(text):
            try:
                found = int(float(raw.replace(",", "")))
            except ValueError:
                continue
            if found not in legal_money:
                violations.append(
                    Violation(",".join(money_fields), "contradicted", sorted(legal_money), f"${raw}")
                )

    legal_weight = {int(record[f]) for f in weight_fields if record.get(f) is not None}
    if legal_weight:
        for raw in _WEIGHT.findall(text):
            try:
                found = int(float(raw.replace(",", "")))
            except ValueError:
                continue
            if found not in legal_weight:
                violations.append(
                    Violation(",".join(weight_fields), "contradicted", sorted(legal_weight), raw)
                )

    return VerifyReport(ok=not violations, violations=violations)
