"""Plant the failure modes your downstream model has to survive — in code.

You cannot test an agent on failures your corpus does not contain. So defects are
injected deliberately, at a known rate, with a label.

**Why this is a post-process and not a prompt.** The obvious move is to ask the
model to write a thread with a vague pickup window. Measured: told "never state the
pickup date or time explicitly", qwen2.5:7b put explicit `Sent:` dates in every
header and had a reply propose "Monday or Tuesday this week".

Worse than unreliable, it is unverifiable. If the model plants the defect, the
defect flag is the model's *claim about what it did* — not your label. Then you are
back to trusting the generator, which is the whole thing this library refuses to do.

So: render clean, verify against the record, then damage the text mechanically. The
flag is true by construction.

Each defect is `f(text, record, rng) -> (text, flag_or_None)`.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Defect = Callable[[str, Dict[str, Any], random.Random], Tuple[str, Optional[str]]]

_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"(?:Mon|Tues?|Wed(?:nes)?|Thur?s?|Fri|Sat(?:ur)?|Sun)(?:day)?\b[^,\n]{0,12}?)"
    r"(?:\s+\d{1,2}:\d{2}\s?(?:[AaPp]\.?[Mm]\.?)?)?",
)
_TIME = re.compile(r"\b\d{1,2}:\d{2}\s?(?:[AaPp]\.?[Mm]\.?)?\b")

_VAGUE = [
    "early next week, AM if you can",
    "sometime next week",
    "start of the week works",
    "early in the week, morning preferred",
    "next week — will confirm the day",
]


def vague_pickup(text: str, record: Dict[str, Any], rng: random.Random):
    """Strip every explicit date and clock time; leave a dispatcher's hand-wave.

    The most common real defect and the one with the worst blast radius: an agent
    that resolves "early next week" into a timestamp has invented a dock booking.
    """
    out = _TIME.sub("", text)
    out = _DATE.sub(lambda m: rng.choice(_VAGUE), out)
    if out == text:
        return text, None
    return re.sub(r" {2,}", " ", out), "vague_pickup"


def unit_confusion(text: str, record: Dict[str, Any], rng: random.Random):
    """Restate the weight in kg but keep the `lb` label. Record stays the truth."""
    lb = record.get("weight_lb")
    if not lb:
        return text, None
    kg = int(round(lb / 2.20462))
    for form in (f"{lb:,}", str(lb)):
        if form in text:
            return text.replace(form, f"{kg:,}", 1), "unit_confusion"
    return text, None


def duplicate_message(text: str, record: Dict[str, Any], rng: random.Random):
    """Resend the top message. The agent must not book the load twice."""
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if len(blocks) < 2:
        return text, None
    dup = blocks[0].replace("Subject:", "Subject: RESEND -", 1)
    return dup + "\n\n" + text, "duplicate_message"


def contradictory_reply(text: str, record: Dict[str, Any], rng: random.Random):
    """Append a reply that disagrees with the record about the piece count."""
    pieces = record.get("pieces")
    if not pieces:
        return text, None
    wrong = pieces + rng.choice([-3, -2, 2, 3])
    reply = (
        "\n\n---\n"
        "Subject: RE: correction\n\n"
        f"Ignore my last - it's {wrong} pallets not {pieces}. Sorry.\n"
    )
    return text + reply, "contradictory_reply"


def buried_hazmat(text: str, record: Dict[str, Any], rng: random.Random):
    """Hide a placard requirement in a signature footer. An escalation trigger."""
    footer = (
        "\n\n--\nShipping notes: consignment includes 2 pallets classified "
        "UN1950 (aerosols, limited qty). Placarding per 49 CFR applies.\n"
    )
    return text + footer, "buried_hazmat"


def typo(text: str, record: Dict[str, Any], rng: random.Random):
    """Drop a character from a prose word. Never touches a digit or an identifier."""
    words = [
        (m.start(), m.group())
        for m in re.finditer(r"\b[a-z]{5,}\b", text)
        if not any(c.isdigit() for c in m.group())
    ]
    if not words:
        return text, None
    start, word = rng.choice(words)
    cut = rng.randrange(1, len(word) - 1)
    broken = word[:cut] + word[cut + 1:]
    return text[:start] + broken + text[start + len(word):], "typo"


CATALOGUE: Dict[str, Defect] = {
    "vague_pickup": vague_pickup,
    "unit_confusion": unit_confusion,
    "duplicate_message": duplicate_message,
    "contradictory_reply": contradictory_reply,
    "buried_hazmat": buried_hazmat,
    "typo": typo,
}


@dataclass
class Damaged:
    text: str
    flags: List[str]


def apply_defects(
    text: str,
    record: Dict[str, Any],
    rng: random.Random,
    rates: Dict[str, float],
    catalogue: Optional[Dict[str, Defect]] = None,
) -> Damaged:
    """Apply each defect independently at its configured rate.

    Returns the damaged text and the flags that actually took effect — a defect
    that found nothing to damage reports nothing, so the manifest never claims a
    failure mode the corpus does not contain.
    """
    cat = catalogue or CATALOGUE
    unknown = [k for k in rates if k not in cat]
    if unknown:
        raise ValueError(f"unknown defect(s): {unknown}")

    flags: List[str] = []
    out = text
    for name in sorted(rates):            # sorted: order must not depend on dict order
        if rng.random() >= rates[name]:
            continue
        out, flag = cat[name](out, record, rng)
        if flag:
            flags.append(flag)
    return Damaged(text=out, flags=flags)


def observed_rates(all_flags: Sequence[Sequence[str]]) -> Dict[str, float]:
    """What fraction of the corpus actually carries each defect.

    Report this, not the configured rate. A defect that never found anything to
    damage is a failure mode your corpus does not contain, whatever the config says.
    """
    n = max(1, len(all_flags))
    counts: Dict[str, int] = {}
    for flags in all_flags:
        for f in flags:
            counts[f] = counts.get(f, 0) + 1
    return {k: round(v / n, 4) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
