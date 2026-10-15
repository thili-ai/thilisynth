"""`carriers-v1` — the carrier master a booking agent searches and vets.

Drill 1 produced tenders and nothing to book them onto: the tender carries a carrier *name* and
one per-mile anchor per equipment type. A booking desk spends most of its day finding a carrier
that can cover the lane and is safe to hand freight to. This module builds the thing it searches.

Built the same way as the tenders: the record is sampled first, from a seed, and the traps are
planted in code, so every trap label is true by construction.

  * One row per carrier name already in `reference.CARRIERS`, because `carrier_invited` in every
    tender is drawn from that list. A master built from any other names would leave tenders
    pointing at carriers that do not exist.
  * A few look-alike rows with near-identical names and different MC numbers, so a lookup by name
    can return the wrong company.
  * Names, MC and DOT numbers are synthetic. They are not drawn from FMCSA, and no real carrier is
    named: putting real companies into fabricated vetting failures is not acceptable.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from ...core.sample import Sampler
from . import reference as ref

# The day the tender threads are sent. Pickup dates in `tenders-v1` run from 2026-09-03 to 2026-09-22.
AS_OF = date(2026, 9, 1)

# Chance that a carrier carries each trap. A trap that fires on nobody is a failure mode the corpus
# does not contain, so each is topped up to MIN_PER_TRAP.
TRAP_RATES: Dict[str, float] = {
    "authority_inactive": 0.04,
    "authority_pending": 0.03,
    "contact_changed": 0.04,
    "insurance_expired": 0.06,
    "insurance_lapsing": 0.05,
    "no_history": 0.07,
    "safety_unsatisfactory": 0.04,
}
MIN_PER_TRAP = 2
N_LOOKALIKES = 4

_PAYMENT_TERMS = ["net 30", "net 30", "net 45", "quick pay"]
_NAME_TWEAKS = [
    (" LLC", " L.L.C."), (" Inc", " Inc."), ("Transport", "Transports"), ("Logistics", "Logistic"),
    ("Freight Lines", "Freight Line"), ("Motor Freight", "Motor Freight Co"), ("Express", "Expres"),
    ("Carriers", "Carrier"), ("Trucking", "Truckin"),
]


@dataclass
class Carrier:
    carrier_id: str
    name: str
    mc_number: str
    dot_number: str
    authority_status: str                 # active | inactive | pending
    insurance_expiry: str                 # ISO date
    safety_rating: str                    # satisfactory | conditional | unsatisfactory | none
    equipment: List[str]
    lanes_served: List[str]               # hub cities, from reference.CITIES
    on_time_rate: Optional[float]
    payment_terms: str
    rate_history: List[float]             # dollars per mile on its last loads
    loads_completed: int
    contact_email: str
    contact_changed_days_ago: Optional[int]
    lookalike_of: Optional[str] = None    # carrier_id this one imitates
    traps: List[str] = field(default_factory=list)   # the labels, true by construction


def _iso(d: date) -> str:
    return d.isoformat()


def _slug(name: str) -> str:
    return "".join(ch for ch in name.lower().split()[0] if ch.isalnum())


def _equipment(sampler: Sampler) -> List[str]:
    names = [e[0] for e in ref.EQUIPMENT]
    chosen = [sampler.zipf(names, exponent=0.9)]
    for _ in range(sampler.integer(0, 2)):
        extra = sampler.zipf(names, exponent=0.9)
        if extra not in chosen:
            chosen.append(extra)
    return chosen


def _unique_number(sampler: Sampler, prefix: str, low: int, high: int, used: set) -> str:
    while True:
        value = f"{prefix}-{sampler.integer(low, high)}"
        if value not in used:
            used.add(value)
            return value


def _clean_carrier(sampler: Sampler, index: int, name: str, used_mc: set, used_dot: set) -> Carrier:
    equipment = _equipment(sampler)
    anchor = ref.RATE_PER_MILE.get(equipment[0], 2.20)
    cities = [c[0] for c in ref.CITIES]
    return Carrier(
        carrier_id=f"CAR-{index + 1:04d}",
        name=name,
        mc_number=_unique_number(sampler, "MC", 100000, 999999, used_mc),
        dot_number=_unique_number(sampler, "DOT", 1000000, 3999999, used_dot),
        authority_status="active",
        insurance_expiry=_iso(AS_OF + timedelta(days=sampler.integer(60, 540))),
        safety_rating=sampler.weighted(["satisfactory", "conditional"], [0.85, 0.15]),
        equipment=equipment,
        lanes_served=sorted(sampler.subset(cities, sampler.integer(6, 14))),
        on_time_rate=round(sampler.integer(82, 99) / 100, 2),
        payment_terms=sampler.pick(_PAYMENT_TERMS),
        rate_history=[round(anchor * sampler.integer(92, 105) / 100, 2) for _ in range(5)],
        loads_completed=sampler.integer(12, 900),
        contact_email=f"dispatch@{_slug(name)}.com",
        contact_changed_days_ago=None,
    )


# --- the traps: each returns True only if it actually changed the carrier ---------------------

def _plant(trap: str, c: Carrier, sampler: Sampler) -> bool:
    if trap in c.traps:
        return False
    if trap == "authority_inactive":
        if c.authority_status != "active":
            return False
        c.authority_status = "inactive"
    elif trap == "authority_pending":
        if c.authority_status != "active":
            return False
        c.authority_status = "pending"
    elif trap == "insurance_expired":
        if "insurance_lapsing" in c.traps:
            return False
        c.insurance_expiry = _iso(AS_OF - timedelta(days=sampler.integer(5, 120)))
    elif trap == "insurance_lapsing":
        # Valid today, gone before most pickups: a carrier that is fine for one tender and not another.
        if "insurance_expired" in c.traps:
            return False
        c.insurance_expiry = _iso(AS_OF + timedelta(days=sampler.integer(3, 18)))
    elif trap == "safety_unsatisfactory":
        if "no_history" in c.traps:
            return False
        c.safety_rating = "unsatisfactory"
    elif trap == "no_history":
        if "safety_unsatisfactory" in c.traps:
            return False
        c.loads_completed, c.rate_history, c.on_time_rate, c.safety_rating = 0, [], None, "none"
    elif trap == "contact_changed":
        c.contact_changed_days_ago = sampler.integer(1, 14)
        c.contact_email = f"{_slug(c.name)}.dispatch@gmail.com"
    else:  # pragma: no cover
        raise ValueError(f"unknown trap: {trap}")
    c.traps.append(trap)
    return True


def _lookalike(sampler: Sampler, index: int, original: Carrier, used_mc: set, used_dot: set) -> Carrier:
    """A carrier whose name is one small edit away from `original`, with its own MC number."""
    name = original.name
    for old, new in _NAME_TWEAKS:
        if old in name:
            name = name.replace(old, new, 1)
            break
    else:
        name = f"{name} Co"
    twin = _clean_carrier(sampler, index, name, used_mc, used_dot)
    twin.lookalike_of = original.carrier_id
    twin.traps.append("name_collision")
    _plant("authority_inactive", twin, sampler)
    _plant("no_history", twin, sampler)
    return twin


def generate_carriers(seed: int = 20260806, lookalikes: int = N_LOOKALIKES) -> List[Carrier]:
    """Build the carrier master. Reproducible from `seed` alone."""
    sampler = Sampler(seed ^ 0xCA77)
    used_mc: set = set()
    used_dot: set = set()
    carriers = [_clean_carrier(sampler, i, name, used_mc, used_dot) for i, name in enumerate(ref.CARRIERS)]

    for trap in sorted(TRAP_RATES):
        for c in carriers:
            if sampler.chance(TRAP_RATES[trap]):
                _plant(trap, c, sampler)
    for trap in sorted(TRAP_RATES):
        while sum(trap in c.traps for c in carriers) < MIN_PER_TRAP:
            eligible = [c for c in carriers if not c.traps]
            if not eligible:
                break                         # nothing clean left to plant on
            _plant(trap, sampler.pick(eligible), sampler)

    originals = sampler.subset([c for c in carriers if not c.traps], lookalikes)
    for k, original in enumerate(originals):
        carriers.append(_lookalike(sampler, len(carriers), original, used_mc, used_dot))
    return carriers


def carrier_ask(c: Carrier, miles: int) -> Optional[int]:
    """What this carrier would ask for a load of `miles`: its average rate per mile, to the $25."""
    if not c.rate_history:
        return None
    raw = sum(c.rate_history) / len(c.rate_history) * miles
    return max(25, int(round(raw / 25)) * 25)


def carrier_floor(c: Carrier, miles: int) -> Optional[int]:
    """The lowest price this carrier will accept for a load of `miles`: 95% of its ask, to the $25 below.

    A carrier asks for its average rate and will come down to this, no further. Negotiation is the
    space between the two: the agent offers, the carrier counters, and no deal is struck below the floor.
    """
    ask = carrier_ask(c, miles)
    if ask is None:
        return None
    return max(25, int(ask * 0.95) // 25 * 25)


def name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def observed_traps(carriers: List[Carrier]) -> Dict[str, int]:
    """How many carriers actually carry each trap. Report this, not the configured rate."""
    counts: Dict[str, int] = {}
    for c in carriers:
        for t in c.traps:
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
