"""What the right answer is for every tender: book a carrier at a rate, or escalate for a reason.

These rules belong to the *dataset*. They are not the learner's `policy.py`. If the grader used the
agent's own policy to decide what is correct, an agent with the wrong threshold would mark itself
right. Here the truth comes from facts planted in the data: a defect flag set by a defect function,
an insurance date in a carrier row, a rate on a card. The learner's policy is then scored against it.

An outcome is derived, never asked of a model, so the labels are free and correct by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .carriers import Carrier, carrier_floor
from .ratecard import RATE_CARD, RateCard

# The defect flags that put a thread beyond the desk's authority, and the reason each stands for.
FLAG_REASON: Dict[str, str] = {
    "buried_hazmat": "hazmat",
    "contradictory_reply": "thread_contradiction",
    "unit_confusion": "weight_implausible",
    "vague_pickup": "date_unresolvable",
}

# When a thread has several reasons, the first in this list is the primary one.
REASON_PRIORITY: List[str] = [
    "hazmat",
    "thread_contradiction",
    "weight_implausible",
    "date_unresolvable",
    "value_over_limit",
    "carrier_unvetted",
    "rate_above_ceiling",
]

CONTACT_CHANGE_WINDOW_DAYS = 30


@dataclass
class Outcome:
    action: str                                   # "book" | "escalate"
    primary_reason: Optional[str]                 # None when booking
    reasons: List[str] = field(default_factory=list)
    acceptable_carriers: List[str] = field(default_factory=list)   # carrier_ids; empty when escalating
    max_buy_usd: int = 0
    invited_carrier_id: Optional[str] = None
    invited_carrier_ok: Optional[bool] = None     # is the carrier already on the thread an acceptable one?
    expected_bookings: int = 0                    # one when booking, however many times the thread arrives
    deliveries: int = 1                           # a duplicated message arrives twice


def vetting_failures(c: Carrier, pickup: str) -> List[str]:
    """Why this carrier may not be handed freight picking up on `pickup`. Empty means it passes."""
    failures = []
    if c.authority_status != "active":
        failures.append(f"authority_{c.authority_status}")
    if c.insurance_expiry < pickup:
        failures.append("insurance_expired_by_pickup")
    if c.safety_rating == "unsatisfactory":
        failures.append("safety_unsatisfactory")
    if c.loads_completed == 0:
        failures.append("no_history")
    if c.contact_changed_days_ago is not None and c.contact_changed_days_ago <= CONTACT_CHANGE_WINDOW_DAYS:
        failures.append("contact_recently_changed")
    return failures


def serves(c: Carrier, record: Dict[str, Any]) -> bool:
    return (
        record["origin_city"] in c.lanes_served
        and record["destination_city"] in c.lanes_served
        and record["equipment"] in c.equipment
    )


def truth_for(
    record: Dict[str, Any],
    defects: Sequence[str],
    carriers: Sequence[Carrier],
    card: RateCard = RATE_CARD,
) -> Outcome:
    """The correct outcome for one tender, given the world it arrives in."""
    reasons = [FLAG_REASON[f] for f in defects if f in FLAG_REASON]

    sell = record["rate_offered_usd"]
    if sell > card.desk_authority_usd:
        reasons.append("value_over_limit")

    max_buy = card.max_buy_for_margin(sell)
    candidates = [c for c in carriers if serves(c, record)]
    vetted = [c for c in candidates if not vetting_failures(c, record["pickup_date"])]
    # Affordable means the carrier will accept a price the desk can pay, which negotiation may reach.
    affordable = [c for c in vetted if (carrier_floor(c, record["miles"]) or 10**9) <= max_buy]
    if not vetted:
        reasons.append("carrier_unvetted")
    elif not affordable:
        reasons.append("rate_above_ceiling")

    ordered = sorted(set(reasons), key=REASON_PRIORITY.index)
    base = [c for c in carriers if c.lookalike_of is None]
    invited = next((c for c in base if c.name == record["carrier_invited"]), None)
    ok_ids = {c.carrier_id for c in affordable}
    return Outcome(
        action="escalate" if ordered else "book",
        primary_reason=ordered[0] if ordered else None,
        reasons=ordered,
        acceptable_carriers=[] if ordered else sorted(ok_ids),
        max_buy_usd=max_buy,
        invited_carrier_id=invited.carrier_id if invited else None,
        invited_carrier_ok=(invited.carrier_id in ok_ids) if invited else None,
        expected_bookings=0 if ordered else 1,
        deliveries=2 if "duplicate_message" in defects else 1,
    )


def build_truth(rows: Sequence[Any], carriers: Sequence[Carrier], card: RateCard = RATE_CARD) -> List[Outcome]:
    """One outcome per corpus row, in order. `rows` are the pipeline's `Row`s."""
    return [truth_for(r.record, r.defects, carriers, card) for r in rows]


def truth_summary(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """What the truth contains. Report this, not what you meant it to contain."""
    n = max(1, len(outcomes))
    booked = [o for o in outcomes if o.action == "book"]
    reasons: Dict[str, int] = {}
    primaries: Dict[str, int] = {}
    for o in outcomes:
        for r in o.reasons:
            reasons[r] = reasons.get(r, 0) + 1
        if o.primary_reason:
            primaries[o.primary_reason] = primaries.get(o.primary_reason, 0) + 1
    invited_known = [o for o in outcomes if o.invited_carrier_ok is not None]
    return {
        "n": len(outcomes),
        "book_share": round(len(booked) / n, 4),
        "escalate_share": round(1 - len(booked) / n, 4),
        "primary_reason_counts": dict(sorted(primaries.items(), key=lambda kv: -kv[1])),
        "any_reason_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "invited_carrier_ok_share": round(sum(o.invited_carrier_ok for o in invited_known) / max(1, len(invited_known)), 4),
        "duplicate_deliveries": sum(o.deliveries > 1 for o in outcomes),
    }
