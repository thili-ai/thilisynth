"""`paperwork-rules-v1` — the numbers Harborline's paperwork desk works to.

The same idea as `ratecard-v1` for the booking desk: the desk's rules live in the dataset, not in the
learner's code, so a policy with a wrong threshold cannot mark itself right.

**The dollar figures are illustrative.** They are chosen to be plausible and to hang together (a wrong load
reference costs more than a wrong weight, which costs more than a wrong commodity description). They are not
industry data and nothing was measured to produce them. The drill's argument is how a threshold follows from
a cost; the exact costs are the course's, and the corpus card says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Cost, in dollars, of posting this field wrong and nobody noticing. Keyed "<doc>.<field>".
_FIELD_COST: Dict[str, int] = {
    "bol.load_ref": 400,
    "pod.load_ref": 400,
    "bol.weight_lb": 150,
    "bol.pieces": 150,
    "pod.pieces_received": 150,
    "bol.consignee": 120,
    "pod.receiver": 120,
    "pod.exceptions": 100,
    "bol.shipper_signature": 80,
    "pod.receiver_signature": 80,
    "bol.hs_code": 60,
    "bol.seal_number": 40,
    "pod.seal_number": 40,
    "bol.pickup_date": 25,
    "pod.delivery_date": 25,
    "bol.bol_number": 20,
    "bol.origin": 15,
    "bol.destination": 15,
    "bol.shipper": 10,
    "bol.commodity": 2,
}

# The tier a field belongs to, for the consequence table the learner builds in lesson 4.
_TIER: Dict[str, str] = {
    "bol.load_ref": "critical", "pod.load_ref": "critical",
    "bol.weight_lb": "high", "bol.pieces": "high", "pod.pieces_received": "high",
    "bol.consignee": "high", "pod.receiver": "high", "pod.exceptions": "high",
    "bol.shipper_signature": "high", "pod.receiver_signature": "high",
    "bol.hs_code": "medium", "bol.seal_number": "medium", "pod.seal_number": "medium",
    "bol.pickup_date": "medium", "pod.delivery_date": "medium",
    "bol.bol_number": "low", "bol.origin": "low", "bol.destination": "low",
    "bol.shipper": "low", "bol.commodity": "low",
}


@dataclass(frozen=True)
class DeskCard:
    as_of: str = "2026-09-01"
    # A BOL weight within this share of the booked weight is a scale difference, not a discrepancy.
    weight_tolerance: float = 0.05
    # Pieces must match exactly.
    pieces_tolerance: int = 0
    # The field accuracy the posted fields must reach for a document to count as straight-through.
    accuracy_floor: float = 0.995
    # What it costs to have a person look at one document and fix it.
    review_cost_usd: float = 3.0
    field_cost_usd: Dict[str, int] = field(default_factory=lambda: dict(_FIELD_COST))
    tier: Dict[str, str] = field(default_factory=lambda: dict(_TIER))

    def cost(self, key: str) -> int:
        """Dollars lost when this field is posted wrong. `key` is '<doc>.<field>'."""
        return self.field_cost_usd[key]


DESK_CARD = DeskCard()
