"""`auditdesk-rules-v1` — the numbers Harborline's billing desk works to.

The same idea as `ratecard-v1` for booking and `paperwork-rules-v1` for the paperwork desk: the desk's rules live in the
dataset, not in the learner's code, so an engine with a wrong rule cannot mark itself right.

**The dollar figures and the dispute behaviour are illustrative.** They are chosen to be plausible and to hang together.
They are not industry data and nothing was measured to produce them. In particular, how often a carrier concedes a
dispute is an assumption of the course. The drill's argument is how a threshold follows from a cost; the exact numbers
are the course's, and the corpus card says so.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditCard:
    as_of: str = "2026-09-01"
    # Detention: the first hours at each stop are free, each further started hour is billed at the rate card's detention rate.
    detention_free_hours: int = 2
    # A difference this small is rounding, not an overcharge.
    variance_tolerance_usd: float = 1.00
    # What it costs the desk to raise and chase one dispute with a carrier.
    dispute_cost_usd: float = 35.0
    # What it costs to have a person look at an invoice and decide.
    review_cost_usd: float = 12.0
    # What it costs, in goodwill and rework, to dispute a charge that was right.
    false_dispute_cost_usd: float = 150.0
    # How much of a disputed overcharge the carrier gives back. A finding that cites its source is far more likely to stick.
    upheld_with_evidence: float = 0.90
    upheld_without_evidence: float = 0.30


AUDIT_CARD = AuditCard()
