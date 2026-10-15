"""`ratecard-v1` — the rates and rules of Harborline's booking desk.

Anchored to the numbers the tenders were built from. A tender's offered rate is
`reference.RATE_PER_MILE[equipment]` times the road mileage, with a spread of 0.88 to 1.18 and
rounding to $25. This card uses the same anchors and the same rounding, so a tender's price and the
card's benchmark agree: the agent's margin arithmetic has one consistent world to work in.

What is in it, and who uses it:

  * the carrier benchmark (the buy side) and a ceiling above it     — `lfbooking`, `lfaudit`
  * the margin the desk must keep, and the value it may book alone  — `lfbooking`
  * a fuel-surcharge index and an accessorial schedule              — `lfaudit`

The sell side is the tender's own offered rate. There is no table of per-shipper contract rates:
none is needed to price a load, and inventing one would add numbers nothing checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from . import reference as ref


def _to_25(x: float) -> int:
    return max(25, int(round(x / 25)) * 25)


def _default_accessorials() -> Dict[str, int]:
    return {
        "detention": 75,            # per hour after two free
        "layover": 250,             # per day
        "lumper fee": 150,
        "driver assist": 100,
        "reconsignment": 200,
        "tarping": 125,
        "stop-off": 100,
        "after-hours delivery": 150,
        "liftgate": 75,
    }


@dataclass(frozen=True)
class RateCard:
    as_of: str = "2026-09-01"
    benchmark_per_mile: Dict[str, float] = field(default_factory=lambda: dict(ref.RATE_PER_MILE))
    # The share of the sell rate the desk must keep. A booking below it is not the agent's to make.
    margin_floor: float = 0.04
    # Carriers may be paid up to this multiple of the benchmark.
    carrier_ceiling_multiplier: float = 1.12
    # The load value (the sell rate) the desk may book without a person. Harborline's rule.
    desk_authority_usd: int = 6000
    fuel_surcharge_per_mile: float = 0.42
    accessorials_usd: Dict[str, int] = field(default_factory=_default_accessorials)

    def benchmark_buy(self, equipment: str, miles: int) -> int:
        """The carrier benchmark for a load: the per-mile anchor times the mileage."""
        return _to_25(self.benchmark_per_mile.get(equipment, 2.20) * miles)

    def buy_ceiling(self, equipment: str, miles: int) -> int:
        return _to_25(self.benchmark_per_mile.get(equipment, 2.20) * miles * self.carrier_ceiling_multiplier)

    def max_buy_for_margin(self, sell_usd: int) -> int:
        """The most the desk may pay a carrier and still keep its margin, to the $25 below."""
        return int((sell_usd * (1 - self.margin_floor)) // 25 * 25)

    def margin(self, sell_usd: int, buy_usd: int) -> float:
        return round((sell_usd - buy_usd) / sell_usd, 4) if sell_usd else 0.0


RATE_CARD = RateCard()
