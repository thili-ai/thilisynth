"""The `LoadTender` record — the ground truth for the `tenders-v1` corpus.

Sampled first, rendered second. Every field here is a label, for free, at any
volume: nothing in this file ever asks a model what the answer is.

Constraints are enforced where they exist in reality — a reefer commodity gets
reefer equipment, weight stays under the trailer's payload limit, and the offered
rate is derived from mileage and equipment rather than drawn from thin air. A
corpus whose internal relationships are random teaches a model relationships that
do not exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...core.sample import Sampler
from . import reference as ref


@dataclass
class LoadTender:
    load_id: str
    shipper: str
    carrier_invited: str
    origin_city: str
    origin_state: str
    destination_city: str
    destination_state: str
    miles: int
    equipment: str
    temp_f: Optional[int]
    commodity: str
    hs_code: str
    weight_lb: int
    pieces: int
    accessorials: List[str]
    rate_offered_usd: int
    pickup_date: str
    pickup_window: str


def sample_tender(sampler: Sampler, n: int) -> LoadTender:
    """Draw one internally-consistent load tender."""
    origin_city, origin_state, o_lat, o_lon = sampler.zipf(ref.CITIES)
    dest_city, dest_state, d_lat, d_lon = sampler.zipf(ref.CITIES)
    while dest_city == origin_city:
        dest_city, dest_state, d_lat, d_lon = sampler.zipf(ref.CITIES)

    hs_code, commodity, needs_reefer, density = sampler.zipf(ref.COMMODITIES)

    eligible = [e for e in ref.EQUIPMENT if e[2] == needs_reefer] or list(ref.EQUIPMENT)
    equip_name, max_payload, is_reefer = sampler.pick(eligible)

    weight = sampler.integer(int(max_payload * 0.45), int(max_payload * 0.97), step=100)
    pieces = sampler.integer(8, 26)
    miles = ref.road_miles(o_lat, o_lon, d_lat, d_lon)

    base = ref.RATE_PER_MILE.get(equip_name, 2.20) * miles
    rate = sampler.money(int(base * 0.88), int(base * 1.18), round_to=25)

    acc = sampler.subset(ref.ACCESSORIALS, sampler.integer(0, 2))

    day = 3 + (n % 20)
    hour = sampler.pick([6, 7, 8, 9, 13, 14])

    return LoadTender(
        load_id=f"TNDR-{40000 + n}",
        shipper=sampler.zipf(ref.SHIPPERS),
        carrier_invited=sampler.zipf(ref.CARRIERS),
        origin_city=origin_city,
        origin_state=origin_state,
        destination_city=dest_city,
        destination_state=dest_state,
        miles=miles,
        equipment=equip_name,
        temp_f=sampler.integer(32, 38) if is_reefer else None,
        commodity=commodity,
        hs_code=hs_code,
        weight_lb=weight,
        pieces=pieces,
        accessorials=acc,
        rate_offered_usd=rate,
        pickup_date=f"2026-09-{day:02d}",
        pickup_window=f"{hour:02d}:00-{hour + 4:02d}:00",
    )


# What a downstream extractor is asked for, and how. See core/schema.py — the
# nullable list is the fix for the `weight_lb: 0` hallucination.
EXTRACTION_FIELDS = [
    "load_id", "origin_city", "destination_city", "equipment",
    "weight_lb", "pieces", "temp_f", "rate_offered_usd",
]
EXTRACTION_NULLABLE = ["weight_lb", "pieces", "temp_f", "rate_offered_usd"]
EXTRACTION_DESCRIPTIONS: Dict[str, Any] = {
    "temp_f": "Required trailer temperature in Fahrenheit, or null if not stated.",
    "weight_lb": "Total shipment weight in pounds, or null if not stated in the text.",
    "rate_offered_usd": "The rate offered, in whole dollars, or null if not stated.",
}
