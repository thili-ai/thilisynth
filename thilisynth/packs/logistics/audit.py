"""`invoices-v1` — what a carrier bills after a load is delivered, and what Harborline actually owes.

A load has a rate confirmation (what was agreed), a gate log (when the truck was at each stop) and a carrier invoice (what the
carrier says it is owed). The invoice is built from the facts and then overcharged in code, so every overcharge label is true
by construction and the amount that should be recovered is known to the cent.

  * Every invoice is for a load in `tenders-v1`, with a carrier from `carriers-v1`, priced against `ratecard-v1`. The delivery date
    comes from `docs-v1`.
  * Carriers word the same charge four different ways. Normalising the wording is most of the work of reading an invoice.
  * Some things look odd and are right: authorised extras, detention the gate log supports, wording in four dialects, rounding of
    under a dollar. They are what make a false dispute possible.
  * The data cannot tell whether an authorised extra was actually performed, so an authorised charge is treated as payable. That is
    a limit of the data, stated in the card.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.sample import Sampler
from . import reference as ref
from .audit_rules import AUDIT_CARD, AuditCard
from .carriers import Carrier, carrier_ask, carrier_floor, generate_carriers
from .ratecard import RATE_CARD, RateCard
from .truth import vetting_failures

# The overcharge classes, and the chance each is planted on a load. Observed rates are what the card reports.
AUDIT_DEFECTS: Dict[str, float] = {
    "rate_mismatch": 0.05,
    "fuel_error": 0.05,
    "detention_overbilled": 0.06,
    "unauthorised_accessorial": 0.05,
    "accessorial_rate_error": 0.08,
    "duplicate_invoice": 0.03,
}

# Where a person can check each class. A finding that cites this source is evidenced.
EVIDENCE_SOURCE: Dict[str, str] = {
    "rate_mismatch": "rate_confirmation.linehaul_usd",
    "fuel_error": "rate_confirmation.miles",
    "detention_overbilled": "gate_log",
    "unauthorised_accessorial": "rate_confirmation.authorised",
    "accessorial_rate_error": "rate_card.accessorials_usd",
    "duplicate_invoice": "invoice.duplicate_of",
}

CODES = ["linehaul", "fuel", "detention", "layover", "lumper fee", "driver assist", "reconsignment", "tarping",
         "stop-off", "after-hours delivery", "liftgate"]

# One wording of every charge per carrier dialect.
DIALECTS: List[Dict[str, str]] = [
    {"linehaul": "Linehaul", "fuel": "Fuel Surcharge", "detention": "Detention", "layover": "Layover",
     "lumper fee": "Lumper Fee", "driver assist": "Driver Assist", "reconsignment": "Reconsignment", "tarping": "Tarping",
     "stop-off": "Stop-off", "after-hours delivery": "After-hours Delivery", "liftgate": "Liftgate"},
    {"linehaul": "LH", "fuel": "FSC", "detention": "DTN HRS", "layover": "LAYOVR", "lumper fee": "LUMPER",
     "driver assist": "DRV ASSIST", "reconsignment": "RECONSIGN", "tarping": "TARP", "stop-off": "STOP OFF",
     "after-hours delivery": "AFTER HRS DEL", "liftgate": "LIFTGATE"},
    {"linehaul": "Line Haul Freight", "fuel": "Fuel Adjustment", "detention": "Waiting Time", "layover": "Layover Day",
     "lumper fee": "Unloading Service", "driver assist": "Driver Assistance", "reconsignment": "Reconsignment Fee",
     "tarping": "Tarp Charge", "stop-off": "Additional Stop", "after-hours delivery": "Out-of-hours Delivery",
     "liftgate": "Lift Gate Service"},
    {"linehaul": "FRT-001 Freight", "fuel": "FUE-100 Fuel", "detention": "ACC-DET Det. time", "layover": "ACC-LAY Layover",
     "lumper fee": "ACC-LMP Lumper", "driver assist": "ACC-DRA Driver assist", "reconsignment": "ACC-RCN Reconsign",
     "tarping": "ACC-TRP Tarping", "stop-off": "ACC-STP Stop-off", "after-hours delivery": "ACC-AHD After hrs",
     "liftgate": "ACC-LFT Liftgate"},
]
N_DIALECTS = len(DIALECTS)
ROUNDS_TO_DOLLAR = {1}                      # dialect 1 bills whole dollars on every line


# --- the data model -------------------------------------------------------------------------------------

@dataclass
class RateConfirmation:
    """What Harborline agreed with the carrier: the contract for the load."""
    load_id: str
    carrier_id: str
    carrier_name: str
    linehaul_usd: int
    miles: int
    authorised: List[str]            # accessorials the booking allows (detention is handled by the gate log, not by authorisation)


@dataclass
class GateEvent:
    stop: str                        # "pickup" | "delivery"
    arrival: str                     # ISO, to the minute
    departure: str

    @property
    def minutes(self) -> int:
        return int((datetime.fromisoformat(self.departure) - datetime.fromisoformat(self.arrival)).total_seconds() // 60)


@dataclass
class InvoiceLine:
    description: str                 # in the carrier's own wording
    qty: float
    unit_price: float
    amount: float


@dataclass
class Invoice:
    invoice_no: str
    load_id: str
    carrier_id: str
    carrier_name: str
    date: str
    lines: List[InvoiceLine]
    total: float
    duplicate_of: Optional[str] = None   # the invoice number this one repeats, when it does


@dataclass
class ExpectedLine:
    """The truth about one invoice line."""
    code: str                        # the normalised charge
    billed_usd: float
    expected_usd: float              # what the rate confirmation, the rate card and the gate log support
    overbilled_usd: float            # the planted overcharge on this line, or 0.0
    cls: Optional[str]               # the class of the planted overcharge


@dataclass
class InvoiceTruth:
    lines: List[ExpectedLine]
    billed_usd: float
    expected_usd: float              # what Harborline owes on this invoice
    overbilled_usd: float            # the sum of the planted overcharges
    classes: List[str]
    action: str                      # "dispute" when anything is overbilled, else "pay"


@dataclass
class AuditedLoad:
    load_id: str
    rate_confirmation: RateConfirmation
    gate_log: List[GateEvent]
    delivered_on: str
    invoices: List[Invoice]
    truths: List[InvoiceTruth]
    defects: List[str]


# --- the rules the truth follows ------------------------------------------------------------------------

def detention_hours(events: Sequence[GateEvent], card: AuditCard = AUDIT_CARD) -> int:
    """Billable detention hours: at each stop, every started hour past the free time."""
    total = 0
    for e in events:
        extra = e.minutes - 60 * card.detention_free_hours
        total += math.ceil(extra / 60) if extra > 0 else 0
    return total


def fuel_usd(miles: int, rate_card: RateCard = RATE_CARD) -> float:
    return round(miles * rate_card.fuel_surcharge_per_mile, 2)


def _round25(x: float) -> int:
    return max(25, int(round(x / 25)) * 25)


def _money(x: float, dialect: int) -> float:
    return float(round(x)) if dialect in ROUNDS_TO_DOLLAR else round(x, 2)


# --- one load -------------------------------------------------------------------------------------------

def _pick_carrier(rec: Dict[str, Any], carriers: Sequence[Carrier], s: Sampler) -> Carrier:
    base = [c for c in carriers if c.lookalike_of is None and c.rate_history]
    serving = [c for c in base if rec["origin_city"] in c.lanes_served and rec["destination_city"] in c.lanes_served
               and rec["equipment"] in c.equipment]
    vetted = [c for c in serving if not vetting_failures(c, rec["pickup_date"])]
    pool = vetted or serving or base
    return s.pick(sorted(pool, key=lambda c: c.carrier_id))


def _gate_log(rec: Dict[str, Any], delivered_on: str, s: Sampler) -> List[GateEvent]:
    def stop(name: str, day: str, hour: int) -> GateEvent:
        arrival = datetime.fromisoformat(day) + timedelta(hours=hour, minutes=s.integer(0, 59))
        dwell = s.integer(25, 105) if s.chance(0.85) else s.integer(135, 450)       # minutes
        return GateEvent(name, arrival.isoformat(timespec="minutes"), (arrival + timedelta(minutes=dwell)).isoformat(timespec="minutes"))
    start_hour = int(rec["pickup_window"][:2])
    return [stop("pickup", rec["pickup_date"], start_hour), stop("delivery", delivered_on, s.integer(7, 15))]


def _make_load(rec: Dict[str, Any], pod_delivered_on: str, i: int, seed: int, carriers: Sequence[Carrier],
               rate_card: RateCard, card: AuditCard) -> AuditedLoad:
    s = Sampler(seed * 1_000_003 + i + 7_000_000)
    defects = [d for d in AUDIT_DEFECTS if s.chance(AUDIT_DEFECTS[d])]
    carrier = _pick_carrier(rec, carriers, s)
    dialect = int(carrier.carrier_id[-4:]) % N_DIALECTS
    words = DIALECTS[dialect]

    ask, floor = carrier_ask(carrier, rec["miles"]), carrier_floor(carrier, rec["miles"])
    agreed = min(ask, max(floor, _round25(ask * 0.96)))
    authorised = [a for a in rec["accessorials"] if a != "detention"]
    rc = RateConfirmation(rec["load_id"], carrier.carrier_id, carrier.name, agreed, rec["miles"], authorised)
    gates = _gate_log(rec, pod_delivered_on, s)
    supported = detention_hours(gates, card)
    billed_acc = [a for a in authorised if s.chance(0.75)]
    if "accessorial_rate_error" in defects and not billed_acc:
        defects.remove("accessorial_rate_error")             # nothing was billed to mis-price, so nothing was planted

    expected: List[Tuple[str, float, float, float, Optional[str]]] = []   # code, billed, expected, overbilled, class
    lines: List[InvoiceLine] = []

    def add(code: str, qty: float, unit: float, billed: float, exp: float, over: float = 0.0, cls: Optional[str] = None):
        lines.append(InvoiceLine(words[code], qty, unit, _money(billed, dialect)))
        expected.append((code, _money(billed, dialect), round(exp, 2), over, cls))

    # linehaul
    billed_line = agreed
    over = 0.0
    if "rate_mismatch" in defects:
        billed_line = _round25(agreed * (1 + s.integer(4, 15) / 100))
        billed_line = billed_line if billed_line > agreed else agreed + 25
        over = billed_line - agreed
    add("linehaul", 1, float(billed_line), billed_line, agreed, over, "rate_mismatch" if over else None)

    # fuel surcharge, on the booked miles unless the carrier inflated them
    miles_billed = rec["miles"]
    if "fuel_error" in defects:
        miles_billed = int(round(rec["miles"] * (1 + s.integer(8, 25) / 100)))
    fuel_exp = fuel_usd(rec["miles"], rate_card)
    fuel_billed = miles_billed * rate_card.fuel_surcharge_per_mile
    over = _money(fuel_billed, dialect) - fuel_exp if "fuel_error" in defects else 0.0
    add("fuel", miles_billed, rate_card.fuel_surcharge_per_mile, fuel_billed, fuel_exp, round(over, 2), "fuel_error" if over else None)

    # authorised accessorials that the carrier billed
    mispriced = s.pick(billed_acc) if "accessorial_rate_error" in defects else None
    for a in billed_acc:
        amt = float(rate_card.accessorials_usd[a])
        if a == mispriced:
            extra = float(s.integer(1, 9) * 5)
            add(a, 1, amt + extra, amt + extra, amt, extra, "accessorial_rate_error")
        else:
            add(a, 1, amt, amt, amt)

    # detention
    det_rate = float(rate_card.accessorials_usd["detention"])
    hours_billed = supported
    if "detention_overbilled" in defects:
        hours_billed = supported + s.integer(1, 4)
    if hours_billed > 0:
        over = (hours_billed - supported) * det_rate
        add("detention", hours_billed, det_rate, hours_billed * det_rate, supported * det_rate, over,
            "detention_overbilled" if over else None)

    # an accessorial the booking never authorised
    if "unauthorised_accessorial" in defects:
        options = [a for a in ref.ACCESSORIALS if a != "detention" and a not in authorised]
        if options:
            a = s.pick(options)
            amt = float(rate_card.accessorials_usd[a])
            add(a, 1, amt, amt, 0.0, amt, "unauthorised_accessorial")

    total = round(sum(l.amount for l in lines), 2)
    day = (datetime.fromisoformat(pod_delivered_on) + timedelta(days=s.integer(1, 10))).date().isoformat()
    inv_no = f"{carrier.name.split()[0][:3].upper()}-{(i * 4093 + 17011) % 900000 + 100000}"
    inv = Invoice(inv_no, rec["load_id"], carrier.carrier_id, carrier.name, day, lines, total)

    ex_lines = [ExpectedLine(c, b, e, o, k) for c, b, e, o, k in expected]
    classes = sorted({l.cls for l in ex_lines if l.cls})
    truth = InvoiceTruth(ex_lines, total, round(sum(l.expected_usd for l in ex_lines), 2),
                         round(sum(l.overbilled_usd for l in ex_lines), 2), classes,
                         "dispute" if classes else "pay")
    invoices, truths = [inv], [truth]

    if "duplicate_invoice" in defects:
        dup_day = (datetime.fromisoformat(day) + timedelta(days=s.integer(3, 15))).date().isoformat()
        dup = Invoice(f"{inv_no}-A" if s.chance(0.5) else f"{inv_no[:3]}-{(i * 4093 + 53011) % 900000 + 100000}",
                      rec["load_id"], carrier.carrier_id, carrier.name, dup_day,
                      [InvoiceLine(l.description, l.qty, l.unit_price, l.amount) for l in lines], total, duplicate_of=inv_no)
        dup_lines = [ExpectedLine(l.code, l.billed_usd, 0.0, l.billed_usd, "duplicate_invoice") for l in ex_lines]
        invoices.append(dup)
        truths.append(InvoiceTruth(dup_lines, total, 0.0, round(total, 2), ["duplicate_invoice"], "dispute"))

    return AuditedLoad(rec["load_id"], rc, gates, pod_delivered_on, invoices, truths, defects)


def generate_audit(records: Sequence[Dict[str, Any]], delivered_on: Sequence[str], seed: int = 20260806,
                   carriers: Optional[Sequence[Carrier]] = None, rate_card: RateCard = RATE_CARD,
                   card: AuditCard = AUDIT_CARD) -> List[AuditedLoad]:
    """Audit data for each tender record, in order. Reproducible from `seed` and the records alone."""
    cars = list(carriers) if carriers is not None else generate_carriers(seed=seed)
    return [_make_load(r, d, i, seed, cars, rate_card, card) for i, (r, d) in enumerate(zip(records, delivered_on))]


def generate_invoices(n: int = 2000, seed: int = 20260806, card: AuditCard = AUDIT_CARD) -> List[AuditedLoad]:
    """`invoices-v1`: the audit data for the first `n` tenders of `tenders-v1`."""
    from . import generate_paperwork, generate_tenders
    records = [r.record for r in generate_tenders(n=n, seed=seed).rows]
    docs = generate_paperwork(records, seed)
    return generate_audit(records, [d.pod_truth.values["delivery_date"] for d in docs], seed, card=card)


# --- what the corpus actually contains ------------------------------------------------------------------

def audit_stats(loads: Sequence[AuditedLoad]) -> Dict[str, Any]:
    """Observed, not configured. This is what the corpus card reports."""
    invoices = [(l, inv, t) for l in loads for inv, t in zip(l.invoices, l.truths)]
    n_inv = max(1, len(invoices))
    planted: Dict[str, int] = {}
    for l in loads:
        for d in l.defects:
            planted[d] = planted.get(d, 0) + 1
    by_class: Dict[str, int] = {}
    over_by_class: Dict[str, float] = {}
    for _, _, t in invoices:
        for ln in t.lines:
            if ln.cls:
                by_class[ln.cls] = by_class.get(ln.cls, 0) + 1
                over_by_class[ln.cls] = over_by_class.get(ln.cls, 0.0) + ln.overbilled_usd
    dialect_counts: Dict[int, int] = {}
    for _, inv, _ in invoices:
        d = int(inv.carrier_id[-4:]) % N_DIALECTS
        dialect_counts[d] = dialect_counts.get(d, 0) + 1
    billed = sum(t.billed_usd for _, _, t in invoices)
    over = sum(t.overbilled_usd for _, _, t in invoices)
    legit_detention = sum(1 for _, _, t in invoices for ln in t.lines if ln.code == "detention" and not ln.cls)
    rounding = [abs(t.billed_usd - t.expected_usd - t.overbilled_usd) for _, _, t in invoices]
    return {
        "loads": len(loads),
        "invoices": len(invoices),
        "dispute_share": round(sum(t.action == "dispute" for _, _, t in invoices) / n_inv, 4),
        "planted_defects": dict(sorted(planted.items(), key=lambda kv: -kv[1])),
        "lines_by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "overbilled_by_class": {k: round(v, 2) for k, v in sorted(over_by_class.items(), key=lambda kv: -kv[1])},
        "billed_usd": round(billed, 2),
        "overbilled_usd": round(over, 2),
        "overbilled_share": round(over / billed, 4) if billed else 0.0,
        "legit_detention_lines": legit_detention,
        "max_rounding_usd": round(max(rounding), 2) if rounding else 0.0,
        "dialects": dict(sorted(dialect_counts.items())),
    }
