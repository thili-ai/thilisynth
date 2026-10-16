"""`docs-v1` — the paperwork that comes back after a load is delivered: a Bill of Lading and a Proof of Delivery.

A page here is not an image. It is what an OCR engine hands you: a list of tokens, each with text and a box.
That is a deliberate choice, and it has a cost the corpus card states plainly: **the scan damage is modelled in
code, not sampled from real scanners.** In exchange every error is attributable, because the generator knows
exactly which token it damaged and why.

Built the way the rest of the pack is built, from a record, in code, from a seed:

  * Every load is one record from `tenders-v1`. The BOL carries the tender's own load id, pieces, weight and
    commodity. The booking the desk checks the documents against is that tender, plus a consignee, which the
    tender email never named.
  * The facts are decided first (what was loaded, what arrived, who signed). The pages are then drawn from
    those facts, on one of four templates, and only afterwards damaged. The outcome (post or hold) is derived
    from the facts, never from the damage and never from the learner's own code.
  * No OCR confidence is provided. A real engine reports one, but a simulated one would be a number the
    generator invented, correlated with the damage by construction. The reader has to earn its own.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.sample import Sampler
from . import reference as ref
from .paperwork_rules import DESK_CARD, DeskCard

# --- the fields -----------------------------------------------------------------------------------------

BOL_FIELDS = [
    "bol_number", "load_ref", "shipper", "consignee", "origin", "destination", "pieces", "weight_lb",
    "commodity", "hs_code", "seal_number", "pickup_date", "shipper_signature",
]
POD_FIELDS = [
    "load_ref", "receiver", "delivery_date", "pieces_received", "exceptions", "seal_number", "receiver_signature",
]
FIELDS = {"bol": BOL_FIELDS, "pod": POD_FIELDS}

# How a field is laid out on templates 2 and 3.
_GROUP = {
    "bol": {"shipper": "party", "consignee": "party", "bol_number": "ref", "load_ref": "ref", "pickup_date": "ref",
            "seal_number": "ref", "pieces": "item", "commodity": "item", "hs_code": "item", "weight_lb": "item",
            "origin": "rest", "destination": "rest", "shipper_signature": "rest"},
    "pod": {"load_ref": "ref", "delivery_date": "ref", "seal_number": "ref", "receiver": "party",
            "pieces_received": "item", "exceptions": "rest", "receiver_signature": "rest"},
}

_LABELS = {
    ("bol", 0): {"bol_number": "B/L No.", "load_ref": "Load Reference", "shipper": "Shipper", "consignee": "Consignee",
                 "origin": "Ship From", "destination": "Ship To", "pieces": "Pieces", "weight_lb": "Weight (lb)",
                 "commodity": "Description", "hs_code": "HS Code", "seal_number": "Seal No.",
                 "pickup_date": "Pickup Date", "shipper_signature": "Shipper Signature"},
    ("bol", 1): {"bol_number": "BILL OF LADING NO", "load_ref": "LOAD REF", "shipper": "SHIPPER", "consignee": "CONSIGNEE",
                 "origin": "ORIGIN", "destination": "DESTINATION", "pieces": "PIECES", "weight_lb": "TOTAL WEIGHT",
                 "commodity": "COMMODITY", "hs_code": "HS", "seal_number": "SEAL", "pickup_date": "PICKUP",
                 "shipper_signature": "SIGNED BY (SHIPPER)"},
    ("bol", 2): {"bol_number": "Document No.", "load_ref": "Customer Ref", "shipper": "From", "consignee": "To (Consignee)",
                 "origin": "Origin", "destination": "Destination", "pieces": "Qty", "weight_lb": "Gross Weight",
                 "commodity": "Goods", "hs_code": "Tariff Code", "seal_number": "Seal Number",
                 "pickup_date": "Date Shipped", "shipper_signature": "Shipper Sign"},
    ("bol", 3): {"bol_number": "BOL #", "load_ref": "Reference", "shipper": "Shipper Name", "consignee": "Deliver To",
                 "origin": "Origin City", "destination": "Dest City", "pieces": "Pcs", "weight_lb": "Gross Wt (lb)",
                 "commodity": "Description of Articles", "hs_code": "HS", "seal_number": "Seal",
                 "pickup_date": "Ship Date", "shipper_signature": "Driver/Shipper Signature"},
    ("pod", 0): {"load_ref": "Load Reference", "receiver": "Received By", "delivery_date": "Delivery Date",
                 "pieces_received": "Pieces Received", "exceptions": "Exceptions", "seal_number": "Seal No.",
                 "receiver_signature": "Receiver Signature"},
    ("pod", 1): {"load_ref": "LOAD REF", "receiver": "RECEIVER", "delivery_date": "DELIVERED",
                 "pieces_received": "PCS RCVD", "exceptions": "NOTES / EXCEPTIONS", "seal_number": "SEAL AT DELIVERY",
                 "receiver_signature": "SIGNED BY (RECEIVER)"},
    ("pod", 2): {"load_ref": "Customer Ref", "receiver": "Delivered To", "delivery_date": "Date Received",
                 "pieces_received": "Qty Received", "exceptions": "Damage or Shortage", "seal_number": "Seal Number",
                 "receiver_signature": "Receiver Sign"},
    ("pod", 3): {"load_ref": "Reference", "receiver": "Consignee Name", "delivery_date": "Rec'd Date",
                 "pieces_received": "Pcs Recd", "exceptions": "Remarks", "seal_number": "Seal",
                 "receiver_signature": "Consignee Signature"},
}
_TITLE = {"bol": "BILL OF LADING", "pod": "PROOF OF DELIVERY"}
_FOOTER = "Received subject to the classifications and tariffs in effect on the date of issue. Page 1 of 1"

N_TEMPLATES = 4
CHAR_W, CHAR_H = 9, 14

# --- consignees: new here, the tender email never named one ---------------------------------------------

_CONSIGNEE_SUFFIX = ["Distribution Center", "Stores Inc", "Retail Group", "Warehouse LLC", "Markets Inc"]
CONSIGNEES = ref._names(ref._SURNAMES, _CONSIGNEE_SUFFIX, 60, seed=20260808)
_SIGNERS = ["J. Ramirez", "T. Okafor", "S. Lindqvist", "P. Moreau", "R. Banerjee", "M. Castellano", "D. Whitcombe"]

# Chance that each paperwork defect is planted on a load. Observed rates are what the card reports.
PAPER_DEFECTS: Dict[str, float] = {
    "shortage": 0.07,
    "damage_noted": 0.06,
    "weight_off": 0.05,
    "consignee_mismatch": 0.04,
    "seal_mismatch": 0.04,
    "unsigned": 0.05,
    "load_ref_mismatch": 0.03,
}

# Scan damage by level: the share of tokens damaged on a page of that level, and how pages are spread.
DAMAGE_LEVEL_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
DAMAGE_TOKEN_RATE = [0.0, 0.05, 0.12, 0.25]
_CONFUSE = {"0": "O", "O": "0", "1": "l", "l": "1", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8",
            "2": "Z", "Z": "2", "6": "G", "G": "6"}

# When a load has several reasons to hold, the first in this list is the primary one.
PAPER_REASON_PRIORITY = [
    "load_ref_mismatch", "consignee_mismatch", "shortage", "damage_noted", "weight_off", "seal_mismatch", "unsigned",
]


# --- the data model -------------------------------------------------------------------------------------

@dataclass
class Token:
    """What an OCR engine returns. Nothing else is visible to a reader."""
    text: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class Page:
    doc_type: str                 # "bol" | "pod"
    template: int
    damage_level: int
    width: int
    height: int
    tokens: List[Token]


@dataclass
class PageTruth:
    """The answers for one page, kept apart from the page so a reader cannot see them."""
    values: Dict[str, str]                    # canonical true value of every field
    roles: List[str]                          # per token: "label" | "value" | "other"
    fields: List[Optional[str]]               # per token: the field a label names or a value belongs to
    clean: List[str]                          # per token: the text before damage
    damaged: List[bool]                       # per token: did damage change it
    damaged_fields: List[str]                 # fields with a damaged value token
    dropped_fields: List[str]                 # fields whose value tokens were all lost
    label_damaged_fields: List[str]           # fields whose label was damaged


@dataclass
class Booking:
    """What Harborline expected: the tender, plus the consignee the tender email never named."""
    load_id: str
    shipper: str
    consignee: str
    origin: str
    destination: str
    pieces: int
    weight_lb: int
    pickup_date: str


@dataclass
class PaperOutcome:
    action: str                               # "post" | "hold"
    primary_reason: Optional[str]
    reasons: List[str] = field(default_factory=list)


@dataclass
class LoadPaperwork:
    load_id: str
    shipper: str
    consignee: str
    booking: Booking
    bol: Page
    pod: Page
    bol_truth: PageTruth
    pod_truth: PageTruth
    defects: List[str]
    outcome: PaperOutcome


# --- values and how they are written on a page ----------------------------------------------------------

def _show(key_field: str, value: str, template: int) -> str:
    """How a canonical value is printed on a template: commas in weights, dates in the template's style."""
    if key_field == "weight_lb":
        return f"{int(value):,}"
    if key_field in ("pickup_date", "delivery_date"):
        d = date.fromisoformat(value)
        return d.strftime("%m/%d/%Y") if template == 2 else d.strftime("%d %b %Y") if template == 3 else value
    return value


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y")


def canonical_value(key_field: str, text: str) -> str:
    """The canonical form of a value as read off a page. What the desk does to a read value before comparing it."""
    t = re.sub(r"\s+", " ", text).strip()
    if key_field in ("weight_lb", "pieces", "pieces_received"):
        return re.sub(r"[^0-9]", "", t)
    if key_field in ("pickup_date", "delivery_date"):
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(t, fmt).date().isoformat()
            except ValueError:
                continue
        return t
    if key_field in ("shipper_signature", "receiver_signature"):
        return "signed" if t else "unsigned"
    return t


# --- laying a page out ----------------------------------------------------------------------------------

def _words(items: List[dict], text: str, x: int, y: int, role: str, fld: Optional[str]) -> int:
    for word in text.split():
        items.append({"text": word, "x": x, "y": y, "role": role, "field": fld})
        x += CHAR_W * len(word) + CHAR_W
    return x


def _place(template: int, doc: str, fields: Sequence[str]) -> Dict[str, Tuple[int, int, int, int]]:
    """(label_x, label_y, value_x, value_y) for every field, on this template."""
    group = _GROUP[doc]
    out: Dict[str, Tuple[int, int, int, int]] = {}
    if template == 0:                                   # label on the left, value to its right
        for k, f in enumerate(fields):
            y = 130 + 34 * k
            out[f] = (50, y, 320, y)
    elif template == 1:                                 # label above value, two columns
        for k, f in enumerate(fields):
            x = 50 + 400 * (k % 2)
            y = 130 + 80 * (k // 2)
            out[f] = (x, y, x, y + 24)
    elif template == 2:                                 # party blocks side by side, a row of references, an item table
        parties = [f for f in fields if group[f] == "party"]
        refs = [f for f in fields if group[f] == "ref"]
        items = [f for f in fields if group[f] == "item"]
        rest = [f for f in fields if group[f] == "rest"]
        for k, f in enumerate(parties):
            out[f] = (50 + 400 * k, 130, 50 + 400 * k, 154)
        for k, f in enumerate(refs):
            out[f] = (50 + 200 * k, 230, 50 + 200 * k, 254)
        for f, x in zip(items, (50, 150, 480, 600)):
            out[f] = (x, 340, x, 364)
        for k, f in enumerate(rest):
            out[f] = (50 + 400 * (k % 2), 450 + 80 * (k // 2), 50 + 400 * (k % 2), 474 + 80 * (k // 2))
    else:                                               # references down the left, stacked parties, an item table
        refs = [f for f in fields if group[f] == "ref"]
        parties = [f for f in fields if group[f] == "party"]
        items = [f for f in fields if group[f] == "item"]
        rest = [f for f in fields if group[f] == "rest"]
        y = 130
        for f in refs:
            out[f] = (50, y, 300, y)
            y += 34
        y += 20
        for f in parties:
            out[f] = (50, y, 50, y + 24)
            y += 70
        y += 20
        for f, x in zip(items, (50, 130, 480, 620)):
            out[f] = (x, y, x, y + 24)
        y += 90
        for f in rest:
            out[f] = (50, y, 320, y)
            y += 34
    return out


def _layout(doc: str, template: int, values: Dict[str, str], signer_for: Dict[str, str]) -> List[dict]:
    items: List[dict] = []
    _words(items, _TITLE[doc], 50, 60, "other", None)
    labels = _LABELS[(doc, template)]
    fields = FIELDS[doc]
    for f, (lx, ly, vx, vy) in _place(template, doc, fields).items():
        _words(items, labels[f], lx, ly, "label", f)
        if f.endswith("_signature"):
            if values[f] == "signed":
                _words(items, "/s/ " + signer_for[f], vx, vy, "value", f)
            continue                                   # a blank signature has a label and no value
        _words(items, _show(f, values[f], template), vx, vy, "value", f)
    _words(items, _FOOTER, 50, 1040, "other", None)
    return items


# --- scan damage ----------------------------------------------------------------------------------------

def _confuse(text: str, s: Sampler) -> str:
    out = []
    for ch in text:
        out.append(_CONFUSE[ch] if ch in _CONFUSE and s.chance(0.6) else ch)
    return "".join(out)


def _damage(items: List[dict], level: int, s: Sampler) -> List[dict]:
    """Apply scan damage to a page's tokens. Returns the new token list, each with `clean`, `damaged` and `dropped`."""
    rate = DAMAGE_TOKEN_RATE[level]
    out: List[dict] = []
    for it in items:
        base = {**it, "clean": it["text"], "damaged": False}
        if it["role"] == "other" or not s.chance(rate if it["role"] == "value" else rate / 2):
            out.append(base)
            continue
        op = s.weighted(["confuse", "split", "drop"], [0.6, 0.2, 0.2 if level >= 2 else 0.0])
        if op == "drop":
            out.append({**base, "dropped": True})
            continue
        if op == "split" and len(it["text"]) >= 5:
            cut = s.integer(2, len(it["text"]) - 2)
            left, right = it["text"][:cut], it["text"][cut:]
            out.append({**base, "text": left, "damaged": True})
            out.append({**base, "text": right, "x": it["x"] + CHAR_W * cut + 3, "damaged": True})
            continue
        changed = _confuse(it["text"], s)
        out.append({**base, "text": changed, "damaged": changed != it["text"]})
    return out


def _finish(page_items: List[dict], doc: str, template: int, level: int, values: Dict[str, str]) -> Tuple[Page, PageTruth]:
    kept = [it for it in page_items if not it.get("dropped")]
    tokens = [Token(it["text"], it["x"], it["y"], CHAR_W * len(it["text"]), CHAR_H) for it in kept]
    value_fields = {it["field"] for it in page_items if it["role"] == "value"}
    surviving = {it["field"] for it in kept if it["role"] == "value"}
    truth = PageTruth(
        values=dict(values),
        roles=[it["role"] for it in kept],
        fields=[it["field"] for it in kept],
        clean=[it["clean"] for it in kept],
        damaged=[it["damaged"] for it in kept],
        damaged_fields=sorted({it["field"] for it in kept if it["role"] == "value" and it["damaged"]}),
        dropped_fields=sorted(value_fields - surviving),
        label_damaged_fields=sorted({it["field"] for it in kept if it["role"] == "label" and it["damaged"]}),
    )
    return Page(doc, template, level, 850, 1100, tokens), truth


# --- one load -------------------------------------------------------------------------------------------

def outcome_for(booking: Booking, bol: Dict[str, str], pod: Dict[str, str], card: DeskCard = DESK_CARD) -> PaperOutcome:
    """Post or hold, derived from the facts on the documents and the booking. Never from the damage."""
    reasons = []
    if pod["load_ref"] != booking.load_id:
        reasons.append("load_ref_mismatch")
    if pod["receiver"] != booking.consignee:
        reasons.append("consignee_mismatch")
    if int(pod["pieces_received"]) < int(bol["pieces"]) - card.pieces_tolerance:
        reasons.append("shortage")
    if "damaged" in pod["exceptions"].lower():
        reasons.append("damage_noted")
    if abs(int(bol["weight_lb"]) - booking.weight_lb) / booking.weight_lb > card.weight_tolerance:
        reasons.append("weight_off")
    if pod["seal_number"] != bol["seal_number"]:
        reasons.append("seal_mismatch")
    if bol["shipper_signature"] != "signed" or pod["receiver_signature"] != "signed":
        reasons.append("unsigned")
    ordered = sorted(set(reasons), key=PAPER_REASON_PRIORITY.index)
    return PaperOutcome("hold" if ordered else "post", ordered[0] if ordered else None, ordered)


def _swap_digits(seal: str, s: Sampler) -> str:
    digits = list(seal)
    idx = [i for i, ch in enumerate(digits) if ch.isdigit()]
    i = s.pick(idx[:-1])
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    if "".join(digits) == seal:                         # two equal neighbours: change one instead
        digits[i] = str((int(digits[i]) + 1) % 10)
    return "".join(digits)


def _make_load(rec: Dict[str, Any], i: int, seed: int, card: DeskCard) -> LoadPaperwork:
    s = Sampler(seed * 1_000_003 + i)
    defects = [d for d in PAPER_DEFECTS if s.chance(PAPER_DEFECTS[d])]

    consignee = s.zipf(CONSIGNEES)
    pieces, weight = rec["pieces"], rec["weight_lb"]
    origin = f"{rec['origin_city']}, {rec['origin_state']}"
    destination = f"{rec['destination_city']}, {rec['destination_state']}"
    booking = Booking(rec["load_id"], rec["shipper"], consignee, origin, destination, pieces, weight, rec["pickup_date"])

    # The BOL weight is the booked weight, or a scale difference inside tolerance, or (weight_off) outside it.
    if "weight_off" in defects:
        factor = 1 + s.weighted([-1, 1], [0.5, 0.5]) * s.integer(8, 20) / 100
    elif s.chance(0.3):
        factor = 1 + s.weighted([-1, 1], [0.5, 0.5]) * s.integer(1, 3) / 100
    else:
        factor = 1.0
    bol_weight = int(round(weight * factor / 10) * 10) if factor != 1.0 else weight

    seal = f"SL{s.integer(1000000, 9999999)}"
    bol_values = {
        "bol_number": f"BL-{(i * 7919 + 104729) % 900000 + 100000}",
        "load_ref": rec["load_id"],
        "shipper": rec["shipper"],
        "consignee": consignee,
        "origin": origin,
        "destination": destination,
        "pieces": str(pieces),
        "weight_lb": str(bol_weight),
        "commodity": rec["commodity"],
        "hs_code": rec["hs_code"],
        "seal_number": seal,
        "pickup_date": rec["pickup_date"],
        "shipper_signature": "signed",
    }

    received = pieces
    exceptions = "None"
    if "shortage" in defects:
        k = s.integer(1, min(3, pieces - 1))
        received = pieces - k
        exceptions = f"Received {received} of {pieces} pallets" if s.chance(0.7) else "None"
    if "damage_noted" in defects:
        n_dmg = s.integer(1, 3)
        exceptions = f"{n_dmg} pallets damaged - {s.pick(['water', 'crushed', 'torn wrap'])}"
    transit = max(1, math.ceil(rec["miles"] / 550)) + s.integer(0, 1)
    delivery = (date.fromisoformat(rec["pickup_date"]) + timedelta(days=transit)).isoformat()
    other_loads = [f"TNDR-{40000 + max(0, i + d)}" for d in (-3, -2, -1, 1, 2, 3)]
    other_loads = [x for x in other_loads if x != rec["load_id"]]

    pod_values = {
        "load_ref": s.pick(other_loads) if "load_ref_mismatch" in defects else rec["load_id"],
        "receiver": s.pick([c for c in CONSIGNEES if c != consignee]) if "consignee_mismatch" in defects else consignee,
        "delivery_date": delivery,
        "pieces_received": str(received),
        "exceptions": exceptions,
        "seal_number": _swap_digits(seal, s) if "seal_mismatch" in defects else seal,
        "receiver_signature": "signed",
    }
    if "unsigned" in defects:
        if s.chance(0.5):
            bol_values["shipper_signature"] = "unsigned"
        else:
            pod_values["receiver_signature"] = "unsigned"

    signer = {"shipper_signature": s.pick(_SIGNERS), "receiver_signature": s.pick(_SIGNERS)}
    t_bol = ref.SHIPPERS.index(rec["shipper"]) % N_TEMPLATES
    t_pod = CONSIGNEES.index(pod_values["receiver"]) % N_TEMPLATES      # the form belongs to the company that receives
    lvl_bol = s.weighted(range(4), DAMAGE_LEVEL_WEIGHTS)
    lvl_pod = s.weighted(range(4), DAMAGE_LEVEL_WEIGHTS)

    bol_page, bol_truth = _finish(_damage(_layout("bol", t_bol, bol_values, signer), lvl_bol, s), "bol", t_bol, lvl_bol, bol_values)
    pod_page, pod_truth = _finish(_damage(_layout("pod", t_pod, pod_values, signer), lvl_pod, s), "pod", t_pod, lvl_pod, pod_values)
    return LoadPaperwork(
        load_id=rec["load_id"], shipper=rec["shipper"], consignee=consignee, booking=booking,
        bol=bol_page, pod=pod_page, bol_truth=bol_truth, pod_truth=pod_truth, defects=defects,
        outcome=outcome_for(booking, bol_values, pod_values, card),
    )


def generate_paperwork(records: Sequence[Dict[str, Any]], seed: int = 20260806, card: DeskCard = DESK_CARD) -> List[LoadPaperwork]:
    """Paperwork for each tender record, in order. Reproducible from `seed` and the records alone."""
    return [_make_load(rec, i, seed, card) for i, rec in enumerate(records)]


def generate_docs(n: int = 2000, seed: int = 20260806, card: DeskCard = DESK_CARD) -> List[LoadPaperwork]:
    """`docs-v1`: paperwork for the first `n` tenders of `tenders-v1`."""
    from . import generate_tenders
    corpus = generate_tenders(n=n, seed=seed)
    return generate_paperwork([r.record for r in corpus.rows], seed, card)


# --- what the corpus actually contains ------------------------------------------------------------------

def paperwork_stats(loads: Sequence[LoadPaperwork]) -> Dict[str, Any]:
    """Observed, not configured. This is what the corpus card reports."""
    n = max(1, len(loads))
    pages = [(p, t) for l in loads for p, t in ((l.bol, l.bol_truth), (l.pod, l.pod_truth))]
    by_level: Dict[int, Dict[str, int]] = {k: {"pages": 0, "tokens": 0, "damaged": 0, "fields": 0, "fields_damaged": 0}
                                           for k in range(4)}
    for p, t in pages:
        row = by_level[p.damage_level]
        value_tokens = sum(r == "value" for r in t.roles)
        row["pages"] += 1
        row["tokens"] += value_tokens
        row["damaged"] += sum(d and r == "value" for d, r in zip(t.damaged, t.roles))
        row["fields"] += len(FIELDS[p.doc_type])
        row["fields_damaged"] += len(set(t.damaged_fields) | set(t.dropped_fields))
    reasons: Dict[str, int] = {}
    primaries: Dict[str, int] = {}
    for l in loads:
        for r in l.outcome.reasons:
            reasons[r] = reasons.get(r, 0) + 1
        if l.outcome.primary_reason:
            primaries[l.outcome.primary_reason] = primaries.get(l.outcome.primary_reason, 0) + 1
    planted: Dict[str, int] = {}
    for l in loads:
        for d in l.defects:
            planted[d] = planted.get(d, 0) + 1
    templates: Dict[int, int] = {}
    for p, _ in pages:
        templates[p.template] = templates.get(p.template, 0) + 1
    return {
        "n": len(loads),
        "post_share": round(sum(l.outcome.action == "post" for l in loads) / n, 4),
        "planted_defects": dict(sorted(planted.items(), key=lambda kv: -kv[1])),
        "reason_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "primary_reason_counts": dict(sorted(primaries.items(), key=lambda kv: -kv[1])),
        "damage_by_level": by_level,
        "templates": dict(sorted(templates.items())),
    }
