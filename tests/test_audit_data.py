"""invoices-v1 and auditdesk-rules-v1: consistent with the tenders, labels true by construction, truth independent of any engine."""
import dataclasses
import math
from datetime import datetime

import pytest

from thilisynth.packs.logistics import (
    AUDIT_CARD, AUDIT_DEFECTS, DIALECTS, EVIDENCE_SOURCE, RATE_CARD, audit_stats, audit_truth_card, auditrules_card,
    detention_hours, fuel_usd, generate_audit, generate_carriers, generate_invoices, generate_paperwork, generate_tenders,
    invoices_card,
)
from thilisynth.packs.logistics import reference as ref
from thilisynth.packs.logistics.audit import CODES
from thilisynth.packs.logistics.carriers import carrier_ask, carrier_floor

N = 1500


@pytest.fixture(scope="module")
def tenders():
    return generate_tenders(n=N)


@pytest.fixture(scope="module")
def records(tenders):
    return [r.record for r in tenders.rows]


@pytest.fixture(scope="module")
def loads(records):
    docs = generate_paperwork(records)
    return generate_audit(records, [d.pod_truth.values["delivery_date"] for d in docs])


def decode(inv):
    """The normalised code of every invoice line, by inverting the carrier's dialect. The test's own normaliser."""
    inverse = {v: k for k, v in DIALECTS[int(inv.carrier_id[-4:]) % len(DIALECTS)].items()}
    return [inverse[l.description] for l in inv.lines]


def test_same_seed_same_invoices(records):
    delivered = ["2026-09-08"] * 60
    a = [dataclasses.asdict(l) for l in generate_audit(records[:60], delivered, seed=5)]
    b = [dataclasses.asdict(l) for l in generate_audit(records[:60], delivered, seed=5)]
    assert a == b and a != [dataclasses.asdict(l) for l in generate_audit(records[:60], delivered, seed=6)]


def test_a_load_does_not_depend_on_how_many_loads_there_are(records):
    delivered = ["2026-09-08"] * 400
    small = generate_audit(records[:40], delivered[:40])
    big = generate_audit(records[:400], delivered)
    assert [dataclasses.asdict(l) for l in small] == [dataclasses.asdict(l) for l in big[:40]]


def test_tenders_and_paperwork_are_untouched_by_this_pack(records):
    a = generate_tenders(n=40).to_jsonl()
    generate_invoices(n=40)
    assert generate_tenders(n=40).to_jsonl() == a


def test_rate_confirmation_is_a_real_deal_with_a_real_carrier(records, loads):
    carriers = {c.carrier_id: c for c in generate_carriers()}
    for rec, l in zip(records, loads):
        rc, c = l.rate_confirmation, carriers[l.rate_confirmation.carrier_id]
        assert rc.load_id == rec["load_id"] == l.load_id and rc.miles == rec["miles"] and c.lookalike_of is None
        assert carrier_floor(c, rec["miles"]) <= rc.linehaul_usd <= carrier_ask(c, rec["miles"])
        assert set(rc.authorised) == set(rec["accessorials"]) - {"detention"}


def test_every_charge_has_four_distinct_wordings_and_they_do_not_collide():
    assert len(DIALECTS) == 4 and all(set(d) == set(CODES) for d in DIALECTS)
    for code in CODES:
        words = [d[code] for d in DIALECTS]
        assert len(set(words)) == 4, code
    for d in DIALECTS:
        assert len(set(d.values())) == len(d), "inside one dialect two charges share a wording"


def test_gate_log_is_consistent(records, loads):
    for rec, l in zip(records, loads):
        pu, de = l.gate_log
        assert pu.stop == "pickup" and de.stop == "delivery" and pu.minutes > 0 and de.minutes > 0
        assert datetime.fromisoformat(pu.departure) < datetime.fromisoformat(de.arrival)
        assert de.arrival[:10] == l.delivered_on


def test_the_truth_follows_the_rules_independently_of_the_generator(loads):
    """Re-derive what is owed from the rate confirmation, the rate card and the gate log, without looking at the planted labels."""
    checked = 0
    for l in loads:
        rc = l.rate_confirmation
        supported = detention_hours(l.gate_log)
        for inv, t in zip(l.invoices, l.truths):
            if inv.duplicate_of:
                assert t.expected_usd == 0.0 and t.classes == ["duplicate_invoice"]
                continue
            codes = decode(inv)
            owed = rc.linehaul_usd + fuel_usd(rc.miles)
            owed += sum(RATE_CARD.accessorials_usd[c] for c in codes if c in rc.authorised)
            owed += supported * RATE_CARD.accessorials_usd["detention"]
            assert abs(owed - t.expected_usd) < 0.005, (l.load_id, owed, t.expected_usd)
            checked += 1
    assert checked > 1000


def test_every_overcharge_label_is_true(loads):
    seen = {c: 0 for c in AUDIT_DEFECTS}
    for l in loads:
        rc = l.rate_confirmation
        supported = detention_hours(l.gate_log)
        first = l.invoices[0]
        by_code = dict(zip(decode(first), first.lines))
        if "rate_mismatch" in l.defects:
            assert by_code["linehaul"].amount > rc.linehaul_usd
        if "fuel_error" in l.defects:
            assert by_code["fuel"].qty > rc.miles
        if "detention_overbilled" in l.defects:
            assert by_code["detention"].qty > supported
        if "unauthorised_accessorial" in l.defects:
            extra = [c for c in by_code if c not in ("linehaul", "fuel", "detention") and c not in rc.authorised]
            assert extra, l.load_id
        if "accessorial_rate_error" in l.defects:
            assert any(c not in ("linehaul", "fuel", "detention") and c in rc.authorised and by_code[c].amount > RATE_CARD.accessorials_usd[c]
                       for c in by_code)
        if "duplicate_invoice" in l.defects:
            dup = l.invoices[1]
            assert dup.duplicate_of == first.invoice_no and dup.invoice_no != first.invoice_no
            assert [(x.description, x.amount) for x in dup.lines] == [(x.description, x.amount) for x in first.lines]
        # the classes on the truth are exactly the planted ones
        assert set(l.truths[0].classes) == set(l.defects) - {"duplicate_invoice"}
        for d in l.defects:
            seen[d] += 1
    assert all(v >= 30 for v in seen.values()), seen


def test_clean_invoices_are_clean_and_only_rounding_differs(loads):
    clean = [(l, i, t) for l in loads for i, t in zip(l.invoices, l.truths) if not t.classes]
    assert len(clean) > 800
    for l, inv, t in clean:
        assert t.action == "pay" and t.overbilled_usd == 0.0
        assert abs(t.billed_usd - t.expected_usd) <= 0.5 < AUDIT_CARD.variance_tolerance_usd


def test_legitimate_oddities_exist(loads):
    st = audit_stats(loads)
    assert st["legit_detention_lines"] > 100, "detention the gate log supports must be common enough to tempt a false dispute"
    assert len(st["dialects"]) == 4 and min(st["dialects"].values()) > 50
    assert any(any(c in l.rate_confirmation.authorised for c in decode(i)) for l in loads for i in l.invoices[:1])


def test_every_class_has_a_checkable_source():
    assert set(EVIDENCE_SOURCE) == set(AUDIT_DEFECTS)


def test_detention_follows_the_free_hours_in_the_card(loads):
    from thilisynth.packs.logistics.audit import GateEvent
    one = GateEvent("delivery", "2026-09-08T10:00", "2026-09-08T12:30")      # 150 minutes
    assert detention_hours([one]) == 1 and detention_hours([one], dataclasses.replace(AUDIT_CARD, detention_free_hours=1)) == 2
    assert detention_hours([GateEvent("d", "2026-09-08T10:00", "2026-09-08T12:00")]) == 0, "exactly two hours is free"


def test_the_cards_state_what_the_data_does_not_contain(loads):
    for text in (invoices_card(loads, 1), auditrules_card(AUDIT_CARD), audit_truth_card(loads)):
        assert "does NOT contain" in text
    assert "optimistic bound" in invoices_card(loads, 1)
    assert "illustrative" in auditrules_card(AUDIT_CARD)
