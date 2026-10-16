"""docs-v1 and paperwork-rules-v1: consistent with the tenders, labels true by construction, truth independent of damage."""
import dataclasses
import json
from datetime import date

import pytest

from thilisynth.packs.logistics import (
    BOL_FIELDS, DESK_CARD, POD_FIELDS, Token, canonical_value, deskcard_card, docs_card, generate_docs,
    generate_paperwork, generate_tenders, outcome_for, paperwork_stats, paperwork_truth_card,
)
from thilisynth.packs.logistics import reference as ref
from thilisynth.packs.logistics.paperwork import (
    DAMAGE_TOKEN_RATE, N_TEMPLATES, PAPER_DEFECTS, PAPER_REASON_PRIORITY, CONSIGNEES,
)

N = 1500


@pytest.fixture(scope="module")
def tenders():
    return generate_tenders(n=N)


@pytest.fixture(scope="module")
def loads(tenders):
    return generate_paperwork([r.record for r in tenders.rows])


def test_same_seed_same_documents(tenders):
    recs = [r.record for r in tenders.rows[:60]]
    a = [dataclasses.asdict(l) for l in generate_paperwork(recs, seed=5)]
    b = [dataclasses.asdict(l) for l in generate_paperwork(recs, seed=5)]
    assert a == b
    assert a != [dataclasses.asdict(l) for l in generate_paperwork(recs, seed=6)]


def test_documents_for_a_load_do_not_depend_on_how_many_loads_there_are(tenders):
    recs = [r.record for r in tenders.rows]
    small = generate_paperwork(recs[:40])
    big = generate_paperwork(recs[:400])
    assert [dataclasses.asdict(l) for l in small] == [dataclasses.asdict(l) for l in big[:40]]


def test_a_page_shows_a_reader_nothing_but_tokens():
    assert [f.name for f in dataclasses.fields(Token)] == ["text", "x", "y", "w", "h"]


def test_every_load_is_built_from_its_tender(tenders, loads):
    for row, l in zip(tenders.rows, loads):
        rec = row.record
        assert l.load_id == rec["load_id"] == l.booking.load_id
        assert l.bol_truth.values["load_ref"] == rec["load_id"]
        assert l.bol_truth.values["pieces"] == str(rec["pieces"]) == str(l.booking.pieces)
        assert l.bol_truth.values["commodity"] == rec["commodity"]
        assert l.bol_truth.values["hs_code"] == rec["hs_code"]
        assert l.booking.weight_lb == rec["weight_lb"]
        assert l.consignee in CONSIGNEES


def test_token_and_truth_arrays_line_up(loads):
    for l in loads:
        for page, truth in ((l.bol, l.bol_truth), (l.pod, l.pod_truth)):
            n = len(page.tokens)
            assert len(truth.roles) == len(truth.fields) == len(truth.clean) == len(truth.damaged) == n
            assert set(truth.values) == set(BOL_FIELDS if page.doc_type == "bol" else POD_FIELDS)


def test_tenders_v1_is_untouched_by_this_pack():
    """The paperwork is built from the tenders and never feeds back into them."""
    a = generate_tenders(n=50).to_jsonl()
    generate_docs(n=50)
    assert generate_tenders(n=50).to_jsonl() == a


def test_undamaged_pages_read_back_exactly(loads):
    """With no damage, the tokens of a field, joined and canonicalised, give the true value."""
    checked = 0
    for l in loads:
        for page, truth in ((l.bol, l.bol_truth), (l.pod, l.pod_truth)):
            if page.damage_level != 0:
                continue
            for f, expected in truth.values.items():
                toks = [t.text for t, r, fld in zip(page.tokens, truth.roles, truth.fields) if r == "value" and fld == f]
                got = canonical_value(f, " ".join(toks))
                assert got == expected, (l.load_id, page.doc_type, f, got, expected)
                checked += 1
    assert checked > 5000


def test_the_four_layouts_put_a_value_where_its_label_says(loads):
    """Templates 0 and 3 print some values beside their labels; 1 prints them below. A real structural difference."""
    seen = {t: {"beside": 0, "below": 0} for t in range(N_TEMPLATES)}
    for l in loads[:400]:
        for page, truth in ((l.bol, l.bol_truth), (l.pod, l.pod_truth)):
            if page.damage_level != 0:
                continue
            first = {}
            for i, (r, f) in enumerate(zip(truth.roles, truth.fields)):
                if f and (r, f) not in first:
                    first[(r, f)] = page.tokens[i]
            for (r, f), tok in list(first.items()):
                if r != "label" or ("value", f) not in first:
                    continue
                v = first[("value", f)]
                if abs(v.y - tok.y) <= 2 and v.x > tok.x:
                    seen[page.template]["beside"] += 1
                elif v.y > tok.y:
                    seen[page.template]["below"] += 1
    assert seen[0]["beside"] > 0 and seen[0]["below"] == 0
    assert seen[1]["below"] > 0 and seen[1]["beside"] == 0
    assert seen[2]["below"] > 0 and seen[3]["below"] > 0 and seen[3]["beside"] > 0


def test_level_zero_is_clean_and_damage_grows_with_level(loads):
    st = paperwork_stats(loads)["damage_by_level"]
    rate = {k: st[k]["damaged"] / max(1, st[k]["tokens"]) for k in st}
    assert rate[0] == 0
    assert rate[1] < rate[2] < rate[3]
    for l in loads:
        for page, truth in ((l.bol, l.bol_truth), (l.pod, l.pod_truth)):
            if page.damage_level == 0:
                assert not any(truth.damaged) and not truth.dropped_fields


def test_a_damaged_token_really_differs_from_its_clean_text(loads):
    n_damaged = 0
    for l in loads:
        for page, truth in ((l.bol, l.bol_truth), (l.pod, l.pod_truth)):
            for tok, clean, damaged in zip(page.tokens, truth.clean, truth.damaged):
                if damaged:
                    n_damaged += 1
                    assert tok.text != clean or len(clean) >= 5      # a split keeps the text of one half only
                else:
                    assert tok.text == clean
    assert n_damaged > 500


def test_every_defect_is_present_and_its_label_is_true(loads):
    st = paperwork_stats(loads)
    for d in PAPER_DEFECTS:
        assert st["planted_defects"].get(d, 0) >= 30, d
    for l in loads:
        o = l.outcome
        if "shortage" in l.defects:
            assert int(l.pod_truth.values["pieces_received"]) < int(l.bol_truth.values["pieces"])
        if "damage_noted" in l.defects:
            assert "damaged" in l.pod_truth.values["exceptions"]
        if "weight_off" in l.defects:
            assert abs(int(l.bol_truth.values["weight_lb"]) - l.booking.weight_lb) / l.booking.weight_lb > DESK_CARD.weight_tolerance
        if "consignee_mismatch" in l.defects:
            assert l.pod_truth.values["receiver"] != l.consignee
        if "seal_mismatch" in l.defects:
            assert l.pod_truth.values["seal_number"] != l.bol_truth.values["seal_number"]
        if "unsigned" in l.defects:
            assert "unsigned" in (l.bol_truth.values["shipper_signature"], l.pod_truth.values["receiver_signature"])
        if "load_ref_mismatch" in l.defects:
            assert l.pod_truth.values["load_ref"] != l.load_id
        # a defect is a reason to hold, and nothing else is
        assert set(o.reasons) == set(l.defects), (l.load_id, l.defects, o.reasons)


def test_natural_scale_differences_inside_tolerance_do_not_hold_a_load(loads):
    inside = [l for l in loads if "weight_off" not in l.defects
              and int(l.bol_truth.values["weight_lb"]) != l.booking.weight_lb]
    assert len(inside) > 100
    assert all("weight_off" not in l.outcome.reasons for l in inside)


def test_the_outcome_comes_from_the_facts_not_the_damage(tenders):
    recs = [r.record for r in tenders.rows[:300]]
    ls = generate_paperwork(recs)
    for l in ls:
        again = outcome_for(l.booking, l.bol_truth.values, l.pod_truth.values)
        assert again == l.outcome
        # damage that changes what a reader sees cannot change the truth
        if l.bol.damage_level == 3 or l.pod.damage_level == 3:
            assert l.outcome.action == ("hold" if l.defects else "post")


def test_primary_reason_follows_the_priority_list(loads):
    for l in loads:
        if l.outcome.reasons:
            assert l.outcome.primary_reason == min(l.outcome.reasons, key=PAPER_REASON_PRIORITY.index)
        else:
            assert l.outcome.action == "post" and l.outcome.primary_reason is None


def test_canonical_value_round_trips_the_printed_forms():
    assert canonical_value("weight_lb", "37,450") == "37450"
    assert canonical_value("pickup_date", "09/03/2026") == "2026-09-03"
    assert canonical_value("pickup_date", "03 Sep 2026") == "2026-09-03"
    assert canonical_value("pickup_date", "2026-09-03") == "2026-09-03"
    assert canonical_value("shipper_signature", "/s/ J. Ramirez") == "signed"
    assert canonical_value("shipper_signature", "") == "unsigned"


def test_the_desk_card_is_consistent():
    card = DESK_CARD
    assert 0 < card.weight_tolerance < 0.2 and card.accuracy_floor == 0.995 and card.review_cost_usd > 0
    for key in [f"bol.{f}" for f in BOL_FIELDS] + [f"pod.{f}" for f in POD_FIELDS]:
        assert card.cost(key) > 0 and card.tier[key] in ("critical", "high", "medium", "low")
    assert card.cost("bol.load_ref") > card.cost("bol.weight_lb") > card.cost("bol.commodity")


def test_the_cards_state_what_the_data_does_not_contain(loads):
    for text in (docs_card(loads, 1), deskcard_card(DESK_CARD), paperwork_truth_card(loads)):
        assert "does NOT contain" in text
    assert "not an image" in docs_card(loads, 1).lower() or "not sampled from real scanners" in docs_card(loads, 1)
    assert "illustrative" in deskcard_card(DESK_CARD)


def test_the_weight_tolerance_belongs_to_the_card_and_changes_the_truth(tenders):
    recs = [r.record for r in tenders.rows[:600]]
    normal = generate_paperwork(recs)
    tight = generate_paperwork(recs, card=dataclasses.replace(DESK_CARD, weight_tolerance=0.001))
    newly_held = [a for a, b in zip(normal, tight) if a.outcome.action == "post" and b.outcome.action == "hold"]
    assert len(newly_held) > 20
    assert all(b.outcome.primary_reason == "weight_off" for a, b in zip(normal, tight)
               if a.outcome.action == "post" and b.outcome.action == "hold")
    # the documents themselves do not depend on the card, only the verdict does
    assert [dataclasses.asdict(a.bol) for a in normal] == [dataclasses.asdict(b.bol) for b in tight]


def test_every_hold_reason_has_a_missed_hold_cost():
    assert set(DESK_CARD.missed_hold_cost_usd) == set(PAPER_REASON_PRIORITY)
    assert all(v > 0 for v in DESK_CARD.missed_hold_cost_usd.values())
