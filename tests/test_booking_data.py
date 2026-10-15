"""carriers-v1, ratecard-v1 and the booking ground truth: consistent with the tenders, labels true by construction."""
import dataclasses
import hashlib
import json
from datetime import date

from thilisynth.packs.logistics import (
    RATE_CARD, build_truth, generate_carriers, generate_tenders, truth_for, truth_summary,
    carriers_card, ratecard_card, truth_card,
)
from thilisynth.packs.logistics import reference as ref
from thilisynth.packs.logistics.carriers import (
    AS_OF, MIN_PER_TRAP, TRAP_RATES, carrier_ask, carrier_floor, name_similarity, observed_traps,
)
from thilisynth.packs.logistics.truth import REASON_PRIORITY, vetting_failures

# sha256 of generate_tenders(n=300, seed=20260806).to_jsonl(), taken from the code as committed before the
# booking data existed. It is what the lfsynth lessons build and check against.
EXPECTED_TENDERS_DIGEST = "36bc3134874ab8a76753cd7510918e6d469fd6e501a29a3c51eeff5efa722bfd"


def test_tenders_v1_is_unchanged_by_the_booking_data():
    """Drill 1's corpus must stay byte-for-byte what lfsynth lesson 8 builds."""
    rows = generate_tenders(n=300, seed=20260806).to_jsonl()
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert len(rows) == 300 and digest == EXPECTED_TENDERS_DIGEST


def test_same_seed_same_carriers():
    a = [dataclasses.asdict(c) for c in generate_carriers(seed=5)]
    b = [dataclasses.asdict(c) for c in generate_carriers(seed=5)]
    assert a == b
    assert a != [dataclasses.asdict(c) for c in generate_carriers(seed=6)]


def test_every_tender_carrier_resolves_to_a_row():
    carriers = generate_carriers()
    names = {c.name for c in carriers if c.lookalike_of is None}
    assert names == set(ref.CARRIERS)
    for row in generate_tenders(n=500).rows:
        assert row.record["carrier_invited"] in names


def test_ids_and_numbers_are_unique():
    carriers = generate_carriers()
    for key in ("carrier_id", "mc_number", "dot_number"):
        values = [getattr(c, key) for c in carriers]
        assert len(values) == len(set(values)), key


def test_every_trap_is_present_and_its_label_is_true():
    carriers = generate_carriers()
    seen = observed_traps(carriers)
    for trap in TRAP_RATES:
        assert seen.get(trap, 0) >= MIN_PER_TRAP, trap
    assert seen["name_collision"] == 4
    for c in carriers:
        assert ("authority_inactive" in c.traps) == (c.authority_status == "inactive")
        assert ("authority_pending" in c.traps) == (c.authority_status == "pending")
        assert ("insurance_expired" in c.traps) == (c.insurance_expiry < AS_OF.isoformat())
        assert ("no_history" in c.traps) == (c.loads_completed == 0)
        assert ("safety_unsatisfactory" in c.traps) == (c.safety_rating == "unsatisfactory")
        assert ("contact_changed" in c.traps) == (c.contact_changed_days_ago is not None)
        if "insurance_lapsing" in c.traps:
            assert AS_OF.isoformat() < c.insurance_expiry <= date(2026, 9, 19).isoformat()


def test_lookalikes_are_near_identical_but_distinct():
    carriers = {c.carrier_id: c for c in generate_carriers()}
    twins = [c for c in carriers.values() if c.lookalike_of]
    assert len(twins) == 4
    for t in twins:
        original = carriers[t.lookalike_of]
        assert t.name != original.name and name_similarity(t.name, original.name) > 0.9
        assert t.mc_number != original.mc_number
        assert t.authority_status != "active"


def test_every_equipment_type_has_carriers():
    carriers = generate_carriers()
    for name, _, _ in ref.EQUIPMENT:
        assert sum(name in c.equipment for c in carriers) >= 5, name


def test_rate_card_agrees_with_the_tenders():
    """A tender's offered rate sits within the generator's 0.88-1.18 spread of the card's benchmark."""
    for row in generate_tenders(n=500).rows:
        r = row.record
        base = ref.RATE_PER_MILE[r["equipment"]] * r["miles"]
        assert 0.88 * base - 25 <= r["rate_offered_usd"] <= 1.18 * base + 25
        assert abs(RATE_CARD.benchmark_buy(r["equipment"], r["miles"]) - base) <= 13


def test_margin_arithmetic():
    assert RATE_CARD.max_buy_for_margin(5000) == 4800          # 4% floor, to the $25 below
    assert RATE_CARD.margin(5000, 4800) == 0.04
    assert RATE_CARD.buy_ceiling("53ft dry van", 1000) > RATE_CARD.benchmark_buy("53ft dry van", 1000)


def test_truth_is_deterministic_and_leaves_no_reason_empty():
    carriers = generate_carriers()
    corpus = generate_tenders(n=400)
    a, b = build_truth(corpus.rows, carriers), build_truth(corpus.rows, carriers)
    assert a == b
    for o in a:
        assert (o.action == "book") == (not o.reasons)
        assert (o.action == "book") == (o.expected_bookings == 1)
        assert o.primary_reason == (o.reasons[0] if o.reasons else None)
        assert o.reasons == sorted(set(o.reasons), key=REASON_PRIORITY.index)


def test_flags_map_to_the_expected_reasons():
    carriers = generate_carriers()
    row = generate_tenders(n=5, seed=3, defect_rates={}).rows[0]
    for flag, reason in [("buried_hazmat", "hazmat"), ("contradictory_reply", "thread_contradiction"),
                         ("unit_confusion", "weight_implausible"), ("vague_pickup", "date_unresolvable")]:
        assert reason in truth_for(row.record, [flag], carriers).reasons
    assert truth_for(row.record, ["typo"], carriers).reasons == truth_for(row.record, [], carriers).reasons
    assert truth_for(row.record, ["duplicate_message"], carriers).deliveries == 2


def test_acceptable_carriers_pass_vetting_and_the_price():
    carriers = {c.carrier_id: c for c in generate_carriers()}
    for row, o in zip(*[(corpus := generate_tenders(n=600)).rows, build_truth(corpus.rows, list(carriers.values()))]):
        for cid in o.acceptable_carriers:
            c = carriers[cid]
            assert not vetting_failures(c, row.record["pickup_date"])
            assert carrier_floor(c, row.record["miles"]) <= o.max_buy_usd
            assert not c.lookalike_of


def test_trap_carriers_are_never_acceptable_for_the_trap_they_carry():
    carriers = {c.carrier_id: c for c in generate_carriers()}
    corpus = generate_tenders(n=1500)
    for row, o in zip(corpus.rows, build_truth(corpus.rows, list(carriers.values()))):
        for cid in o.acceptable_carriers:
            traps = set(carriers[cid].traps)
            assert not traps & {"authority_inactive", "authority_pending", "insurance_expired",
                                "safety_unsatisfactory", "no_history", "name_collision", "contact_changed"}


def test_both_outcomes_and_every_reason_are_common():
    carriers = generate_carriers()
    s = truth_summary(build_truth(generate_tenders(n=3000).rows, carriers))
    assert 0.25 <= s["book_share"] <= 0.75
    for reason in REASON_PRIORITY:
        assert s["any_reason_counts"].get(reason, 0) >= 50, reason


def test_cards_render_and_state_their_limits():
    carriers = generate_carriers()
    corpus = generate_tenders(n=300)
    outcomes = build_truth(corpus.rows, carriers)
    for text in (carriers_card(carriers, 20260806), ratecard_card(RATE_CARD), truth_card(outcomes, RATE_CARD)):
        assert text.startswith("# Corpus Card") and "does NOT contain" in text


def test_a_carrier_will_come_down_but_not_below_its_floor():
    for c in generate_carriers():
        for miles in (300, 1200, 2000):
            ask, floor = carrier_ask(c, miles), carrier_floor(c, miles)
            if c.rate_history:
                assert floor <= ask and floor >= int(ask * 0.94) // 25 * 25
            else:
                assert ask is None and floor is None
