"""The pipeline: reproducibility, the sample-render-verify-damage order, and labels."""
import random

from thilisynth.core.defects import apply_defects, observed_rates
from thilisynth.core.sample import Sampler, split_by
from thilisynth.packs.logistics import (
    TENDER_DEFECTS, extraction_schema, generate_tenders,
)


def test_same_seed_same_corpus():
    """A corpus a grader cannot regenerate is not a deliverable."""
    a = generate_tenders(n=25, seed=7)
    b = generate_tenders(n=25, seed=7)
    assert [r.text for r in a.rows] == [r.text for r in b.rows]
    assert [r.defects for r in a.rows] == [r.defects for r in b.rows]


def test_different_seed_different_corpus():
    a = generate_tenders(n=25, seed=7)
    b = generate_tenders(n=25, seed=8)
    assert [r.text for r in a.rows] != [r.text for r in b.rows]


def test_scripted_renderer_never_contradicts_its_record():
    """Templates cannot hallucinate. That is the point of having them."""
    c = generate_tenders(n=100, seed=11)
    assert c.rejected == []
    assert c.reject_rate == 0.0


def test_records_are_internally_consistent():
    c = generate_tenders(n=100, seed=3)
    for row in c.rows:
        r = row.record
        assert r["origin_city"] != r["destination_city"]
        assert r["weight_lb"] > 0
        assert 100 <= r["miles"] <= 3200
        rate_per_mile = r["rate_offered_usd"] / r["miles"]
        assert 1.4 < rate_per_mile < 4.0, f"implausible rate: ${rate_per_mile:.2f}/mi"
        # A reefer commodity must travel in reefer equipment.
        if r["temp_f"] is not None:
            assert "reefer" in r["equipment"]


def test_defect_flags_describe_the_text():
    """The flag is a label, not a claim — so it must be true of the bytes."""
    c = generate_tenders(n=150, seed=5)
    for row in c.rows:
        if "buried_hazmat" in row.defects:
            assert "UN1950" in row.text
        if "duplicate_message" in row.defects:
            assert "RESEND" in row.text
        if "contradictory_reply" in row.defects:
            assert "Ignore my last" in row.text


def test_observed_rates_track_configured_rates():
    c = generate_tenders(n=400, seed=13)
    observed = c.defect_rates
    for name, configured in TENDER_DEFECTS.items():
        assert name in observed, f"{name} configured but never fired"
        assert abs(observed[name] - configured) < 0.10, f"{name}: {observed[name]} vs {configured}"


def test_a_defect_that_never_fires_is_not_reported():
    """Silence beats a false claim: the manifest must not promise a missing failure mode."""
    rng = random.Random(0)
    record = {"weight_lb": None, "pieces": None}
    damaged = apply_defects("no numbers here", record, rng, {"unit_confusion": 1.0})
    assert damaged.flags == []
    assert observed_rates([damaged.flags]) == {}


def test_extraction_schema_lets_the_model_say_null():
    """The fix for the measured `weight_lb: 0` hallucination."""
    schema = extraction_schema()
    assert schema["properties"]["weight_lb"]["type"] == ["integer", "null"]
    assert schema["properties"]["load_id"]["type"] == "string"   # identifiers stay required


def test_split_is_by_shipper_not_by_row():
    c = generate_tenders(n=200, seed=17)
    records = [r.record for r in c.rows]
    train, test = split_by(records, "shipper", holdout=0.25, seed=1)
    train_shippers = {r["shipper"] for r in train}
    test_shippers = {r["shipper"] for r in test}
    assert train_shippers.isdisjoint(test_shippers)
    assert test


def test_zipf_is_actually_skewed():
    s = Sampler(42)
    draws = [s.zipf(list("abcdefghij")) for _ in range(1000)]
    counts = sorted(s.distribution(draws).values(), reverse=True)
    assert counts[0] > counts[-1] * 3, "zipf sampling produced a flat distribution"


def test_vague_pickup_leaves_email_headers_alone():
    """A dispatcher being vague does not make the mail server vague.

    An earlier version rewrote every date in the document, producing
    "Sent: early in the week, morning preferred, September 1, 2026" — a header no
    mail client has ever generated, and an instant tell that the corpus is fake.
    """
    import dataclasses
    from thilisynth.core.defects import vague_pickup
    from thilisynth.core.sample import Sampler
    from thilisynth.packs.logistics.records import sample_tender
    from thilisynth.packs.logistics.specs import TENDER_EMAIL
    from thilisynth.renderers.scripted import ScriptedRenderer

    record = dataclasses.asdict(sample_tender(Sampler(20260806), 0))
    clean = ScriptedRenderer(seed=1).render(record, TENDER_EMAIL)
    damaged, flag = vague_pickup(clean, record, random.Random(0))

    assert flag == "vague_pickup"
    sent = [l for l in damaged.splitlines() if l.startswith("Sent:")]
    assert sent, "the thread lost its headers"
    for line in sent:
        assert "week" not in line, f"header was rewritten: {line}"
        assert ":" in line.split("Sent:")[1], f"header lost its clock time: {line}"

    body = "\n".join(l for l in damaged.splitlines() if not l.startswith(("From:", "To:", "Sent:", "Subject:")))
    assert record["pickup_date"] not in body
    assert record["pickup_window"] not in body
    assert "between -" not in body and "window -" not in body, "punctuation debris left behind"
