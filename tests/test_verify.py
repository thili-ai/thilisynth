"""The verification loop — the module that exists because a model got it wrong.

The canonical case, reproduced here as a test: qwen2.5:1.5b was handed
`rate_offered_usd: 1850` and wrote "$18,500" into the email. If that row enters the
corpus, its label says 1850 and its text says 18500, and nothing downstream can
tell.
"""
from thilisynth.core.verify import verify

RECORD = {
    "load_id": "TNDR-48812",
    "origin_city": "Fresno",
    "destination_city": "Reno",
    "equipment": "53ft reefer",
    "weight_lb": 41200,
    "rate_offered_usd": 1850,
}
KW = {
    "must_appear": ["load_id", "origin_city", "destination_city", "equipment"],
    "money_fields": ["rate_offered_usd"],
    "weight_fields": ["weight_lb"],
}


def test_clean_render_passes():
    text = (
        "Subject: Load TNDR-48812 Fresno CA -> Reno NV\n"
        "53ft reefer, 41,200 lbs. Offering $1,850 all in."
    )
    assert verify(text, RECORD, **KW).ok


def test_catches_the_ten_x_money_error():
    """The measured 1.5B failure: $1,850 rendered as $18,500."""
    text = (
        "Subject: Load TNDR-48812 Fresno CA -> Reno NV\n"
        "53ft reefer, 41,200 lbs. Offering $18,500 all in."
    )
    report = verify(text, RECORD, **KW)
    assert not report.ok
    assert len(report.contradictions) == 1
    assert "18,500" in str(report)


def test_catches_a_wrong_weight():
    text = "Load TNDR-48812 Fresno to Reno, 53ft reefer, 4,120 lbs, $1,850."
    report = verify(text, RECORD, **KW)
    assert any(v.field == "weight_lb" for v in report.contradictions)


def test_an_omitted_value_is_not_a_contradiction():
    """Real emails omit things. That is realism, not corruption.

    Weight and rate never appear here. They are not in `must_appear` — only
    identifiers and places are — so nothing is missing, and since no money or
    weight token appears at all there is nothing to contradict. The render passes,
    and it should: an email that says "call me on rate" is a normal email.
    """
    text = "Load TNDR-48812 Fresno to Reno. 53ft reefer. Call me on rate."
    report = verify(text, RECORD, **KW)
    assert report.ok
    assert report.contradictions == []


def test_an_omitted_must_appear_field_is_missing():
    text = "Load TNDR-48812 Fresno to Reno. Call me on rate."   # no equipment
    report = verify(text, RECORD, **KW)
    assert not report.ok
    assert {v.field for v in report.missing} == {"equipment"}


def test_may_be_absent_exempts_a_field():
    text = "Load TNDR-48812 Fresno to Reno. $1,850. 41,200 lbs."
    assert verify(text, RECORD, may_be_absent=["equipment"], **{**KW, "must_appear": ["equipment"]}).ok


def test_equipment_corruption_is_caught():
    """'53ft reefer' -> '53ft Rer efer', also measured on 1.5B."""
    text = "Load TNDR-48812 Fresno to Reno, 53ft Rer efer, 41,200 lbs, $1,850."
    report = verify(text, RECORD, **KW)
    assert not report.ok
    assert any(v.field == "equipment" and v.kind == "missing" for v in report.violations)
