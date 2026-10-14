"""The freight pack — everything domain-specific about `tenders-v1` lives here.

`thilisynth.core` must never import from a pack. That constraint is the whole
reason this library is worth more than the drill it was written for: the next
industry corpus (prior-auth letters, FNOL calls, dispute tickets) is a new pack
against an unchanged core.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

from ...core.pipeline import Corpus, generate
from ...core.sample import Sampler
from ...renderers.base import Renderer
from ...renderers.scripted import ScriptedRenderer
from . import reference as ref
from .records import (
    EXTRACTION_DESCRIPTIONS,
    EXTRACTION_FIELDS,
    EXTRACTION_NULLABLE,
    LoadTender,
    sample_tender,
)
from .specs import TENDER_EMAIL

# Which record fields `core.verify` checks, and how.
TENDER_VERIFY = {
    "must_appear": ["load_id", "origin_city", "destination_city", "equipment"],
    "money_fields": ["rate_offered_usd"],
    "weight_fields": ["weight_lb"],
}

# Defect rates for tenders-v1. Tuned so every downstream failure mode is present
# often enough to measure, and rare enough that a clean tender is still the norm.
TENDER_DEFECTS = {
    "vague_pickup": 0.30,
    "unit_confusion": 0.08,
    "duplicate_message": 0.10,
    "contradictory_reply": 0.12,
    "buried_hazmat": 0.06,
    "typo": 0.35,
}

SOURCES = [
    "US freight hubs (real cities, tonnage-ordered) — `reference.CITIES`",
    "HS commodity headings (WCO Harmonized System) — `reference.COMMODITIES`",
    "Trailer types and payload limits — `reference.EQUIPMENT`",
    "Company names — **synthetic**, built from real naming patterns, not copied from FMCSA",
]

KNOWN_GAPS = [
    "No real load-tender corpus exists publicly, so email-thread fidelity is unvalidated.",
    "Company names are synthetic; carrier-level realism (DOT/MC numbers, fleet size, "
    "safety rating) needs the FMCSA Company Census and is not loaded yet.",
    "US truckload conventions only. Indian freight (LR copies, e-way bills, GST) is absent.",
    "Single-leg loads only — no multi-stop, no intermodal, no cross-dock.",
    "Mileage is great-circle distance x 1.17, which under-reads mountain routes: Fresno to Reno "
    "comes out near 226 miles against about 340 by road.",
]


def generate_tenders(
    n: int = 500,
    seed: int = 20260806,
    renderer: Optional[Renderer] = None,
    defect_rates: Optional[Dict[str, float]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Corpus:
    """Build the `tenders-v1` corpus that `lflogistics-lfbooking` consumes."""
    r = renderer or ScriptedRenderer(seed=seed)
    return generate(
        name="tenders-v1",
        n=n,
        sample_fn=sample_tender,
        render_fn=lambda rec: r.render(rec, TENDER_EMAIL),
        to_dict=dataclasses.asdict,
        verify_kwargs=TENDER_VERIFY,
        defect_rates=TENDER_DEFECTS if defect_rates is None else defect_rates,
        seed=seed,
        renderer_name=r.name,
        on_progress=on_progress,
    )


def extraction_schema() -> Dict[str, Any]:
    """The JSON Schema a downstream extractor is constrained with.

    Note `EXTRACTION_NULLABLE`: every field that can legitimately be absent from an
    email is nullable, so the model can answer `null` instead of inventing a value.
    Drop that and you get `weight_lb: 0`.
    """
    from ...core.schema import json_schema
    return json_schema(
        LoadTender,
        fields=EXTRACTION_FIELDS,
        nullable=EXTRACTION_NULLABLE,
        descriptions=EXTRACTION_DESCRIPTIONS,
    )


__all__ = [
    "LoadTender", "sample_tender", "generate_tenders", "extraction_schema",
    "TENDER_EMAIL", "TENDER_DEFECTS", "TENDER_VERIFY", "SOURCES", "KNOWN_GAPS", "ref",
]
