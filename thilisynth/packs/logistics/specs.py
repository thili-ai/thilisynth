"""Render specs for the freight pack — one prompt and one script per surface.

Both describe the same artifact. Keeping them side by side is what makes the two
renderers comparable: run the corpus through `ScriptedRenderer`, then through
`OllamaRenderer`, and the diff is exactly what the model bought you — plus the
fidelity you now have to verify.
"""
from __future__ import annotations

import random
from typing import Any, Dict

from ...renderers.base import RenderSpec

# --- the real-model path ---------------------------------------------------
# No literal braces: OllamaRenderer calls .format(record=..., **record).

TENDER_PROMPT = """You write realistic freight logistics emails. You are not an assistant \
and you never mention being one.

Load tender record:
{record}

Write the email thread a freight broker would actually receive for this load.

- A forwarded thread, 3 messages, newest at top, separated by "-----Original Message-----".
- Realistic From/Sent/To/Subject headers on each message.
- Use the record's values exactly. Do not round, restate or convert any number.
- Dispatchers are terse. No marketing language, no bullet-point summaries, no signature blocks
  longer than two lines.
- Output the email thread only. No preamble, no explanation, no markdown headings.
"""


# --- the deterministic path ------------------------------------------------

_OPENERS = [
    "Got a load for you.", "Covering this one today.", "New tender, see below.",
    "Need a truck on this.", "Can you cover this one?",
]
_CARRIER_ASKS = [
    "Can you confirm the temp set point before we dispatch?",
    "Need the pickup number before the driver rolls.",
    "Is there a lumper at origin on this?",
    "Who do we call at the dock for check-in?",
    "Any chance on the rate? Tight for that lane.",
]
_CLOSERS = ["Thanks", "Thanks,", "Appreciate it", "LMK", "Regards"]


def _addr(company: str, mailbox: str) -> str:
    slug = company.split()[0].lower()
    return f"{mailbox}@{slug}.com"


def tender_thread(r: Dict[str, Any], rng: random.Random) -> str:
    """A 3-message forwarded tender thread, deterministic from the record."""
    lane = f"{r['origin_city']}, {r['origin_state']} -> {r['destination_city']}, {r['destination_state']}"
    subject = f"Load {r['load_id']} {lane}"
    broker = _addr("Redstone Brokerage", "ops")
    carrier = _addr(r["carrier_invited"], "dispatch")
    shipper = _addr(r["shipper"], "shipping")

    temp = f" Reefer at {r['temp_f']}F continuous." if r.get("temp_f") is not None else ""
    acc = ""
    if r.get("accessorials"):
        acc = f" Note {' and '.join(r['accessorials'])} may apply."

    msg3 = (
        f"From: {carrier}\n"
        f"Sent: Tuesday, September 1, 2026 4:12 PM\n"
        f"To: {broker}\n"
        f"Subject: RE: {subject}\n\n"
        f"{rng.choice(_CARRIER_ASKS)}\n\n"
        f"{rng.choice(_CLOSERS)}\n"
    )
    msg2 = (
        f"From: {broker}\n"
        f"Sent: Tuesday, September 1, 2026 2:48 PM\n"
        f"To: {carrier}\n"
        f"Subject: FW: {subject}\n\n"
        f"{rng.choice(_OPENERS)} {r['equipment']}, {r['pieces']} pallets, "
        f"{r['weight_lb']:,} lbs of {r['commodity'].lower()}.{temp}\n"
        f"Pickup {r['pickup_date']} between {r['pickup_window']}. "
        f"Offering ${r['rate_offered_usd']:,} all in.{acc}\n\n"
        f"{rng.choice(_CLOSERS)}\n"
    )
    msg1 = (
        f"From: {shipper}\n"
        f"Sent: Tuesday, September 1, 2026 11:05 AM\n"
        f"To: {broker}\n"
        f"Subject: {subject}\n\n"
        f"Tendering {r['load_id']} for {r['pickup_date']}, window {r['pickup_window']}.\n"
        f"{r['pieces']} pallets, {r['weight_lb']:,} lbs, {r['equipment']}. "
        f"Commodity is {r['commodity'].lower()} (HS {r['hs_code']}).\n"
        f"Target ${r['rate_offered_usd']:,} to {r['destination_city']}.\n\n"
        f"{rng.choice(_CLOSERS)}\n"
    )
    sep = "\n-----Original Message-----\n"
    return msg3 + sep + msg2 + sep + msg1


TENDER_EMAIL = RenderSpec(
    kind="tender_email_thread",
    prompt=TENDER_PROMPT,
    script=tender_thread,
    max_tokens=900,
)
