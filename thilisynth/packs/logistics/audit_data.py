"""Build the billing desk's datasets in one command.

    python -m thilisynth.packs.logistics.audit_data --out corpora/audit-v1

Writes `invoices-v1` (rate confirmations, gate logs, invoices) and its truth, `auditdesk-rules-v1`, each with its corpus card.
Everything is reproducible from the seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from . import AUDIT_CARD, audit_truth_card, auditrules_card, generate_invoices, invoices_card


def build(out: Path, seed: int = 20260806, n_loads: int = 2000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    loads = generate_invoices(n=n_loads, seed=seed)
    with (out / "invoices.jsonl").open("w") as fh:
        for l in loads:
            fh.write(json.dumps({
                "load_id": l.load_id, "rate_confirmation": dataclasses.asdict(l.rate_confirmation),
                "gate_log": [dataclasses.asdict(g) for g in l.gate_log], "delivered_on": l.delivered_on,
                "invoices": [dataclasses.asdict(i) for i in l.invoices],
            }) + "\n")
    with (out / "truth.jsonl").open("w") as fh:
        for l in loads:
            fh.write(json.dumps({"load_id": l.load_id, "defects": l.defects,
                                 "invoices": [dataclasses.asdict(t) for t in l.truths]}) + "\n")
    (out / "auditdesk_rules.json").write_text(json.dumps(dataclasses.asdict(AUDIT_CARD), indent=2))
    (out / "INVOICES_CARD.md").write_text(invoices_card(loads, seed))
    (out / "AUDITDESK_RULES_CARD.md").write_text(auditrules_card(AUDIT_CARD))
    (out / "AUDIT_TRUTH_CARD.md").write_text(audit_truth_card(loads))
    return {"loads": len(loads), "invoices": sum(len(l.invoices) for l in loads), "out": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build invoices-v1, auditdesk-rules-v1 and the audit ground truth.")
    p.add_argument("--out", default="corpora/audit-v1")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--loads", type=int, default=2000)
    a = p.parse_args()
    print(build(Path(a.out), a.seed, a.loads))


if __name__ == "__main__":
    main()
