"""Build the paperwork desk's datasets in one command.

    python -m thilisynth.packs.logistics.paperwork_data --out corpora/paperwork-v1

Writes `docs-v1` (pages and their truth), `paperwork-rules-v1` and the outcome truth, each with its corpus card.
Everything is reproducible from the seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from . import DESK_CARD, deskcard_card, docs_card, generate_docs, paperwork_truth_card


def build(out: Path, seed: int = 20260806, n_loads: int = 2000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    loads = generate_docs(n=n_loads, seed=seed)
    with (out / "docs.jsonl").open("w") as fh:
        for l in loads:
            fh.write(json.dumps({
                "load_id": l.load_id, "shipper": l.shipper, "consignee": l.consignee,
                "booking": dataclasses.asdict(l.booking),
                "bol": dataclasses.asdict(l.bol), "pod": dataclasses.asdict(l.pod),
            }) + "\n")
    with (out / "truth.jsonl").open("w") as fh:
        for l in loads:
            fh.write(json.dumps({
                "load_id": l.load_id, "defects": l.defects, "outcome": dataclasses.asdict(l.outcome),
                "bol": dataclasses.asdict(l.bol_truth), "pod": dataclasses.asdict(l.pod_truth),
            }) + "\n")
    (out / "paperwork_rules.json").write_text(json.dumps(dataclasses.asdict(DESK_CARD), indent=2))
    (out / "DOCS_CARD.md").write_text(docs_card(loads, seed))
    (out / "PAPERWORK_RULES_CARD.md").write_text(deskcard_card(DESK_CARD))
    (out / "PAPERWORK_TRUTH_CARD.md").write_text(paperwork_truth_card(loads))
    return {"loads": len(loads), "out": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build docs-v1, paperwork-rules-v1 and the paperwork ground truth.")
    p.add_argument("--out", default="corpora/paperwork-v1")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--loads", type=int, default=2000)
    a = p.parse_args()
    print(build(Path(a.out), a.seed, a.loads))


if __name__ == "__main__":
    main()
