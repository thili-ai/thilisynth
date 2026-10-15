"""Build the booking desk's datasets in one command.

    python -m thilisynth.packs.logistics.booking_data --out corpora/booking-v1

Writes `carriers-v1`, `ratecard-v1` and the ground truth for `tenders-v1`, each with its corpus card.
Everything is reproducible from the seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from . import (
    RATE_CARD, build_truth, carriers_card, generate_carriers, generate_tenders, ratecard_card, truth_card,
)


def build(out: Path, seed: int = 20260806, n_tenders: int = 5000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    carriers = generate_carriers(seed=seed)
    corpus = generate_tenders(n=n_tenders, seed=seed)
    outcomes = build_truth(corpus.rows, carriers)

    with (out / "carriers.jsonl").open("w") as fh:
        for c in carriers:
            fh.write(json.dumps(dataclasses.asdict(c)) + "\n")
    (out / "ratecard.json").write_text(json.dumps(dataclasses.asdict(RATE_CARD), indent=2))
    with (out / "truth.jsonl").open("w") as fh:
        for row, o in zip(corpus.rows, outcomes):
            fh.write(json.dumps({"load_id": row.record["load_id"], **dataclasses.asdict(o)}) + "\n")

    (out / "CARRIERS_CARD.md").write_text(carriers_card(carriers, seed))
    (out / "RATECARD_CARD.md").write_text(ratecard_card(RATE_CARD))
    (out / "TRUTH_CARD.md").write_text(truth_card(outcomes, RATE_CARD))
    return {"carriers": len(carriers), "tenders": len(corpus.rows), "out": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build carriers-v1, ratecard-v1 and the booking ground truth.")
    p.add_argument("--out", default="corpora/booking-v1")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--tenders", type=int, default=5000)
    a = p.parse_args()
    print(build(Path(a.out), a.seed, a.tenders))


if __name__ == "__main__":
    main()
