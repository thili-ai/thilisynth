"""Build the `tenders-v1` corpus and write its corpus card.

    python examples/generate_tenders.py                      # scripted, no model
    python examples/generate_tenders.py --renderer ollama    # real local model
    python examples/generate_tenders.py --renderer ollama --model qwen2.5:1.5b -n 20

The interesting run is the comparison. The scripted renderer never contradicts its
record — templates cannot hallucinate. A real model does, at a rate you should
measure rather than assume, and `reject_rate` in the corpus card is that number.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from thilisynth.core.card import corpus_card                      # noqa: E402
from thilisynth.core.measure import field_coverage, summary       # noqa: E402
from thilisynth.packs.logistics import (                          # noqa: E402
    KNOWN_GAPS, SOURCES, TENDER_DEFECTS, generate_tenders,
)
from thilisynth.renderers.ollama import OllamaRenderer            # noqa: E402
from thilisynth.renderers.scripted import ScriptedRenderer        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--renderer", choices=["scripted", "ollama"], default="scripted")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--out", default="corpora/tenders-v1")
    args = ap.parse_args()

    if args.renderer == "ollama":
        renderer = OllamaRenderer(model=args.model, seed=args.seed)
        model_note = args.model
    else:
        renderer = ScriptedRenderer(seed=args.seed)
        model_note = None

    t0 = time.time()
    done = [0]

    def progress(i: int, n: int) -> None:
        if i % max(1, n // 20) == 0 or i == n:
            done[0] = i
            print(f"  {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"generating {args.n} tenders with {renderer.name} ({model_note or 'templates'})")
    corpus = generate_tenders(n=args.n, seed=args.seed, renderer=renderer, on_progress=progress)
    elapsed = time.time() - t0

    texts = [r.text for r in corpus.rows]
    records = [r.record for r in corpus.rows]
    stats = summary(texts, records, entity_field="shipper")
    stats["seconds"] = round(elapsed, 1)
    stats["seconds_per_row"] = round(elapsed / max(1, len(corpus.rows)), 2)
    coverage = field_coverage(
        texts, records, ["load_id", "weight_lb", "rate_offered_usd", "pieces", "equipment"]
    )
    for k, v in coverage.items():
        stats[f"coverage_{k}"] = v

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w") as fh:
        for row in corpus.to_jsonl():
            fh.write(json.dumps(row) + "\n")
    if corpus.rejected:
        with (out / "rejected.jsonl").open("w") as fh:
            for row in corpus.rejected:
                fh.write(json.dumps(row) + "\n")
    (out / "CORPUS_CARD.md").write_text(
        corpus_card(corpus, TENDER_DEFECTS, stats, SOURCES, KNOWN_GAPS, model=model_note)
    )

    print(f"\n{len(corpus.rows)} rows -> {out}/corpus.jsonl")
    print(f"rejected (contradicted their record): {len(corpus.rejected)}  "
          f"({corpus.reject_rate:.2%})")
    print(f"defects observed: {corpus.defect_rates}")
    print(f"card: {out}/CORPUS_CARD.md")
    for v in corpus.rejected[:5]:
        print(f"  REJECT {v['record']['load_id']}: {v['reason']}")


if __name__ == "__main__":
    main()
