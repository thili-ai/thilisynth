# thilisynth

Every industry AI project starts at the same wall: **the data is confidential and there is no public
substitute.** There is no open corpus of freight load tenders, no public TMS, no dataset of insurance
first-notice-of-loss calls. So you generate one — and almost everybody generates it wrong.

The wrong way is to ask a model for "a Bill of Lading". You get a plausible document and **no label** —
nothing but the model's own opinion about what it just wrote.

`thilisynth` inverts that:

> **The LLM generates the noise. It never generates the ground truth.**

A sampler draws the structured record from distributions shaped like the real world — real freight
lanes, real HS commodity codes, real trailer payload limits. A local LLM then *renders* that record as
the messy human surface a system would actually receive: a forwarded email thread with three replies
and a typo. The record you sampled **is** the label — free, at any volume, correct by construction.

Then every render is diffed back against its record, because models lie about the data you handed them.

```
sample  ->  render  ->  VERIFY  ->  damage  ->  corpus card
  |           |           |           |
  |           |           |           `-- defects applied in code, so the flag is a
  |           |           |               label and not the model's claim
  |           |           `-- reject renders that contradict the record, BEFORE any
  |           |               deliberate damage
  |           `-- the model writes only the surface
  `-- the record is the ground truth, and it already exists
```

> ⚠️ **Educational reference, not production.** A few hundred lines of stdlib Python you can hold in
> your head. It is built to be read and modified, not deployed.

## Built lesson-by-lesson in a Learn Fast course

This is the reference implementation for **[Generate Your Own Industry Dataset](https://learnfast.thili.ai)**,
drill 1 of the Freight & Logistics track. You build it yourself; this repo is here to unblock you and
to check your work.

| Lesson | What you build | Files |
|---|---|---|
| 1–2 | The wall. Stand up Ollama, ask it for a Bill of Lading 20 times, count the distinct shippers — and ask where the label is | `examples/model_check.py` |
| 3 | Record first, text second. The `LoadTender` schema and a seeded sampler over real reference tables | `core/sample.py`, `packs/logistics/records.py`, `packs/logistics/reference.py` |
| 4 | Schema-constrained decoding — and where it stops helping | `core/schema.py`, `renderers/ollama.py` |
| 5 | Rendering the mess, then checking it against the record | `renderers/`, `core/verify.py` |
| 6 | Injecting failure modes on purpose, in code, with labels | `core/defects.py` |
| 7 | Diversity, measured — distinct-n, type-token ratio, Zipf slope | `core/measure.py` |
| 8 | Documents and OCR damage | `core/docrender.py` *(not built yet)* |
| 9 | Volume — batch generation | `renderers/vllm.py` *(not built yet)* |
| 10 | Fidelity against real documents | `core/measure.py` |
| 11 | Model collapse, reproduced on a laptop | *(notebook only)* |
| 12 | The pack — prove the core is domain-agnostic by writing a second one | `packs/` |

## Quickstart (no model, no keys, no GPU)

```bash
git clone https://github.com/thili-ai/thilisynth && cd thilisynth
python examples/generate_tenders.py -n 200
```

```python
from thilisynth.packs.logistics import generate_tenders

corpus = generate_tenders(n=500, seed=20260806)   # deterministic, no model needed
print(corpus.rows[0].record)      # the label
print(corpus.rows[0].text)        # the email a broker would receive
print(corpus.rows[0].defects)     # ['vague_pickup', 'typo']
print(corpus.defect_rates)        # what the corpus ACTUALLY contains
```

The default `ScriptedRenderer` is templates, not intelligence, and it is not pretending otherwise. It
exists so the pipeline is **reproducible** (same seed, same bytes, any machine), **testable** (every
test here runs with no model in the loop), and **runnable on an 8 GB laptop**.

## Running it for real

```bash
ollama pull qwen2.5:7b
python examples/generate_tenders.py -n 25 --renderer ollama --model qwen2.5:7b
python examples/model_check.py            # compare a small model against a big one
```

**Model floor, measured on an M3.** `qwen2.5:1.5b` is not usable for rendering. Handed
`rate_offered_usd: 1850` it wrote **"$18,500"**, turned `53ft reefer` into "53ft Rer efer", and signed
the email "Qwen Support Team". `qwen2.5:7b` renders acceptably at ~11 tok/s. Smaller models are fine
for the structured-output path, where a schema guarantees the shape anyway.

That 10× money error is why `core/verify.py` exists, and why it is not optional.

## Two things this library will teach you the hard way

**Constrained decoding guarantees the shape, never the truth.** Ollama's `format` parameter forces
valid JSON. Measured: `qwen2.5:7b` returned `weight_lb: 0` for a weight that simply was not in the
source text — because the schema marked the field *required* and non-nullable, which is an instruction
to invent a value. Make it nullable and let the model say `null`.

**A model will not plant a defect on request.** Told "never state the pickup date or time explicitly",
`qwen2.5:7b` put explicit dates in every header. Worse than unreliable, it is unverifiable: if the model
plants the defect, the flag is its claim about what it did, not your label. So `core/defects.py` damages
the *rendered text* mechanically, after verification.

## What's inside

```
thilisynth/
  core/           domain-agnostic. NEVER imports from packs/
    schema.py     dataclass -> JSON Schema; required vs nullable
    sample.py     seeded sampling, Zipf-shaped entities, split-by-entity
    verify.py     diff a render against its record; missing vs contradicted
    defects.py    the failure-mode catalogue, applied in code
    measure.py    distinct-n, type-token ratio, Zipf slope, field coverage
    pipeline.py   sample -> render -> verify -> damage -> manifest
    card.py       CORPUS_CARD.md, including what the corpus does NOT contain
  renderers/
    base.py       Renderer: record + spec -> text
    scripted.py   deterministic, no model
    ollama.py     a local model, stdlib urllib
  packs/
    logistics/    freight: LoadTender, reference tables, render specs
```

`core/` not importing from `packs/` is the load-bearing constraint. The next industry corpus —
prior-auth letters, FNOL calls, dispute tickets — is a new pack against an unchanged core.

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

Every test runs against `ScriptedRenderer`, so the suite is fast and never flaky.

## License

MIT © thili.ai
