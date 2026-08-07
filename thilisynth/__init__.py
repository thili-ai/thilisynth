"""thilisynth — generate an industry dataset when the real data is confidential.

The rule the whole library is built on:

    The LLM generates the noise. It never generates the ground truth.

Sample the record first, from distributions shaped like the real world. Then have a
local model render it as the messy human surface a system would actually receive.
The record you sampled *is* the label — free, at any volume, and correct by
construction. Ask a model for "a Bill of Lading" instead and you have a plausible
document and no label at all.

    from thilisynth.packs.logistics import generate_tenders
    corpus = generate_tenders(n=500)          # deterministic, no model needed

`core/` is domain-agnostic and must never import from `packs/`. That split is what
makes the next industry corpus cheap.
"""
from .core.card import corpus_card
from .core.pipeline import Corpus, Row, generate
from .core.sample import Sampler, split_by
from .core.schema import json_schema
from .core.verify import verify, VerifyReport
from .renderers.base import Renderer, RenderSpec
from .renderers.scripted import ScriptedRenderer

__version__ = "0.1.0"

__all__ = [
    "generate", "Corpus", "Row", "Sampler", "split_by", "json_schema",
    "verify", "VerifyReport", "Renderer", "RenderSpec", "ScriptedRenderer",
    "corpus_card", "__version__",
]
