"""A deterministic renderer — no model, no network, no GPU.

Borrowed wholesale from `thili-ai/thilicodex`'s `ScriptedModel`. It exists for
three reasons, in order of importance:

1. **Reproducibility.** A corpus a grader cannot regenerate is not a deliverable.
   Same seed, same bytes, on any machine, forever.
2. **Correctness.** Every test in this repo runs against it, so the pipeline —
   sampling, verification, defect injection, measurement — is tested with no model
   in the loop and no flakiness.
3. **Hardware.** Rendering well needs a 7B model (measured: 1.5B corrupts the
   fields it was handed). A student on an 8 GB laptop completes the whole drill
   through this renderer and runs the real one on a slice.

It is templates, not intelligence, and it is not trying to hide that. The point of
comparing it with `OllamaRenderer` is to see exactly what the model buys you — and
what it costs you in fidelity you now have to verify.
"""
from __future__ import annotations

import random
from typing import Any, Dict

from .base import Renderer, RenderSpec


class ScriptedRenderer(Renderer):
    """Render from templates supplied by the pack, seeded per record."""

    name = "scripted"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def render(self, record: Dict[str, Any], spec: RenderSpec) -> str:
        if spec.script is None:
            raise ValueError(
                f"RenderSpec(kind={spec.kind!r}) has no `script`; "
                "ScriptedRenderer needs one. Packs must supply both a prompt and a script."
            )
        # Seed off the record so the same record always renders the same way,
        # independently of how many records were generated before it.
        rng = random.Random(f"{self.seed}:{record.get('load_id', '')}:{spec.kind}")
        return spec.script(record, rng).strip()
