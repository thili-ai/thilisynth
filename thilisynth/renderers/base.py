"""The Renderer interface — the one seam between the corpus and the LLM.

A renderer turns a **record** (the ground truth, already sampled) into the messy
human surface a system would really receive: an email thread, a phone transcript,
a scanned form.

That direction is the whole design. The record is the label; the renderer only
writes the wrapping. Generate it the other way around — ask a model for "a Bill of
Lading" — and you have a plausible document and no label at all, only the model's
opinion about what it just wrote.

Everything downstream calls `render(record, spec) -> str` and nothing else, which
is why a deterministic `ScriptedRenderer` can stand in for a real model with no
other code noticing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Renderer(ABC):
    """Anything that turns a record into surface text."""

    name = "renderer"

    @abstractmethod
    def render(self, record: Dict[str, Any], spec: "RenderSpec") -> str:
        raise NotImplementedError


class RenderSpec:
    """What surface to produce, and how.

    A pack supplies these. `prompt` drives a real model; `script` drives the
    deterministic one. Both describe the same artifact, which is what keeps the two
    paths comparable.
    """

    def __init__(self, kind: str, prompt: str = "", script: Any = None, max_tokens: int = 900):
        self.kind = kind          # e.g. "tender_email_thread"
        self.prompt = prompt
        self.script = script
        self.max_tokens = max_tokens
