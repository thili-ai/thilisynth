"""The real backend: a local Ollama server, over stdlib urllib (no dependencies).

Local on purpose. Freight data is customer-confidential and rate cards are
commercially sensitive — a 3PL cannot post either to a hosted API, which is exactly
why production stacks in this sector run on-prem. It also means the drill costs the
student nothing to run.

    ollama pull qwen2.5:7b

**Model floor, measured 2026-08-06 on an M3.** `qwen2.5:1.5b` is not usable for
rendering: handed `rate_offered_usd: 1850` it wrote "$18,500", mangled "53ft reefer"
into "53ft Rer efer", and signed the email as "Qwen Support Team". `qwen2.5:7b`
renders acceptably at ~11 tok/s. Smaller models are fine for the structured-output
path (`generate_json`), where the schema guarantees the shape anyway.

Whatever you use, the output goes through `core.verify` before it enters a corpus.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .base import Renderer, RenderSpec


DEFAULT_HOST = "http://127.0.0.1:11434"


def default_host() -> str:
    """The Ollama address: `OLLAMA_HOST` if set (the variable the `ollama` CLI itself uses), else localhost.

    Inside a container, set it to http://host.docker.internal:11434 once, and no notebook needs editing.
    A bare "host:port" gets an http:// in front of it.
    """
    host = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
    return host if "://" in host else f"http://{host}"


class OllamaRenderer(Renderer):
    """Render surface text with a local Ollama model."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: Optional[str] = None,
        temperature: float = 0.8,
        seed: Optional[int] = None,
        timeout: int = 600,
    ):
        self.model = model
        self.host = (host or default_host()).rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.last_stats: Dict[str, Any] = {}

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it with `ollama serve` and `ollama pull {self.model}`. ({e})"
            ) from e

    def complete(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: Optional[float] = None,
    ) -> str:
        """Free-form completion. Records tokens and seconds in `last_stats`.

        This is the plain "talk to the model" entry point; `render()` is the same
        call with a record formatted into a pack's prompt template.
        """
        options: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": max_tokens,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        out = self._post({
            "model": self.model, "prompt": prompt, "stream": False, "options": options,
        })
        tokens = out.get("eval_count", 0)
        seconds = round(out.get("eval_duration", 0) / 1e9, 3)
        self.last_stats = {
            "tokens": tokens,
            "seconds": seconds,
            "tokens_per_second": round(tokens / seconds, 1) if seconds else 0.0,
        }
        return out["response"].strip()

    def render(self, record: Dict[str, Any], spec: RenderSpec) -> str:
        if not spec.prompt:
            raise ValueError(f"RenderSpec(kind={spec.kind!r}) has no `prompt`.")
        prompt = spec.prompt.format(record=json.dumps(record, indent=2), **record)
        return self.complete(prompt, max_tokens=spec.max_tokens)

    def generate_json(self, prompt: str, schema: Dict[str, Any], max_tokens: int = 400) -> Dict[str, Any]:
        """Schema-constrained decoding — the shape is guaranteed, the truth is not.

        Ollama's `format` parameter takes a JSON Schema and forces valid output. It
        will not make the content correct: measured, qwen2.5:7b returned
        `weight_lb: 0` for a weight that was simply absent from the source, because
        the schema marked the field required and non-nullable. Make fields nullable
        and let the model say `null`.
        """
        out = self._post({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        })
        tokens = out.get("eval_count", 0)
        seconds = round(out.get("eval_duration", 0) / 1e9, 3)
        self.last_stats = {
            "tokens": tokens,
            "seconds": seconds,
            "tokens_per_second": round(tokens / seconds, 1) if seconds else 0.0,
        }
        return json.loads(out["response"])
