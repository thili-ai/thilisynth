"""Renderers — record in, surface text out. Scripted for tests, Ollama for real."""
from .base import Renderer, RenderSpec
from .scripted import ScriptedRenderer

__all__ = ["Renderer", "RenderSpec", "ScriptedRenderer"]
