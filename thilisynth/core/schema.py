"""Record schemas, and the JSON Schema we hand a model to constrain its decoding.

A record is a plain dataclass. `json_schema()` turns one into the JSON Schema that
Ollama's `format` parameter (or Outlines, or a GBNF grammar) uses to guarantee the
model emits the right *shape*.

The distinction this module exists to make explicit — and the one lesson 4 is built
on — is **required vs nullable**:

    required + not nullable  ->  the model MUST produce a value.
                                 If the source text does not contain one, it will
                                 invent it. Measured: qwen2.5:7b returned
                                 `weight_lb: 0` for a weight that was simply absent.

    required + nullable      ->  the model may say `null`, which is the honest
                                 answer and the one you want.

Constrained decoding guarantees the shape. It never guarantees the truth.
"""
from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, List, Optional, Type

_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


def _json_type(py_type: Any) -> str:
    """Map a Python annotation to a JSON Schema type name."""
    # Unwrap Optional[X] / X | None, which dataclass fields use for absent values.
    args = getattr(py_type, "__args__", None)
    if args:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if non_none:
            py_type = non_none[0]
    origin = typing.get_origin(py_type) or py_type
    return _JSON_TYPES.get(origin, "string")


def json_schema(
    record_cls: Type,
    fields: Optional[List[str]] = None,
    nullable: Optional[List[str]] = None,
    required: Optional[List[str]] = None,
    descriptions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a JSON Schema for `record_cls`, or for a subset of its fields.

    Args:
        fields:      which dataclass fields to include (default: all).
        nullable:    fields whose type becomes ["<type>", "null"] so the model can
                     honestly report "not present in the source".
        required:    fields the model must emit (default: all included fields).
                     Anything required and NOT nullable is a licence to hallucinate.
        descriptions: per-field hints; models do read these.
    """
    all_names = [f.name for f in dataclasses.fields(record_cls)]
    names = fields or all_names
    unknown = [n for n in names if n not in all_names]
    if unknown:
        raise ValueError(f"{record_cls.__name__} has no field(s): {unknown}")

    nullable = set(nullable or [])
    descriptions = descriptions or {}
    # get_type_hints, not `field.type` — under `from __future__ import annotations`
    # (which every module here uses) `field.type` is the *string* "int", so a naive
    # lookup silently types every field as "string" and the schema stops
    # constraining anything.
    try:
        types = typing.get_type_hints(record_cls)
    except Exception:                                   # pragma: no cover
        types = {f.name: f.type for f in dataclasses.fields(record_cls)}

    props: Dict[str, Any] = {}
    for name in names:
        jtype: Any = _json_type(types[name])
        if name in nullable:
            jtype = [jtype, "null"]
        spec: Dict[str, Any] = {"type": jtype}
        if name in descriptions:
            spec["description"] = descriptions[name]
        props[name] = spec

    return {
        "type": "object",
        "properties": props,
        "required": list(required if required is not None else names),
    }


def as_dict(record: Any) -> Dict[str, Any]:
    """A record as a plain dict — what gets written to the corpus as the label."""
    return dataclasses.asdict(record)
