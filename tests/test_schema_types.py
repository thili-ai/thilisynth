from typing import List, Optional

from thilisynth.core.schema import _json_type, json_schema
from thilisynth.packs.logistics.records import LoadTender


def test_scalars_and_optionals():
    assert _json_type(str) == "string"
    assert _json_type(int) == "integer"
    assert _json_type(Optional[int]) == "integer"
    assert _json_type(Optional[str]) == "string"


def test_list_is_an_array_not_a_string():
    assert _json_type(List[str]) == "array"
    assert json_schema(LoadTender, fields=["accessorials"])["properties"]["accessorials"]["type"] == "array"
