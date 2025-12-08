from __future__ import annotations

# ruff: noqa: S101 - assertions express expectations in test cases
import json
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar, cast

import pytest
from x_make_common_x.json_contracts import validate_payload, validate_schema

from x_make_py_mod_sideload_x import main_json
from x_make_py_mod_sideload_x.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)

FixtureFunc = TypeVar("FixtureFunc", bound=Callable[[], dict[str, object]])

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "json_contracts"


def _round_trip(data: dict[str, object]) -> dict[str, object]:
    raw_obj = cast("object", json.loads(json.dumps(data)))
    if not isinstance(raw_obj, Mapping):
        message = "Round-trip payload must remain a mapping"
        raise TypeError(message)
    result: dict[str, object] = {}
    for key, value in raw_obj.items():
        result[str(key)] = value
    return result


def _module_fixture(func: FixtureFunc) -> FixtureFunc:
    decorator: Callable[[FixtureFunc], object] = pytest.fixture(scope="module")
    return cast("FixtureFunc", decorator(func))


def _load_fixture(name: str) -> dict[str, object]:
    path = FIXTURE_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as handle:
        raw_payload = cast("object", json.load(handle))
    if not isinstance(raw_payload, Mapping):
        message = f"Fixture payload must be an object: {name}"
        raise TypeError(message)
    payload_dict: dict[str, object] = {}
    for key, value in raw_payload.items():
        payload_dict[str(key)] = value
    return payload_dict


def _write_module(base_dir: Path, dotted_name: str, content: str) -> Path:
    parts = dotted_name.split(".")
    if len(parts) > 1:
        package = base_dir.joinpath(*parts[:-1])
        package.mkdir(parents=True, exist_ok=True)
        init = package / "__init__.py"
        if not init.exists():
            init.write_text("\n", encoding="utf-8")
        module_name = parts[-1]
        module_path = package / f"{module_name}.py"
    else:
        module_path = base_dir / f"{dotted_name}.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return module_path


@_module_fixture
def sample_input() -> dict[str, object]:
    return _load_fixture("input")


@_module_fixture
def sample_output() -> dict[str, object]:
    return _load_fixture("output")


@_module_fixture
def sample_error() -> dict[str, object]:
    return _load_fixture("error")


def test_schemas_are_valid() -> None:
    for schema in (INPUT_SCHEMA, OUTPUT_SCHEMA, ERROR_SCHEMA):
        validate_schema(schema)


def test_sample_payloads_match_schema(
    sample_input: dict[str, object],
    sample_output: dict[str, object],
    sample_error: dict[str, object],
) -> None:
    validate_payload(sample_input, INPUT_SCHEMA)
    validate_payload(sample_output, OUTPUT_SCHEMA)
    validate_payload(sample_error, ERROR_SCHEMA)


def test_main_json_runs_successfully(
    sample_input: dict[str, object],
    tmp_path: Path,
) -> None:
    payload = _round_trip(sample_input)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    module_path = _write_module(
        workspace,
        "demo.module",
        """
        VALUE = 42

        def run() -> int:
            return VALUE
        """,
    )

    parameters_obj = payload.get("parameters")
    assert isinstance(parameters_obj, dict)
    parameters = cast("dict[str, object]", parameters_obj)
    parameters["base_path"] = str(workspace)

    result = main_json(payload)

    validate_payload(result, OUTPUT_SCHEMA)
    status_value = result.get("status")
    assert isinstance(status_value, str)
    assert status_value == "success"
    object_kind_value = result.get("object_kind")
    attribute_value = result.get("attribute")
    assert isinstance(object_kind_value, str)
    assert isinstance(attribute_value, str)
    assert object_kind_value == "attribute"
    assert attribute_value == "run"
    module_file_value = result.get("module_file")
    assert isinstance(module_file_value, str)
    assert Path(module_file_value).resolve() == module_path.resolve()
    metadata_obj = result.get("metadata")
    assert isinstance(metadata_obj, dict)
    typed_metadata: dict[str, object] = {}
    for key, value in metadata_obj.items():
        assert isinstance(key, str)
        typed_metadata[key] = cast("object", value)
    attribute_type = typed_metadata.get("attribute_type")
    assert isinstance(attribute_type, str)
    assert attribute_type == "function"


def test_main_json_reports_missing_attribute(
    sample_input: dict[str, object],
    tmp_path: Path,
) -> None:
    payload = _round_trip(sample_input)
    parameters_obj = payload.get("parameters")
    assert isinstance(parameters_obj, dict)
    parameters = cast("dict[str, object]", parameters_obj)
    parameters["attribute"] = "missing"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_module(
        workspace,
        "demo.module",
        """
        class Tool:
            pass
        """,
    )
    parameters["base_path"] = str(workspace)

    result = main_json(payload)

    validate_payload(result, ERROR_SCHEMA)
    status_value = result.get("status")
    message_value = result.get("message")
    assert isinstance(status_value, str)
    assert isinstance(message_value, str)
    assert status_value == "failure"
    assert message_value == "attribute resolution failed"


def test_main_json_reports_validation_error() -> None:
    result = main_json({"command": "x_make_py_mod_sideload_x", "parameters": {}})

    validate_payload(result, ERROR_SCHEMA)
    status_value = result.get("status")
    message_value = result.get("message")
    assert isinstance(status_value, str)
    assert isinstance(message_value, str)
    assert status_value == "failure"
    assert message_value == "input payload failed validation"
