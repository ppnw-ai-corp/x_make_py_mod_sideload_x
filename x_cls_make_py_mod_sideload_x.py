"""Minimal sideload helper with typed sideload interface."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from os import PathLike, fsdecode, fspath
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import IO, TYPE_CHECKING, NamedTuple, Protocol, cast

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec


from x_make_common_x.json_contracts import validate_payload

from x_make_py_mod_sideload_x.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)

StrPath = str | PathLike[str]

SCHEMA_VERSION = "x_make_py_mod_sideload_x.run/1.0"


class _ValidationErrorProtocol(Protocol):
    message: str
    path: tuple[object, ...]
    schema_path: tuple[object, ...]


class _JsonSchemaModule(Protocol):
    ValidationError: type[Exception]


def _load_validation_error() -> type[Exception]:
    module = cast("_JsonSchemaModule", importlib.import_module("jsonschema"))
    return module.ValidationError


ValidationErrorType = _load_validation_error()

_EMPTY_MAPPING: Mapping[str, object] = MappingProxyType({})


class _CoreInputs(NamedTuple):
    base_path: str
    module_name: str
    attribute_name: str | None
    loader_options: Mapping[str, object]


class _LoadedModule(NamedTuple):
    module: ModuleType
    module_file: str


@dataclass(slots=True)
class _AttributeResolutionContext:
    module_file: str
    metadata: dict[str, object]
    messages: list[str]


# Legacy-compatible entry point
def _resolve_module_file(base_path: StrPath, module: str) -> str:
    path_str = fspath(base_path)
    if not path_str:
        message = "base_path must be a non-empty string"
        raise ValueError(message)

    base_dir = Path(path_str)
    if not base_dir.exists():
        message = f"base_path does not exist: {base_dir}"
        raise FileNotFoundError(message)

    module_path = Path(module)
    if module_path.is_absolute() and module_path.is_file():
        return module_path.as_posix()

    candidates: list[Path] = []
    if module.endswith(".py"):
        candidates.append(base_dir / module)
    else:
        dotted_parts = module.split(".")
        if len(dotted_parts) > 1:
            *pkg_parts, mod_part = dotted_parts
            candidates.append(base_dir.joinpath(*pkg_parts, f"{mod_part}.py"))
        candidates.append(base_dir / f"{module}.py")
        candidates.append(base_dir / module / "__init__.py")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()

    message = (
        f"Cannot resolve module file for module={module} under base_path={base_dir}"
    )
    raise ImportError(message)


def _create_spec(module_file: str) -> ModuleSpec:
    spec = importlib.util.spec_from_file_location(
        f"sideload_{abs(hash(module_file))}",
        module_file,
    )
    if spec is None or spec.loader is None:
        message = f"Failed to create module spec for {module_file}"
        raise ImportError(message)
    return spec


def _load_module(base_path: StrPath, module: str) -> ModuleType:
    module_file = _resolve_module_file(base_path, module)
    spec = _create_spec(module_file)
    module_obj = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        message = f"Loader missing for module spec {spec.name}"
        raise ImportError(message)
    loader.exec_module(module_obj)
    return module_obj


def _get_attribute(module_obj: ModuleType, attr_name: str) -> object:
    if not hasattr(module_obj, attr_name):
        module_file_raw: object | None = getattr(module_obj, "__file__", None)
        if isinstance(module_file_raw, str):
            module_file = module_file_raw
        elif module_file_raw is None:
            module_file = "<unknown>"
        elif isinstance(module_file_raw, PathLike):
            module_file = fsdecode(module_file_raw)
        else:
            module_file = str(module_file_raw)
        message = (
            f"{ModuleType.__name__} loaded from "
            f"{module_file} has no attribute {attr_name!r}"
        )
        raise AttributeError(message)

    attr: object = getattr(module_obj, attr_name)
    if inspect.isclass(attr):
        attr_type = cast("type[object]", attr)
        return attr_type()
    return attr


class ModuleLoader(Protocol):
    def load_module(self, base_path: StrPath, module: str) -> ModuleType: ...

    def get_attribute(self, module_obj: ModuleType, attr_name: str) -> object: ...


class DefaultModuleLoader:
    def load_module(self, base_path: StrPath, module: str) -> ModuleType:
        return _load_module(base_path, module)

    def get_attribute(self, module_obj: ModuleType, attr_name: str) -> object:
        return _get_attribute(module_obj, attr_name)


class PyModuleSideload:
    """Utility class that sideloads Python modules safely."""

    def __init__(self, module_loader: ModuleLoader | None = None) -> None:
        self._module_loader: ModuleLoader = module_loader or DefaultModuleLoader()

    @property
    def module_loader(self) -> ModuleLoader:
        """Expose the module loader for advanced integrations and testing."""

        return self._module_loader

    def run(
        self, base_path: StrPath, module: str, obj: str | None = None
    ) -> ModuleType | object:
        module_obj = self._module_loader.load_module(base_path, module)
        if obj is None:
            return module_obj
        return self._module_loader.get_attribute(module_obj, obj)


class ModuleSideloadRunner(PyModuleSideload):
    def run(
        self, base_path: StrPath, module: str, obj: str | None = None
    ) -> ModuleType | object:
        """Load a module file under base_path and return module or attribute.

        base_path: directory containing modules or packages
        module: a filename (foo.py), a dotted name (pkg.mod) or a module name
        obj: optional attribute name to return from the module
        """
        return super().run(base_path, module, obj)


def _failure_payload(
    message: str, *, details: Mapping[str, object] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"status": "failure", "message": message}
    if details:
        payload["details"] = dict(details)
    with suppress(ValidationErrorType):
        validate_payload(payload, ERROR_SCHEMA)
    return payload


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _ensure_json_mapping(raw: object) -> Mapping[str, object]:
    if isinstance(raw, Mapping):
        typed = cast("Mapping[str, object]", raw)
        materialized = {str(key): value for key, value in typed.items()}
        return MappingProxyType(materialized)
    return _EMPTY_MAPPING


def _parameters_from_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    parameters_obj = payload.get("parameters")
    if isinstance(parameters_obj, Mapping):
        typed = cast("Mapping[str, object]", parameters_obj)
        return MappingProxyType(dict(typed))
    return _EMPTY_MAPPING


def _validate_required_inputs(
    base_path: str | None, module_name: str | None
) -> tuple[str, str] | None:
    if not base_path:
        return ("base_path", "base_path must be a non-empty string")
    if not module_name:
        return ("module", "module must be a non-empty string")
    return None


def _validate_input_schema(payload: Mapping[str, object]) -> dict[str, object] | None:
    try:
        validate_payload(payload, INPUT_SCHEMA)
    except ValidationErrorType as exc:
        error = cast("_ValidationErrorProtocol", exc)
        return _failure_payload(
            "input payload failed validation",
            details={
                "error": error.message,
                "path": [str(part) for part in error.path],
                "schema_path": [str(part) for part in error.schema_path],
            },
        )
    return None


def _extract_inputs(payload: Mapping[str, object]) -> _CoreInputs | dict[str, object]:
    parameters = _parameters_from_payload(payload)
    base_path = _string_or_none(parameters.get("base_path"))
    module_name = _string_or_none(parameters.get("module"))
    attribute_name = _string_or_none(parameters.get("attribute"))
    loader_options = _ensure_json_mapping(parameters.get("loader_options"))

    missing = _validate_required_inputs(base_path, module_name)
    if missing is not None:
        field, message = missing
        return _failure_payload(message, details={"field": field})

    return _CoreInputs(
        base_path or "", module_name or "", attribute_name, loader_options
    )


def _module_file_from_module(module_obj: ModuleType) -> str | None:
    module_file_raw: object | None = getattr(module_obj, "__file__", None)
    if isinstance(module_file_raw, str):
        return module_file_raw
    if isinstance(module_file_raw, PathLike):
        return fsdecode(module_file_raw)
    return None


def _load_target_module(
    runner: ModuleSideloadRunner, *, base_path: str, module_name: str
) -> _LoadedModule | dict[str, object]:
    try:
        module_obj_loaded = runner.module_loader.load_module(base_path, module_name)
    except (FileNotFoundError, ImportError, ValueError, OSError) as exc:
        return _failure_payload(
            "module resolution failed",
            details={
                "error": str(exc),
                "base_path": base_path,
                "module": module_name,
            },
        )

    module_file = _module_file_from_module(module_obj_loaded)
    if module_file is None:
        module_file = _resolve_module_file(base_path, module_name)

    return _LoadedModule(module_obj_loaded, module_file)


def _resolve_attribute_if_requested(
    runner: ModuleSideloadRunner,
    module_obj: ModuleType,
    attribute_name: str | None,
    *,
    context: _AttributeResolutionContext,
) -> str | dict[str, object]:
    if not attribute_name:
        return "module"
    try:
        attribute_result = runner.module_loader.get_attribute(
            module_obj, attribute_name
        )
    except AttributeError as exc:
        return _failure_payload(
            "attribute resolution failed",
            details={
                "error": str(exc),
                "attribute": attribute_name,
                "module_file": context.module_file,
            },
        )
    context.metadata["attribute_type"] = type(attribute_result).__name__
    context.messages.append(f"Resolved attribute {attribute_name}")
    return "attribute"


def _compose_success_payload(
    *,
    module_file: str,
    attribute_name: str | None,
    object_kind: str,
    messages: Sequence[str],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "module_file": module_file,
        "attribute": attribute_name,
        "object_kind": object_kind,
    }
    if messages:
        payload["messages"] = list(messages)
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def _validate_output_schema(result: Mapping[str, object]) -> dict[str, object] | None:
    try:
        validate_payload(result, OUTPUT_SCHEMA)
    except ValidationErrorType as exc:
        error = cast("_ValidationErrorProtocol", exc)
        return _failure_payload(
            "generated output failed schema validation",
            details={
                "error": error.message,
                "path": [str(part) for part in error.path],
                "schema_path": [str(part) for part in error.schema_path],
            },
        )
    return None


def main_json(
    payload: Mapping[str, object], *, ctx: object | None = None
) -> dict[str, object]:
    del ctx
    schema_failure = _validate_input_schema(payload)
    if schema_failure:
        return schema_failure

    inputs = _extract_inputs(payload)
    if isinstance(inputs, dict):
        return inputs

    runner = ModuleSideloadRunner()
    metadata: dict[str, object] = {"module_name": inputs.module_name}
    if inputs.loader_options:
        metadata["loader_options"] = dict(inputs.loader_options)

    load_result = _load_target_module(
        runner,
        base_path=inputs.base_path,
        module_name=inputs.module_name,
    )
    if isinstance(load_result, dict):
        return load_result

    messages = [f"Loaded {inputs.module_name}"]
    resolution_context = _AttributeResolutionContext(
        module_file=load_result.module_file,
        metadata=metadata,
        messages=messages,
    )
    object_kind = _resolve_attribute_if_requested(
        runner,
        load_result.module,
        inputs.attribute_name,
        context=resolution_context,
    )
    if isinstance(object_kind, dict):
        return object_kind

    result_payload = _compose_success_payload(
        module_file=load_result.module_file,
        attribute_name=inputs.attribute_name,
        object_kind=object_kind,
        messages=messages,
        metadata=metadata,
    )

    output_failure = _validate_output_schema(result_payload)
    if output_failure:
        return output_failure
    return result_payload


def _load_json_payload(file_path: str | None) -> Mapping[str, object]:
    def _load(stream: IO[str]) -> Mapping[str, object]:
        raw_payload = cast("object", json.load(stream))
        if not isinstance(raw_payload, Mapping):
            message = "JSON payload must be a mapping"
            raise TypeError(message)
        payload_map = cast("Mapping[object, object]", raw_payload)
        payload_dict: dict[str, object] = {}
        for key, value in payload_map.items():
            if not isinstance(key, str):
                message = "JSON payload keys must be strings"
                raise TypeError(message)
            payload_dict[key] = value
        return MappingProxyType(payload_dict)

    if file_path:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            return _load(handle)
    return _load(sys.stdin)


def _run_json_cli(args: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="x_make_py_mod_sideload_x JSON runner")
    parser.add_argument(
        "--json", action="store_true", help="Read JSON payload from stdin"
    )
    parser.add_argument("--json-file", type=str, help="Path to JSON payload file")
    parsed = parser.parse_args(args)

    json_flag_obj: object = cast("object", getattr(parsed, "json", False))
    read_from_stdin = bool(json_flag_obj)
    json_file_obj: object = cast("object", getattr(parsed, "json_file", None))
    json_file = json_file_obj if isinstance(json_file_obj, str) else None

    if not (read_from_stdin or json_file):
        parser.error("JSON input required. Use --json for stdin or --json-file <path>.")

    payload = _load_json_payload(None if read_from_stdin else json_file)
    result = main_json(payload)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


# Packaging-friendly aliases
x_cls_make_py_mod_sideload_x = ModuleSideloadRunner
xclsmakepymodsideloadx = ModuleSideloadRunner

__all__ = [
    "ModuleSideloadRunner",
    "PyModuleSideload",
    "main_json",
    "x_cls_make_py_mod_sideload_x",
    "xclsmakepymodsideloadx",
]


if __name__ == "__main__":
    _run_json_cli(sys.argv[1:])
