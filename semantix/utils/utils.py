"""Utility functions for the semantix package."""

import ast
import importlib
import re
import sys
from enum import Enum
from types import FrameType, ModuleType
from typing import Any, Optional


def get_type(_type: Any) -> str:  # noqa: ANN401
    """Get the type annotation of the input type."""
    if hasattr(_type, "__origin__") and _type.__origin__ is not None:
        if _type.__origin__ is list:
            return f"list[{get_type(_type.__args__[0])}]"
        if _type.__origin__ is dict:
            return f"dict[{get_type(_type.__args__[0])}, {get_type(_type.__args__[1])}]"
        if _type.__origin__ is tuple:
            return f"tuple[{', '.join([get_type(x) for x in _type.__args__])}]"
        if _type.__origin__ is set:
            return f"set[{get_type(_type.__args[0])}]"
    if hasattr(_type, "__args__"):
        return " | ".join(get_type(arg) for arg in _type.__args__).replace(
            "NoneType", "None"
        )
    return str(_type.__name__) if isinstance(_type, type) else str(_type)


def get_object_string(obj: Any, type_collector: list = []) -> str:  # noqa: ANN401
    """Get the string representation of the input object."""
    if isinstance(obj, str):
        return f'"{obj}"'
    elif isinstance(obj, (int, float, bool)):
        return str(obj)
    elif isinstance(obj, list):
        return (
            "["
            + ", ".join(get_object_string(item, type_collector) for item in obj)
            + "]"
        )
    elif isinstance(obj, tuple):
        return (
            "("
            + ", ".join(get_object_string(item, type_collector) for item in obj)
            + ")"
        )
    elif isinstance(obj, dict):
        return (
            "{"
            + ", ".join(
                f"{get_object_string(key, type_collector)}: {get_object_string(value, type_collector)}"
                for key, value in obj.items()
            )
            + "}"
        )
    elif isinstance(obj, Enum):
        type_collector.append(obj.__class__.__name__)
        return f"{obj.__class__.__name__}.{obj.name}"
    elif hasattr(obj, "__dict__"):
        type_collector.append(obj.__class__.__name__)
        args = ", ".join(
            f"{key}={get_object_string(value)}" for key, value in vars(obj).items()
        )
        return f"{obj.__class__.__name__}({args})"
    else:
        return str(obj)


def extract_non_primary_type(type_str: str) -> list:
    """Extract non-primary types from the type string."""
    if not type_str:
        return []
    pattern = r"(?:\[|,\s*|\|)([a-zA-Z_][a-zA-Z0-9_]*)|([a-zA-Z_][a-zA-Z0-9_]*)"
    matches = re.findall(pattern, type_str)
    primary_types = [
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "Any",
        "None",
        "Image",
        "Video",
    ]
    non_primary_types = [m for t in matches for m in t if m and m not in primary_types]
    return non_primary_types


def get_type_from_value(data: Any) -> str:  # noqa: ANN401
    """Get the type annotation of the input data."""
    if isinstance(data, dict):
        class_name = next(
            (value.__class__.__name__ for value in data.values() if value is not None),
            None,
        )
        if class_name:
            return f"dict[str, {class_name}]"
        else:
            return "dict[str, Any]"
    elif isinstance(data, list):
        if data:
            class_name = data[0].__class__.__name__
            return f"list[{class_name}]"
        else:
            return "list"
    else:
        return str(type(data).__name__)


def get_semstr(
    frame: FrameType,
    obj: Any,  # noqa: ANN401
    var_name: str = "",
    module: Optional[ModuleType] = None,
) -> tuple:
    """Get the semantic meaning of the input object."""
    var_name = (
        var_name
        if var_name
        else next((var for var, val in frame.f_locals.items() if val is obj), "")
    )
    _module = module if module else sys.modules[frame.f_globals["__name__"]]

    if var_name:
        meaning = getattr(_module, f"{var_name}_meaning", None)
    if not meaning and var_name:
        with open(frame.f_code.co_filename, "r") as file:
            tree = ast.parse(file.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == var_name:
                        if node.module:
                            module = importlib.import_module(node.module)
                        else:
                            raise Exception(
                                "Module not found."
                            )  # Don't know whether this will happen
                        var_name, meaning = get_semstr(frame, obj, var_name, module)
                        break
    return var_name, meaning
