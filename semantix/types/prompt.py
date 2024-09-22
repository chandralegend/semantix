"""Module to represent the prompt types."""

from enum import Enum
from types import FrameType
from typing import Any, Callable, Dict, List, Type, Union

from pydantic import BaseModel

from semantix.types.semantic import Semantic
from semantix.utils.helpers import pydantic_to_dataclass
from semantix.utils.utils import (
    extract_non_primary_type,
    get_object_string,
    get_type,
    get_type_from_value,
)


class TypeExplanation:
    """Class to represent the type explanation."""

    def __init__(self, frame: FrameType, type: str) -> None:
        """Initializes the TypeExplanation class."""
        self.type = frame.f_globals[type]

    def get_type_repr(self, type_collector: list = []) -> str:
        """Get the type representation."""
        if issubclass(self.type, BaseModel):
            __doc__ = self.type.__doc__
            self.type = pydantic_to_dataclass(self.type, self.type.__name__)
            self.type.__doc__ = __doc__
        semstr = self.type.__doc__ if self.type.__doc__ else ""
        _name = self.type.__name__
        usage_example_list = []
        for param, annotation in self.type.__init__.__annotations__.items():
            if param == "return":
                continue
            if isinstance(annotation, type) and issubclass(annotation, Semantic):
                type_repr = get_type(annotation.wrapped_type)
                type_collector.extend(extract_non_primary_type(type_repr))
                usage_example_list.append(
                    f'{param}: {type_repr} - {annotation._meaning}"'
                )
            else:
                type_repr = get_type(annotation)
                type_collector.extend(extract_non_primary_type(type_repr))
                usage_example_list.append(f"{param}: {type_repr}")
        usage_example = ", ".join(usage_example_list)
        if semstr:
            return f"- {semstr} ({_name}) (class) -> {_name}({usage_example})".strip()
        return f"- {_name} (class) -> {_name}({usage_example})".strip()

    def get_type_repr_enum(self) -> str:
        """Get the type representation."""
        semstr = self.type.__doc__ if self.type.__doc__ else ""
        _name = self.type.__name__
        usage_example_list = []
        for param, _ in self.type.__members__.items():
            usage_example_list.append(f"{_name}.{param}")
        usage_example = ", ".join(usage_example_list)
        if semstr:
            return f"- {semstr} ({_name}) (Enum) -> {usage_example}".strip()
        return f"- {_name} (Enum) -> {usage_example}".strip()

    def __str__(self) -> str:
        """Returns the string representation of the TypeExplanation class."""
        if issubclass(self.type, Enum):
            return self.get_type_repr_enum()
        return self.get_type_repr()

    def get_nested_types(self) -> list:
        """Get the nested types."""
        type_collector: List[str] = []
        if not issubclass(self.type, Enum):
            self.get_type_repr(type_collector)
        return type_collector


class Information:
    """Class to represent the information."""

    def __init__(self, semstr: str, name: str, value: Any) -> None:  # noqa: ANN401
        """Initializes the Information class."""
        self.value = value
        self.name = name
        self.semstr = semstr

    @property
    def type(self) -> str:
        """Get the type of the information."""
        return get_type_from_value(self.value)

    def get_content(self, contains_media: bool) -> Union[List, str]:
        """Returns the list of dictionaries representation of the InputInformation class."""
        input_type = self.type
        if input_type == "Image" or input_type == "Video":
            return [
                f"- {self.semstr if self.semstr else ''} ({self.name}) ({input_type}) = ",
                self.value,
            ]
        return str(self) if not contains_media else [str(self)]

    def __str__(self) -> str:
        """Returns the string representation of the Information class."""
        if self.semstr:
            return f"- {self.semstr} ({self.name}) ({self.type}) = {get_object_string(self.value)}".strip()
        return f"- {self.name} ({self.type}) = {get_object_string(self.value)}".strip()

    def get_types(self) -> list:
        """Get the types of the information."""
        type_collector = extract_non_primary_type(self.type)
        get_object_string(self.value, type_collector)
        return type_collector


class OutputHint:
    """Class to represent the output hint."""

    def __init__(self, semstr: str, type: Type[Any]) -> None:  # noqa: ANN401
        """Initializes the OutputHint class."""
        self.semstr = semstr
        self.type = get_type(type)

    def __str__(self) -> str:
        """Returns the string representation of the OutputHint class."""
        if self.semstr:
            return f"- {self.semstr} ({self.type})".strip()
        return f"- {self.type}".strip()

    def get_types(self) -> list:
        """Get the types of the output."""
        type_collector = extract_non_primary_type(self.type)
        return type_collector


class Tool:
    """Base class for tools."""

    def __init__(self, func: Callable, semstr: str = "") -> None:
        """Initialize the tool."""
        self.func = func
        self.semstr = semstr

    @property
    def get_params(self) -> List[Dict]:
        """Get the parameters of the tool."""
        params = []
        for param, annotation in self.func.__annotations__.items():
            if isinstance(annotation, type) and issubclass(annotation, Semantic):
                params.append(
                    {
                        "name": param,
                        "type": get_type(annotation.wrapped_type),
                        "semstr": annotation._meaning,
                    }
                )
            else:
                params.append(
                    {
                        "name": param,
                        "type": get_type(annotation),
                        "semstr": "",
                    }
                )
        return params

    def __call__(self, *args, **kwargs) -> str:  # noqa
        """Forward function of the tool."""
        return self.func(*args, **kwargs)

    def get_usage_example(self) -> str:
        """Get the usage example of the tool."""
        get_param_str = lambda x: (  # noqa E731
            f'{x["name"]}: {x["type"]} - "{x["semstr"]}"'
            if x["semstr"]
            else f'{x["name"]}: {x["type"]}'
        )
        return f"{self.func.__name__}({', '.join([get_param_str(x) for x in self.get_params if x['name'] != 'return'])})"  # noqa E501

    def get_return_annotation(self) -> str:
        """Get the return annotation of the tool."""
        return_annotation = self.func.__annotations__.get("return")
        if (
            return_annotation
            and isinstance(return_annotation, type)
            and issubclass(return_annotation, Semantic)
        ):
            return f'returns "{return_annotation._meaning}":{get_type(return_annotation.wrapped_type)}'
        return f"returns {get_type(return_annotation)}"

    def __str__(self) -> str:
        """String representation of the tool."""
        if self.semstr:
            return f"- {self.semstr} ({self.func.__name__}) -> {self.get_usage_example()} {self.get_return_annotation()}".strip()  # noqa E501
        return f"- {self.func.__name__} -> {self.get_usage_example()} {self.get_return_annotation()}".strip()


class ReActOutput:
    """Class to represent the ReAct output."""

    def __init__(self, thought: str, action: str, observation: str) -> None:
        """Initializes the ReActOutput class."""
        self.thought = thought
        self.action = action
        self.observation = observation

    def __repr__(self) -> str:
        """Returns the string representation of the ReActOutput class."""
        return f"\t- Thought: {self.thought}\n\t- Action: {self.action}\n\t- Observation: {self.observation})"
