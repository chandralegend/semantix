"""This module contains the classes and functions to represent the types and information needed for the library."""

import inspect
import sys
from enum import Enum
from types import FrameType
from typing import Any, Callable, Dict, Generic, List, Type, TypeVar, Union

from semantix.utils import (
    extract_non_primary_type,
    get_object_string,
    get_type,
    get_type_from_value,
)


T = TypeVar("T")


class SemanticMeta(type):
    """Metaclass for the Semantic class."""

    def __new__(
        mcs, name: str, bases: tuple, namespace: dict, **kwargs: dict  # noqa: N804
    ) -> Any:  # noqa: ANN401
        """Creates a new instance of the class."""
        cls = super().__new__(mcs, name, bases, namespace)
        if "meaning" in kwargs and hasattr(cls, "meaning"):
            cls._meaning = kwargs["meaning"]  # type: ignore
        return cls

    def __getitem__(cls, params: tuple) -> Type[T]:
        """Get the item from the class."""
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Semantic requires two parameters: type and meaning")
        typ, meaning = params
        curr_frame = inspect.currentframe()
        if curr_frame:
            frame = curr_frame.f_back
        if not frame:
            raise Exception("Cannot get the current frame.")
        var_name = list(frame.f_locals.keys())[-1]
        # Set the meaning of the variable in the module's global scope
        if var_name:
            setattr(
                sys.modules[frame.f_globals["__name__"]], f"{var_name}_meaning", meaning
            )
        return type(
            f"MT_{get_type(typ)}", (cls,), {"wrapped_type": typ, "_meaning": meaning}
        )


class Semantic(Generic[T], metaclass=SemanticMeta):
    """Class to represent the semantic type."""

    wrapped_type: Type[T]
    _meaning: str = ""

    def __new__(cls, *args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        """Creates a new instance of the class."""
        return cls.wrapped_type(*args, **kwargs)

    def __instancecheck__(self, instance: Any) -> bool:  # noqa: ANN401
        """Check if the instance is of the class."""
        return isinstance(instance, self.wrapped_type)

    def __subclasscheck__(self, subclass: Type) -> bool:
        """Check if the subclass is of the class."""
        return issubclass(subclass, self.wrapped_type)

    def __repr__(self) -> str:
        """Get the representation of the class."""
        return f"{self.wrapped_type.__name__} {self._meaning}"


class SemanticClass:
    """Class to represent the semantic class."""

    @classmethod
    def init(cls, *args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        """Initialize the class."""
        # TODO: Implement the initialization of the class with llms
        return cls.__class__(*args, **kwargs)


class TypeExplanation:
    """Class to represent the type explanation."""

    def __init__(self, frame: FrameType, type: str) -> None:
        """Initializes the TypeExplanation class."""
        self.type = frame.f_globals[type]

    def get_type_repr(self, type_collector: list = []) -> str:
        """Get the type representation."""
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
                    f'{param}="{annotation._meaning}":{type_repr}'
                )
            else:
                type_repr = get_type(annotation)
                type_collector.extend(extract_non_primary_type(type_repr))
                usage_example_list.append(f"{param}={type_repr}")
        usage_example = ", ".join(usage_example_list)
        return f"{semstr} ({_name}) (class) eg:- {_name}({usage_example})".strip()

    def get_type_repr_enum(self) -> str:
        """Get the type representation."""
        semstr = self.type.__doc__ if self.type.__doc__ else ""
        _name = self.type.__name__
        usage_example_list = []
        for param, _ in self.type.__members__.items():
            usage_example_list.append(f"{_name}.{param}")
        usage_example = ", ".join(usage_example_list)
        return f"{semstr} ({_name}) (enum) eg:- {usage_example}".strip()

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

    def get_content(self, contains_media: bool) -> Union[List[Dict], str]:
        """Returns the list of dictionaries representation of the InputInformation class."""
        input_type = self.type
        if input_type == "Image":
            img_base64, img_type = self.value.process()
            return [
                {
                    "type": "text",
                    "text": f"{self.semstr if self.semstr else ''} ({self.name}) (Image) = ".strip(),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{img_type};base64,{img_base64}"},
                },
            ]
        elif input_type == "Video":
            video_frames = self.value.process()
            return [
                {
                    "type": "text",
                    "text": f"{self.semstr if self.semstr else ''} ({self.name}) (Video) = ".strip(),
                },
                *(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low",
                        },
                    }
                    for frame in video_frames
                ),
            ]
        return (
            str(self)
            if not contains_media
            else [
                {
                    "type": "text",
                    "text": str(self),
                }
            ]
        )

    def __str__(self) -> str:
        """Returns the string representation of the Information class."""
        return f"{self.semstr} ({self.name}) ({self.type}) = {get_object_string(self.value)}".strip()

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
        return f"{self.semstr if self.semstr else ''} ({self.type})".strip()

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
            f'{x["name"]}="{x["semstr"]}":{x["type"]}'
            if x["semstr"]
            else f'{x["name"]}={x["type"]}'
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
        return " | ".join(
            [
                f"{self.semstr} ({self.func.__name__})",
                self.get_return_annotation(),
                f"usage eg. {self.get_usage_example()}",
            ]
        )


class ReActOutput:
    """Class to represent the ReAct output."""

    def __init__(self, thought: str, action: str, observation: str) -> None:
        """Initializes the ReActOutput class."""
        self.thought = thought
        self.action = action
        self.observation = observation

    def __repr__(self) -> str:
        """Returns the string representation of the ReActOutput class."""
        return f"ReActOutput(thought={self.thought}, action={self.action}, observation={self.observation})"
