from typing import Generic, TypeVar, Type, Any, Callable, Dict, List
import sys
import inspect

from semantix.utils import (
    extract_non_primary_type,
    get_object_string,
    get_type_from_value,
    get_type,
)


T = TypeVar("T")


class SemanticMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        if "meaning" in kwargs:
            cls._meaning = kwargs["meaning"]
        return cls

    def __getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Semantic requires two parameters: type and meaning")
        typ, meaning = params
        frame = inspect.currentframe().f_back
        var_name = list(frame.f_locals.keys())[-1]
        if var_name:
            setattr(
                sys.modules[frame.f_globals["__name__"]], f"{var_name}_meaning", meaning
            )
        return type(
            f"MT_{get_type(typ)}", (cls,), {"wrapped_type": typ, "_meaning": meaning}
        )


class Semantic(Generic[T], metaclass=SemanticMeta):
    wrapped_type: Type[T]
    _meaning: str = ""

    def __new__(cls, *args, **kwargs):
        return cls.wrapped_type(*args, **kwargs)

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, cls.wrapped_type)

    def __subclasscheck__(cls, subclass: Type) -> bool:
        return issubclass(subclass, cls.wrapped_type)

    def __repr__(cls) -> str:
        return f"{cls.wrapped_type.__name__} {cls._meaning}"


class SemanticClass:
    @classmethod
    def init(cls, *args, **kwargs):
        return cls.__class__(*args, **kwargs)

    def __object_repr__(self):
        semstr = self.__class__.__doc__ if self.__class__.__doc__ else ""
        _name = self.__class__.__name__
        var_name = [
            var
            for var, val in inspect.currentframe().f_back.f_locals.items()
            if val is self
        ][0]
        return f"{semstr} ({var_name}) ({_name}) = {get_object_string(self)}".strip()

    @classmethod
    def __type_repr__(cls):
        semstr = cls.__doc__ if cls.__doc__ else ""
        _name = cls.__name__
        usage_example_list = []
        for param, annotation in cls.__init__.__annotations__.items():
            if isinstance(annotation, type) and issubclass(annotation, Semantic):
                usage_example_list.append(
                    f'{param}="{annotation._meaning}":{get_type(annotation.wrapped_type)}'
                )
            else:
                usage_example_list.append(f"{param}={get_type(annotation)}")
        usage_example = ", ".join(usage_example_list)
        return f"{semstr} ({_name}) (class) eg:- {usage_example}".strip()


class InputInformation:
    """Class to represent the input information."""

    def __init__(self, semstr: str, name: str, value: Any) -> None:  # noqa: ANN401
        """Initializes the InputInformation class."""
        self.semstr = semstr
        self.name = name
        self.value = value

    def __str__(self) -> str:
        """Returns the string representation of the InputInformation class."""
        type_anno = get_type_from_value(self.value)
        return f"{self.semstr if self.semstr else ''} ({self.name}) ({type_anno}) = {get_object_string(self.value)}".strip()  # noqa: E501

    def to_list_dict(self) -> List[Dict]:
        """Returns the list of dictionaries representation of the InputInformation class."""
        input_type = get_type_from_value(self.value)
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
        return [
            {
                "type": "text",
                "text": str(self),
            }
        ]

    def get_types(self) -> list:
        """Get the types of the input."""
        return extract_non_primary_type(get_type_from_value(self.value))


class OutputHint:
    """Class to represent the output hint."""

    def __init__(self, semstr: str, type: str) -> None:  # noqa: ANN401
        """Initializes the OutputHint class."""
        self.semstr = semstr
        self.type = get_type(type)

    def __str__(self) -> str:
        """Returns the string representation of the OutputHint class."""
        return f"{self.semstr if self.semstr else ''} ({self.type})".strip()

    def get_types(self) -> list:
        """Get the types of the output."""
        return extract_non_primary_type(self.type)


class Information:
    """Class to represent the information."""

    def __init__(self, value: Any, name: str, semstr: str) -> None:
        """Initializes the Information class."""
        self.value = value
        self.name = name
        self.semstr = semstr

    def __str__(self) -> str:
        """Returns the string representation of the Information class."""
        type_anno = get_type_from_value(self.value)
        return f"{self.semstr} ({self.name}) ({type_anno}) = {get_object_string(self.value)}".strip()

    def get_types(self) -> list:
        """Get the types of the information."""
        return extract_non_primary_type(self.type)


class Tool:
    """Base class for tools."""

    def __init__(self, func: Callable, semstr: str = "") -> None:
        """Initialize the tool."""
        self.func = func
        self.semstr = semstr

    @property
    def get_params(self) -> List[Dict]:
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
        return f"{self.func.__name__}({', '.join([get_param_str(x) for x in self.get_params if x['name'] != 'return'])})"

    def get_return_annotation(self) -> str:
        """Get the return annotation of the tool."""
        return_annotation = self.func.__annotations__.get("return", None)
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
