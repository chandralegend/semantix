from typing import Generic, TypeVar, Type, Any, Callable, Tuple, Dict, List
import base64
import importlib
import sys
import inspect
import importlib.util
from io import BytesIO

from semantix.utils import (
    extract_non_primary_type,
    get_object_string,
    get_type_from_value,
    get_type,
)

cv2 = importlib.import_module("cv2") if importlib.util.find_spec("cv2") else None
PILImage = (
    importlib.import_module("PIL.Image") if importlib.util.find_spec("PIL") else None
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
    def init(self, **kwargs):
        raise NotImplementedError

    def __object_repr__(self):
        raise NotImplementedError

    @classmethod
    def __type_repr__(cls):
        semstr = cls.__doc__
        _name = cls.__name__
        _type = "class"
        return f"{_type} {_name} {semstr}"


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
        self.type = type

    def __str__(self) -> str:
        """Returns the string representation of the OutputHint class."""
        return f"{self.semstr if self.semstr else ''} ({self.type})".strip()

    def get_types(self) -> list:
        """Get the types of the output."""
        return extract_non_primary_type(self.type)


class Information:
    """Class to represent the information."""

    def __init__(self, name: str, value: Any, semstr: str = "") -> None:
        """Initializes the Information class."""
        self.name = name
        self.value = value
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
        return f"returns {get_type(return_annotation.wrapped_type)}"

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


# class Video:
#     """Class to represent a video."""

#     def __init__(self, file_path: str, seconds_per_frame: int = 2) -> None:
#         """Initializes the Video class."""
#         assert (
#             cv2 is not None
#         ), "Please install the required dependencies by running `pip install semantix[video]`."
#         self.file_path = file_path
#         self.seconds_per_frame = seconds_per_frame

#     def process(
#         self,
#     ) -> list:
#         """Processes the video and returns a list of base64 encoded frames."""
#         assert (
#             cv2 is not None
#         ), "Please install the required dependencies by running `pip install semantix[video]`."

#         assert self.seconds_per_frame > 0, "Seconds per frame must be greater than 0"

#         base64_frames = []

#         video = cv2.VideoCapture(self.file_path)
#         total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = video.get(cv2.CAP_PROP_FPS)
#         video_total_seconds = total_frames / fps
#         assert (
#             video_total_seconds > self.seconds_per_frame
#         ), "Video is too short for the specified seconds per frame"
#         assert (
#             video_total_seconds < 4
#         ), "Video is too long. Please use a video less than 4 seconds long."

#         frames_to_skip = int(fps * self.seconds_per_frame)
#         curr_frame = 0
#         while curr_frame < total_frames - 1:
#             video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
#             success, frame = video.read()
#             if not success:
#                 break
#             _, buffer = cv2.imencode(".jpg", frame)
#             base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
#             curr_frame += frames_to_skip
#         video.release()
#         return base64_frames


# class Image:
#     """Class to represent an image."""

#     def __init__(self, file_path: str) -> None:
#         """Initializes the Image class."""
#         assert (
#             PILImage is not None
#         ), "Please install the required dependencies by running `pip install mtllm[image]`."
#         self.file_path = file_path

#     def process(self) -> Tuple[str, str]:
#         """Processes the image and returns a base64 encoded image and its format."""
#         assert (
#             PILImage is not None
#         ), "Please install the required dependencies by running `pip install mtllm[image]`."
#         image = PILImage.open(self.file_path)
#         img_format = image.format
#         with BytesIO() as buffer:
#             image.save(buffer, format=img_format, quality=100)
#             return (
#                 base64.b64encode(buffer.getvalue()).decode("utf-8"),
#                 img_format.lower(),
#             )


# class TypeExplanation:
#     """Class to represent a type explanation."""

#     def __init__(self, type_item: str, mod_registry: SemRegistry) -> None:
#         """Initializes the TypeExplanation class."""
#         self.type_item = type_item
#         self.explanation, self._nested_types = self.get_type_explanation(mod_registry)

#     def get_type_explanation(self, mod_registry: SemRegistry) -> tuple[str, set[str]]:
#         """Get the type explanation of the input type string."""
#         scope, sem_info = mod_registry.lookup(name=self.type_item)
#         if isinstance(sem_info, SemInfo) and sem_info.type:
#             sem_info_scope = SemScope(sem_info.name, sem_info.type, scope)
#             _, type_info = mod_registry.lookup(scope=sem_info_scope)
#             type_info_types = []
#             type_example_list = []
#             type_example: str
#             if sem_info.type == "Enum" and isinstance(type_info, list):
#                 for enum_item in type_info:
#                     (
#                         type_example_list.append(
#                             f'{sem_info.name}.{enum_item.name} : "{enum_item.semstr}"'
#                             if enum_item.semstr
#                             else f"{sem_info.name}.{enum_item.name}"
#                         )
#                     )
#                 type_example = ", ".join(type_example_list)
#             elif sem_info.type in ["obj", "class", "node", "edge"] and isinstance(
#                 type_info, list
#             ):
#                 for arch_item in type_info:
#                     if arch_item.type in ["obj", "class", "node", "edge"]:
#                         continue
#                     type_example_list.append(
#                         f'{arch_item.name}="{arch_item.semstr}":{arch_item.type}'
#                         if arch_item.semstr
#                         else f"{arch_item.name}={arch_item.type}"
#                     )
#                     if arch_item.type and extract_non_primary_type(arch_item.type):
#                         type_info_types.extend(extract_non_primary_type(arch_item.type))
#                 type_example = f"{sem_info.name}({', '.join(type_example_list)})"
#             return (
#                 f"{sem_info.semstr} ({sem_info.name}) ({sem_info.type}) eg:- {type_example}".strip(),  # noqa: E501
#                 set(type_info_types),
#             )
#         return "", set()

#     def __str__(self) -> str:
#         """Returns the string representation of the TypeExplanation class."""
#         return self.explanation

#     @property
#     def nested_types(self) -> set[str]:
#         """Get the nested types of the type."""
#         return self._nested_types
