"""This module contains the classes and functions to represent the types and information needed for the library."""

import inspect
import sys
from typing import Any, Generic, Type, TypeVar

from semantix.utils.utils import get_type


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


class Output:
    """Class to represent the output."""

    def __init__(self, **kwargs: dict) -> None:  # noqa: ANN401
        """Initialize the output class."""
        self.kwargs = {key.replace("-", "_"): value for key, value in kwargs.items()}

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Get the attribute of the class."""
        return self.kwargs.get(name)

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        """Get the item of the class."""
        return self.kwargs.get(name)

    def __repr__(self) -> str:
        """Get the representation of the class."""
        x = "\n".join([f"### {key} ###\n{value}" for key, value in self.kwargs.items()])
        return f"Output:\n{x}"


class SemanticClass:
    """Class to represent the semantic class."""

    @classmethod
    def init(cls, *args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        """Initialize the class."""
        # TODO: Implement the initialization of the class enhances
        return cls.__class__(*args, **kwargs)
