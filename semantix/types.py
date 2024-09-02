from typing import Generic, TypeVar, Type, Any

T = TypeVar('T')

class SemanticMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        if 'meaning' in kwargs:
            cls._meaning = kwargs['meaning']
        return cls

    def __getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Semantic requires two parameters: type and meaning")
        typ, meaning = params
        return type(f"MT_{typ.__name__}", (cls,), {"wrapped_type": typ, "_meaning": meaning})

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
    