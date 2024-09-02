from typing import Callable
from semantix.types import Semantic

def with_llm(meaning: str, model, info: list = [], method: str = 'Normal', tools: list[Callable]= [], **kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(*args)
            print(**kwargs)
            print(f"{func.__name__} {meaning}")
            for param, annotation in func.__annotations__.items():
                if isinstance(annotation, type) and issubclass(annotation, Semantic):
                    print(f"{annotation._meaning} {annotation.wrapped_type} {param}")
                else:
                    print(f"{param}: {annotation}")
            return func(*args, **kwargs)
        return wrapper
    return decorator