from typing import Callable, List
from semantix.types import Semantic, Tool


def with_llm(
    meaning: str,
    model,
    info: list = [],
    method: str = "Normal",
    tools: List[Callable] = [],
    **kwargs,
):
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


def tool(meaning: str) -> Tool:
    def decorator(func):
        return Tool(func, meaning)

    return decorator
