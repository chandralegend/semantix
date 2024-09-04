from typing import Callable, List
from semantix.types import Semantic, Tool, Information, InputInformation, OutputHint
from semantix.utils import get_semstr
import inspect


def with_llm(
    meaning: str,
    model,
    info: list = [],
    method: str = "Normal",
    tools: List[Callable] = [],
    **kwargs,
):
    frame = inspect.currentframe().f_back

    def decorator(func):
        def wrapper(*args, **kwargs):
            information = [Information(obj, *get_semstr(frame, obj)) for obj in info]
            input_information = []
            return_hint = None
            for param, annotation in func.__annotations__.items():
                if isinstance(annotation, type) and issubclass(annotation, Semantic):
                    if param == "return":
                        return_hint = OutputHint(
                            annotation._meaning, annotation.wrapped_type
                        )
                        continue
                    input_information.append(
                        InputInformation(annotation._meaning, param, kwargs[param])
                    )
                else:
                    if param == "return":
                        return_hint = OutputHint("", annotation)
                        continue
                    input_information.append(InputInformation("", param, kwargs[param]))
            action = f"{func.__name__} {meaning}"
            context = func.__doc__
            print(
                f"""
            Action: {action}
            Context: {context}
            Information: {[str(i) for i in information]}
            Input Information: {[str(i) for i in input_information]}
            Tools: {[str(t) for t in tools]}
            Return Hint: {return_hint}
            """
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def tool(meaning: str) -> Tool:
    def decorator(func):
        return Tool(func, meaning)

    return decorator
