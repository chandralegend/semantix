from typing import Callable, List, Union
from semantix.types import Semantic, Tool, Information, OutputHint, TypeExplanation
from semantix.inference import (
    InferenceEngine,
    PromptInfo,
    OutputFixPromptInfo,
    ExtractOutputPromptInfo,
)
from semantix.utils import get_semstr
import inspect


def with_llm(
    meaning: str,
    model,
    info: list = [],
    method: str = "Normal",
    tools: List[Union[Callable, Tool]] = [],
    model_params: dict = {},
):
    curr_frame = inspect.currentframe()
    if curr_frame:
        frame = curr_frame.f_back
    else:
        raise Exception("Cannot get the current frame.")

    def decorator(func):
        def wrapper(**kwargs):
            informations = [Information(obj, *get_semstr(frame, obj)) for obj in info]
            _tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
            input_informations = []
            return_hint = None
            for param, annotation in func.__annotations__.items():
                if isinstance(annotation, type) and issubclass(annotation, Semantic):
                    if param == "return":
                        return_hint = OutputHint(
                            annotation._meaning, annotation.wrapped_type
                        )
                        continue
                    input_informations.append(
                        Information(annotation._meaning, param, kwargs[param])
                    )
                else:
                    if param == "return":
                        return_hint = OutputHint("", annotation)
                        continue
                    input_informations.append(Information("", param, kwargs[param]))
            action = f"{meaning} ({func.__name__})"
            context = func.__doc__

            types = set()
            for i in [
                *informations,
                *input_informations,
                return_hint if return_hint else [],
            ]:
                types.update(i.get_types())
            type_explanations = [TypeExplanation(frame, t) for t in types]
            for t in type_explanations:
                types.update(t.get_nested_types())
            type_explanations = [TypeExplanation(frame, t) for t in types]

            inference_engine = InferenceEngine(
                model=model,
                method=method,
                prompt_info=PromptInfo(
                    action=action,
                    context=context,
                    informations=informations,
                    input_informations=input_informations,
                    tools=_tools,
                    return_hint=return_hint,
                    type_explanations=type_explanations,
                ),
                extract_output_prompt_info=ExtractOutputPromptInfo(
                    return_hint=return_hint, type_explanations=type_explanations
                ),
                output_fix_prompt_info=OutputFixPromptInfo(
                    return_hint=return_hint,
                    type_explanations=type_explanations,
                ),
                model_params=model_params,
            )
            return inference_engine.run(frame)

        return wrapper

    return decorator


def tool(meaning: str) -> Callable:
    def decorator(func):
        return Tool(func, meaning)

    return decorator
