"""Decorators for defining semantic types and tools."""

import inspect
from typing import Any, Callable, List, Union

from semantix.inference import (
    ExtractOutputPromptInfo,
    InferenceEngine,
    OutputFixPromptInfo,
    PromptInfo,
)
from semantix.llms.base import BaseLLM
from semantix.types import Information, OutputHint, Semantic, Tool, TypeExplanation
from semantix.utils import get_semstr


def with_llm(
    meaning: str,
    model: BaseLLM,
    info: list = [],
    method: str = "Normal",
    tools: List[Union[Callable, Tool]] = [],
    model_params: dict = {},
) -> Callable:
    """Converts a function into a semantic function with LLM capabilities."""
    curr_frame = inspect.currentframe()
    if curr_frame:
        frame = curr_frame.f_back
    else:
        raise Exception(
            "Cannot get the current frame."
        )  # Don't know whether this will happen
    if not frame:
        raise Exception(
            "Cannot get the previous frame."
        )  # Don't know whether this will happen

    def decorator(func: Callable) -> Callable:
        def wrapper(**kwargs: dict) -> Any:  # noqa
            informations = [Information(obj, *get_semstr(frame, obj)) for obj in info]
            _tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
            input_informations = []
            return_hint: OutputHint
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
            assert (
                return_hint
            ), "Return type is not defined. Please define the return type."
            action = f"{meaning} ({func.__name__})"
            context = func.__doc__ if func.__doc__ else ""

            types = set()
            for i in [
                *informations,
                *input_informations,
                return_hint if return_hint else [],
            ]:
                types.update(i.get_types())  # type: ignore
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
    """Converts a function into a tool."""

    def decorator(func: Callable) -> Tool:
        return Tool(func, meaning)

    return decorator
