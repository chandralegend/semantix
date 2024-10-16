"""Decorators for defining semantic types and tools."""

import inspect
from typing import Any, Callable, List, Literal, Union

from semantix.inference import (
    ExtractOutputPromptInfo,
    InferenceEngine,
    OutputFixPromptInfo,
    PromptInfo,
)
from semantix.llms.base import BaseLLM
from semantix.types.prompt import Information, OutputHint, Tool, TypeExplanation
from semantix.types.semantic import Semantic
from semantix.utils.utils import get_semstr


def enhance(
    meaning: str,
    model: BaseLLM,
    info: list = [],
    method: Literal["Normal", "Reason", "CoT", "ReAct", "Reflection"] = "Normal",
    tools: List[Union[Callable, Tool]] = [],
    retries: int = 2,
    return_additional_info: bool = False,
    **kwargs: dict,
) -> Callable:
    """Convert a function into a semantic function with enhanced LLM capabilities.

    Args:
        meaning (str): A description of the function's purpose or intended behavior.
        model (BaseLLM): The Large Language Model instance to be used for enhancement.
        info (list, optional): Additional information or context to be provided to the LLM. Defaults to [].
        method (str, optional): The enhancement method to be applied. Defaults to "Normal". Options are: "Normal", "Reason", "CoT", "ReAct", "Reflection".
        tools (List[Union[Callable, Tool]], optional): A list of functions or Tool objects that the LLM can use. Defaults to [].
        retries (int, optional): The number of retry attempts for LLM operations. Defaults to 2.
        return_additional_info (bool, optional): Whether to return the output and additional information. Defaults to False.
        **kwargs (dict): Additional keyword arguments to be passed to the LLM.

    Returns:
        Callable: A wrapped version of the original function with enhanced LLM capabilities.

    The enhanced function will utilize the specified LLM and method to process inputs and generate outputs.
    The 'tools' parameter allows for the integration of external functions or APIs that the LLM can call upon during execution.
    Proper error handling and retry logic are implemented to ensure robustness.

    Example:
    ```python
    @enhance(
        meaning="Summarize text",
        model=my_llm_instance,
        method="CoT",
        temperature=0.7
    )
    def summarize_text(text: str) -> str:
        ...
    ```

    For more information on available methods and their descriptions, please refer documentation.
    """  # noqa: E501
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
    model_params = kwargs

    def decorator(func: Callable) -> Callable:
        def wrapper(**kwargs: dict) -> Any:  # noqa
            informations = []
            for i in info:
                var_name, meaning = get_semstr(frame, i)
                informations.append(Information(meaning, var_name, i))
            _tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
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
            return inference_engine.run(frame, retries + 1, return_additional_info)

        return wrapper

    return decorator


def tool(meaning: str) -> Callable:
    """Converts a function into a tool.

    Args:
        meaning (str): A description of the tool's purpose or intended behavior.

    Returns:
        Callable: A wrapped version of the original function as a Tool object.

    The tool can be used by the LLM to perform specific tasks or operations during execution.

    Example:
    ```python
    @tool("Summarize text")
    def summarize_text(text: str) -> str:
        # Implementation
    ```

    For more information on how to use tools with the LLM, please refer to the documentation.
    """

    def decorator(func: Callable) -> Tool:
        return Tool(func, meaning)

    return decorator
