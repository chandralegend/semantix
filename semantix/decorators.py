from typing import Callable, List, Union
from semantix.types import Semantic, Tool, Information, OutputHint, TypeExplanation
from semantix.inference import InferenceEngine, PromptInfo
from semantix.utils import get_semstr, extract_non_primary_type
import inspect


def with_llm(
    meaning: str,
    model,
    info: list = [],
    method: str = "Normal",
    tools: List[Union[Callable, Tool]] = [],
    model_params: dict = {},
):
    frame = inspect.currentframe().f_back

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
                types.update(extract_non_primary_type(i.type))
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
                model_params=model_params,
            )
            return inference_engine.run()

        return wrapper

    return decorator


def tool(meaning: str) -> Tool:
    def decorator(func):
        return Tool(func, meaning)

    return decorator
