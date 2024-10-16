"""Base Large Language Model (LLM) class."""

import inspect
import logging
import re
import traceback
from typing import Any, Callable, Dict, List, Literal, Union

from loguru import logger


from semantix.inference import (
    ExtractOutputPromptInfo,
    InferenceEngine,
    OutputFixPromptInfo,
    PromptInfo,
)
from semantix.types import Image, Video
from semantix.types.prompt import Information, OutputHint, Tool, TypeExplanation
from semantix.types.semantic import Semantic
from semantix.utils.utils import get_semstr


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

NORMAL = """
Follow the following template to provide the answer.

```output
Provide the output in the desired output type.
```
"""

REASON = """
Follow the following template to provide the answer.

```reasoning
Lets Reason to achieve the goal in this section.
```
```output
Provide the output in the desired output type.
```
"""

CHAIN_OF_THOUGHT = """
Follow the following template to provide the answer.

```chain-of-thoughts
Lets Think Step by Step to achieve the goal.
```
```output
Provide the output in the desired output type.
```
"""

REFLECTION = """
Follow the following template to provide the answer.

```chain-of-thoughts
Lets Think Step by Step to achieve the goal.
```
```reflection
Lets Reflect on The Thought Process, and check the validity of the thought process.
```
```output
Provide the output in the desired output type.
```
"""

PLANNER = """
Follow the following template to provide the answer.

```plan
Step by Step Plan approach to achieve the goal. (No Code)
```
Execute the Plan as follows:
for i, step in plan:
```step_i
Execution of the the step.
```
Finally,
```output
Provide the output in the desired output type.
```
"""

REACT = ""

EXTRACT_OUTPUT_INSTRUCTION = """
Above output is not in the desired output format.
Follow the following template to provide the answer.

```output
Only provide the output in this section in the desired output type.
```
"""

OUTPUT_FIX_INSTRUCTION = """
Above Error is encountered when trying to evaluate the Model Output.
Follow the following template to provide the answer.

```debug
Debug the error and fix the output.
```
```output
Provide the output in the desired output type.
```
"""


class BaseLLM:
    """Base Large Language Model (LLM) class."""

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"

    MESSAGE_DESCRIPTIONS = {
        "informations": "## Additional Information",
        "input_informations": "## Inputs",
        "context": "## Context",
        "type_explanations": "## Type Definitions",
        "return_hint": "## Output Type Definition",
        "action": "# Goal:",
        "tools": "## Tools",
        "output_fix_error": "## Error Encountered",
        "output_fix_output": "## Previous Output",
        "extract_output_output": "## Model Output",
    }
    SYSTEM_PROMPT = ""
    METHOD_PROMPTS = {
        "Normal": NORMAL,
        "Reason": REASON,
        "CoT": CHAIN_OF_THOUGHT,
        "ReAct": REACT,
        "Reflection": REFLECTION,
        "Planner": PLANNER,
    }
    EXTRACT_OUTPUT_INSTRUCTION = EXTRACT_OUTPUT_INSTRUCTION
    OUTPUT_FIX_INSTRUCTION = OUTPUT_FIX_INSTRUCTION
    SYSTEM_MESSAGES = {
        "extract_output": "You are an expert in extracting the output in the desired format.",
        "output_fix": "You are an expert in debugging python errors.",
    }

    class Message:
        """Class to represent the message."""

        class Content:
            """Class to represent the content."""

            def __init__(
                self, items: List[Union[str, Information]], desc: str = ""
            ) -> None:
                """Initialize the content."""
                self.items = items
                self.desc = desc

            def __instancecheck__(self, instance: Any) -> bool:  # noqa: ANN401
                """Check if the instance is an instance of the content."""
                return all(isinstance(i, instance) for i in self.items)

            @property
            def format(self) -> Union[str, List[Dict]]:
                """Format the content."""
                if isinstance(self.items[0], Information):
                    contains_media = any(
                        i.type in ["Video", "Image"] for i in self.items  # type: ignore
                    )
                    contents: List[Union[str, Dict]] = []
                    if self.desc:
                        (
                            contents.append(self.desc)
                            if not contains_media
                            else contents.append(
                                {
                                    "type": "text",
                                    "text": self.desc,
                                }
                            )
                        )
                    for i in self.items:
                        content = i.get_content(contains_media)  # type: ignore
                        if isinstance(content, str):
                            contents.append(content)
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, str):
                                    contents.append({"type": "text", "text": c})
                                elif isinstance(c, Image):
                                    img_base64, img_type = c.process()
                                    contents.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/{img_type};base64,{img_base64}",
                                                "detail": c.quality,
                                            },
                                        }
                                    )
                                elif isinstance(c, Video):
                                    frames = c.process()
                                    contents.extend(
                                        [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpg;base64,{frame}",
                                                    "detail": c.quality,
                                                },
                                            }
                                            for frame in frames
                                        ]
                                    )
                                else:
                                    raise ValueError(f"Unknown content type: {type(c)}")
                    return contents if contains_media else "\n".join(contents).strip()  # type: ignore
                contents = [self.desc] + self.items  # type: ignore
                return "\n".join(contents).strip()  # type: ignore

        def __init__(self, role: str, content: Content) -> None:
            """Initialize the message."""
            self.role = role
            self.content = content

        def to_dict(self) -> dict:
            """Convert the message to a dictionary."""
            return {"role": self.role, "content": self.content.format}

        def __str__(self) -> str:
            """Get the string representation of the message."""
            content_items = self.content.items
            x = [self.content.desc]
            if isinstance(content_items[0], Information):
                contains_media = any(
                    i.type in ["Video", "Image"] for i in content_items  # type: ignore
                )
                for i in content_items:
                    content = i.get_content(contains_media)  # type: ignore
                    if isinstance(content, str):
                        x.append(content)
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, str):
                                x.append(c)
                            elif isinstance(c, Image):
                                x.append(f"Image: {c.file_path}")
                            elif isinstance(c, Video):
                                x.append(f"Video: {c.file_path}")
                            else:
                                raise ValueError(f"Unknown content type: {type(c)}")
            else:
                x.extend(content_items)  # type: ignore
            return "\n".join(x).strip()

    def __init__(self, verbose: bool = False, max_retries: int = 3) -> None:
        """Initialize the Large Language Model (LLM) client."""
        self.verbose = verbose
        self.max_retries = max_retries

    def get_message_desc(self, key: str) -> str:
        """Get the message description."""
        return self.MESSAGE_DESCRIPTIONS.get(key, "")

    def get_system_message(self, variant: str = "") -> Message:
        """Get the system message."""
        return self.Message(
            self.SYSTEM_ROLE,
            self.Message.Content(
                [self.SYSTEM_MESSAGES.get(variant, self.SYSTEM_PROMPT)]
            ),
        )

    def method_message(self, method: str) -> Message:
        """Get the method message."""
        return self.Message(
            self.USER_ROLE, self.Message.Content([self.METHOD_PROMPTS[method]])
        )

    def __infer__(self, messages: list, model_params: dict) -> str:
        """Infer a response from the input meaning."""
        raise NotImplementedError

    @staticmethod
    def _msgs_to_str(messages: List[Message]) -> str:
        """Convert the messages to a string."""
        return "\n".join([str(m) for m in messages])

    def __call__(self, messages: List[Message], model_params: dict) -> str:
        """Infer a response from the input text."""
        if self.verbose:
            logger.info(f"Model Input\n{self._msgs_to_str(messages)}")
        _messages = [m.to_dict() for m in messages]
        return self.__infer__(_messages, model_params)

    def simplify_messages(self, messages: List[dict]) -> List[dict]:
        """Simplify the messages by combining consecutive messages from the same role."""
        new_msgs: List[dict] = []
        for msg in messages:
            if not new_msgs:
                new_msgs.append(msg)
            elif isinstance(msg["content"], str):
                last_msg = new_msgs[-1]
                if last_msg["role"] == msg["role"] and isinstance(
                    last_msg["content"], str
                ):
                    last_msg["content"] = "\n".join(
                        [last_msg["content"], msg["content"]]
                    )
                else:
                    new_msgs.append(msg)
            else:
                new_msgs.append(msg)
        return new_msgs

    def resolve_output(
        self,
        model_output: str,
        extract_output_prompt_info: "ExtractOutputPromptInfo",
        output_fix_prompt_info: "OutputFixPromptInfo",
        _globals: dict,
        _locals: dict,
    ) -> dict:
        """Resolve the output string to return the reasoning and output."""
        if self.verbose:
            logger.info(f"Model Output\n{model_output}")
        outputs = dict(re.findall(r"```(.*?)\n(.*?)```", model_output, re.DOTALL))
        if "output" not in outputs:
            output = self._extract_output(
                model_output,
                extract_output_prompt_info,
            )
        else:
            output = outputs["output"].strip()
        obj = self.to_object(output, output_fix_prompt_info, _globals, _locals)
        outputs["output"] = obj
        return outputs

    def _extract_output(
        self, model_output: str, extract_output_prompt_info: "ExtractOutputPromptInfo"
    ) -> str:
        """Extract the output from the model output."""
        if self.verbose:
            logger.info("Extracting output from the model output.")
        output_extract_messages = extract_output_prompt_info.get_messages(
            self, model_output
        )
        _messages = [m.to_dict() for m in output_extract_messages]
        output_extract_output = self.__infer__(_messages, {})
        if self.verbose:
            logger.info(f"Extracted Output: {output_extract_output}")
        outputs = dict(re.findall(r"```(.*?)\n(.*?)```", model_output, re.DOTALL))
        return outputs["output"].strip()

    def to_object(
        self,
        output: str,
        output_fix_prompt_info: "OutputFixPromptInfo",
        _globals: dict,
        _locals: dict,
        error: str = "",
        num_retries: int = 0,
    ) -> Any:  # noqa: ANN401
        """Convert the output string to an object."""
        if output_fix_prompt_info.return_hint.type == "str":
            return output
        if num_retries >= self.max_retries:
            raise ValueError("Failed to convert output to object. Max tries reached.")
        if error:
            fixed_output = self._fix_output(output, output_fix_prompt_info, error)
            return self.to_object(
                fixed_output,
                output_fix_prompt_info,
                _globals,
                _locals,
                error="",
                num_retries=num_retries + 1,
            )
        try:
            return eval(output, _globals, _locals)
        except Exception as e:
            if num_retries == self.max_retries - 1:
                traceback_str = traceback.format_exc()
                error_str = "\n".join([traceback_str, str(e)])
            else:
                error_str = str(e)
            return self.to_object(
                output,
                output_fix_prompt_info,
                _globals,
                _locals,
                error=error_str,
                num_retries=num_retries + 1,
            )

    def _fix_output(
        self, output: str, output_fix_prompt_info: "OutputFixPromptInfo", error: str
    ) -> str:
        """Fix the output string."""
        if self.verbose:
            logger.info(f"Error: {error}, Fixing the output.")
        output_fix_messages = [
            m.to_dict()
            for m in output_fix_prompt_info.get_messages(self, output, error)
        ]
        output_fix_output = self.__infer__(output_fix_messages, {})
        if self.verbose:
            logger.info(f"Fixed Output: {output_fix_output}")
        outputs = dict(re.findall(r"```(.*?)\n(.*?)```", output_fix_output, re.DOTALL))
        return outputs["output"].strip()

    def enhance(
        self,
        meaning: str = "",
        info: list = [],
        method: Literal[
            "Normal", "Reason", "CoT", "ReAct", "Reflection", "Planner"
        ] = "Normal",
        tools: List[Union[Callable, Tool]] = [],
        retries: int = 2,
        return_additional_info: bool = False,
        **kwargs: dict,
    ) -> Callable:
        """Convert a function into a semantic function with enhanced LLM capabilities.

        Args:
            meaning (str, optional): A description of the function's purpose or intended behavior.
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
                    var_name, semstr = get_semstr(frame, i)
                    informations.append(Information(semstr, var_name, i))
                _tools = [
                    tool if isinstance(tool, Tool) else Tool(tool) for tool in tools
                ]
                input_informations = []
                return_hint: OutputHint
                for param, annotation in func.__annotations__.items():
                    if isinstance(annotation, type) and issubclass(
                        annotation, Semantic
                    ):
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
                    model=self,
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
