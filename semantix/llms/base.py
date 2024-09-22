"""Base Large Language Model (LLM) class."""

import logging
import re
import traceback
from typing import Any, Dict, List, TYPE_CHECKING, Union

from loguru import logger

from semantix.types import Image, Video
from semantix.types.prompt import Information

if TYPE_CHECKING:
    from semantix.inference import ExtractOutputPromptInfo, OutputFixPromptInfo


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

NORMAL = """
Provide the answer in the desired output type definition. Follow the following template to provide the answer.

```output
Only provide the output in this section in the desired output type.
```
"""  # noqa E501

REASON = """
Provide the answer in the desired output type definition. Follow the following template to provide the answer.

```reasoning
Think step by step to achieve the goal in this section.
```
```output
Only provide the output in this section in the desired output type.
```
"""

CHAIN_OF_THOUGHT = """
Provide the answer in the desired output type definition. Follow the following template to provide the answer.

```chain-of-thoughts
Think Step by Step to achieve the goal.
```
```output
Only provide the output in this section in the desired output type.
```
"""  # noqa E501

REACT = ""

REFLECTION = ""

EXTRACT_OUTPUT_INSTRUCTION = """
Above output is not in the desired output format. Extract the output in the desired format. Follow the following template to provide the answer.

```output
Only provide the output in this section in the desired output type.
```
"""  # noqa E501

OUTPUT_FIX_INSTRUCTION = """
Above output is not in the desired Output Type. Follow the following template to provide the answer.

```debug
Debug the error and fix the output.
```
```output
Only provide the output in this section in the desired output type.
```
"""  # noqa E501


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
        "return_hint": "## Desired Output Type Definition",
        "action": "# **Goal**:",
        "tools": "## Tools",
        "output_fix_error": "## Error Encountered",
        "output_fix_output": "## Previous Output",
        "extract_output_output": "## Model Output",
    }
    SYSTEM_PROMPT = "You are an expert and sticks to the instructions given. You are instructed NOT to provide code snippets but to FOLLOW the instructions to achieve the goal in the desired output."  # noqa E501
    METHOD_PROMPTS = {
        "Normal": NORMAL,
        "Reason": REASON,
        "CoT": CHAIN_OF_THOUGHT,
        "ReAct": REACT,
        "Reflection": REFLECTION,
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
            self.SYSTEM_ROLE, self.Message.Content([self.METHOD_PROMPTS[method]])
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
        output_extract_output = self.__infer__(output_extract_messages, {})
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
