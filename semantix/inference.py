"""Inference engine for running the model and generating prompts."""

from types import FrameType
from typing import List

from semantix.llms.base import BaseLLM
from semantix.types import Information, OutputHint, Tool, TypeExplanation


class PromptInfo:
    """Class to represent the prompt information. (According to Meaning-Typed Prompting Technique)."""

    def __init__(
        self,
        action: str,
        context: str,
        informations: List[Information],
        input_informations: List[Information],
        tools: List[Tool],
        return_hint: OutputHint,
        type_explanations: List[TypeExplanation],
    ) -> None:
        """Initializes the PromptInfo class."""
        self.informations = informations
        self.context = context
        self.input_informations = input_informations
        self.return_hint = return_hint
        self.type_explanations = type_explanations
        self.action = action
        self.tools = tools

    def get_messages(self, model: BaseLLM) -> list:
        """Get the messages for the prompt."""
        messages = [model.system_message]
        if self.informations:
            messages.append(self.get_info_msg(self.informations, model, "informations"))
        if self.input_informations:
            messages.append(
                self.get_info_msg(self.input_informations, model, "input_informations")
            )
        if self.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"{model.get_message_desc('context')}\n{self.context}",
                }
            )
        if self.type_explanations:
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            model.get_message_desc("type_explanations"),
                            *[str(t) for t in self.type_explanations],
                        ]
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('return_hint')}\n{self.return_hint}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('action')}\n{self.action}",
            }
        )
        if self.tools:
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        [model.get_message_desc("tools"), *[str(t) for t in self.tools]]
                    ),
                }
            )
        return messages

    def get_info_msg(
        self, inputs: List[Information], model: BaseLLM, info_type: str
    ) -> dict:
        """Get the information message."""
        contains_media = any(i.type == "Video" or i.type == "Image" for i in inputs)
        contents = [
            (
                model.get_message_desc(info_type)
                if not contains_media
                else {
                    "type": "text",
                    "text": model.get_message_desc(info_type),
                }
            )
        ]
        for i in inputs:
            content = i.get_content(contains_media)
            if isinstance(content, list):
                contents.extend(content)
            else:
                contents.append(content)
        return {
            "role": "user",
            "content": contents if contains_media else "\n".join(contents),  # type: ignore
        }


class ExtractOutputPromptInfo:
    """Class to represent the extract output prompt information."""

    def __init__(
        self, return_hint: OutputHint, type_explanations: List[TypeExplanation]
    ) -> None:
        """Initializes the ExtractOutputPromptInfo class."""
        self.return_hint = return_hint
        self.type_explanations = type_explanations

    def get_messages(self, model: BaseLLM, output: str) -> list:
        """Get the messages for the extract output prompt."""
        messages = []
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('extract_output_output')}\n{output}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('extract_output_return_hint')}\n{self.return_hint}",
            }
        )
        if self.type_explanations:
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            model.get_message_desc("extract_output_type_explanations"),
                            *[str(t) for t in self.type_explanations],
                        ]
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": model.EXTRACT_OUTPUT_INSTRUCTION,
            }
        )
        return messages


class OutputFixPromptInfo:
    """Class to represent the output fix prompt information."""

    def __init__(
        self, return_hint: OutputHint, type_explanations: List[TypeExplanation]
    ) -> None:
        """Initializes the OutputFixPromptInfo class."""
        self.return_hint = return_hint
        self.type_explanations = type_explanations

    def get_messages(self, model: BaseLLM, output: str, error: str) -> list:
        """Get the messages for the output fix prompt."""
        messages = []
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('output_fix_output')}\n{output}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('output_fix_return_hint')}\n{self.return_hint}",
            }
        )
        if self.type_explanations:
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            model.get_message_desc("output_fix_type_explanations"),
                            *[str(t) for t in self.type_explanations],
                        ]
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": f"{model.get_message_desc('output_fix_error')}\n{error}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": model.OUTPUT_FIX_INSTRUCTION,
            }
        )
        return messages


class InferenceEngine:
    """Class to represent the inference engine."""

    def __init__(
        self,
        model: BaseLLM,
        method: str,
        prompt_info: PromptInfo,
        extract_output_prompt_info: ExtractOutputPromptInfo,
        output_fix_prompt_info: OutputFixPromptInfo,
        model_params: dict,
    ) -> None:
        """Initializes the InferenceEngine class."""
        self.model = model
        self.method = method
        self.prompt_info = prompt_info
        self.extract_output_prompt_info = extract_output_prompt_info
        self.output_fix_prompt_info = output_fix_prompt_info
        self.model_params = model_params

    def run(self, frame: FrameType) -> str:  # noqa: ANN401
        """Run the inference engine."""
        messages = self.prompt_info.get_messages(self.model)
        messages.append(self.model.method_message(self.method))
        _locals = frame.f_locals
        _globals = frame.f_globals
        model_output = self.model(messages, self.model_params)
        return self.model.resolve_output(
            model_output,
            self.extract_output_prompt_info,
            self.output_fix_prompt_info,
            _globals,
            _locals,
        )
