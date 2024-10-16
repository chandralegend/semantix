"""Inference engine for running the model and generating prompts."""

from types import FrameType
from typing import Any, List, TYPE_CHECKING

from loguru import logger

from semantix.types.prompt import Information, OutputHint, Tool, TypeExplanation
from semantix.types.semantic import Output

if TYPE_CHECKING:
    from semantix.llms.base import BaseLLM


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

    def get_messages(self, model: "BaseLLM") -> List["BaseLLM.Message"]:
        """Get the messages for the prompt."""
        messages = [model.get_system_message()] if model.SYSTEM_PROMPT else []
        messages.append(
            model.Message(
                model.SYSTEM_ROLE,
                model.Message.Content(
                    [f"{model.get_message_desc('action')} {self.action}"]
                ),
            )
        )
        if self.context:
            messages.append(
                model.Message(
                    model.SYSTEM_ROLE,
                    model.Message.Content(
                        [self.context], model.get_message_desc("context")
                    ),
                )
            )
        messages.append(
            model.Message(
                model.SYSTEM_ROLE,
                model.Message.Content(
                    [str(self.return_hint)], model.get_message_desc("return_hint")
                ),
            )
        )
        if self.tools:
            messages.append(
                model.Message(
                    model.SYSTEM_ROLE,
                    model.Message.Content(
                        [str(t) for t in self.tools], model.get_message_desc("tools")
                    ),
                )
            )
        if self.type_explanations:
            messages.append(
                model.Message(
                    model.SYSTEM_ROLE,
                    model.Message.Content(
                        [str(t) for t in self.type_explanations],
                        model.get_message_desc("type_explanations"),
                    ),
                )
            )
        if self.input_informations:
            messages.append(
                model.Message(
                    model.USER_ROLE,
                    model.Message.Content(
                        self.input_informations,  # type: ignore
                        model.get_message_desc("input_informations"),
                    ),
                )
            )
        if self.informations:
            messages.append(
                model.Message(
                    model.USER_ROLE,
                    model.Message.Content(
                        self.informations, model.get_message_desc("informations")  # type: ignore
                    ),
                )
            )
        return messages


class ExtractOutputPromptInfo:
    """Class to represent the extract output prompt information."""

    def __init__(
        self, return_hint: OutputHint, type_explanations: List[TypeExplanation]
    ) -> None:
        """Initializes the ExtractOutputPromptInfo class."""
        self.return_hint = return_hint
        self.type_explanations = type_explanations

    def get_messages(self, model: "BaseLLM", output: str) -> List["BaseLLM.Message"]:
        """Get the messages for the extract output prompt."""
        messages = [model.get_system_message("extract_output")]
        messages.append(
            model.Message(
                model.SYSTEM_ROLE,
                model.Message.Content(
                    [str(self.return_hint)], model.get_message_desc("return_hint")
                ),
            )
        )
        if self.type_explanations:
            messages.append(
                model.Message(
                    model.SYSTEM_ROLE,
                    model.Message.Content(
                        [str(t) for t in self.type_explanations],
                        model.get_message_desc("type_explanations"),
                    ),
                )
            )
        messages.append(
            model.Message(
                model.USER_ROLE,
                model.Message.Content(
                    [output], model.get_message_desc("extract_output_output")
                ),
            )
        )
        messages.append(
            model.Message(
                model.USER_ROLE,
                model.Message.Content([model.EXTRACT_OUTPUT_INSTRUCTION]),
            )
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

    def get_messages(
        self, model: "BaseLLM", output: str, error: str
    ) -> List["BaseLLM.Message"]:
        """Get the messages for the output fix prompt."""
        messages = [model.get_system_message("output_fix")]
        messages.append(
            model.Message(
                model.SYSTEM_ROLE,
                model.Message.Content(
                    [str(self.return_hint)], model.get_message_desc("return_hint")
                ),
            )
        )
        if self.type_explanations:
            messages.append(
                model.Message(
                    model.SYSTEM_ROLE,
                    model.Message.Content(
                        [str(t) for t in self.type_explanations],
                        model.get_message_desc("type_explanations"),
                    ),
                )
            )
        messages.append(
            model.Message(
                model.USER_ROLE,
                model.Message.Content(
                    [output], model.get_message_desc("output_fix_output")
                ),
            )
        )
        messages.append(
            model.Message(
                model.USER_ROLE,
                model.Message.Content(
                    [error], model.get_message_desc("output_fix_error")
                ),
            )
        )
        messages.append(
            model.Message(
                model.USER_ROLE,
                model.Message.Content([model.OUTPUT_FIX_INSTRUCTION]),
            )
        )
        return messages


class InferenceEngine:
    """Class to represent the inference engine."""

    def __init__(
        self,
        model: "BaseLLM",
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

    def run(
        self, frame: FrameType, retries: int, return_additional_info: bool
    ) -> Any:  # noqa: ANN401
        """Run the inference engine."""
        messages = self.prompt_info.get_messages(self.model)
        messages.append(self.model.method_message(self.method))
        _locals = frame.f_locals
        _globals = frame.f_globals
        for i in range(retries + 1):
            model_output_str = self.model(messages, self.model_params)
            try:
                model_output = self.model.resolve_output(
                    model_output_str,
                    self.extract_output_prompt_info,
                    self.output_fix_prompt_info,
                    _globals,
                    _locals,
                )
                output = Output(**model_output)
                if return_additional_info:
                    return output
                return output.output
            except Exception as e:
                if self.model.verbose and i < retries:
                    err_msg = f"Error encountered: {e}. Retrying... ({i+1}/{retries})"
                    logger.exception(err_msg)
        else:
            raise Exception(f"Failed to perform the operation after {retries} retries.")
