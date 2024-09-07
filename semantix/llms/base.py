"""Base Large Language Model (LLM) class."""

import logging
import re
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from semantix.inference import ExtractOutputPromptInfo, OutputFixPromptInfo


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

NORMAL = """
Generate and return the output result(s) only, adhering to the provided Type in the following format

[Output] <result>
"""  # noqa E501

REASON = """
Reason and return the output result(s) only, adhering to the provided Type in the following format

[Reasoning] <Reason>
[Output] <Result>
"""

CHAIN_OF_THOUGHT = """
Generate and return the output result(s) only, adhering to the provided Type in the following format. Perform the operation in a chain of thoughts.(Think Step by Step)

[Chain of Thoughts] <Chain of Thoughts>
[Output] <Result>
"""  # noqa E501

REACT = """
You are given with a list of tools you can use to do different things. To achieve the given [Action], incrementally think and provide tool_usage necessary to achieve what is thought. Provide your answer adhering in the following format. tool_usage is a function call with the necessary arguments. Only provide one [THOUGHT] and [TOOL USAGE] at a time.

[Thought] <Thought>
[Tool Usage] <tool_usage>
"""  # noqa E501


class BaseLLM:
    """Base Large Language Model (LLM) class."""

    MESSAGE_DESCRIPTIONS = {
        "informations": "[Information]",
        "input_informations": "[Inputs Information]",
        "context": "[Context]",
        "type_explanations": "[Type Explanations]",
        "return_hint": "[Output Information]",
        "action": "[Action]",
        "tools": "[Tools]",
    }
    SYSTEM_PROMPT = "This is an operation you must perform and return the output values. Neither, the methodology, extra sentences nor the code are not needed. Input/Type formatting: Explanation of the Input (variable_name) (type) = value"  # noqa E501
    METHOD_PROMPTS = {
        "Normal": NORMAL,
        "Reason": REASON,
        "Chain-of-Thoughts": CHAIN_OF_THOUGHT,
        "ReAct": REACT,
    }
    EXTRACT_OUTPUT_INSTRUCTION = "Above output is not in the desired Output Format/Type. Please provide the output in the desired type. Do not repeat the previously provided output. Important: Do not provide the code or the methodology. Only provide the output in the desired format."  # noqa E501
    OUTPUT_FIX_INSTRUCTION = "Above output is not in the desired Output Format/Type. Please provide the output in the desired type. Do not repeat the previously provided output. Important: Do not provide the code or the methodology. Only provide the output in the desired format."  # noqa E501

    def __init__(self, verbose: bool = False, max_retries: int = 3) -> None:
        """Initialize the Large Language Model (LLM) client."""
        self.verbose = verbose
        self.max_retries = max_retries

    def get_message_desc(self, key: str) -> str:
        """Get the message description."""
        return self.MESSAGE_DESCRIPTIONS[key]

    @property
    def system_message(self) -> dict:
        """Get the system message."""
        return {"role": "system", "content": self.SYSTEM_PROMPT}

    def method_message(self, method: str) -> dict:
        """Get the method message."""
        return {"role": "system", "content": self.METHOD_PROMPTS[method]}

    def __infer__(self, messages: list, model_params: dict) -> str:
        """Infer a response from the input meaning."""
        raise NotImplementedError

    @staticmethod
    def _msgs_to_str(messages: list) -> str:
        """Convert the messages to a string."""
        return "\n\n".join([str(m["content"]) for m in messages])

    def __call__(self, messages: list, model_params: dict) -> str:
        """Infer a response from the input text."""
        if self.verbose:
            logger.info(f"Model Input\n{self._msgs_to_str(messages)}")
        return self.__infer__(messages, model_params)

    def resolve_output(
        self,
        model_output: str,
        extract_output_prompt_info: "ExtractOutputPromptInfo",
        output_fix_prompt_info: "OutputFixPromptInfo",
        _globals: dict,
        _locals: dict,
    ) -> Any:  # noqa: ANN401
        """Resolve the output string to return the reasoning and output."""
        if self.verbose:
            logger.info(f"Model Output\n{model_output}")
        output_match = re.search(r"\[Output\](.*)", model_output, re.DOTALL)
        if not output_match:
            output = self._extract_output(
                model_output,
                extract_output_prompt_info,
            )
        else:
            output = output_match.group(1).strip()
        return self.to_object(output, output_fix_prompt_info, _globals, _locals)

    def _extract_output(
        self, model_output: str, extract_output_prompt_info: "ExtractOutputPromptInfo"
    ) -> str:
        """Extract the output from the meaning out string."""
        if self.verbose:
            logger.info("Extracting output from the meaning out string.")
        output_extract_messages = extract_output_prompt_info.get_messages(
            self, model_output
        )
        output_extract_output = self.__infer__(output_extract_messages, {})
        if self.verbose:
            logger.info(f"Extracted Output: {output_extract_output}")
        return output_extract_output

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
            return self.to_object(
                output,
                output_fix_prompt_info,
                _globals,
                _locals,
                error=str(e),
                num_retries=num_retries + 1,
            )

    def _fix_output(
        self, output: str, output_fix_prompt_info: "OutputFixPromptInfo", error: str
    ) -> str:
        """Fix the output string."""
        if self.verbose:
            logger.info(f"Error: {error}, Fixing the output.")
        output_fix_messages = output_fix_prompt_info.get_messages(self, output, error)
        output_fix_output = self.__infer__(output_fix_messages, {})
        return output_fix_output
