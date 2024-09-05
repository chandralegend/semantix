from typing import List, Optional
from semantix.types import (
    Tool,
    Information,
    OutputHint,
    TypeExplanation,
)
from semantix.llms.base import BaseLLM
from icecream import ic


class PromptInfo:
    def __init__(
        self,
        action: str,
        context: str,
        informations: List[Information],
        input_informations: List[Information],
        tools: List[Tool],
        return_hint: OutputHint,
        type_explanations: List[TypeExplanation],
    ):
        self.informations = informations
        self.context = context
        self.input_informations = input_informations
        self.return_hint = return_hint
        self.type_explanations = type_explanations
        self.action = action
        self.tools = tools

    def get_prompt(self, model: BaseLLM) -> list:
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
    ) -> Optional[dict]:
        contains_media = any([i.type == "Video" or i.type == "Image" for i in inputs])
        contents = [model.get_message_desc(info_type)]
        for i in inputs:
            content = i.get_content(contains_media)
            (
                content.extend(content)
                if isinstance(content, list)
                else contents.append(content)
            )
        return {
            "role": "user",
            "content": contents if contains_media else "\n".join(contents),
        }


class InferenceEngine:
    def __init__(self, model, method, prompt_info: PromptInfo, model_params: dict):
        self.model = model
        self.method = method
        self.prompt_info = prompt_info
        self.model_params = model_params

        ic(prompt_info.get_prompt(model))

    def run(self):
        return "This is the result of the inference engine"
