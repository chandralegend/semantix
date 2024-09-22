"""Anthropic API client for Language Learning Models (LLMs)."""

import os
from typing import List, Optional

from semantix.llms.base import BaseLLM
from semantix.types import Image, Video
from semantix.types.prompt import Information


class Anthropic(BaseLLM):
    """Anthropic API client for Language Learning Models (LLMs)."""

    class Message(BaseLLM.Message):
        """Message class for the Anthropic API client."""

        class Content(BaseLLM.Message.Content):
            """Content class for the Anthropic API client."""

            @property
            def format(self) -> str:
                """Return the content in the correct format."""
                if isinstance(self.items[0], Information):
                    contains_media = any(
                        i.type in ["Video", "Image"] for i in self.items  # type: ignore
                    )
                    contents: List = []
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
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": f"image/{img_type}",
                                                "data": img_base64,
                                            },
                                        }
                                    )
                                elif isinstance(c, Video):
                                    frames = c.process()
                                    contents.extend(
                                        [
                                            {
                                                "type": "image_url",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": "image/jpeg",
                                                    "data": frame,
                                                },
                                            }
                                            for frame in frames
                                        ]
                                    )
                                else:
                                    raise ValueError(f"Unknown content type: {type(c)}")
                    return contents if contains_media else "\n".join(contents).strip()  # type: ignore
                contents = [self.desc] + self.items
                return "\n".join(contents).strip()  # type: ignore

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the Anthropic API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the Anthropic API. Defaults to "claude-3.5-sonnet-20240620".
            api_key (str, optional): The API key for the Anthropic API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the Anthropic API.

        You can find the full list of parameters here: https://docs.anthropic.com/en/api/messages
        Check out the available models here: https://docs.anthropic.com/en/docs/about-claude/models
        """
        import anthropic

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.default_params = {
            "model": model,
            "max_tokens": max_tokens,
            **kwargs,
        }

    def __infer__(self, messages: list, model_params: dict = {}) -> str:
        """Infer a response from the input meaning."""
        params = {
            **self.default_params,
            **model_params,
        }
        system_message = messages.pop(0)
        # Anthropic API requires the system message to be seperate and not part of the messages
        # Also, user and assistant roles should be one after another without consecutive messages from the same role
        messages = [
            {
                "role": (
                    self.USER_ROLE
                    if message["role"] == self.SYSTEM_ROLE
                    else message["role"]
                ),
                "content": message["content"],
            }
            for message in messages
        ]
        messages = self.simplify_messages(messages)
        output = self.client.messages.create(
            system=system_message["content"], messages=messages, **params
        )
        return output.content[0].text

    def simplify_messages(self, messages: List[dict]) -> List[dict]:
        """Simplify the messages to the required format."""
        simplified = super().simplify_messages(messages)
        new_messages: List[dict] = []
        for message in simplified:
            if not new_messages:
                new_messages.append(message)
                continue
            last_message = new_messages[-1]
            if last_message["role"] == message["role"]:
                if isinstance(last_message["content"], list):
                    if isinstance(message["content"], list):
                        last_message["content"].extend(message["content"])
                    else:
                        last_message["content"].append(
                            {"type": "text", "text": message["content"]}
                        )
                elif isinstance(message["content"], list):
                    new_messages[-1] = {
                        "role": last_message["role"],
                        "content": [
                            {"type": "text", "text": last_message["content"]},
                            *message["content"],
                        ],
                    }
                else:
                    new_messages[-1] = {
                        "role": last_message["role"],
                        "content": "\n".join(
                            [last_message["content"], message["content"]]
                        ),
                    }
            else:
                new_messages.append(message)
        return new_messages
