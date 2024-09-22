"""OpenAI API client for Language Learning Models (LLMs)."""

import os
from typing import Optional

from semantix.llms.base import BaseLLM


class OpenAI(BaseLLM):
    """OpenAI API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the OpenAI API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the OpenAI API. Defaults to "gpt-4o-mini".
            api_key (str, optional): The API key for the OpenAI API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the OpenAI API.

        You can find the full list of parameters here: https://platform.openai.com/docs/api-reference/chat
        """
        import openai

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
        self.default_params = {
            "model": model,
            **kwargs,
        }

    def __infer__(self, messages: list, model_params: dict = {}) -> str:
        """Infer a response from the input meaning."""
        params = {
            **self.default_params,
            **model_params,
        }
        messages = self.simplify_messages(messages)
        output = self.client.chat.completions.create(messages=messages, **params)
        return output.choices[0].message.content
