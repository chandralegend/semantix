"""Groq API client for Language Learning Models (LLMs)."""

import os
from typing import Optional

from semantix.llms.base import BaseLLM


class Groq(BaseLLM):
    """Groq API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "llama3-8b-8192",
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the Groq API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the Groq API. Defaults to "llama3-8b-8192".
            api_key (str, optional): The API key for the Groq API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the Groq API.

        Check out models here: https://console.groq.com/docs/models
        """
        import groq

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.client = groq.Groq(api_key=api_key)
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
        output = self.client.chat.complete.create(messages=messages, **params)
        return output.choices[0].message.content
