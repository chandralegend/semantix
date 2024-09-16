"""MistralAI API client for Language Learning Models (LLMs)."""

import os
from typing import Optional

from semantix.llms.base import BaseLLM


class Mistral(BaseLLM):
    """MistralAI API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the MistralAI API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the MistralAI API. Defaults to "mistral-large-latest".
            api_key (str, optional): The API key for the MistralAI API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the MistralAI API.

        Check out models here: https://docs.mistral.ai/getting-started/models/
        """
        import mistralai

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.client = mistralai.Mistral(api_key=api_key)
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
