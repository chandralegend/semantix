"""Together API client for Language Learning Models (LLMs)."""

import os
from typing import Optional

from semantix.llms.base import BaseLLM


class Together(BaseLLM):
    """Together API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the Together API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the Together API. Defaults to "meta-llama/Llama-3-8b-chat-hf".
            api_key (str, optional): The API key for the Together API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the Together API.

        Check out models here: https://docs.together.ai/docs/chat-models
        """
        import together

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.client = together.Together(api_key=api_key)
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
        output = self.client.chat.completions.create(messages=messages, **params)
        return output.choices[0].message.content
