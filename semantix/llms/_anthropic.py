"""Anthropic API client for Language Learning Models (LLMs)."""

import os
from typing import Optional

from semantix.llms.base import BaseLLM


class Anthropic(BaseLLM):
    """Anthropic API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "claude-3-5-sonnet-20240620",
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
            **kwargs,
        }

    def __infer__(self, messages: list, model_params: dict = {}) -> str:
        """Infer a response from the input meaning."""
        params = {
            **self.default_params,
            **model_params,
        }
        output = self.client.messages.create(messages=messages, **params)
        return output.content[0]["text"]
