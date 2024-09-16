"""Cohere API client for Language Learning Models (LLMs)."""

import os
from typing import Optional, Tuple

from semantix.llms.base import BaseLLM


class Cohere(BaseLLM):
    """Cohere API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "command-r-plus-08-2024",
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the Cohere API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the Cohere API. Defaults to "command-r-plus-08-2024".
            api_key (str, optional): The API key for the Cohere API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the Cohere API.

        You can find the full list of parameters here: https://docs.cohere.com/reference/chat
        Check out the available models here: https://docs.cohere.com/docs/models
        """
        import cohere

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(api_key=api_key)
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
        chat_history, message = self.process_messages(messages)
        output = self.client.chat(
            chat_history=chat_history,
            message=message,
            **params,
        )
        return output.text

    @staticmethod
    def process_messages(messages: list) -> Tuple[list, str]:
        """Process the messages to the required format."""
        message = messages.pop()["content"]
        chat_history = [{"role": "USER", "message": msg["content"]} for msg in messages]
        return chat_history, message
