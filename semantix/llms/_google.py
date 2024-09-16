"""Google Generative AI API client for Language Learning Models (LLMs)."""

import os
from typing import Optional

from semantix.llms.base import BaseLLM


class Gemini(BaseLLM):
    """Google Generative AI API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the Google Generative AI API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
            model (str, optional): The model to use for the Google Generative AI API. Defaults to "gemini-1.5-flash".
            api_key (str, optional): The API key for the Google Generative AI API. Defaults to None.
            **kwargs (dict): Additional keyword arguments to be passed to the Google Generative AI API.

        Check out models here: https://ai.google.dev/gemini-api/docs/models/gemini
        """
        import google.generativeai as genai

        super().__init__(verbose, max_retries)
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.default_params = {
            **kwargs,
        }

    def __infer__(self, messages: list, model_params: dict = {}) -> str:
        """Infer a response from the input meaning."""
        # TODO: Cannot understand the API documentation for Google Generative AI API.
        raise NotImplementedError("Not yet implemented.")
