"""OpenAI API client for Language Learning Models (LLMs)."""

from semantix.llms.base import BaseLLM


class OpenAI(BaseLLM):
    """OpenAI API client for Language Learning Models (LLMs)."""

    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "gpt-4o-mini",
        **kwargs: dict,
    ) -> None:
        """Initialize the OpenAI API client.

        Args:
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            max_retries (int, optional): The maximum number of retries for API requests. Defaults to 3.
            model (str, optional): The model to use for the OpenAI API. Defaults to "gpt-4o-mini".
            **kwargs (dict): Additional keyword arguments to be passed to the OpenAI API.

        You can find the full list of parameters here: https://platform.openai.com/docs/api-reference/chat
        """
        import openai

        super().__init__(verbose, max_retries)
        self.client = openai.OpenAI()
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
