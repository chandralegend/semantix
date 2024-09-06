from semantix.llms.base import BaseLLM


class OpenAI(BaseLLM):
    def __init__(
        self,
        verbose: bool = False,
        max_retries: int = 3,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> None:
        import openai

        super().__init__(verbose, max_retries)
        self.client = openai.OpenAI()
        self.default_params = {
            "model": model,
            **kwargs,
        }

    def __infer__(self, messages: list, model_params: dict = {}) -> str:
        params = {
            **self.default_params,
            **model_params,
        }
        output = self.client.chat.completions.create(messages=messages, **params)
        return output.choices[0].message.content
