"""Cohere API client for Language Learning Models (LLMs)."""

import os
from typing import List, Optional, Tuple

from semantix.llms.base import BaseLLM


class Cohere(BaseLLM):
    """Cohere API client for Language Learning Models (LLMs)."""

    SYSTEM_ROLE = "SYSTEM"
    USER_ROLE = "USER"
    ASSISTANT_ROLE = "CHATBOT"

    class Message(BaseLLM.Message):
        """Message class for the Cohere API client."""

        def to_dict(self) -> dict:
            """Convert the message to a dictionary."""
            return {"role": self.role, "message": self.content.format}

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
        simplified_messages = self.simplify_messages(messages)
        chat_history, message = self.process_messages(simplified_messages)
        output = self.client.chat(
            chat_history=chat_history,
            message=message,
            **params,
        )
        return output.text

    @staticmethod
    def process_messages(messages: list) -> Tuple[list, str]:
        """Process the messages to the required format."""
        message = messages.pop()["message"]
        return messages, message

    def simplify_messages(self, messages: List[dict]) -> List[dict]:
        """Simplify the messages."""
        messages = super().simplify_messages(messages)
        new_msgs: List[dict] = []
        for msg in messages:
            if not new_msgs:
                new_msgs.append(msg)
            elif isinstance(msg["message"], str):
                last_msg = new_msgs[-1]
                if last_msg["role"] == msg["role"] and isinstance(
                    last_msg["message"], str
                ):
                    last_msg["message"] = "\n".join(
                        [last_msg["message"], msg["message"]]
                    )
                else:
                    new_msgs.append(msg)
            else:
                new_msgs.append(msg)
        return new_msgs
