"""Builtin LLMs for Semantix."""

from semantix.llms._openai import OpenAI
from semantix.llms.base import BaseLLM

__all__ = ["OpenAI", "BaseLLM"]
