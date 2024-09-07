"""Semantix is a Python library that give superpowers to your code."""

from semantix.decorators import tool, with_llm
from semantix.media import Image, Video
from semantix.types import Semantic, SemanticClass

__all__ = ["Semantic", "with_llm", "tool", "Image", "Video", "SemanticClass"]
