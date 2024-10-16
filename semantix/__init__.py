"""Semantix is a Python library that give superpowers to your code."""

import semantix.llms as llms
from semantix.decorators import enhance, tool
from semantix.types.semantic import Semantic

__all__ = ["Semantic", "enhance", "tool", "llms"]
