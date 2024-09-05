from semantix.types import Semantic
from semantix.llms.base import BaseLLM
from semantix.decorators import with_llm, tool
from enum import Enum


@with_llm("Summarize the Given Text", BaseLLM())
def nested_fn(text: Semantic[str, "Text to Summarize"]) -> Semantic[str, "Summary"]:  # type: ignore
    ...


nested_var: Semantic[str, "Text to Summarize"] = "This is a nested variable"


@tool("This is a tool")
def nested_tool(a: Semantic[str, "Text to Summarize"]) -> None:
    print(a)


class SomeClass:
    """Some Class semstr"""

    def __init__(self, a: Semantic[str, "a_str"], b: int) -> None:
        self.a = a
        self.b = b


class SomeEnum(Enum):
    """Some Enum semstr"""

    A = "A"
    B = "B"
