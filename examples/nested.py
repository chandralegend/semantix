from semantix.types import Semantic
from semantix.decorators import with_llm, tool


@with_llm("Summarize the Given Text", None)
def nested_fn(text: Semantic[str, "Text to Summarize"]) -> Semantic[str, "Summary"]:  # type: ignore
    ...


nested_var: Semantic[str, "Text to Summarize"] = "This is a nested variable"


@tool("This is a tool")
def nested_tool(a: Semantic[str, "Text to Summarize"]) -> None:
    print(a)
