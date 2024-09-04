from semantix.decorators import with_llm
from semantix.types import Semantic, SemanticClass

from examples.nested import nested_fn, nested_var, nested_tool

var: Semantic[str, "var_str"] = "This is a variable"


@with_llm("This is a test", None, info=[var, nested_var], tools=[nested_tool])
def test_fn(a: Semantic[str, "a_str"], b: int) -> Semantic[str, "b_str"]:
    """Some Context"""
    ...  # type: ignore


test_fn(a="Hello", b=1)

nested_fn(text="Hello")
