from semantix.decorators import with_llm
from semantix.types import Semantic, SemanticClass
from semantix.llms.base import BaseLLM


from examples.nested import nested_fn, nested_var, nested_tool, SomeClass, SomeEnum


var: Semantic[str, "var_str"] = "This is a variable"
someclass_var: Semantic[SomeClass, "someclass_var_str"] = SomeClass("Hello", 1)


@with_llm(
    "This is a test",
    BaseLLM(),
    info=[var, nested_var, someclass_var],
    tools=[nested_tool],
)
def test_fn(a: Semantic[str, "a_str"], b: int) -> Semantic[SomeEnum, "b_str"]:
    """Some Context"""
    ...  # type: ignore


test_fn(a="Hello", b=1)

nested_fn(text="Hello")
