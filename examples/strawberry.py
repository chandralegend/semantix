from semantix import Semantic, enhance
from semantix.llms import OpenAI

llm = OpenAI()


@enhance(
    "Count the occurrences of a letter in a given word",
    llm,
    method="Reflection",
    return_additional_info=True,
)
def count(
    word: str, letter: str
) -> Semantic[int, "Occurrences of a Letter in a Word"]: ...  # type: ignore


print(count(word="strawberry", letter="r"))  # 3
