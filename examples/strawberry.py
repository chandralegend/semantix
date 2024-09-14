from semantix import Semantic, enhance
from semantix.llms import OpenAI

llm = OpenAI(verbose=True)


@enhance("Count the occurrences of a letter in a given word", llm, method="Reflection")
def count(
    word: str, letter: str
) -> Semantic[int, "Occurrences of a Letter in a Word"]: ...


print(count(word="strawberry", letter="r"))  # 3
