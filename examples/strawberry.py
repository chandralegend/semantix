from semantix import Semantic, enhance
from semantix.llms import OpenAI

llm = OpenAI()


@enhance(
    "Count the occurrences of a letter in a given word",
    llm,
    method="Reason",
    return_additional_info=True,
)
def count(word: str, letter: str) -> int:
    """Seperate the word into letters and go through each
    letter to check if it is equal to the given letter"""
    ...


answer = count(word="strawberry", letter="r")
print(answer)
