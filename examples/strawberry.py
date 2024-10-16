from semantix.llms import OpenAI

llm = OpenAI(verbose=True)


@llm.enhance(
    "Count the occurrences of a letter in a given word",
    method="Reflection",
    return_additional_info=True,
)
def count(word: str, letter: str) -> int: ...


answer = count(word="strawberry", letter="r")
print(answer.output)
