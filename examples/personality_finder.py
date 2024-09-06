from semantix import Semantic, with_llm
from semantix.llms.openai import OpenAI
from enum import Enum

llm = OpenAI()


class Personality(Enum):
    """Personality of the Person"""

    INTROVERT = "Introvert"
    EXTROVERT = "Extrovert"


class Person:
    """Person"""

    def __init__(
        self,
        full_name: Semantic[str, "Fullname of the Person"],  # type: ignore
        yod: Semantic[int, "Year of Death"],  # type: ignore
        personality: Semantic[Personality, "Personality of the Person"],  # type: ignore
    ):
        self.full_name = full_name
        self.yod = yod
        self.personality = personality

    def __repr__(self) -> str:
        return f"{self.full_name} was a {self.personality.value} person who died in {self.yod}"


@with_llm("Get Person Information use common knowledge", llm)
def get_person_info(
    name: Semantic[str, "Name of the Person"], # type: ignore
) -> Semantic[Person, "Person"]:  # type: ignore
    ...


if __name__ == "__main__":
    person_obj = get_person_info(name="Martin Luther King Jr.")
    print(person_obj)
