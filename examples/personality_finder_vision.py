from semantix import Semantic, with_llm
from semantix.llms.openai import OpenAI
from semantix.media import Image
from enum import Enum

llm = OpenAI(verbose=True)

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
        life_work: Semantic[str, "Life Work of the Person"],  # type: ignore
    ):
        self.full_name = full_name
        self.yod = yod
        self.personality = personality
        self.life_work = life_work

    def __repr__(self) -> str:
        return f"{self.full_name} was a {self.personality.value} person who died in {self.yod}. He was known for his work in {self.life_work}"

@with_llm("Get Person Information use common knowledge", llm, method="Reason")
def get_person_info(
    name: Semantic[Image, "Image of a Person"], # type: ignore
) -> Semantic[Person, "Person"]:
    ...


if __name__ == "__main__":
    person_obj = get_person_info(name=Image("mandela.jpg"))
    print(person_obj)
