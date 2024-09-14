from enum import Enum
from typing import List

from semantix import Semantic, enhance
from semantix.llms.openai import OpenAI
from semantix.types import Image

llm = OpenAI()


class Personality(Enum):
    """Personality of the Person"""

    INTROVERT = "Introvert"
    EXTROVERT = "Extrovert"


class LifeWork:
    """Life Work of the Person"""

    def __init__(self, work: str, year: int, description: str):
        self.work = work
        self.year = year
        self.description = description

    def __repr__(self) -> str:
        return f"{self.work} ({self.year}) - {self.description}"

    def __str__(self) -> str:
        return f"{self.work} ({self.year}) - {self.description}"


class Person:
    """Person"""

    def __init__(
        self,
        full_name: Semantic[str, "Fullname of the Person"],  # type: ignore
        yod: Semantic[int, "Year of Death"],  # type: ignore
        personality: Semantic[Personality, "Personality of the Person"],  # type: ignore
        life_works: Semantic[List[LifeWork], "Life's Works of the Person"],  # type: ignore
    ):
        self.full_name = full_name
        self.yod = yod
        self.personality = personality
        self.life_works = life_works

    def __repr__(self) -> str:
        repr_str = (
            f"{self.full_name} was a {self.personality.value} person who died in {self.yod}\n\nLife's Work:\n"
            ""
        )
        for i, work in enumerate(self.life_works):
            repr_str += f"{i+1}. {work}\n"
        return repr_str


@enhance("Get Person Information use common knowledge", llm, method="Reason")
def get_person_info(
    img: Semantic[Image, "Image of a Person"],  # type: ignore
) -> Person: ...


if __name__ == "__main__":
    person_obj = get_person_info(img=Image("examples/mandela.jpg"))
    print(person_obj)
