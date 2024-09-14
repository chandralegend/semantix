from dataclasses import dataclass
from enum import Enum
from typing import List

from semantix import Semantic, enhance
from semantix.llms import OpenAI
from semantix.types import Image

llm = OpenAI()


class Personality(Enum):
    """Personality of the Person"""

    INTROVERT = "Introvert"
    EXTROVERT = "Extrovert"


@dataclass
class LifeWork:
    """Life Work of the Person"""

    work: str
    year: int
    description: str

    def __repr__(self) -> str:
        return f"{self.work} ({self.year}) - {self.description}"


@dataclass
class Person:
    full_name: str
    yod: Semantic[int, "Year of Death"]
    personality: Semantic[Personality, "Personality of the Person"]
    life_works: Semantic[List[LifeWork], "Life's Works of the Person"]

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
