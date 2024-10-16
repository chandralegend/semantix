from dataclasses import dataclass
from enum import Enum

import semantix as sx
from semantix.llms import OpenAI

llm = OpenAI(verbose=True)


class Personality(Enum):
    """Personality of the Person"""

    INTROVERT = "Introvert"
    EXTROVERT = "Extrovert"


@dataclass
class Person:
    """Person Class"""

    full_name: str
    yod: sx.Semantic[int, "Year of Death"]  # type: ignore
    personality: Personality


@llm.enhance("Get Person Information use common knowledge")
def get_person_info(name: sx.Semantic[str, "Name of the Person"]) -> Person:  # type: ignore
    ...


if __name__ == "__main__":
    person_obj = get_person_info(name="Albert Einstein")
    print(
        f"{person_obj.full_name} is an {person_obj.personality.value} who died in {person_obj.yod}"
    )
