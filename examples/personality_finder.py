from dataclasses import dataclass
from enum import Enum

from semantix import Semantic, enhance
from semantix.llms.openai import OpenAI

llm = OpenAI()


class Personality(Enum):
    """Personality of the Person"""

    INTROVERT = "Introvert"
    EXTROVERT = "Extrovert"


@dataclass
class Person:
    full_name: str
    yod: Semantic[int, "Year of Death"]  # type: ignore
    personality: Semantic[Personality, "Personality of the Person"]  # type: ignore


@enhance("Get Person Information use common knowledge", llm)
def get_person_info(name: Semantic[str, "Name of the Person"]) -> Person:  # type: ignore
    ...


if __name__ == "__main__":
    person_obj = get_person_info(name="Albert Einstein")
    print(
        f"{person_obj.full_name} is an {person_obj.personality.value} who died in {person_obj.yod}"
    )
