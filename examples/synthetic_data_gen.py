from dataclasses import dataclass
from pydantic import BaseModel, field_validator

from semantix import Semantic, enhance
from semantix.llms import OpenAI

llm = OpenAI()


class Address(BaseModel):
    """Address Information"""

    street: str
    city: str
    postal_code: int
    country: str

    @field_validator("postal_code")
    def validate_postal_code(cls, v):
        if len(str(v)) != 6:
            raise ValueError("Postal code must be 6 digits.")
        return v


@dataclass
class User:
    """User Information"""

    name: str
    age: int
    address: Address


@enhance(
    "Generate a random person's information. The name must be chosen at random. Make it something you wouldn't normally choose.",
    llm,
)
def generate_user_data() -> Semantic[User, "Synthetic User Data"]: ...  # type: ignore


if __name__ == "__main__":
    for _ in range(5):
        user = generate_user_data()
        print(user)
