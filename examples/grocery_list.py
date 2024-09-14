from semantix import Semantic, enhance
from semantix.llms.openai import OpenAI
from dataclasses import dataclass
from enum import Enum
from typing import List

llm = OpenAI(temperature=0.0)


class Store(Enum):
    """Stores to buy from"""

    FARMERS_MARKET = "Farmers Market"
    GROCERY_STORE = "Grocery Store"
    CONVENIENCE_STORE = "Convenience Store"
    PHARMACY = "Pharmacy"
    HARDWARE_STORE = "Hardware Store"


@dataclass
class Item:
    """An item to buy"""

    name: str
    quantity: int
    store: Semantic[Store, "Where to buy from"]  # type: ignore


@enhance("Create a Item List from the Call Transcript", llm, method="Reason")
def create_list(text: Semantic[str, "The text of the call"]) -> List[Item]:  # type: ignore
    ...


call_transcript = """
Hey Mike, on your way home can you please pick up milk, 10 eggs, bread and dozen bananas.
and also light bulb in the kitchen is out, we need to fix that. also i am having a headache,
can you get me few aspirin.
"""

items = create_list(text=call_transcript)
for item in items:
    print(f"{item.quantity} {item.name} from {item.store.value}")
