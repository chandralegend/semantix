# Minimal Working Example

Here we will walk you through a minimal working example of using semantix to convert your phone call to a grocery list.

!> Makesure to have the `semantix` package installed. If not, follow the installation instructions [here](#installation).

Lets Assume Following is the transcription of th call you had:

```markdown
Hey Mike, on your way home can you please pick up milk, 10 eggs, bread and dozen bananas.
and also light bulb in the kitchen is out, we need to fix that. also i am having a headache,
can you get me few aspirin.
```

Following is the code to convert the above call to a grocery list:

```python
from dataclasses import dataclass
from enum import Enum
from typing import List

from semantix import Semantic, enhance
from semantix.llms.openai import OpenAI

# Initialize the LLM
llm = OpenAI(temperature=0.0)

# Define the Store Enum
class Store(Enum):
    """Stores to buy from"""
    FARMERS_MARKET = "Farmers Market"
    GROCERY_STORE = "Grocery Store"
    CONVENIENCE_STORE = "Convenience Store"
    PHARMACY = "Pharmacy"
    HARDWARE_STORE = "Hardware Store"

# Define the Item Class
@dataclass
class Item:
    """An item to buy"""
    name: str
    quantity: int
    store: Semantic[Store, "Where to buy from"]

# Define the function to create the list
@enhance("Create a Item List from the Call Transcript", llm, method="Reason")
def create_list(text: Semantic[str, "The text of the call"]) -> List[Item]:
    ...

call_transcript = """
Hey Mike, on your way home can you please pick up milk, 10 eggs, bread and dozen bananas.
and also light bulb in the kitchen is out, we need to fix that. also i am having a headache,
can you get me few aspirin.
"""

items = create_list(text=call_transcript)
for item in items:
    print(f"{item.quantity} {item.name} from {item.store.value}")
```

This code will output the following:

```markdown
1 milk from Grocery Store
10 eggs from Grocery Store
1 bread from Grocery Store
12 bananas from Grocery Store
1 aspirin from Pharmacy
1 light bulb from Hardware Store
```

Violà! You have successfully converted your phone call to a grocery list.

This is just a simple example of how you can use semantix to infuse meaning into your code. You can use the same
approach to enhance any function in your codebase and make it more intelligent and context-aware.
