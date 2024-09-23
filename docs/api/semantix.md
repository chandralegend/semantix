# Sematix API Reference

This page documents the API reference for the `semantix` module.

## Semantic

```python
class Semantic(Generic[T, D])
```

A generic class to represent a semantic type.

### Parameters

- `T` : Type
    - The type of the value.
- `D` : str
    - The description of the value.

### Example

```python
from semantix import Semantic

age: Semantic[int, "Age of the Person"] = 25

def calculate_force(
    m: Semantic[float, "Mass of the Object"],
    a: Semantic[float, "Acceleration of the Object"])
-> Semantic[float, "Force of the Object"]:
    return m * a

@dataclass
class Person:
    name: Semantic[str, "Name of the Person"]
    age: Semantic[int, "Age of the Person"]
    height: Semantic[float, "Height of the Person"]
```

## enhance

```python
@enhance(meaning: str, llm: BaseLLM, method: str = "Normal", tools: List[Callable] = [], retries=2, return_additional_info=False **kwargs)
```

A decorator to enhance the function with LLM capabilities.

### Parameters

- `meaning` : str
    - The meaning of the function.
- `llm` : BaseLLM
    - The Large Language Model to use.
- `method` : str, optional
    - The method to use for the enhancement. Default is `"Normal"`.
    - Options are `"Normal"`, `"Reason"`, `"CoT"`, `"ReAct"`, `"Reflection"`.
- `tools` : List[Callable | Tool], optional
    - List of tools/functions to be used by the LLM. Default is `[]`.
- `retries` : int, optional
    - The number of retries to use. Default is `2`.
- `return_additional_info` : bool, optional
    - Whether to return additional information in the form of `Output` Object. Default is `False`.
- `**kwargs`
    - Additional keyword arguments to pass to the LLM.
    - For example, `temperature`, `max_tokens`, etc. The list of arguments depends on the LLM.

### Example

```python
from semantix import enhance
from semantix.llms.openai import OpenAI

llm = OpenAI()

@enhance("Get Person Informations use common knowledge", llm)
def get_person(name: Semantic[str, "Name of the Person"]) -> Person:
    ...
```
