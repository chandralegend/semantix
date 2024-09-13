# Best Practices

Semantix is a powerful tool that can be used in many different ways. Here are some best practices to help you get the most out of it.
at the sametime semantix expects users to follow python best practices as well.

## When to use Semantic Types, Meaning or Docstrings?

Using Semantic Types is not a must in Semantix, but it can greatly improve the results of your code and the output of the LLMs. If your function, Class, variable names are not self explanatory, you can use Semantic Types to provide more context to the LLMs.

For example, assume that you want to generate a Joke with a punchline. You can write the code as follows:

```python
from semantix import Semantic, with_llm
from semantix.llms import OpenAI
from typing import Tuple

@with_llm(llm=OpenAI())
def generate_joke() -> Semantic[Tuple[str, str], "Joke and Punchline"]:
    ...
```

As you can see, i have left the `meaning` parameter of the `with_llm` empty. This is because the function name provides enough context about the task. However, in the case of the output type, i have used a Semantic Type to provide more context to the LLM. because if I just provide `Tuple[str, str]` the LLM will not know the order the Joker and the Punchline should be in the output.

If you have a function that is not self explanatory, you can use the `meaning` parameter to provide more context to the LLM. For example:

```python
@with_llm("Generate a joke with a punchline", llm=OpenAI())
def generate() -> Semantic[Tuple[str, str], "Joke and Punchline"]:
    ...
```

Same goes for the function signatures. If the function signature is not self explanatory, you can use Semantic Types to provide more context to the LLMs. For example:

```python
from semantix.types import Image

@with_llm("Asnwer the given question about the image", llm=OpenAI())
def answer_question(image: Image, q: Semantic[str, "Question"]) -> str:
    ...
```

Here the `image` parameter provides enough context to the LLM, but the `q` parameter does not. So, I have used a Semantic Type to provide more context to the LLM.

This is also true for Custom Types, Classes, and Enums. If you have a custom type that is not self explanatory, you can use a Docstring to provide more context to the LLMs. For example:

```python
class Item:
    """Represents an item in the inventory."""
    # rest of the code

class Fruit:
    # Doesnt require a docstring as the name is self explanatory
    # rest of the code
```

## When to use Semantix?

Semantix is a powerful tool, but that doesn't mean you should use it for every task.

Here are some guidelines on when to use Semantix:
- When you want structured outputs from LLMs.
- WHen you want reasoning capabilities in your code.
- When you want to prototype GenAI Applications quickly.

Here are some guidelines on when not to use Semantix:
- For tasks that can be achieved through SOTA libraries. eg: Object Detection, PII Detection, etc.
- For basic chatbots, etc. Use something like Langchain, Rasa or Dialogflow.
- For basic NLP tasks. Use something like Huggingface Transformers, Spacy, etc.

Rest of the best practices are same as the Python best practices.
