# Tools

## What is a tool?

A tool is a glorified function that can help your Symbolic AI to get additional information or perform a specific task.
Tools can be used to enhance the capabilities of your Symbolic AI model, allowing it to perform more complex reasoning
and inference tasks.

## Why do we need tools?

Other than providing additional information, tools are essential because most of the current symbolic AI models are not
good at handling tasks such as Mathematics, etc. Tools can help your Symbolic AI model to perform these tasks with ease.

## How to use tools?

There are two ways to create tools in Semantix:

### 1. Use Inbuilt tools

Semantix comes with a set of inbuilt tools that you can use out of the box. These tools are designed to help you perform
common tasks such as Mathematics, Getting Information from wikipedia, Reading a file, Writing a file, Searching on the web, etc.

Here is an example of how you can use an inbuilt tool in Semantix:

```python
from semantix.tools import math
from semantix.tools.wikipedia import search, summary
from semantix.tools.file_utils import read_file, write_file
from semantix.tools.serper import search
from semantix.tools.exa_search import search
```

### 2. Create Custom tools

You can also create your own custom tools in Semantix. To create a custom tool, you need to define a function that takes
the necessary inputs and returns the desired output. You can then use the `@tool` decorator to register the function as a tool.

Here is an example of how you can create a custom tool in Semantix:

```python
from semantix import tool

@tool('adds two numbers')
def add(a: int, b: int) -> int:
    return a + b
```

You can make more expressive tools by using Semantic types in your tools. Here is an example of how you can create a tool
that takes a sentence and a target language and translates the sentence to the target language:

```python
from semantix import tool, Semantic

@tool('translates a sentence to a target language')
def translate(sentence: Semantic[str, 'Sentence'], target_language: Semantic[str, 'Language']) -> Semantic[str, 'Translated Sentence']:
    # Code to translate the sentence to the target language
```

Also you can use your traditional functions as tools too. But make sure to use type hints, and descriptive variable names to make your tools more expressive.
