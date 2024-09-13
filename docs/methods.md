# Methods

Methods in semantix are different instructons that you can use in `with_llm` to get most out of the models. Currently we support the following methods:

## Normal

This is the default method that is used when no method is specified. In this method, the model will try to generate the output directly without any additional thinking process.
This is suitable for simple taks such as translation, summarization, etc.

```python
@with_llm("Translate the given sentence to Arabic", llm=OpenAI(), method="Normal")
def translate_to_arabic(sentence: str) -> Semantic[str, "Arabic Sentence"]:
    ...
```

## Reason

This method is used when you want the model to reason about the input and generate the output. This is suitable for tasks that require reasoning such as question answering, etc. but if the task is very hard, this method will not work as expected.

```python
@with_llm("Answer the given question", llm=OpenAI(), method="Reason")
def answer_question(question: str) -> str:
    ...
```

## Chain-of-Thoughts

This method generates the output after a series of thinking steps. This is suitable for tasks that require multiple steps of thinking such as math problems, etc.

```python
@with_llm("Solve the given math problem", llm=OpenAI(), method="Chain-of-Thoughts")
def solve_math_problem(problem: str) -> Semantic[str, "Solution"]:
    ...
```

## ReAct

This method is a iterative method. In order to use this, you need to provide the `with_llm` decorator with a list of tools that you want the llm to use in the thinking process. LLM are notorious for failing in tasks such as calculations, by this method you can provide amath tool such that llm can deligate the task to the tool and get the output and continue the thinking process. Also this method is useful when you want the llm to acquire additional context from the internet or other sources.

```python
@with_llm("Answer the given question", llm=OpenAI(), method="ReAct", tools=[google_search, wikipedia_summary])
def answer_question(question: str) -> str:
    ...

answer_question("Who is the wife of the current president of the United States?")
```

This will first use the google search tool to get the name of the current president of the United States and then use the name to get the name of the wife of the president from wikipedia.

## Reflection (Coming Soon)

This method is used when you want the model to reflect on the output and correct it if necessary. This is suitable for tasks that require reflection such as writing, etc.

```python
@with_llm("Write a short story", llm=OpenAI(), method="Reflection")
def write_short_story() -> str:
    """Make sure the story is less than 3 sentences"""
    ...
```
