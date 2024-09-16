# LLMs API Reference

This page documents the API reference for the `semantix.llms` module.

## BaseLLM

Base class to represent the Large Language Model. Every LLM should inherit from this class.

### Parameters

- `verbose` : bool, optional
    - Whether to print the logs, input prompts, outputs. Default is `False`.
- `max_retries` : int, optional
    - max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.

### Example

```python
from semantix.llms import BaseLLM

class MyLLM(BaseLLM):
    def __init__(self, verbose=False, max_retries=3, **kwargs):
        super().__init__(verbose=verbose, max_retries=max_retries)
        # Your code here
```

## OpenAI

A class to represent the OpenAI Large Language Model.

### Parameters

- `verbose` : bool, optional
    - Whether to print the logs, input prompts, outputs. Default is `False`.
- `max_retries` : int, optional
    - max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
- `model` : str, optional
    - The model to use. Default is `"gpt-4o-mini"`. Currentyl only chat models are supported. Check the [OpenAI API](https://arc.net/l/quote/gkgqwbpgt) for more details.
- `api_key` : str, optional
    - The API key to use. Default is `None`. If `None`, it will look for the `OPENAI_API_KEY` environment variable.
- `**kwargs`
    - Any Default parameters to be used in inference. Check the [OpenAI API](https://platform.openai.com/docs/api-reference/chat) for more details.

### Example

```python
from semantix.llms import OpenAI

llm = OpenAI(verbose=True, max_retries=5, model="gpt-4o-mini", api_key="YOUR_API_KEY", temperature=0.5)
```

## Anthropic

A class to represent the Anthropic Large Language Model.

### Parameters

- `verbose` : bool, optional
    - Whether to print the logs, input prompts, outputs. Default is `False`.
- `max_retries` : int, optional
    - max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
- `model` : str, optional
    - The model to use. Default is `"claude-3-5-sonnet-20240620"`. Check the [Anthropic API](https://docs.anthropic.com/en/docs/about-claude/models) for more details.
- `api_key` : str, optional
    - The API key to use. Default is `None`. If `None`, it will look for the `ANTHROPIC_API_KEY` environment variable.
- `**kwargs`
    - Any Default parameters to be used in inference. Check the [Anthropic API](https://docs.anthropic.com/en/api/messages) for more details.

### Example

```python
from semantix.llms import Anthropic

llm = Anthropic(verbose=True, max_retries=5, model="claude-3-5-sonnet-20240620", api_key="YOUR_API_KEY", temperature=0.5)
```

## Cohere

A class to represent the Cohere Large Language Model.

### Parameters

- `verbose` : bool, optional
    - Whether to print the logs, input prompts, outputs. Default is `False`.
- `max_retries` : int, optional
    - max_retries (int, optional): The maximum number of self healing steps allowed. Defaults to 3.
- `model` : str, optional
    - The model to use. Default is `"command-r-plus-08-2024"`. Check the [Cohere API](https://docs.cohere.com/docs/models) for more details.
- `api_key` : str, optional
    - The API key to use. Default is `None`. If `None`, it will look for the `COHERE_API_KEY` environment variable.
- `**kwargs`
    - Any Default parameters to be used in inference. Check the [Cohere API](https://docs.cohere.com/reference/chat) for more details.

### Example

```python
from semantix.llms import Cohere

llm = Cohere(verbose=True, max_retries=5, model="command-r-plus-08-2024", api_key="YOUR_API_KEY", temperature=0.5)
```
