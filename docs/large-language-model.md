# Large Language Model

Large Language Models plays crucial role in the workings of Semantix. It can be considered as one of the pillars holding the bridge between traditional
programs and AI-powered Programs. LLMs allows the semantix library with Symbolic AI Reasoning capabilities to understand the context of what you are
trying to achieve through your code.

!> Semantix doesnt send your code to any external servers. All the processing is done locally on your machine only the Meaning Typed prompts are sent to the LLMs.

## What LLMS are supported?

Semantix currently tested on the following LLMs but not limited to:
1. OpenAI's GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o, GPT-4o-mini
2. Anthropic's Claude 3 Haiku, Sonnet, Opus and Claude 3.5 Sonnet
3. Cohere's Command-R, Command-R-plus
4. Google's Gemini Models
5. Mistral's mixtral models
6. Opensource Models like Gemma-2b, LLama-3-8b and up, Llama-3.1 models

We are constantly adding support for more models. If you want to add support for a specific model, please open an issue on the GitHub repository.

## How to use LLMs in Semantix?

Initiating a LLM is very simple in Semantix. You just need to import the correct model from the `semantix.llms` module.

for example, to use OpenAI's GPT-4o-mini model, you can do the following:

!> Makesure necessary dependencies are installed for the model you are trying to use. for example, to use OpenAI models you need to run `pip install semantix[openai]`.

```python
from semantix.llms import OpenAI

llm = OpenAI(model="gpt-4o-mini") # GPT-4o-mini is the default model
```

Similarly, you can use other models by importing them from the `semantix.llms` module.

!> Make sure you have the necessary API keys or access tokens for the models you want to use. You can find more information about obtaining API keys in the documentation of the respective models.

## How to use Local LLMs?

Semantix also supports huggingface and ollama models. You can use them by importing them from the `semantix.llms` module.
for example, to use Gemma 2b model locally through huggingface, you can do the following:

```python
from semantix.llms import Huggingface

llm = Huggingface(model="google/gemma-2b")
```

!> Some models requires permission to access. Make sure to follow the necessary steps to access the models.


## How to pass different parameters to LLMs?

You can pass different parameters to the LLMs easily in the initialization of the LLM object. For example, to set the temperature parameter for the GPT-4o-mini model, you can do the following:

```python
llm = OpenAI(model="gpt-4o-mini", temperature=0.5)
```

In here, the temperature parameter is set to 0.5 and is considered as the default value for the model for the rest of the sessions. This can be overridden in different usecases. for example, you can override the temperature parameter for 2 with llm function call as follows:

```python
@with_llm("Some meaning", llm, temperature=0.7)
def some_function():
    ...

@with_llm("Some other meaning", llm, temperature=0.3)
def some_other_function():
    ...

@with_llm("Some other other meaning", llm)
def some_other_other_function():
    ...
```

In here, the `some_function` will use the temperature parameter as 0.7, `some_other_function` will use the temperature parameter as 0.3 and `some_other_other_function` will use the default temperature parameter (0.5) set in the initialization of the llm object.

To check the supported parameters for the LLMs, you can refer to the documentation of the respective models.

## How to create a custom LLM?

Each and every in-built LLM classes in Semantix are subclasses of the `BaseLLM` class. You can create your own custom LLM by subclassing the `BaseLLM` class and implementing the necessary methods.

For example, to create a custom LLM that always returns a name of the model, you can do the following:

```python
from semantix.llms import BaseLLM

class CustomLLM(BaseLLM):
    def __init__(self,verbose: bool = False, max_retries: int = 3, model: str = "custom_model", **kwargs):
        super().__init__(verbose, max_retries)
        self.default_params = {
            "model": model,
            **kwargs
        }

    def __infer__(self, messages: list, model_params: dict={}) -> str: # This signature should be maintained
        return self.default_params["model"]
```

In here, the `CustomLLM` class is created by subclassing the `BaseLLM` class and implementing the `__infer__` method. The `__infer__` method should take a list of messages and a dictionary of model parameters (dict) as input and return the inferred output (str).

You can add the necessary code to initialize your custom model inside the `__init__` method. You can also add any additional methods or properties to the custom LLM class as needed.

In order for semantix to correctly use your custom LLM, __infer__ method should be implemented with the same signature as above. The `model_params` parameter is used to pass the parameters to the LLM. and your inference logic should be implemented inside the `__infer__` method.

Nor you can use your custom LLM as same as the in-built LLMs in Semantix.

```python
llm = CustomLLM(model="my_custom_model")
```
