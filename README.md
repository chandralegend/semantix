<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/light.png">
    <img alt="Semantix: Infusing Meaning into Code with Large Language Models" width="500px" src="https://i.ibb.co/SR2hqgh/1.png">
  </picture>

  [![PyPI version](https://img.shields.io/pypi/v/semantix.svg)](https://pypi.org/project/semantix/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandralegend/semantix/blob/main/try.ipynb) ![License](https://img.shields.io/badge/License-MIT-blue.svg)

Semantix provides a simple but powerful way to infuse meaning into functions, variables and classes to leverage the power of Large Language models to generate structured typed outputs `without the need of JSON Schema or any other abstractions.`

</div>

## Key Features:

- **Semantic Type**: Add meaning to your variables. No need of additional abstractions like `InputField`, `OutputField` etc.
- **AutoPrompting**: Semantix Generate prompts using the `Meaning Typed Prompting` Technique.
- **Supercharged Functions**: Automatically augment functions enhance-powered capabilities. No Function body is needed.
- **Minimal Overhead**: Seamlessly integrate into existing Python codebases with minimal overhead.

## Minimal Example

```python
from enum import Enum
from dataclasses import dataclass

from semantix import Semantic, enhance
from semantix.llms import OpenAI
from semantix.types import Image

llm = OpenAI()

class Personality(Enum):
    """Personality of the Person"""

    INTROVERT = "Introvert"
    EXTROVERT = "Extrovert"

@dataclass
class Person:
    full_name: str
    yod: Semantic[int, "Year of Death"]
    personality: Semantic[Personality, "Personality of the Person"]

@enhance("Get Person Informations use common knowledge", llm)
def get_person(name: Semantic[str, "Name of the Person"]) -> Person:
    ...

person_obj = get_person(name="Albert Einstein")
print(f"{person_obj.full_name} is an {person_obj.personality.value} who died in {person_obj.yod}")
# Albert Einstein is an Introvert who died in 1955
```

## Supports Vision

```python
from semantix.types import Image

@enhance("Get Person Informations use common knowledge", llm)
def get_person(img: Semantic[Image, "Image of the Person"]) -> Person:
    ...

person_obj = get_person(img=Image("mandela.jpg"))
print(f"{person_obj.full_name} is an {person_obj.personality.value} who died in {person_obj.yod}")
# Nelson Mandela is an Extrovert who died in 2013
```

## Installation
All you need is:

```bash
pip install semantix
```

To install the very latest from `main`:

```bash
pip install git+https://github.com/chandralegend/semantix.git
````

Or open our intro notebook in Google Colab: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/chandralegend/semantix/blob/main/try.ipynb)

By default, Semantix doesn't install any llm packages. You can install them separately:

```bash
pip install semantix[openai]
pip install semantix[anthropic]
pip install semantix[openai, anthropic] # Install both
```

If you want to use MultiModal capabilities, you can install the following:

```bash
pip install semantix[image]
pip install semantix[video]
```

## Citation

If you find Semantix helpful, give it a ⭐️ on [GitHub](https://github/chandralegend/semantix)!
and If you have used Semantix in your project, add the badge to your README.md file.

![https://github/chandralegend/semantix](https://img.shields.io/badge/Powered%20by-Semantix-8A2BE2)

```markdown
[![https://github/chandralegend/semantix](https://img.shields.io/badge/Powered%20by-Semantix-8A2BE2)](https://github/chandralegend/semantix)
```

If you used Semantix in your research, please cite it as follows:

```bibtex
@misc{semantix,
  author = {Chandra Irugalbandara},
  title = {Semantix: Infusing Meaning into Code with Large Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github/chandralegend/semantix}}
}
```

## Contributing
Please read [CONTRIBUTING.md](docs/contributing.md) for a quick guide on how to contribute to Semantix.
