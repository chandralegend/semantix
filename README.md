<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://i.ibb.co/NT3Xbfp/2.png">
    <source media="(prefers-color-scheme: light)" srcset="https://i.ibb.co/SR2hqgh/1.png">
    <img alt="Semantix: Infusing Meaning into Code with Large Language Models" width="500px" src="https://i.ibb.co/SR2hqgh/1.png">
  </picture>

  [![PyPI version](https://img.shields.io/pypi/v/semantix.svg)](https://pypi.org/project/semantix/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandralegend/semantix/blob/main/try.ipynb) ![License](https://img.shields.io/badge/License-MIT-blue.svg)
</div>

Semantix empowers developers to infuse meaning into their code through enhanced variable typing (semantic typing). By leveraging the power of large language models (LLMs) behind the scenes, Semantix transforms ordinary functions into intelligent, context-aware operations without explicit LLM calls.

## Key Features:

- **Semantic Type System**: Define rich, meaningful types that carry contextual information.
- **Function Enhancement**: Automatically augment functions with LLM-powered capabilities.
- **Implicit Intelligence**: Leverage advanced NLP and reasoning without direct API calls.
- **Developer-Friendly**: Seamlessly integrate into existing Python codebases with minimal overhead.

Semantix bridges the gap between traditional programming and AI-assisted development, allowing you to write more expressive, powerful code with ease.

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
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for a quick guide on how to contribute to Semantix.
