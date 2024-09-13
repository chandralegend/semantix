# Installation

All you need is:

```bash
pip install semantix
```

By default, Semantix doesn't install any llm packages. You can install them separately:

<!-- tabs:start -->

#### **OpenAI**
```bash
pip install semantix[openai]
```
#### **Anthropic**
```bash
pip install semantix[anthropic]
```
#### **Groq**
```bash
pip install semantix[groq]
```
#### **Cohere**
```bash
pip install semantix[cohere]
```
#### **TogetherAI**
```bash
pip install semantix[togetherai]
```
#### **MistralAI**
```bash
pip install semantix[mistralai]
```
#### **HuggingFace**
```bash
pip install semantix[huggingface]
```

<!-- tabs:end -->

If you want to use multiple llm packages, you can install them together:

```bash
pip install semantix[openai, anthropic, ...]
```

If you want to use MultiModal capabilities, you can install the following:

```bash
pip install semantix[image, video]
```
