# FAQ

### Whats the Difference between `semantix` and other libraries such as `DSPy`, `Instructor` etc?

Libraries such as `DSPy`, `Instructor` etc. are JSON Schema based Structured Output Generators. Though they are powerful, they come with few caveats.

1. Uses PyDantic Models. Though pydantic provide robust type validation, uses have to extend all the types to `BaseModel` which is not always necessary.
2. Unnecessary Abstractions. Libraries like `DSPy` uses unnecessary abstractions to embed meaning into variables (`InputField`, `OutputField` etc). Classes like `dspy.Signature` are not necessary because already python function signatures, type hints and output type hints are expressive enough to convey the objective.
3. In most LLMs, characters such as `{`, `}`, `:`, `"` etc are considered as seperate tokens. So in the inference you will be charged for each of these tokens alot because JSON Schema are heavily composed of these characters.

`Semantix` on the other hand uses `Semantic` types to embed meaning into variables. `Semantic` types are just a wrapper around the original type and a string that represents the meaning of the type. This allows you to write more expressive, powerful code with ease without any use of unnecessary abstractions. and it is not a must as well. You can use `Semantic` types only when you need to embed meaning into variables.
