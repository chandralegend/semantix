# Semantic Type

Semantix allows you to define rich, meaningful types that carry contextual information to traditional types. This enables
Symbolic AI capabilities in your codebase, allowing you to write more expressive, powerful code with ease.

## Why Semantic Types?

Traditional programming languages have a limited type system that only allows you to define basic types like `int`, `str`,
`float`, etc. These types are useful for basic operations, but they lack the ability to carry additional context or meaning.
For example, consider the following function signature:

```python
def add(a: int, b: int) -> int:
    return a + b
```

In this function, the types `int` are used to represent the input and output values. However, these types do not convey any
additional information about the meaning or context of the values. What if we wanted to specify that `a` and `b` represent
temperatures in Celsius, or that the result of the addition should be a temperature in Fahrenheit? Traditional types are
insufficient for expressing this kind of rich, contextual information.

Semantic types address this limitation by allowing you to define types that carry additional meaning or context. For example,
you could define a semantic type `Temperature` that represents a temperature value along with its unit of measurement. This
allows you to write functions that operate on temperature values in a way that is more expressive and meaningful:

```python
from semantix import Semantic

def add(
    a: Semantic[int, 'Temperature in Celsius'],
    b: Semantic[int, 'Temperature in Celsius']
) -> Semantic[int, 'Temperature in Fahrenheit']:
    ...
```

In this function, the semantic types `Temperature in Celsius` and `Temperature in Fahrenheit` provide additional context about
the meaning of the input and output values. This makes the function more expressive and allows Symbolic AI models to reason
about the values in a more intelligent way.

## How to Define Semantic Types

Semantic types in Semantix are defined using the `Semantic` class, which takes two type arguments: the base type and a string
that represents the semantic meaning of the type. For example, you can define a semantic type `Temperature` as follows:

```python
from semantix import Semantic

Temperature = Semantic[float, 'Temperature']
```

This defines a semantic type `Temperature` that represents a floating-point value with the semantic meaning of a temperature.

You can then use this semantic type in function signatures to provide additional context about the meaning of the values:

```python

def convert_to_fahrenheit(
    temp_celsius: Semantic[float, 'Temperature in Celsius']
) -> Semantic[float, 'Temperature in Fahrenheit']:
    ...
```

You can use nested types to define more complex semantic types. For example, you could define a list of temperatures as follows:

```python
from semantix import Semantic

Temperature = Semantic[float, 'Temperature']
TemperatureList = Semantic[list[float], 'List of Temperatures']
```

!>  For python 3.9 and lower, you need to use `typing.List` instead of `list`.

By using semantic types in your code, you can unlock the full power of Symbolic AI and write more expressive, powerful code.
