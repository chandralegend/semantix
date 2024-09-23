# Utilities API Reference

The `semantix.utils` module provides a set of utility functions.

## create_class

```python
create_class(
    classname: str, fields: Dict[str, Tuple[Type, Any]], desc: str = ""
) -> Any
```

Helper function to create a new dataclass with the given fields. This is helpful when creating classes with a large number of parameters.

### Parameters

- `classname` (str): The name of the class to be created.
- `fields` (Dict[str, Tuple[Type, Any]): A dictionary containing the fields of the class, where the keys are the field names and the values are tuples containing the field type and default value.
- `desc` (str, optional): Description of the class. Defaults to "".

## Example

```python
from semantix.utils import create_class

fields = {
    "name": (str, ""),
    "age": (int, 0),
    "height": (float, 0.0),
}

Person = create_class("Person", fields, "A class to represent a person.")
```

## create_enum

```python
create_enum(
    classname: str, fields: Dict[str, Any], desc: str = ""
) -> Enum
```

Helper function to create a new Enum with the given fields. This is helpful when creating enums with a large number of values.

### Parameters

- `classname` (str): The name of the Enum class.
- `fields` (Dict[str, Any]): A dictionary containing the field names and values for the Enum.
- `desc` (str, optional): A description for the Enum. Defaults to "".

## Example

```python
from semantix.utils import create_enum

colors = {
    "RED": 1,
    "GREEN": 2,
    "BLUE": 3,
}

Color = create_enum("Color", colors, "An Enum to represent colors.")
```
