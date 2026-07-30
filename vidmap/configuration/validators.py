"""Pydantic dataclass decorator with shallow repr for Options classes."""

import dataclasses

from pydantic.dataclasses import dataclass as _pydantic_dc
from pydantic_core import ArgsKwargs


def _shallow_repr(self):
    """Repr that collapses nested dataclasses to ClassName(...)."""
    parts = []
    for f in dataclasses.fields(self):
        val = getattr(self, f.name)
        if dataclasses.is_dataclass(val) and not isinstance(val, type):
            parts.append(f"{f.name}={type(val).__name__}(...)")
        else:
            parts.append(f"{f.name}={val!r}")
    return f"{type(self).__name__}({', '.join(parts)})"


def dataclass(_cls=None, /, **kwargs):
    """Pydantic dataclass with shallow repr (nested dataclasses shown as ClassName(...))."""

    def wrap(c):
        dc = _pydantic_dc(c, **kwargs)
        dc.__repr__ = _shallow_repr
        return dc

    if _cls is None:
        return wrap
    return wrap(_cls)


def instantiate_nested_options(raw, nested_types):
    """Instantiate declared nested dataclasses from dictionary values."""
    is_args_kwargs = isinstance(raw, ArgsKwargs)
    if is_args_kwargs:
        values = {} if raw.kwargs is None else dict(raw.kwargs)
    elif isinstance(raw, dict):
        values = dict(raw)
    else:
        return raw

    for name, options_type in nested_types.items():
        if name in values and isinstance(values[name], dict):
            values[name] = options_type(**values[name])

    if is_args_kwargs:
        args = () if raw.args is None else raw.args
        return ArgsKwargs(args, values)
    return values
