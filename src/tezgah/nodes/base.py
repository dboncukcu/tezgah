import inspect


class Node:
    name: str
    leaf: bool = True

    def reads(self) -> list[str]:
        raise NotImplementedError

    def writes(self) -> list[str]:
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError


def named_params(signature):
    return [
        param
        for param in signature.parameters.values()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def str_list(value, owner, field):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"{owner}: {field} entries must be strings, got {item!r}")
        return list(value)
    raise TypeError(f"{owner}: {field} must be a string or a list of strings, got {type(value).__name__}")


def str_map(value, owner, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{owner}: {field} must be a dict, got {type(value).__name__}")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"{owner}: {field} must map str to str, got {key!r}: {item!r}")
    return dict(value)


def key_map(value, owner, field):
    if value is None:
        return {}
    if isinstance(value, dict):
        return str_map(value, owner, field)
    return {key: key for key in str_list(value, owner, field)}


def node_label(kind, name):
    if not isinstance(name, str):
        raise TypeError(f"{kind} name must be a string, got {type(name).__name__}")
    return name, f"{kind} '{name}'"


def make_binding(signature, inputs, owner):
    if inputs is None:
        if signature is None:
            raise TypeError(f"{owner}: cannot read the signature, pass inputs explicitly")
        return {param.name: param.name for param in named_params(signature)}
    if isinstance(inputs, dict):
        return str_map(inputs, owner, "inputs")
    return {key: key for key in str_list(inputs, owner, "inputs")}


def node_executor(value, owner):
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError(f"{owner}: executor must be an executor instance (an object with submit()), not a string; executor names belong to run()")
    if not hasattr(value, "submit"):
        raise TypeError(f"{owner}: executor must have a submit() method, got {type(value).__name__}")
    return value


def child_path(parent, name):
    return f"{parent}.{name}" if parent else name


def as_node(value, owner):
    from .step import Step

    if isinstance(value, Node):
        return value
    if callable(value):
        return Step(value)
    raise TypeError(
        f"{owner}: expected Step, Pipeline, Map, Loop, Branch or a callable, got {type(value).__name__}"
    )
