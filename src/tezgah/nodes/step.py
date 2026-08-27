import inspect
import time

from ..errors import ContractError
from .base import Node, make_binding, named_params, node_executor, node_label, str_list


def call_with_retries(fn, call_args, retries, wait):
    attempt = 0
    while True:
        try:
            return fn(**call_args)
        except Exception:
            if attempt >= retries:
                raise
            attempt += 1
            if wait:
                time.sleep(wait)


class Step(Node):
    leaf = True

    def __init__(self, fn, inputs=None, outputs=None, name=None, when=None,
                 wait_for=None, executor=None, retries=0, wait=0.0):
        if not callable(fn):
            raise TypeError(f"Step fn must be callable, got {type(fn).__name__}")
        self.fn = fn
        fn_name = getattr(fn, "__name__", None)

        if name is None:
            name = fn_name
        if name is None:
            raise TypeError(f"Step: {fn!r} has no __name__, pass name explicitly")
        self.name, label = node_label("Step", name)

        try:
            self.signature = inspect.signature(fn)
        except (TypeError, ValueError):
            self.signature = None

        self.inputs = make_binding(self.signature, inputs, label)

        if outputs is None:
            if fn_name is None:
                raise TypeError(f"{label}: {fn!r} has no __name__, pass outputs explicitly")
            outputs = [fn_name]
        else:
            outputs = str_list(outputs, label, "outputs")
        if len(set(outputs)) != len(outputs):
            raise ValueError(f"{label}: duplicate keys in outputs {outputs}")
        self.outputs = outputs

        if when is not None and not callable(when):
            raise TypeError(f"{label}: when must be callable, got {type(when).__name__}")
        self.when = when
        if when is None:
            self.when_params = []
        else:
            try:
                when_signature = inspect.signature(when)
            except (TypeError, ValueError):
                raise TypeError(f"{label}: cannot read the signature of when, use a callable with named parameters") from None
            self.when_params = [param.name for param in named_params(when_signature)]

        self.wait_for = [] if wait_for is None else str_list(wait_for, label, "wait_for")
        self.executor = node_executor(executor, label)

        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError(f"{label}: retries must be an int >= 0, got {retries!r}")
        if not isinstance(wait, (int, float)) or isinstance(wait, bool) or wait < 0:
            raise ValueError(f"{label}: wait must be a number >= 0, got {wait!r}")
        self.retries = retries
        self.wait = wait

    def reads(self):
        return list(dict.fromkeys(list(self.inputs.values()) + self.when_params))

    def writes(self):
        return list(self.outputs)

    def execute(self, values):
        if self.when is not None:
            if not self.when(**{param: values[param] for param in self.when_params}):
                return "skipped", {}, 0.0, {}

        call_args = {param: values[key] for param, key in self.inputs.items()}
        start = time.perf_counter()
        result = call_with_retries(self.fn, call_args, self.retries, self.wait)
        ms = (time.perf_counter() - start) * 1000

        if not self.outputs:
            return "ok", {}, ms, {}
        if len(self.outputs) == 1:
            return "ok", {self.outputs[0]: result}, ms, {}
        if not isinstance(result, dict):
            raise ContractError(
                f"{self.name}: {len(self.outputs)} outputs declared, fn must return a mapping, got {type(result).__name__}"
            )
        missing = [key for key in self.outputs if key not in result]
        if missing:
            raise ContractError(f"{self.name}: returned mapping is missing output keys {missing}")
        return "ok", {key: result[key] for key in self.outputs}, ms, {}
