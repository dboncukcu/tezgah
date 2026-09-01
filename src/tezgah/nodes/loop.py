import builtins
import inspect
import time

from .base import Node, as_node, child_path, key_map, named_params, node_executor, node_label, str_list


def _as_range(spec, label):
    if isinstance(spec, builtins.range):
        return spec
    if isinstance(spec, bool):
        raise TypeError(f"{label}: range must be an int, [start, stop], [start, stop, step] or a range, got a bool")
    if isinstance(spec, int):
        return builtins.range(spec)
    if isinstance(spec, (list, tuple)) and 1 <= len(spec) <= 3:
        parts = list(spec)
        for part in parts:
            if not isinstance(part, int) or isinstance(part, bool):
                raise TypeError(f"{label}: range parts must be ints, got {part!r}")
        if len(parts) == 3 and parts[2] == 0:
            raise ValueError(f"{label}: range step cannot be zero")
        return builtins.range(*parts)
    raise TypeError(
        f"{label}: range must be an int, [start, stop], [start, stop, step] or a range, got {type(spec).__name__}"
    )


class Loop(Node):
    leaf = False

    def __init__(self, body, carry, range, index=None, until=None, trace=None,
                 outputs=None, executor=None, name="loop", wait_for=None):
        self.name, label = node_label("Loop", name)

        self.body = as_node(body, f"{label} body")
        self.carry = key_map(carry, label, "carry")
        if not self.carry:
            raise ValueError(f"{label}: carry cannot be empty, a loop carries state from turn to turn")

        self.turns = _as_range(range, label)
        if index is not None and not isinstance(index, str):
            raise TypeError(f"{label}: index must be a string, got {type(index).__name__}")
        if index is not None and index in self.carry:
            raise ValueError(f"{label}: index and carry cannot share the name '{index}'")
        self.index = index

        if until is not None and not callable(until):
            raise TypeError(f"{label}: until must be callable, got {type(until).__name__}")
        self.until = until
        if until is None:
            self.until_params = []
        else:
            try:
                until_signature = inspect.signature(until)
            except (TypeError, ValueError):
                raise TypeError(f"{label}: cannot read the signature of until, use a callable with named parameters") from None
            self.until_params = [param.name for param in named_params(until_signature)]

        self.trace = key_map(trace, label, "trace")
        self.outputs = key_map(outputs, label, "outputs")
        written = list(self.trace.values()) + list(self.outputs.values())
        if len(set(written)) != len(written):
            raise ValueError(f"{label}: trace and outputs write overlapping parent keys {written}")

        self.wait_for = [] if wait_for is None else str_list(wait_for, label, "wait_for")
        self.executor = node_executor(executor, label)

    def broadcast_keys(self):
        skip = set(self.carry) | ({self.index} if self.index else set())
        return [key for key in self.body.reads() if key not in skip]

    def reads(self):
        return list(dict.fromkeys(list(self.carry.values()) + self.broadcast_keys()))

    def writes(self):
        return list(dict.fromkeys(list(self.trace.values()) + list(self.outputs.values())))

    def execute(self, values, kernel, path, pool=None):
        start = time.perf_counter()
        carry = {name: values[outer] for name, outer in self.carry.items()}
        broadcast = {key: values[key] for key in self.broadcast_keys()}
        body_pool = self.body.executor or self.executor or pool
        traces = {outer: [] for outer in self.trace.values()}
        turn_records = []
        status = "ok"

        for position, value in enumerate(self.turns):
            frame = {**carry, **broadcast}
            if self.index:
                frame[self.index] = value
            turn_path = f"{path}[{position}]"
            body_path = child_path(turn_path, self.body.name)
            kernel.emit(turn_path, "iter_started", node=self.name)
            future = kernel.submit(self.body, frame, body_path, body_pool)
            kernel.wait([future])
            turn_status, written, record = kernel.deliver(future)
            kernel.emit(turn_path, "iter_finished", node=self.name, status=turn_status, ms=record.get("ms", 0))
            turn_records.append(record)
            if turn_status == "failed":
                status = "failed"
                break
            for name in self.carry:
                carry[name] = written[name]
            for inner, outer in self.trace.items():
                traces[outer].append(written[inner])
            if self.until is not None and self.until(**{p: written[p] for p in self.until_params}):
                break

        ms = (time.perf_counter() - start) * 1000
        if status == "failed":
            written = {}
        else:
            written = {outer: carry[name] for name, outer in self.outputs.items()}
            written.update({outer: traces[outer] for outer in traces})
        extra = {"turns": turn_records, "traces": dict(traces)}
        return status, written, ms, extra
