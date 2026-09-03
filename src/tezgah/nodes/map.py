import time

from ..errors import ContractError
from .base import Node, as_node, child_path, key_map, node_executor, node_label, str_list


class Map(Node):
    leaf = False

    def __init__(self, body, over, item="item", index=None, collect=None,
                 parallel=False, executor=None, name="map", wait_for=None):
        self.name, label = node_label("Map", name)

        self.body = as_node(body, f"{label} body")
        if not isinstance(over, str):
            raise TypeError(f"{label}: over must be a bus key string, got {type(over).__name__}")
        self.over = over
        if not isinstance(item, str):
            raise TypeError(f"{label}: item must be a string, got {type(item).__name__}")
        self.item = item
        if index is not None and not isinstance(index, str):
            raise TypeError(f"{label}: index must be a string, got {type(index).__name__}")
        if index == item:
            raise ValueError(f"{label}: item and index cannot share the name '{item}'")
        self.index = index
        self.collect = key_map(collect, label, "collect")
        parent_keys = list(self.collect.values())
        if len(set(parent_keys)) != len(parent_keys):
            raise ValueError(f"{label}: two collect entries write the same parent key {parent_keys}")

        if isinstance(parallel, bool):
            self.parallel = parallel
        elif isinstance(parallel, int):
            if parallel < 1:
                raise ValueError(f"{label}: parallel must be False, True or an int >= 1, got {parallel!r}")
            self.parallel = parallel
        else:
            raise TypeError(f"{label}: parallel must be False, True or an int, got {type(parallel).__name__}")

        self.wait_for = [] if wait_for is None else str_list(wait_for, label, "wait_for")
        self.executor = node_executor(executor, label)

    def broadcast_keys(self):
        skip = {self.item} | ({self.index} if self.index else set())
        return [key for key in self.body.reads() if key not in skip]

    def reads(self):
        return list(dict.fromkeys([self.over] + self.broadcast_keys()))

    def writes(self):
        return list(self.collect.values())

    def window(self, kernel):
        if isinstance(self.parallel, bool):
            return kernel.window if self.parallel else 1
        return self.parallel

    def execute(self, values, kernel, path, pool=None):
        start = time.perf_counter()
        collection = values[self.over]
        try:
            items = list(collection)
        except TypeError:
            raise ContractError(
                f"{self.name}: over key '{self.over}' is not iterable, got {type(collection).__name__}"
            )

        broadcast = {key: values[key] for key in self.broadcast_keys()}
        body_pool = self.body.executor or self.executor or pool
        results = {outer: [None] * len(items) for outer in self.collect.values()}
        records: list = [None] * len(items)
        flying = {}
        next_index = 0
        broken = False

        while next_index < len(items) or flying:
            while not broken and len(flying) < self.window(kernel) and next_index < len(items):
                i = next_index
                iteration = {self.item: items[i], **broadcast}
                if self.index:
                    iteration[self.index] = i
                iter_path = f"{path}[{i}]"
                body_path = child_path(iter_path, self.body.name)
                kernel.emit(iter_path, "iter_started", node=self.name)
                flying[kernel.submit(self.body, iteration, body_path, body_pool)] = i
                next_index += 1
            if not flying:
                break
            for future in kernel.wait(list(flying)):
                i = flying.pop(future)
                status, written, record = kernel.deliver(future)
                records[i] = record
                kernel.emit(f"{path}[{i}]", "iter_finished", node=self.name, status=status, ms=record.get("ms", 0))
                if status == "failed":
                    broken = True
                    continue
                for inner, outer in self.collect.items():
                    results[outer][i] = written[inner]

        ms = (time.perf_counter() - start) * 1000
        status = "failed" if broken else "ok"
        written = dict(results) if status == "ok" else {}
        extra = {"count": len(items), "iterations": records}
        return status, written, ms, extra
