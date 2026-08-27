import time

from .base import Node, as_node, child_path, key_map, node_executor, node_label, str_list


class Pipeline(Node):
    leaf = False

    def __init__(self, nodes, inputs=None, outputs=None, name="pipeline",
                 executor=None, wait_for=None):
        self.name, label = node_label("Pipeline", name)

        self.nodes = [as_node(node, label) for node in nodes]
        self.inputs = key_map(inputs, label, "inputs")
        self.outputs = key_map(outputs, label, "outputs")
        inner_names = list(self.inputs.values())
        if len(set(inner_names)) != len(inner_names):
            raise ValueError(f"{label}: two inputs map to the same inner name")
        outer_names = list(self.outputs.values())
        if len(set(outer_names)) != len(outer_names):
            raise ValueError(f"{label}: two outputs map to the same outer key")

        self.wait_for = [] if wait_for is None else str_list(wait_for, label, "wait_for")
        self.executor = node_executor(executor, label)
        self.subscribers = []

    def reads(self):
        return list(self.inputs)

    def writes(self):
        return list(self.outputs.values())

    def deps(self):
        writer = {}
        for node in self.nodes:
            for key in node.writes():
                writer.setdefault(key, node)
        dependencies = {}
        for node in self.nodes:
            upstream = {
                writer[key].name
                for key in node.reads()
                if key in writer and writer[key] is not node
            }
            upstream.update(node.wait_for)
            dependencies[node.name] = upstream
        return dependencies

    def validate(self):
        from ..validate import validate_pipeline

        validate_pipeline(self)

    def execute(self, values, kernel, path, pool=None):
        start = time.perf_counter()
        frame = {inner: values[outer] for outer, inner in self.inputs.items()}
        deps = self.deps()
        statuses, records = {}, {}
        flying = {}
        launched = set()

        while len(statuses) < len(self.nodes):
            moved = False
            for node in self.nodes:
                if node.name in statuses or node.name in launched:
                    continue
                states = [statuses.get(name) for name in deps[node.name]]
                if any(state in ("failed", "upstream_failed") for state in states):
                    node_path = child_path(path, node.name)
                    statuses[node.name] = "upstream_failed"
                    records[node.name] = {"node": node.name, "path": node_path, "status": "upstream_failed", "ms": 0}
                    kernel.emit(node_path, "skipped", node=node.name, status="upstream_failed")
                    moved = True
                elif all(state in ("ok", "skipped") for state in states):
                    node_path = child_path(path, node.name)
                    node_values = {key: frame[key] for key in node.reads()}
                    flying[kernel.submit(node, node_values, node_path, node.executor or pool)] = node.name
                    launched.add(node.name)
                    moved = True
            if not flying:
                if not moved:
                    raise RuntimeError(f"{self.name}: scheduler stalled (unvalidated graph?)")
                continue
            for future in kernel.wait(list(flying)):
                name = flying.pop(future)
                launched.discard(name)
                status, written, record = kernel.deliver(future)
                frame.update(written)
                statuses[name] = status
                records[name] = record

        ms = (time.perf_counter() - start) * 1000
        status = "failed" if any(s in ("failed", "upstream_failed") for s in statuses.values()) else "ok"
        written = {outer: frame[inner] for inner, outer in self.outputs.items()} if status == "ok" else {}
        extra = {"nodes": [records[node.name] for node in self.nodes]}
        return status, written, ms, extra
