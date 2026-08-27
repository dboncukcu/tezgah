import inspect
import time

from ..errors import ContractError
from .base import Node, as_node, child_path, make_binding, node_executor, node_label, str_list


class Branch(Node):
    leaf = False

    def __init__(self, decide, inputs=None, cases=None, default=None,
                 executor=None, name="branch", wait_for=None):
        self.name, label = node_label("Branch", name)

        if not callable(decide):
            raise TypeError(f"{label}: decide must be callable, got {type(decide).__name__}")
        self.decide = decide
        try:
            self.signature = inspect.signature(decide)
        except (TypeError, ValueError):
            self.signature = None
        self.inputs = make_binding(self.signature, inputs, label)

        if not isinstance(cases, dict) or not cases:
            raise TypeError(f"{label}: cases must be a non empty dict of label to node")
        self.cases = {case_label: as_node(case, f"{label} case {case_label!r}") for case_label, case in cases.items()}
        self.default = None if default is None else as_node(default, f"{label} default")

        self.wait_for = [] if wait_for is None else str_list(wait_for, label, "wait_for")
        self.executor = node_executor(executor, label)

    def branches(self):
        nodes = list(self.cases.values())
        if self.default is not None:
            nodes.append(self.default)
        return nodes

    def reads(self):
        keys = list(self.inputs.values())
        for branch in self.branches():
            keys.extend(branch.reads())
        return list(dict.fromkeys(keys))

    def writes(self):
        return list(self.branches()[0].writes())

    def execute(self, values, kernel=None, path=None, pool=None):
        if kernel is None:
            raise ValueError("Branch.execute needs a kernel; nodes are run by tezgah, not by hand")
        if path is None:
            raise ValueError("Branch.execute needs a path; nodes are run by tezgah, not by hand")

        start = time.perf_counter()
        call_args = {param: values[key] for param, key in self.inputs.items()}
        case_label = self.decide(**call_args)
        branch = self.cases.get(case_label, self.default)
        if branch is None:
            raise ContractError(f"{self.name}: no case for label {case_label!r} and no default")

        branch_path = child_path(path, branch.name)
        future = kernel.submit(branch, values, branch_path, branch.executor or self.executor or pool)
        kernel.wait([future])
        status, written, record = kernel.deliver(future)

        ms = (time.perf_counter() - start) * 1000
        extra = {"label": case_label, "chosen": record}
        return status, written, ms, extra
