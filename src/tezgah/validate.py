import inspect
import warnings

from .errors import UnusedOutputWarning, ValidationError
from .nodes import Branch, Loop, Map, Pipeline, Step
from .nodes.base import child_path, named_params


def validate_pipeline(pipe):
    problems, unused = [], []
    _check_frame(pipe, "", problems, unused, frozenset())
    if problems:
        raise ValidationError(problems)
    for message in unused:
        warnings.warn(message, UnusedOutputWarning, stacklevel=3)


def _check_node(node, path, problems, unused, ancestors):
    if isinstance(node, Step):
        _check_step(node, path, problems)
    elif isinstance(node, Pipeline):
        _check_frame(node, path, problems, unused, ancestors)
    elif isinstance(node, Map):
        _check_map(node, path, problems, unused, ancestors)
    elif isinstance(node, Loop):
        _check_loop(node, path, problems, unused, ancestors)
    elif isinstance(node, Branch):
        _check_branch(node, path, problems, unused, ancestors)


def _check_frame(pipe, path, problems, unused, ancestors):
    if id(pipe) in ancestors:
        problems.append(f"{path or pipe.name}: pipeline '{pipe.name}' contains itself")
        return
    ancestors = ancestors | {id(pipe)}
    where = path or pipe.name

    names = set()
    for node in pipe.nodes:
        if node.name in names:
            problems.append(f"{where}: two nodes share the name '{node.name}'")
        names.add(node.name)

    writers = {inner: None for inner in pipe.inputs.values()}
    for node in pipe.nodes:
        for key in node.writes():
            if key in writers:
                current = writers[key]
                owner = "the pipeline inputs" if current is None else f"node '{current.name}'"
                problems.append(f"{where}: key '{key}' is written by both {owner} and node '{node.name}'")
            else:
                writers[key] = node

    for node in pipe.nodes:
        for key in node.reads():
            if key not in writers:
                problems.append(f"{where}: node '{node.name}' reads key '{key}' but nothing in this frame produces it")

    for inner in pipe.outputs:
        if inner not in writers:
            problems.append(f"{where}: declared output '{inner}' is not produced in this frame")

    for node in pipe.nodes:
        for target in node.wait_for:
            if target not in names:
                problems.append(f"{where}: node '{node.name}' waits for unknown node '{target}'")

    _find_cycles(pipe, where, problems, names)

    read_keys = set()
    for node in pipe.nodes:
        read_keys.update(node.reads())
    exported = set(pipe.outputs)
    for node in pipe.nodes:
        for key in node.writes():
            if key not in read_keys and key not in exported:
                unused.append(f"{where}: output '{key}' of node '{node.name}' is never used")

    for node in pipe.nodes:
        _check_node(node, child_path(path, node.name), problems, unused, ancestors)


def _check_step(step, path, problems):
    if step.when is not None and step.outputs:
        problems.append(f"{path}: when is only allowed on side effect steps, '{step.name}' declares outputs")
    _check_binding(path, f"node '{step.name}'", step.signature, step.inputs, problems)


def _check_binding(path, owner, signature, binding, problems):
    if signature is None:
        return
    names = {param.name for param in named_params(signature)}
    takes_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    for param in binding:
        if param not in names and not takes_kwargs:
            problems.append(f"{path}: {owner} has no parameter with that name: '{param}'")
    for param in named_params(signature):
        if param.default is inspect.Parameter.empty and param.name not in binding:
            problems.append(f"{path}: {owner} is missing required parameter '{param.name}'")


def _check_map(node, path, problems, unused, ancestors):
    if id(node) in ancestors:
        problems.append(f"{path}: map '{node.name}' contains itself")
        return
    ancestors = ancestors | {id(node)}

    frame_keys = {node.item} | ({node.index} if node.index else set()) | set(node.broadcast_keys())
    overlap = sorted(set(node.body.writes()) & frame_keys)
    if overlap:
        problems.append(f"{path}: body writes {overlap} but those keys already exist in the iteration frame (item, index, broadcast)")
    if node.collect and not node.body.writes():
        problems.append(f"{path}: collect was requested but the body produces no outputs")
    if not node.collect and node.body.writes():
        unused.append(f"{path}: body outputs are discarded (collect not set)")

    _check_node(node.body, path, problems, unused, ancestors)


def _check_loop(node, path, problems, unused, ancestors):
    if id(node) in ancestors:
        problems.append(f"{path}: loop '{node.name}' contains itself")
        return
    ancestors = ancestors | {id(node)}

    body_writes = set(node.body.writes())
    body_reads = set(node.body.reads())
    for name in node.carry:
        if name not in body_writes:
            problems.append(f"{path}: body does not export carry key '{name}'")
        if name not in body_reads:
            unused.append(f"{path}: carry key '{name}' is never read by the body")
    for param in node.until_params:
        if param not in body_writes:
            problems.append(f"{path}: until reads '{param}' but the body does not export it")
    for inner in node.trace:
        if inner not in body_writes:
            problems.append(f"{path}: trace key '{inner}' is not exported by the body")
    for name in node.outputs:
        if name not in node.carry:
            problems.append(f"{path}: outputs key '{name}' is not a carry key")

    overlap = sorted(body_writes & set(node.broadcast_keys()))
    if overlap:
        problems.append(f"{path}: body writes {overlap} but those keys are broadcast from the parent frame")

    used = set(node.carry) | set(node.until_params) | set(node.trace)
    for key in sorted(body_writes - used):
        unused.append(f"{path}: body export '{key}' is never used")

    _check_node(node.body, path, problems, unused, ancestors)


def _check_branch(node, path, problems, unused, ancestors):
    if id(node) in ancestors:
        problems.append(f"{path}: branch '{node.name}' contains itself")
        return
    ancestors = ancestors | {id(node)}

    _check_binding(path, f"branch '{node.name}' decide", node.signature, node.inputs, problems)

    reference, reference_desc = None, None
    for label, branch in node.cases.items():
        writes = set(branch.writes())
        if reference is None:
            reference, reference_desc = writes, f"case {label!r}"
        elif writes != reference:
            problems.append(
                f"{path}: {reference_desc} produces {sorted(reference)} but case {label!r} produces {sorted(writes)}; all branches must produce the same outputs"
            )
    if node.default is not None and reference is not None:
        writes = set(node.default.writes())
        if writes != reference:
            problems.append(
                f"{path}: {reference_desc} produces {sorted(reference)} but the default branch produces {sorted(writes)}; all branches must produce the same outputs"
            )

    for branch in node.branches():
        _check_node(branch, path, problems, unused, ancestors)


def _find_cycles(pipe, where, problems, names):
    deps = pipe.deps()
    color = {}
    stack = []

    def visit(name):
        color[name] = "gray"
        stack.append(name)
        for upstream in sorted(deps.get(name, ())):
            if upstream not in names:
                continue
            if color.get(upstream) == "gray":
                cycle = stack[stack.index(upstream):] + [upstream]
                problems.append(f"{where}: dependency cycle {' -> '.join(cycle)}")
            elif upstream not in color:
                visit(upstream)
        stack.pop()
        color[name] = "black"

    for node in pipe.nodes:
        if node.name not in color:
            visit(node.name)
