import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tezgah import Branch, Loop, Map, Pipeline, Step, run


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.max = 0

    def enter(self):
        with self.lock:
            self.now += 1
            self.max = max(self.max, self.now)

    def leave(self):
        with self.lock:
            self.now -= 1


class TrackedPool:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.closed = False
        self._pool = ThreadPoolExecutor(max_workers=2)

    def submit(self, fn, *args):
        with self.lock:
            self.calls += 1
        return self._pool.submit(fn, *args)

    def shutdown(self):
        self.closed = True
        self._pool.shutdown(wait=True)


def test_node_executor_serializes_steps():
    gpu = ThreadPoolExecutor(max_workers=1)
    tracker = Tracker()

    def work_a():
        tracker.enter()
        time.sleep(0.15)
        tracker.leave()
        return "a"

    def work_b():
        tracker.enter()
        time.sleep(0.15)
        tracker.leave()
        return "b"

    pipe = Pipeline(
        nodes=[
            Step(work_a, outputs=["a"], executor=gpu),
            Step(work_b, outputs=["b"], executor=gpu),
            Step(lambda a, b: a + b, inputs=["a", "b"], outputs=["ab"], name="glue"),
        ],
        outputs=["ab"],
    )
    started = time.perf_counter()
    report = run(pipe, inputs={}, executor="thread", workers=4)
    elapsed = time.perf_counter() - started
    gpu.shutdown()
    assert report.outputs == {"ab": "ab"}
    assert tracker.max == 1
    assert elapsed >= 0.29


def test_pipeline_executor_inherited_by_children():
    pool = TrackedPool()
    pipe = Pipeline(
        nodes=[
            Step(lambda x: x + 1, inputs=["x"], outputs=["y"], name="first"),
            Step(lambda y: y * 2, inputs=["y"], outputs=["z"], name="second"),
        ],
        inputs=["x"],
        outputs=["z"],
        executor=pool,
    )
    report = run(pipe, inputs={"x": 3}, executor="thread", workers=2)
    assert report.outputs == {"z": 8}
    assert pool.calls == 2
    assert pool.closed is False
    pool.shutdown()


def test_pipeline_executor_inherited_through_nesting():
    pool = TrackedPool()
    inner = Pipeline(
        nodes=[Step(lambda x: x + 1, inputs=["x"], outputs=["y"], name="deep")],
        inputs={"x": "x"},
        outputs={"y": "y"},
        name="inner",
    )
    outer = Pipeline(
        nodes=[inner],
        inputs=["x"],
        outputs=["y"],
        executor=pool,
    )
    report = run(outer, inputs={"x": 1}, executor="thread", workers=2)
    assert report.outputs == {"y": 2}
    assert pool.calls == 1
    pool.shutdown()


def test_map_executor_used_for_body():
    pool = TrackedPool()
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2, 3], outputs=["nums"], name="src"),
            Map(Step(lambda n: n * 2, inputs=["n"], outputs=["out"], name="double"),
                over="nums", item="n", collect="outs", parallel=3, executor=pool),
        ],
        outputs=["outs"],
    )
    report = run(pipe, inputs={}, executor="thread", workers=2)
    assert report.outputs == {"outs": [2, 4, 6]}
    assert pool.calls == 3
    pool.shutdown()


def test_map_body_executor_wins_over_map_executor():
    body_pool = TrackedPool()
    map_pool = TrackedPool()
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2], outputs=["nums"], name="src"),
            Map(Step(lambda n: n * 2, inputs=["n"], outputs=["out"], name="double", executor=body_pool),
                over="nums", item="n", collect="outs", executor=map_pool),
        ],
        outputs=["outs"],
    )
    report = run(pipe, inputs={}, executor="thread", workers=2)
    assert report.outputs == {"outs": [2, 4]}
    assert body_pool.calls == 2
    assert map_pool.calls == 0
    body_pool.shutdown()
    map_pool.shutdown()


def test_loop_executor_used_for_body():
    pool = TrackedPool()
    pipe = Pipeline(
        nodes=[Loop(body=Step(lambda total: total + 1, inputs=["total"], outputs=["total"], name="inc"),
                    carry={"total": "seed"}, max_iter=3, outputs={"total": "out"}, executor=pool)],
        inputs=["seed"],
        outputs=["out"],
    )
    report = run(pipe, inputs={"seed": 0}, executor="thread", workers=2)
    assert report.outputs == {"out": 3}
    assert pool.calls == 3
    pool.shutdown()


def test_branch_executor_used_for_chosen():
    pool = TrackedPool()
    pipe = Pipeline(
        nodes=[
            Branch(lambda value: "small" if value < 10 else "big", inputs=["value"],
                   cases={"small": Step(lambda value: value * 2, inputs=["value"], outputs=["scaled"], name="sm"),
                          "big": Step(lambda value: value // 2, inputs=["value"], outputs=["scaled"], name="bg")},
                   executor=pool),
        ],
        inputs=["value"],
        outputs=["scaled"],
    )
    report = run(pipe, inputs={"value": 4}, executor="thread", workers=2)
    assert report.outputs == {"scaled": 8}
    assert pool.calls == 1
    pool.shutdown()


def test_node_executor_works_in_serial_mode():
    gpu = ThreadPoolExecutor(max_workers=1)
    seen_threads = []

    def work(x):
        seen_threads.append(threading.current_thread().name)
        return x + 1

    pipe = Pipeline(nodes=[Step(work, inputs=["x"], outputs=["y"], executor=gpu)], inputs=["x"], outputs=["y"])
    report = run(pipe, inputs={"x": 1})
    gpu.shutdown()
    assert report.outputs == {"y": 2}
    assert seen_threads and all(name != "MainThread" for name in seen_threads)


def test_node_executor_matches_default_outputs():
    gpu = ThreadPoolExecutor(max_workers=1)
    pipe = Pipeline(
        nodes=[
            Step(lambda x: x + 1, inputs=["x"], outputs=["y"], name="a", executor=gpu),
            Step(lambda y: y * 2, inputs=["y"], outputs=["z"], name="b", executor=gpu),
        ],
        inputs=["x"],
        outputs=["z"],
    )
    serial = run(Pipeline(
        nodes=[
            Step(lambda x: x + 1, inputs=["x"], outputs=["y"], name="a"),
            Step(lambda y: y * 2, inputs=["y"], outputs=["z"], name="b"),
        ],
        inputs=["x"],
        outputs=["z"],
    ), inputs={"x": 5})
    pooled = run(pipe, inputs={"x": 5}, executor="thread", workers=2)
    gpu.shutdown()
    assert serial.outputs == pooled.outputs == {"z": 12}


def test_node_executor_string_rejected():
    with pytest.raises(TypeError, match="not a string"):
        Step(lambda x: x, inputs=["x"], outputs=["y"], executor="thread")
    with pytest.raises(TypeError, match="not a string"):
        Pipeline(nodes=[Step(lambda x: x, inputs=["x"], outputs=["y"])], executor="thread")
    with pytest.raises(TypeError, match="not a string"):
        Map(Step(lambda n: n, inputs=["n"], outputs=["o"]), over="nums", executor="thread")
    with pytest.raises(TypeError, match="not a string"):
        Loop(body=Step(lambda t: t, inputs=["t"], outputs=["t"]), carry={"t": "t"}, max_iter=1, executor="thread")
    with pytest.raises(TypeError, match="not a string"):
        Branch(lambda: "a", cases={"a": Step(lambda: 1, outputs=["o"])}, executor="thread")


def test_node_executor_without_submit_rejected():
    with pytest.raises(TypeError, match="submit"):
        Step(lambda x: x, inputs=["x"], outputs=["y"], executor=object())
