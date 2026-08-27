import threading
import time

import pytest

from tezgah import Map, Pipeline, RunError, Step, run, subscribe


def snooze_a():
    time.sleep(0.2)
    return "a"


def snooze_b():
    time.sleep(0.2)
    return "b"


def test_thread_runs_independent_steps_together():
    pipe = Pipeline(
        nodes=[
            Step(snooze_a, outputs=["a"]),
            Step(snooze_b, outputs=["b"]),
            Step(lambda a, b: a + b, inputs=["a", "b"], outputs=["ab"], name="glue"),
        ],
        outputs=["ab"],
    )
    started = time.perf_counter()
    report = run(pipe, inputs={}, executor="thread", workers=2)
    elapsed = time.perf_counter() - started
    assert report.outputs == {"ab": "ab"}
    assert elapsed < 0.35


def test_sibling_containers_run_together():
    def sleeper():
        time.sleep(0.2)
        return "v"

    left = Pipeline(nodes=[Step(sleeper, outputs=["v"], name="l")], outputs={"v": "left"}, name="left")
    right = Pipeline(nodes=[Step(sleeper, outputs=["v"], name="r")], outputs={"v": "right"}, name="right")
    pipe = Pipeline(nodes=[left, right], outputs=["left", "right"])
    started = time.perf_counter()
    report = run(pipe, inputs={}, executor="thread", workers=2)
    elapsed = time.perf_counter() - started
    assert report.outputs == {"left": "v", "right": "v"}
    assert elapsed < 0.35


def test_thread_map_parallel_is_deterministic():
    def slow_double(n):
        time.sleep(0.1)
        return n * 2

    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2, 3, 4], outputs=["nums"], name="src"),
            Map(Step(slow_double, inputs=["n"], outputs=["out"]), over="nums", item="n",
                collect="doubled", parallel=4),
        ],
        outputs=["doubled"],
    )
    started = time.perf_counter()
    report = run(pipe, inputs={}, executor="thread", workers=4)
    elapsed = time.perf_counter() - started
    assert report.outputs == {"doubled": [2, 4, 6, 8]}
    assert elapsed < 0.3


def test_thread_matches_serial():
    def build():
        return Pipeline(
            nodes=[
                Step(lambda: "bir iki\nuc dort bes", outputs=["text"], name="source"),
                Step(lambda text: len(text.split()), inputs=["text"], outputs=["words"], name="words"),
                Step(lambda text: len(text.splitlines()), inputs=["text"], outputs=["lines"], name="lines"),
                Step(lambda words, lines: f"{words}/{lines}", inputs=["words", "lines"], outputs=["merged"], name="merge"),
                Step(lambda: [1, 2, 3], outputs=["nums"], name="nums_src"),
                Map(Step(lambda n: n * 3, inputs=["n"], outputs=["tripled"], name="triple"),
                    over="nums", item="n", collect="tripled_list", parallel=3),
            ],
            outputs=["merged", "tripled_list"],
        )

    inputs = {}
    seri = run(build(), inputs=inputs)
    threaded = run(build(), inputs=inputs, executor="thread", workers=3)
    threaded_again = run(build(), inputs=inputs, executor="thread", workers=2)
    assert seri.outputs == threaded.outputs == threaded_again.outputs


def test_worker_limit_respected():
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def task(i):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        return i

    pipe = Pipeline(
        nodes=[
            Step(lambda: list(range(6)), outputs=["nums"], name="src"),
            Map(Step(task, inputs=["i"], outputs=["out"]), over="nums", item="i", collect="outs", parallel=6),
        ],
        outputs=["outs"],
    )
    report = run(pipe, inputs={}, executor="thread", workers=2)
    assert report.outputs == {"outs": [0, 1, 2, 3, 4, 5]}
    assert state["max"] == 2


def test_parallel_true_window_is_workers():
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def body(n):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        return n * 2

    pipe = Pipeline(
        nodes=[
            Step(lambda: list(range(5)), outputs=["nums"], name="src"),
            Map(Step(body, inputs=["n"], outputs=["out"]), over="nums", item="n", collect="outs", parallel=True),
        ],
        outputs=["outs"],
    )
    report = run(pipe, inputs={}, executor="thread", workers=2)
    assert report.outputs == {"outs": [0, 2, 4, 6, 8]}
    assert state["max"] == 2


def test_thread_failure_propagation():
    def boom(x):
        raise ValueError("patladi")

    pipe = Pipeline(
        nodes=[
            Step(boom, inputs=["x"], outputs=["a"]),
            Step(lambda a: a, inputs=["a"], outputs=["b"], name="after"),
            Step(lambda x: x + 1, inputs=["x"], outputs=["c"], name="free"),
        ],
        inputs=["x"],
        outputs={"b": "b", "c": "c"},
    )
    with pytest.raises(RunError) as err:
        run(pipe, inputs={"x": 1}, executor="thread", workers=2)
    statuses = {rec["node"]: rec["status"] for rec in err.value.tree["nodes"]}
    assert statuses == {"boom": "failed", "after": "upstream_failed", "free": "ok"}


def test_thread_map_failure_stops_new_iterations():
    lock = threading.Lock()
    calls = []

    def maybe_boom(i):
        with lock:
            calls.append(i)
        if i == 0:
            raise RuntimeError("iter failed")
        time.sleep(0.1)
        return i

    pipe = Pipeline(
        nodes=[
            Step(lambda: [0, 1, 2, 3, 4, 5], outputs=["nums"], name="src"),
            Map(Step(maybe_boom, inputs=["i"], outputs=["out"]), over="nums", item="i",
                collect="outs", parallel=2, name="mapper"),
        ],
        outputs=["outs"],
    )
    with pytest.raises(RunError) as err:
        run(pipe, inputs={}, executor="thread", workers=2)
    assert set(calls) == {0, 1}
    assert err.value.failures[0][0] == "mapper[0].maybe_boom"


def test_thread_events_intact():
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2, 3], outputs=["nums"], name="src"),
            Map(Step(lambda n: n + 1, inputs=["n"], outputs=["bumped"], name="bump"),
                over="nums", item="n", collect="bumped_list", parallel=3, name="mapper"),
        ],
        outputs=["bumped_list"],
    )
    events = []
    subscribe(pipe, events.append)
    run(pipe, inputs={}, executor="thread", workers=3)
    started = [event for event in events if event["kind"] == "started"]
    terminal = {"finished", "failed", "skipped"}
    assert started
    for event in started:
        assert any(
            other["path"] == event["path"] and other["kind"] in terminal
            for other in events
        )


def test_thread_records_and_outputs(tmp_path):
    pipe = Pipeline(
        nodes=[
            Step(snooze_a, outputs=["a"]),
            Step(snooze_b, outputs=["b"]),
            Step(lambda a, b: a + b, inputs=["a", "b"], outputs=["ab"], name="glue"),
        ],
        outputs=["ab"],
    )
    report = run(pipe, inputs={}, executor="thread", workers=2, record_dir=str(tmp_path))
    assert report.tree["status"] == "ok"
    assert [rec["status"] for rec in report.tree["nodes"]] == ["ok", "ok", "ok"]
    import json

    summary = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len([line for line in lines if json.loads(line)["kind"] == "started"]) == 3
