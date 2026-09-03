import time

import pytest

from tezgah import Map, Pipeline, RunError, Step, run
from tezgah.executors import DaskPool


def read_text(source):
    return source


def count_words(text):
    return len(text.split())


def count_lines(text):
    return len(text.splitlines())


def summarize(words, lines):
    return f"{words}/{lines}"


def snooze_150():
    time.sleep(0.15)
    return "a"


def snooze_150_b():
    time.sleep(0.15)
    return "b"


def slow_double(n):
    time.sleep(0.1)
    return n * 2


def boom(x):
    raise ValueError("nope")


def diamond():
    return Pipeline(
        nodes=[
            Step(read_text, inputs={"source": "raw"}, outputs=["text"]),
            Step(count_words, inputs=["text"], outputs=["words"]),
            Step(count_lines, inputs=["text"], outputs=["lines"]),
            Step(summarize, inputs=["words", "lines"], outputs=["summary"]),
        ],
        inputs=["raw"],
        outputs=["summary"],
    )


def test_dask_diamond():
    report = run(diamond(), inputs={"raw": "one two\nthree"}, executor="dask", workers=2)
    assert report.outputs == {"summary": "3/2"}
    assert report.tree["status"] == "ok"


def test_dask_matches_serial():
    serial = run(diamond(), inputs={"raw": "a b\nc d"})
    with_dask = run(diamond(), inputs={"raw": "a b\nc d"}, executor="dask", workers=2)
    assert serial.outputs == with_dask.outputs


def test_dask_runs_independent_steps_together():
    pipe = Pipeline(
        nodes=[
            Step(snooze_150, outputs=["a"]),
            Step(snooze_150_b, outputs=["b"]),
            Step(lambda a, b: a + b, inputs=["a", "b"], outputs=["ab"], name="glue"),
        ],
        outputs=["ab"],
    )
    started = time.perf_counter()
    report = run(pipe, inputs={}, executor="dask", workers=2)
    elapsed = time.perf_counter() - started
    assert report.outputs == {"ab": "ab"}
    assert elapsed < 3.0


def test_dask_map_collect_order():
    pipe = Pipeline(
        nodes=[
            Step(lambda: [3, 1, 2], outputs=["nums"], name="src"),
            Map(Step(slow_double, outputs=["out"]), over="nums", item="n",
                collect={"out": "doubled"}, parallel=3),
        ],
        outputs=["doubled"],
    )
    report = run(pipe, inputs={}, executor="dask", workers=3)
    assert report.outputs == {"doubled": [6, 2, 4]}


def test_dask_map_matches_serial():
    def build():
        return Pipeline(
            nodes=[
                Step(lambda: [1, 2, 3], outputs=["nums"], name="src"),
                Map(Step(slow_double, outputs=["out"]), over="nums", item="n", collect={"out": "outs"}, parallel=3),
            ],
            outputs=["outs"],
        )

    serial = run(build(), inputs={})
    dask_run = run(build(), inputs={}, executor="dask", workers=2)
    assert serial.outputs == dask_run.outputs == {"outs": [2, 4, 6]}


def test_dask_failure_propagation():
    pipe = Pipeline(
        nodes=[
            Step(boom, inputs=["x"], outputs=["a"]),
            Step(lambda a: a, inputs=["a"], outputs=["b"], name="after"),
        ],
        inputs=["x"],
        outputs=["b"],
    )
    with pytest.raises(RunError) as err:
        run(pipe, inputs={"x": 1}, executor="dask", workers=2)
    assert [path for path, _ in err.value.failures] == ["boom"]
    statuses = {rec["node"]: rec["status"] for rec in err.value.tree["nodes"]}
    assert statuses == {"boom": "failed", "after": "upstream_failed"}


def test_dask_pool_instance_not_owned_by_engine():
    pool = DaskPool(workers=2)
    pipe = Pipeline(nodes=[Step(lambda x: x + 1, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    report = run(pipe, inputs={"x": 1}, executor=pool)
    assert report.outputs == {"y": 2}
    pool.shutdown()


def test_nested_pipeline_through_dask():
    def bump(v):
        return v + 1

    inner = Pipeline(
        nodes=[Step(bump, inputs=["v"], outputs=["v_next"])],
        inputs={"first": "v"},
        outputs={"v_next": "second"},
        name="inner",
    )
    pipe = Pipeline(
        nodes=[
            Step(bump, inputs={"v": "start"}, outputs=["first"], name="first_inc"),
            inner,
            Step(bump, inputs={"v": "second"}, outputs=["third"], name="last_inc"),
        ],
        inputs=["start"],
        outputs={"third": "result"},
    )
    report = run(pipe, inputs={"start": 1}, executor="dask", workers=2)
    assert report.outputs == {"result": 4}
