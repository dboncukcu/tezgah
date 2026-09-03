import importlib.util
import threading
import time

import pytest

from tezgah import Map, Pipeline, Step, run


def read_text(source):
    return source


def count_words(text):
    return len(text.split())


def count_lines(text):
    return len(text.splitlines())


def snooze_a():
    time.sleep(0.2)
    return "a"


def snooze_b():
    time.sleep(0.2)
    return "b"


def diamond():
    return Pipeline(
        nodes=[
            Step(read_text, inputs={"source": "raw"}, outputs=["text"]),
            Step(count_words, inputs=["text"], outputs=["words"]),
            Step(count_lines, inputs=["text"], outputs=["lines"]),
            Step(lambda words, lines: f"{words}/{lines}", inputs=["words", "lines"], outputs=["summary"], name="fmt"),
        ],
        inputs=["raw"],
        outputs=["summary"],
    )


class FakeFuture:
    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None
        self.callbacks = []
        self.lock = threading.Lock()

    def _finish(self):
        self.event.set()
        with self.lock:
            callbacks = list(self.callbacks)
        for callback in callbacks:
            callback(self)

    def set_result(self, value):
        self.value = value
        self._finish()

    def set_exception(self, error):
        self.error = error
        self._finish()

    def add_done_callback(self, callback):
        with self.lock:
            if not self.event.is_set():
                self.callbacks.append(callback)
                return
        callback(self)

    def result(self):
        self.event.wait()
        if self.error is not None:
            raise self.error
        return self.value

    def done(self):
        return self.event.is_set()


class FakePool:
    def submit(self, fn, *args):
        future = FakeFuture()

        def work():
            try:
                future.set_result(fn(*args))
            except Exception as exc:
                future.set_exception(exc)

        threading.Thread(target=work, daemon=True).start()
        return future


class InlinePool:
    def __init__(self):
        self.calls = 0
        self.shutdown_called = False

    def submit(self, fn, *args):
        self.calls += 1
        future = FakeFuture()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self):
        self.shutdown_called = True


def test_instance_pool_used_and_owned_by_user():
    pool = InlinePool()
    report = run(diamond(), inputs={"raw": "one two\nthree"}, executor=pool)
    assert report.outputs == {"summary": "3/2"}
    assert pool.calls == 4
    assert pool.shutdown_called is False


def test_instance_pool_matches_serial():
    serial = run(diamond(), inputs={"raw": "a b\nc"})
    pooled = run(diamond(), inputs={"raw": "a b\nc"}, executor=InlinePool())
    assert serial.outputs == pooled.outputs


def test_mixed_futures_run_together():
    pipe = Pipeline(
        nodes=[
            Step(snooze_a, outputs=["a"]),
            Pipeline(nodes=[Step(snooze_b, outputs=["b"], name="inner")], outputs={"b": "b"}, name="sub"),
            Step(lambda a, b: a + b, inputs=["a", "b"], outputs=["ab"], name="glue"),
        ],
        outputs=["ab"],
    )
    started = time.perf_counter()
    report = run(pipe, inputs={}, executor=FakePool())
    elapsed = time.perf_counter() - started
    assert report.outputs == {"ab": "ab"}
    assert elapsed < 0.35


def test_map_through_custom_pool():
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2, 3], outputs=["nums"], name="src"),
            Map(Step(lambda n: n * 2, inputs=["n"], outputs=["out"], name="double"),
                over="nums", item="n", collect={"out": "outs"}, parallel=3),
        ],
        outputs=["outs"],
    )
    report = run(pipe, inputs={}, executor=FakePool())
    assert report.outputs == {"outs": [2, 4, 6]}


def test_failure_through_custom_pool():
    def boom(x):
        raise ValueError("nope")

    pipe = Pipeline(
        nodes=[
            Step(boom, inputs=["x"], outputs=["a"]),
            Step(lambda a: a, inputs=["a"], outputs=["b"], name="after"),
        ],
        inputs=["x"],
        outputs=["b"],
    )
    from tezgah import RunError

    with pytest.raises(RunError) as err:
        run(pipe, inputs={"x": 1}, executor=FakePool())
    assert [path for path, _ in err.value.failures] == ["boom"]
    statuses = {rec["node"]: rec["status"] for rec in err.value.tree["nodes"]}
    assert statuses == {"boom": "failed", "after": "upstream_failed"}


def test_dask_name_without_dask_installed():
    if importlib.util.find_spec("dask") is not None:
        pytest.skip("dask is installed")
    pipe = Pipeline(nodes=[Step(lambda x: x, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    with pytest.raises(ImportError, match="dask"):
        run(pipe, inputs={"x": 1}, executor="dask")


def test_unknown_executor_lists_options():
    pipe = Pipeline(nodes=[Step(lambda x: x, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    with pytest.raises(ValueError, match="'serial', 'thread', 'dask'"):
        run(pipe, inputs={"x": 1}, executor="warp")


def test_serial_runs_inline():
    pipe = Pipeline(nodes=[Step(lambda x: x + 1, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    assert run(pipe, inputs={"x": 1}, executor="serial").outputs == {"y": 2}
