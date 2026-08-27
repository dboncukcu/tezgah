import json
import os
import queue
import sys
import threading
import uuid
import warnings
from concurrent import futures as _futures
from datetime import datetime

from .errors import RunError, ValidationError
from .executors import resolve_executor
from .nodes import Pipeline


class Report:
    def __init__(self, outputs, tree, run_id):
        self.outputs = outputs
        self.tree = tree
        self.run = run_id


def _run_container(future, node, values, kernel, path, pool):
    try:
        future.set_result(node.execute(values, kernel, path, pool))
    except Exception as exc:
        future.set_exception(exc)


class Kernel:
    def __init__(self, run_id, sinks, window=1, pool=None):
        self.run_id = run_id
        self.sinks = sinks
        self.window = window
        self.pool = pool
        self.failures = []
        self._ledger = {}
        self._state = threading.Lock()
        self._emit = threading.Lock()

    def submit(self, node, values, path, pool=None):
        self.emit(path, "started", node=node.name)
        target = pool if pool is not None else self.pool
        if target is None:
            future = _futures.Future()
            with self._state:
                self._ledger[future] = (node, path)
            try:
                if node.leaf:
                    result = node.execute(values)
                else:
                    result = node.execute(values, self, path, pool)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future
        if node.leaf:
            future = target.submit(node.execute, values)
        else:
            future = _futures.Future()
            threading.Thread(target=_run_container, args=(future, node, values, self, path, target), daemon=True).start()
        with self._state:
            self._ledger[future] = (node, path)
        return future

    def wait(self, futures):
        if not futures:
            return []
        if all(isinstance(f, _futures.Future) for f in futures):
            done, _ = _futures.wait(futures, return_when=_futures.FIRST_COMPLETED)
            return list(done)
        completions = queue.SimpleQueue()
        for future in futures:
            future.add_done_callback(completions.put)
        done = [completions.get()]
        while True:
            try:
                extra = completions.get_nowait()
            except queue.Empty:
                break
            if not any(extra is d for d in done):
                done.append(extra)
        return done

    def deliver(self, future):
        with self._state:
            node, path = self._ledger.pop(future)
        try:
            status, written, ms, extra = future.result()
        except Exception as exc:
            with self._state:
                self.failures.append((path, exc))
            record = {
                "node": node.name,
                "path": path,
                "status": "failed",
                "ms": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.emit(path, "failed", node=node.name, status="failed", ms=0, error=record["error"])
            return "failed", {}, record
        record = {"node": node.name, "path": path, "status": status, "ms": round(ms, 3), **extra}
        if status == "skipped":
            self.emit(path, "skipped", node=node.name, status="skipped")
        else:
            self.emit(path, "finished", node=node.name, status="ok", ms=record["ms"])
        return status, written, record

    def emit(self, path, kind, node=None, **extra):
        event = {"schema": 1, "run": self.run_id, "path": path, "node": node, "kind": kind}
        event.update(extra)
        event["t"] = datetime.now().isoformat(timespec="milliseconds")
        with self._emit:
            for sink in self.sinks:
                try:
                    sink(event)
                except Exception as exc:
                    warnings.warn(f"event sink failed: {type(exc).__name__}: {exc}")

    def shutdown(self):
        if self.pool is not None and hasattr(self.pool, "shutdown"):
            self.pool.shutdown()


class _FileSink:
    def __init__(self, directory):
        self.file = open(os.path.join(directory, "events.jsonl"), "w", encoding="utf-8")

    def __call__(self, event):
        self.file.write(json.dumps(event, ensure_ascii=False, default=repr) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def subscribe(pipe, callback):
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"subscribe expects a Pipeline, got {type(pipe).__name__}")
    if not callable(callback):
        raise TypeError(f"subscribe callback must be callable, got {type(callback).__name__}")
    pipe.subscribers.append(callback)
    return callback


def run(pipe, inputs=None, executor=None, workers=None, record_dir=None):
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"run expects a Pipeline, got {type(pipe).__name__}")
    pipe.validate()

    given = dict(inputs or {})
    problems = []
    for outer in pipe.inputs:
        if outer not in given:
            problems.append(f"run: missing input '{outer}'")
    for key in given:
        if key not in pipe.inputs:
            problems.append(f"run: unexpected input '{key}'")
    if problems:
        raise ValidationError(problems)

    run_id = "r_" + uuid.uuid4().hex[:8]
    started_t = datetime.now().isoformat(timespec="milliseconds")
    sinks = list(pipe.subscribers)

    pool = resolve_executor(executor, workers)
    owns_pool = pool is not None and isinstance(executor, str)
    if pool is None:
        kernel = Kernel(run_id, sinks)
    else:
        kernel = Kernel(run_id, sinks, window=workers or os.cpu_count() or 1, pool=pool)

    file_sink = None
    out_file = err_file = None
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
            file_sink = _FileSink(record_dir)
            sinks.append(file_sink)
            out_file = open(os.path.join(record_dir, "stdout.txt"), "w", encoding="utf-8")
            err_file = open(os.path.join(record_dir, "stderr.txt"), "w", encoding="utf-8")
            sys.stdout = _Tee(old_stdout, out_file)
            sys.stderr = _Tee(old_stderr, err_file)
        status, written, ms, extra = pipe.execute(given, kernel, "", pipe.executor)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        if owns_pool:
            kernel.shutdown()
        if out_file is not None:
            out_file.close()
        if err_file is not None:
            err_file.close()

    tree = {"node": pipe.name, "path": pipe.name, "status": status, "ms": round(ms, 3), **extra}
    if file_sink is not None:
        summary = {"schema": 1, "run": run_id, "status": status, "t": started_t, "ms": tree["ms"], "tree": tree}
        with open(os.path.join(record_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=repr)
        file_sink.close()

    if kernel.failures:
        raise RunError(kernel.failures, tree) from kernel.failures[0][1]
    return Report(written, tree, run_id)
