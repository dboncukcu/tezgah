import json
import sys
import time

from tezgah import Loop, Map, Pipeline, RunCatalog, Step, load_run, run, subscribe


def build_pipe():
    return Pipeline(
        nodes=[
            Step(lambda: 0, outputs=["seed"], name="seed_src"),
            Loop(
                body=Step(lambda total: total + 1, inputs=["total"], outputs=["total"], name="inc"),
                carry={"total": "seed"},
                range=2,
                trace={"total": "totals"},
                outputs={"total": "out"},
                name="grow",
            ),
        ],
        outputs=["out", "totals"],
    )


def test_load_run_queries(tmp_path):
    target = tmp_path / "kayit"
    report = run(build_pipe(), inputs={}, record_dir=str(target))
    record = load_run(str(target))
    assert record.run == report.run
    assert record.status == "ok"
    assert record.summary["run"] == report.run
    assert record.find("seed_src")["status"] == "ok"
    assert record.find("grow")["traces"] == {"totals": [1, 2]}
    assert record.failed() == []
    assert "grow[0].inc" in record.durations()
    assert any(event["kind"] == "iter_finished" for event in record.events_for("grow[0]"))
    assert len(record.nodes()) >= 5


def test_load_run_failed_nodes(tmp_path):
    def boom(x):
        raise ValueError("nope")

    pipe = Pipeline(
        nodes=[
            Step(boom, inputs=["x"], outputs=["a"], name="kaboom"),
            Step(lambda a: a, inputs=["a"], outputs=["b"], name="after"),
        ],
        inputs=["x"],
        outputs=["b"],
    )
    from tezgah import RunError

    try:
        run(pipe, inputs={"x": 1}, record_dir=str(tmp_path))
    except RunError:
        record = load_run(str(tmp_path))
        paths = [r["path"] for r in record.failed()]
        assert paths == ["kaboom", "after"]
    else:
        raise AssertionError("RunError expected")


def test_timeline(tmp_path):
    target = tmp_path / "run1"
    run(build_pipe(), inputs={}, record_dir=str(target))
    record = load_run(str(target))
    bars = {bar["path"]: bar for bar in record.timeline()}
    assert "seed_src" in bars
    assert "grow" in bars
    assert "grow[0].inc" in bars
    assert bars["seed_src"]["status"] == "ok"
    assert bars["seed_src"]["start"] <= bars["seed_src"]["end"]


def test_timeline_excludes_upstream_failed(tmp_path):
    def boom(x):
        raise ValueError("nope")

    pipe = Pipeline(
        nodes=[
            Step(boom, inputs=["x"], outputs=["a"], name="kaboom"),
            Step(lambda a: a, inputs=["a"], outputs=["b"], name="after"),
        ],
        inputs=["x"],
        outputs=["b"],
    )
    from tezgah import RunError

    try:
        run(pipe, inputs={"x": 1}, record_dir=str(tmp_path))
    except RunError:
        pass
    record = load_run(str(tmp_path))
    paths = {bar["path"] for bar in record.timeline()}
    assert "kaboom" in paths
    assert "after" not in paths


def test_stdout_stderr(tmp_path):
    def talk():
        print("hello")
        return "x"

    def shout(x):
        print("warn", file=sys.stderr)
        return "y"

    pipe = Pipeline(
        nodes=[
            Step(talk, outputs=["x"], name="talker"),
            Step(shout, inputs=["x"], outputs=["y"], name="shouter"),
        ],
        outputs=["y"],
    )
    target = tmp_path / "logs"
    run(pipe, inputs={}, record_dir=str(target))
    record = load_run(str(target))
    assert "hello" in record.stdout()
    assert "warn" in record.stderr()


def test_malformed_events_line_skipped(tmp_path):
    target = tmp_path / "run1"
    run(build_pipe(), inputs={}, record_dir=str(target))
    with open(target / "events.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"broken": ')
    record = load_run(str(target))
    assert record.events
    assert all("kind" in event for event in record.events)


def test_catalog_list_and_get(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    run(build_pipe(), inputs={}, record_dir=str(runs / "a"))
    time.sleep(0.01)
    second = run(build_pipe(), inputs={}, record_dir=str(runs / "b"))
    (runs / "garbage").mkdir()

    catalog = RunCatalog(str(runs))
    entries = catalog.list()
    assert [entry["dir"] for entry in entries] == ["b", "a"]
    assert entries[0]["run"] == second.run
    assert entries[0]["status"] == "ok"
    assert entries[0]["t"]

    record = catalog.get("b")
    assert record.run == second.run
    assert catalog.get("garbage") is None
    assert catalog.get("nope") is None
    assert catalog.get("..") is None
    assert catalog.get("../runs/a") is None


def test_catalog_empty_root(tmp_path):
    catalog = RunCatalog(str(tmp_path / "yok"))
    assert catalog.list() == []
    assert catalog.get("a") is None


def test_run_json_has_timestamp(tmp_path):
    target = tmp_path / "run1"
    run(build_pipe(), inputs={}, record_dir=str(target))
    summary = json.loads((target / "run.json").read_text(encoding="utf-8"))
    assert "T" in summary["t"]
    assert len(summary["t"].split(".")[-1]) == 3


def test_event_timestamps_millisecond(tmp_path):
    events = []
    pipe = build_pipe()
    subscribe(pipe, events.append)
    run(pipe, inputs={}, record_dir=str(tmp_path / "run1"))
    assert events
    assert all(len(event["t"].split(".")[-1]) == 3 for event in events)


def test_map_iterations_in_nodes(tmp_path):
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2], outputs=["nums"], name="src"),
            Map(Step(lambda n: n + 1, inputs=["n"], outputs=["out"], name="bump"),
                over="nums", item="n", collect={"out": "doubled"}, name="mapper"),
        ],
        outputs=["doubled"],
    )
    run(pipe, inputs={}, record_dir=str(tmp_path / "run1"))
    record = load_run(str(tmp_path / "run1"))
    paths = {node["path"] for node in record.nodes()}
    assert "mapper[0].bump" in paths
    assert "mapper[1].bump" in paths
    assert record.find("mapper")["count"] == 2
