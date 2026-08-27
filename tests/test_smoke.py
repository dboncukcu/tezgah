import json
from functools import partial

import pytest

import tezgah
from tezgah import (
    Branch,
    Loop,
    Map,
    Pipeline,
    RunError,
    Step,
    UnusedOutputWarning,
    ValidationError,
    run,
    subscribe,
)


def add1(x):
    return x + 1


def make(x):
    return x + 1


def use(y):
    return y * 2


def numbers():
    return [3, 1, 2]


def test_version_available():
    assert tezgah.__version__
    assert isinstance(tezgah.__version__, str)


def test_defaults_from_fn():
    step = Step(add1)
    assert step.name == "add1"
    assert step.inputs == {"x": "x"}
    assert step.outputs == ["add1"]


def test_inputs_list_and_dict():
    assert Step(add1, inputs=["x"]).inputs == {"x": "x"}
    assert Step(add1, inputs={"x": "raw"}).inputs == {"x": "raw"}


def test_outputs_single_string():
    assert Step(add1, outputs="y").outputs == ["y"]


def test_bare_callable_wrapped():
    pipe = Pipeline(nodes=[add1], inputs=["x"], outputs=["add1"])
    assert isinstance(pipe.nodes[0], Step)
    assert pipe.nodes[0].name == "add1"


def test_step_construction_errors():
    with pytest.raises(TypeError):
        Step(3)
    with pytest.raises(TypeError):
        Step(partial(add1))
    with pytest.raises(ValueError):
        Step(add1, outputs=["a", "a"])
    with pytest.raises(ValueError):
        Step(add1, outputs=["o"], retries=-1)
    with pytest.raises(ValueError):
        Step(add1, outputs=["o"], wait=-0.5)


def test_pipeline_construction_errors():
    with pytest.raises(TypeError):
        Pipeline(nodes=[3])
    with pytest.raises(ValueError):
        Pipeline(nodes=[add1], inputs={"a": "x", "b": "x"})
    with pytest.raises(ValueError):
        Pipeline(nodes=[add1], outputs={"a": "x", "b": "x"})


def test_diamond():
    def read_text(source):
        return source

    def count_words(text):
        return len(text.split())

    def count_lines(text):
        return len(text.splitlines())

    pipe = Pipeline(
        nodes=[
            Step(read_text, inputs={"source": "raw"}, outputs=["text"]),
            Step(count_words, inputs=["text"], outputs=["words"]),
            Step(count_lines, inputs=["text"], outputs=["lines"]),
            Step(lambda words, lines: f"{words}/{lines}", inputs=["words", "lines"], outputs=["summary"], name="fmt"),
        ],
        inputs=["raw"],
        outputs=["summary"],
    )
    report = run(pipe, inputs={"raw": "one two\nthree"})
    assert report.outputs == {"summary": "3/2"}
    assert report.tree["status"] == "ok"
    assert [rec["status"] for rec in report.tree["nodes"]] == ["ok", "ok", "ok", "ok"]


def test_declaration_order_irrelevant():
    pipe = Pipeline(
        nodes=[
            Step(add1, inputs={"x": "y"}, outputs=["z"], name="second"),
            Step(add1, inputs=["x"], outputs=["y"], name="first"),
        ],
        inputs=["x"],
        outputs=["z"],
    )
    assert run(pipe, inputs={"x": 1}).outputs == {"z": 3}


def test_multi_output_mapping():
    def split(a, b):
        return {"q": a // b, "r": a % b, "extra": 0}

    pipe = Pipeline(
        nodes=[Step(split, inputs=["a", "b"], outputs=["q", "r"])],
        inputs=["a", "b"],
        outputs=["q", "r"],
    )
    assert run(pipe, inputs={"a": 7, "b": 3}).outputs == {"q": 2, "r": 1}


def test_multi_output_requires_mapping():
    def broken(a):
        return a

    pipe = Pipeline(nodes=[Step(broken, inputs=["a"], outputs=["q", "r"])], inputs=["a"], outputs=["q", "r"])
    with pytest.raises(RunError, match="must return a mapping"):
        run(pipe, inputs={"a": 1})


def test_multi_output_missing_key():
    def half(a):
        return {"q": a}

    pipe = Pipeline(nodes=[Step(half, inputs=["a"], outputs=["q", "r"])], inputs=["a"], outputs=["q", "r"])
    with pytest.raises(RunError, match="missing output keys"):
        run(pipe, inputs={"a": 1})


def test_when_skips_and_wait_for():
    calls = []

    def side_effect(flag):
        calls.append("called")

    pipe = Pipeline(
        nodes=[
            Step(side_effect, inputs=["flag"], outputs=[], when=lambda flag: flag),
            Step(lambda: "done", outputs=["done"], wait_for=["side_effect"]),
        ],
        inputs=["flag"],
        outputs=["done"],
    )
    report = run(pipe, inputs={"flag": False})
    assert calls == []
    assert report.outputs == {"done": "done"}
    statuses = {rec["node"]: rec["status"] for rec in report.tree["nodes"]}
    assert statuses["side_effect"] == "skipped"
    run(pipe, inputs={"flag": True})
    assert calls == ["called"]


def test_skipped_event_sequence():
    pipe = Pipeline(
        nodes=[Step(lambda flag: None, inputs=["flag"], outputs=[], when=lambda flag: flag, name="side")],
        inputs=["flag"],
        outputs=[],
    )
    events = []
    subscribe(pipe, events.append)
    run(pipe, inputs={"flag": False})
    assert [event["kind"] for event in events] == ["started", "skipped"]
    assert events[1]["status"] == "skipped"


def test_failure_propagation():
    def boom(x):
        raise ValueError("nope")

    pipe = Pipeline(
        nodes=[
            Step(boom, inputs=["x"], outputs=["a"]),
            Step(add1, inputs={"x": "a"}, outputs=["b"], name="after_boom"),
            Step(add1, inputs=["x"], outputs=["c"], name="independent"),
        ],
        inputs=["x"],
        outputs={"b": "b", "c": "c"},
    )
    with pytest.raises(RunError) as err:
        run(pipe, inputs={"x": 1})
    assert [path for path, _ in err.value.failures] == ["boom"]
    assert isinstance(err.value.__cause__, ValueError)
    statuses = {rec["node"]: rec["status"] for rec in err.value.tree["nodes"]}
    assert statuses == {"boom": "failed", "after_boom": "upstream_failed", "independent": "ok"}


def test_map_collect_order_and_broadcast():
    def scale(n, factor):
        return n * factor

    pipe = Pipeline(
        nodes=[
            Step(numbers, outputs=["nums"]),
            Map(Step(scale, outputs=["scaled"]), over="nums", item="n", collect="scaled_list", name="scaler"),
        ],
        inputs=["factor"],
        outputs=["scaled_list"],
    )
    report = run(pipe, inputs={"factor": 10})
    assert report.outputs == {"scaled_list": [30, 10, 20]}
    assert report.tree["nodes"][1]["count"] == 3


def test_map_index():
    pipe = Pipeline(
        nodes=[
            Step(numbers, outputs=["nums"]),
            Map(Step(lambda n, i: f"{i}:{n}", inputs=["n", "i"], outputs=["tagged"], name="tag"),
                over="nums", item="n", index="i", collect="tags"),
        ],
        outputs=["tags"],
    )
    assert run(pipe, inputs={}).outputs == {"tags": ["0:3", "1:1", "2:2"]}


def test_map_multi_output_body():
    def split(n):
        return {"double": n * 2, "half": n / 2}

    pipe = Pipeline(
        nodes=[
            Step(numbers, outputs=["nums"]),
            Map(Step(split, outputs=["double", "half"]), over="nums", item="n", collect="pairs"),
        ],
        outputs=["pairs"],
    )
    report = run(pipe, inputs={})
    assert report.outputs["pairs"][0] == {"double": 6, "half": 1.5}
    assert len(report.outputs["pairs"]) == 3


def test_map_empty_collection():
    def empty():
        return []

    pipe = Pipeline(
        nodes=[
            Step(empty, outputs=["nums"]),
            Map(Step(lambda n: n, inputs=["n"], outputs=["same"], name="echo"), over="nums", item="n", collect="out"),
        ],
        outputs=["out"],
    )
    assert run(pipe, inputs={}).outputs == {"out": []}


def test_map_over_not_iterable():
    def five():
        return 5

    pipe = Pipeline(
        nodes=[
            Step(five, outputs=["nums"]),
            Map(Step(lambda n: n, inputs=["n"], outputs=["same"], name="echo"), over="nums", item="n", collect="out"),
        ],
        outputs=["out"],
    )
    with pytest.raises(RunError, match="not iterable"):
        run(pipe, inputs={})


def test_map_pipeline_body():
    body = Pipeline(
        nodes=[
            Step(lambda w: w.strip(), inputs=["w"], outputs=["clean"], name="clean_word"),
            Step(lambda clean: clean.upper(), inputs=["clean"], outputs=["loud"], name="shout"),
        ],
        inputs={"word": "w"},
        outputs={"loud": "loud"},
        name="normalize",
    )
    pipe = Pipeline(
        nodes=[
            Step(lambda: [" a ", "b "], outputs=["words"], name="words_src"),
            Map(body=body, over="words", item="word", collect="normalized"),
        ],
        outputs=["normalized"],
    )
    assert run(pipe, inputs={}).outputs == {"normalized": ["A", "B"]}


def test_nested_map():
    pipe = Pipeline(
        nodes=[
            Step(lambda: [[1, 2], [3]], outputs=["rows"], name="rows_src"),
            Map(
                body=Map(Step(lambda cell: cell * 2, inputs=["cell"], outputs=["doubled"], name="double_cell"),
                         over="row", item="cell", collect="doubled_row", name="inner"),
                over="rows", item="row", collect="matrix", name="outer",
            ),
        ],
        outputs=["matrix"],
    )
    assert run(pipe, inputs={}).outputs == {"matrix": [[2, 4], [6]]}


def test_loop_until_convergence_with_trace():
    def improve(candidate):
        nxt = candidate / 2
        return {"candidate_next": nxt, "err": abs(nxt)}

    body = Pipeline(
        nodes=[Step(improve, inputs=["candidate"], outputs=["candidate_next", "err"])],
        inputs=["candidate"],
        outputs={"candidate_next": "candidate", "err": "err"},
        name="refine",
    )
    pipe = Pipeline(
        nodes=[
            Loop(
                body=body,
                carry={"candidate": "seed_value"},
                until=lambda err: err < 0.001,
                max_iter=1000,
                trace={"err": "err_history"},
                outputs={"candidate": "best"},
            )
        ],
        inputs=["seed_value"],
        outputs=["best", "err_history"],
    )
    report = run(pipe, inputs={"seed_value": 1.0})
    assert report.outputs["best"] < 0.001
    history = report.outputs["err_history"]
    assert history == sorted(history, reverse=True)
    assert 0 < len(history) < 1000


def test_loop_max_iter_without_until():
    body = Pipeline(
        nodes=[Step(lambda c: c + 1, inputs=["c"], outputs=["c_next"], name="bump")],
        inputs=["c"],
        outputs={"c_next": "c"},
        name="body",
    )
    pipe = Pipeline(
        nodes=[Loop(body=body, carry={"c": "start"}, max_iter=5, outputs={"c": "final"})],
        inputs=["start"],
        outputs=["final"],
    )
    assert run(pipe, inputs={"start": 0}).outputs == {"final": 5}


def test_loop_step_body_direct_carry():
    pipe = Pipeline(
        nodes=[Loop(body=Step(lambda total: total + total, inputs=["total"], outputs=["total"], name="double_up"),
                    carry={"total": "seed"}, max_iter=3, outputs={"total": "grown"})],
        inputs=["seed"],
        outputs=["grown"],
    )
    assert run(pipe, inputs={"seed": 1}).outputs == {"grown": 8}


def test_loop_broadcast_constant():
    body = Step(lambda total, inc: total + inc, inputs=["total", "inc"], outputs=["total"], name="add_up")
    pipe = Pipeline(
        nodes=[Loop(body=body, carry={"total": "seed"}, max_iter=4, outputs={"total": "sum"})],
        inputs=["seed", "inc"],
        outputs=["sum"],
    )
    assert run(pipe, inputs={"seed": 0, "inc": 3}).outputs == {"sum": 12}


def test_loop_body_failure():
    def blow(total):
        raise RuntimeError("turn failed")

    pipe = Pipeline(
        nodes=[Loop(body=Step(blow, inputs=["total"], outputs=["total"]), carry={"total": "t"}, max_iter=3, outputs={"total": "out"})],
        inputs=["t"],
        outputs=["out"],
    )
    with pytest.raises(RunError) as err:
        run(pipe, inputs={"t": 1})
    assert err.value.failures[0][0] == "loop[0].blow"


def test_loop_validation():
    body = Step(lambda total: total + 1, inputs=["total"], outputs=["total"], name="inc")
    with pytest.raises(TypeError, match="max_iter"):
        Loop(body=body, carry={"total": "t"})

    narrowed = Pipeline(
        nodes=[Step(lambda c: c, inputs=["c"], outputs=["other"], name="noop")],
        inputs=["c"],
        outputs={"other": "other"},
        name="body",
    )
    pipe = Pipeline(nodes=[Loop(body=narrowed, carry={"c": "x"}, max_iter=3)], inputs=["x"], outputs=[])
    with pytest.raises(ValidationError, match="does not export carry key 'c'"):
        pipe.validate()

    pipe = Pipeline(
        nodes=[Loop(body=body, carry={"total": "t"}, until=lambda q: True, max_iter=3)],
        inputs=["t"],
        outputs=[],
    )
    with pytest.raises(ValidationError, match="until reads 'q'"):
        pipe.validate()

    pipe = Pipeline(
        nodes=[Loop(body=body, carry={"total": "t"}, max_iter=3, trace={"nope": "hist"})],
        inputs=["t"],
        outputs=["hist"],
    )
    with pytest.raises(ValidationError, match="trace key 'nope'"):
        pipe.validate()

    def widen(total):
        return {"total": total + 1, "extra": total}

    widened = Step(widen, inputs=["total"], outputs=["total", "extra"], name="widen")
    pipe = Pipeline(
        nodes=[Loop(body=widened, carry={"total": "t"}, max_iter=3, outputs={"extra": "o"})],
        inputs=["t"],
        outputs=["o"],
    )
    with pytest.raises(ValidationError, match="not a carry key"):
        pipe.validate()


def test_branch_dispatch():
    def kind_of(value):
        return "small" if value < 10 else "big"

    pipe = Pipeline(
        nodes=[
            Branch(kind_of, inputs=["value"],
                   cases={"small": Step(lambda value: value * 2, inputs=["value"], outputs=["scaled"], name="sm"),
                          "big": Step(lambda value: value // 2, inputs=["value"], outputs=["scaled"], name="bg")}),
        ],
        inputs=["value"],
        outputs=["scaled"],
    )
    assert run(pipe, inputs={"value": 4}).outputs == {"scaled": 8}
    assert run(pipe, inputs={"value": 40}).outputs == {"scaled": 20}


def test_branch_default():
    def kind_of(value):
        return "small" if value < 10 else "big"

    pipe = Pipeline(
        nodes=[
            Branch(kind_of, inputs=["value"],
                   cases={"small": Step(lambda value: "kucuk", inputs=["value"], outputs=["labeled"], name="sm")},
                   default=Step(lambda value: "varsayilan", inputs=["value"], outputs=["labeled"], name="fallback"),
                   name="route"),
        ],
        inputs=["value"],
        outputs=["labeled"],
    )
    report = run(pipe, inputs={"value": 30})
    assert report.outputs == {"labeled": "varsayilan"}
    record = report.tree["nodes"][0]
    assert record["label"] == "big"
    assert record["chosen"]["node"] == "fallback"


def test_branch_unmatched_without_default():
    def kind_of(value):
        return "small" if value < 10 else "big"

    pipe = Pipeline(
        nodes=[
            Branch(kind_of, inputs=["value"],
                   cases={"small": Step(lambda value: value, inputs=["value"], outputs=["labeled"], name="sm")}),
        ],
        inputs=["value"],
        outputs=["labeled"],
    )
    with pytest.raises(RunError, match="no case for label"):
        run(pipe, inputs={"value": 30})


def test_branch_result_flows_downstream():
    def kind_of(value):
        return "small" if value < 10 else "big"

    pipe = Pipeline(
        nodes=[
            Branch(kind_of, inputs=["value"],
                   cases={"small": Step(lambda value: value * 2, inputs=["value"], outputs=["scaled"], name="sm"),
                          "big": Step(lambda value: value // 2, inputs=["value"], outputs=["scaled"], name="bg")},
                   name="route"),
            Step(lambda scaled: f"<{scaled}>", inputs=["scaled"], outputs=["boxed"], name="box"),
        ],
        inputs=["value"],
        outputs=["boxed"],
    )
    assert run(pipe, inputs={"value": 4}).outputs == {"boxed": "<8>"}
    assert run(pipe, inputs={"value": 40}).outputs == {"boxed": "<20>"}


def test_branch_outputs_must_match():
    def kind_of(value):
        return "small" if value < 10 else "big"

    pipe = Pipeline(
        nodes=[
            Branch(kind_of, inputs=["value"],
                   cases={"small": Step(lambda value: value, inputs=["value"], outputs=["x"], name="sm"),
                          "big": Step(lambda value: value, inputs=["value"], outputs=["y"], name="bg")}),
        ],
        inputs=["value"],
        outputs=["x"],
    )
    with pytest.raises(ValidationError, match="must produce the same outputs"):
        pipe.validate()


def test_nested_pipeline_boundary():
    bump = Pipeline(
        nodes=[Step(add1, inputs={"x": "v"}, outputs=["v_next"])],
        inputs={"first": "v"},
        outputs={"v_next": "second"},
        name="bump",
    )
    outer = Pipeline(
        nodes=[
            Step(add1, inputs={"x": "start"}, outputs=["first"], name="first_inc"),
            bump,
            Step(add1, inputs={"x": "second"}, outputs=["third"], name="last_inc"),
        ],
        inputs=["start"],
        outputs={"third": "result"},
    )
    report = run(outer, inputs={"start": 1})
    assert report.outputs == {"result": 4}
    bump_record = report.tree["nodes"][1]
    assert bump_record["node"] == "bump"
    assert bump_record["path"] == "bump"
    assert bump_record["nodes"][0]["path"] == "bump.add1"


def test_no_closure_over_outer_frame():
    inner = Pipeline(
        nodes=[
            Step(add1, inputs={"x": "v"}, outputs=["v_next"]),
            Step(add1, inputs={"x": "outer_secret"}, outputs=["peek"], name="peeker"),
        ],
        inputs=["v"],
        outputs=["v_next"],
        name="inner",
    )
    outer = Pipeline(
        nodes=[
            Step(add1, inputs={"x": "x"}, outputs=["outer_secret"], name="mk"),
            inner,
        ],
        inputs=["x", "v"],
        outputs=["v_next"],
    )
    with pytest.raises(ValidationError, match="'peeker' reads key 'outer_secret'"):
        outer.validate()


def test_nested_export_collision():
    inner = Pipeline(
        nodes=[Step(add1, inputs={"x": "v"}, outputs=["v_next"])],
        inputs={"x": "v"},
        outputs={"v_next": "y"},
        name="inner",
    )
    outer = Pipeline(
        nodes=[
            Step(add1, inputs={"x": "x"}, outputs=["y"], name="mk"),
            inner,
        ],
        inputs=["x"],
        outputs=["y"],
    )
    with pytest.raises(ValidationError, match="written by both"):
        outer.validate()


def test_events_subscriber():
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2], outputs=["nums"], name="src"),
            Map(Step(lambda n: n + 1, inputs=["n"], outputs=["bumped"], name="bump"),
                over="nums", item="n", collect="bumped_list", name="mapper"),
        ],
        outputs=["bumped_list"],
    )
    events = []
    subscribe(pipe, events.append)
    run(pipe, inputs={})
    kinds = {(event["path"], event["kind"]) for event in events}
    assert ("src", "started") in kinds
    assert ("src", "finished") in kinds
    assert ("mapper", "started") in kinds
    assert ("mapper[0]", "iter_started") in kinds
    assert ("mapper[1]", "iter_finished") in kinds
    assert ("mapper[0].bump", "finished") in kinds
    for event in events:
        assert event["schema"] == 1
        assert event["run"].startswith("r_")
        assert "t" in event


def test_subscriber_error_does_not_break_run():
    pipe = Pipeline(
        nodes=[
            Step(lambda: [1, 2], outputs=["nums"], name="src"),
            Map(Step(lambda n: n + 1, inputs=["n"], outputs=["bumped"], name="bump"),
                over="nums", item="n", collect="bumped_list", name="mapper"),
        ],
        outputs=["bumped_list"],
    )

    def bad_sink(event):
        raise RuntimeError("sink boom")

    subscribe(pipe, bad_sink)
    with pytest.warns(UserWarning, match="event sink failed"):
        report = run(pipe, inputs={})
    assert report.outputs == {"bumped_list": [2, 3]}


def test_record_dir(tmp_path):
    pipe = Pipeline(
        nodes=[Step(lambda: print("hello") or "x", outputs=["x"], name="printer")],
        outputs=["x"],
    )
    report = run(pipe, inputs={}, record_dir=str(tmp_path))
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["run"] == report.run
    assert {event["kind"] for event in parsed} == {"started", "finished"}
    summary = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert summary["run"] == report.run
    assert summary["status"] == "ok"
    assert summary["tree"]["nodes"][0]["node"] == "printer"
    assert "hello" in (tmp_path / "stdout.txt").read_text(encoding="utf-8")


def test_loop_events_and_traces_in_run_json(tmp_path):
    body = Step(lambda total: total + 1, inputs=["total"], outputs=["total"], name="inc")
    pipe = Pipeline(
        nodes=[Loop(body=body, carry={"total": "seed"}, max_iter=2,
                    trace={"total": "totals"}, outputs={"total": "out"}, name="grow")],
        inputs=["seed"],
        outputs=["out", "totals"],
    )
    events = []
    subscribe(pipe, events.append)
    run(pipe, inputs={"seed": 0}, record_dir=str(tmp_path))
    kinds = {(event["path"], event["kind"]) for event in events}
    assert ("grow[0]", "iter_started") in kinds
    assert ("grow[1]", "iter_finished") in kinds
    summary = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    loop_record = summary["tree"]["nodes"][0]
    assert loop_record["traces"] == {"totals": [1, 2]}
    assert len(loop_record["turns"]) == 2


def test_validation_missing_producer():
    pipe = Pipeline(nodes=[Step(use, inputs=["y"], outputs=["z"])], inputs=["x"], outputs=["z"])
    with pytest.raises(ValidationError, match="reads key 'y'"):
        pipe.validate()


def test_validation_double_write():
    pipe = Pipeline(
        nodes=[
            Step(make, inputs=["x"], outputs=["y"]),
            Step(use, inputs={"y": "x"}, outputs=["y"], name="again"),
        ],
        inputs=["x"],
        outputs=["y"],
    )
    with pytest.raises(ValidationError, match="written by both"):
        pipe.validate()


def test_validation_input_key_rewritten():
    pipe = Pipeline(nodes=[Step(make, inputs=["x"], outputs=["x"])], inputs=["x"], outputs=["x"])
    with pytest.raises(ValidationError, match="pipeline inputs"):
        pipe.validate()


def test_validation_declared_output_not_produced():
    pipe = Pipeline(nodes=[Step(make, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["z"])
    with pytest.raises(ValidationError, match="declared output 'z'"):
        pipe.validate()


def test_validation_signature_binding():
    def add(a, b):
        return a + b

    pipe = Pipeline(nodes=[Step(add, inputs={"a": "x"}, outputs=["s"])], inputs=["x"], outputs=["s"])
    with pytest.raises(ValidationError, match="required parameter 'b'"):
        pipe.validate()

    pipe = Pipeline(nodes=[Step(make, inputs={"x": "x", "q": "x"}, outputs=["y"])], inputs=["x"], outputs=["y"])
    with pytest.raises(ValidationError, match="no parameter with that name"):
        pipe.validate()


def test_validation_when_requires_no_outputs():
    pipe = Pipeline(
        nodes=[Step(make, inputs=["x"], outputs=["y"], when=lambda x: True)],
        inputs=["x"],
        outputs=["y"],
    )
    with pytest.raises(ValidationError, match="when"):
        pipe.validate()


def test_validation_wait_for_unknown_node():
    pipe = Pipeline(
        nodes=[Step(make, inputs=["x"], outputs=["y"], wait_for=["ghost"])],
        inputs=["x"],
        outputs=["y"],
    )
    with pytest.raises(ValidationError, match="unknown node 'ghost'"):
        pipe.validate()


def test_validation_duplicate_names():
    pipe = Pipeline(
        nodes=[
            Step(make, inputs=["x"], outputs=["y"]),
            Step(make, inputs=["x"], outputs=["z"]),
        ],
        inputs=["x"],
        outputs=["y", "z"],
    )
    with pytest.raises(ValidationError, match="share the name 'make'"):
        pipe.validate()


def test_validation_cycle():
    pipe = Pipeline(
        nodes=[
            Step(make, inputs=["x"], outputs=["y"], name="a", wait_for=["b"]),
            Step(use, inputs=["y"], outputs=["z"], name="b"),
        ],
        inputs=["x"],
        outputs=["z"],
    )
    with pytest.raises(ValidationError, match="cycle"):
        pipe.validate()


def test_validation_problems_collected():
    def add(a, b):
        return a + b

    pipe = Pipeline(nodes=[Step(add, inputs={"a": "x"}, outputs=["s"])], inputs=["x"], outputs=["t"])
    with pytest.raises(ValidationError) as err:
        pipe.validate()
    assert len(err.value.problems) == 2


def test_validation_pipeline_contains_itself():
    pipe = Pipeline(nodes=[Step(make, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    pipe.nodes.append(pipe)
    with pytest.raises(ValidationError, match="contains itself"):
        pipe.validate()


def test_validation_unused_output_warns():
    pipe = Pipeline(nodes=[Step(make, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=[])
    with pytest.warns(UnusedOutputWarning, match="never used"):
        pipe.validate()


def test_map_discarded_outputs_warn():
    def scale(n, factor):
        return n * factor

    pipe = Pipeline(
        nodes=[
            Step(numbers, outputs=["nums"]),
            Map(Step(scale, outputs=["scaled"]), over="nums", item="n"),
        ],
        inputs=["factor"],
        outputs=[],
    )
    with pytest.warns(UnusedOutputWarning, match="discarded"):
        pipe.validate()


def test_map_body_writing_iteration_key_rejected():
    pipe = Pipeline(
        nodes=[
            Step(numbers, outputs=["nums"]),
            Map(Step(lambda n: n, inputs=["n"], outputs=["n"], name="echo"), over="nums", item="n", collect="out"),
        ],
        outputs=["out"],
    )
    with pytest.raises(ValidationError, match="already exist in the iteration frame"):
        pipe.validate()


def test_run_input_mismatch():
    pipe = Pipeline(nodes=[Step(add1, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    with pytest.raises(ValidationError, match="missing input 'x'"):
        run(pipe, inputs={})
    with pytest.raises(ValidationError, match="unexpected input 'q'"):
        run(pipe, inputs={"x": 1, "q": 2})


def test_unknown_executor():
    pipe = Pipeline(nodes=[Step(add1, inputs=["x"], outputs=["y"])], inputs=["x"], outputs=["y"])
    with pytest.raises(ValueError, match="unknown executor"):
        run(pipe, inputs={"x": 1}, executor="warp")


def test_retry_recovers():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("not yet")
        return "done"

    pipe = Pipeline(nodes=[Step(flaky, outputs=["out"], retries=3)], outputs=["out"])
    assert run(pipe, inputs={}).outputs == {"out": "done"}
    assert len(calls) == 3


def test_retry_exhausted():
    calls = []

    def always_bad():
        calls.append(1)
        raise RuntimeError("still bad")

    pipe = Pipeline(nodes=[Step(always_bad, outputs=["out"], retries=1)], outputs=["out"])
    with pytest.raises(RunError):
        run(pipe, inputs={})
    assert len(calls) == 2
