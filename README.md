# tezgah

A small, executor-agnostic pipeline engine that **statically validates flow graphs built from Python functions before running them**.

You describe your computation as a graph of named data keys flowing between plain Python functions. tezgah checks the whole graph before a single call is made, runs independent branches side by side on the executor of your choice (serial, thread pool, Dask process cluster, or any pool you bring), and emits a structured event stream you can record, query, or serve.

```python
from tezgah import Pipeline, Step, Map, run

pipe = Pipeline(
    nodes=[
        Step(read_logs, outputs=["raw"]),
        Step(parse_errors, inputs=["raw"], outputs=["errors"]),
        Step(parse_timings, inputs=["raw"], outputs=["timings"]),
        Step(merge, inputs=["errors", "timings"], outputs=["table"]),
        Map(body=Step(enrich, inputs=["row", "config"]), over="table",
            item="row", collect={"enrich": "enriched"}, parallel=4),
        Step(write_report, inputs=["enriched"], outputs=["report_path"]),
    ],
    inputs=["log_dir", "config"],
    outputs=["report_path"],
)

report = run(pipe, inputs={"log_dir": "logs/", "config": cfg}, executor="thread", workers=4)
```

`parse_errors` and `parse_timings` have no data dependency, so they run side by side automatically. `merge` waits for both. The `Map` fans the body out over the collection with at most 4 iterations in flight, and collects results **in input order**, regardless of completion order.

---

## Table of contents

- [Why](#why)
- [Install](#install)
- [Core concepts](#core-concepts)
- [The five building blocks](#the-five-building-blocks)
- [Static validation](#static-validation)
- [Execution model](#execution-model)
- [Runtime, in pictures](#runtime-in-pictures)
- [Executors](#executors)
- [Node-level executors (resource locking)](#node-level-executors-resource-locking)
- [Events and observability](#events-and-observability)
- [Records and the query layer](#records-and-the-query-layer)
- [Error model](#error-model)
- [Architecture notes](#architecture-notes)
- [Non-goals](#non-goals)
- [Development](#development)

## Why

Pipeline scripts rot because structure lives in code order, validation happens at runtime, and every project rebuilds the same scaffolding. tezgah makes the structure **declarative and checkable**:

- **The graph is validated before it runs.** Two-hour runs that would crash at minute 40 fail in two seconds, with *all* problems listed in one error.
- **Independence is structural, not scheduled.** Whether two nodes run together is decided by the absence of edges between them, not by list order or explicit task plumbing.
- **Results are executor-independent.** The same graph produces byte-identical outputs on serial, thread, and Dask executors; only timing changes.
- **Zero required dependencies.** The core is pure stdlib. Dask is an optional extra.
- **No string DSL.** The only strings in the grammar are names (bus keys, node names). Configuration is Python; logic is Python.

## Install

Requires Python 3.13 or newer.

```bash
pip install tezgah
```

Optional Dask executor support:

```bash
pip install 'tezgah[dask]'
```

From source, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/dboncukcu/tezgah.git
cd tezgah
uv sync
```

Check the installation:

```python
>>> import tezgah
>>> tezgah.__version__
'0.1.1'
```

## Core concepts

**Bus.** Named values flow between nodes through a shared namespace of keys. At runtime the concrete form is a **frame**: a flat key → value mapping. Every `Pipeline` opens its own frame; every `Map` iteration and every `Loop` turn runs in its own small frame.

**Single assignment.** Within a frame, a key is written exactly once — binding a pipeline input counts as a write. Two writers of the same key is a static error. This one rule buys two properties: parallelism without data races, and results that do not depend on execution order (the determinism guarantee).

**Edges.** Run order is derived from declared relations, never from list order:

- **Data edge**: a node's input key is another node's output key.
- **Control edge**: `wait_for=["node_name"]` — forces order without carrying data.

A node becomes ready only when *all* incoming edges are satisfied (AND semantics). A `wait_for` target that was skipped by `when` counts as satisfied ("finished or irrelevant").

**Frames are closed.** A nested pipeline sees only its declared inputs; nothing leaks in from the outer frame. Renaming at the boundary is free — the same sub-pipeline can bind to different keys in different parents.

**Users never see futures.** User functions receive plain values and return plain values. Futures are an engine-internal currency.

## The five building blocks

### Step — one function call

```python
Step(fn, inputs=None, outputs=None, name=None, when=None,
     wait_for=None, executor=None, retries=0, wait=0.0, unpack=False)
```

`fn` is any callable. Everything else is optional wiring.

**Input binding** — three forms, all resolving to `{parameter: bus_key}`:

| form | example | meaning |
|---|---|---|
| omitted | `Step(parse)` | derived from the signature: every named parameter binds to the same-named key (defaults included) |
| list | `inputs=["raw"]` | each key binds to the same-named parameter |
| dict | `inputs={"df": "raw"}` | rename: parameter `df` receives key `raw` |

With an explicit form, parameters with defaults may stay unbound (the Python default applies). Binding a parameter the function does not have is a static error (unless the function takes `**kwargs`).

**Output rules.** `unpack` says how the return value meets `outputs`; the number of outputs never decides anything.

| `unpack` | `outputs` | function must return | written to the frame |
|---|---|---|---|
| `False` (default) | omitted | anything | the whole value under `fn.__name__` |
| `False` | one key | anything, a mapping included | the whole value under that key |
| `False` | many keys | — | construction-time `ValueError`: one value cannot be wrapped under several names |
| `True` | one or many keys | a mapping containing those keys | each declared key picked by name; a missing key is a runtime `ContractError` |
| any | `[]` | anything (return value discarded) | nothing — a side-effect step |

With `unpack=True`, keys the mapping carries beyond the declared outputs are not written; the run keeps going and the kernel raises an `UnusedOutputWarning` naming them, and the node record lists them under `dropped`. Undeclared values still cannot leak into the frame.

```python
def split(a, b):
    return {"q": a // b, "r": a % b, "extra": 0}

Step(split, inputs=["a", "b"], outputs=["q", "r"], unpack=True)   # q and r written, "extra" warned about
Step(split, inputs=["a", "b"], outputs=["parts"])                  # parts = the whole mapping
```

**when** — a condition callable allowed *only on side-effect steps* (`outputs=[]`). Its parameters are looked up in the frame; `False` marks the step `skipped` and `fn` is never called. Steps with outputs may not carry `when` — that would create "maybe present" keys.

**retries / wait** — on failure the step is retried `retries` more times, sleeping `wait` seconds between attempts. Deliberately just two counters; no backoff machinery.

A bare callable in a node list is wrapped automatically:

```python
Pipeline(nodes=[double, clean])        # both become Steps
```

### Pipeline — a frame with nodes inside

```python
Pipeline(nodes, inputs=None, outputs=None, name="pipeline",
         executor=None, wait_for=None)
```

`nodes` is a list; **the list order is not the run order** — edges decide. `inputs` declares what this frame receives from outside; `outputs` declares what it exports. Dictionary forms follow data-flow direction:

```python
Pipeline(
    nodes=[...],
    inputs={"outer_key": "inner_name"},    # outer → inner
    outputs={"inner_name": "outer_key"},   # inner → outer
)
```

Example — a reusable sub-pipeline bound to different names in two parents:

```python
bump = Pipeline(
    nodes=[Step(inc, inputs={"v": "v"}, outputs=["v_next"])],
    inputs={"first": "v"},
    outputs={"v_next": "second"},
    name="bump",
)
# parent A: bump reads "first"; parent B: bump reads "initial" via inputs={"initial": "v"}
```

Failure semantics: a failed node marks its dependents `upstream_failed` and that propagates transitively; **independent branches keep running**. The engine finishes everything runnable, then raises `RunError` carrying failed paths, original exceptions, and the full status tree.

### Map — one body, per element

```python
Map(body, over, item="item", index=None, collect=None,
    parallel=False, executor=None, name="map", wait_for=None)
```

`over` is the *key* of the collection in the parent frame (not the collection itself). Each iteration runs in its own small frame containing:

- `item` — the element's name in the body frame
- `index` — the iteration number's name (if given); the natural hook for deterministic per-iteration seeding
- **broadcast** inputs — every key the body reads besides `item`/`index` is read **once** from the parent frame and handed to all iterations as the same object. Mutating a broadcast object in-place is the user's responsibility.

```python
Map(body=Step(scale, inputs=["n", "factor"]),
    over="nums", item="n", index="i",
    collect={"scale": "scaled"}, parallel=4)
# scale(n=element, i=position, factor=parent_frame["factor"])  per element
```

`collect` maps body outputs to parent keys, in data-flow direction like a pipeline's `outputs`: `{"scale": "scaled"}` gathers the body's `scale` value from every iteration into one list written as `scaled`; a list (`["scale"]`) keeps the names. Each collected key becomes its own list of plain values, one element per item. **Order is always input order, never completion order** — even with parallelism on, output is deterministic. A body output that is not collected is discarded and the validator warns; a `collect` key the body does not produce is a validation error.

If one iteration fails, no new ones start, in-flight iterations finish, and the `Map` node fails.

### Loop — sequential turns with carried state

```python
Loop(body, carry, range, index=None, until=None, trace=None,
     outputs=None, executor=None, name="loop", wait_for=None)
```

Turns are **never** parallelized — the dependency between turns is the point. The parameters:

- `carry`: `{carry_name: parent_frame_key}` — state passed turn to turn. **Closure rule:** the body must export every carry key; turn *i*'s output is turn *i+1*'s input.
- `range`: **mandatory** iteration source and termination guarantee, with Python semantics: an int, `[start, stop]`, `[start, stop, step]`, or a `range` object. Exhausting it without `until` firing is a normal end, not an error; an empty range runs zero turns and exports the initial carry.
- `index`: optional name; each turn the current range value enters the body frame under it, so `range=[1, 81], index="epoch"` reads exactly like `for epoch in range(1, 81)`. The loop owns the numbering: start the range wherever your domain counts from.
- `until`: called after each turn with named parameters looked up in that turn's exports; `True` ends the loop.
- `trace`: `{inner_key: parent_key}` — accumulates one value per turn into a list.
- `outputs`: `{carry_name: parent_key}` — exports the final carry values; keys are limited to carry names.

The canonical refinement pattern (rename inside, remap at the boundary — required because of single assignment inside a pipeline frame):

```python
refine = Pipeline(
    nodes=[Step(improve, inputs=["candidate"], outputs=["candidate_next", "err"], unpack=True)],
    inputs=["candidate"],
    outputs={"candidate_next": "candidate", "err": "err"},
)

Loop(body=refine,
     carry={"candidate": "seed_value"},
     range=1000,
     until=lambda err: err < 1e-6,
     trace={"err": "err_history"},
     outputs={"candidate": "best"})
```

A bare `Step` body can write the carry name directly (no intermediate frame); the closure rule forces it to produce the key anyway.

### Branch — pick one path by label

```python
Branch(decide, inputs=None, cases=None, default=None,
       executor=None, name="branch", wait_for=None)
```

`decide` is a callable returning a label; its parameters bind like `Step` inputs. The chosen case runs **in the parent frame** — it reads and writes parent keys directly. The **integrity rule**: all cases (and `default`, if present) must produce the same set of output keys, so downstream nodes never see "maybe present" keys. An unmatched label without a default is a runtime `ContractError`.

```python
Branch(kind_of, inputs=["value"],
       cases={
           "small": Step(shrink, inputs=["value"], outputs=["scaled"]),
           "big":   Step(grow,   inputs=["value"], outputs=["scaled"]),
       },
       default=Step(pass_through, inputs=["value"], outputs=["scaled"]))
```

### Comparison

|  | Pipeline | Map | Loop | Branch |
|---|---|---|---|---|
| body | different nodes | one body | one body | one node per label |
| runs | each node once | once per element | sequential turns | one case once |
| parallelism | across branches | across iterations | never | inside the chosen case |
| flow | inside the frame | element in, list out | carry across turns | in the parent frame |

## Static validation

`pipe.validate()` (also called automatically by `run`) walks the whole graph recursively, **collects every problem**, and reports them together in one error:

```
ValidationError: 2 problems found:
  1. outer: node 'peeker' reads key 'outer_secret' but nothing in this frame produces it
  2. outer: node 'first' is missing required parameter 'v'
```

The rulebook:

1. Every consumed key has a producer in the same frame.
2. Single assignment holds (pipeline input binding counts as a write).
3. Declared pipeline outputs are actually produced inside.
4. Signature binding fits: every bound parameter exists; every required parameter is bound.
5. `when` only on side-effect steps; `wait_for` targets exist.
6. No dependency cycles (data + control edges, DFS with cycle path reported).
7. Node names are unique per frame; no container contains itself.
8. Loop closure: carry exported, `until`/`trace` keys exported, `outputs` ⊆ carry.
9. Map: body does not write `item`/`index`/broadcast keys; every `collect` key is a body output.

Unused outputs are **warnings, not errors** (`UnusedOutputWarning`). Value-dependent conditions (iterability, picklability) cannot be known statically and fail at runtime with explicit messages.

## Execution model

`run()` is the single entry point:

```python
run(pipe, inputs=None, executor=None, workers=None, record_dir=None)
```

- `inputs` — must match the pipeline's declared inputs exactly (missing/unexpected keys raise before the run).
- `executor` — `"serial"` (default), `"thread"`, `"dask"`, or any object with `submit()`.
- `workers` — pool size for thread/dask; defaults to `os.cpu_count()`.
- `record_dir` — write `events.jsonl`, `run.json`, `stdout.txt`, `stderr.txt` here.

Returns a `Report`:

```python
report.outputs     # {declared_key: value}
report.tree        # status tree: node name, path, status, ms, nested records
report.run         # run id, e.g. "r_7f3a2c9b"
```

Node statuses: `ok`, `failed`, `skipped` (`when` was false), `upstream_failed` (a dependency failed).

**Determinism guarantee.** Because of single assignment, every key has exactly one value regardless of execution order. The same graph run serially, on threads, or on Dask produces identical `report.outputs`. What differs is timing and the interleaving of events from independent nodes.

**Map windows.** `parallel` controls how many iterations are *in flight* (submitted, not necessarily running):

- `False` → window of 1 (iterations strictly one at a time)
- `N` → at most N in flight
- `True` → window equals `workers` ("pool saturation"; deliberate guardrail — see Architecture notes)

## Runtime, in pictures

The timelines below are the runtime semantics of `run()`. Every picture is asserted by tests: timings in `tests/test_thread.py`, window equality and collection order in `tests/test_thread.py` and `tests/test_dask.py`.

### One graph, two timelines

List order never decides anything — edges do. The same graph under different executors:

```text
graph:                       (every step takes ~t)

  read_logs --+-> parse_errors --+
              |                  +-> merge
              +-> parse_timings -+

executor="serial" (default):        time -------------------------->

  read_logs       ####
  parse_errors         ####
  parse_timings             ####
  merge                          ####
  total: 4t

executor="thread", workers=2:      time ---------------->

  read_logs       ####
  parse_errors         ####        <- both edges were satisfied the
  parse_timings         ####       <- moment read_logs finished; two
  merge                      ####     workers -> side by side
  total: 3t
```

`parse_errors` and `parse_timings` never touch each other's outputs; that absence of an edge is what puts them on separate workers.

### Submit is not run

`submit()` hands work to a queue and returns immediately. The `workers` bound applies to what *runs*, not to what is *submitted*:

```text
workers=2, three independent steps ready at once: S1 S2 S3

submit(S1)   submit(S2)   submit(S3)        submit returns immediately
    |            |            |
    v            v            v
   S1           S2          [ S3 ]          RUNNING     WAITING
 worker1      worker2        queue

worker1 finishes S1 --> picks up S3 from the queue.
At no moment do three leaves run.
```

### Map windows

```text
Map(..., parallel=3) with workers=2, over [n0..n4]

          t0      t1      t2      t3
  w1     n0 ##   n2 ##   n4 ##
  w2     n1 ##   n3 ##

  in flight (submitted, not finished):  {n0,n1,n2} -> {n2,n3,n4} -> ...
  window  = at most 3 in flight
  workers = at most 2 actually running

  collected list = [r0, r1, r2, r3, r4]     input order, always.
                                      (r3 may finish before r2; it still
                                       lands at index 3)
```

### Where waiting happens: two pools

Containers wait; leaves compute. Waiting happens in a place where it blocks no one:

```text
run(..., executor="thread", workers=N)

  containers (Pipeline / Map / Loop / Branch .execute)
      each runs in its own spawned thread
      their job is to wait; a parked thread costs memory, not CPU,
      and holds no pool slot
              |
              |  submit(leaf)
              v
  work pool: N workers
      only user functions run here
      they never wait for anything; they always finish


  why the split — if containers ran INSIDE the bounded pool:

    w1: [ Pipeline A -- waiting for its children ]   slot held forever
    w2: [ Pipeline B -- waiting for its children ]   slot held forever
    queue: A.c1, A.c2, B.c1, ...                     never gets a worker
    => deadlock

  tezgah's rule: waiters hold no pool slots. The waiting graph mirrors
  the node graph, which validation proved acyclic -> no deadlock, by
  construction rather than by discipline.
```

### Loop turns

Turns cannot overlap by definition — turn *i+1* consumes turn *i*'s carry:

```text
  carry:  seed --> [turn 0] --> [turn 1] --> [turn 2] --> outputs (final carry)
                    t0 ####      t1 ####      t2 ####

  trace:              e0         e0,e1       e0,e1,e2      (one value per turn)

  until is checked after every turn: the loop stops the moment it fires;
  the range is the hard bound either way.
```

## Executors

| name | backed by | use for |
|---|---|---|
| `"serial"` | inline calls | default; debugging; full determinism |
| `"thread"` | `ThreadPoolExecutor` | independent branches; I/O-bound work; numpy/pandas/torch (C kernels release the GIL) |
| `"dask"` | `LocalCluster` (processes) | pure-Python heavy compute across cores |
| instance | anything with `submit()` | your own pool |

```python
run(pipe, inputs=..., executor="serial")
run(pipe, inputs=..., executor="thread", workers=8)
run(pipe, inputs=..., executor="dask", workers=4)
run(pipe, inputs=..., executor=my_pool)          # instance: you own its lifetime
```

Process boundaries (Dask): everything crossing the boundary is pickled — use module-level functions as `Step` fns, and pass references (file paths) instead of large data. The usual rule of thumb applies: process parallelism pays off only when `gained_compute > serialization + transport + process_spawn`.

**Writing your own pool** — the whole protocol:

```python
class MyPool:
    def submit(self, fn, *args): ...   # schedule, return a future (any shape)
```

Optionally implement `shutdown()` for lifecycle. Futures must expose `result()` and, when mixed with non-stdlib futures, `add_done_callback()`. That's it — the engine recognizes the stdlib `concurrent.futures` shape natively and falls back to a callback-completion queue otherwise. Rate limiting, priority queues, resource quotas: policy is yours, mechanism is the engine's.

## Node-level executors (resource locking)

Any node may declare its own executor **instance** (strings are reserved for `run()` and rejected here). Unset nodes inherit from their enclosing container, which inherits from the run-level pool.

```python
from concurrent.futures import ThreadPoolExecutor

gpu = ThreadPoolExecutor(max_workers=1)   # one worker = a lock with a queue

Pipeline(nodes=[
    Step(prepare_data, outputs=["batch"]),                                # main pool
    Step(train,    inputs=["batch"], outputs=["model"],  executor=gpu),   # serialized
    Step(evaluate, inputs=["model"],  outputs=["score"], executor=gpu),   # serialized
    Step(log_metrics, inputs=["score"], outputs=["log"]),                 # main pool
])
```

GPU-touching steps line up one by one with no lock code, no waiting code — while the rest of the graph flows through the main pool. The same pattern fits DB connections, rate-limited APIs, licensed tools. Precedence for a body: the body's own executor wins over the container's.

Pools declared on nodes are owned by you — the engine shuts down only pools it created itself.

## Events and observability

The contract is the **event schema**, not any particular consumer:

```json
{"schema": 1, "run": "r_7f3a2c9b", "path": "cv.fold[2].train", "node": "train_fold",
 "kind": "finished", "status": "ok", "ms": 8412.3, "t": "2026-08-25T19:30:11.123"}
```

- `path` is the hierarchical frame path: container names joined by dots, Map iterations and Loop turns numbered with brackets. `cv.fold[2].train` = the `train` step of iteration 2 of the `fold` Map inside the `cv` sub-pipeline.
- Values never appear in events — only key names and metadata.
- Kinds: `started`, `finished`, `failed`, `skipped`, `iter_started`, `iter_finished`.

Subscribe to the stream:

```python
from tezgah import subscribe

def sink(event):
    print(event["path"], event["kind"], event.get("ms"))

subscribe(pipe, sink)
run(pipe, inputs={...})
```

A crashing sink is converted to a warning; it never breaks the run. With `record_dir`, a file sink writes `events.jsonl` line by line while the run is live, plus a `run.json` summary at the end; stdout/stderr are captured to files while still flowing to the console.

## Records and the query layer

`record_dir` produces:

```
runs/x1/
  events.jsonl    # the event stream, one JSON object per line
  run.json        # summary: run id, status, t, ms, full status tree
  stdout.txt      # captured stdout (tee'd to console)
  stderr.txt
```

The discovery contract: *a directory containing `run.json` is a run*. No index files to maintain, crash-safe by construction.

```python
from tezgah import load_run, RunCatalog

record = load_run("runs/x1")
record.run, record.status, record.ms, record.t
record.summary                        # the full run.json dict, as-is
record.tree                           # status tree
record.nodes()                        # flat list of node records
record.find("cv.fold[2].train")       # one node by path
record.failed()                       # failed + upstream_failed records
record.durations()                    # {path: ms}
record.events_for("cv.fold[2]")       # events of one node
record.timeline()                     # [{path, node, status, start, end}] for Gantt views
record.stdout(), record.stderr()      # captured output

catalog = RunCatalog("runs")
catalog.list()                        # newest first: [{dir, run, status, ms, t}]
catalog.get("x1")                     # RunRecord or None (path-traversal safe)
```

`records.py` is deliberately a pure query layer — no HTTP, no frameworks. A dashboard is a thin adapter over it:

```python
# your project (fastapi is NOT a tezgah dependency)
app = FastAPI()
catalog = RunCatalog("runs")

@app.get("/api/runs")
def list_runs():
    return catalog.list()

@app.get("/api/runs/{name}/timeline")
def timeline(name: str):
    return catalog.get(name).timeline()
```

## Error model

| type | when | payload |
|---|---|---|
| `ValidationError` | before the run, statically | `.problems` — the complete numbered list |
| `ContractError` | during the run | mapping violations, non-iterable `over`, unmatched branch label |
| `RunError` | end of a failed run | `.failures` — `[(node_path, original_exception)]`; `.tree` — status tree; first original exception chained via `__cause__` |

```python
from tezgah import RunError

try:
    run(pipe, inputs={"x": 1})
except RunError as err:
    for path, exc in err.failures:
        print(path, type(exc).__name__, exc)
```

## Architecture notes

The design in one sentence: **nodes carry their own execution rules; the kernel only serves** — submit (run a child), wait (one of these finishes), deliver (consume a result), emit (publish an event).

- **The kernel is graph-blind.** It knows no node types, no dependencies, no scheduling policy. Its only per-node knowledge is the `leaf` flag: leaves go to a bounded work pool and never wait; containers spawn their own unbounded thread to wait in.
- **Deadlock-free by construction.** Waiting containers hold no pool slots ("waiters don't hold CPUs"); the waiting graph mirrors the statically acyclic node graph, so a cycle in waits is impossible.
- **Frames are written on one side only.** `execute()` receives values and *returns* what it would write; the caller applies the writes to its own frame. No locks on frames, no races by construction.
- **No two engines.** Serial mode is the same kernel with pools replaced by inline execution — one code path, fully deterministic. Parallel mode swaps in the two pools without touching node code.
- **`parallel=True` means `workers`.** A window larger than the pool gains nothing for leaf bodies, and for container bodies each in-flight iteration costs a spawn thread — unbounded windows would be a thread bomb. True unbounded is available by passing an explicit integer if you know why.

Non-goals, deliberate: no scheduler/daemon (processes are born on call and die when done), no multi-machine orchestration, no string DSL in configuration, no shared object store, no backoff machinery. `follow()` (live tailing) and resume are shelved until needed.

## Development

```bash
uv sync                                   # env with pytest + dask
uv run pytest -q                          # full suite (104 tests)
uv run pytest tests/test_smoke.py -q      # semantics only, fast
```

The test suite is the executable contract of the semantics above, and runs the same scenarios across serial, thread, custom-pool, and Dask executors — asserting identical outputs.

Project layout:

```
tezgah/
  src/tezgah/
    nodes/        # the grammar: base, step, pipeline, map, loop, branch
    validate.py   # static analysis, one function per rule
    engine.py     # Kernel (submit/wait/deliver/emit) + run() gate
    executors.py  # executor resolution + lazy-imported DaskPool
    records.py    # run.json/events.jsonl readers: RunRecord, RunCatalog
    errors.py
  tests/
```

License: MIT — see [LICENSE](LICENSE).
