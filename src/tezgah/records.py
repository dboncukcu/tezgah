import json
import os


class RunRecord:
    def __init__(self, summary, events, directory):
        self.summary = summary
        self.directory = directory
        self.run = summary.get("run")
        self.status = summary.get("status")
        self.ms = summary.get("ms")
        self.t = summary.get("t")
        self.tree = summary.get("tree") or {}
        self.events = events

    def nodes(self):
        return list(_walk(self.tree))

    def find(self, path):
        for record in _walk(self.tree):
            if record.get("path") == path:
                return record
        return None

    def failed(self):
        return [
            record
            for record in self.nodes()[1:]
            if record.get("status") in ("failed", "upstream_failed")
        ]

    def durations(self):
        return {record.get("path"): record.get("ms") for record in _walk(self.tree)}

    def events_for(self, path):
        return [event for event in self.events if event.get("path") == path]

    def timeline(self):
        starts = {}
        bars = []
        for event in self.events:
            kind = event.get("kind")
            path = event.get("path")
            if kind == "started":
                starts[path] = event.get("t")
            elif kind in ("finished", "failed", "skipped") and path in starts:
                bars.append({
                    "path": path,
                    "node": event.get("node"),
                    "status": event.get("status") or kind,
                    "start": starts.pop(path),
                    "end": event.get("t"),
                })
        return bars

    def stdout(self):
        return _read_text(self.directory, "stdout.txt")

    def stderr(self):
        return _read_text(self.directory, "stderr.txt")


class RunCatalog:
    def __init__(self, root: str):
        self.root = root

    def list(self):
        entries = []
        if not os.path.isdir(self.root):
            return entries
        for name in os.listdir(self.root):
            summary_path = os.path.join(self.root, name, "run.json")
            if not os.path.isfile(summary_path):
                continue
            try:
                with open(summary_path, encoding="utf-8") as handle:
                    summary = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            entries.append({
                "dir": name,
                "run": summary.get("run"),
                "status": summary.get("status"),
                "ms": summary.get("ms"),
                "t": summary.get("t"),
            })
        entries.sort(key=lambda entry: entry.get("t") or "", reverse=True)
        return entries

    def get(self, name: str):
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            return None
        record_dir = os.path.join(self.root, name)
        if not os.path.isfile(os.path.join(record_dir, "run.json")):
            return None
        return load_run(record_dir)


def load_run(record_dir: str):
    with open(os.path.join(record_dir, "run.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    events = []
    events_path = os.path.join(record_dir, "events.jsonl")
    if os.path.exists(events_path):
        with open(events_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return RunRecord(summary, events, record_dir)


def _walk(record):
    if not record:
        return
    yield record
    for key in ("nodes", "iterations", "turns"):
        for child in record.get(key) or ():
            yield from _walk(child)
    chosen = record.get("chosen")
    if chosen:
        yield from _walk(chosen)


def _read_text(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()
