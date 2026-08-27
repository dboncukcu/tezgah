from importlib.metadata import PackageNotFoundError, version

from .engine import Report, run, subscribe
from .errors import ContractError, RunError, TezgahError, UnusedOutputWarning, ValidationError
from .nodes import Branch, Loop, Map, Node, Pipeline, Step
from .records import RunCatalog, RunRecord, load_run

try:
    __version__ = version("tezgah")
except PackageNotFoundError:
    __version__ = None

__all__ = [
    "Branch",
    "ContractError",
    "Loop",
    "Map",
    "Node",
    "Pipeline",
    "Report",
    "RunCatalog",
    "RunError",
    "RunRecord",
    "Step",
    "TezgahError",
    "UnusedOutputWarning",
    "ValidationError",
    "__version__",
    "load_run",
    "run",
    "subscribe",
]
