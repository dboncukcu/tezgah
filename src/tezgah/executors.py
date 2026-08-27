import os


def resolve_executor(executor, workers):
    if executor is None or executor == "seri":
        return None
    if executor == "thread":
        from concurrent import futures

        return futures.ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1)
    if executor == "dask":
        return DaskPool(workers=workers)
    if hasattr(executor, "submit"):
        return executor
    raise ValueError(f"unknown executor {executor!r}, available: 'seri', 'thread', 'dask' or any object with submit()")


class DaskPool:
    def __init__(self, workers=None, processes=True):
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as exc:
            raise ImportError(
                "DaskPool needs dask; install it with: uv add 'tezgah[dask]' (or pip install 'tezgah[dask]')"
            ) from exc
        self.cluster = LocalCluster(
            n_workers=workers or os.cpu_count() or 1,
            processes=processes,
            dashboard_address=None,
        )
        self.client = Client(self.cluster)

    def submit(self, fn, *args):
        return self.client.submit(fn, *args, pure=False)

    def shutdown(self):
        self.client.close()
        self.cluster.close()
