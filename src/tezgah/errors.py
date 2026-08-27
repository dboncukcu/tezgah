class TezgahError(Exception):
    pass


class ValidationError(TezgahError):
    def __init__(self, problems):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        if len(self.problems) == 1:
            message = self.problems[0]
        else:
            listing = "\n".join(
                f"  {i}. {problem}" for i, problem in enumerate(self.problems, start=1)
            )
            message = f"{len(self.problems)} problems found:\n{listing}"
        super().__init__(message)


class ContractError(TezgahError):
    pass


class RunError(TezgahError):
    def __init__(self, failures, tree=None):
        self.failures = list(failures)
        self.tree = tree
        details = "; ".join(
            f"{path}: {type(exc).__name__}: {exc}" for path, exc in self.failures
        )
        super().__init__(f"{len(self.failures)} failed node(s): {details}")


class UnusedOutputWarning(UserWarning):
    pass
