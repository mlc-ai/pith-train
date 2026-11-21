"""
Session options shared by every test (imports only pytest, so the CPU-only checks run without a GPU):

- --shard I/N keeps every N-th collected test starting at I, so N processes (one per GPU) split a
  tier between them: CUDA_VISIBLE_DEVICES=I pytest -m spec --shard I/N.
- A test marked multigpu that skips (too few GPUs or ranks) fails instead, unless
  --allow-multigpu-skip: a multi-GPU run that silently tests nothing is the failure mode this
  guards against.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--shard", default=None, help="I/N: run every N-th test starting at I")
    parser.addoption("--allow-multigpu-skip", action="store_true", help="let multigpu tests skip")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard = config.getoption("--shard")
    if shard is None:
        return
    index, count = (int(part) for part in shard.split("/"))
    if not 0 <= index < count:
        raise pytest.UsageError(f"--shard {shard}: need 0 <= I < N")
    keep = [item for position, item in enumerate(items) if position % count == index]
    config.hook.pytest_deselected(items=[item for item in items if item not in keep])
    items[:] = keep


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker("multigpu") is not None:
        if not item.config.getoption("--allow-multigpu-skip"):
            report.outcome = "failed"
            reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
            report.longrepr = f"multigpu test skipped ({reason}); run it on enough GPUs or pass --allow-multigpu-skip"
