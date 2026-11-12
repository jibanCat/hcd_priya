"""Global pytest config: enable JAX float64 before any test creates arrays.

The Phase-2b emulator's structural identities are bit-level and require double
precision (see hcd_analysis/emulator/__init__.py, which also enables x64 on
import). Setting it here at collection time guarantees x64 even if a test
imports jax before importing the emulator package. No effect on the numpy-based
non-emulator tests.

Also defines the ``deferred_legacy(cause)`` marker (see tests/DEFERRED.md): a test of the pre-2026-10 inference
and closure stack whose interface was retired at gate E is expected to fail ONLY with the exact error of its cause.
Any other failure fails the test, and a pass fails it too (strict), so each marker is removed when the test is
rebuilt against the current interface.
"""

import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tests.gate_helpers import DEFERRED_CAUSES  # noqa: E402  (the patterns live with the guards)


def pytest_sessionfinish(session, exitstatus):
    """Every test session, however targeted, ends with the module-origin check (gate A review, BT8 recommendation):
    a project module imported from another checkout fails the session."""
    import sys
    from tests.gate_helpers import foreign_project_modules

    foreign = foreign_project_modules(sys.modules)
    if foreign:
        print(
            f"\nCHECKOUT INTEGRITY FAILURE: project modules loaded from another checkout: {foreign[:10]}"
        )
        session.exitstatus = 1


def pytest_configure(config):
    # register the `slow` marker (the ~8-min NUTS 0-divergence funnel smoke is marked slow).
    config.addinivalue_line("markers", "slow: marks slow tests (deselect with '-m \"not slow\"')")
    config.addinivalue_line(
        "markers",
        "deferred_legacy(cause): retired pre-2026-10 interface; expected to fail "
        "only with the cause's exact error (tests/DEFERRED.md)",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    marker = item.get_closest_marker("deferred_legacy")
    if marker is None:
        return
    rep = outcome.get_result()
    cause = marker.args[0]
    pattern, reason = DEFERRED_CAUSES[cause]  # an unknown cause is a KeyError: fail loudly
    if call.excinfo is not None and rep.failed and call.when in ("setup", "call"):
        if re.search(pattern, call.excinfo.exconly()):
            rep.outcome = "skipped"
            rep.wasxfail = f"deferred_legacy {cause}: {reason}"
    elif call.when == "call" and rep.passed:
        rep.outcome = "failed"
        rep.longrepr = (
            f"[XPASS(strict)] deferred_legacy {cause} passed: the cause is gone, remove the marker "
            f"(tests/DEFERRED.md)"
        )
