"""TDD for the RCINJ out-of-span (OOS) injection switch in run_stepA.

The NORC res_corr TRUTH-injection gate (RCINJ block in run_stepA.build_config) injects, by default, the
ACTUAL small-box res_corr table ({"actual_res_corr": True}). This test covers the env switch that swaps the
injection to the OUT-OF-SPAN basis member (res_corr_injection_basis.npz; the adversarial worst-n_s direction
C^-1-orthogonal to the 2-param alpha_res span, already scaled to the +-5% He-II envelope). The metals precedent
(NUTS found an out-of-span leak after an in-span pass) motivates this arm.

The selection is factored into a pure helper ``_rcinj_primary_spec(repo, environ)`` so it is testable without
building a ctx or running NUTS.
"""
import importlib.util
import os

_RUNSTEPA = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_stepA.py")


def _load_runstepA():
    spec = importlib.util.spec_from_file_location("run_stepA", _RUNSTEPA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_rcinj_primary_spec"), "run_stepA must expose _rcinj_primary_spec"
    return mod


def test_default_is_actual_res_corr():
    """Unset RCINJ_OOS_MEMBER -> the deployed default (inject the actual small-box res_corr table)."""
    m = _load_runstepA()
    spec = m._rcinj_primary_spec(repo="/x", environ={})
    assert spec == {"actual_res_corr": True, "strength": 1.0}


def test_oos_member_switches_to_basis():
    """RCINJ_OOS_MEMBER='b1' -> inject the basis member via the (path, member, strength) form; NO actual_res_corr."""
    m = _load_runstepA()
    spec = m._rcinj_primary_spec(repo="/x", environ={"RCINJ_OOS_MEMBER": "b1"})
    assert spec["path"] == "/x/hcd_analysis/_emulator_data/res_corr_injection_basis.npz"
    assert spec["member"] == "b1"
    assert spec["strength"] == 1.0
    assert "actual_res_corr" not in spec


def test_strength_scales_both_forms():
    """RCINJ_RC_STRENGTH scales either injection form (the +-1sigma systematic scan)."""
    m = _load_runstepA()
    a = m._rcinj_primary_spec(repo="/x", environ={"RCINJ_RC_STRENGTH": "0.5"})
    assert a["strength"] == 0.5 and a.get("actual_res_corr") is True
    b = m._rcinj_primary_spec(repo="/x", environ={"RCINJ_OOS_MEMBER": "b2", "RCINJ_RC_STRENGTH": "1.5"})
    assert b["member"] == "b2" and b["strength"] == 1.5


def test_defaults_to_process_environ():
    """environ=None reads os.environ (the production path); default (unset) is actual_res_corr."""
    m = _load_runstepA()
    saved = {k: os.environ.pop(k) for k in ("RCINJ_OOS_MEMBER", "RCINJ_RC_STRENGTH") if k in os.environ}
    try:
        spec = m._rcinj_primary_spec()
        assert spec.get("actual_res_corr") is True and spec["strength"] == 1.0
    finally:
        os.environ.update(saved)
