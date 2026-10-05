"""Run the UPSTREAM reference implementation (lya_emulator_full @ 5e98627 + fake_spectra 2.2.3, emu-3.9 environment)
in a subprocess and return JSON, so parity tests compare against the actual upstream code (gate B, emulator-debug
campaign 2026-10). The upstream repository and environment are separate from this checkout (absolute paths allowed)."""
import json
import os
import subprocess

import pytest

EMU39_PY = "/home/mfho/.conda/envs/emu-3.9/bin/python"
UPSTREAM = "/home/mfho/lya_emulator_full"
UPSTREAM_PRODUCT = f"{UPSTREAM}/kodiaq_2_2_4_6-48-48"
GSL_LIB = "/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib"


def require_upstream():
    missing = [p for p in (EMU39_PY, UPSTREAM, UPSTREAM_PRODUCT, GSL_LIB) if not os.path.exists(p)]
    if not missing:
        return
    msg = f"upstream reference unavailable: {missing}"
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1)")
    pytest.skip(msg)


def run_upstream(code, extra_pythonpath=None, timeout=900):
    """Execute ``code`` with the upstream interpreter; the code must print one JSON document as its last line."""
    require_upstream()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{GSL_LIB}:/home/mfho/.conda/envs/emu-3.9/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = UPSTREAM + (os.pathsep + extra_pythonpath if extra_pythonpath else "")
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("JAX_PLATFORMS", None)
    r = subprocess.run([EMU39_PY, "-c", code], capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"upstream subprocess failed ({r.returncode}):\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])
