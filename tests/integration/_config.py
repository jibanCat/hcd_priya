"""Integration tests on the real release data (D5 plan v3.1 E2): NOT run in CI. They run only when the environment
variable ``HCD_D5_INTEGRATION_CONFIG`` names a JSON configuration (kept with the private release records, outside this
repository) that gives the data root, the layout under it and the baselines recorded before the change. No real path or
real-data value is written in these files. Run locally, one file per process, under the shared lock:

    HCD_D5_INTEGRATION_CONFIG=/path/to/integration_config.json python -B -m pytest -q tests/integration/<file>.py
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest

ENV = "HCD_D5_INTEGRATION_CONFIG"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load():
    p = os.environ.get(ENV)
    if not p:
        return None
    if not os.path.isfile(p):
        raise pytest.UsageError(
            f"{ENV} names {p!r}, which is not a file (unset it to skip the integration tests)"
        )
    with open(p) as fh:
        return json.load(fh)


CFG = load()
needs_data = pytest.mark.skipif(
    CFG is None, reason=f"integration data not configured ({ENV} unset)"
)


def path(rel):
    """An absolute path under the configured data root (absolute entries are returned as they are)."""
    return rel if os.path.isabs(rel) else os.path.join(CFG["data_root"], rel)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run(args, cwd, timeout=3600):
    """A fresh process of this checkout's interpreter (bytecode off, user site off, CPU JAX)."""
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        JAX_PLATFORMS="cpu",
        HDF5_USE_FILE_LOCKING="FALSE",
    )
    return subprocess.run(
        [sys.executable, "-B", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=timeout,
    )


def require_clean_checkout():
    from hcd_analysis import lineage as L

    code = L.code_identity(REPO)
    if not (isinstance(code.get("commit"), str) and code.get("dirty") is False):
        pytest.fail(
            f"the export integration tests run from a clean, committed checkout (code identity {code})"
        )
    return code


def flat(o, p=""):
    """{leaf path: value} of a JSON document."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            out.update(flat(v, f"{p}/{k}"))
        return out or {p: {}}
    if isinstance(o, list):
        out = {}
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}[{i}]"))
        return out or {p: []}
    return {p: o}
