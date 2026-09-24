"""Synthetic tests for scripts/eboss_unblind_once.py (no real data, no NUTS, no real blind.lock).

Builds a fake blinded chain directory exactly in the run_real_fit.export_getdist layout, a fake blind.lock
(written with the frozen writer, fixed commit string), a fake analysis.lock carrying the lock's sha256, a fake
authorization record, then checks: dry-run writes nothing and never prints the offset; the formal run
recovers the physical draws exactly; the stamp makes a second run refuse; every tamper refuses.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
from hcd_analysis.emulator import blinding as BL
from hcd_analysis.emulator.inference import PARAM_NAMES

REPO = "/home/mfho/hcd_priya"
SCRIPT = os.path.join(REPO, "scripts", "eboss_unblind_once.py")
NAMES = list(PARAM_NAMES) + [f"tau0_z{i}" for i in range(13)] + ["alpha_lls", "alpha_subdla", "alpha_dla"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _build(tmp_path, n_chains=2, n=50, seed=0):
    rng = np.random.default_rng(seed)
    d = tmp_path / "chains"; d.mkdir()
    lock = tmp_path / "blind.lock"
    BL.write_blind_lock(str(lock), "test_project", commit="deadbeef")
    offset = BL.offset_from_lock(str(lock))
    phys = []
    for c in range(n_chains):
        draws = rng.uniform(0.2, 0.8, size=(n, len(NAMES)))
        draws[:, 0] = 0.9 + 0.01 * rng.standard_normal(n)          # ns
        draws[:, 1] = 1.5e-9 + 1e-10 * rng.standard_normal(n)      # Ap
        phys.append(draws.copy())
        shown = BL.apply_blind(draws, offset, columns=NAMES)
        table = np.column_stack([np.ones(n), rng.uniform(100, 200, n), shown])
        np.savetxt(d / f"real_eboss.{c + 1}.txt", table, fmt=["%.8g"] * table.shape[1],
                   header="weight  minusloglike  " + "  ".join(NAMES))
    (d / "real_eboss.paramnames").write_text("".join(f"{n}\t{n}\n" for n in NAMES))
    health = dict(leg="eBOSS", survey="eboss", blinded=True, blind_params=["ns", "Ap"], n_chains=n_chains, n_draws=n)
    (d / "real_eboss.health.json").write_text(json.dumps(health))
    with open(d / "SHA256SUMS", "w") as f:
        for name in sorted(os.listdir(d)):
            if name != "SHA256SUMS":
                f.write(f"{_sha(d / name)}  {name}\n")
    alock = tmp_path / "analysis.lock"
    alock.write_text(json.dumps(dict(blinding=dict(blind_lock_sha256=_sha(lock)))))
    auth = tmp_path / "PI-24.md"; auth.write_text("PI DECISIONS #24: authorized\n")
    return d, lock, alock, auth, offset, phys


def _run(d, lock, alock, auth, extra=(), confirm="eBOSS", auth_sha=None, events=None):
    cmd = [sys.executable, SCRIPT, "--chain-dir", str(d), "--root", "real_eboss", "--blind-lock", str(lock),
           "--analysis-lock", str(alock), "--authorization", str(auth), "--authorization-sha256", auth_sha or _sha(auth),
           "--events-log", str(events or (d.parent / "unblind_events.log")), "--confirm", confirm,
           "--code-repo", REPO, "--notes-repo", REPO] + list(extra)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=REPO, JAX_PLATFORMS="cpu")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_dry_run_writes_nothing_and_hides_offset(tmp_path):
    d, lock, alock, auth, offset, phys = _build(tmp_path)
    before = sorted(os.listdir(d))
    r = _run(d, lock, alock, auth, extra=["--dry-run"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "DRY-RUN OK"
    assert sorted(os.listdir(d)) == before
    for v in offset.values():
        assert f"{v:.6g}" not in r.stdout + r.stderr


def test_formal_run_recovers_physical_draws_exactly_and_stamps(tmp_path):
    d, lock, alock, auth, offset, phys = _build(tmp_path)
    r = _run(d, lock, alock, auth)
    assert r.returncode == 0, r.stderr
    for c in range(2):
        t = np.loadtxt(d / f"real_eboss.unblinded.{c + 1}.txt", ndmin=2)
        np.testing.assert_allclose(t[:, 2:], phys[c], rtol=0, atol=2e-8 * np.abs(phys[c]).max() + 1e-16)
        # relative precision of %.8g on the shown values, then exact subtraction
        assert np.allclose(t[:, 2 + 2:], phys[c][:, 2:], rtol=1e-7, atol=0)
    stamp = json.load(open(d / "UNBLINDED.stamp"))
    assert stamp["leg"] == "eBOSS" and stamp["applied_offset"] == {k: float(v) for k, v in offset.items()}
    assert set(stamp["blinded_chain_sha256"]) == {"real_eboss.1.txt", "real_eboss.2.txt"}
    assert (d / "real_eboss.unblinded.paramnames").read_text() == (d / "real_eboss.paramnames").read_text()
    ev = (tmp_path / "unblind_events.log").read_text().strip().splitlines()
    assert len(ev) == 1 and json.loads(ev[0])["leg"] == "eBOSS"
    # blinded files untouched
    for want, name in [l.split() for l in (d / "SHA256SUMS").read_text().splitlines()]:
        assert _sha(d / name) == want
    # second run refuses (unblind-once)
    r2 = _run(d, lock, alock, auth)
    assert r2.returncode == 3 and "already unblinded" in r2.stderr


@pytest.mark.parametrize("tamper", ["confirm", "auth_sha", "lock_sha", "sums", "not_blinded", "output_exists"])
def test_refusals(tmp_path, tamper):
    d, lock, alock, auth, offset, phys = _build(tmp_path)
    kw = {}
    if tamper == "confirm":
        kw["confirm"] = "DESI"
    elif tamper == "auth_sha":
        kw["auth_sha"] = "0" * 64
    elif tamper == "lock_sha":
        alock.write_text(json.dumps(dict(blinding=dict(blind_lock_sha256="1" * 64))))
    elif tamper == "sums":
        p = d / "real_eboss.1.txt"; p.write_text(p.read_text().replace("1", "2", 1))
    elif tamper == "not_blinded":
        h = json.load(open(d / "real_eboss.health.json")); h["blinded"] = False
        (d / "real_eboss.health.json").write_text(json.dumps(h))
        lines = [l for l in (d / "SHA256SUMS").read_text().splitlines() if "health" not in l]
        (d / "SHA256SUMS").write_text("\n".join(lines) + f"\n{_sha(d / 'real_eboss.health.json')}  real_eboss.health.json\n")
    elif tamper == "output_exists":
        (d / "real_eboss.unblinded.1.txt").write_text("x")
    before = sorted(os.listdir(d))
    r = _run(d, lock, alock, auth, **kw)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert r.stderr.startswith("REFUSE:")
    assert sorted(os.listdir(d)) == before
    assert not (d / "UNBLINDED.stamp").exists()


def test_module_loads_without_jax_import_at_top():
    spec = importlib.util.spec_from_file_location("ubo", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert callable(mod.main) and callable(mod.verify_sha256sums)
