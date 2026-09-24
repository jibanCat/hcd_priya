"""Synthetic tests for scripts/eboss_unblind_once.py v2 (no real data, no NUTS, no real blind.lock).

Builds a fake WRAPPER-PRODUCED blinded chain directory inside a throw-away git repo (the notes-repo stand-in): chains in the
run_real_fit.export_getdist layout, SHA256SUMS, EXECUTION_RECORD.json (schema v1, exit 0, consumed), health.json, a
nuisance npz/json pair, a fake blind.lock (frozen writer, fixed commit), a fake analysis.lock and authorization record.
"""
import hashlib
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
GREEN = dict(rhat_max=1.004, ess_bulk_min=900.0, ess_tail_min=700.0, ebfmi_min=0.9, n_divergent=0, treedepth_sat_frac=0.0)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)


def _build(tmp_path, n_chains=2, n=500, seed=0, health=None, nuisance=True, record=True, commit=True, rail=False):
    rng = np.random.default_rng(seed)
    notes = tmp_path / "notes"; notes.mkdir()
    _git(notes, "init", "-q"); _git(notes, "config", "user.email", "t@t"); _git(notes, "config", "user.name", "t")
    d = notes / "artifacts" / "chains" / "blinded" / "eboss_blind_test"; d.mkdir(parents=True)
    lock = tmp_path / "blind.lock"
    BL.write_blind_lock(str(lock), "test_project", commit="deadbeef")
    offset = BL.offset_from_lock(str(lock))
    phys = []
    for c in range(n_chains):
        draws = rng.uniform(0.2, 0.8, size=(n, len(NAMES)))
        draws[:, 0] = 0.9 + 0.01 * rng.standard_normal(n)
        draws[:, 1] = 1.5e-9 + 1e-10 * rng.standard_normal(n)
        phys.append(draws.copy())
        shown = BL.apply_blind(draws, offset, columns=NAMES)
        table = np.column_stack([np.ones(n), rng.uniform(100, 200, n), shown])
        np.savetxt(d / f"real_eboss.{c + 1}.txt", table, fmt=["%.8g"] * table.shape[1], header="weight  minusloglike  " + "  ".join(NAMES))
    (d / "real_eboss.paramnames").write_text("".join(f"{nm}\t{nm}\n" for nm in NAMES))
    h = dict(leg="eBOSS", survey="eboss", blinded=True, blind_params=["ns", "Ap"], n_chains=n_chains, n_draws=n, **(health or GREEN))
    (d / "real_eboss.health.json").write_text(json.dumps(h))
    if nuisance:
        sites = {k: 0.01 * (1 + 0.05 * rng.standard_normal((n_chains, n))) for k in ("f_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z0")}
        sites["f_res_amp"] = 0.01 * rng.standard_normal((n_chains, n))
        np.savez(d / "real_eboss.nuisance.npz", **sites)
        js = dict(sites={k: dict(frac_near_lo=(0.3 if rail else 0.0), frac_near_hi=0.0) for k in sites})
        (d / "real_eboss.nuisance.json").write_text(json.dumps(js))
    payload = sorted(f for f in os.listdir(d))
    with open(d / "SHA256SUMS", "w") as f:
        for name in payload:
            f.write(f"{_sha(d / name)}  {name}\n")
    if record:
        rec = dict(schema="eboss_realfit_execution_record.v1", exit_status=0, formal_execution_consumed=True,
                   outputs_sha256={name: _sha(d / name) for name in payload}, code_head="abc", wrapper_sha256="def")
        (d / "EXECUTION_RECORD.json").write_text(json.dumps(rec))
    if commit:
        _git(notes, "add", "-A"); _git(notes, "commit", "-q", "-m", "chains")
    alock = tmp_path / "analysis.lock"; alock.write_text(json.dumps(dict(blinding=dict(blind_lock_sha256=_sha(lock)))))
    auth = tmp_path / "PI-24.md"; auth.write_text("PI DECISIONS #24: authorized\n")
    return d, notes, lock, alock, auth, offset, phys


def _run(d, notes, lock, alock, auth, extra=(), confirm="eBOSS", auth_sha=None):
    cmd = [sys.executable, SCRIPT, "--chain-dir", str(d), "--root", "real_eboss", "--blind-lock", str(lock), "--analysis-lock", str(alock),
           "--authorization", str(auth), "--authorization-sha256", auth_sha or _sha(auth), "--events-log", str(notes / "artifacts" / "unblind_events.log"),
           "--confirm", confirm, "--code-repo", REPO, "--notes-repo", str(notes)] + list(extra)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=REPO, JAX_PLATFORMS="cpu")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_dry_run_green_writes_nothing_and_hides_offset(tmp_path):
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path)
    before = sorted(os.listdir(d))
    r = _run(d, notes, lock, alock, auth, extra=["--dry-run"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "DRY-RUN OK GREEN"
    assert sorted(os.listdir(d)) == before
    for v in offset.values():
        assert f"{v:.6g}" not in r.stdout + r.stderr and f"{v:.4g}" not in r.stdout + r.stderr


def test_formal_run_exact_inverse_stamp_and_unblind_once(tmp_path):
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path)
    r = _run(d, notes, lock, alock, auth)
    assert r.returncode == 0, r.stderr
    for c in range(2):
        t = np.loadtxt(d / f"real_eboss.unblinded.{c + 1}.txt", ndmin=2)
        assert np.allclose(t[:, 2:], phys[c], rtol=1e-7, atol=0)
    assert not any(f.endswith(".tmp") for f in os.listdir(d))
    stamp = json.load(open(d / "UNBLINDED.stamp"))
    assert stamp["schema"] == "UNBLINDED.stamp.v2" and stamp["health_gate"]["label"] == "GREEN"
    assert stamp["applied_offset"] == {k: float(v) for k, v in offset.items()}
    for want, name in [l.split() for l in (d / "SHA256SUMS").read_text().splitlines()]:
        assert _sha(d / name) == want                      # blinded payload untouched
    ev = (notes / "artifacts" / "unblind_events.log").read_text().strip().splitlines()
    assert len(ev) == 1 and json.loads(ev[0])["health_gate"] == "GREEN"
    r2 = _run(d, notes, lock, alock, auth)
    assert r2.returncode == 3 and "already unblinded" in r2.stderr


def test_amber_proceeds_with_label(tmp_path):
    amber = dict(GREEN, rhat_max=1.015, ess_tail_min=250.0)
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path, health=amber)
    r = _run(d, notes, lock, alock, auth, extra=["--dry-run"])
    assert r.returncode == 0 and r.stdout.strip() == "DRY-RUN OK AMBER", r.stderr


def test_rails_alone_give_amber(tmp_path):
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path, rail=True)
    r = _run(d, notes, lock, alock, auth, extra=["--dry-run"])
    assert r.returncode == 0 and r.stdout.strip() == "DRY-RUN OK AMBER", r.stderr


@pytest.mark.parametrize("bad", [dict(rhat_max=1.03), dict(n_divergent=2), dict(ebfmi_min=0.2), dict(ess_tail_min=150.0), dict(treedepth_sat_frac=0.1)])
def test_red_refuses(tmp_path, bad):
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path, health=dict(GREEN, **bad))
    r = _run(d, notes, lock, alock, auth)
    assert r.returncode == 3 and "health gate RED" in r.stderr
    assert not (d / "UNBLINDED.stamp").exists() and not (d / "real_eboss.unblinded.1.txt").exists()


@pytest.mark.parametrize("tamper", ["confirm", "auth_sha", "lock_sha", "sums", "no_record", "record_mismatch", "untracked", "uncommitted", "corrupt_chain2", "output_exists"])
def test_refusals_write_nothing(tmp_path, tamper):
    kw = {}
    if tamper == "no_record":
        kw["record"] = False
    if tamper == "untracked":
        kw["commit"] = False
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path, **kw)
    rk = {}
    if tamper == "confirm":
        rk["confirm"] = "DESI"
    elif tamper == "auth_sha":
        rk["auth_sha"] = "0" * 64
    elif tamper == "lock_sha":
        alock.write_text(json.dumps(dict(blinding=dict(blind_lock_sha256="1" * 64))))
    elif tamper == "sums":
        p = d / "real_eboss.1.txt"; p.write_text(p.read_text().replace("1", "2", 1)); _git(notes, "commit", "-qam", "tamper")
    elif tamper == "record_mismatch":
        rec = json.load(open(d / "EXECUTION_RECORD.json")); rec["outputs_sha256"]["real_eboss.1.txt"] = "0" * 64
        (d / "EXECUTION_RECORD.json").write_text(json.dumps(rec)); _git(notes, "commit", "-qam", "tamper")
    elif tamper == "uncommitted":
        (d / "real_eboss.health.json").write_text((d / "real_eboss.health.json").read_text() + "\n")
    elif tamper == "corrupt_chain2":
        p = d / "real_eboss.2.txt"; lines = p.read_text().splitlines(); lines[3] = "not a number"; p.write_text("\n".join(lines) + "\n")
        with open(d / "SHA256SUMS", "w") as f:
            for name in sorted(x for x in os.listdir(d) if x not in ("SHA256SUMS", "EXECUTION_RECORD.json")):
                f.write(f"{_sha(d / name)}  {name}\n")
        rec = json.load(open(d / "EXECUTION_RECORD.json")); rec["outputs_sha256"]["real_eboss.2.txt"] = _sha(p)
        (d / "EXECUTION_RECORD.json").write_text(json.dumps(rec)); _git(notes, "add", "-A"); _git(notes, "commit", "-qm", "corrupt")
    elif tamper == "output_exists":
        (d / "real_eboss.unblinded.1.txt").write_text("x")
    before = sorted(os.listdir(d))
    r = _run(d, notes, lock, alock, auth, **rk)
    assert r.returncode == 3, (tamper, r.stdout, r.stderr)
    assert r.stderr.startswith("REFUSE:")
    assert sorted(os.listdir(d)) == before
    assert not (d / "UNBLINDED.stamp").exists()
    for v in offset.values():
        assert f"{v:.6g}" not in r.stdout + r.stderr


def test_june_like_directory_without_record_is_refused(tmp_path):
    d, notes, lock, alock, auth, offset, phys = _build(tmp_path, record=False)
    r = _run(d, notes, lock, alock, auth, extra=["--dry-run"])
    assert r.returncode == 3 and "EXECUTION_RECORD.json missing" in r.stderr
