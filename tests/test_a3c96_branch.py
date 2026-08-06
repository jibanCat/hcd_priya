"""Tests for scripts/a3c96_branch.py (A3c N=96 adjudication branch classification).

The 512-combo sweep mirrors the firstarm oracle-sweep convention: an INDEPENDENTLY coded
oracle (rows + branch precedence re-implemented here from the pre-registration text, not
from the module under test) checked against the module over the full factor product.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    path = os.path.join(REPO, "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def BR():
    return _load_script("a3c96_branch.py")


N = 96


def _gate_leg(ns=(0.05, 1.02, "UNIFORM"), Ap=(0.06, 1.05, "UNIFORM"),
              tau0=(-0.05, 1.01, "UNIFORM"), n=N, survey="KS"):
    def ch(mean, std, verdict):
        return dict(pull=dict(mean=mean, std=std, sem=std / np.sqrt(n), n=n),
                    rank=dict(ks_p=(0.5 if verdict == "UNIFORM" else 1e-5),
                              verdict=verdict, n=n, n_outside_band=0, mean=0.5))
    legs = dict(
        n_mocks=n,
        gate_ns=("PASS" if abs(ns[0]) <= 0.30 and ns[1] <= 1.1 else "FAIL"),
        gate_Ap=("PASS" if abs(Ap[0]) <= 0.30 and Ap[1] <= 1.1 else "FAIL"),
        gate_rank_ns=ns[2], gate_rank_Ap=Ap[2],
        pulls={"ns": ch(*ns)["pull"], "Ap": ch(*Ap)["pull"], "tau0amp": ch(*tau0)["pull"]},
        rank_uniformity={"ns": ch(*ns)["rank"], "Ap": ch(*Ap)["rank"],
                         "tau0amp": ch(*tau0)["rank"]},
        prior=dict(survey=survey),
        repaired_sectors=dict(gated=False, sites={}),
    )
    return dict(legs={survey: legs})


def _sd_json(survey="KS", n=N, conj=True, health_ok=True, n_div=0):
    health = {
        "f_res_amp": dict(rank_ks_p=(0.4 if health_ok else 1e-5), healthy=health_ok,
                          scatter=0.04, scatter_expected=0.005),
        "f_res_slope": dict(rank_ks_p=0.6, healthy=True, scatter=0.08,
                            scatter_expected=0.1),
    }
    return dict(survey=survey, n=n, conjuncts_ok=conj, health=health,
                health_ok=all(h["healthy"] for h in health.values()),
                n_div_total=n_div, L_median=150.0, L_range=[34, 300])


@pytest.fixture(scope="module")
def intact(tmp_path_factory):
    """An OUTDIR with the exact 96-mock census + a matching tranche-1 sha256 inventory."""
    import hashlib
    d = tmp_path_factory.mktemp("outdir")
    lines = []
    for m in range(N):
        p = d / f"mock_{m:04d}.pkl"
        p.write_bytes(b"payload-%04d" % m)
        if m < 48:
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest()
                         + f"  mock_{m:04d}.pkl")
    sha = d / "t1.sha256"
    sha.write_text("\n".join(lines) + "\n")
    return str(d), str(sha)


def _classify(BR, tmp_path, gate, sd, intact):
    outdir, sha = intact
    g = tmp_path / "gate.json"
    s = tmp_path / "sd.json"
    g.write_text(json.dumps(gate))
    s.write_text(json.dumps(sd))
    return BR.run(str(g), str(s), sha, outdir)


# ------------------------------- the named branch cases -------------------------------

def test_branch_A_all_healthy(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(), intact)
    assert out["branch"] == "A" and out["disposition"]["row"] == 1


def test_branch_B_ns_sd_only(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(ns=(0.05, 1.15, "UNIFORM")), _sd_json(), intact)
    assert out["branch"] == "B" and out["disposition"]["row"] == 6


def test_branch_C_Ap_sd_only_is_new_failure_mode(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(Ap=(0.06, 1.2, "UNIFORM")), _sd_json(), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 6


def test_branch_C_both_sd_only(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(ns=(0.05, 1.15, "UNIFORM"),
                                            Ap=(0.06, 1.2, "UNIFORM")), _sd_json(), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 6


def test_branch_C_mean_fail(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(ns=(0.5, 1.02, "UNIFORM")), _sd_json(), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 4


def test_branch_C_rank_fail(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(Ap=(0.06, 1.05, "NON-UNIFORM")), _sd_json(), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 3


def test_branch_C_tau0_limb(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(tau0=(0.5, 1.01, "UNIFORM")), _sd_json(), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 3


def test_branch_C_health_fail(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(health_ok=False), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 2


def test_divergence_lifts_A_to_C(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(n_div=1), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 1


def test_divergence_lifts_B_to_C(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(ns=(0.05, 1.15, "UNIFORM")),
                    _sd_json(n_div=2), intact)
    assert out["branch"] == "C" and out["disposition"]["row"] == 6


# ------------------------------------ branch D ----------------------------------------

def test_D_disposition_refusal(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(conj=False), intact)
    assert out["branch"] == "D" and out["disposition"] is None
    assert any("refusal" in r for r in out["reasons"])


def test_D_wrong_population_n48(BR, tmp_path, intact):
    out = _classify(BR, tmp_path, _gate_leg(n=48), _sd_json(n=48), intact)
    assert out["branch"] == "D"
    assert any("N=48 readout" in r for r in out["reasons"])


def test_D_tranche1_sha_mismatch(BR, tmp_path, tmp_path_factory):
    import hashlib
    d = tmp_path_factory.mktemp("outdir_bad")
    lines = []
    for m in range(N):
        p = d / f"mock_{m:04d}.pkl"
        p.write_bytes(b"payload-%04d" % m)
        if m < 48:
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest()
                         + f"  mock_{m:04d}.pkl")
    sha = d / "t1.sha256"
    sha.write_text("\n".join(lines) + "\n")
    (d / "mock_0007.pkl").write_bytes(b"REGENERATED")  # tranche-1 member changed
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(), (str(d), str(sha)))
    assert out["branch"] == "D"
    assert any("CHANGED" in r for r in out["reasons"])


def test_D_missing_and_stray_and_smoke(BR, tmp_path, tmp_path_factory, intact):
    _, sha = intact
    d = tmp_path_factory.mktemp("outdir_census")
    for m in range(N - 1):                       # mock_0095 missing
        (d / f"mock_{m:04d}.pkl").write_bytes(b"x")
    (d / "mock_0099.pkl").write_bytes(b"stray")
    (d / "mock_0000.smoke.pkl").write_bytes(b"smoke")
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(), (str(d), sha))
    assert out["branch"] == "D"
    joined = " ".join(out["reasons"])
    assert "smoke" in joined and "mock_0095.pkl" in joined and "mock_0099.pkl" in joined


def test_D_sha_inventory_wrong_names(BR, tmp_path, intact, tmp_path_factory):
    outdir, _ = intact
    bad = tmp_path_factory.mktemp("sha") / "bad.sha256"
    bad.write_text("0" * 64 + "  mock_0050.pkl\n")   # not a tranche-1 name
    out = _classify(BR, tmp_path, _gate_leg(), _sd_json(), (outdir, str(bad)))
    assert out["branch"] == "D"


# ------------------------- the 512-combo independent-oracle sweep ---------------------

def _oracle(ns_mean_ok, ns_sd_ok, ap_mean_ok, ap_sd_ok, ns_rank_ok, ap_rank_ok,
            tau0_esc, health_ok, n_div):
    """Rows + branches re-derived from the pre-registration text, independently."""
    mean_fail = (not ns_mean_ok) or (not ap_mean_ok)
    rank_fail = (not ns_rank_ok) or (not ap_rank_ok)
    sd_only = [ch for ch, m_ok, s_ok in (("ns", ns_mean_ok, ns_sd_ok),
                                         ("Ap", ap_mean_ok, ap_sd_ok)) if m_ok and not s_ok]
    if mean_fail:
        row = 4
    elif rank_fail:
        row = 3
    elif tau0_esc:
        row = 3
    elif sd_only:
        row = 6
    elif not health_ok:
        row = 2
    else:
        row = 1
    if row in (2, 3, 4):
        branch = "C"
    elif row == 6:
        branch = "B" if sd_only == ["ns"] else "C"
    else:
        branch = "A"
    if n_div > 0 and branch in ("A", "B"):
        branch = "C"
    return row, branch


def test_512_combo_oracle_sweep(BR, tmp_path, intact):
    combos = 0
    for ns_mean_ok in (True, False):
        for ns_sd_ok in (True, False):
            for ap_mean_ok in (True, False):
                for ap_sd_ok in (True, False):
                    for ns_rank in ("UNIFORM", "NON-UNIFORM"):
                        for ap_rank in ("UNIFORM", "NON-UNIFORM"):
                            for tau0_esc in (False, True):
                                for health_ok in (True, False):
                                    for n_div in (0, 3):
                                        gate = _gate_leg(
                                            ns=(0.05 if ns_mean_ok else 0.5,
                                                1.02 if ns_sd_ok else 1.2, ns_rank),
                                            Ap=(0.06 if ap_mean_ok else -0.6,
                                                1.05 if ap_sd_ok else 1.3, ap_rank),
                                            tau0=(0.5 if tau0_esc else -0.05, 1.01,
                                                  "UNIFORM"))
                                        sd = _sd_json(health_ok=health_ok, n_div=n_div)
                                        out = _classify(BR, tmp_path, gate, sd, intact)
                                        row, branch = _oracle(
                                            ns_mean_ok, ns_sd_ok, ap_mean_ok, ap_sd_ok,
                                            ns_rank == "UNIFORM", ap_rank == "UNIFORM",
                                            tau0_esc, health_ok, n_div)
                                        assert out["disposition"]["row"] == row, (
                                            f"row mismatch at combo {combos}")
                                        assert out["branch"] == branch, (
                                            f"branch mismatch at combo {combos}: "
                                            f"got {out['branch']}, oracle {branch}")
                                        combos += 1
    assert combos == 512
