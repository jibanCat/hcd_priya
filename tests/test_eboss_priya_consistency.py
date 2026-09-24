"""Synthetic tests for scripts/eboss_priya_consistency.py (frozen section-3 rules; exact tau0 inversion)."""
import importlib.util
import json
import os

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"
SCRIPT = os.path.join(REPO, "scripts", "eboss_priya_consistency.py")
Z = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6]
NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"] + \
        [f"tau0_z{i}" for i in range(13)] + ["alpha_lls", "alpha_subdla", "alpha_dla"]


def _mod():
    spec = importlib.util.spec_from_file_location("epc", SCRIPT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _ref(tmp_path):
    ref = dict(reference="test", primary_chain="chain3", chains=dict(
        chain3=dict(n_P=dict(median=0.898, err_plus=0.012, err_minus=0.013), tau0=dict(median=1.221, err_plus=0.021, err_minus=0.012),
                    dtau0=dict(median=-0.270, err_plus=0.029, err_minus=0.029), alpha_q=dict(lower68=2.85),
                    v_scale_h=dict(median=0.695), A_P_1e_9=dict(upper68=1.33)),
        **{"chain1_fiducial_z2.6_4.6": dict(n_P=dict(median=1.009), A_P_1e9=dict(median=1.69), **{"A_P_1e-9": dict(median=1.69)},
                                            tau0=dict(median=1.082), dtau0=dict(median=-0.013))}))
    p = tmp_path / "ref.json"; p.write_text(json.dumps(ref)); return p


def _lock(tmp_path):
    p = tmp_path / "analysis.lock"; p.write_text(json.dumps(dict(legs=dict(eBOSS=dict(z=Z))))); return p


def _chains(tmp_path, ns=0.898, Ap=1.25e-9, tau0_amp=1.22, dtau0=-0.27, n=400, n_chains=2, sig_ns=0.012, seed=1, alphaq=2.45):
    m = _mod(); rng = np.random.default_rng(seed)
    d = tmp_path / "chains"; d.mkdir(exist_ok=True)
    z = np.asarray(Z)
    for c in range(n_chains):
        X = np.empty((n, len(NAMES)))
        X[:, 0] = ns + sig_ns * rng.standard_normal(n)
        X[:, 1] = np.clip(Ap + 0.05e-9 * rng.standard_normal(n), 1.2e-9, None)
        X[:, 2] = 3.8; X[:, 3] = 2.9; X[:, 4] = alphaq + 0.02 * rng.standard_normal(n); X[:, 5] = 0.70; X[:, 6] = 0.143; X[:, 7] = 7.2; X[:, 8] = 0.05
        amp = tau0_amp + 0.01 * rng.standard_normal(n); dt = dtau0 + 0.02 * rng.standard_normal(n)
        X[:, 9:22] = amp[:, None] * ((1 + z)[None, :] / 4.0) ** dt[:, None] * m.kim_tau0(z)[None, :]
        X[:, 22] = 0.17; X[:, 23] = 0.06; X[:, 24] = 0.004
        table = np.column_stack([np.ones(n), np.zeros(n), X])
        np.savetxt(d / f"real_eboss.unblinded.{c + 1}.txt", table, fmt=["%.10g"] * table.shape[1], header="weight  minusloglike  " + "  ".join(NAMES))
    (d / "real_eboss.unblinded.paramnames").write_text("".join(f"{n}\t{n}\n" for n in NAMES))
    (d / "real_eboss.health.json").write_text(json.dumps(dict(rhat_max=1.003, ess_bulk_min=900, ess_tail_min=700, ebfmi_min=0.9, n_divergent=0,
                                                              treedepth_sat_frac=0.0, n_chains=n_chains, n_draws=n, seed=1)))
    return d


def test_tau0_inversion_is_exact():
    m = _mod(); rng = np.random.default_rng(3)
    amp = rng.uniform(0.75, 1.25, 20); dt = rng.uniform(-0.4, 0.25, 20); z = np.asarray(Z)
    ladder = amp[:, None] * ((1 + z)[None, :] / 4.0) ** dt[:, None] * m.kim_tau0(z)[None, :]
    a2, d2 = m.recover_tau0_amp_dtau0(ladder, z)
    np.testing.assert_allclose(a2, amp, rtol=1e-12); np.testing.assert_allclose(d2, dt, atol=1e-12)


@pytest.mark.parametrize("ns,expect", [(0.898, "CONSISTENT"), (0.910, "CONSISTENT"), (0.930, "SHIFTED"), (0.960, "DISCREPANT"), (0.85, "SHIFTED"), (0.84, "DISCREPANT")])
def test_ns_labels_follow_the_frozen_rule(tmp_path, ns, expect):
    m = _mod(); d = _chains(tmp_path, ns=ns)
    out = tmp_path / "out"
    m.main(["--chain-dir", str(d), "--reference", str(_ref(tmp_path)), "--analysis-lock", str(_lock(tmp_path)), "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["tests"]["ns"]["label"] == expect and r["headline_label"] == expect
    if expect != "CONSISTENT":
        assert r["attribution"] is not None and "alphaq" in r["attribution"]["linear_response"]
    else:
        assert r["attribution"] is None


@pytest.mark.parametrize("Ap,expect", [(1.25e-9, "CONSISTENT"), (1.40e-9, "CONSISTENT"), (1.55e-9, "SHIFTED"), (2.0e-9, "DISCREPANT")])
def test_Ap_one_sided_rule(tmp_path, Ap, expect):
    m = _mod(); d = _chains(tmp_path, Ap=Ap)
    out = tmp_path / "out"
    m.main(["--chain-dir", str(d), "--reference", str(_ref(tmp_path)), "--analysis-lock", str(_lock(tmp_path)), "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["tests"]["Ap"]["label"] == expect


def test_secondary_labels_and_outputs(tmp_path):
    m = _mod(); d = _chains(tmp_path, tau0_amp=1.10, dtau0=-0.10)
    out = tmp_path / "out"
    m.main(["--chain-dir", str(d), "--reference", str(_ref(tmp_path)), "--analysis-lock", str(_lock(tmp_path)), "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["tests"]["tau0_amp"]["label"] == "DISCREPANT"      # |delta| 0.12 > 0.10
    assert r["tests"]["dtau0"]["label"] == "SHIFTED"            # |delta| 0.17: > max(1s, 0.06), <= 0.20
    assert abs(r["summaries"]["tau0_amp"]["median"] - 1.10) < 0.01
    md = open(str(out) + ".md").read()
    assert "Preregistered tests" in md and "attribution" in md.lower()
    assert set(r["green"]) == {"rhat", "ess_bulk", "ess_tail", "divergences", "ebfmi"} and all(r["green"].values())
