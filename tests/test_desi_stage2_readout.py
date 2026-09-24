"""Synthetic tests for scripts/desi_stage2_readout.py v2: runner-format pkls/JSONs built from known posteriors with AR(1)
autocorrelated chains (the stored draws are an independent 600-draw chain thinned by the stored step, as in production)."""
import importlib.util, json, os, pickle
import numpy as np, pytest
S = importlib.util.spec_from_file_location("s2r", "/home/mfho/hcd_priya/scripts/desi_stage2_readout.py"); M = importlib.util.module_from_spec(S); S.loader.exec_module(M)
NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"] + [f"tau0_z{i}" for i in range(13)] + ["alpha_lls", "alpha_subdla", "alpha_dla"]
SD = np.array([0.05, 0.03, 0.06, 0.25]); MU = np.array([0.4, 0.95, -0.1, -1.6])


def _ar1(rng, n, corr, shift=0.0, phi=0.6):
    C = np.eye(4); C[0, 2] = C[2, 0] = corr; R = np.linalg.cholesky(np.outer(SD, SD) * C)
    e = rng.standard_normal((n, 4)); z = np.empty((n, 4)); z[0] = e[0]
    for i in range(1, n):
        z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * e[i]
    return MU + shift * SD * np.array([1, 0, 0, 0]) + z @ R.T


def _mock(tmp, m, rng, corr_stored=-0.2, corr_strong=-0.2, shift=0.0, step=5, sampler_ok=True, phi=0.6):
    L = int(np.ceil(600 / step))
    Xs = _ar1(rng, 600, corr_stored, 0.0, phi)[::step]
    dr = rng.uniform(0.2, 0.8, size=(L, 25)); dr[:, 0] = Xs[:, 0]
    stored = dict(names=NAMES, draws=dr, truth_vec=rng.uniform(0.3, 0.7, 25), L=L, n_div=0, run_cfg={}, kept_global=np.ones(13, bool), dropped={"DESI": []}, ll_true=-1.0,
                  truth_alpha_hcd_z=np.zeros((13, 3)), sites_extra=dict(tau0_amp=dict(draws=Xs[:, 1], truth=0.95), dtau0=dict(draws=Xs[:, 2], truth=-0.1), k_SiIII_DESI_z1=dict(draws=10 ** Xs[:, 3], truth=10 ** -1.6)))
    os.makedirs(tmp / "stored", exist_ok=True); pickle.dump(stored, open(tmp / "stored" / f"mock_{m:04d}.pkl", "wb"))
    chains = []
    for c in range(4):
        X = _ar1(rng, 600, corr_strong, shift, phi); d = rng.uniform(0.2, 0.8, size=(600, 25)); d[:, 0] = X[:, 0]
        chains.append(dict(chain=c, draws=d, samples=dict(tau0_amp=X[:, 1], dtau0=X[:, 2], k_SiIII_DESI_z1=10 ** X[:, 3]), n_div=0))
    raw = dict(mock=m, names=NAMES, strong=dict(chains=chains, names=NAMES, per_chain_div=[0, 0, 0, 0]))
    os.makedirs(tmp / "s2", exist_ok=True); pickle.dump(raw, open(tmp / "s2" / f"stage2_mock_{m:04d}.pkl", "wb"))
    json.dump(dict(mock=m, replica=dict(step=step, compare=dict(bit_identical=False)),
                   strong=dict(battery=dict(rhat_max=1.003, ess_bulk_min=900, ess_tail_min=800, ebfmi_min=0.9, treedepth_sat_frac=0.0, n_divergent=0),
                               gate=dict(pilot_gate_passed=sampler_ok, passed_sampler_criteria=sampler_ok)), identity=dict(truth_vec_equal=True, ll_true_equal=True)),
              open(tmp / "s2" / f"stage2_mock_{m:04d}.json", "w"))


def test_stored_step_is_unique_or_refuses():
    assert M.stored_step(120) == 5 and M.stored_step(150) == 4 and M.stored_step(300) == 2 and M.stored_step(200) == 3
    with pytest.raises(SystemExit):
        M.stored_step(601)


@pytest.mark.parametrize("scenario,expect", [("mb", "M-B-orientation"), ("ma_corr", "M-A"), ("ma_shift", "M-A"), ("ambig", "AMBIGUOUS")])
def test_decision_rule_on_planted_scenarios(tmp_path, scenario, expect):
    rng = np.random.default_rng(1); M.B_FINITE_L = 300
    for m in list(M.TAIL) + list(M.CONTROL):
        tail = m in M.TAIL
        if scenario == "mb":
            _mock(tmp_path, m, rng, corr_stored=-0.3, corr_strong=-0.3)
        elif scenario == "ma_corr":
            _mock(tmp_path, m, rng, corr_stored=(-0.7 if tail else -0.3), corr_strong=(0.3 if tail else -0.3))
        elif scenario == "ma_shift":
            _mock(tmp_path, m, rng, corr_stored=-0.3, corr_strong=-0.3, shift=(2.5 if tail else 0.0))
        else:
            _mock(tmp_path, m, rng, corr_stored=0.1, corr_strong=0.1)
    out = tmp_path / "readout"
    M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "s2"), "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["decision"]["label"].startswith(expect), r["decision"]
    assert len(r["per_mock"]) == 8 and all("finite_L" in m and "sd_ratio_band95" in m["finite_L"] for m in r["per_mock"])
    assert (out.with_suffix(".md").exists() or (tmp_path / "readout.md").exists())
    with pytest.raises(SystemExit):
        M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "s2"), "--out", str(out)])   # exactly-once


def test_failed_sampler_gate_makes_incomplete_and_empty_set_is_incomplete(tmp_path):
    rng = np.random.default_rng(2); M.B_FINITE_L = 200
    for m in list(M.TAIL) + list(M.CONTROL):
        _mock(tmp_path, m, rng, sampler_ok=(m != 9))
    out = tmp_path / "r"
    M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "s2"), "--out", str(out)])
    d = json.load(open(str(out) + ".json"))["decision"]; assert d["label"] == "INCOMPLETE" and d["excluded_failed_sampler_gate"] == [9]
    out2 = tmp_path / "r2"
    M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "empty"), "--out", str(out2)])
    assert json.load(open(str(out2) + ".json"))["decision"]["label"] == "INCOMPLETE"


def test_null_calibration_of_the_corr_band_with_autocorrelated_chains(tmp_path):
    """Same posterior on both sides (AR(1) chains): the stored corr should fall outside the block-bootstrap 95 percent band
    about 5 percent of the time (accept up to 0.12 at this sample size)."""
    M.B_FINITE_L = 300; out = 0; n = 0
    for pop in range(40):
        for step in (2, 5):
            rng = np.random.default_rng(1000 + pop * 10 + step)
            d = tmp_path / f"p{pop}_{step}"; _mock(d, 45, rng, corr_stored=-0.2, corr_strong=-0.2, step=step)
            res = M.analyze_mock(str(d / "stored" / "mock_0045.pkl"), str(d / "s2" / "stage2_mock_0045.pkl"), str(d / "s2" / "stage2_mock_0045.json"), 7)
            out += int(not res["corr_inside_band"]); n += 1
    rate = out / n
    assert rate <= 0.12, rate
