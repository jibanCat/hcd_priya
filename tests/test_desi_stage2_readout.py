"""Synthetic tests for scripts/desi_stage2_readout.py: runner-format pkls/JSONs built from known posteriors."""
import importlib.util, json, os, pickle
import numpy as np, pytest
S = importlib.util.spec_from_file_location("s2r", "/home/mfho/hcd_priya/scripts/desi_stage2_readout.py"); M = importlib.util.module_from_spec(S); S.loader.exec_module(M)
NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"] + [f"tau0_z{i}" for i in range(13)] + ["alpha_lls", "alpha_subdla", "alpha_dla"]


def _mock(tmp, m, rng, corr_stored=-0.2, corr_strong=-0.2, shift=0.0, L=120, step=5):
    """stored: L draws from N(mu, Sigma_stored); strong: 4 x 600 from N(mu + shift*sd, Sigma_strong); k in linear units in files."""
    sd = np.array([0.05, 0.03, 0.06, 0.25])
    def mk(n, c, sh):
        C = np.eye(4); C[0, 2] = C[2, 0] = c; R = np.linalg.cholesky(np.outer(sd, sd) * C)
        mu = np.array([0.4, 0.95, -0.1, -1.6]) + sh * sd * np.array([1, 0, 0, 0])
        return mu + (R @ rng.standard_normal((4, n))).T
    Xs = mk(L, corr_stored, 0.0)
    dr = rng.uniform(0.2, 0.8, size=(L, 25)); dr[:, 0] = Xs[:, 0]
    stored = dict(names=NAMES, draws=dr, truth_vec=rng.uniform(0.3, 0.7, 25), L=L, n_div=0, run_cfg={}, kept_global=np.ones(13, bool), dropped={"DESI": []}, ll_true=-1.0,
                  truth_alpha_hcd_z=np.zeros((13, 3)), sites_extra=dict(tau0_amp=dict(draws=Xs[:, 1], truth=0.95), dtau0=dict(draws=Xs[:, 2], truth=-0.1), k_SiIII_DESI_z1=dict(draws=10 ** Xs[:, 3], truth=10 ** -1.6)))
    os.makedirs(tmp / "stored", exist_ok=True); pickle.dump(stored, open(tmp / "stored" / f"mock_{m:04d}.pkl", "wb"))
    chains = []
    for c in range(4):
        X = mk(600, corr_strong, shift); d = rng.uniform(0.2, 0.8, size=(600, 25)); d[:, 0] = X[:, 0]
        chains.append(dict(chain=c, draws=d, samples=dict(tau0_amp=X[:, 1], dtau0=X[:, 2], k_SiIII_DESI_z1=10 ** X[:, 3]), n_div=0))
    raw = dict(mock=m, names=NAMES, strong=dict(chains=chains, names=NAMES, per_chain_div=[0, 0, 0, 0]))
    os.makedirs(tmp / "s2", exist_ok=True); pickle.dump(raw, open(tmp / "s2" / f"stage2_mock_{m:04d}.pkl", "wb"))
    json.dump(dict(mock=m, replica=dict(step=step, compare=dict(bit_identical=False)), strong=dict(battery=dict(rhat_max=1.003, ess_bulk_min=900, ess_tail_min=800, ebfmi_min=0.9, treedepth_sat_frac=0.0, n_divergent=0), gate=dict(pilot_gate_passed=True)), identity={}),
              open(tmp / "s2" / f"stage2_mock_{m:04d}.json", "w"))


@pytest.mark.parametrize("scenario,expect", [("mb", "M-B-orientation"), ("ma_corr", "M-A"), ("ma_shift", "M-A"), ("ambig", "AMBIGUOUS")])
def test_decision_rule_on_planted_scenarios(tmp_path, scenario, expect):
    rng = np.random.default_rng(1); M.B_FINITE_L = 400
    for m in list(M.TAIL) + list(M.CONTROL):
        tail = m in M.TAIL
        if scenario == "mb":
            _mock(tmp_path, m, rng, corr_stored=-0.3, corr_strong=-0.3)
        elif scenario == "ma_corr":
            _mock(tmp_path, m, rng, corr_stored=(-0.6 if tail else -0.3), corr_strong=(0.3 if tail else -0.3))
        elif scenario == "ma_shift":
            _mock(tmp_path, m, rng, corr_stored=-0.3, corr_strong=-0.3, shift=(2.5 if tail else 0.0))
        else:
            _mock(tmp_path, m, rng, corr_stored=0.1, corr_strong=0.1)     # agree but no negative coupling -> ambiguous
    out = tmp_path / "readout"
    M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "s2"), "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["decision"]["label"].startswith(expect), r["decision"]
    assert len(r["per_mock"]) == 8 and all("finite_L" in m and "stats" in m for m in r["per_mock"])
    with pytest.raises(SystemExit):
        M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "s2"), "--out", str(out)])   # exactly-once


def test_incomplete_set_is_labelled_incomplete(tmp_path):
    rng = np.random.default_rng(2); M.B_FINITE_L = 200
    for m in (45, 9):
        _mock(tmp_path, m, rng)
    out = tmp_path / "r"
    M.main(["--stored-dir", str(tmp_path / "stored"), "--stage2-dir", str(tmp_path / "s2"), "--out", str(out), "--mocks", "45", "9"])
    assert json.load(open(str(out) + ".json"))["decision"]["label"] == "INCOMPLETE"
