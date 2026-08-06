"""A3d KS f_res pin-vs-draw paired diagnostic analyzer (PI #9 3f, Option B) — TDD suite.

The analyzer's contract, pre-registered in `2026-08-05-A3d-PREREGISTRATION.md`:
  * REFUSES (PairingError) to print any number unless every conjunct passes;
  * PRIMARY attribution statistic = paired per-mock tau0_amp pull delta (draw - pin), read
    against an EXHAUSTIVE five-cell decision table keyed on the 95% t-CI [lo, hi] with the
    pre-registered materiality threshold M = 0.30;
  * n_s / A_p / dtau0 are SECONDARY transmission checks: reported, never celled;
  * the pin arm is the PRESERVED r6x deployed KS population (r6x pkl schema, meta-identified,
    smoke pkls excluded); the draw arm is the run_prod_sbc_shard schema with the frozen cfg.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_ksfd_paired.py -q
"""
import importlib.util
import os
import pickle

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    path = os.path.join(REPO, "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def A():
    return _load_script("analyze_ksfd_paired.py")


# --------------------------------------------------------------------------------------------
# Synthetic fixtures. 25-entry truth_vec (9 theta + 13 tau0 rungs + 3 alphas), names as the
# deployed KS leg emits them, sites_extra with the 8 shared KS latents + the 2 f_res sites.
# --------------------------------------------------------------------------------------------
NAMES = (["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]
         + [f"tau0_z{i}" for i in range(13)] + ["alpha_lls", "alpha_subdla", "alpha_dla"])
SHARED = ("dla_raw", "dtau0", "eps_lls", "kappa_lls", "m_sub", "t_dla", "t_sub", "tau0_amp")
L_DRAWS = 60


def _site(rng, truth, loc=None, sd=0.02):
    loc = truth if loc is None else loc
    return dict(draws=rng.normal(loc, sd, L_DRAWS), truth=truth)


def _per_mock(m, arm_salt=0, fres_drawn=False, tau0_pull=0.0, ll_true=None):
    """One per-mock record. TRUTHS are seeded by m ONLY (bitwise-shared across arms, as the
    real pairing produces); DRAWS are salted per arm so paired deltas carry realistic noise.
    `tau0_pull` sets the tau0_amp posterior offset in pull units."""
    tv = np.asarray(np.random.default_rng(500 + m).uniform(0.05, 0.95, 25))
    draws = np.random.default_rng(900 + m + 7919 * arm_salt).normal(tv, 0.05, (L_DRAWS, 25))
    se = {}
    rng2 = np.random.default_rng(300 + m + 7919 * arm_salt)
    for i, k in enumerate(SHARED):
        tr = float(np.random.default_rng(40 + m * 31 + i).uniform(0.2, 1.2))
        loc = tr + (tau0_pull * 0.02 if k == "tau0_amp" else 0.0)
        se[k] = dict(draws=rng2.normal(loc, 0.02, L_DRAWS), truth=tr)
    if fres_drawn:
        fa = float(np.random.default_rng(60 + m).normal(0.0, 0.15))
        fs = float(np.random.default_rng(61 + m).normal(0.0, 0.5))
        se["f_res_amp"] = dict(draws=rng2.normal(fa, 0.1, L_DRAWS), truth=fa)
        se["f_res_slope"] = dict(draws=rng2.normal(fs, 0.4, L_DRAWS), truth=fs)
    else:
        se["f_res_amp"] = dict(draws=rng2.normal(0, 0.14, L_DRAWS), truth=float("nan"))
        se["f_res_slope"] = dict(draws=rng2.normal(0, 0.5, L_DRAWS), truth=float("nan"))
    return dict(truth_vec=tv, draws=draws, names=list(NAMES), L=100, n_div=0,
                ll_true=(ll_true if ll_true is not None else -97.0 - m),
                ll_draws=np.random.default_rng(70 + m).normal(-100, 2, L_DRAWS),
                sites_extra=se)


def _write_pin(dirpath, A, n=12, smoke_extra=False, meta_over=None, **pmkw):
    os.makedirs(dirpath, exist_ok=True)
    for m in range(n):
        # FIXTURE FIDELITY (P3 review finding 8): the REAL r6x meta carries arm/leg/seed/NUTS/
        # smoke but NOT `survey` -- that lives at the top level only, so the analyzer's
        # meta-then-top-level fallback path is exercised by every test.
        meta = {k: v for k, v in A.FROZEN_PIN_META.items() if k != "survey"}
        meta.update(meta_over or {})
        d = dict(arm=meta["arm"], leg=meta["leg"], survey=A.FROZEN_PIN_META["survey"],
                 idxs=[m], per_mock=[_per_mock(m, **pmkw)],
                 meta=dict(meta, smoke=meta.get("smoke", False)))
        with open(os.path.join(dirpath, f"r6x_ks_deployed_shard_{m:03d}.pkl"), "wb") as f:
            pickle.dump(d, f)
    if smoke_extra:
        d = dict(arm="deployed", leg="KS", survey="KS", idxs=[0],
                 per_mock=[_per_mock(0, ll_true=-1.0)], meta=dict(A.FROZEN_PIN_META, smoke=True))
        with open(os.path.join(dirpath, "r6x_ks_deployed_shard_000.smoke.pkl"), "wb") as f:
            pickle.dump(d, f)


def _write_draw(dirpath, A, n=12, cfg_over=None, dll=+37.0, **pmkw):
    os.makedirs(dirpath, exist_ok=True)
    for m in range(n):
        rec = _per_mock(m, arm_salt=1, fres_drawn=True, ll_true=-97.0 - m + dll, **pmkw)
        rec["run_cfg"] = dict(A.FROZEN_DRAW_CFG, **(cfg_over or {}))
        rec["truth_site_semantics"] = dict(not_self_drawn=[])
        with open(os.path.join(dirpath, f"mock_{m:04d}.pkl"), "wb") as f:
            pickle.dump(rec, f)


def _happy(tmp_path, A, tau0_pull_draw=0.0, tau0_pull_pin=0.0):
    pin = str(tmp_path / "pin"); draw = str(tmp_path / "draw")
    _write_pin(pin, A, tau0_pull=tau0_pull_pin)
    _write_draw(draw, A, tau0_pull=tau0_pull_draw)
    return draw, pin


# ------------------------------------ decision table -----------------------------------------
def test_attribution_cell_exhaustive(A):
    M = 0.30
    assert A.attribution_cell(-0.5, -0.1, M) == "UNEXPECTED-DIRECTION"
    assert A.attribution_cell(-0.1, +0.1, M) == "NULL"
    assert A.attribution_cell(-0.1, +0.5, M) == "UNDER-RESOLVED"
    assert A.attribution_cell(+0.05, +0.2, M) == "NON-NULL-IMMATERIAL"
    assert A.attribution_cell(+0.1, +0.9, M) == "NON-NULL-MATERIAL"


def test_attribution_cell_boundaries(A):
    """Tie conventions, fixed in advance: lo <= 0 counts as zero-inclusion; hi >= M counts as
    materiality NOT excluded; hi < 0 strictly for the unexpected direction."""
    M = 0.30
    assert A.attribution_cell(0.0, 0.2, M) == "NULL"                 # lo == 0 -> zero included
    assert A.attribution_cell(-0.1, 0.30, M) == "UNDER-RESOLVED"    # hi == M -> not excluded
    assert A.attribution_cell(1e-12, 0.30, M) == "NON-NULL-MATERIAL"
    assert A.attribution_cell(-0.3, 0.0, M) == "NULL"                # hi == 0 -> not unexpected
    for lo, hi in [(-1, -0.01), (-1, 0.1), (-1, 1), (0.01, 0.1), (0.01, 1)]:
        assert A.attribution_cell(lo, hi, M) in A.CELLS


def test_cell_requires_ordered_ci(A):
    with pytest.raises(ValueError):
        A.attribution_cell(0.5, 0.1, 0.30)


# ------------------------------------ happy path ---------------------------------------------
def test_happy_path_returns_stats_and_cell(A, tmp_path):
    draw, pin = _happy(tmp_path, A)
    out = A.run(draw, pin, expect_n=12)
    assert out["n_pairs"] == 12
    assert set(out["paired"]) >= {"tau0amp", "ns", "Ap", "dtau0"}
    t0 = out["paired"]["tau0amp"]
    for k in ("mean", "sd", "sem", "t", "ci95", "rho", "wilcoxon_p", "n_negative"):
        assert k in t0
    assert out["cell"] in A.CELLS
    assert out["primary"] == "tau0amp"
    # secondary channels never carry a cell
    assert "cell" not in out["paired"]["ns"]
    # the full-explanation reference is recomputed from the pin arm, not hard-coded
    assert abs(out["full_explanation_ref"] - (-np.mean(out["pin_pulls"]["tau0amp"]))) < 1e-12
    # draw-arm f_res sector stats present (the intervention measurement)
    assert set(out["fres_draw"]) == {"f_res_amp", "f_res_slope"}
    for s in out["fres_draw"].values():
        for k in ("pull_mean", "pull_sd", "rank_ks_p", "scatter", "scatter_expected"):
            assert k in s


def test_detects_seeded_positive_delta(A, tmp_path):
    """Draw arm offset +0.6 pull units vs pin at 0 -> a decisively positive paired delta."""
    draw, pin = _happy(tmp_path, A, tau0_pull_draw=+0.6, tau0_pull_pin=0.0)
    out = A.run(draw, pin, expect_n=12)
    assert out["paired"]["tau0amp"]["mean"] > 0.3
    assert out["cell"] in ("NON-NULL-MATERIAL", "NON-NULL-IMMATERIAL")


# ------------------------------------ conjunct refusals --------------------------------------
def test_refuses_missing_draw_pkl(A, tmp_path):
    draw, pin = _happy(tmp_path, A)
    os.remove(os.path.join(draw, "mock_0007.pkl"))
    with pytest.raises(A.PairingError, match="C1"):
        A.run(draw, pin, expect_n=12)


def test_refuses_truth_vec_mismatch(A, tmp_path):
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0003.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["truth_vec"] = np.asarray(rec["truth_vec"]).copy()
    rec["truth_vec"][0] += 1e-9
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C2"):
        A.run(draw, pin, expect_n=12)


def test_refuses_shared_site_truth_mismatch(A, tmp_path):
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0002.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["sites_extra"]["eps_lls"]["truth"] += 1e-9
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C2"):
        A.run(draw, pin, expect_n=12)


def test_refuses_pin_fres_not_pinned(A, tmp_path):
    """A pin-side pkl with a FINITE f_res truth is not the pinned population."""
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(pin, "r6x_ks_deployed_shard_004.pkl")
    d = pickle.load(open(p, "rb"))
    d["per_mock"][0]["sites_extra"]["f_res_amp"]["truth"] = 0.05
    pickle.dump(d, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C3"):
        A.run(draw, pin, expect_n=12)


def test_refuses_draw_fres_pinned(A, tmp_path):
    """A draw-side pkl with a NaN f_res truth means the self-draw did not happen."""
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0005.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["sites_extra"]["f_res_amp"]["truth"] = float("nan")
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C4"):
        A.run(draw, pin, expect_n=12)


def test_refuses_draw_fres_repeated_truth(A, tmp_path):
    """Across-mock DISTINCTNESS: a repeated f_res truth (the decoy A1c amendment 5 closed)."""
    draw, pin = _happy(tmp_path, A)
    p5, p6 = (os.path.join(draw, f"mock_{m:04d}.pkl") for m in (5, 6))
    r5, r6 = pickle.load(open(p5, "rb")), pickle.load(open(p6, "rb"))
    r6["sites_extra"]["f_res_amp"]["truth"] = r5["sites_extra"]["f_res_amp"]["truth"]
    pickle.dump(r6, open(p6, "wb"))
    with pytest.raises(A.PairingError, match="C4"):
        A.run(draw, pin, expect_n=12)


def test_refuses_lawbreaking_fres_truths(A, tmp_path):
    """f_res truths must be consistent with the DEPLOYED law Normal(0, 0.15): a cluster of
    +5-sigma truths refuses at alpha 1e-3."""
    draw, pin = _happy(tmp_path, A)
    for m in range(12):
        p = os.path.join(draw, f"mock_{m:04d}.pkl")
        rec = pickle.load(open(p, "rb"))
        rec["sites_extra"]["f_res_amp"]["truth"] = 0.75 + 0.001 * m   # ~5 sigma, one-sided
        pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C4"):
        A.run(draw, pin, expect_n=12)


def test_refuses_wrong_draw_cfg(A, tmp_path):
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0001.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["run_cfg"] = dict(rec["run_cfg"], seed=20260614)     # the A3c seed: wrong population
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C5"):
        A.run(draw, pin, expect_n=12)


def test_refuses_missing_sampler_stamp(A, tmp_path):
    """A draw pkl WITHOUT the sampler stamps is not the pre-registered A3d population."""
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0009.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["run_cfg"].pop("seed")
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C5"):
        A.run(draw, pin, expect_n=12)


def test_refuses_pin_meta_mismatch(A, tmp_path):
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(pin, "r6x_ks_deployed_shard_008.pkl")
    d = pickle.load(open(p, "rb"))
    d["meta"]["n_samples"] = 600                              # not the r6x population
    pickle.dump(d, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C6"):
        A.run(draw, pin, expect_n=12)


def test_smoke_pkl_is_excluded_not_loaded(A, tmp_path):
    """The 2026-08-05 erratum class: a *.smoke.pkl in the pin dir must be IGNORED by the glob,
    and the run must still succeed on the 12 production pkls."""
    pin = str(tmp_path / "pin"); draw = str(tmp_path / "draw")
    _write_pin(pin, A, smoke_extra=True)
    _write_draw(draw, A)
    out = A.run(draw, pin, expect_n=12)
    assert out["n_pairs"] == 12


def test_refuses_smoke_meta_on_production_name(A, tmp_path):
    """meta.smoke=True on a production-named pin pkl -> refuse (C6), never silently include."""
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(pin, "r6x_ks_deployed_shard_002.pkl")
    d = pickle.load(open(p, "rb"))
    d["meta"]["smoke"] = True
    pickle.dump(d, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C6"):
        A.run(draw, pin, expect_n=12)


def test_refuses_no_propagation(A, tmp_path):
    """ll_true EQUAL pin-vs-draw on any mock = the f_res perturbation did not reach the data."""
    pin = str(tmp_path / "pin"); draw = str(tmp_path / "draw")
    _write_pin(pin, A)
    _write_draw(draw, A, dll=0.0)                             # identical ll_true
    with pytest.raises(A.PairingError, match="C7"):
        A.run(draw, pin, expect_n=12)


def test_refuses_duplicate_pin_census(A, tmp_path):
    """P3 review finding 1: a 13th same-config production-named pkl in the pin dir must refuse
    (census check), never silently last-glob-win over the preserved chain."""
    draw, pin = _happy(tmp_path, A)
    src = os.path.join(pin, "r6x_ks_deployed_shard_003.pkl")
    dup = os.path.join(pin, "r6x_ks_deployed_shard_012.pkl")   # matches the glob, wrong census
    with open(src, "rb") as f:
        d = pickle.load(f)
    with open(dup, "wb") as f:
        pickle.dump(d, f)
    with pytest.raises(A.PairingError, match="C1"):
        A.run(draw, pin, expect_n=12)


def test_refuses_draw_not_self_drawn_provenance(A, tmp_path):
    """P3 review finding 2a (C4b): the runner's own truth_site_semantics must agree nothing was
    left un-drawn; a non-empty not_self_drawn refuses."""
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0004.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["truth_site_semantics"] = dict(not_self_drawn=["f_res_amp"])
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(A.PairingError, match="C4"):
        A.run(draw, pin, expect_n=12)


def test_health_reported_not_gated(A, tmp_path):
    """n_div > 0 is DISCLOSED, never an exclusion: the pair count stays 12."""
    draw, pin = _happy(tmp_path, A)
    p = os.path.join(draw, "mock_0010.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["n_div"] = 3
    pickle.dump(rec, open(p, "wb"))
    out = A.run(draw, pin, expect_n=12)
    assert out["n_pairs"] == 12
    assert out["health"]["draw_n_div_total"] == 3
    assert out["health"]["flags"], "a nonzero n_div must raise a visible health flag"
