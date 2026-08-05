"""THE FINITE-L CORRECTED NULL (PI ruling 3d.1, 2026-08-05).

The pull divides by the SAMPLE sd of L ESS-thinned draws, so under an exact null the pull is
sqrt(1 + 1/L) times a Student t with L-1 dof, and its population sd is

    sqrt[ (1 + 1/L) * (L-1) / (L-3) ]   --   NOT 1.

Over A1's realized L (median 150, range 32-300) that is 1.0125, which moves the healthy-arm
joint-gate pass probability 0.815 -> 0.782 and P(realized sd > 1.0332) 0.349 -> 0.395. The PI
adopted this corrected null as the INTERPRETATION REFERENCE, evaluated on the landed arm's OWN
L distribution (never inherited from A1), with the explicit guard that the ~1.3% correction
must NOT excuse or downgrade a genuine sd > 1.1 failure. The frozen gate is UNCHANGED.

These tests pin the closed form to values verified against simulation, the arm-level
aggregation rule (root-mean-VARIANCE, not mean of sds), and the analyzer wiring end-to-end.

Run: /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_finite_l_null.py -q
"""
import ast
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/mfho/hcd_priya")

runner = importlib.import_module("scripts.run_prod_sbc_shard")

_ANALYZER_NS = None


def _analyzer():
    """Load only the analyzer's definitions (it is a SCRIPT whose body requires ROOT/PREFIX)."""
    global _ANALYZER_NS
    if _ANALYZER_NS is not None:
        return _ANALYZER_NS
    path = "/home/mfho/hcd_priya/scripts/analyze_sbc_perleg.py"
    tree = ast.parse(open(path).read(), filename=path)
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {"__name__": "analyze_sbc_perleg_defs"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    _ANALYZER_NS = type("NS", (), ns)
    return _ANALYZER_NS


# ------------------------------------------------------------------- the closed form

def test_null_sd_matches_the_simulation_verified_values():
    """The four reference values from the 2026-07-29 record, each verified against simulation
    to 3 dp before being recorded."""
    an = _analyzer()
    for L, want in ((300, 1.0050), (150, 1.0101), (99, 1.0155), (32, 1.0499)):
        assert an.finite_L_null_sd([L]) == pytest.approx(want, abs=2e-4), L


def test_null_sd_aggregates_variances_not_sds():
    """Arm-level: sqrt(mean over mocks of the per-mock null VARIANCE). Averaging sds instead
    understates the aggregate whenever L is heterogeneous."""
    an = _analyzer()
    # var(4) = (1+1/4)*(3/1) = 3.75 ; var(300) = 1.010090 ; sqrt(mean) = 1.5427
    assert an.finite_L_null_sd([4, 300]) == pytest.approx(1.5427, abs=2e-4)
    # and NOT the mean of sds, (1.9365 + 1.0050)/2 = 1.4708
    assert an.finite_L_null_sd([4, 300]) != pytest.approx(1.4708, abs=5e-3)


def test_null_sd_refuses_L_at_or_below_3():
    """The variance is undefined at L <= 3. Every real arm has L >= 32; an L <= 3 mock is
    itself a pathology and must fail loud, not average quietly."""
    an = _analyzer()
    with pytest.raises(ValueError):
        an.finite_L_null_sd([3, 150])


# ------------------------------------------------------------------- healthy-arm context

def test_healthy_context_reproduces_the_recorded_probabilities():
    """Closed form (normal mean x chi-square sd, independent under normality). The 2026-07-29
    record quoted 0.8155/0.7814 from simulation; the exact values are 0.8147/0.7818 -- the
    ~0.001 differences are that simulation's noise, re-verified 2026-08-05."""
    an = _analyzer()
    c1 = an.healthy_arm_context(48, 1.0)
    assert c1["p_joint_gate_bias0"] == pytest.approx(0.8147, abs=5e-4)
    assert c1["p_sd_gt_decisive"] == pytest.approx(0.3488, abs=5e-4)
    c2 = an.healthy_arm_context(48, 1.0125)
    assert c2["p_joint_gate_bias0"] == pytest.approx(0.7818, abs=5e-4)
    assert c2["p_sd_gt_decisive"] == pytest.approx(0.3951, abs=5e-4)


def test_healthy_context_carries_the_decisive_sd_threshold():
    """sd* = 0.30*sqrt(N)/t(0.975,N-1): above it a decisive pass is arithmetically impossible.
    At N=48 that is the pre-registered 1.0332."""
    an = _analyzer()
    assert an.healthy_arm_context(48, 1.0)["sd_decisive_max"] == pytest.approx(1.0332, abs=2e-4)


def test_healthy_context_closed_form_agrees_with_the_exact_t_simulation():
    """The closed form treats the arm mean as normal and (N-1)S^2/sigma^2 as chi-square; the
    exact null pull is sqrt(1+1/L)*t_{L-1}. At L=150 the two must agree to ~0.01 -- this is the
    test that licenses quoting closed-form numbers as THE corrected null."""
    an = _analyzer()
    L, n_arm, n_sim = 150, 48, 20000
    rng = np.random.default_rng(20260805)
    pulls = np.sqrt(1.0 + 1.0 / L) * rng.standard_t(L - 1, size=(n_sim, n_arm))
    mean_ok = np.abs(pulls.mean(axis=1)) <= 0.30
    sd = pulls.std(axis=1, ddof=1)
    joint = float(np.mean(mean_ok & (sd <= 1.1)))
    c = an.healthy_arm_context(n_arm, an.finite_L_null_sd([L]))
    assert c["p_joint_gate_bias0"] == pytest.approx(joint, abs=0.015)


# ------------------------------------------------------------------- analyzer wiring

METAL = ["f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1", "k_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1"]
FRES = ["f_res_amp", "f_res_slope"]


def _synthetic_leg_hetL(dirpath, L_list, seed=0):
    """A structurally real eBOSS ARM-P population with HETEROGENEOUS L across mocks, so the
    printed finite-L null sd is a genuine aggregate rather than one L's value."""
    import pickle as _pkl
    rng = np.random.default_rng(seed)
    names = (["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz",
              "bhfeedback"] + [f"tau0_z{i}" for i in range(13)]
             + ["alpha_lls", "alpha_subdla", "alpha_dla"])
    P = len(names)
    os.makedirs(dirpath, exist_ok=True)
    cfg = dict(runner.RUN_CFG_DEFAULTS, survey="eBOSS", leg="eBOSS",
               metal_prior="flatlog2node",
               hcd_prior_signature="50befc941edfc4c7" + "0" * 48,
               metal_selfdraw=True, fres_selfdraw=True)
    for m, L in enumerate(L_list):
        truth = np.concatenate([rng.uniform(0.2, 0.8, 9), np.linspace(0.2, 1.2, 13),
                                rng.uniform(0.3, 0.6, 3)])
        draws = truth[None, :] + rng.normal(0, 0.05, (L, P))
        se = {}
        for nm in METAL:
            se[nm] = {"draws": rng.normal(0.01, 0.002, L),
                      "truth": float(np.exp(rng.uniform(np.log(0.003), np.log(0.03))))}
        for nm, sig in zip(FRES, (0.05, 0.5)):
            se[nm] = {"draws": rng.normal(0.0, sig, L), "truth": float(rng.normal(0.0, sig))}
        d = dict(sim=f"s{m}", names=names, truth_vec=truth, draws=draws, L=L, n_div=0,
                 ll_true=-1931.7, ll_draws=rng.normal(-1930, 5, L), run_cfg=cfg,
                 sites_extra=se,
                 truth_site_semantics={"self_draw": METAL + FRES, "not_self_drawn": [],
                                       "note": ""})
        _pkl.dump(d, open(os.path.join(dirpath, f"mock_{m:04d}.pkl"), "wb"))


def test_readout_prints_and_persists_the_finite_L_null(tmp_path):
    """PI ruling 3d.1 end-to-end: the analyzer must PRINT the corrected null computed from the
    landed arm's own L distribution, carry the no-excuse guard, and persist it in the gate JSON
    -- an interpretation reference nobody can compute post-hoc is exactly the round-3 defect
    class (computed-then-discarded) again."""
    root = tmp_path / "root"
    L_list = [40, 60, 80, 120, 40, 60, 80, 120]
    _synthetic_leg_hetL(str(root / "prod_sbc_leg_eboss"), L_list)
    env = dict(os.environ, PYTHONPATH="/home/mfho/hcd_priya",
               SBC_PERLEG_OUTDIR=str(tmp_path / "figs"), MPLBACKEND="Agg")
    r = subprocess.run(
        ["/home/mfho/.conda/envs/emu-jax/bin/python3",
         "/home/mfho/hcd_priya/scripts/analyze_sbc_perleg.py", str(root), "t_finL"],
        capture_output=True, text=True, env=env, cwd="/home/mfho/hcd_priya", timeout=900)
    assert r.returncode == 0, f"analyzer failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"

    assert "finite-L null" in r.stdout, "the corrected null must be PRINTED at readout"
    assert "sd > 1.1" in r.stdout, "the printed context must carry the no-excuse guard"

    an = _analyzer()
    want_sd = an.finite_L_null_sd(L_list)
    j = json.load(open(tmp_path / "figs" / "t_finL_gate.json"))
    fl = j["legs"]["eBOSS"]["finite_L_null"]
    assert fl["sd"] == pytest.approx(want_sd, abs=1e-4)
    assert fl["gated"] is False, "an interpretation reference, never a gate criterion"
    assert 0.0 < fl["p_joint_gate_bias0"] < 1.0
    assert 0.0 < fl["p_sd_gt_decisive"] < 1.0
    assert "excuse" in fl["note"].lower() or "sd > 1.1" in fl["note"]
