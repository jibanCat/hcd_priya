"""Synthetic-fixture tests for scripts/a2c_mech_adjud.py (PI #14 mechanism adjudication).

NO-PEEK: synthetic only; the tool's first real-data contact is its single post-review run.
"""
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def MA():
    spec = importlib.util.spec_from_file_location(
        "ma", os.path.join(REPO, "scripts", "a2c_mech_adjud.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- frozen scope boundaries must REFUSE, not silently comply ---------------------------

def test_layer_o_refuses_loudly(MA):
    with pytest.raises(MA.AdjudRefusal, match="LAYER O IS INADMISSIBLE"):
        MA.layer_o_refuse()


def test_h2_condition_a_refuses_loudly(MA):
    with pytest.raises(MA.AdjudRefusal, match="NON-ADJUDICABLE"):
        MA.h2_condition_a_refuse()


def test_frozen_constants_are_the_preregistered_values(MA):
    assert MA.NS_SPLIT_PRIMARY == 1.00      # PI-frozen, never optimized
    assert MA.NS_SPLIT_SECONDARY == 0.995   # the code's own pre-existing boundary
    assert MA.KNN_K == 3
    assert MA.LOWK_CUT == 0.0102
    assert MA.MDE_RHO == 0.40 and MA.H3_COMPAT_FLOOR == 0.30


# --- coordinate algebra -------------------------------------------------------------------

def test_physical_ns_mapping(MA):
    assert np.isclose(MA.physical_ns(0.0), 0.80)
    assert np.isclose(MA.physical_ns(1.0), 1.05)
    assert np.isclose(MA.physical_ns(0.8), 1.00)   # the frozen split in unit coords


def test_tau_eff_coefficients_match_verified_arithmetic(MA):
    co = MA.tau_eff_coefficients()
    assert np.isclose(co["zbar"], 3.3351, atol=1e-3)
    assert np.isclose(co["c_bar"], 0.080458, atol=1e-5)
    assert np.isclose(co["c_bar_desi"], 0.037216, atol=1e-5)
    assert np.isclose(co["admixture_ratio"], 2.162, atol=2e-3)
    assert int(co["desi_mask"].sum()) == 11


def test_project_to_tau_eff_is_the_exact_jacobian(MA):
    co = MA.tau_eff_coefficients()
    out = MA.project_to_tau_eff(0.5, 2.0, co["c"])
    assert np.allclose(out, 0.5 + 2.0 * co["c"])


# --- H1 support metric ---------------------------------------------------------------------

def test_local_support_recovers_known_distances(MA):
    design = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    truth = np.array([[0.0, 0.0]])
    kth, nn1 = MA.local_support(truth, design, k=3)
    assert np.isclose(nn1[0], 0.0)
    assert np.isclose(kth[0], 1.0)       # 3rd nearest is the other unit-distance corner


def test_local_support_is_larger_in_a_sparse_corner(MA):
    rng = np.random.default_rng(1)
    dense = rng.uniform(0, 0.5, size=(50, 3))
    sparse_pt = np.array([[0.98, 0.5, 0.5]])
    design = np.vstack([dense, [[0.9, 0.5, 0.5]]])
    kth_sparse, _ = MA.local_support(sparse_pt, design, k=3)
    kth_dense, _ = MA.local_support(np.array([[0.25, 0.25, 0.25]]), design, k=3)
    assert kth_sparse[0] > kth_dense[0]


def test_local_support_refuses_bad_shapes(MA):
    with pytest.raises(MA.AdjudRefusal, match="dimension mismatch"):
        MA.local_support(np.zeros((3, 4)), np.zeros((5, 3)))
    with pytest.raises(MA.AdjudRefusal, match="exceeds"):
        MA.local_support(np.zeros((3, 2)), np.zeros((2, 2)), k=5)


# --- partial correlation + permutation null ------------------------------------------------

def test_spearman_partial_collapses_a_confounded_association(MA):
    rng = np.random.default_rng(2)
    z = rng.normal(size=300)
    x = z + 0.2 * rng.normal(size=300)
    y = z + 0.2 * rng.normal(size=300)
    from scipy.stats import spearmanr
    assert abs(spearmanr(x, y).statistic) > 0.85
    assert abs(MA.spearman_partial(x, y, z)) < 0.3


def test_spearman_partial_preserves_a_genuine_direct_link(MA):
    rng = np.random.default_rng(3)
    z = rng.normal(size=300)
    x = z + rng.normal(size=300)
    y = 0.9 * x + 0.3 * rng.normal(size=300)
    assert abs(MA.spearman_partial(x, y, z)) > 0.6


def test_permutation_p_is_calibrated(MA):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(4)
    ps = []
    for _ in range(40):
        a = rng.normal(size=48); b = rng.normal(size=48)
        ps.append(MA.permutation_p(lambda u, v: spearmanr(u, v).statistic, a, b, B=200)["p"])
    ps = np.asarray(ps)
    assert 0.0 <= float(np.mean(ps < 0.05)) < 0.18
    assert 0.3 < float(np.mean(ps)) < 0.7


def test_permutation_p_reports_mc_se_and_never_zero(MA):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(5)
    a = rng.normal(size=48); b = a + 0.05 * rng.normal(size=48)
    out = MA.permutation_p(lambda u, v: spearmanr(u, v).statistic, a, b, B=500)
    assert out["p"] >= 1.0 / 501.0
    assert out["mc_se"] > 0


def test_holm(MA):
    assert MA.holm([0.001, 0.9]) == [True, False]
    assert MA.holm([0.02, 0.03]) == [True, True]


# --- H2 exposure ----------------------------------------------------------------------------

def test_f_lowk_exposure_fraction(MA):
    k = np.array([0.001, 0.005, 0.02, 0.05])
    u = np.array([1.0, 1.0, 1.0, 1.0])
    out = MA.f_lowk_exposure(k, u)
    assert out["n_lowk"] == 2
    assert np.isclose(out["F_lowk"], 0.5)
    assert "EXPOSURE ONLY" in out["interpretation"]


def test_f_lowk_refuses_degenerate_score(MA):
    with pytest.raises(MA.AdjudRefusal, match="degenerate"):
        MA.f_lowk_exposure([0.001, 0.02], [0.0, 0.0])


# --- H3 projection ---------------------------------------------------------------------------

def test_absorber_projection_recovers_a_planted_response(MA):
    rng = np.random.default_rng(6)
    co = MA.tau_eff_coefficients()
    L = 200
    A = rng.normal(size=(L, 3))
    ln_amp = 0.4 * A[:, 0] - 0.2 * A[:, 1] + 0.01 * rng.normal(size=L)
    dtau = 0.3 * A[:, 2] + 0.01 * rng.normal(size=L)
    out = MA.absorber_projection(np.exp(ln_amp), dtau, A, co)
    assert np.allclose(out["dln_amp_dabs"], [0.4, -0.2, 0.0], atol=0.02)
    assert np.allclose(out["ddtau_dabs"], [0.0, 0.0, 0.3], atol=0.02)
    assert out["r2_amp"] > 0.95 and out["r2_dtau"] > 0.95


def test_absorber_projection_refuses_length_mismatch(MA):
    co = MA.tau_eff_coefficients()
    with pytest.raises(MA.AdjudRefusal, match="length mismatch"):
        MA.absorber_projection(np.ones(10), np.ones(10), np.ones((9, 3)), co)


# --- driver end-to-end (synthetic) ----------------------------------------------------------

def _mk_recs(MA, n=48, seed=9):
    rng = np.random.default_rng(seed)
    names = (["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz",
              "bhfeedback"] + [f"tau0_z{i}" for i in range(13)]
             + ["alpha_lls", "alpha_subdla", "alpha_dla"])
    co = MA.tau_eff_coefficients(); z = co["z"]
    recs = []
    for m in range(n):
        L = int(rng.choice([75, 150, 300]))
        truth = np.zeros(len(names)); draws = np.zeros((L, len(names)))
        for nm in names[:9] + ["alpha_lls", "alpha_subdla", "alpha_dla"]:
            j = names.index(nm)
            truth[j] = rng.uniform(0, 1) if j < 9 else rng.normal()
            draws[:, j] = rng.normal(truth[j], 0.3, L)
        amp_t, dt_t = rng.uniform(0.75, 1.25), rng.uniform(-0.4, 0.25)
        amp_d, dt_d = rng.normal(amp_t, 0.02, L), rng.normal(dt_t, 0.02, L)
        lt = amp_t * ((1 + z) / 4.0) ** dt_t * 0.0023 * (1 + z) ** 3.65
        ld = amp_d[:, None] * ((1 + z) / 4.0) ** dt_d[:, None] * 0.0023 * (1 + z) ** 3.65
        for i in range(13):
            j = names.index(f"tau0_z{i}"); truth[j] = lt[i]; draws[:, j] = ld[:, i]
        recs.append(dict(names=names, draws=draws, truth=truth, L=L,
                         sites_extra={"tau0_amp": {"draws": amp_d, "truth": amp_t},
                                      "dtau0": {"draws": dt_d, "truth": dt_t}}))
    return recs


def test_driver_end_to_end_and_scope_recorded(MA):
    rng = np.random.default_rng(10)
    recs = _mk_recs(MA)
    design = rng.uniform(0, 1, size=(60, 9))
    limits = np.column_stack([np.zeros(9), np.ones(9)])
    out = MA.run_adjudication(recs, design, limits, B=300)
    # frozen scope determinations must be RECORDED, not computed
    assert out["LayerO"]["status"] == "INADMISSIBLE" and out["LayerO"]["computed"] is False
    assert out["H2"]["condition_A"]["status"] == "NON_ADJUDICABLE"
    assert out["H2"]["condition_A"]["computed"] is False
    assert out["H1_H2_interaction"]["status"] == "NOT_TESTABLE"
    # H1 and H3 present
    assert out["H1"]["status"] in ("ADJUDICATED", "NON_IDENTIFIED")
    assert "collinearity" in out["H1"]
    assert out["H3"]["coefficients"]["n_desi_constrained"] == 11
    assert "GEOMETRIC" in out["H3"]["caveat"]
    assert len(out["H3"]["projected_by_rung"]) == 13


def test_driver_never_computes_layer_o(MA):
    """The driver must record Layer O without ever producing a rank value."""
    recs = _mk_recs(MA, n=12)
    design = np.random.default_rng(11).uniform(0, 1, size=(60, 9))
    limits = np.column_stack([np.zeros(9), np.ones(9)])
    out = MA.run_adjudication(recs, design, limits, B=100)
    blob = repr(out["LayerO"])
    assert "rank" not in blob.lower() or "INADMISSIBLE" in blob
    assert not any(k in out["LayerO"] for k in ("p", "observed", "statistic"))
