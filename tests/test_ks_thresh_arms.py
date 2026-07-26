"""TDD for the threshold-study registry (scripts/ks_thresh_arms.py).

The load-bearing assertions are the EXACT ENDPOINTS: at f=0 every construction must
reproduce the K0 baseline and at f=1 the corresponding X-battery corner arm. Those two
anchors are what make the mixture algebra checkable against arms that have already run,
and the pre-registration requires them to be asserted before any production fit.
No fits, no ctx build.
"""
import numpy as np
import pytest

import scripts.ks_thresh_arms as TH
import scripts.ks_xsel_arms as XA

Z = np.round(np.arange(2.2, 4.61, 0.2), 10)
NZ = Z.size


def drawn_rows(seed=3):
    """A plausible admissible drawn truth (interior of the mixture simplex)."""
    rng = np.random.default_rng(seed)
    rows = np.abs(rng.normal(0, 0.02, (NZ, 3))) + np.array([0.35, 0.06, 0.005])[None, :]
    assert XA.truth_admissible(rows)
    return rows


# ------------------------------------------------------------------ alpha-space fractions

@pytest.mark.parametrize("cls_idx", [0, 1])
def test_alpha_fraction_endpoints(cls_idx):
    a = drawn_rows()
    at0 = TH.fraction_alpha_rows(a, cls_idx, 0.0)
    assert np.array_equal(at0, a)                                  # f=0 IS the K0 baseline
    at1 = TH.fraction_alpha_rows(a, cls_idx, 1.0)
    assert np.allclose(at1, XA.corner_alpha_rows(NZ, cls_idx), rtol=0, atol=1e-15)


def test_alpha_fraction_is_convex_and_admissible():
    a = drawn_rows()
    for f in (0.0, 0.05, 0.1, 0.25, 0.3, 0.5, 0.75, 1.0):
        rows = TH.fraction_alpha_rows(a, 0, f)
        assert XA.truth_admissible(rows), f"left the simplex at f={f}"
        # linear interpolation, class by class
        assert np.allclose(rows, (1 - f) * a + f * XA.corner_alpha_rows(NZ, 0))
        # the OTHER classes are diluted, never zeroed outright (that was the naive form)
        if 0.0 < f < 1.0:
            assert np.all(rows[:, 1] > 0) and np.all(rows[:, 1] < a[:, 1] + 1e-15)


def test_alpha_fraction_does_not_mutate_the_caller():
    a = drawn_rows()
    before = a.copy()
    TH.fraction_alpha_rows(a, 0, 0.4)
    assert np.array_equal(a, before)


def test_naive_form_is_not_what_we_build():
    """Guard against regressing to `f * e_cls`, which would confound the selection fraction
    with removing the baseline HCD content."""
    a = drawn_rows()
    f = 0.3
    naive = np.zeros((NZ, 3))
    naive[:, 0] = f
    assert not np.allclose(TH.fraction_alpha_rows(a, 0, f), naive)
    # concretely: the subDLA content must survive at (1-f) of its drawn value
    assert np.allclose(TH.fraction_alpha_rows(a, 0, f)[:, 1], (1 - f) * a[:, 1])


@pytest.mark.parametrize("f", [-0.01, 1.01])
def test_alpha_fraction_refuses_out_of_range(f):
    with pytest.raises(AssertionError, match=r"\[0, 1\]"):
        TH.fraction_alpha_rows(drawn_rows(), 0, f)


def test_alpha_fraction_refuses_dla_direction():
    with pytest.raises(AssertionError, match="data-side"):
        TH.fraction_alpha_rows(drawn_rows(), 2, 0.3)


# ------------------------------------------------------------------ profiled fractions

def test_profiled_fraction_has_the_requested_mean_and_the_x4_shape():
    f_z = TH.profiled_fraction_f_z(0.30, Z)
    assert np.isclose(f_z.mean(), 0.30)                      # matched-mean, by construction
    shape = np.atleast_1d(XA.f_sel(Z))
    assert np.allclose(f_z / f_z.max(), shape / shape.max())  # same inverted-U SHAPE as X4
    assert f_z.argmax() == shape.argmax()


def test_profiled_fraction_reproduces_x4_at_its_own_mean():
    """Rescaling to the mean of f_sel itself must return f_sel: the X4 arm is the f_mean =
    mean(f_sel) member of this family."""
    shape = np.atleast_1d(XA.f_sel(Z))
    f_z = TH.profiled_fraction_f_z(float(shape.mean()), Z)
    assert np.allclose(f_z, shape)


def test_profiled_alpha_endpoints_and_admissibility():
    a = drawn_rows()
    assert np.array_equal(TH.profiled_fraction_alpha_rows(a, 0, 0.0, Z), a)
    rows = TH.profiled_fraction_alpha_rows(a, 0, 0.30, Z)
    assert XA.truth_admissible(rows)
    f_z = TH.profiled_fraction_f_z(0.30, Z)
    assert np.allclose(rows, (1 - f_z)[:, None] * a + f_z[:, None] * np.array([1.0, 0, 0]))


def test_profiled_fraction_refuses_an_unreachable_mean():
    """A mean so high that the peak would exceed 1 must fail loud, not silently clip."""
    with pytest.raises(AssertionError, match="exceeds 1"):
        TH.profiled_fraction_f_z(0.95, Z)


# ------------------------------------------------------------------ data-side (DLA) fractions

def test_ratio_endpoints_against_k0_and_x1():
    rng = np.random.default_rng(11)
    R = 0.80 + 0.05 * rng.random(40)          # a corrected-fork-like suppression ratio
    assert np.allclose(TH.thresh_ratio_rows(0.0, R), 1.0)     # f=0 -> leg byte-identical
    assert np.allclose(TH.thresh_ratio_rows(1.0, R), R)       # f=1 -> exactly X1
    f = 0.25
    assert np.allclose(TH.thresh_ratio_rows(f, R), 1.0 - f + f * R)


def test_ratio_is_the_sample_average_of_the_two_populations():
    """P_mix = (1-f) P_clean + f P_swap, with P_swap = P_clean * R."""
    rng = np.random.default_rng(5)
    P_clean = 1.0 + rng.random(30)
    R = 0.8 + 0.1 * rng.random(30)
    for f in (0.1, 0.25, 0.5):
        direct = (1 - f) * P_clean + f * (P_clean * R)
        via_ratio = P_clean * TH.thresh_ratio_rows(f, R)
        assert np.allclose(direct, via_ratio)


def test_ratio_refuses_nonpositive_result():
    with pytest.raises(AssertionError, match="non-positive"):
        TH.thresh_ratio_rows(1.0, np.array([0.5, -0.1]))


# ------------------------------------------------------------------ registry hygiene

def test_arms_and_cells():
    assert TH.arm_ids() == ("T1_dla010", "T2_dla025", "T3_lls030", "T4_lls030p")
    cells = TH.batch_cells()
    assert len(cells) == 64 and len(set(cells)) == 64
    assert cells[0] == ("T1_dla010", 0) and cells[-1] == ("T4_lls030p", 15)


def test_t3_and_t4_share_the_mean_fraction():
    """The z-profile contrast is only interpretable at a MATCHED mean fraction."""
    assert TH.ARMS["T3_lls030"]["f"] == TH.ARMS["T4_lls030p"]["f"]
    assert TH.ARMS["T3_lls030"]["cls"] == TH.ARMS["T4_lls030p"]["cls"]


def test_shard_names_cannot_collide_with_the_x_battery():
    n = TH.shard_pkl_name("T1_dla010", 3)
    assert n.startswith("ks_thresh_") and n.endswith("_shard_003.pkl")
    assert not n.startswith("ks_xsel_")
    assert TH.shard_pkl_name("T1_dla010", 3, smoke=True).endswith(".smoke.pkl")
    with pytest.raises(AssertionError, match="unknown threshold arm"):
        TH.shard_pkl_name("X1_dla100", 0)


def test_signature_differs_from_the_x_battery_registry():
    """The two campaigns must never pool: different payloads -> different signatures."""
    sig_t = TH.registry_signature()
    sig_x = XA.registry_signature()
    assert sig_t != sig_x
    assert len(sig_t) == 64


def test_signature_is_stable_and_pins_the_truth_table():
    assert TH.registry_signature() == TH.registry_signature()
    payload = TH._registry_payload()
    assert payload["truth_table_sha256"] == XA.XSEL_TRUTH_TABLE_SHA256
    assert payload["campaign"] == "threshold-study-1"
    # the deferred subDLA arms must not have crept into the payload
    assert set(payload["arms"]) == set(TH.ARMS) == set(TH.arm_ids())


def test_budget_matches_the_preregistration():
    assert 140 <= TH.campaign_cost_cpuh() <= 160     # ~150 CPU-h for T1-T4
