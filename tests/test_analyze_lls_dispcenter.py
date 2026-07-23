"""TDD suite for scripts/analyze_lls_dispcenter.py (the displaced-CENTER LLS closure arm).

The LLS mirror of tests/test_analyze_dla_selfdraw.py. Same paired clean-vs-boosted shard
schema, but the campaign measures a DIFFERENT estimand: the closing panel (2026-07-19, PI
decision 2) commissioned this arm to REPLACE the linear "~0.96 sigma_post n_s per 1-prior-sigma
LLS center error" line in the n_s budget with a NUTS measurement. So there is no pre-declared
pass/fail gate here -- the product is a normalized center-error line, and the analyzer must
report it per unit prior-sigma of truth displacement rather than per this campaign's particular
boost.

Contract under test:
  * ingest asserts mirroring the DLA analyzer (reduced cov, forward_signature consistency +
    non-None, single survey, single boost, no duplicate mock idxs, smoke pkls ignored) plus
    the LLS-specific truth contract: boosted alpha_lls == boost x clean, every other truth
    entry bit-identical, AND the z-resolved truth_alpha_hcd_z[:, 0] column scaled by the same
    boost with the other class columns untouched
  * campaign completeness fail-loud by DEFAULT, with an explicit allow_incomplete escape that
    RECORDS the missing mock idxs in meta (job 54175284 task 2 hit the 18h wall -> N=7 of 8;
    a reduced-N readout is admissible only if it is stamped, never silent)
  * the prior width is read LIVE from meta["prior"], not hardcoded, and must agree across shards
  * center_error_line normalizes the pooled paired delta by the stamped prior-sigma displacement
"""
import pickle

import numpy as np
import pytest

from scripts.analyze_lls_dispcenter import (
    ALPHA,
    center_error_line,
    contraction_stats,
    delivered_displacement,
    displacement_sigma,
    load_shards,
    paired_mean_shift,
    prior_sigma,
    response_slope,
)

SIG = "68f71a3d" + "0" * 56
NAMES = ["ns", "Ap", "tau0_z0", "alpha_lls", "alpha_subdla", "alpha_dla"]
J_LLS = NAMES.index("alpha_lls")
MU = 0.17211216067355312
SIGMA = 0.04939619011330974
BOOST = 1.287


def _rec(names, truth, mean, sd, L=100, seed=0, n_div=0, alpha_z=None):
    """Synthetic fit record matching the dispcenter shard pkl schema."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(mean, sd, size=(L, len(names)))
    if alpha_z is None:
        alpha_z = np.tile(np.array([truth[J_LLS], 0.06, 0.004]), (13, 1))
    return {
        "sim": "leg_a_prior",
        "truth_vec": np.asarray(truth, dtype=float),
        "draws": draws,
        "L": L,
        "names": list(names),
        "kept_global": np.ones(13, dtype=bool),
        "n_div": n_div,
        "truth_alpha_hcd_z": np.asarray(alpha_z, dtype=float),
        "sites_extra": {
            "tau0_amp": {"draws": rng.normal(1.0, 0.02, size=L), "truth": 1.0},
            "dtau0": {"draws": rng.normal(0.0, 0.05, size=L), "truth": 0.0},
        },
    }


def _pkl(path, shard, clean, boost, sig=SIG, cov_reduced=True, boost_val=BOOST, n_mocks=None,
         survey="desi", sigma=SIGMA, disp=1.0):
    payload = {
        "arm": "lls_dispcenter",
        "survey": survey,
        "idxs": [shard],
        "clean_per_mock": clean,
        "boost_per_mock": boost,
        "meta": {
            "shard": shard,
            "n_mocks": 2 if n_mocks is None else n_mocks,
            "boost": boost_val,
            "prior": {
                "alpha_lls_mu": MU,
                "alpha_lls_sigma": sigma,
                "frac_sigma": 0.287,
                "boost": boost_val,
                "latent_sigma_disp_center": disp,
                "latent_sigma_disp_median": disp,
            },
            "forward": {
                "res_corr_on": False,
                "fix_alpha_res": True,
                "dla_cov_reduced": cov_reduced,
                "dla_forward_frac": 1.0,
                "forward_signature": sig,
            },
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _paired_records(alpha_clean=MU, seed=1, boost_val=BOOST, **kw):
    """One clean/boost pair honouring the full truth contract: alpha_lls scaled by boost in
    BOTH the pivot vector and the z-resolved LLS column, everything else bit-identical."""
    truth_c = np.array([0.4, 0.8, 0.3, alpha_clean, 0.06, 0.004])
    truth_b = truth_c.copy()
    truth_b[J_LLS] = boost_val * alpha_clean
    az_c = np.tile(np.array([alpha_clean, 0.06, 0.004]), (13, 1))
    az_b = az_c.copy()
    az_b[:, 0] *= boost_val
    rc = _rec(NAMES, truth_c, truth_c, 0.05, seed=seed, alpha_z=az_c, **kw)
    rb = _rec(NAMES, truth_b, truth_c, 0.05, seed=seed + 1000, alpha_z=az_b, **kw)
    return rc, rb


# ---------------------------------------------------------------- ingest

def test_load_shards_reads_and_pairs(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 1, [rc2], [rb2])
    clean, boost, meta = load_shards(str(tmp_path))
    assert len(clean) == 2 and len(boost) == 2
    assert meta["boost"] == pytest.approx(BOOST)
    assert clean[0]["truth_vec"][J_LLS] == pytest.approx(MU)
    assert clean[1]["truth_vec"][J_LLS] == pytest.approx(0.19)
    assert meta["missing_idxs"] == []


def test_alpha_name_is_lls():
    assert ALPHA == "alpha_lls"


def test_load_shards_rejects_unreduced_cov(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], cov_reduced=False)
    with pytest.raises(AssertionError, match="dla_cov_reduced"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_signature_mismatch(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 1, [rc2], [rb2],
         sig="deadbeef" + "0" * 56)
    with pytest.raises(AssertionError, match="forward_signature"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_none_signature(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], sig=None, n_mocks=1)
    with pytest.raises(AssertionError, match="signature"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_mixed_survey(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], survey="desi")
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_eboss_shard_001.pkl", 1, [rc2], [rb2], survey="eboss")
    with pytest.raises(AssertionError, match="survey"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_mixed_boost(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2, boost_val=1.2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 1, [rc2], [rb2], boost_val=1.2)
    with pytest.raises(AssertionError, match="boost"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_mixed_prior_width(tmp_path):
    """The center-error line is normalized by the prior width; two widths in one campaign make
    the normalization meaningless, so it must fail loud rather than silently pool."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 1, [rc2], [rb2], sigma=2 * SIGMA)
    with pytest.raises(AssertionError, match="sigma"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_broken_truth_pairing(tmp_path):
    rc, rb = _paired_records()
    rb["truth_vec"][J_LLS] = rc["truth_vec"][J_LLS] * 1.4          # not the stamped boost
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="boost"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_nonlls_truth_drift(tmp_path):
    rc, rb = _paired_records()
    rb["truth_vec"][0] += 1e-6                                     # cosmology truth must match
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="truth"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_unscaled_z_row(tmp_path):
    """The runner scales the pivot alpha_hcd[0] AND every z-resolved alpha_hcd_z[:, 0] row.
    A pivot-only boost would displace a different quantity than the one being reported."""
    rc, rb = _paired_records()
    rb["truth_alpha_hcd_z"][:, 0] = rc["truth_alpha_hcd_z"][:, 0]  # z rows left unscaled
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="alpha_hcd_z"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_other_class_z_drift(tmp_path):
    """Only the LLS column may move; a subDLA/DLA z-row change means a different injection."""
    rc, rb = _paired_records()
    rb["truth_alpha_hcd_z"][:, 1] *= 1.01
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="alpha_hcd_z"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_duplicate_shard_idxs(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 0, [rc2], [rb2], n_mocks=1)
    with pytest.raises(AssertionError, match="duplicate"):
        load_shards(str(tmp_path))


def test_load_shards_ignores_smoke_pkls(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.smoke.pkl", 0, [rc2], [rb2], n_mocks=1)
    clean, _, _ = load_shards(str(tmp_path))
    assert len(clean) == 1
    assert clean[0]["truth_vec"][J_LLS] == pytest.approx(MU)


def test_load_shards_rejects_incomplete_campaign_by_default(tmp_path):
    """N=7 of 8 must NOT gate silently -- the default path is fail-loud, exactly as the DLA
    analyzer treats a hole in the shard sequence."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=8)
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_003.pkl", 3, [rc2], [rb2], n_mocks=8)
    with pytest.raises(AssertionError, match="missing"):
        load_shards(str(tmp_path))


def test_load_shards_allow_incomplete_records_missing_idxs(tmp_path):
    """The escape hatch is explicit AND self-documenting: the missing mock idxs come back in
    meta so every downstream readout stamps the reduced N instead of hiding it."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=8)
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_003.pkl", 3, [rc2], [rb2], n_mocks=8)
    clean, boost, meta = load_shards(str(tmp_path), allow_incomplete=True)
    assert len(clean) == 2 and len(boost) == 2
    assert meta["missing_idxs"] == [1, 2, 4, 5, 6, 7]
    assert meta["n_mocks_expected"] == 8
    assert meta["n_mocks_used"] == 2


def test_allow_incomplete_still_rejects_duplicates(tmp_path):
    """Relaxing completeness must not relax the disjointness guard -- a double-counted mock is
    a different failure from a missing one."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=8)
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 0, [rc2], [rb2], n_mocks=8)
    with pytest.raises(AssertionError, match="duplicate"):
        load_shards(str(tmp_path), allow_incomplete=True)


# ---------------------------------------------------------------- prior / displacement

def test_prior_sigma_and_displacement_read_live_from_meta(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1, disp=1.0)
    _, _, meta = load_shards(str(tmp_path))
    assert prior_sigma(meta) == pytest.approx(SIGMA)
    assert displacement_sigma(meta) == pytest.approx(1.0)


def test_displacement_sigma_uses_the_stamped_center_value(tmp_path):
    """A half-sigma arm (--boost 0.6435 say) must normalize by 0.5, not by 1."""
    rc, rb = _paired_records(boost_val=1.1435)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1,
         boost_val=1.1435, disp=0.5)
    _, _, meta = load_shards(str(tmp_path))
    assert displacement_sigma(meta) == pytest.approx(0.5)


# ---------------------------------------------------------------- the center-error line

def test_center_error_line_normalizes_by_displacement():
    """The reported budget line is sigma_post of cosmology bias per ONE prior-sigma of LLS
    center error. A half-sigma arm measuring the same raw delta implies twice the line."""
    deltas = [0.20, 0.20, 0.20, 0.20]
    full = center_error_line(deltas, disp=1.0)
    half = center_error_line(deltas, disp=0.5)
    assert full["per_prior_sigma"] == pytest.approx(0.20)
    assert half["per_prior_sigma"] == pytest.approx(0.40)
    assert full["mean_raw"] == pytest.approx(0.20)


def test_center_error_line_reports_student_t_upper_bound():
    """Small-n interval uses the exact Student-t quantile (DLA analyzer closing-panel fix 7),
    normalized the same way as the mean."""
    deltas = [0.1, 0.2, 0.3, 0.4]
    line = center_error_line(deltas, disp=1.0)
    d = np.asarray(deltas)
    se = d.std(ddof=1) / np.sqrt(d.size)
    assert line["se_per_prior_sigma"] == pytest.approx(se)
    assert line["ub95"] == pytest.approx(abs(d.mean()) + 3.1824 * se, rel=1e-3)


def test_center_error_line_beats_the_linear_reference():
    """The line exists to be compared against the 0.96 linear rescale it replaces."""
    line = center_error_line([-0.09] * 7, disp=1.0)
    assert line["linear_reference"] == pytest.approx(0.96)
    assert line["shrink_vs_linear"] == pytest.approx(0.09 / 0.96, rel=1e-6)


def test_center_error_line_drops_none_deltas():
    line = center_error_line([0.2, None, 0.2], disp=1.0)
    assert line["n"] == 2
    assert line["per_prior_sigma"] == pytest.approx(0.2)


def test_center_error_line_accepts_per_mock_displacements():
    """The injection is MULTIPLICATIVE on each mock's own truth, so the displacement actually
    delivered varies mock to mock. Normalizing each mock by its OWN displacement is what makes
    the pooled number a response-per-unit-prior-sigma rather than a response at a nominal
    design point that most mocks never sat at."""
    line = center_error_line([0.2, 0.2], disp=[1.0, 0.5])
    # per-mock: 0.2/1.0 = 0.2 and 0.2/0.5 = 0.4 -> mean 0.3
    assert line["per_prior_sigma"] == pytest.approx(0.3)
    assert line["n"] == 2


def test_center_error_line_per_mock_disp_drops_none_in_step():
    """A dropped delta must drop its OWN displacement, not shift the alignment."""
    line = center_error_line([0.2, None, 0.2], disp=[1.0, 0.1, 0.5])
    assert line["n"] == 2
    assert line["per_prior_sigma"] == pytest.approx(0.3)


# ---------------------------------------------------------------- delivered displacement

def test_delivered_displacement_is_per_mock_not_the_stamped_nominal(tmp_path):
    """meta.prior.latent_sigma_disp_center is evaluated at the prior CENTRE. A mock whose truth
    sits below the centre receives a SMALLER absolute displacement, because boost multiplies
    that mock's own truth while the prior sd is a fixed absolute number:
        delivered = (boost - 1) * truth_clean / sigma_prior = truth_clean / mu   (at frac width)
    Quoting the nominal 1.000 for a mock that received 0.67 overstates the denominator."""
    rc, rb = _paired_records(alpha_clean=MU)                    # exactly at the centre -> 1.000
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=2)
    rc2, rb2 = _paired_records(alpha_clean=0.5 * MU, seed=2)    # half the centre -> 0.500
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 1, [rc2], [rb2], n_mocks=2)
    clean, boost, meta = load_shards(str(tmp_path))
    dd = delivered_displacement(clean, boost, prior_sigma(meta))
    assert dd[0] == pytest.approx(1.0, rel=1e-9)
    assert dd[1] == pytest.approx(0.5, rel=1e-9)
    # and the stamped nominal is NOT the delivered mean
    assert displacement_sigma(meta) == pytest.approx(1.0)
    assert np.mean(dd) == pytest.approx(0.75, rel=1e-9)


# ---------------------------------------------------------------- estimator robustness

def test_paired_mean_shift_is_zero_when_the_posterior_does_not_move():
    """The response estimand is 'how far did the posterior move', normalized by its width.
    If the two arms have the SAME posterior mean, the answer is 0 -- even when the two arms
    have different posterior WIDTHS and the truth sits far from the mean."""
    truth = np.array([0.4, 0.8, 0.3, MU, 0.06, 0.004])
    rc = _rec(NAMES, truth, truth + 0.5, 0.05, L=20000, seed=71)   # wide-ish arm
    rb = _rec(NAMES, truth, truth + 0.5, 0.02, L=20000, seed=72)   # same mean, narrower
    assert paired_mean_shift(rc, rb, "ns") == pytest.approx(0.0, abs=0.02)


def test_paired_delta_bias_z_is_contaminated_by_unequal_arm_widths():
    """Regression pin for the defect this replaced. The DLA-arm convention
    Delta bias_z = (mu_b - T)/s_b - (mu_c - T)/s_c picks up a spurious -T*(1/s_b - 1/s_c) term
    when the arms' posterior widths differ, which they do here (ESS-thinned draw counts run
    20..150). On the real campaign that pushed tau0_amp -- the site with the largest
    truth/sigma ratio -- from +0.14 to +0.23. This test pins that the MEAN-SHIFT estimator does
    not inherit the artifact; it is the reason the primary line is not Delta bias_z."""
    from scripts.analyze_dla_selfdraw import paired_delta_named
    truth = np.array([0.4, 0.8, 0.3, MU, 0.06, 0.004])
    rc = _rec(NAMES, truth, truth + 0.5, 0.05, L=20000, seed=81)
    rb = _rec(NAMES, truth, truth + 0.5, 0.02, L=20000, seed=82)
    contaminated = paired_delta_named(rc, rb, "ns")
    clean_est = paired_mean_shift(rc, rb, "ns")
    assert abs(contaminated) > 5.0          # the artifact is enormous here, not a rounding gap
    assert abs(clean_est) < 0.02


# ---------------------------------------------------------------- response + contraction

def _tight_pair(alpha_clean, seed, track=0.0, boost_val=BOOST):
    truth_c = np.array([0.4, 0.8, 0.3, alpha_clean, 0.06, 0.004])
    truth_b = truth_c.copy()
    truth_b[J_LLS] = boost_val * alpha_clean
    rc = _rec(NAMES, truth_c, truth_c, 0.005, L=4000, seed=seed)
    rb = _rec(NAMES, truth_c, truth_c, 0.005, L=4000, seed=seed + 1000)
    rb["truth_vec"] = truth_b
    rb["draws"][:, J_LLS] += track * (truth_b[J_LLS] - truth_c[J_LLS])
    return rc, rb


def test_response_slope_full_tracking():
    rows = [_tight_pair(a, 20 + i, track=1.0) for i, a in enumerate([0.15, 0.17, 0.19])]
    assert response_slope(rows)["slope"] == pytest.approx(1.0, abs=0.15)


def test_response_slope_prior_dominated():
    rows = [_tight_pair(a, 30 + i, track=0.0) for i, a in enumerate([0.15, 0.17, 0.19])]
    assert abs(response_slope(rows)["slope"]) < 0.15


def test_response_slope_jackknife_se():
    rows = [_tight_pair(a, 50 + i, track=1.0) for i, a in enumerate([0.15, 0.17, 0.19, 0.21])]
    s = response_slope(rows)
    assert 0 < s["slope_se"] < 0.2


def test_contraction_stats_uses_the_live_prior_sd():
    """Per-COLUMN sds, so a wrong column index cannot pass: only alpha_lls has sd 0.045."""
    truth = np.array([0.4, 0.8, 0.3, MU, 0.06, 0.004])
    sds = np.array([0.001, 0.002, 0.003, 0.045, 0.005, 0.006])
    recs = [_rec(NAMES, truth, truth, sds, L=4000, seed=40 + i) for i in range(3)]
    c = contraction_stats(recs, prior_sd=SIGMA)
    assert c["post_sd_mean"] == pytest.approx(0.045, abs=0.003)
    assert c["sd_ratio"] == pytest.approx(0.045 / SIGMA, rel=0.1)
    assert c["contraction"] == pytest.approx(1.0 - c["sd_ratio"])
    assert c["prior_sd_empirical"] == pytest.approx(0.0, abs=1e-12)   # identical truths


def test_contraction_stats_reads_the_alpha_column_not_a_neighbour():
    """Regression pin for the tautology the reviewer caught: with per-column sds, asking for
    the wrong site returns a visibly different number."""
    truth = np.array([0.4, 0.8, 0.3, MU, 0.06, 0.004])
    sds = np.array([0.001, 0.002, 0.003, 0.045, 0.005, 0.006])
    recs = [_rec(NAMES, truth, truth, sds, L=4000, seed=45 + i) for i in range(3)]
    assert contraction_stats(recs, prior_sd=SIGMA, alpha="alpha_subdla")["post_sd_mean"] \
        == pytest.approx(0.005, abs=0.001)


# ---------------------------------------------------------------- sign / bound correctness

def test_center_error_line_handles_a_negative_displacement_arm():
    """The docstring promises a minus-sigma arm (--boost 0.713) pools into the same units.
    Both the SE and the 95% bound are MAGNITUDES and must stay positive; only the point
    estimate carries a sign."""
    line = center_error_line([-0.2, -0.3, -0.25, -0.15], disp=-1.0)
    assert line["per_prior_sigma"] == pytest.approx(0.225)     # double sign flip
    assert line["se_per_prior_sigma"] > 0
    assert line["ub95"] > 0


def test_center_error_line_ub95_uses_the_magnitude_of_the_mean():
    """The real n_s line is NEGATIVE; without abs() the frozen bound would be understated."""
    neg = center_error_line([-0.1, -0.2, -0.3, -0.4], disp=1.0)
    pos = center_error_line([0.1, 0.2, 0.3, 0.4], disp=1.0)
    assert neg["ub95"] == pytest.approx(pos["ub95"])
    assert neg["ub95"] > 0.25


# ---------------------------------------------------------------- more ingest guards

def test_load_shards_rejects_missing_truth_alpha_hcd_z(tmp_path):
    """The z-row contract must not silently no-op when the key is absent -- it is the only
    check that the z-resolved injection matches the pivot one."""
    rc, rb = _paired_records()
    rc.pop("truth_alpha_hcd_z")
    rb.pop("truth_alpha_hcd_z")
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="truth_alpha_hcd_z"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_sites_extra_truth_drift(tmp_path):
    """tau0_amp/dtau0 lines are computed from sites_extra truths, so their pairing must be
    asserted too -- two of the four reported rows rested on an unchecked invariant."""
    rc, rb = _paired_records()
    rb["sites_extra"]["tau0_amp"]["truth"] += 1e-9
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="sites_extra"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_mixed_displacement(tmp_path):
    """disp = (boost-1)*mu/sigma, so equal boost AND equal sigma still do not imply equal
    displacement -- the headline's divisor needs its own guard."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], disp=1.0)
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    _pkl(tmp_path / "lls_dispcenter_desi_shard_001.pkl", 1, [rc2], [rb2], disp=0.25)
    with pytest.raises(AssertionError, match="displacement"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_mixed_prior_mu(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(alpha_clean=0.19, seed=2)
    p = tmp_path / "lls_dispcenter_desi_shard_001.pkl"
    _pkl(p, 1, [rc2], [rb2])
    d = pickle.load(open(p, "rb"))
    d["meta"]["prior"]["alpha_lls_mu"] = 3 * MU
    pickle.dump(d, open(p, "wb"))
    with pytest.raises(AssertionError, match="alpha_lls_mu"):
        load_shards(str(tmp_path))


# ---------------------------------------------------------------- report contract

def test_main_stamps_reduced_N_on_every_quotable_line(tmp_path, capsys):
    """The reduced-N contract is only real if the lines a reader COPIES carry the stamp.
    main() was previously exercised by no test at all."""
    from scripts.analyze_lls_dispcenter import main
    rc, rb = _paired_records()
    _pkl(tmp_path / "lls_dispcenter_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=4)
    for i, a in enumerate([0.16, 0.18], start=1):
        rci, rbi = _paired_records(alpha_clean=a, seed=10 * i)
        _pkl(tmp_path / f"lls_dispcenter_desi_shard_00{i}.pkl", i, [rci], [rbi], n_mocks=4)
    npz = tmp_path / "out.npz"
    import sys
    argv = sys.argv
    sys.argv = ["analyze_lls_dispcenter.py", "--shard-dir", str(tmp_path),
                "--allow-incomplete", "--npz-out", str(npz)]
    try:
        main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "BUDGET LINE OF RECORD" in out
    assert "N=3" in out
    assert "MISSING [3]" in out
    # every per-parameter line a reader copies carries the N stamp too
    for line in out.splitlines():
        if "per prior-sigma" in line and "CENTER-ERROR" not in line:
            assert "N=3" in line, f"unstamped quotable line: {line}"
    z = np.load(npz)
    assert int(z["n_mocks_used"]) == 3 and int(z["n_mocks_expected"]) == 4
    assert list(z["missing_idxs"]) == [3]
    # the realized displacement must be recoverable from the frozen artifact
    assert "truth_clean" in z and "truth_boost" in z
    assert z["delivered_disp"].shape == (3,)
