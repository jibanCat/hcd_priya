"""TDD suite for scripts/analyze_dla_selfdraw.py (the displaced-truth DLA selfdraw analyzer).

Contract under test (2026-07-17 handoff analyzer spec + 2026-07-18 meta-verdict confounders):
  * ingest asserts: meta.forward.dla_cov_reduced is True, forward_signature AND meta.boost
    identical across shards, campaign completeness (union of idxs == range(n_mocks)), per-pkl
    truth pairing (boosted alpha_dla == boost x clean, every non-DLA truth entry bit-identical
    between arms)
  * per-record stats indexed BY NAME via rec["names"] (25 = 9 theta + 13 tau0_z + 3 alpha)
  * paired Delta bias_z pooling with the |mean| + 2*SE confidence-bound gate (0.30 sigma)
  * alpha_dla displaced-truth response slope (paired Delta post-mean / Delta truth)
  * clean-arm SBC rank u = frac(draws < truth) + central-68/95 coverage counts
  * contraction vs the deployed prior sd (0.00925) + the empirical prior sd cross-check
  * L-weighted secondary pooling (retained draws L=17..150 -> unequal per-mock MC error)
"""
import pickle

import numpy as np
import pytest

from scripts.analyze_dla_selfdraw import (
    GATE,
    PRIOR_SD_ALPHA_DLA,
    bias_z_named,
    central_coverage,
    contraction_stats,
    load_shards,
    paired_delta_named,
    pool_deltas,
    rank_stats,
    rank_u,
    response_slope,
    verdict,
)

SIG = "68f71a3d" + "0" * 56


def _rec(names, truth, mean, sd, L=100, seed=0, n_div=0, tau0_truth=1.0, tau0_mean=1.0,
         tau0_sd=0.02):
    """Synthetic fit record matching the shard pkl schema. Draws are gaussian at (mean, sd)
    per named column; sites_extra carries tau0_amp/dtau0 in the {draws, truth} format."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(mean, sd, size=(L, len(names)))
    return {
        "sim": "leg_a_prior",
        "truth_vec": np.asarray(truth, dtype=float),
        "draws": draws,
        "L": L,
        "names": list(names),
        "kept_global": np.ones(13, dtype=bool),
        "n_div": n_div,
        "sites_extra": {
            "tau0_amp": {"draws": rng.normal(tau0_mean, tau0_sd, size=L), "truth": tau0_truth},
            "dtau0": {"draws": rng.normal(0.0, 0.05, size=L), "truth": 0.0},
        },
    }


NAMES = ["ns", "Ap", "tau0_z0", "alpha_lls", "alpha_subdla", "alpha_dla"]


def _pkl(path, shard, clean, boost, sig=SIG, cov_reduced=True, boost_val=1.5, n_mocks=None):
    payload = {
        "arm": "dla_selfdraw",
        "survey": "desi",
        "idxs": [shard],
        "clean_per_mock": clean,
        "boost_per_mock": boost,
        "meta": {
            "shard": shard,
            "n_mocks": 2 if n_mocks is None else n_mocks,
            "boost": boost_val,
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


def _paired_records(alpha_clean=0.004, seed=1, **kw):
    """One clean/boost record pair honouring the truth contract (boost = 1.5x on alpha_dla
    ONLY; all other truth entries bit-identical)."""
    truth_c = np.array([0.4, 0.8, 0.3, 0.18, 0.06, alpha_clean])
    truth_b = truth_c.copy()
    truth_b[5] = 1.5 * alpha_clean
    mean = truth_c.copy()
    rc = _rec(NAMES, truth_c, mean, 0.05, seed=seed, **kw)
    rb = _rec(NAMES, truth_b, mean, 0.05, seed=seed + 1000, **kw)
    return rc, rb


# ---------------------------------------------------------------- ingest

def test_load_shards_reads_and_pairs(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(alpha_clean=0.006, seed=2)
    _pkl(tmp_path / "dla_selfdraw_desi_shard_001.pkl", 1, [rc2], [rb2])
    clean, boost, meta = load_shards(str(tmp_path))
    assert len(clean) == 2 and len(boost) == 2
    assert meta["boost"] == 1.5
    # order follows shard index
    assert clean[0]["truth_vec"][5] == pytest.approx(0.004)
    assert clean[1]["truth_vec"][5] == pytest.approx(0.006)


def test_load_shards_rejects_unreduced_cov(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb], cov_reduced=False)
    with pytest.raises(AssertionError, match="dla_cov_reduced"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_signature_mismatch(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb])
    rc2, rb2 = _paired_records(seed=2)
    _pkl(tmp_path / "dla_selfdraw_desi_shard_001.pkl", 1, [rc2], [rb2], sig="deadbeef" + "0" * 56)
    with pytest.raises(AssertionError, match="forward_signature"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_broken_truth_pairing(tmp_path):
    rc, rb = _paired_records()
    rb["truth_vec"][5] = rc["truth_vec"][5] * 1.4          # not the stamped boost
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb])
    with pytest.raises(AssertionError, match="boost"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_nondla_truth_drift(tmp_path):
    rc, rb = _paired_records()
    rb["truth_vec"][0] += 1e-6                             # non-DLA truth must be bit-identical
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    with pytest.raises(AssertionError, match="truth"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_incomplete_campaign(tmp_path):
    """A hole in the shard sequence (or any missing mock) must be fail-loud, not a silent
    reduced-N gate (consistency agent 2 MAJOR)."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=16)
    rc2, rb2 = _paired_records(alpha_clean=0.006, seed=2)
    _pkl(tmp_path / "dla_selfdraw_desi_shard_003.pkl", 3, [rc2], [rb2], n_mocks=16)
    with pytest.raises(AssertionError, match="missing"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_duplicate_shard_idxs(tmp_path):
    """A duplicated mock index must not silently double-count through the gate (closing-panel
    fix 2; the set-only completeness check regressed the PR#14 disjointness mandate)."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb], n_mocks=1)
    rc2, rb2 = _paired_records(alpha_clean=0.006, seed=2)
    # second FILE carrying the SAME mock idx 0: the set-only check sees a complete {0} while
    # the gate would silently pool 2 mocks (the true escape case)
    _pkl(tmp_path / "dla_selfdraw_desi_shard_001.pkl", 0, [rc2], [rb2], n_mocks=1)
    with pytest.raises(AssertionError, match="duplicate"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_none_signature(tmp_path):
    """An all-None forward_signature must not pass the consistency check vacuously
    (closing-panel fix 5)."""
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb], sig=None, n_mocks=1)
    with pytest.raises(AssertionError, match="signature"):
        load_shards(str(tmp_path))


def test_load_shards_rejects_mixed_boost(tmp_path):
    rc, rb = _paired_records()
    _pkl(tmp_path / "dla_selfdraw_desi_shard_000.pkl", 0, [rc], [rb], boost_val=1.5)
    rc2, rb2 = _paired_records(alpha_clean=0.006, seed=2)
    rb2["truth_vec"][5] = 1.4 * rc2["truth_vec"][5]
    _pkl(tmp_path / "dla_selfdraw_desi_shard_001.pkl", 1, [rc2], [rb2], boost_val=1.4)
    with pytest.raises(AssertionError, match="boost"):
        load_shards(str(tmp_path))


# ---------------------------------------------------------------- per-record stats

def test_bias_z_named_recovers_known_shift():
    truth = np.array([0.4, 0.8, 0.3, 0.18, 0.06, 0.004])
    rec = _rec(NAMES, truth, truth, 0.05, L=4000, seed=3)
    b = bias_z_named(rec, "ns")
    assert abs(b) < 0.1                                    # unbiased draws -> ~0
    rec2 = _rec(NAMES, truth, truth + 0.05, 0.05, L=4000, seed=4)
    assert bias_z_named(rec2, "ns") == pytest.approx(1.0, abs=0.1)


def test_bias_z_named_sites_extra():
    truth = np.array([0.4, 0.8, 0.3, 0.18, 0.06, 0.004])
    rec = _rec(NAMES, truth, truth, 0.05, L=4000, seed=5, tau0_truth=1.0, tau0_mean=1.02,
               tau0_sd=0.02)
    assert bias_z_named(rec, "tau0_amp") == pytest.approx(1.0, abs=0.1)


def test_rank_u_and_coverage():
    truth = np.array([0.4, 0.8, 0.3, 0.18, 0.06, 0.004])
    rec = _rec(NAMES, truth, truth, 0.05, L=4000, seed=6)
    u = rank_u(rec, "Ap")
    assert 0.35 < u < 0.65                                 # truth at the centre
    cov = central_coverage([0.5, 0.5, 0.99, 0.01, 0.6])
    assert cov["n"] == 5
    assert cov["in68"] == 3                                # |u-0.5|<=0.34 for 0.5,0.5,0.6
    assert cov["in95"] == 3                                # 0.99 and 0.01 fall outside [0.025,0.975]


# ---------------------------------------------------------------- pooling + gate

def test_paired_delta_and_pooling_zero_shift():
    pairs = [_paired_records(alpha_clean=a, seed=10 + i, L=4000)
             for i, a in enumerate([0.003, 0.004, 0.005, 0.006])]
    deltas = [paired_delta_named(rc, rb, "ns") for rc, rb in pairs]
    pooled = pool_deltas(deltas)
    assert abs(pooled["mean"]) < 0.15
    # ub uses the exact small-n Student-t 97.5% quantile, not the fixed 2 (closing-panel
    # fix 7): t_{3,0.975} = 3.1824
    assert pooled["ub"] == pytest.approx(abs(pooled["mean"]) + 3.1824 * pooled["se"],
                                         rel=1e-3)
    assert verdict(pooled["ub"]) == "PASS"


def test_pool_deltas_flags_real_shift():
    deltas = [0.5, 0.6, 0.55, 0.45]                        # a real ~0.5 sigma paired shift
    pooled = pool_deltas(deltas)
    assert pooled["ub"] > GATE
    assert verdict(pooled["ub"]) == "FAIL"


def test_pool_deltas_weighted_secondary():
    deltas = [1.0, 0.0]
    pooled = pool_deltas(deltas, weights=[3.0, 1.0])
    assert pooled["wmean"] == pytest.approx(0.75)
    assert pool_deltas(deltas)["mean"] == pytest.approx(0.5)


def test_pool_deltas_weights_align_past_none():
    """A None delta must drop its OWN weight, not shift the alignment (consistency agent 2
    minor: weights[:n] truncation would pair wrong L with wrong delta)."""
    pooled = pool_deltas([1.0, None, 0.0], weights=[3.0, 5.0, 1.0])
    assert pooled["n"] == 2
    assert pooled["wmean"] == pytest.approx(0.75)


# ---------------------------------------------------------------- alpha_dla response

def _tight_pair(alpha_clean, seed, track=0.0):
    """Paired records with a TIGHT alpha_dla posterior (sd 0.005) so the slope estimate's MC
    noise (~sd/sqrt(L) per arm) is well inside the test tolerance. `track` shifts the boost-arm
    alpha_dla draws by track * (Delta truth)."""
    truth_c = np.array([0.4, 0.8, 0.3, 0.18, 0.06, alpha_clean])
    truth_b = truth_c.copy()
    truth_b[5] = 1.5 * alpha_clean
    rc = _rec(NAMES, truth_c, truth_c, 0.005, L=4000, seed=seed)
    rb = _rec(NAMES, truth_c, truth_c, 0.005, L=4000, seed=seed + 1000)
    rb["truth_vec"] = truth_b
    rb["draws"][:, 5] += track * (truth_b[5] - truth_c[5])
    return rc, rb


def test_response_slope_full_tracking():
    """Posterior mean moves by the full truth displacement -> slope 1."""
    rows = [_tight_pair(a, 20 + i, track=1.0) for i, a in enumerate([0.003, 0.004, 0.005])]
    s = response_slope(rows)
    assert s["slope"] == pytest.approx(1.0, abs=0.15)


def test_response_slope_prior_dominated():
    """Posterior mean does not move at all -> slope 0 (fully prior-pinned)."""
    rows = [_tight_pair(a, 30 + i, track=0.0) for i, a in enumerate([0.003, 0.004, 0.005])]
    s = response_slope(rows)
    assert abs(s["slope"]) < 0.15


def test_response_slope_jackknife_se():
    """Slope must come with a jackknife SE (consistency agent 1: the vs-heuristic comparison
    needs an uncertainty). Tight identical-tracking fixture -> small positive SE."""
    rows = [_tight_pair(a, 50 + i, track=1.0) for i, a in enumerate([0.003, 0.004, 0.005, 0.006])]
    s = response_slope(rows)
    assert s["slope_se"] > 0
    assert s["slope_se"] < 0.2


def test_rank_stats_uniformity_z():
    """mean-u + its z vs uniform (consistency agent 1 MAJOR 2: one-sided rank depletion must be
    surfaced, the softplus note only covers the MEAN bias_z)."""
    r = rank_stats([0.5] * 16)
    assert r["mean_u"] == pytest.approx(0.5)
    assert r["z"] == pytest.approx(0.0, abs=1e-12)
    r2 = rank_stats([0.2] * 16)
    assert r2["z"] == pytest.approx((0.2 - 0.5) / (np.sqrt(1.0 / 12.0) / 4.0))


# ---------------------------------------------------------------- contraction

def test_contraction_stats():
    truth = np.array([0.4, 0.8, 0.3, 0.18, 0.06, 0.004])
    recs = [_rec(NAMES, truth, truth, 0.05, L=4000, seed=40 + i) for i in range(3)]
    c = contraction_stats(recs, prior_sd=PRIOR_SD_ALPHA_DLA)
    assert c["post_sd_mean"] == pytest.approx(0.05, abs=0.01)
    assert c["sd_ratio"] == pytest.approx(0.05 / PRIOR_SD_ALPHA_DLA, rel=0.25)
    # empirical prior sd from the truth draws (all identical here -> 0)
    assert c["prior_sd_empirical"] == pytest.approx(0.0, abs=1e-12)
