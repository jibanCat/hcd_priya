"""Analyze the displaced-CENTER LLS closure arm (paired clean vs boosted).

The LLS mirror of analyze_dla_selfdraw.py, for a different question. The closing panel
(2026-07-19, PI decision 2) commissioned this arm to replace a LINEAR line in the n_s budget --
"~0.96 sigma_post of n_s bias per 1-prior-sigma error in the LLS incidence center" -- with a
NUTS measurement, because this project has repeatedly seen NUTS reverse linear intuition
(the metal n_s mechanism, the Option-B decorrelation). So this analyzer reports a MEASUREMENT,
not a verdict: there is no pre-declared pass/fail gate for this arm.

Headline product: the CENTER-ERROR LINE = shift in the cosmology posterior, in sigma_post, per
ONE prior-sigma of LLS incidence center error, for n_s and A_p (plus the mean-flux pair, since
tau0 is the suspected n_s channel).

TWO estimator decisions, both forced by review (2026-07-21, two independent agents):

1. NORMALIZE BY THE REALIZED DISPLACEMENT, PER MOCK -- not by the design-time stamp.
   The injection is MULTIPLICATIVE on each mock's own truth (apply_lls_truth_boost scales
   alpha by `boost`), while the prior sd is a FIXED absolute number. So mock m actually
   receives (boost-1)*truth_clean_m / sigma_prior = truth_clean_m / mu prior-sigma, which
   equals the stamped `latent_sigma_disp_center` only for a mock whose truth landed exactly on
   the prior center. On the campaign of record the 7 clean truths sit 0.46 prior-sigma low, so
   the realized displacements run 0.674..1.127 (mean 0.867) and dividing by the stamped 1.000
   understated every line by ~15%. `latent_sigma_disp_center` is kept as a provenance stamp
   only. The realized per-mock displacements are written to the npz so a downstream reader can
   recompute the normalization from the frozen artifact.

2. THE PAIRED STATISTIC IS A MEAN SHIFT, NOT A DIFFERENCE OF bias_z.
   The DLA selfdraw arm pools Delta bias_z = (mu_b - T)/s_b - (mu_c - T)/s_c, which is right
   for its EQUIVALENCE gate (a calibration statement). Here the estimand is "how far did the
   posterior move", and that difference-of-bias_z picks up a spurious -T*(1/s_b - 1/s_c) term
   whenever the two arms' posterior widths differ -- which they do, since ESS-thinned draw
   counts run 20..150. Immaterial for n_s/A_p (whose truth/sigma ratios are modest) but it
   moved tau0_amp from +0.14 to +0.23 on the real data. PRIMARY is therefore
   (mu_b - mu_c)/s_pooled; the DLA-convention Delta bias_z is reported as a SECONDARY line for
   continuity with that arm.

Each shard pkl holds ONE paired mock: a CLEAN run (truth drawn from the deployed prior) and a
BOOSTED run whose alpha_lls truth is scaled by meta.boost in the pivot AND in every z-resolved
alpha_hcd_z[:, 0] row, at the SAME (seed, mock index). Truth theta and the cosmic noise draw
are shared, so the paired per-mock difference cancels the shared noise.

Ingest asserts (mirroring the DLA analyzer, which is the certified original and is NOT edited
here): reduced DESI covariance, forward_signature + hcd_prior_signature present and identical,
single survey, single boost, single prior width, single prior center, single stamped
displacement, no duplicate mock idxs, smoke pkls filtered, and the per-pair truth contract
across truth_vec, truth_alpha_hcd_z (both the boosted LLS column and the untouched others) and
every sites_extra truth.

Campaign completeness is fail-loud BY DEFAULT. Job 54175284 task 2 hit the 18h wall, so the
campaign of record is N=7 of 8; --allow-incomplete opts into that readout and STAMPS the
missing mock idxs onto every quotable line. A reduced-N readout is admissible only if it is
stamped, never silent.

Pure numpy on purpose -- safe on the login node (no JAX import).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/analyze_lls_dispcenter.py \
     --shard-dir /scratch/cavestru_root/cavestru1/mfho/lls_dispcenter --allow-incomplete \
     [--npz-out out.npz] [--fig-out fig.png]
"""
import argparse
import functools
import glob
import os
import pickle

import numpy as np

# The DLA selfdraw analyzer is the certified original (its readout is a signed gate of record),
# so it is imported, never edited. These helpers are indexed BY SITE NAME and carry no DLA
# specialization; sharing them keeps one implementation of the rank/coverage statistics and of
# the L-weighted pooling (including the closing-panel fixes: exact Student-t bound, None-safe
# weight alignment, rank-uniformity z).
from scripts.analyze_dla_selfdraw import (
    _col_draws_truth,
    bias_z_named,
    central_coverage,
    paired_delta_named,
    pool_deltas,
    rank_stats,
    rank_u,
)

print = functools.partial(print, flush=True)

ALPHA = "alpha_lls"
PAIRED_PARAMS = ("ns", "Ap", "tau0_amp", "dtau0")
# The linear rescale this arm replaces. It is an n_s-ONLY number (closing panel: "0.5 sigma per
# prior-sigma measured at width 0.15 rescales to ~0.96 sigma n_s at 0.287"); there is no A_p
# linear line, so the comparison column is printed for n_s alone.
LINEAR_REFERENCE = 0.96


# ---------------------------------------------------------------- ingest

def load_shards(shard_dir, allow_incomplete=False):
    """Load every lls_dispcenter shard pkl in `shard_dir`, ordered by shard index. Returns
    (clean_per_mock, boost_per_mock, meta) with the two lists index-aligned, and `meta`
    annotated with `missing_idxs` / `n_mocks_expected` / `n_mocks_used`.

    Every assert here rejects a campaign that is not what it claims to be. `allow_incomplete`
    relaxes EXACTLY ONE of them (the completeness check) and records what was missing; it does
    not relax disjointness, which is a different failure entirely.
    """
    paths = sorted(glob.glob(os.path.join(shard_dir, "lls_dispcenter_*_shard_*.pkl")))
    paths = [p for p in paths if ".smoke" not in os.path.basename(p)]
    assert paths, f"no lls_dispcenter shard pkls under {shard_dir}"
    clean, boost, meta0, idxs = [], [], None, []
    ref = {}                      # the campaign-identity fields, fixed by the first shard
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        sv = d.get("survey")
        assert sv is not None, f"{p}: no survey key (cannot verify single-survey pooling)"
        fwd, pri = d["meta"]["forward"], d["meta"]["prior"]
        assert fwd.get("dla_cov_reduced") is True, \
            f"{p}: dla_cov_reduced is not True (stale-cov pkl)"
        sig = fwd.get("forward_signature")
        assert sig, f"{p}: forward_signature is missing/None (vacuous consistency check)"
        # Every field below feeds either the estimand or its divisor, so each gets its own
        # cross-shard guard. Note disp = (boost-1)*mu/sigma: equal boost AND equal sigma still
        # do not imply equal displacement, so it cannot be inferred from the others.
        cur = {
            "survey": sv,
            "forward_signature": sig,
            "hcd_prior_signature": fwd.get("hcd_prior_signature"),
            "boost": float(d["meta"]["boost"]),
            "alpha_lls_sigma": float(pri["alpha_lls_sigma"]),
            "alpha_lls_mu": float(pri["alpha_lls_mu"]),
            "displacement": float(pri["latent_sigma_disp_center"]),
        }
        if not ref:
            ref, meta0 = cur, d["meta"]
        for k, v in cur.items():
            assert v == ref[k], \
                (f"{p}: {k} {v!r} != {ref[k]!r} (mixed-campaign pooling; the center-error line "
                 f"is normalized by these, so pooling two values is meaningless)")
        assert len(d["clean_per_mock"]) == len(d["boost_per_mock"]), \
            f"{p}: clean/boost per-mock lists differ in length (pairing broken)"
        idxs.extend(int(i) for i in d["idxs"])
        for rc, rb in zip(d["clean_per_mock"], d["boost_per_mock"]):
            _assert_pair_contract(p, rc, rb, cur["boost"])
            clean.append(rc)
            boost.append(rb)
    assert len(idxs) == len(set(idxs)), \
        (f"duplicate mock idxs {sorted(i for i in set(idxs) if idxs.count(i) > 1)}: a "
         f"double-counted mock must not pool through the readout")
    want = set(range(int(meta0["n_mocks"])))
    missing = sorted(want - set(idxs))
    if missing and not allow_incomplete:
        raise AssertionError(
            f"campaign incomplete: missing mock idxs {missing} (got {len(idxs)}/{len(want)}); "
            f"pass allow_incomplete=True to read out the reduced-N campaign with the missing "
            f"idxs stamped onto every quotable line")
    meta0["missing_idxs"] = missing
    meta0["n_mocks_expected"] = int(meta0["n_mocks"])
    meta0["n_mocks_used"] = len(idxs)
    assert len(clean) == len(idxs), (len(clean), len(idxs))
    return clean, boost, meta0


def _assert_pair_contract(p, rc, rb, boost):
    """Per-pair truth contract: alpha_lls scaled by `boost` in the pivot AND in every z-resolved
    LLS row; every other truth entry, every other class column, and every sites_extra truth
    bit-identical. Nothing here may silently no-op -- a missing key is a failure, not a skip."""
    j = list(rc["names"]).index(ALPHA)
    tc, tb = np.asarray(rc["truth_vec"]), np.asarray(rb["truth_vec"])
    assert np.isclose(tb[j], boost * tc[j], rtol=1e-12, atol=0), \
        f"{p}: boosted alpha_lls truth {tb[j]} != boost {boost} x clean {tc[j]}"
    keep = np.ones(len(tc), dtype=bool)
    keep[j] = False
    assert np.array_equal(tc[keep], tb[keep]), \
        f"{p}: non-LLS truth entries differ between arms (pairing broken)"
    azc, azb = rc.get("truth_alpha_hcd_z"), rb.get("truth_alpha_hcd_z")
    assert azc is not None and azb is not None, \
        (f"{p}: truth_alpha_hcd_z is absent -- the z-row contract is the only check that the "
         f"z-resolved injection matches the pivot one, and it must not silently no-op")
    azc, azb = np.asarray(azc), np.asarray(azb)
    assert np.allclose(azb[:, 0], boost * azc[:, 0], rtol=1e-12, atol=0), \
        (f"{p}: truth_alpha_hcd_z[:, 0] is not the stamped boost {boost} x clean -- a "
         f"pivot-only displacement would report a different quantity than the arm injected")
    assert np.array_equal(azc[:, 1:], azb[:, 1:]), \
        f"{p}: truth_alpha_hcd_z non-LLS class columns differ between arms (pairing broken)"
    # tau0_amp / dtau0 lines are computed from these, so their pairing is load-bearing too.
    for k in set(rc.get("sites_extra", {})) | set(rb.get("sites_extra", {})):
        a, b = rc["sites_extra"][k]["truth"], rb["sites_extra"][k]["truth"]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, f"{p}: sites_extra[{k!r}] truth differs between arms ({a} vs {b})"


def prior_sigma(meta):
    """The deployed alpha_lls prior sd, live from the shard stamp (never hardcoded)."""
    return float(meta["prior"]["alpha_lls_sigma"])


def displacement_sigma(meta):
    """The DESIGN-time displacement stamp, evaluated by the runner at the prior center. Kept
    for provenance; the estimator uses `delivered_displacement` instead (see module docstring
    point 1)."""
    return float(meta["prior"]["latent_sigma_disp_center"])


def delivered_displacement(clean, boost, prior_sd):
    """Per-mock REALIZED displacement in prior-sigma: (truth_boost - truth_clean)/sigma_prior.
    Because the boost is multiplicative this equals truth_clean/mu, so a mock whose truth landed
    below the prior center received LESS than the nominal one sigma."""
    tc = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in clean], dtype=float)
    tb = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in boost], dtype=float)
    return (tb - tc) / float(prior_sd)


# ---------------------------------------------------------------- paired estimators

def paired_mean_shift(rec_clean, rec_boost, name):
    """PRIMARY paired statistic: (post_mean_boost - post_mean_clean) / pooled post sd.

    The cosmology truth is identical between arms, so this is purely "how far the posterior
    moved" in units of its own width. Unlike a difference of bias_z it does not pick up a
    term proportional to the truth times the arms' sd mismatch. Returns None if degenerate."""
    dc, _ = _col_draws_truth(rec_clean, name)
    db, _ = _col_draws_truth(rec_boost, name)
    sc, sb = float(dc.std(ddof=1)), float(db.std(ddof=1))
    s = np.sqrt(0.5 * (sc ** 2 + sb ** 2))
    if not np.isfinite(s) or s <= 0:
        return None
    return (float(db.mean()) - float(dc.mean())) / s


def mc_noise_floor(rec_clean, rec_boost, name):
    """Monte-Carlo-only sd of the paired statistic for one mock, from the retained-draw counts:
    sqrt(1/L_c + 1/L_b) in pooled-sd units. With L as low as 20 this is a large share of the
    observed mock-to-mock scatter, so the report states it rather than letting a reader read
    chain length as physics."""
    dc, _ = _col_draws_truth(rec_clean, name)
    db, _ = _col_draws_truth(rec_boost, name)
    return float(np.sqrt(1.0 / dc.size + 1.0 / db.size))


# ---------------------------------------------------------------- the center-error line

def center_error_line(deltas, disp):
    """The budget line: sigma_post of cosmology shift per ONE prior-sigma of LLS center error.

    `deltas` are the per-mock paired statistics; `disp` is either a scalar or a PER-MOCK
    sequence of realized displacements in prior-sigma (the latter is what the report uses --
    see module docstring point 1). Each mock is normalized by its OWN displacement before
    pooling, so the pooled number is a response per unit prior-sigma rather than a response at
    a nominal design point most mocks never sat at.

    The interval is the exact small-n Student-t bound (DLA analyzer closing-panel fix 7). SE and
    the bound are MAGNITUDES: a minus-sigma arm (--boost 0.713) keeps its sign on the point
    estimate only, so both arms pool into the same units.
    """
    from scipy.stats import t as _t
    dd = np.asarray(deltas, dtype=object)
    keep = [i for i, x in enumerate(deltas) if x is not None]
    d = np.asarray([deltas[i] for i in keep], dtype=float)
    if np.ndim(disp) == 0:
        disp_k = np.full(d.size, float(disp))
    else:
        disp_k = np.asarray([float(np.asarray(disp)[i]) for i in keep], dtype=float)
    assert d.size, "no usable paired deltas"
    assert np.all(disp_k != 0), "a realized displacement is 0 -- cannot normalize"
    per = d / disp_k                      # normalize each mock by its OWN displacement
    n = per.size
    mean = float(per.mean())
    se = float(per.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tq = float(_t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return {
        "n": n,
        "per_mock": per,
        "mean_raw": float(d.mean()),
        "per_prior_sigma": mean,
        "se_per_prior_sigma": abs(se),
        "t975": tq,
        "ub95": abs(mean) + tq * abs(se),
        "disp_used": disp_k,
        "linear_reference": LINEAR_REFERENCE,
        "shrink_vs_linear": abs(mean) / LINEAR_REFERENCE,
        "_n_dropped": int(dd.size - n),
    }


# ---------------------------------------------------------------- alpha_lls response

def response_slope(pairs, alpha=ALPHA):
    """Displaced-truth response slope: sum_m Delta post_mean / sum_m Delta truth, with a
    leave-one-out jackknife SE. 1 = the posterior tracks the displaced truth fully, 0 = fully
    prior-pinned. On this arm a LOW slope is the expected signature of the prior-dominated LLS
    sector, and it is exactly why the linear rescale over-predicted the cosmology response."""
    dmean, dtruth = [], []
    for rc, rb in pairs:
        dc, tc = _col_draws_truth(rc, alpha)
        db, tb = _col_draws_truth(rb, alpha)
        dmean.append(float(db.mean()) - float(dc.mean()))
        dtruth.append(tb - tc)
    dmean, dtruth = np.asarray(dmean), np.asarray(dtruth)
    n = dmean.size
    slope = float(dmean.sum() / dtruth.sum())
    if n > 1:
        loo = np.asarray([(dmean.sum() - dmean[i]) / (dtruth.sum() - dtruth[i])
                          for i in range(n)])
        se = float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))
    else:
        se = float("nan")
    return {"slope": slope, "slope_se": se, "dmean": dmean, "dtruth": dtruth}


# ---------------------------------------------------------------- contraction

def contraction_stats(recs, prior_sd, alpha=ALPHA):
    """alpha_lls posterior sd vs the deployed prior sd, plus the empirical prior sd recomputed
    from the clean-arm truth draws (which ARE prior samples) as a free cross-check."""
    sds = np.asarray([float(_col_draws_truth(r, alpha)[0].std()) for r in recs])
    truths = np.asarray([_col_draws_truth(r, alpha)[1] for r in recs])
    emp = float(truths.std(ddof=1)) if truths.size > 1 else 0.0
    ratio = float(sds.mean() / prior_sd)
    return {
        "post_sd_mean": float(sds.mean()),
        "post_sd": sds,
        "sd_ratio": ratio,
        "contraction": 1.0 - ratio,
        "prior_sd_empirical": emp,
    }


def _corr(rec, name_a, name_b):
    da, _ = _col_draws_truth(rec, name_a)
    db, _ = _col_draws_truth(rec, name_b)
    if da.std() <= 0 or db.std() <= 0:
        return np.nan
    return float(np.corrcoef(da, db)[0, 1])


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/scratch/cavestru_root/cavestru1/mfho/lls_dispcenter")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="read out a reduced-N campaign; the missing mock idxs are stamped "
                         "onto every quotable line (job 54175284 task 2 hit the 18h wall)")
    ap.add_argument("--npz-out", default=None)
    ap.add_argument("--fig-out", default=None)
    args = ap.parse_args()

    clean, boost, meta = load_shards(args.shard_dir, allow_incomplete=args.allow_incomplete)
    n = len(clean)
    sd_prior = prior_sigma(meta)
    disp = delivered_displacement(clean, boost, sd_prior)
    stamp = f"N={n}" + (f", MISSING {meta['missing_idxs']}" if meta["missing_idxs"] else "")

    print(f"== LLS displaced-CENTER analyzer: {n} paired mocks, boost {meta['boost']}, "
          f"forward_signature {meta['forward']['forward_signature'][:16]}... ==")
    if meta["missing_idxs"]:
        print(f"!! REDUCED-N READOUT: {meta['n_mocks_used']}/{meta['n_mocks_expected']} mocks; "
              f"MISSING mock idxs {meta['missing_idxs']}. Every number below is stamped "
              f"{stamp} and must be quoted as such.")
    print(f"   alpha_lls prior: mu {meta['prior']['alpha_lls_mu']:.6f} sd {sd_prior:.6f} "
          f"(frac {meta['prior']['frac_sigma']})")
    print(f"   REALIZED displacement (prior-sigma, per mock): {np.round(disp, 3).tolist()}")
    print(f"     range {disp.min():.3f}..{disp.max():.3f}, mean {disp.mean():.3f} -- the "
          f"design-time stamp is {displacement_sigma(meta):.3f}. The injection is "
          f"MULTIPLICATIVE, so each mock's realized displacement is its own truth/mu; every "
          f"line below is normalized per mock by the realized value, NOT by the stamp.")
    Lc = np.asarray([r["L"] for r in clean])
    Lb = np.asarray([r["L"] for r in boost])
    div = sum(int(r["n_div"]) for r in clean + boost)
    print(f"   retained draws L: clean {Lc.min()}..{Lc.max()}, boost {Lb.min()}..{Lb.max()}; "
          f"total divergences {div}")

    # -- alpha_lls recovery, both arms
    con = contraction_stats(clean, prior_sd=sd_prior)
    for armname, recs in (("clean", clean), ("boost", boost)):
        bz = np.asarray([x for x in (bias_z_named(r, ALPHA) for r in recs) if x is not None],
                        dtype=float)
        us = [rank_u(r, ALPHA) for r in recs]
        cov, rs = central_coverage(us), rank_stats(us)
        note = "" if armname == "clean" else "  (truth = displaced; coverage NOT an SBC statement)"
        print(f"\n-- alpha_lls [{armname}]{note}")
        print(f"   bias_z mean {bz.mean():+.3f} +/- {bz.std(ddof=1)/np.sqrt(bz.size):.3f} "
              f"(median {np.median(bz):+.3f})")
        print(f"   coverage: {cov['in68']}/{cov['n']} in central-68, {cov['in95']}/{cov['n']} "
              f"in central-95; mean rank u {rs['mean_u']:.3f} (z {rs['z']:+.2f} vs uniform)")
    # The DLA arm's "softplus skew explains a positive clean-arm mean pull" excuse does NOT
    # transfer (alpha_lls is a direct TruncatedNormal, no softplus), so the clean-arm pull is
    # interpreted here rather than left on the page unexplained.
    tc = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in clean])
    mu_p = float(meta["prior"]["alpha_lls_mu"])
    pinned = np.asarray([(mu_p - t) / float(_col_draws_truth(r, ALPHA)[0].std())
                         for r, t in zip(clean, tc)])
    bz_clean = np.asarray([bias_z_named(r, ALPHA) for r in clean], dtype=float)
    print(f"   INTERPRETATION: with contraction {con['contraction']:.3f} the posterior IS "
          f"essentially the prior, so any truth drawn below the prior center reads as a "
          f"positive bias_z by construction. Pure prior-pinning predicts mean "
          f"{pinned.mean():+.3f} vs measured {bz_clean.mean():+.3f} "
          f"(corr {np.corrcoef(bz_clean, pinned)[0, 1]:+.2f}) -- prior domination, not "
          f"miscalibration.")

    # -- response + contraction
    resp = response_slope(list(zip(clean, boost)))
    print(f"\n-- displaced-truth response: slope {resp['slope']:.3f} +/- {resp['slope_se']:.3f} "
          f"(jackknife; 1 = full tracking, 0 = fully prior-pinned)")
    print(f"-- contraction: post_sd {con['post_sd_mean']:.5f} / prior_sd {sd_prior:.5f} "
          f"= {con['sd_ratio']:.3f} (contraction {con['contraction']:.3f}); empirical prior sd "
          f"from the {n} clean truths: {con['prior_sd_empirical']:.5f} "
          f"(vs deployed {sd_prior:.5f} -- a small/narrow draw is a finite-N accident at "
          f"{stamp}, not a prior error)")

    # -- THE PRODUCT: the center-error line
    print(f"\n-- CENTER-ERROR LINE ({stamp}): posterior shift in sigma_post per ONE prior-sigma "
          f"of LLS center error")
    print(f"   PRIMARY estimator = (post_mean_boost - post_mean_clean)/pooled_sd, normalized "
          f"per mock by the REALIZED displacement. Measurement, NOT a gate: the DLA selfdraw "
          f"arm's 0.30 equivalence gate was defined at a different displacement (+0.408 latent "
          f"sigma) in a different parameterization and is NOT comparable row by row.")
    lines, secondary = {}, {}
    for p in PAIRED_PARAMS:
        shifts = [paired_mean_shift(rc, rb, p) for rc, rb in zip(clean, boost)]
        line = center_error_line(shifts, disp)
        lines[p] = line
        # secondary: the DLA-convention Delta bias_z, and its L-weighted variant
        dbz = [paired_delta_named(rc, rb, p) for rc, rb in zip(clean, boost)]
        sec = center_error_line(dbz, disp)
        secondary[p] = (sec, pool_deltas([x / d for x, d in zip(dbz, disp) if x is not None],
                                         weights=np.minimum(Lc, Lb)))
        t = line["per_prior_sigma"] / line["se_per_prior_sigma"]
        mc = float(np.mean([mc_noise_floor(rc, rb, p) for rc, rb in zip(clean, boost)]))
        ref = f"  [vs linear {LINEAR_REFERENCE:.2f}: x{line['shrink_vs_linear']:.2f}]" \
            if p == "ns" else ""
        print(f"   {p:>9}: {line['per_prior_sigma']:+.3f} +/- {line['se_per_prior_sigma']:.3f} "
              f"sigma_post per prior-sigma (t {t:+.2f}, ub95 {line['ub95']:.3f}, {stamp})"
              f"{ref}")
        print(f"             secondary: Delta bias_z {sec['per_prior_sigma']:+.3f}, "
              f"L-weighted {secondary[p][1]['wmean']:+.3f}; per-mock MC-only sd {mc:.3f} vs "
              f"observed {line['per_mock'].std(ddof=1):.3f} "
              f"(MC share of variance {min(1.0, (mc/line['per_mock'].std(ddof=1))**2):.0%})")

    ns_l = lines["ns"]
    print(f"\n== BUDGET LINE OF RECORD (n_s): {ns_l['per_prior_sigma']:+.3f} +/- "
          f"{ns_l['se_per_prior_sigma']:.3f} sigma_post per 1-prior-sigma LLS center error "
          f"(95% bound {ns_l['ub95']:.3f}), {stamp}. Replaces the linear "
          f"{LINEAR_REFERENCE:.2f} assumption.")

    # -- degeneracy diagnostics
    print("\n-- corr(alpha_lls; .) pooled over clean-arm mocks")
    for p in ("tau0_amp", "Ap", "ns"):
        cs = np.asarray([_corr(r, ALPHA, p) for r in clean])
        print(f"   {p:>9}: mean {np.nanmean(cs):+.3f} (range {np.nanmin(cs):+.3f}.."
              f"{np.nanmax(cs):+.3f})")

    if args.npz_out:
        np.savez(
            args.npz_out,
            n_mocks_used=meta["n_mocks_used"], n_mocks_expected=meta["n_mocks_expected"],
            missing_idxs=np.asarray(meta["missing_idxs"], dtype=int),
            displacement_stamp=displacement_sigma(meta), delivered_disp=disp,
            prior_sd=sd_prior, prior_mu=mu_p,
            truth_clean=np.asarray([_col_draws_truth(r, ALPHA)[1] for r in clean]),
            truth_boost=np.asarray([_col_draws_truth(r, ALPHA)[1] for r in boost]),
            L_clean=Lc, L_boost=Lb,
            dmean=resp["dmean"], dtruth=resp["dtruth"], slope=resp["slope"],
            slope_se=resp["slope_se"], post_sd=con["post_sd"],
            **{f"shift_{p}": lines[p]["per_mock"] for p in PAIRED_PARAMS},
            **{f"line_{p}": np.asarray([lines[p]["per_prior_sigma"],
                                        lines[p]["se_per_prior_sigma"], lines[p]["ub95"]])
               for p in PAIRED_PARAMS},
        )
        print(f"\nnpz written: {args.npz_out}")

    if args.fig_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        tb = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in boost])
        pc = np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in clean])
        pb = np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in boost])
        sb = np.asarray([float(_col_draws_truth(r, ALPHA)[0].std()) for r in boost])
        ax.errorbar(tc, pc, yerr=con["post_sd"], fmt="o", label="clean", alpha=0.8)
        ax.errorbar(tb, pb, yerr=sb, fmt="s", label=f"boost ({meta['boost']}x)", alpha=0.8)
        lim = [0, max(tb.max(), pb.max()) * 1.15]
        ax.plot(lim, lim, "k--", lw=0.8, label="y = truth")
        ax.axhline(mu_p, color="g", ls=":", lw=1.0, label="prior centre")
        ax.set_xlabel("alpha_lls truth")
        ax.set_ylabel("posterior mean")
        ax.set_title(f"recovery (response slope {resp['slope']:.2f})")
        ax.legend(fontsize=8)
        ax = axes[1]
        for i, p in enumerate(PAIRED_PARAMS):
            d = lines[p]["per_mock"]
            ax.scatter(np.full(d.size, i) + np.linspace(-0.15, 0.15, d.size), d, s=14, alpha=0.7)
            ax.errorbar([i], [lines[p]["per_prior_sigma"]],
                        yerr=[2 * lines[p]["se_per_prior_sigma"]], fmt="D", color="k", capsize=4)
        ax.axhline(0, color="k", lw=0.8)
        for g in (LINEAR_REFERENCE, -LINEAR_REFERENCE):
            ax.axhline(g, color="r", ls="--", lw=0.8)
        ax.set_xticks(range(len(PAIRED_PARAMS)), PAIRED_PARAMS)
        ax.set_ylabel("sigma_post per 1 prior-sigma centre error")
        ax.set_title(f"centre-error line ({stamp}); red = linear {LINEAR_REFERENCE}")
        fig.tight_layout()
        fig.savefig(args.fig_out, dpi=150)
        print(f"figure written: {args.fig_out}")


if __name__ == "__main__":
    main()
