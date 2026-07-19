"""Analyze the displaced-truth PRIYA DLA selfdraw closure shards (paired clean vs boosted).

Each shard pkl stores ONE paired mock: a CLEAN run (truth drawn from the deployed prior) and a
BOOSTED run whose alpha_dla truth (pivot + every z row) is scaled by meta.boost (1.5 = 15% of lit
incidence, +0.408 latent sigma) at the SAME (seed, mock index) — the truth theta AND the cosmic
noise draw are shared, so the paired per-mock difference cancels the shared noise AND the A3
metal-floor A_p baseline. PI report spec (2026-07-17 handoff) for clean and boosted arms:

  * alpha_dla recovery + coverage (SBC rank u; central 68/95 — coverage is meaningful on the
    CLEAN arm only, whose truth is a prior draw)
  * posterior mean vs injected truth + the displaced-truth RESPONSE slope with jackknife SE
    (paired Delta post-mean / Delta truth; alpha_dla shows ~21% marginal sd contraction on the
    single DESI leg — the audit's "~0.4 expected" slope is the 1-D shrinkage heuristic
    1-(post/prior)^2, a lower reference that does not bind in the multi-parameter posterior)
  * posterior contraction vs the deployed prior (sd 0.00925) + the empirical prior sd from the
    16 clean truth draws as a cross-check
  * paired Delta bias_z for tau0_amp, dtau0, A_p, n_s in sigma_post units
    (gate: |mean| + 2*SE < 0.30 sigma; softplus skew makes UNPAIRED clean-arm mean pulls ~+0.65
    WITHOUT miscalibration — do not read the clean column as a failure)
  * divergences / retained draws L (L=17..150 across fits -> the L-weighted pooled mean is
    reported as a secondary line) / railing / corr(alpha_dla; tau0_amp, A_p, n_s)

Ingest asserts (reject stale-cov pkls): meta.forward.dla_cov_reduced is True and forward_signature
identical across shards; per-pair truth contract (boosted alpha_dla == boost x clean, all other
truth entries bit-identical). Pure numpy on purpose — safe on the login node (no JAX import).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/analyze_dla_selfdraw.py --shard-dir /scratch/cavestru_root/cavestru1/mfho/dla_selfdraw \
     [--npz-out out.npz] [--fig-out fig.png]
"""
import argparse
import functools
import glob
import os
import pickle

import numpy as np

print = functools.partial(print, flush=True)

GATE = 0.30                       # |mean Delta bias_z| + 2*SE pass threshold (sigma_post)
PRIOR_SD_ALPHA_DLA = 0.00925      # deployed alpha_dla(z=3) prior sd (2026-07-17 handoff)
ALPHA = "alpha_dla"
PAIRED_PARAMS = ("ns", "Ap", "tau0_amp", "dtau0")


# ---------------------------------------------------------------- ingest

def load_shards(shard_dir):
    """Load every dla_selfdraw shard pkl in `shard_dir`, ordered by shard index. Returns
    (clean_per_mock, boost_per_mock, meta) with the two lists index-aligned. Asserts the
    forward/pairing contract; any violation is a stale or corrupt campaign, not a soft skip."""
    paths = sorted(glob.glob(os.path.join(shard_dir, "dla_selfdraw_*_shard_*.pkl")))
    assert paths, f"no dla_selfdraw shard pkls under {shard_dir}"
    clean, boost, meta0, sig0, idxs = [], [], None, None, []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        fwd = d["meta"]["forward"]
        assert fwd.get("dla_cov_reduced") is True, \
            f"{p}: dla_cov_reduced is not True (stale-cov pkl)"
        sig = fwd.get("forward_signature")
        assert sig, f"{p}: forward_signature is missing/None (vacuous consistency check)"
        if sig0 is None:
            sig0, meta0 = sig, d["meta"]
        assert sig == sig0, \
            f"{p}: forward_signature {sig!r} != {sig0!r} (mixed-forward campaign)"
        b = float(d["meta"]["boost"])
        assert b == float(meta0["boost"]), \
            f"{p}: boost {b} != {meta0['boost']} (mixed-boost campaign)"
        assert len(d["clean_per_mock"]) == len(d["boost_per_mock"]), \
            f"{p}: clean/boost per-mock lists differ in length (pairing broken)"
        idxs.extend(int(i) for i in d["idxs"])
        for rc, rb in zip(d["clean_per_mock"], d["boost_per_mock"]):
            j = list(rc["names"]).index(ALPHA)
            tc, tb = np.asarray(rc["truth_vec"]), np.asarray(rb["truth_vec"])
            assert np.isclose(tb[j], b * tc[j], rtol=1e-12, atol=0), \
                f"{p}: boosted alpha_dla truth {tb[j]} != boost {b} x clean {tc[j]}"
            keep = np.ones(len(tc), dtype=bool)
            keep[j] = False
            assert np.array_equal(tc[keep], tb[keep]), \
                f"{p}: non-DLA truth entries differ between arms (pairing broken)"
            clean.append(rc)
            boost.append(rb)
    want = set(range(int(meta0["n_mocks"])))
    assert len(idxs) == len(set(idxs)), \
        (f"duplicate mock idxs {sorted(i for i in set(idxs) if idxs.count(i) > 1)}: a "
         f"double-counted mock must not pool through the gate")
    assert set(idxs) == want, \
        (f"campaign incomplete: missing mock idxs {sorted(want - set(idxs))} "
         f"(got {len(idxs)}/{len(want)}); a partial campaign must not gate silently")
    assert len(clean) == len(idxs) == int(meta0["n_mocks"]), (len(clean), len(idxs))
    return clean, boost, meta0


# ---------------------------------------------------------------- per-record stats

def _col_draws_truth(rec, name):
    """(draws_1d, truth) for a named site: a draws column for names[] sites, or the
    sites_extra {draws, truth} record for derived sites (tau0_amp, dtau0)."""
    names = list(rec["names"])
    if name in names:
        j = names.index(name)
        return np.asarray(rec["draws"])[:, j], float(rec["truth_vec"][j])
    e = rec["sites_extra"][name]
    return np.asarray(e["draws"]), float(e["truth"])


def bias_z_named(rec, name):
    """Signed bias_z = (post_mean - truth) / post_sd. None if the posterior sd is 0."""
    d, t = _col_draws_truth(rec, name)
    sd = float(d.std())
    if sd <= 0:
        return None
    return (float(d.mean()) - t) / sd


def rank_u(rec, name):
    """SBC rank u = frac(draws < truth); uniform on a calibrated (clean, prior-draw) arm."""
    d, t = _col_draws_truth(rec, name)
    return float((d < t).mean())


def central_coverage(us):
    """Central-interval coverage counts from rank statistics: u in [0.5-0.34, 0.5+0.34] for the
    68% interval, [0.025, 0.975] for the 95% interval."""
    u = np.asarray(list(us), dtype=float)
    return {
        "n": int(u.size),
        "in68": int(((u >= 0.16) & (u <= 0.84)).sum()),
        "in95": int(((u >= 0.025) & (u <= 0.975)).sum()),
    }


def rank_stats(us):
    """mean rank u + its z vs Uniform(0,1) (sd 1/sqrt(12n)). Rank uniformity holds under exact
    calibration REGARDLESS of posterior skew — the softplus mean-pull argument does not excuse a
    one-sided rank depletion, so this is surfaced separately from the interval counts."""
    u = np.asarray(list(us), dtype=float)
    n = u.size
    return {
        "mean_u": float(u.mean()),
        "z": float((u.mean() - 0.5) / (np.sqrt(1.0 / 12.0) / np.sqrt(n))),
    }


# ---------------------------------------------------------------- pooling + gate

def paired_delta_named(rec_clean, rec_boost, name):
    """Per-mock paired Delta bias_z = bias_z(boost) - bias_z(clean); the shared noise draw and
    the A3 metal-floor baseline cancel. None if either sd is degenerate."""
    bc, bb = bias_z_named(rec_clean, name), bias_z_named(rec_boost, name)
    if bc is None or bb is None:
        return None
    return bb - bc


def pool_deltas(deltas, weights=None):
    """Pool per-mock deltas: mean, SE (ddof=1), median, the confidence bound
    ub = |mean| + t_{n-1,0.975} * SE (the exact small-n Student-t quantile, NOT a fixed 2 —
    closing-panel fix 7; t_{15,0.975} = 2.131 at the campaign N=16), and an optional weighted
    mean (secondary line for the unequal retained-draw counts L). A None delta (degenerate
    posterior sd) drops its OWN weight, keeping the alignment."""
    from scipy.stats import t as _t
    keep = [i for i, x in enumerate(deltas) if x is not None]
    d = np.asarray([deltas[i] for i in keep], dtype=float)
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tq = float(_t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    out = {
        "n": n,
        "mean": float(d.mean()),
        "se": se,
        "median": float(np.median(d)),
        "t975": tq,
        "ub": abs(float(d.mean())) + tq * se,
    }
    if weights is not None:
        w = np.asarray([list(weights)[i] for i in keep], dtype=float)
        out["wmean"] = float((w * d).sum() / w.sum())
    return out


def verdict(ub):
    """Confidence-bound gate: PASS iff |mean| + 2*SE < GATE sigma_post."""
    return "PASS" if ub < GATE else "FAIL"


# ---------------------------------------------------------------- alpha_dla response

def response_slope(pairs):
    """Displaced-truth response slope: sum_m Delta post_mean(alpha_dla) / sum_m Delta truth,
    with a leave-one-out jackknife SE. 1 = full tracking, 0 = fully prior-pinned. NOTE: the
    "~0.4 expected" number from the 2026-07-17 audit is the 1-D Gaussian shrinkage heuristic
    1-(post_sd/prior_sd)^2, which does NOT bind in the multi-parameter posterior (the marginal
    alpha_dla sd is degeneracy-inflated while the displaced truth moves the data) — treat it as
    a lower reference, not a prediction. Returns per-mock arrays for the npz/figure."""
    dmean, dtruth = [], []
    for rc, rb in pairs:
        dc, tc = _col_draws_truth(rc, ALPHA)
        db, tb = _col_draws_truth(rb, ALPHA)
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
    return {
        "slope": slope,
        "slope_se": se,
        "dmean": dmean,
        "dtruth": dtruth,
    }


# ---------------------------------------------------------------- contraction

def contraction_stats(recs, prior_sd=PRIOR_SD_ALPHA_DLA):
    """alpha_dla posterior sd vs the prior sd, plus the empirical prior sd recomputed from the
    truth draws themselves (the clean-arm truths ARE prior samples — a free cross-check of the
    0.00925 constant)."""
    sds = np.asarray([float(_col_draws_truth(r, ALPHA)[0].std()) for r in recs])
    truths = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in recs])
    emp = float(truths.std(ddof=1)) if truths.size > 1 else 0.0
    return {
        "post_sd_mean": float(sds.mean()),
        "post_sd": sds,
        "sd_ratio": float(sds.mean() / prior_sd),
        "contraction": float(1.0 - sds.mean() / prior_sd),
        "prior_sd_empirical": emp,
    }


# ---------------------------------------------------------------- diagnostics

def _corr(rec, name_a, name_b):
    da, _ = _col_draws_truth(rec, name_a)
    db, _ = _col_draws_truth(rec, name_b)
    if da.std() <= 0 or db.std() <= 0:
        return np.nan
    return float(np.corrcoef(da, db)[0, 1])


def _railing_frac(rec, floor_frac=0.05, ref=0.004414):
    """Softplus-zero pileup diagnostic: fraction of alpha_dla draws below floor_frac x `ref`
    (the deployed prior MEDIAN, 0.004414; no hard bounds exist, near-zero pileup is the railing
    mode). main() passes --prior-median through."""
    d, _ = _col_draws_truth(rec, ALPHA)
    return float((d < floor_frac * ref).mean())


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/scratch/cavestru_root/cavestru1/mfho/dla_selfdraw")
    ap.add_argument("--prior-median", type=float, default=0.004414,
                    help="deployed alpha_dla prior median (railing reference)")
    ap.add_argument("--npz-out", default=None)
    ap.add_argument("--fig-out", default=None)
    args = ap.parse_args()

    clean, boost, meta = load_shards(args.shard_dir)
    n = len(clean)
    print(f"== DLA selfdraw analyzer: {n} paired mocks, boost {meta['boost']}, "
          f"forward_signature {meta['forward']['forward_signature'][:16]}... ==")
    Lc = np.asarray([r["L"] for r in clean])
    Lb = np.asarray([r["L"] for r in boost])
    div = sum(int(r["n_div"]) for r in clean + boost)
    print(f"retained draws L: clean {Lc.min()}..{Lc.max()}, boost {Lb.min()}..{Lb.max()}; "
          f"total divergences {div}")

    # -- alpha_dla recovery, both arms
    for arm, recs in (("clean", clean), ("boost", boost)):
        bz = np.asarray([x for x in (bias_z_named(r, ALPHA) for r in recs)
                         if x is not None], dtype=float)   # None-guard (closing-panel fix 8)
        us = [rank_u(r, ALPHA) for r in recs]
        cov = central_coverage(us)
        pm = np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in recs])
        tr = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in recs])
        rs = rank_stats(us)
        note = "" if arm == "clean" else "  (truth = displaced; coverage NOT an SBC statement)"
        print(f"\n-- alpha_dla [{arm}]{note}")
        print(f"   bias_z mean {bz.mean():+.3f} +/- {bz.std(ddof=1)/np.sqrt(n):.3f} "
              f"(median {np.median(bz):+.3f}); softplus-skew note: unpaired clean-arm MEAN "
              f"pulls ~+0.65 are EXPECTED without miscalibration (mean-only argument)")
        print(f"   post-mean vs truth: <mean> {pm.mean():.6f} vs <truth> {tr.mean():.6f}")
        print(f"   coverage: {cov['in68']}/{cov['n']} in central-68, "
              f"{cov['in95']}/{cov['n']} in central-95; mean rank u {rs['mean_u']:.3f} "
              f"(z {rs['z']:+.2f} vs uniform — skew does NOT excuse rank non-uniformity)")
        rail = np.asarray([_railing_frac(r, ref=args.prior_median) for r in recs])
        print(f"   railing (draws < 5% prior median): max frac {rail.max():.3f}")

    # -- displaced-truth response + contraction
    resp = response_slope(list(zip(clean, boost)))
    con = contraction_stats(clean)
    print(f"\n-- displaced-truth response: slope {resp['slope']:.3f} +/- {resp['slope_se']:.3f} "
          f"(jackknife; 1 = full tracking; the 1-D shrinkage heuristic 1-(post/prior)^2 "
          f"= {1 - con['sd_ratio']**2:.2f} is a lower REFERENCE only — it does not bind in "
          f"the multi-parameter posterior)")
    print(f"-- contraction: post_sd {con['post_sd_mean']:.5f} / prior_sd {PRIOR_SD_ALPHA_DLA} "
          f"= {con['sd_ratio']:.3f} (contraction {con['contraction']:.3f}); empirical prior sd "
          f"from the {n} clean truths: {con['prior_sd_empirical']:.5f}")

    # -- paired cosmology/mean-flux deltas (the PI gate)
    print(f"\n-- paired Delta bias_z (boost - clean), EQUIVALENCE gate |mean|+2SE < {GATE} "
          f"sigma_post (a bounded-response gate, not a zero-test)")
    any_fail = False
    pooled_all = {}
    for p in PAIRED_PARAMS:
        deltas = [paired_delta_named(rc, rb, p) for rc, rb in zip(clean, boost)]
        pooled = pool_deltas(deltas, weights=np.minimum(Lc, Lb))
        pooled_all[p] = pooled
        v = verdict(pooled["ub"])
        any_fail |= (v == "FAIL")
        t = pooled["mean"] / pooled["se"]
        print(f"   {p:>9}: mean {pooled['mean']:+.3f} +/- {pooled['se']:.3f} (t {t:+.2f}, "
              f"median {pooled['median']:+.3f}, L-weighted {pooled['wmean']:+.3f}) "
              f"ub {pooled['ub']:.3f} {v}")
        if v == "PASS" and abs(t) > 2:
            print(f"             ^ CAVEAT: nonzero at {abs(t):.1f} SE — a small real response "
                  f"bounded within the gate; naive linear scaling to a ~{GATE/abs(pooled['mean']):.1f}x "
                  f"larger truth error would approach the budget")

    # -- degeneracy diagnostics
    print("\n-- corr(alpha_dla; .) pooled over clean-arm mocks")
    for p in ("tau0_amp", "Ap", "ns"):
        cs = np.asarray([_corr(r, ALPHA, p) for r in clean])
        print(f"   {p:>9}: mean {np.nanmean(cs):+.3f} (range {np.nanmin(cs):+.3f}.."
              f"{np.nanmax(cs):+.3f})")

    print("\n== OVERALL:", "FAIL — a paired delta exceeds the confidence-bound gate; per the PI "
          "disposition diagnose the tau0 channel first, do NOT add a residual template."
          if any_fail else
          "PASS — alpha_dla tracks the displaced truth and every paired delta is BOUNDED within "
          "the 0.30 equivalence gate (see per-param t and any CAVEAT lines above — bounded, not "
          "necessarily zero).")

    if args.npz_out:
        np.savez(
            args.npz_out,
            L_clean=Lc, L_boost=Lb,
            dmean=resp["dmean"], dtruth=resp["dtruth"], slope=resp["slope"],
            post_sd=con["post_sd"],
            truth_clean=np.asarray([_col_draws_truth(r, ALPHA)[1] for r in clean]),
            truth_boost=np.asarray([_col_draws_truth(r, ALPHA)[1] for r in boost]),
            postmean_clean=np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in clean]),
            postmean_boost=np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in boost]),
            **{f"delta_{p}": np.asarray([paired_delta_named(rc, rb, p)
                                         for rc, rb in zip(clean, boost)], dtype=float)
               for p in PAIRED_PARAMS},
        )
        print(f"npz written: {args.npz_out}")

    if args.fig_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        tc = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in clean])
        tb = np.asarray([_col_draws_truth(r, ALPHA)[1] for r in boost])
        pc = np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in clean])
        pb = np.asarray([float(_col_draws_truth(r, ALPHA)[0].mean()) for r in boost])
        sc = con["post_sd"]
        sb = np.asarray([float(_col_draws_truth(r, ALPHA)[0].std()) for r in boost])
        ax.errorbar(tc, pc, yerr=sc, fmt="o", label="clean", alpha=0.8)
        ax.errorbar(tb, pb, yerr=sb, fmt="s", label="boost (1.5x)", alpha=0.8)
        lim = [0, max(tb.max(), pb.max()) * 1.15]
        ax.plot(lim, lim, "k--", lw=0.8, label="y = truth")
        ax.set_xlabel("alpha_dla truth")
        ax.set_ylabel("posterior mean")
        ax.set_title(f"recovery (response slope {resp['slope']:.2f})")
        ax.legend()
        ax = axes[1]
        for i, p in enumerate(PAIRED_PARAMS):
            d = np.asarray([paired_delta_named(rc, rb, p) for rc, rb in zip(clean, boost)],
                           dtype=float)
            ax.scatter(np.full(d.size, i) + np.linspace(-0.15, 0.15, d.size), d, s=14, alpha=0.7)
            m = pooled_all[p]
            ax.errorbar([i], [m["mean"]], yerr=[2 * m["se"]], fmt="D", color="k", capsize=4)
        ax.axhline(0, color="k", lw=0.8)
        for g in (GATE, -GATE):
            ax.axhline(g, color="r", ls=":", lw=0.8)
        ax.set_xticks(range(len(PAIRED_PARAMS)), PAIRED_PARAMS)
        ax.set_ylabel("paired Delta bias_z (sigma_post)")
        ax.set_title("boost - clean paired deltas (gate 0.30)")
        fig.tight_layout()
        fig.savefig(args.fig_out, dpi=150)
        print(f"figure written: {args.fig_out}")


if __name__ == "__main__":
    main()
