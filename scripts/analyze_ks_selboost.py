"""Analyze the KS selection-function mock-challenge shards (spec 2026-07-18 Sec 6 + Amendment 2
A2.6-A2.7, PI-signed 2026-07-19). Arm-resolved paired readout against the shared K0 clean set:

  PART 1 (binding): per-arm in-envelope equivalence — every envelope="in" mis-centered arm
    (K2, K3, K4, K6, K7; K5 iff its measured profile is in-envelope, the pre-registered A2.8-e
    rule) must pass |mean_m Delta bias_z(p)| + t_{n-1,0.975} SE < 0.30 sigma_post for BOTH
    p = A_p and n_s (exact small-n t quantile, reused pool_deltas convention).
  PART 2 (binding): the mis-centered-arm sensitivity number
    S = max over ALL mis-centered arms a and p in {A_p, n_s} of [|mean Delta bias_z| + 2 SE]/D_a,
    D_a = sqrt(mean_z [ln B_a(z)]^2)/sigma_frac_deployed over the kept KS z rows, sigma from the
    STAMPED deployed width (never hard-coded). Proposed threshold S < 0.50. C-arms, if present,
    count in the max (recorded D4) with C2 flagged separately.
  PART 3 (reported): KS-SEL_p = |S_A,p| DA_sel + |S_eta1,p| De1_sel + |S_eta2,p| De2_sel
    + 2 SE_proj at the K4 u_paper coordinates (until a companion count supplies credible
    half-widths), plus the derived (De1_max, De2_max) trade-off line.

  Response surface: WLS of the per-arm paired deltas on the 3-vector design (dA, de1, de2)
  through the origin across surface_fit arms (K5 excluded — subDLA contamination; C2 excluded —
  out-of-envelope corner), jackknife-over-mocks SEs; rank-deficient designs restrict to the
  identifiable columns (fail-soft with the restriction recorded). Collapse checks: per-arm
  residual vs the surface prediction > 2 SE, and the K4-vs-K6 e2-sign symmetry test.

ARM-RESOLVED FAIL-LOUD INGEST (cross-pkl pairing certificate): identical forward_signature /
prior-constants stamp / registry_signature / core run_kw across every pkl; per-pair theta9
truth identity; boosted pivot truth == B_arm(3.0) x clean (rtol 1e-12); row-level
truth_alpha_hcd_z boosted-class columns == B_arm(z_global) x clean columns; every non-boosted
truth entry bit-identical; per-arm mock completeness; stamped B vectors re-verified against the
profiles; smoke pkls filtered. Any violation is a stale or corrupt campaign, not a soft skip.

Pure numpy on LEGACY-era campaigns — login-node safe (profile evaluation via the numpy
registry twin scripts/ks_selboost_arms.eval_profile, hook-agreement test-enforced). A
MAPPED-era campaign lazily imports jax.numpy + dndx_wc inside _mapped_lls_geometry (the
D_exact/reachability companion; x64 is set by the hcd_analysis.emulator package import).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/analyze_ks_selboost.py --shard-dir /scratch/cavestru_root/cavestru1/mfho/ks_selboost \
     [--npz-out out.npz] [--fig-dir figures/analysis/07_ks_selection]
"""
import argparse
import functools
import glob
import os
import pickle

import numpy as np

from scripts.analyze_dla_selfdraw import (GATE, _col_draws_truth, bias_z_named, central_coverage,
                                          paired_delta_named, pool_deltas, rank_stats, rank_u,
                                          verdict)
import scripts.ks_selboost_arms as AR

print = functools.partial(print, flush=True)

S_THRESHOLD = 0.50                    # Part-2 proposed threshold (sigma_post per prior-sigma)
ALPHA = "alpha_lls"
GATE_PARAMS = ("ns", "Ap")            # the binding params
PAIRED_PARAMS = ("ns", "Ap", "tau0_amp", "dtau0")   # tau0/dtau0 MANDATORY reporting
_CLS_TO_COL = {"lls_truth_boost": ("alpha_lls", 0), "subdla_truth_boost": ("alpha_subdla", 1),
               "dla_truth_boost": ("alpha_dla", 2)}
_RUN_KW_CORE = ("n_warmup", "n_samples", "seed", "max_tree_depth", "dense_mass")


# ---------------------------------------------------------------- ingest

def _stamp_tuple(meta):
    """The homogeneity-checked stamp of one pkl: (era, forward sig, prior sig, prior constants,
    registry sig, core run_kw). The era leads the tuple (2026-07-23) so a mixed-era pool is
    diagnosable as such (the full prior_constants equality below would catch it anyway)."""
    fwd = meta["forward"]
    return (stamp_era(meta["prior_constants"]),
            fwd.get("forward_signature"), fwd.get("hcd_prior_signature"),
            tuple(sorted((k, _hashable(v)) for k, v in meta["prior_constants"].items())),
            meta["arm_stamp"]["registry_signature"],
            tuple((k, meta["run_kw"].get(k)) for k in _RUN_KW_CORE))


def _hashable(v):
    """prior_constants values as comparable tuples (the mapped-era geometry stamps are lists)."""
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def stamp_era(prior_constants):
    """The campaign era from the STAMPED parameterization VALUE (2026-07-23, CS design review:
    key-PRESENCE would misclassify an R6 legacy-override pkl, which stamps
    'alpha_pivot_powerlaw_v1' explicitly). Key absent = the pre-stamp 108-fit history =
    legacy."""
    p = (prior_constants or {}).get("hcd_parameterization", "alpha_pivot_powerlaw_v1")
    era = {"dndx_mapped_v2": "mapped", "alpha_pivot_powerlaw_v1": "legacy"}.get(p)
    assert era is not None, f"unknown stamped hcd_parameterization {p!r}"
    return era


def load_campaign(shard_dir):
    """Load the campaign: (clean {mock: rec}, arms {arm_id: {mock: rec}}, meta {arm_id: meta}).
    All ingest asserts fire here (see the module docstring)."""
    paths = sorted(glob.glob(os.path.join(shard_dir, "ks_selboost_*_shard_*.pkl")))
    paths = [p for p in paths if ".smoke" not in os.path.basename(p)]   # F6a: never pool smoke
    assert paths, f"no ks_selboost shard pkls under {shard_dir}"
    clean, arms, meta_by_arm = {}, {}, {}
    ref = None
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        assert d.get("survey") == "ks", f"{p}: survey {d.get('survey')!r} != 'ks'"
        meta = d["meta"]
        # R6 REFUSAL (2026-07-23): R6 paired-comparison pkls (both arms stamp r6_override=True)
        # are consumed ONLY by analyze_r6_pairs.py — they are matched old-vs-new evidence, not
        # campaign members. Their ks_r6_* prefix keeps them out of the glob above; this stamp
        # check catches a mis-named/moved file loudly rather than silently pooling it.
        assert not (meta.get("prior_constants") or {}).get("r6_override"), \
            f"{p}: r6_override pkl in a campaign dir — R6 pkls go to analyze_r6_pairs.py"
        st = _stamp_tuple(meta)
        if ref is None:
            ref = st
        assert st[0] == ref[0], \
            f"{p}: ERA mismatch ({st[0]} vs {ref[0]}) — a mixed-era campaign must not pool " \
            f"(legacy alpha-space and mapped dN/dX pkls are different prior geometries)"
        assert st[1] and st[1] == ref[1], \
            f"{p}: forward_signature mismatch (mixed-forward campaign must not pool)"
        assert st[2] and st[2] == ref[2], \
            f"{p}: hcd_prior_signature mismatch (drifted prior constants)"
        assert st[3] == ref[3], \
            f"{p}: prior_constants stamp mismatch (stale-prior pkl must not pool)"
        assert st[4] and st[4] == ref[4], \
            f"{p}: registry_signature mismatch (mixed-registry campaign must not pool)"
        # the CURRENT registry must equal the stamped one: gate membership (part1/surface_fit)
        # is read from the live registry, so a post-campaign registry edit silently changing
        # gate membership must fail loud here (reviewer-2 hardening; the profiles themselves
        # are additionally protected by the stamped-B re-check below).
        assert st[4] == AR.registry_signature(), \
            f"{p}: stamped registry_signature != the CURRENT scripts/ks_selboost_arms.py " \
            f"registry (the registry drifted after the campaign ran; gate membership would " \
            f"silently change — reconcile deliberately before analyzing)"
        assert st[5] == ref[5], \
            f"{p}: core run_kw mismatch {st[5]} != {ref[5]} (cross-pkl pairing requires " \
            f"identical sampler settings + seed)"
        arm = d["arm"]
        assert len(d["per_mock"]) == len(d["idxs"]), f"{p}: per_mock/idxs length mismatch"
        # re-verify the STAMPED evaluated boost vectors against the stamped profiles (defense
        # in depth: a hand-edited stamp or an evaluator drift fails here).
        stamp = meta["arm_stamp"]
        profiles = stamp["profiles"]
        if profiles is not None:
            zg = np.asarray(stamp["z_global"], float)
            for cls, prof in profiles.items():
                B_re = np.atleast_1d(AR.eval_profile(prof, zg))
                B_st = np.asarray(stamp["B_z_global"][cls], float)
                assert np.allclose(B_st, B_re, rtol=1e-9, atol=0), \
                    f"{p}: stamped B_z_global[{cls}] does not match the stamped profile"
        tgt = clean if arm == "K0_clean" else arms.setdefault(arm, {})
        for m, rec in zip(d["idxs"], d["per_mock"]):
            assert int(m) not in tgt, f"{p}: duplicate mock idx {m} for arm {arm}"
            tgt[int(m)] = rec
        meta_by_arm[arm] = meta
    assert clean, "no K0_clean shards found (the shared pairing baseline is required)"
    # completeness per arm (a partial campaign must not gate silently)
    for arm, meta in meta_by_arm.items():
        got = set(clean) if arm == "K0_clean" else set(arms[arm])
        want = set(range(int(meta["n_mocks"])))
        assert got == want, (
            f"campaign incomplete for {arm}: missing mock idxs {sorted(want - got)} "
            f"(got {len(got)}/{len(want)})")
    # cross-pkl pairing contract per (arm, mock)
    for arm, recs in arms.items():
        stamp = meta_by_arm[arm]["arm_stamp"]
        zg = np.asarray(stamp["z_global"], float)
        boosted = {cls: (np.atleast_1d(AR.eval_profile(prof, zg)),
                         float(AR.eval_profile(prof, AR.Z_PIVOT)))
                   for cls, prof in stamp["profiles"].items()}
        for m, rb in recs.items():
            assert m in clean, f"{arm} mock {m}: no clean partner"
            rc = clean[m]
            names = list(rc["names"])
            assert names == list(rb["names"]), f"{arm} mock {m}: names mismatch"
            tc, tb = np.asarray(rc["truth_vec"], float), np.asarray(rb["truth_vec"], float)
            # theta9 identity: the same-seed truth draw certificate (theta block = the leading
            # entries before the first tau0 ladder entry — robust to optional trailing
            # latent/metal columns in the packed layout).
            n_theta = next((i for i, n in enumerate(names) if n.startswith("tau0_z")),
                           len(names) - 3)
            assert np.array_equal(tc[:n_theta], tb[:n_theta]), \
                f"{arm} mock {m}: theta truth differs from the clean partner (pairing broken)"
            # pivot contract per boosted class; all other truth entries bit-identical
            keep = np.ones(len(tc), bool)
            for cls, (Bz, B3) in boosted.items():
                nm, col = _CLS_TO_COL[cls]
                j = names.index(nm)
                keep[j] = False
                assert np.isclose(tb[j], B3 * tc[j], rtol=1e-12, atol=0), \
                    f"{arm} mock {m}: boosted {nm} pivot truth {tb[j]} != B(3)={B3} x clean {tc[j]}"
            assert np.array_equal(tc[keep], tb[keep]), \
                f"{arm} mock {m}: non-boosted truth entries differ between arms (pairing broken)"
            # row-level contract on the z-resolved truth
            azc = np.asarray(rc["truth_alpha_hcd_z"], float)
            azb = np.asarray(rb["truth_alpha_hcd_z"], float)
            assert azc.shape == azb.shape == (zg.size, 3), \
                f"{arm} mock {m}: truth_alpha_hcd_z shape {azb.shape} != ({zg.size}, 3)"
            boosted_cols = []
            for cls, (Bz, B3) in boosted.items():
                _, col = _CLS_TO_COL[cls]
                boosted_cols.append(col)
                assert np.allclose(azb[:, col], Bz * azc[:, col], rtol=1e-12, atol=0), \
                    f"{arm} mock {m}: row contract violated on column {col} " \
                    f"(truth rows != B(z_global) x clean rows)"
            for col in set(range(3)) - set(boosted_cols):
                assert np.array_equal(azb[:, col], azc[:, col]), \
                    f"{arm} mock {m}: non-boosted truth_alpha_hcd_z column {col} differs " \
                    f"(row contract violated)"
    return clean, arms, meta_by_arm


# ---------------------------------------------------------------- design coordinates

def design_coords(lnB, z, w=None):
    """WLS of ln B(z) on {1, x, x^2}, x = ln((1+z)/(1+z_pivot)) (A2.6: quadratic basis).
    Returns (dA, de1, de2, r_rms) with r_rms the weighted RMS residual after the quadratic
    (the out-of-family curvature coordinate). Uniform weights unless ``w`` given (the Fisher
    per-z weights are an optional readout input; uniform numbers are INDICATIVE, A2.2)."""
    z = np.asarray(z, float)
    lnB = np.asarray(lnB, float)
    w = np.ones_like(z) if w is None else np.asarray(w, float)
    x = np.log((1.0 + z) / (1.0 + AR.Z_PIVOT))
    X = np.stack([np.ones_like(x), x, x * x], axis=1)
    W = w / w.sum()
    A = X.T @ (W[:, None] * X)
    b = X.T @ (W * lnB)
    beta = np.linalg.solve(A, b)
    resid = lnB - X @ beta
    r_rms = float(np.sqrt((W * resid ** 2).sum()))
    return float(beta[0]), float(beta[1]), float(beta[2]), r_rms


def displacement_D(lnB, sigma_frac, w=None):
    """D_a = sqrt(weighted mean of [ln B(z)]^2) / sigma_frac — the truth displacement in
    deployed-prior-sigma units (reduces to |ln b|/sigma for flat arms; never hard-coded)."""
    lnB = np.asarray(lnB, float)
    w = np.ones_like(lnB) if w is None else np.asarray(w, float)
    return float(np.sqrt((w * lnB ** 2).sum() / w.sum()) / float(sigma_frac))


def _ks_band_rows(z):
    lo, hi = AR.KS_Z_RANGE
    return (np.asarray(z, float) >= lo - 1e-9) & (np.asarray(z, float) <= hi + 1e-9)


def _mapped_lls_geometry(pc, zg, sel):
    """MAPPED-era LLS geometry from the STAMPED reference (2026-07-23, Bayesian design review
    Q2/Q5.1): per kept-z-row (sigma_lnalpha, alpha_centre, alpha_ceiling), all in alpha space.
    sigma_lnalpha(z) = g(z) * sqrt(sigma_eps^2 + x^2 sigma_kappa^2) with g(z) the finite-
    difference d ln alpha_LLS / d ln A through the exact occupancy map at the stamped reference
    (x = ln((1+z)/(1+3))); the ceiling is the map's saturation alpha at amplitude e^12."""
    import jax.numpy as jnp
    from hcd_analysis.emulator.dndx_wc import w_c_corrected
    ref = np.asarray(pc["ks_dndx_ref_z"], float)[sel]        # (n,3)
    xbar = np.asarray(pc["ks_xbar_z"], float)[sel]           # (n,)
    z = np.asarray(zg, float)[sel]
    s_eps = float(pc["ks_dndx_sigma_eps"])
    s_kap = float(pc["ks_dndx_sigma_kappa"])

    def _alpha_lls(amp_factor):
        d = ref * np.array([amp_factor, 1.0, 1.0])
        out = w_c_corrected(jnp.asarray(d), jnp.asarray(xbar), jnp.asarray(z))
        return np.asarray(out, float)[:, 1]
    h = 0.05
    centre = _alpha_lls(1.0)
    g = (np.log(_alpha_lls(np.exp(h))) - np.log(_alpha_lls(np.exp(-h)))) / (2.0 * h)
    x = np.log((1.0 + z) / 4.0)
    sigma_ln = g * np.sqrt(s_eps ** 2 + (x * s_kap) ** 2)
    ceiling = _alpha_lls(np.exp(12.0))
    return sigma_ln, centre, ceiling


def arm_coordinates(meta_arm, w=None):
    """Design coordinates + displacement of one arm from its STAMPS: the LLS boost vector on
    the kept KS z rows (z_global restricted to [2.4, 4.6]) and the stamped deployed fractional
    width. Returns dict(n_z, dA, de1, de2, r, D, z, lnB, in_envelope, era, D_exact,
    unreachable_z).

    D SEMANTICS BY ERA (2026-07-23, design pair): D = |lnB|_rms / lls_frac_sigma_ks (0.40) in
    BOTH eras — the truth boosts are alpha-space multiplicative in both, and in the mapped era
    0.40 = g_LLS(z=3) x sigma_eps(0.5310) to <1e-4 relative (pinned-literal quantization of
    the sigma_eps width-translation construction; domain-review-measured 0.399993), so 0.40
    is the correct pivot-linearized alpha-space width there too. The
    gate statistic keeps this D for cross-era comparability with the 108-fit history. The
    mapped era ADDS the companion D_exact (per-z sigma_lnalpha(z): tilt width + map-saturation
    g(z) both included) and the truth-REACHABILITY flag (a boosted centre beyond the occupancy
    ceiling is a structural unreachability, not a sensitivity statement); a gate verdict that
    flips between D and D_exact normalization is a NEEDS-PI readout item, not an analyzer
    default. Legacy era: D_exact=None, unreachable_z=None (additive keys only)."""
    stamp = meta_arm["arm_stamp"]
    zg = np.asarray(stamp["z_global"], float)
    sel = _ks_band_rows(zg)
    z = zg[sel]
    B = np.asarray(stamp["B_z_global"]["lls_truth_boost"], float)[sel]
    lnB = np.log(B)
    pc = meta_arm["prior_constants"]
    sigma_frac = float(pc["lls_frac_sigma_ks"])
    dA, de1, de2, r = design_coords(lnB, z, w)
    lo, hi = AR.ENVELOPE_REL
    in_env = True
    for cls, Bv in stamp["B_z_global"].items():
        Bv = np.asarray(Bv, float)[sel]
        in_env &= bool(np.all(Bv >= lo - 1e-9) and np.all(Bv <= hi + 1e-9))
    era = stamp_era(pc)
    D_exact, unreachable_z = None, None
    if era == "mapped":
        sigma_ln, centre, ceiling = _mapped_lls_geometry(pc, zg, sel)
        wv = np.ones_like(lnB) if w is None else np.asarray(w, float)
        D_exact = float(np.sqrt((wv * (lnB / sigma_ln) ** 2).sum() / wv.sum()))
        unreachable_z = [float(zz) for zz, bc, ce in zip(z, B * centre, ceiling)
                         if bc > ce * (1.0 - 1e-9)]
    return dict(n_z=int(sel.sum()), dA=dA, de1=de1, de2=de2, r=r,
                D=displacement_D(lnB, sigma_frac, w), z=z, lnB=lnB, in_envelope=in_env,
                era=era, D_exact=D_exact, unreachable_z=unreachable_z)


def _part1_member(arm, coords):
    """Part-1 gate-set membership (A2.6/A2.8-e): the registry part1 flag; 'if_in_envelope'
    (K5) resolves from the STAMPED profile at readout."""
    flag = AR.ARMS[arm]["part1"] if arm in AR.ARMS else True
    if flag == "if_in_envelope":
        return coords["in_envelope"]
    return bool(flag)


# ---------------------------------------------------------------- per-arm stats

def _pairs(clean, recs):
    return [(clean[m], recs[m]) for m in sorted(recs)]


def response_slope_named(pairs, name):
    """Displaced-truth response slope for a named site (pivot-slope; jackknife SE) — the
    parameterized twin of analyze_dla_selfdraw.response_slope. For a near-1 pivot boost
    (|sum Delta truth| ~ 0) the pivot slope is ill-defined -> NaN with the shape-arm caveat."""
    dmean, dtruth = [], []
    for rc, rb in pairs:
        dc, tc = _col_draws_truth(rc, name)
        db, tb = _col_draws_truth(rb, name)
        dmean.append(float(db.mean()) - float(dc.mean()))
        dtruth.append(tb - tc)
    dmean, dtruth = np.asarray(dmean), np.asarray(dtruth)
    n = dmean.size
    denom = dtruth.sum()
    if abs(denom) < 1e-12 * max(1.0, np.abs(dtruth).max() * n):
        return dict(slope=float("nan"), slope_se=float("nan"), dmean=dmean, dtruth=dtruth,
                    caveat="pivot displacement ~0 (shape-only arm): pivot-slope undefined")
    slope = float(dmean.sum() / denom)
    if n > 1:
        loo = np.asarray([(dmean.sum() - dmean[i]) / (dtruth.sum() - dtruth[i])
                          for i in range(n)])
        se = float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))
    else:
        se = float("nan")
    return dict(slope=slope, slope_se=se, dmean=dmean, dtruth=dtruth, caveat=None)


def paired_delta_meanconv(rec_clean, rec_boost, name):
    """The SECOND bias_z convention (spec Sec 6): (mean_b - mean_c)/sd_c (clean-arm sd
    normalization), reported alongside the own-sd paired delta."""
    dc, _ = _col_draws_truth(rec_clean, name)
    db, _ = _col_draws_truth(rec_boost, name)
    sd = float(dc.std())
    if sd <= 0:
        return None
    return (float(db.mean()) - float(dc.mean())) / sd


# ---------------------------------------------------------------- pooled surface

def fit_surface(deltas_by_arm, coords_by_arm, arms_fit):
    """WLS fit of Delta bias_z = S_A dA + S_e1 de1 + S_e2 de2 through the origin across the
    per-mock deltas of ``arms_fit`` (K5/C2 excluded upstream). Rank-deficient designs restrict
    to the identifiable columns (recorded in 'cols'). Jackknife-over-mocks SEs (delete one
    SHARED mock id across every arm — the paired deltas of one mock are correlated through the
    shared clean fit)."""
    rows, obs, mock_of = [], [], []
    for a in arms_fit:
        v = (coords_by_arm[a]["dA"], coords_by_arm[a]["de1"], coords_by_arm[a]["de2"])
        for m, dlt in deltas_by_arm[a].items():
            if dlt is None:
                continue
            rows.append(v)
            obs.append(dlt)
            mock_of.append(m)
    if not rows:
        return None
    X = np.asarray(rows, float)
    y = np.asarray(obs, float)
    # identifiable columns: nonzero design AND full rank after restriction
    cols = [j for j in range(3) if np.abs(X[:, j]).max() > 1e-9]
    Xr = X[:, cols]
    while cols and np.linalg.matrix_rank(Xr) < len(cols):
        cols = cols[:-1]
        Xr = X[:, cols]

    def _solve(Xr_, y_):
        beta_, *_ = np.linalg.lstsq(Xr_, y_, rcond=None)
        return beta_

    beta = _solve(Xr, y)
    mock_ids = sorted(set(mock_of))
    loo = []
    for m in mock_ids:
        keep = np.asarray([mm != m for mm in mock_of])
        if keep.sum() >= len(cols):
            loo.append(_solve(Xr[keep], y[keep]))
    se = np.full(len(cols), np.nan)
    if len(loo) > 1:
        loo = np.asarray(loo)
        g = len(loo)
        se = np.sqrt((g - 1) / g * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0))
    names = ["S_A", "S_eta1", "S_eta2"]
    out = {names[j]: 0.0 for j in range(3)}
    out_se = {f"{names[j]}_se": float("nan") for j in range(3)}
    for i, j in enumerate(cols):
        out[names[j]] = float(beta[i])
        out_se[f"{names[j]}_se"] = float(se[i])
    pred = {a: float(np.dot([coords_by_arm[a]["dA"], coords_by_arm[a]["de1"],
                             coords_by_arm[a]["de2"]],
                            [out["S_A"], out["S_eta1"], out["S_eta2"]])) for a in arms_fit}
    return dict(**out, **out_se, cols=[["S_A", "S_eta1", "S_eta2"][j] for j in cols],
                n_obs=int(y.size), pred_by_arm=pred)


# ---------------------------------------------------------------- campaign summary

def summarize_campaign(shard_dir, w=None):
    """The full numeric readout as a dict: ingest + per-arm gates + Part 1/2/3 + surface +
    collapse checks + diagnostics. main_report prints it."""
    clean, arms, meta = load_campaign(shard_dir)
    coords = {a: arm_coordinates(meta[a], w) for a in arms}
    sigma_frac = float(meta[next(iter(meta))]["prior_constants"]["lls_frac_sigma_ks"])

    deltas = {a: {p: {m: paired_delta_named(clean[m], recs[m], p) for m in sorted(recs)}
                  for p in PAIRED_PARAMS} for a, recs in arms.items()}
    pooled = {a: {p: pool_deltas(list(deltas[a][p].values()),
                                 weights=[min(clean[m]["L"], arms[a][m]["L"])
                                          for m in sorted(arms[a])])
                  for p in PAIRED_PARAMS} for a in arms}

    # PART 1: binding per-arm in-envelope equivalence gates (exact t-quantile ub)
    part1 = {}
    for a in arms:
        if not _part1_member(a, coords[a]):
            continue
        part1[a] = {p: dict(pooled[a][p], verdict=verdict(pooled[a][p]["ub"]))
                    for p in GATE_PARAMS}

    # PART 2: S = max over the BINDING mis-centered arms of (|mean| + 2 SE)/D. FLAG A
    # RESOLUTION (spec launch log, PI-signed A2.8-c): the null-bound arm K5 is EXCLUDED from
    # the binding max (registry part2_binding=False) — at D~0.13 its N=8 noise floor 2SE/D~1.3
    # makes it mechanically un-failable/un-passable on noise alone; it reports a SEPARATE
    # null-bound consistency line. This lands the signed decision in code (the earlier "loud
    # flag / PI disposition" wording is superseded; readout 2026-07-20).
    def _is_binding(a):
        return bool(AR.ARMS[a].get("part2_binding", True)) if a in AR.ARMS else True

    per_arm2, S, argmax = {}, 0.0, None
    null_bound = {}
    for a in arms:
        D = coords[a]["D"]
        binding = _is_binding(a)
        entry = dict(D=D, flagged_out_of_envelope=(not coords[a]["in_envelope"]),
                     part2_binding=binding)
        for p in GATE_PARAMS:
            pl = pooled[a][p]
            term = (abs(pl["mean"]) + 2.0 * pl["se"]) / D if D > 0 else float("nan")
            entry[p] = term
            if binding and np.isfinite(term) and term > S:
                S, argmax = term, (a, p)
        entry["noise_floor_2se_over_D"] = max(
            (2.0 * pooled[a][p]["se"] / D if D > 0 else float("nan")) for p in GATE_PARAMS)
        entry["noise_dominated"] = bool(np.isfinite(entry["noise_floor_2se_over_D"])
                                        and entry["noise_floor_2se_over_D"] >= S_THRESHOLD)
        per_arm2[a] = entry
        if not binding:
            # null-bound consistency line (reported, NOT gated): raw mean +/- 2SE per param
            null_bound[a] = {p: dict(mean=float(pooled[a][p]["mean"]),
                                     se=float(pooled[a][p]["se"]),
                                     within_2se_of_zero=bool(abs(pooled[a][p]["mean"])
                                                             <= 2.0 * pooled[a][p]["se"]))
                             for p in GATE_PARAMS}
    # raw all-arms max (transparency only; NOT the criterion) so the ill-conditioned K5 term
    # is visible but never the headline.
    _raw_terms = [(per_arm2[a][p], a, p) for a in arms for p in GATE_PARAMS
                  if np.isfinite(per_arm2[a][p])]
    S_raw, argmax_raw = max(((t, (a, p)) for t, a, p in _raw_terms), default=(0.0, None))
    # COMPANION S under D_exact (mapped era only; REPORTED, never the gate — design pair
    # 2026-07-23): same max, D_exact normalization. A verdict flip between S and S_exact
    # across the 0.50 threshold is a NEEDS-PI readout item.
    S_exact, argmax_exact = None, None
    if any(coords[a].get("D_exact") is not None for a in arms):
        S_exact = 0.0
        for a in arms:
            De = coords[a].get("D_exact")
            if De is None or not _is_binding(a):
                continue
            for p in GATE_PARAMS:
                pl = pooled[a][p]
                t = (abs(pl["mean"]) + 2.0 * pl["se"]) / De if De > 0 else float("nan")
                if np.isfinite(t) and t > S_exact:
                    S_exact, argmax_exact = float(t), (a, p)
    part2 = dict(S=float(S), argmax=argmax, threshold=S_THRESHOLD, per_arm=per_arm2,
                 verdict=("PASS" if S < S_THRESHOLD else "FAIL"),
                 S_raw_allarms=float(S_raw), argmax_raw=argmax_raw,
                 null_bound_line=null_bound,
                 S_exact=S_exact, argmax_exact=argmax_exact,
                 verdict_flip_needs_pi=bool(S_exact is not None
                                            and (S < S_THRESHOLD) != (S_exact < S_THRESHOLD)),
                 noise_dominated_arms=[a for a, e in per_arm2.items() if e["noise_dominated"]])

    # pooled 3-vector response surface (surface_fit arms only: K5 excluded — its subDLA
    # component would contaminate the LLS surface; C2 excluded — out-of-envelope corner)
    arms_fit = [a for a in arms
                if (AR.ARMS[a]["surface_fit"] if a in AR.ARMS else True)]
    surface = {p: (fit_surface({a: deltas[a][p] for a in arms_fit}, coords, arms_fit)
                   if arms_fit else None)
               for p in GATE_PARAMS}

    # collapse checks (A2.6): per-arm residual vs prediction; e2 sign symmetry needs K4+K6
    collapse = {}
    for p in GATE_PARAMS:
        sf = surface[p]
        if sf is None:
            continue
        resid = {}
        for a in arms_fit:
            pl = pooled[a][p]
            resid[a] = dict(resid=pl["mean"] - sf["pred_by_arm"][a], se=pl["se"],
                            exceeds_2se=abs(pl["mean"] - sf["pred_by_arm"][a]) > 2 * pl["se"])
        e2sym = None
        if "K4_u_paper" in arms and "K6_inv_u" in arms:
            vals = {}
            for a in ("K4_u_paper", "K6_inv_u"):
                de2 = coords[a]["de2"]
                base = sf["S_A"] * coords[a]["dA"] + sf["S_eta1"] * coords[a]["de1"]
                vals[a] = dict(S_e2=(pooled[a][p]["mean"] - base) / de2,
                               se=pooled[a][p]["se"] / abs(de2))
            diff = abs(vals["K4_u_paper"]["S_e2"] - vals["K6_inv_u"]["S_e2"])
            sig = np.hypot(vals["K4_u_paper"]["se"], vals["K6_inv_u"]["se"])
            e2sym = dict(vals=vals, diff=diff, se=float(sig), nonlinear=bool(diff > 2 * sig))
        collapse[p] = dict(resid=resid, e2_symmetry=e2sym,
                           demote=any(r["exceeds_2se"] for r in resid.values())
                           or bool(e2sym and e2sym["nonlinear"]))

    # PART 3: projection at the K4 coordinates (or unavailable)
    part3 = {}
    sel_arm = "K4_u_paper" if "K4_u_paper" in coords else None
    for p in GATE_PARAMS:
        sf = surface[p]
        if sf is None or sel_arm is None:
            part3[p] = None
            continue
        c4 = coords[sel_arm]
        se_proj = np.nansum([abs(c4["dA"]) * sf["S_A_se"], abs(c4["de1"]) * sf["S_eta1_se"],
                             abs(c4["de2"]) * sf["S_eta2_se"]])
        ks_sel = (abs(sf["S_A"]) * abs(c4["dA"]) + abs(sf["S_eta1"]) * abs(c4["de1"])
                  + abs(sf["S_eta2"]) * abs(c4["de2"]) + 2 * se_proj)
        # derived trade-off (De1, De2) budget line: 0.30 = |S_A| DA_sel + |S_e1| De1 + |S_e2| De2
        room = GATE - abs(sf["S_A"]) * abs(c4["dA"]) - 2 * se_proj
        part3[p] = dict(sel_arm=sel_arm, DA_sel=c4["dA"], De1_sel=c4["de1"], De2_sel=c4["de2"],
                        KS_SEL=float(ks_sel), se_proj=float(se_proj), budget=GATE,
                        room_after_amplitude=float(room),
                        De1_max_at_De2_0=(float(room / abs(sf["S_eta1"]))
                                          if sf["S_eta1"] not in (0.0,) else float("nan")),
                        De2_max_at_De1_0=(float(room / abs(sf["S_eta2"]))
                                          if sf["S_eta2"] not in (0.0,) else float("nan")))

    era = stamp_era(meta[next(iter(meta))]["prior_constants"])
    _pc0 = meta[next(iter(meta))]["prior_constants"]
    width_convention = (
        f"alpha-space sigma/mu = {sigma_frac} (stamped lls_frac_sigma_ks)" if era == "legacy"
        else f"pivot-linearized alpha-space {sigma_frac} = g_LLS(3) x sigma_eps"
             f"({_pc0.get('ks_dndx_sigma_eps')}) to <1e-4; exponent width kappa="
             f"{_pc0.get('ks_dndx_sigma_kappa')} NOT in D (see D_exact companion)")
    return dict(clean=clean, arms=arms, meta=meta, coords=coords, sigma_frac=sigma_frac,
                deltas=deltas, pooled=pooled, part1=part1, part2=part2, part3=part3,
                surface=surface, collapse=collapse,
                era=era, width_convention=width_convention)


# ---------------------------------------------------------------- report

def _diag_lines(clean, arms):
    lines = []
    Lc = np.asarray([clean[m]["L"] for m in sorted(clean)])
    div = sum(int(clean[m]["n_div"]) for m in clean)
    us = [rank_u(clean[m], ALPHA) for m in sorted(clean)]
    cov = central_coverage(us)
    rs = rank_stats(us)
    lines.append(f"clean arm: L {Lc.min()}..{Lc.max()}, divergent draws {div}; alpha_lls rank "
                 f"u mean {rs['mean_u']:.3f} (z {rs['z']:+.2f}), coverage "
                 f"{cov['in68']}/{cov['n']} c68 {cov['in95']}/{cov['n']} c95 "
                 f"(low-u depletion watch, spec risk 11)")
    for a, recs in sorted(arms.items()):
        Lb = np.asarray([recs[m]["L"] for m in sorted(recs)])
        div = sum(int(recs[m]["n_div"]) for m in recs)
        rsb = rank_stats([rank_u(recs[m], ALPHA) for m in sorted(recs)])
        lines.append(f"{a}: N={len(recs)} L {Lb.min()}..{Lb.max()} divergent draws {div}; "
                     f"alpha_lls rank u mean {rsb['mean_u']:.3f} (displaced truth: NOT an SBC "
                     f"statement, a depletion watch)")
    return lines


def _alias_lines(clean, arms):
    lines = ["alias diagnostics corr(alpha_lls; .) per arm (read f_res BEFORE interpreting an "
             "n_s delta as a selection penalty — spec risk 7):"]
    for a, recs in sorted(arms.items()):
        cs = {}
        for p in ("tau0_amp", "Ap", "ns", "f_res_amp"):
            vals = []
            for m in recs:
                try:
                    da, _ = _col_draws_truth(recs[m], ALPHA)
                    db, _ = _col_draws_truth(recs[m], p)
                except (KeyError, ValueError):
                    continue
                if da.std() > 0 and db.std() > 0:
                    vals.append(float(np.corrcoef(da, db)[0, 1]))
            cs[p] = np.mean(vals) if vals else np.nan
        lines.append("   " + a + ": " + "  ".join(f"{p} {cs[p]:+.3f}" for p in cs))
    return lines


def main_report(shard_dir, npz_out=None, fig_dir=None, w=None):
    res = summarize_campaign(shard_dir, w)
    clean, arms, meta = res["clean"], res["arms"], res["meta"]
    print(f"== KS selection-boost analyzer: {len(clean)} clean mocks, "
          f"{sum(len(r) for r in arms.values())} boosted fits over {len(arms)} arms; "
          f"forward_signature "
          f"{meta[next(iter(meta))]['forward']['forward_signature'][:16]}... ==")
    print(f"ERA: {res['era'].upper()}  (width convention: {res['width_convention']})")
    _wline = (f"stamped deployed LLS width sigma/mu = {res['sigma_frac']}"
              if res["era"] == "legacy" else
              f"D normalization {res['sigma_frac']} (= g_LLS(3) x sigma_eps; the DEPLOYED "
              f"mapped width is sigma_eps in log-dN/dX)")
    print(f"{_wline} (all D_a in these units); "
          f"z-weights: {'uniform (INDICATIVE, A2.2)' if w is None else 'Fisher-supplied'}")
    if res["era"] == "mapped":
        _p2 = res["part2"]
        print(f"   mapped-era companion: S_exact={_p2['S_exact']} (argmax {_p2['argmax_exact']}) "
              f"vs gate S={_p2['S']:.3f}"
              + ("  ** VERDICT FLIP vs D_exact -> NEEDS-PI **" if _p2["verdict_flip_needs_pi"]
                 else ""))
        for a in sorted(res["coords"]):
            uz = res["coords"][a].get("unreachable_z")
            if uz:
                print(f"   [reachability] arm {a}: boosted truth centre BEYOND the occupancy "
                      f"ceiling (at reference sub/DLA occupancy) at z={uz} — a FAIL here is "
                      f"structural unreachability, not sensitivity")
    for ln in _diag_lines(clean, arms):
        print("   " + ln)

    print("\n-- design coordinates (WLS on {1, x, x^2} over the kept KS z rows)")
    for a in sorted(arms):
        c = res["coords"][a]
        print(f"   {a:>15}: dA {c['dA']:+.3f}  de1 {c['de1']:+.3f}  de2 {c['de2']:+.3f}  "
              f"r {c['r']:.3f}  D {c['D']:.3f}  envelope "
              f"{'in' if c['in_envelope'] else 'OUT'}")

    print("\n-- alpha_lls displaced-truth response per arm (pivot-slope; shape arms carry a "
          "caveat)")
    for a in sorted(arms):
        rsl = response_slope_named(_pairs(clean, arms[a]), ALPHA)
        cav = f"  [{rsl['caveat']}]" if rsl["caveat"] else ""
        print(f"   {a:>15}: slope {rsl['slope']:.3f} +/- {rsl['slope_se']:.3f}{cav}")

    print(f"\n-- PART 1 (binding): per-arm in-envelope equivalence, |mean|+t SE < {GATE} "
          f"sigma_post; both conventions reported")
    any_fail = False
    for a in sorted(res["part1"]):
        for p in GATE_PARAMS:
            g = res["part1"][a][p]
            mc = [paired_delta_meanconv(clean[m], arms[a][m], p) for m in sorted(arms[a])]
            mc = np.asarray([x for x in mc if x is not None], float)
            any_fail |= g["verdict"] == "FAIL"
            print(f"   {a:>15} {p:>3}: mean {g['mean']:+.3f} +/- {g['se']:.3f} "
                  f"(sd_c-conv {np.mean(mc):+.3f}) ub {g['ub']:.3f} {g['verdict']}")
    print("   tau0/dtau0 (mandatory reporting, not gated):")
    for a in sorted(arms):
        g1, g2 = res["pooled"][a]["tau0_amp"], res["pooled"][a]["dtau0"]
        print(f"   {a:>15} tau0_amp {g1['mean']:+.3f}+/-{g1['se']:.3f}  "
              f"dtau0 {g2['mean']:+.3f}+/-{g2['se']:.3f}")

    p2 = res["part2"]
    print(f"\n-- PART 2 (binding): S = max (|mean|+2SE)/D over the BINDING mis-centered arms "
          f"(null-bound K5 EXCLUDED per the PI-signed FLAG A resolution) = "
          f"{p2['S']:.3f} at {p2['argmax']} (threshold {p2['threshold']}) {p2['verdict']}")
    print(f"   (raw all-arms max including the ill-conditioned K5 term = {p2['S_raw_allarms']:.3f} "
          f"at {p2['argmax_raw']} — reported for transparency, NOT the criterion)")
    for a in sorted(p2["per_arm"]):
        e = p2["per_arm"][a]
        flag = "  [OUT-OF-ENVELOPE, flagged separately]" if e["flagged_out_of_envelope"] else ""
        if not e["part2_binding"]:
            flag += ("  [NULL-BOUND arm — EXCLUDED from the binding max (FLAG A, signed); "
                     "sensitivity reported on the null-bound line below]")
        elif e["noise_dominated"]:
            flag += (f"  [noise-floor note: 2SE/D = {e['noise_floor_2se_over_D']:.2f}]")
        print(f"   {a:>15}: D {e['D']:.3f}  ns {e['ns']:.3f}  Ap {e['Ap']:.3f}{flag}")
    if p2["null_bound_line"]:
        for a, nb in p2["null_bound_line"].items():
            bits = "  ".join(f"{p} {nb[p]['mean']:+.3f}+/-{nb[p]['se']:.3f}"
                             f"{'(<2SE of 0)' if nb[p]['within_2se_of_zero'] else '(>2SE!)'}"
                             for p in GATE_PARAMS)
            print(f"   NULL-BOUND consistency line [{a}]: {bits}  "
                  f"(reported, not gated; near-null displacement D~{p2['per_arm'][a]['D']:.2f})")
    print("   [DISCLOSURE] weights = uniform INDICATIVE (spec FLAG B: the Fisher-weight "
          "generator is not built; D_a/S/surface are indicative. Part 1 is weight-independent, "
          "so the FAIL does not depend on this.)")

    print("\n-- response surface (through-origin WLS; jackknife-over-mocks SE) + collapse")
    for p in GATE_PARAMS:
        sf = res["surface"][p]
        if sf is None:
            print(f"   {p}: no surface (no arms)")
            continue
        print(f"   {p}: S_A {sf['S_A']:+.3f}+/-{sf['S_A_se']:.3f}  "
              f"S_eta1 {sf['S_eta1']:+.4f}+/-{sf['S_eta1_se']:.4f}  "
              f"S_eta2 {sf['S_eta2']:+.5f}+/-{sf['S_eta2_se']:.5f}  "
              f"(identifiable cols: {sf['cols']}, n_obs {sf['n_obs']})")
        col = res["collapse"].get(p)
        if col:
            worst = max(col["resid"].values(), key=lambda r: abs(r["resid"]))
            print(f"      collapse: demote={col['demote']} (worst residual {worst['resid']:+.3f}"
                  f" vs 2SE {2 * worst['se']:.3f}; e2-symmetry "
                  f"{col['e2_symmetry'] if col['e2_symmetry'] else 'needs K4+K6'})")

    print("\n-- PART 3 (reported): projected credible-error at the K4 u_paper coordinates")
    for p in GATE_PARAMS:
        p3 = res["part3"][p]
        if p3 is None:
            print(f"   {p}: not computable (K4 arm absent from this pool)")
            continue
        print(f"   {p}: KS-SEL = {p3['KS_SEL']:.3f} (budget {p3['budget']}; se_proj "
              f"{p3['se_proj']:.3f}); trade-off De1_max@De2=0 {p3['De1_max_at_De2_0']:.2f}, "
              f"De2_max@De1=0 {p3['De2_max_at_De1_0']:.2f}")

    for ln in _alias_lines(clean, arms):
        print(ln)

    print("\n== OVERALL:",
          "FAIL — a binding gate exceeded its budget (Part 1 and Part 2 are BOTH required for "
          "KS unblinding; see the pre-registered failure branch, spec Sec 6)."
          if (any_fail or p2["verdict"] == "FAIL") else
          "PASS — every Part-1 in-envelope arm is bounded within the 0.30 equivalence gate and "
          "the Part-2 sensitivity S is below 0.50 sigma_post per prior-sigma.")

    if npz_out:
        arms_sorted = sorted(arms)
        np.savez(
            npz_out,
            arms=np.asarray(arms_sorted),
            D=np.asarray([res["coords"][a]["D"] for a in arms_sorted]),
            dA=np.asarray([res["coords"][a]["dA"] for a in arms_sorted]),
            de1=np.asarray([res["coords"][a]["de1"] for a in arms_sorted]),
            de2=np.asarray([res["coords"][a]["de2"] for a in arms_sorted]),
            **{f"mean_{p}": np.asarray([res["pooled"][a][p]["mean"] for a in arms_sorted])
               for p in PAIRED_PARAMS},
            **{f"se_{p}": np.asarray([res["pooled"][a][p]["se"] for a in arms_sorted])
               for p in PAIRED_PARAMS},
            **{f"ub_{p}": np.asarray([res["pooled"][a][p]["ub"] for a in arms_sorted])
               for p in PAIRED_PARAMS},
            S=p2["S"],                         # the BINDING S (null-bound K5 excluded, signed)
            S_argmax=np.asarray(str(p2["argmax"])),
            S_raw_allarms=p2["S_raw_allarms"],  # transparency: includes the ill-conditioned K5
            part2_binding=np.asarray([res["part2"]["per_arm"][a]["part2_binding"]
                                      for a in arms_sorted]),
            weights_mode=np.asarray("uniform_INDICATIVE_flagB"),
        )
        print(f"npz written: {npz_out}")

    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arms_sorted = sorted(arms)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        for i, a in enumerate(arms_sorted):
            for j, p in enumerate(GATE_PARAMS):
                d = np.asarray([v for v in res["deltas"][a][p].values() if v is not None])
                xx = i + (j - 0.5) * 0.25
                ax.scatter(np.full(d.size, xx) + np.linspace(-0.06, 0.06, d.size), d, s=10,
                           alpha=0.6, color=f"C{j}", label=p if i == 0 else None)
                m = res["pooled"][a][p]
                ax.errorbar([xx], [m["mean"]], yerr=[2 * m["se"]], fmt="D", color="k",
                            capsize=3, ms=4)
        for g in (GATE, -GATE):
            ax.axhline(g, color="r", ls=":", lw=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(arms_sorted)), arms_sorted, rotation=30, ha="right")
        ax.set_ylabel("paired Delta bias_z (sigma_post)")
        ax.set_title("per-arm paired deltas (gate 0.30)")
        ax.legend()
        ax = axes[1]
        D = [res["coords"][a]["D"] for a in arms_sorted]
        for j, p in enumerate(GATE_PARAMS):
            mm = [res["pooled"][a][p]["mean"] for a in arms_sorted]
            ss = [res["pooled"][a][p]["se"] for a in arms_sorted]
            ax.errorbar(D, mm, yerr=ss, fmt="o", color=f"C{j}", label=p, alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("displacement D_a (prior-sigma units)")
        ax.set_ylabel("mean paired Delta bias_z")
        ax.set_title(f"campaign response (Part-2 S = {p2['S']:.3f})")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(fig_dir, "ks_selboost_campaign_summary.png")
        fig.savefig(out, dpi=150)
        print(f"figure written: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/scratch/cavestru_root/cavestru1/mfho/ks_selboost")
    ap.add_argument("--npz-out", default=None)
    ap.add_argument("--fig-dir", default=None,
                    help="e.g. figures/analysis/07_ks_selection (mock closure: committable)")
    ap.add_argument("--fisher-weights-npz", default=None,
                    help="optional per-z Fisher weights npz (arrays z, w) from the deployed KS "
                         "forward; default uniform weights (INDICATIVE numbers, A2.2)")
    args = ap.parse_args()
    w = None
    if args.fisher_weights_npz:
        f = np.load(args.fisher_weights_npz)
        w = np.asarray(f["w"], float)
    main_report(args.shard_dir, npz_out=args.npz_out, fig_dir=args.fig_dir, w=w)


if __name__ == "__main__":
    main()
