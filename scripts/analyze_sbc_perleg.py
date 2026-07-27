#!/usr/bin/env python3
"""PER-LEG SBC read-out — the deployed analysis is PER-LEG (DESI / KS / eBOSS fit
SEPARATELY), so this emits a 3-ROW table (one per leg) and NEVER pools across legs:
per-leg posteriors are NOT combinable (different data, different cov, different param
sets — KS has 25 params, DESI/eBOSS 26 with a_SiIII).

It is the per-leg sibling of ``analyze_sbc_heldout.py`` (held-out / HONEST mocks, leg_a=
False; the mocks CONTAIN emulator error). Same style: text summary FIRST, works
incrementally on whatever has landed (no hard mock-count assert), tau0 amp/dtau0 reported
alongside cosmology (feedback-report-tau0-dtau0-bias), and the ``tau0_amp_slope``
regression convention is COPIED VERBATIM from analyze_sbc_heldout.py so the numbers match.

THE KEY NEW DIAGNOSTIC — POSTERIOR CONTRACTION per param:
    contraction = 1 - mean_over_mocks(Var_post) / Var_prior      (in the named draw-space)
flags PRIOR-DOMINATED params (contraction < 0.1) so a leg that CANNOT constrain n_s (e.g.
eBOSS) is not mis-read as a clean gate PASS. Load-bearing expectation: eBOSS n_s is
prior-dominated; KS/DESI n_s should show real contraction.

  WHICH Var_prior + WHY (documented, see CONTRACTION_NOTE below):
  The stored ``draws``/``truth_vec`` are NOT raw whitened NUTS coordinates — they are the
  numpyro DETERMINISTICS in the NAMED space (ns..bhfeedback = theta_unit on the unit cube;
  tau0_z* = the actual tau0(z) ladder = tau0_vec; alpha_* = the actual incidence; a_SiIII =
  the actual coeff). So pulls are the direct (mu-truth)/sd, exactly as analyze_sbc_heldout.
  We therefore compute Var_prior IN THAT SAME NAMED SPACE, per param-type, from the KNOWN
  deployed prior spec (NOT from the across-mock truth spread — the truths span the held-out
  SIM grid, NOT the prior: n_s truth-std ~0.03 << prior-std ~0.29, and the tau0 truth is a
  FIXED ladder with zero across-mock spread, so a truth-spread Var_prior would be broken /
  negative-contraction). Concretely:
    * ns, Ap  (gated, load-bearing): Uniform[lo,hi] on the unit cube -> Var=(hi-lo)^2/12.
              EXACT (these are the headline gate params).
    * tau0amp, dtau0: forward-sample the ACTUAL Uniform priors (tau0_amp~U, dtau0~U), build
              the ladder with the deployed _kim curve, and regress with the IDENTICAL
              tau0_amp_slope -> Var_prior of the regressed quantity. EXACT-IN-CONVENTION.
    * alpha_lls/subdla/dla: reconstruct a TruncatedNormal(mu, frac*mu, low=0) prior from the
              cross-mock truth MEAN as mu and the deployed HCD_PRIOR_FRAC_SIGMA -> sample ->
              Var_prior. CAVEAT: the per-survey LLS center-boost / width-override (KS) are
              NOT applied here, so the alpha contraction is INDICATIVE not exact; the gated
              ns/Ap contraction is exact.

Usage:
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/analyze_sbc_perleg.py [ROOT] [OUT_PREFIX]

  ROOT        default /scratch/cavestru_root/cavestru1/mfho   (leg dirs prod_sbc_leg_<leg>)
  OUT_PREFIX  default sbc_perleg     (figure -> notes 05_likelihood/<prefix>_gate.png)
You can also override individual leg dirs with LEG_DIRS below (constant) or env.
"""
import pickle, glob, json, os, sys
import numpy as np

# --- deployed constants (so the prior reconstruction matches the run convention exactly) ---
try:
    from hcd_analysis.emulator.data import KIM_AMP, KIM_SLOPE
    from hcd_analysis.emulator.closure_legb import (
        TAU0_AMP_RANGE, DTAU0_RANGE, TAU0_PIVOT_Z)
    from hcd_analysis.emulator.inference import HCD_PRIOR_FRAC_SIGMA
except Exception as e:                                   # keep the read-out usable bare
    print(f"[perleg] WARN could not import deployed constants ({e}); using literals")
    KIM_AMP, KIM_SLOPE = 0.0023, 3.65
    TAU0_AMP_RANGE, DTAU0_RANGE, TAU0_PIVOT_Z = (0.75, 1.25), (-0.4, 0.25), 3.0
    HCD_PRIOR_FRAC_SIGMA = (0.15, 0.40, 0.50)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/cavestru_root/cavestru1/mfho"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "sbc_perleg"
OUT = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
os.makedirs(OUT, exist_ok=True)

# The 3 deployed legs (label -> dir). DESI may be empty (0 pkls) -> handled gracefully.
LEG_DIRS = {
    "DESI":  os.path.join(ROOT, "prod_sbc_leg_desi"),
    "KS":    os.path.join(ROOT, "prod_sbc_leg_ks"),
    "eBOSS": os.path.join(ROOT, "prod_sbc_leg_eboss"),
}

# tau0 ladder z-grid: the GLOBAL 13-rung ladder z = 2.2 + 0.2*i (present on EVERY leg).
Z_TAU0 = np.array([2.2 + 0.2 * i for i in range(13)])
lx = np.log(1.0 + Z_TAU0)
lx_c = lx - lx.mean()                         # centered -> intercept = ln(tau0) at z-bar

CONTRACTION_NOTE = ("Var_prior: ns/Ap Uniform-analytic (exact); tau0amp/dtau0 prior-forward-"
                    "sample (exact-in-convention); alpha_* TruncN(truthmean,frac*mu) recon "
                    "(indicative — per-survey LLS boost/width NOT applied)")
PRIOR_DOM_THRESH = 0.1


def tau0_amp_slope(tau0_ladder):
    """Regress ln(tau0(z)) on centered ln(1+z): (amp=exp(intercept@zbar), slope=dtau0).
    COPIED VERBATIM from analyze_sbc_heldout.py so the numbers match."""
    y = np.log(np.clip(tau0_ladder, 1e-8, None))
    slope = np.sum(lx_c * (y - y.mean())) / np.sum(lx_c ** 2)
    intercept = y.mean()                      # value of ln(tau0) at z-bar (lx centered)
    return np.exp(intercept), slope


# ---------- prior-variance helpers (the contraction denominator, named draw-space) ----------
_RNG = np.random.default_rng(0)


def _uniform_prior_var(lo, hi):
    return (hi - lo) ** 2 / 12.0


def _tau0_regressed_prior_var(n=20000):
    """Var_prior of the REGRESSED tau0amp & dtau0, by forward-sampling the actual Uniform
    priors (tau0_amp~U(TAU0_AMP_RANGE), dtau0~U(DTAU0_RANGE)), building the ladder with the
    deployed _kim curve on Z_TAU0, and regressing with the IDENTICAL tau0_amp_slope."""
    amp = _RNG.uniform(TAU0_AMP_RANGE[0], TAU0_AMP_RANGE[1], n)
    dt = _RNG.uniform(DTAU0_RANGE[0], DTAU0_RANGE[1], n)
    kim = KIM_AMP * (1.0 + Z_TAU0) ** KIM_SLOPE
    alpha_z = amp[:, None] * ((1.0 + Z_TAU0)[None, :] / (1.0 + TAU0_PIVOT_Z)) ** dt[:, None]
    ladder = alpha_z * kim[None, :]                       # (n, 13)
    ra = np.empty(n); rs = np.empty(n)
    for i in range(n):
        ra[i], rs[i] = tau0_amp_slope(ladder[i])
    return float(np.var(ra, ddof=1)), float(np.var(rs, ddof=1))


def _truncnorm_prior_var(mu, frac, n=20000):
    """Var_prior of a TruncatedNormal(mu, frac*mu, low=0) sampled draw — the deployed alpha
    incidence prior FORM (per-survey center/width overrides NOT applied; indicative)."""
    if not np.isfinite(mu) or mu <= 0:
        return np.nan
    sd = frac * mu
    s = _RNG.normal(mu, sd, n)
    s = s[s > 0.0]                                        # low=0 truncation
    return float(np.var(s, ddof=1)) if len(s) > 1 else np.nan


# ---------------------------- per-leg analysis ----------------------------
def analyze_leg(label, src):
    """Return a per-leg result dict, or None when no usable pkls have landed."""
    if not os.path.isdir(src):
        print(f"[{label}] dir does not exist: {src} -> no pkls yet")
        return None
    files = sorted(glob.glob(os.path.join(src, "mock_*.pkl")))
    if not files:
        print(f"[{label}] no pkls yet in {src}")
        return None

    mocks, kept_files, names = [], [], None
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except (EOFError, pickle.UnpicklingError):
            print(f"[{label}] SKIP partial/corrupt {os.path.basename(f)} (still writing?)")
            continue
        if names is None:
            names = list(d["names"])
        mocks.append(d)
        kept_files.append(f)         # parallel to mocks (files includes skipped pkls)
    M = len(mocks)
    if M == 0:
        print(f"[{label}] all landed pkls are partial — re-run shortly")
        return None

    # RUN-CFG POOLING HOMOGENEITY (2026-07-23, CS design review Q2): the _run_mock CLASH guard
    # only fires on resume-over-existing; two runs with DISJOINT mock indices into one dir never
    # met it, and this analyzer used to pool with zero verification. Every pooled pkl's
    # default-completed run_cfg must be identical — a closure pkl and an ARM-P (deployed-prior)
    # pkl differ on survey/hcd_prior_signature and REFUSE here.
    from scripts.run_prod_sbc_shard import effective_run_cfg
    _effs = [effective_run_cfg(m.get("run_cfg")) for m in mocks]
    _ref = _effs[0]
    for f, e in zip(kept_files, _effs):
        assert e == _ref, (f"[{label}] run_cfg POOLING MISMATCH at {os.path.basename(f)}: "
                           f"{e} != {_ref} — mixed SBC populations in one dir; separate them "
                           f"before analyzing")
    # ARM-P leg identity: a deployed-prior population must be THIS leg's survey.
    if _ref["survey"] is not None:
        assert _ref["survey"] == label, (f"[{label}] deployed-prior pkls stamped survey="
                                         f"{_ref['survey']!r} pooled under the {label} leg")
        assert _ref["hcd_prior_signature"], \
            f"[{label}] deployed-prior population lacks the hcd_prior_signature pin"

    P = len(names)
    j_ns, j_ap = names.index("ns"), names.index("Ap")
    TAU0_IDX = [names.index(f"tau0_z{z}") for z in range(13)]
    assert len(TAU0_IDX) == 13, f"[{label}] expected 13 tau0 rungs, got {len(TAU0_IDX)}"
    has_siiii = "a_SiIII" in names                       # DESI/eBOSS only

    # named scalar params we pull (alpha_dla can be ~0 -> sd tiny, guarded)
    SCALAR = ["ns", "Ap", "alpha_subdla", "alpha_lls", "alpha_dla"]
    pulls = {k: [] for k in SCALAR + ["tau0amp", "dtau0"]}
    postvar = {k: [] for k in SCALAR}                    # per-mock Var_post (named space)
    postvar_tau = {"tau0amp": [], "dtau0": []}
    truths = {k: [] for k in SCALAR}
    ll_rank_frac, ndiv_all, L_all = [], [], []
    # SBC RANKS (2026-07-26): the EXACT calibration statistic, and a PRE-REGISTERED gate
    # criterion ("rank-uniformity: KS p>0.05 AND ECDF inside the Beta(k,N+1-k) band") that
    # this analyzer did not previously compute. The pull mean is NOT exact -- it can be
    # nonzero for a skewed posterior under perfect calibration -- so a pull FAIL with
    # uniform ranks and a pull FAIL with non-uniform ranks are different findings.
    ranks = {k: [] for k in SCALAR + ["tau0amp", "dtau0"]}
    # METAL-FLOOR baseline (2026-07-26): the criteria PRE-DECLARE that a flat-log metal
    # prior reappears as an A_p baseline on metal-free mocks and is "NOT emulator bias".
    # The prior is LogUniform on a positive decrement, so it has no mass at zero and settles
    # at a near-CONSTANT floor. That constancy is exactly why a within-leg correlation of
    # the A_p pull against the metal amplitude has NO power against it; report the floor's
    # size and its across-mock constancy instead.
    metal_nodes = {}

    for d in mocks:
        dr = np.asarray(d["draws"]); t = np.asarray(d["truth_vec"])
        L = dr.shape[0]; L_all.append(L)
        ndiv_all.append(int(d.get("n_div", -1)))

        def pull(j):
            col = dr[:, j]
            sd = col.std(ddof=1) if L > 1 else np.nan    # N=1 guard -> nan
            mu = col.mean()
            return ((mu - t[j]) / sd if (np.isfinite(sd) and sd > 0) else np.nan), \
                   (col.var(ddof=1) if L > 1 else np.nan)

        for k in SCALAR:
            j = names.index(k)
            p, v = pull(j)
            pulls[k].append(p); postvar[k].append(v); truths[k].append(t[j])
            ranks[k].append(float(np.mean(dr[:, j] < t[j])) if L > 1 else np.nan)

        # metal-node floor bookkeeping (present only on the metal-floated legs)
        for nm_site, rec_site in (d.get("sites_extra") or {}).items():
            if not (nm_site.startswith("f_Si") or nm_site.startswith("k_Si")):
                continue
            x = np.asarray(rec_site.get("draws"), float)
            if x.size == 0:
                continue
            metal_nodes.setdefault(nm_site, {"post_mean": [], "post_sd": [], "truth": []})
            metal_nodes[nm_site]["post_mean"].append(float(x.mean()))
            metal_nodes[nm_site]["post_sd"].append(float(x.std(ddof=1)) if x.size > 1 else np.nan)
            metal_nodes[nm_site]["truth"].append(float(rec_site.get("truth", np.nan)))

        # tau0 amp/dtau0: regress each draw's ladder + the truth ladder
        amps = np.empty(L); slopes = np.empty(L)
        for i in range(L):
            amps[i], slopes[i] = tau0_amp_slope(dr[i, TAU0_IDX])
        amp_t, slope_t = tau0_amp_slope(t[TAU0_IDX])
        sa = amps.std(ddof=1) if L > 1 else np.nan
        ss = slopes.std(ddof=1) if L > 1 else np.nan
        pulls["tau0amp"].append((amps.mean() - amp_t) / sa if (np.isfinite(sa) and sa > 0) else np.nan)
        pulls["dtau0"].append((slopes.mean() - slope_t) / ss if (np.isfinite(ss) and ss > 0) else np.nan)
        ranks["tau0amp"].append(float(np.mean(amps < amp_t)) if L > 1 else np.nan)
        ranks["dtau0"].append(float(np.mean(slopes < slope_t)) if L > 1 else np.nan)
        postvar_tau["tau0amp"].append(amps.var(ddof=1) if L > 1 else np.nan)
        postvar_tau["dtau0"].append(slopes.var(ddof=1) if L > 1 else np.nan)

        if d.get("ll_true") is not None and d.get("ll_draws") is not None:
            lld = np.asarray(d["ll_draws"]); llt = float(d["ll_true"])
            ll_rank_frac.append(float(np.mean(lld < llt)))

    # ---- Var_prior in the named space (the contraction denominator) ----
    # ns/Ap: Uniform on the unit cube. We don't have ctx bounds in the pkl, but the unit cube
    # default is [0,1] for both ns and Ap (sampling_unit_bounds: lo=hi-edge only for the IGM
    # params herei/heref/alphaq). Use [0,1] -> Var=1/12 (documented).
    var_prior = {"ns": _uniform_prior_var(0.0, 1.0), "Ap": _uniform_prior_var(0.0, 1.0)}
    ta_pv, dt_pv = _tau0_regressed_prior_var()
    var_prior["tau0amp"], var_prior["dtau0"] = ta_pv, dt_pv
    fl, fs, fd = HCD_PRIOR_FRAC_SIGMA
    for k, frac in (("alpha_lls", fl), ("alpha_subdla", fs), ("alpha_dla", fd)):
        mu_truth = float(np.nanmean(truths[k])) if len(truths[k]) else np.nan
        var_prior[k] = _truncnorm_prior_var(mu_truth, frac)

    def contraction(k):
        vp = var_prior.get(k, np.nan)
        pv = postvar[k] if k in postvar else postvar_tau.get(k, [])
        pv = np.asarray(pv, float); pv = pv[np.isfinite(pv)]
        if not np.isfinite(vp) or vp <= 0 or len(pv) == 0:
            return np.nan
        return 1.0 - float(np.mean(pv)) / vp

    res = {
        "label": label, "src": src, "n_mocks": M, "n_params": P, "has_siiii": has_siiii,
        "div_total": int(sum(max(0, x) for x in ndiv_all)),
        "L_median": int(np.median(L_all)), "L_range": [int(min(L_all)), int(max(L_all))],
        "ll_rank_frac_mean": float(np.mean(ll_rank_frac)) if ll_rank_frac else None,
        "pulls": {k: np.asarray(v, float) for k, v in pulls.items()},
        "ranks": {k: np.asarray(v, float) for k, v in ranks.items()},
        "rank_uniformity": {k: rank_uniformity(v) for k, v in ranks.items()},
        "metal_floor": metal_floor_summary(metal_nodes),
        "contraction": {k: contraction(k) for k in SCALAR + ["tau0amp", "dtau0"]},
        "var_prior": var_prior,
        # the homogeneity-verified effective config (2026-07-23): the certificate's per-leg
        # prior identity (signed PI item 5: the certificate must NAME the parameterization).
        "run_cfg_effective": _ref,
    }
    return res


def pull_stats(arr):
    a = np.asarray(arr, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(mean=np.nan, std=np.nan, sem=np.nan, n=0)
    m = float(a.mean())
    s = float(a.std(ddof=1)) if len(a) > 1 else np.nan    # N=1 -> nan guard
    sem = s / np.sqrt(len(a)) if (len(a) > 1) else np.nan
    return dict(mean=m, std=s, sem=sem, n=int(len(a)))


def rank_uniformity(rank_fracs, n_grid=None):
    """PRE-REGISTERED rank criterion: KS p>0.05 against Uniform(0,1) AND the rank ECDF
    inside the analytic Beta(k, N+1-k) pointwise band.

    Returns mean rank, the KS p-value, the count of order statistics outside the pointwise
    95% band, and the verdict. The ECDF points are strongly correlated, so the band count is
    reported as CORROBORATION of a systematic shift, never as an independent test: one
    offset pushes many points out at once. The KS p is the criterion."""
    u = np.asarray([x for x in rank_fracs if np.isfinite(x)], float)
    n = u.size
    if n < 4:
        return dict(n=n, mean=np.nan, ks_p=np.nan, n_outside_band=None,
                    verdict="underpowered(N<4)")
    try:
        from scipy import stats as _st
        ks_p = float(_st.kstest(u, "uniform").pvalue)
        us = np.sort(u)
        out = int(sum(not (_st.beta.ppf(0.025, k, n + 1 - k) <= us[k - 1]
                           <= _st.beta.ppf(0.975, k, n + 1 - k)) for k in range(1, n + 1)))
    except Exception:                                     # keep the read-out usable bare
        return dict(n=n, mean=float(u.mean()), ks_p=np.nan, n_outside_band=None,
                    verdict="scipy-unavailable")
    return dict(n=n, mean=float(u.mean()), ks_p=ks_p, n_outside_band=out,
                verdict=("UNIFORM" if ks_p > 0.05 else "NON-UNIFORM"))


def metal_floor_summary(metal_nodes):
    """PRE-DECLARED metal-floor baseline. On metal-free mocks the LogUniform node prior has
    no mass at zero, so the nodes settle at a near-constant floor that biases A_p WITHOUT
    producing any across-mock correlation. Reports the floor size and its constancy
    (across-mock spread over the mean within-mock posterior sd): a ratio well below 1 means
    the floor is effectively the same in every mock, and therefore that any within-leg
    correlation test against it is UNINFORMATIVE rather than exculpatory."""
    if not metal_nodes:
        return None
    out = {"nodes": {}, "mocks_carry_metal_truth": None}
    truth_finite = []
    for nm, rec in sorted(metal_nodes.items()):
        pm = np.asarray(rec["post_mean"], float)
        ps = np.asarray(rec["post_sd"], float)
        tr = np.asarray(rec["truth"], float)
        within = float(np.nanmean(ps)) if np.isfinite(ps).any() else np.nan
        across = float(pm.std(ddof=1)) if pm.size > 1 else np.nan
        truth_finite.append(bool(np.isfinite(tr).any()))
        out["nodes"][nm] = dict(
            post_mean=float(pm.mean()), within_mock_sd=within, across_mock_sd=across,
            constancy_ratio=(across / within if (np.isfinite(within) and within > 0) else np.nan),
            truth_present=bool(np.isfinite(tr).any()))
    out["mocks_carry_metal_truth"] = bool(any(truth_finite))
    ratios = [v["constancy_ratio"] for v in out["nodes"].values()
              if np.isfinite(v["constancy_ratio"])]
    out["max_constancy_ratio"] = float(max(ratios)) if ratios else np.nan
    out["floor_active"] = bool((not out["mocks_carry_metal_truth"]) and out["nodes"])
    return out


def gate_cosmo(st, gate_mean=0.3, gate_std=1.1):
    """n_s / A_p gate: |pull mean|<=0.3 AND pull std<=1.1, only when N>=4 (else underpowered)."""
    n = st["n"]
    if n < 4:
        return "underpowered(N<4)"
    ok_m = abs(st["mean"]) <= gate_mean
    ok_s = np.isfinite(st["std"]) and st["std"] <= gate_std
    return "PASS" if (ok_m and ok_s) else "FAIL"


def is_prior_dom(c):
    """PRIOR-DOMINATED = the posterior barely contracts: 0 <= c < threshold. A NEGATIVE
    contraction (Var_post > Var_prior_recon) is NOT prior-domination — it means the prior-var
    RECONSTRUCTION under-shoots (the alpha_dla one-sided softplus prior, whose recon as a plain
    TruncN(tiny-mu) is too narrow; alpha_dla is secondary), so we do NOT flag it prior-dom."""
    return np.isfinite(c) and 0.0 <= c < PRIOR_DOM_THRESH


def ctag(c):
    if not np.isfinite(c):
        return "  c=n/a"
    flag = "  PRIOR-DOM" if is_prior_dom(c) else ("  (recon<post)" if c < 0 else "")
    return f"  c={c:+.2f}{flag}"


# ============================== RUN ==============================
print("=" * 92)
print("PER-LEG held-out SBC read-out  —  posteriors NOT combinable (3 separate fits)")
print(f"ROOT={ROOT}")
print(f"contraction = 1 - mean(Var_post)/Var_prior ; PRIOR-DOMINATED if < {PRIOR_DOM_THRESH}")
print(f"  [{CONTRACTION_NOTE}]")
print("=" * 92)

results = {}
for label, src in LEG_DIRS.items():
    print(f"\n--- {label} ---")
    r = analyze_leg(label, src)
    if r is None:
        continue
    results[label] = r
    # per-leg headline lines (text-first, like analyze_sbc_heldout)
    for k in ("ns", "Ap", "tau0amp", "dtau0", "alpha_subdla", "alpha_lls", "alpha_dla"):
        st = pull_stats(r["pulls"][k])
        gate = ""
        if k in ("ns", "Ap"):
            gate = "  GATE:" + gate_cosmo(st)
        print(f"  {k:13s} mean={st['mean']:+.3f} std={st['std'] if np.isfinite(st['std']) else float('nan'):.3f}"
              f" sem={st['sem'] if np.isfinite(st['sem']) else float('nan'):.3f} N={st['n']}"
              f"{ctag(r['contraction'][k])}{gate}")
    # PRE-REGISTERED rank-uniformity criterion (the EXACT statistic; the pull mean is not
    # exact for a skewed posterior, so these two lines must be read together)
    print("  -- rank uniformity (PRE-REGISTERED: KS p>0.05; Beta-band count is corroboration) --")
    for k in ("ns", "Ap", "tau0amp", "dtau0"):
        ru = r["rank_uniformity"].get(k, {})
        if not ru or not np.isfinite(ru.get("ks_p", np.nan)):
            continue
        band = ("" if ru["n_outside_band"] is None
                else f"  outside Beta band {ru['n_outside_band']}/{ru['n']}")
        print(f"  {k:13s} mean rank={ru['mean']:.3f} (ideal 0.5)  KS p={ru['ks_p']:.4f}  "
              f"{ru['verdict']}{band}")
    mf = r.get("metal_floor")
    if mf:
        print("  -- metal-node floor (PRE-DECLARED A_p baseline on metal-free mocks) --")
        print(f"  mocks carry metal truth: {mf['mocks_carry_metal_truth']}   "
              f"floor active: {mf['floor_active']}   max constancy ratio "
              f"{mf['max_constancy_ratio']:.3f}")
        for nm, v in mf["nodes"].items():
            print(f"    {nm:22s} post={v['post_mean']:.5f}  within-mock sd={v['within_mock_sd']:.5f}"
                  f"  across-mock sd={v['across_mock_sd']:.5f}  ratio={v['constancy_ratio']:.3f}")
        if mf["floor_active"]:
            print("    NOTE: the floor is near-CONSTANT across mocks, so a within-leg correlation of"
                  "\n    the A_p pull against the metal amplitude has NO POWER against it. Testing it"
                  "\n    requires a metals-off refit, not a correlation.")
    if r["ll_rank_frac_mean"] is not None:
        print(f"  loglik-rank   mean={r['ll_rank_frac_mean']:.3f} (ideal 0.5)")
    print(f"  run health    div_total={r['div_total']}  L median={r['L_median']} "
          f"range={r['L_range']}  params={r['n_params']}{'  (+a_SiIII)' if r['has_siiii'] else ''}")

if not results:
    print("\n[perleg] no leg has usable pkls yet — nothing to tabulate. Re-run as mocks land.")
    sys.exit(0)

# ---------------------------- the 3-ROW TABLE ----------------------------
print("\n" + "=" * 92)
print("PER-LEG GATE TABLE  (NOT combinable — read each row independently)")
print("=" * 92)
hdr = (f"{'leg':6s} {'N':>3s} | {'n_s pull(mean±std)':>20s} {'gate':>6s} {'c_ns':>7s} | "
       f"{'A_p pull':>16s} {'c_Ap':>7s} | {'tau0amp':>9s} {'dtau0':>9s} | {'llrank':>6s} {'ndiv':>5s}")
print(hdr)
print("-" * len(hdr))
for label in ("DESI", "KS", "eBOSS"):
    if label not in results:
        print(f"{label:6s}  -- no pkls yet --")
        continue
    r = results[label]
    sn, sa = pull_stats(r["pulls"]["ns"]), pull_stats(r["pulls"]["Ap"])
    sta, sdt = pull_stats(r["pulls"]["tau0amp"]), pull_stats(r["pulls"]["dtau0"])
    cns, cap = r["contraction"]["ns"], r["contraction"]["Ap"]
    llr = r["ll_rank_frac_mean"]
    ns_str = f"{sn['mean']:+.2f}±{sn['std']:.2f}" if np.isfinite(sn['std']) else f"{sn['mean']:+.2f}±nan"
    ap_str = f"{sa['mean']:+.2f}±{sa['std']:.2f}" if np.isfinite(sa['std']) else f"{sa['mean']:+.2f}±nan"
    cns_str = (f"{cns:+.2f}*" if is_prior_dom(cns) else
               (f"{cns:+.2f}" if np.isfinite(cns) else "n/a"))
    cap_str = (f"{cap:+.2f}*" if is_prior_dom(cap) else
               (f"{cap:+.2f}" if np.isfinite(cap) else "n/a"))
    print(f"{label:6s} {r['n_mocks']:3d} | {ns_str:>20s} {gate_cosmo(sn):>6s} {cns_str:>7s} | "
          f"{ap_str:>16s} {cap_str:>7s} | {sta['mean']:+9.2f} {sdt['mean']:+9.2f} | "
          f"{(llr if llr is not None else float('nan')):6.2f} {r['div_total']:5d}")
print("-" * len(hdr))
print("  c_* = posterior contraction (1 - <Var_post>/Var_prior);  '*' = PRIOR-DOMINATED (<0.1)")

# ---------------------------- per-leg certification note ----------------------------
print("\nPER-LEG CERTIFICATION ROLES (posteriors are NOT combinable):")
print("  DESI  certifies A_p / LLS / low-k       (n_s contraction modest; A_p the anchor)")
print("  KS    certifies n_s / high-k tilt       (the HONEST n_s certifier — read c_ns)")
print("  eBOSS certifies A_p (anchor); its n_s IS constrained (c_ns ~0.96, NOT prior-dominated, verified")
print("        2026-06-21) — so an n_s gate FAIL there is a REAL pull bias, not an unconstrained posterior.")

# ---------------------------- JSON ----------------------------
def jpull(arr):
    return pull_stats(arr)

out_json = os.path.join(OUT, f"{PREFIX}_gate.json")
jdump = {
    "root": ROOT, "per_leg_not_combinable": True,
    "contraction_method": CONTRACTION_NOTE, "prior_dom_thresh": PRIOR_DOM_THRESH,
    "z_tau0": Z_TAU0.tolist(),
    "legs": {},
}
for label in ("DESI", "KS", "eBOSS"):
    if label not in results:
        jdump["legs"][label] = {"status": "no_pkls_yet", "src": LEG_DIRS[label]}
        continue
    r = results[label]
    leg = {
        "src": r["src"], "n_mocks": r["n_mocks"], "n_params": r["n_params"],
        "has_siiii": r["has_siiii"], "div_total": r["div_total"],
        "L_median": r["L_median"], "L_range": r["L_range"],
        "ll_rank_frac_mean": r["ll_rank_frac_mean"],
        "gate_ns": gate_cosmo(pull_stats(r["pulls"]["ns"])),
        "gate_Ap": gate_cosmo(pull_stats(r["pulls"]["Ap"])),
        "pulls": {k: jpull(r["pulls"][k]) for k in r["pulls"]},
        # PRE-REGISTERED criteria added 2026-07-26 (previously absent from this certificate)
        "rank_uniformity": {k: {kk: (None if (isinstance(vv, float) and not np.isfinite(vv))
                                     else vv) for kk, vv in v.items()}
                            for k, v in r["rank_uniformity"].items()},
        "gate_rank_ns": r["rank_uniformity"]["ns"]["verdict"],
        "gate_rank_Ap": r["rank_uniformity"]["Ap"]["verdict"],
        "metal_floor": r.get("metal_floor"),
        "contraction": {k: (float(v) if np.isfinite(v) else None)
                        for k, v in r["contraction"].items()},
        "var_prior": {k: (float(v) if np.isfinite(v) else None)
                      for k, v in r["var_prior"].items()},
        # PER-LEG PRIOR IDENTITY (signed PI item 5, 2026-07-23): the certificate names the
        # parameterization this leg's SBC population fit under. deployed_prior_certificate=False
        # marks a closure-prior (survey=None) population — NOT a deployed-prior certification.
        "prior": {
            "survey": r["run_cfg_effective"]["survey"],
            "hcd_parameterization": r["run_cfg_effective"]["hcd_parameterization"],
            "hcd_prior_signature": r["run_cfg_effective"]["hcd_prior_signature"],
            "deployed_prior_certificate": r["run_cfg_effective"]["survey"] is not None,
        },
    }
    jdump["legs"][label] = leg
json.dump(jdump, open(out_json, "w"), indent=2)
print(f"\nwrote {out_json}")

# ---------------------------- FIGURE ----------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [lab for lab in ("DESI", "KS", "eBOSS")]                 # fixed leg order
    present = [lab for lab in order if lab in results]
    xpos = np.arange(len(order))
    colors = {"DESI": "#4477AA", "KS": "#228833", "eBOSS": "#CCBB44"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (1) n_s mean-pull +/- sem vs the +/-0.3 gate band
    # (2) A_p mean-pull +/- sem vs the +/-0.3 gate band
    for ax, par, ttl in ((axes[0], "ns", r"$n_s$ mean pull $\pm$ sem"),
                         (axes[1], "Ap", r"$A_p$ mean pull $\pm$ sem")):
        ax.axhspan(-0.3, 0.3, color="0.85", label="gate band |mean|<=0.3")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        for i, lab in enumerate(order):
            if lab not in results:
                ax.text(i, 0, "no\npkls", ha="center", va="center", fontsize=9, color="0.5")
                continue
            st = pull_stats(results[lab]["pulls"][par])
            yerr = st["sem"] if np.isfinite(st["sem"]) else 0.0
            ax.errorbar(i, st["mean"], yerr=yerr, fmt="o", ms=9, capsize=5,
                        color=colors[lab], lw=2)
            c = results[lab]["contraction"][par]
            tag = f"N={st['n']}\nc={c:+.2f}" + ("\nPRIOR-DOM" if is_prior_dom(c) else "")
            ax.annotate(tag, (i, st["mean"]), textcoords="offset points",
                        xytext=(10, 0), fontsize=8, va="center")
        ax.set_xticks(xpos); ax.set_xticklabels(order)
        ax.set_ylim(-1.6, 1.6); ax.set_ylabel("pull mean"); ax.set_title(ttl)
        ax.legend(fontsize=8, loc="upper right")

    # (3) contraction bars per leg (ns, Ap) so prior-dominated legs are visually obvious
    ax = axes[2]
    width = 0.38
    bar_color = {"ns": "#EE6677", "Ap": "#66CCEE"}
    for off, par, lbl in ((-width / 2, "ns", "n_s"), (width / 2, "Ap", "A_p")):
        vals = [results[lab]["contraction"][par] if lab in results else np.nan for lab in order]
        vals = [v if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar(xpos + off, vals, width, label=lbl, color=bar_color[par])
        for b, lab in zip(bars, order):
            if lab in results and is_prior_dom(results[lab]["contraction"][par]):
                b.set_hatch("xx"); b.set_edgecolor("crimson")
    ax.axhline(PRIOR_DOM_THRESH, color="crimson", ls="--", lw=1.2,
               label=f"prior-dom < {PRIOR_DOM_THRESH}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(order)
    ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.05)
    ax.set_ylabel("posterior contraction"); ax.set_title("contraction (hatched = prior-dom)")
    ax.legend(fontsize=8, loc="lower left")

    npres = len(present)
    fig.suptitle(f"PER-LEG held-out SBC — {npres}/3 legs landed (NOT combinable)  |  "
                 f"gate: ns,Ap |mean|<=0.3 & std<=1.1; contraction flags prior-dom",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figp = os.path.join(OUT, f"{PREFIX}_gate.png")
    fig.savefig(figp, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"wrote {figp}")
except Exception as e:
    import traceback
    print(f"(figure skipped: {e})")
    traceback.print_exc()
