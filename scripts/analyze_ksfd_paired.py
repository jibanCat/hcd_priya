#!/usr/bin/env python3
"""A3d: the KS f_res PIN-vs-DRAW paired attribution diagnostic (PI #9 section 3f, Option B).

NARROW PURPOSE (the PI's words): test whether the known f_res-truth pinning materially
contributes to the KS tau0_amp pull. PRIMARY attribution statistic: the paired per-mock
tau0_amp pull delta (draw - pin). n_s and A_p are SECONDARY transmission checks: reported,
never celled, never a gate. Pre-registration: `2026-08-05-A3d-PREREGISTRATION.md`
(notes repo); the decision table below is that document's table, in code, exhaustive.

BINDING INTERPRETATION (PI #9 3f.3, printed with every readout):
  * a NULL result may EXCLUDE f_res pinning as a material explanation of the KS pull;
  * a NON-NULL result may SUPPORT f_res pinning as a contributor;
  * NEITHER result by itself distinguishes finite-sample noise from another genuine
    KS-specific mechanism (the pin and draw arms share the same seed, so the same noise
    realization is frozen into both);
  * this is an ATTRIBUTION diagnostic, not a substitute for the A3c certification arm.

THE ARMS. PIN = the 12 PRESERVED r6x deployed KS pkls (seed 20260724, NUTS 250/300, f_res
truths pinned/NaN; r6x schema: top-level dict with per_mock + meta, EMPTY run_cfg). DRAW =
12 new fits from `run_prod_sbc_shard --deployed-prior --leg KS --fres-selfdraw --seed
20260724 --n-warmup 250 --n-samples 300` (per-mock mock_%04d.pkl schema, frozen run_cfg
incl. the 2026-08-05 sampler-population stamps). The flags-off tiny-NUTS probe reproduced
the r6x mock BITWISE (truth_vec, ll_true to the last digit, every sites_extra truth), so
the pairing is exact by construction and CONJUNCT C2 verifies it per mock anyway.

WHAT IT REFUSES TO DO. On any conjunct failure the arms are not the populations they claim
to be and this module raises PairingError instead of printing a number. A *.smoke.pkl in
the pin directory is EXCLUDED by name and refused by meta (the 2026-08-05 erratum class:
the r6x eBOSS table was contaminated by exactly such a file).

Usage:
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/analyze_ksfd_paired.py DRAW_DIR R6X_KS_DIR [EXPECT_N]
"""
import glob
import os
import pickle
import sys

import numpy as np
from scipy import stats as _st

N_PAIRS_DEFAULT = 12

# The pre-registered materiality threshold on the f_res contribution to the tau0_amp pull,
# in sigma_post pull units (the campaign's standing 0.30 budget unit). Fixed in advance.
MATERIALITY_M = 0.30

# The full-explanation reference is DEFINED as -(mean pin-arm tau0_amp pull) and RECOMPUTED
# from the pin pkls at readout; the pre-registration records its value on the fixed pin
# artifacts as +0.6332 and the readout asserts the recomputation matches within FULL_TOL
# (the pin arm is a preserved, immutable population; a mismatch means the wrong pkls).
FULL_EXPLANATION_PINNED = +0.6332
FULL_TOL = 5e-4

# Frozen draw-arm run_cfg: measured from the flags-on pairing probe + the 2026-08-05
# sampler-population stamps. Every draw pkl must carry EXACTLY this dict (C5).
FROZEN_DRAW_CFG = dict(
    leg_a=True, cemu_variant="current", amp_sigma=0.0, leg="KS", fold=0,
    tau0_prior_sigma=0.0, subdla_truth_boost=1.0, res_corr_on=False,
    sample_res=True, f_res_amp_sigma=0.15, metal_prior="uniform",
    fres_selfdraw=True, metal_selfdraw=False, diag_no_sample_metals=False,
    ks_kmax=0.065, survey="KS", hcd_parameterization="dndx_mapped_v2",
    hcd_prior_signature="50befc941edfc4c789286d2054d0107f1eb19a5672bcb86131426fb101eea216",
    single_member=False,
    seed=20260724, n_warmup=250, n_samples=300, max_tree_depth=10,
)

# Frozen pin-arm identity: the r6x deployed KS population's own meta fields (C6).
FROZEN_PIN_META = dict(arm="deployed", leg="KS", survey="KS",
                       seed=20260724, n_warmup=250, n_samples=300, max_tree_depth=10,
                       smoke=False)

# The 8 shared KS truth sites that must pair BITWISE (C2), on top of the 25-entry truth_vec.
SHARED_TRUTH_SITES = ("dla_raw", "dtau0", "eps_lls", "kappa_lls", "m_sub", "t_dla", "t_sub",
                      "tau0_amp")

# Deployed f_res laws for the C4 KS-consistency check (PROD_FORWARD_BY_LEG: KS sigma 0.15).
FRES_LAW_SD = {"f_res_amp": 0.15, "f_res_slope": 0.5}
LAW_ALPHA = 1e-3

CHANNELS = ("tau0amp", "ns", "Ap", "dtau0")
PRIMARY = "tau0amp"

CELLS = ("UNEXPECTED-DIRECTION", "NULL", "UNDER-RESOLVED",
         "NON-NULL-IMMATERIAL", "NON-NULL-MATERIAL")


class PairingError(RuntimeError):
    """A conjunct failed: the arms are not the pre-registered populations. NO numbers."""


def attribution_cell(lo, hi, M=MATERIALITY_M):
    """The pre-registered EXHAUSTIVE decision table over the 95% CI [lo, hi] of the mean
    paired tau0_amp delta (draw - pin). Tie conventions fixed in advance: lo <= 0 counts as
    zero-inclusion; hi >= M counts as materiality NOT excluded; hi < 0 strictly is the
    unexpected direction (drawing the truths made the pull MORE negative), a PI stop
    condition (3f.6). The five cells are mutually exclusive and cover every ordered CI."""
    if not (lo <= hi):
        raise ValueError(f"CI must be ordered, got [{lo}, {hi}]")
    if hi < 0:
        return "UNEXPECTED-DIRECTION"
    if lo <= 0:
        return "NULL" if hi < M else "UNDER-RESOLVED"
    return "NON-NULL-IMMATERIAL" if hi < M else "NON-NULL-MATERIAL"


# ---------------------------------------------------------------------------------------------
def _fail(conj, msg):
    raise PairingError(f"[{conj}] {msg} -- REFUSING to print any paired number.")


def _load_draw(draw_dir, n):
    recs = {}
    for m in range(n):
        p = os.path.join(draw_dir, f"mock_{m:04d}.pkl")
        if not os.path.exists(p):
            _fail("C1", f"draw arm incomplete: missing {p}")
        with open(p, "rb") as f:
            recs[m] = pickle.load(f)
    return recs

def _load_pin(pin_dir, n):
    files = sorted(glob.glob(os.path.join(pin_dir, "r6x_ks_deployed_shard_*.pkl")))
    files = [f for f in files if "smoke" not in os.path.basename(f)]   # the erratum class
    # P3 review finding 1: refuse a wrong CENSUS outright. A stray same-config re-fit dropped
    # into the preserved dir under a matching name would otherwise silently last-glob-win over
    # the preserved chain (same truths, same meta: C2/C3/C6 blind to it).
    if len(files) != n:
        _fail("C1", f"pin arm census wrong: {len(files)} shard pkls (smoke excluded), need "
                    f"exactly {n}: {sorted(os.path.basename(f) for f in files)}")
    recs = {}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        m = int(d["idxs"][0])
        if m in recs:
            _fail("C1", f"pin arm mock index {m} appears in more than one shard pkl")
        recs[m] = d
    if sorted(recs) != list(range(n)):
        _fail("C1", f"pin arm incomplete: have mocks {sorted(recs)}, need 0..{n-1}")
    return recs


def _truth(se, k):
    return float(se[k]["truth"])


def _check_conjuncts(draw, pin, n):
    # C2: bitwise base-truth pairing per mock (truth_vec + the 8 shared sites_extra truths)
    for m in range(n):
        dpm, ppm = draw[m], pin[m]["per_mock"][0]
        if not np.array_equal(np.asarray(dpm["truth_vec"]), np.asarray(ppm["truth_vec"])):
            _fail("C2", f"mock {m}: truth_vec differs between arms (pairing broken)")
        for k in SHARED_TRUTH_SITES:
            a, b = _truth(dpm["sites_extra"], k), _truth(ppm["sites_extra"], k)
            if not (np.isfinite(a) and np.isfinite(b) and a == b):
                _fail("C2", f"mock {m}: shared truth site {k} differs ({a!r} vs {b!r})")
    # C3: pin-side f_res truths PINNED (NaN) -- the defect population, verbatim
    for m in range(n):
        se = pin[m]["per_mock"][0]["sites_extra"]
        for k in FRES_LAW_SD:
            if np.isfinite(_truth(se, k)):
                _fail("C3", f"pin mock {m}: {k} truth is FINITE -- not the pinned population")
    # C4: draw-side f_res truths DRAWN: finite, nonzero, distinct across mocks, law-consistent
    for k, sd in FRES_LAW_SD.items():
        vals = np.array([_truth(draw[m]["sites_extra"], k) for m in range(n)])
        if not np.all(np.isfinite(vals)):
            _fail("C4", f"draw arm: {k} truth NaN on some mock -- self-draw did not happen")
        if np.any(vals == 0.0):
            _fail("C4", f"draw arm: {k} truth exactly 0 (the pinned value) on some mock")
        if np.unique(vals).size != n:
            _fail("C4", f"draw arm: repeated {k} truth across mocks (decoy population)")
        p = _st.kstest(vals / sd, "norm").pvalue
        if p < LAW_ALPHA:
            _fail("C4", f"draw arm: {k} truths inconsistent with the deployed law "
                        f"Normal(0, {sd}) (KS p {p:.2e} < {LAW_ALPHA})")
    # C4b (P3 review finding 2a): the runner's own provenance must agree that nothing was left
    # un-drawn -- the committed backstop of the by-hand probe check in pre-registration P1.
    for m in range(n):
        nsd = draw[m].get("truth_site_semantics", {}).get("not_self_drawn", "MISSING")
        if nsd != []:
            _fail("C4", f"draw mock {m}: truth_site_semantics.not_self_drawn = {nsd!r} "
                        f"(expected []) -- the self-draw provenance disagrees")
    # C5: draw-side frozen run_cfg, identical across mocks and equal to the registration
    for m in range(n):
        cfg = draw[m].get("run_cfg")
        if cfg != FROZEN_DRAW_CFG:
            missing = set(FROZEN_DRAW_CFG) - set(cfg or {})
            extra = set(cfg or {}) - set(FROZEN_DRAW_CFG)
            diff = {k: (None if not cfg else cfg.get(k), FROZEN_DRAW_CFG[k])
                    for k in FROZEN_DRAW_CFG if cfg is None or cfg.get(k) != FROZEN_DRAW_CFG[k]}
            _fail("C5", f"draw mock {m}: run_cfg != FROZEN_DRAW_CFG "
                        f"(missing {sorted(missing)}, extra {sorted(extra)}, diff {diff})")
    # C6: pin-side identity via its own meta + top-level fields
    for m in range(n):
        d = pin[m]
        meta = d.get("meta", {})
        for k, v in FROZEN_PIN_META.items():
            got = meta.get(k, d.get(k))
            if k == "smoke":
                got = bool(meta.get("smoke", False))
            if got != v:
                _fail("C6", f"pin mock {m}: identity field {k}={got!r}, expected {v!r}")
    # C7: mock/truth NON-IDENTITY (P3 review finding 2b wording): ll_true equality would mean
    # the two arms are byte-level the same computation, refusing the trivial no-op. It does NOT
    # by itself prove the drawn truth reached the mock DATA (a recorded-but-unforwarded truth
    # also moves ll_true, evaluated at a different point); the propagation cover is the
    # A1c-validated forwarding path + the runner's fail-loud selfdraw assert + C4b + the
    # reported fres_draw response block.
    for m in range(n):
        a, b = float(draw[m]["ll_true"]), float(pin[m]["per_mock"][0]["ll_true"])
        if a == b:
            _fail("C7", f"mock {m}: ll_true IDENTICAL pin-vs-draw -- the arms are the same "
                        f"computation (no-op intervention)")


def _pull(pm, ch):
    if ch in ("ns", "Ap"):
        j = pm["names"].index(ch if ch != "Ap" else "Ap")
        dr = np.asarray(pm["draws"])[:, j]; tr = float(np.asarray(pm["truth_vec"])[j])
    else:
        key = "tau0_amp" if ch == "tau0amp" else ch
        se = pm["sites_extra"][key]
        dr = np.asarray(se["draws"]); tr = float(se["truth"])
    return (float(np.mean(dr)) - tr) / float(np.std(dr, ddof=1))


def _finite_l_sd(Ls):
    return float(np.sqrt(np.mean([(1 + 1 / L) * (L - 1) / (L - 3) for L in Ls])))


def run(draw_dir, pin_dir, expect_n=N_PAIRS_DEFAULT):
    n = int(expect_n)
    draw = _load_draw(draw_dir, n)
    pin = _load_pin(pin_dir, n)
    _check_conjuncts(draw, pin, n)

    pin_pm = {m: pin[m]["per_mock"][0] for m in range(n)}
    pulls_d = {ch: np.array([_pull(draw[m], ch) for m in range(n)]) for ch in CHANNELS}
    pulls_p = {ch: np.array([_pull(pin_pm[m], ch) for m in range(n)]) for ch in CHANNELS}

    # The pin arm is a fixed, preserved population, so its realized pull is a constant of the
    # registration (+0.6332). The recomputation is compared below and a mismatch is FLAGGED in
    # health (printed, never silently adjusted); C2/C6 already refuse mixed/wrong populations.
    full_ref = float(-np.mean(pulls_p[PRIMARY]))
    paired = {}
    tcrit = _st.t.ppf(0.975, n - 1)
    for ch in CHANNELS:
        D = pulls_d[ch] - pulls_p[ch]
        sd = float(D.std(ddof=1)); sem = sd / np.sqrt(n); mean = float(D.mean())
        w = _st.wilcoxon(D) if np.any(D != 0) else None
        paired[ch] = dict(mean=mean, sd=sd, sem=sem,
                          t=(mean / sem if sem > 0 else float("nan")),
                          ci95=(mean - tcrit * sem, mean + tcrit * sem),
                          wilcoxon_p=(float(w.pvalue) if w else 1.0),
                          rho=float(np.corrcoef(pulls_d[ch], pulls_p[ch])[0, 1]),
                          n_negative=int(np.sum(D < 0)), deltas=D)

    lo, hi = paired[PRIMARY]["ci95"]
    cell = attribution_cell(lo, hi, MATERIALITY_M)

    # draw-arm f_res sector: the intervention measurement of the 5b-bis calibration symptom
    fres = {}
    for k, tau in FRES_LAW_SD.items():
        trs = np.array([_truth(draw[m]["sites_extra"], k) for m in range(n)])
        mus = np.array([float(np.mean(draw[m]["sites_extra"][k]["draws"])) for m in range(n)])
        sds = np.array([float(np.std(draw[m]["sites_extra"][k]["draws"], ddof=1))
                        for m in range(n)])
        ranks = np.array([float(np.mean(np.asarray(draw[m]["sites_extra"][k]["draws"]) < trs[i]))
                          for i, m in enumerate(range(n))])
        pulls = (mus - trs) / sds
        fres[k] = dict(pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
                       rank_ks_p=float(_st.kstest(ranks, "uniform").pvalue),
                       scatter=float(mus.std(ddof=1)),
                       scatter_expected=float(np.sqrt(max(tau**2 - float(np.mean(sds**2)), 0.0))))

    Ld = [int(draw[m]["L"]) for m in range(n)]
    Lp = [int(pin_pm[m]["L"]) for m in range(n)]
    ndiv_d = sum(int(draw[m].get("n_div", 0)) for m in range(n))
    ndiv_p = sum(int(pin_pm[m].get("n_div", 0)) for m in range(n))
    flags = []
    if ndiv_d:
        flags.append(f"draw arm has {ndiv_d} divergences (disclosed; pairs are NEVER dropped)")
    if ndiv_p:
        flags.append(f"pin arm has {ndiv_p} divergences (disclosed)")
    if abs(full_ref - FULL_EXPLANATION_PINNED) > FULL_TOL:
        flags.append(f"full-explanation ref recomputed {full_ref:+.4f} != registered "
                     f"{FULL_EXPLANATION_PINNED:+.4f} (tolerance {FULL_TOL}); wrong pin pkls?")

    return dict(n_pairs=n, primary=PRIMARY, paired=paired, cell=cell,
                materiality_M=MATERIALITY_M, full_explanation_ref=full_ref,
                full_in_ci=bool(lo <= full_ref <= hi),
                pin_pulls=pulls_p, draw_pulls=pulls_d, fres_draw=fres,
                finite_L=dict(draw=_finite_l_sd(Ld), pin=_finite_l_sd(Lp),
                              L_draw=Ld, L_pin=Lp),
                health=dict(draw_n_div_total=ndiv_d, pin_n_div_total=ndiv_p, flags=flags))


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        sys.exit(2)
    draw_dir, pin_dir = argv[1], argv[2]
    n = int(argv[3]) if len(argv) > 3 else N_PAIRS_DEFAULT
    out = run(draw_dir, pin_dir, expect_n=n)

    print(f"=== A3d PAIRED f_res PIN-vs-DRAW DIAGNOSTIC (n={out['n_pairs']} pairs, KS leg) ===")
    print(f"pin arm finite-L null sd {out['finite_L']['pin']:.4f} "
          f"(L {min(out['finite_L']['L_pin'])}-{max(out['finite_L']['L_pin'])}); "
          f"draw arm {out['finite_L']['draw']:.4f} "
          f"(L {min(out['finite_L']['L_draw'])}-{max(out['finite_L']['L_draw'])})")
    for ch in CHANNELS:
        p = out["paired"][ch]
        lab = "PRIMARY " if ch == PRIMARY else "secondary"
        print(f"  [{lab}] {ch:8s} delta {p['mean']:+.4f} +/- {p['sd']:.4f} "
              f"(sem {p['sem']:.4f}, t {p['t']:+.2f}, CI95 [{p['ci95'][0]:+.4f}, "
              f"{p['ci95'][1]:+.4f}], wilcoxon p {p['wilcoxon_p']:.4f}, rho {p['rho']:+.2f}, "
              f"{p['n_negative']}/{out['n_pairs']} negative)")
    print(f"  full-explanation reference (recomputed from the pin arm): "
          f"{out['full_explanation_ref']:+.4f}; inside the primary CI: {out['full_in_ci']}")
    for k, s in out["fres_draw"].items():
        print(f"  draw-arm {k}: pull {s['pull_mean']:+.3f} +/- {s['pull_sd']:.3f}, "
              f"rank KS-p {s['rank_ks_p']:.3f}, across-mock sd(post mean) {s['scatter']:.4f} "
              f"vs {s['scatter_expected']:.4f} expected under a drawn truth")
    for f in out["health"]["flags"]:
        print(f"  HEALTH FLAG: {f}")
    print(f"\nATTRIBUTION CELL (pre-registered, exhaustive; M = {out['materiality_M']}): "
          f"**{out['cell']}**")
    print("BINDING INTERPRETATION (PI #9 3f.3): a NULL cell may exclude f_res pinning as a")
    print("material explanation; a NON-NULL cell may support it as a contributor; NEITHER")
    print("distinguishes finite-sample noise from another genuine KS-specific mechanism;")
    print("this diagnostic is NOT a substitute for A3c. UNEXPECTED-DIRECTION returns to the PI.")


if __name__ == "__main__":
    main(sys.argv)
