#!/usr/bin/env python
"""PI #17 §12 BOUNDED n_s-EDGE DIAGNOSTIC.

Preregistered in `2026-08-08-NS-EDGE-DIAGNOSTIC-PREREG.md`. Reads ONLY committed per-mock
pkls, committed logs, and frozen artifacts. Runs NO fit, NO mock, NO posterior sampling.

THE QUESTION (PI #17 §12): did the divergence-triggered retry merely recover from
initialization-dependent numerical failure, or did it systematically select against a
scientifically supported high-curvature posterior region near the upper n_s boundary?

COORDINATE. Everything is in `theta_unit[0]`, the unit coordinate NUTS samples, in which the
n_s prior is EXACTLY Uniform(0,1) (`closure_legb.py:2650` + `data.py:32,:56`, both
[0.8, 1.05], so n_s occupies the full unit interval). Physical n_s = 0.8 + 0.25*theta_unit.
The unit coordinate is used because it is leg-invariant and INVARIANT TO THE BLIND
(`blinding.py:78-88` applies a post-hoc +-0.2165 offset to physical n_s only).

PRIMARY (one comparison, threshold-free, one-sided upper):
    T = mean of u over the 17 retried execution units,   u = theta_unit[0] of the truth.
NULL: permute the mock-index labels jointly over the shared block {0..47} across all three
arms, and independently over the KS-only block {48..95}. This is design-exact: mocks 0..47
carry byte-identical truths across arms AND share their NUTS base_seed, so an
independent-per-arm permutation would understate the null variance and be anti-conservative.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/scratch/cavestru_root/cavestru1/mfho/cert_2026-07"

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS. Every one is traceable to code or a frozen artifact; none is
# taken from a comment, from memory, or from post-hoc scanning (PI #17 sec-12).
# --------------------------------------------------------------------------- #
ARMS = {                                    # arm -> (pkl dir, n_mocks)
    "A1c_eBOSS": ("armp_eBOSS_corrected_v2", 48),
    "A2c_DESI": ("armp_DESI_corrected_v1", 48),
    "A3c_KS": ("armp_KS_corrected_v1", 96),
}
# Complete, from all 6701 SLURM logs; re-verified 2026-08-08. PI #17 sec-4 states the same.
RETRIED = {
    "A1c_eBOSS": [1, 4, 7, 10, 15, 37, 39, 41],
    "A2c_DESI": [4, 5, 7, 8, 14],
    "A3c_KS": [5, 25, 49, 73],
}
N_SHARED = 48                               # indices 0..47: truths identical across arms
NS_LO_PHYS, NS_HI_PHYS = 0.8, 1.05          # data.py:32 PARAM_LIMITS[0] == data.py:56
# Design-hull top, read from the sha256-pinned LF cache (60 unique 9-D design points).
DESIGN_MAX_NS_PHYS = 1.0395833
DESIGN_MAX_U = (DESIGN_MAX_NS_PHYS - NS_LO_PHYS) / (NS_HI_PHYS - NS_LO_PHYS)
# KS-only MF C_emu edge box, from the FROZEN artifact figures/analysis/04_emulator/
# mf_cemu_floor.npz key `ns_box` -- NOT from source and NOT the rejected 0.995 comment.
NS_BOX = (0.86, 0.98)
B_PERM = 200_000
ALPHA = 0.05
SEED = 20260808


def init_dist_analysis():
    """PI #17 sec-12 items 3 and 11, answered ANALYTICALLY -- no replay, no sampling.

    `_run_nuts_legb` uses NumPyro's default `init_to_median(num_samples=15)`. On the
    `theta_unit` site (`dist.Uniform(...).to_event(1)`, closure_legb.py:2650) that strategy
    draws 15 prior samples, takes the componentwise median in the UNCONSTRAINED coordinate,
    and maps back. The Uniform auto-bijector is the logit, which is monotone per component,
    so the unconstrained median equals the constrained median exactly.

    The median of 15 iid Uniform(0,1) draws is the 8th order statistic, which is EXACTLY
    Beta(8, 8). The initialization distribution is therefore known in closed form and does
    NOT depend on the seed -- so the per-attempt initialization values, which were never
    logged, are not needed to answer whether initialization can reach the upper n_s region.
    """
    from scipy import stats
    b = stats.beta(8, 8)
    n_init = 192 + 18                       # 192 first attempts + 18 retry attempts
    thresholds = {
        "u_0.7200_ns_0.98_KS_ns_box_top": 0.72,
        "u_0.7800_ns_0.995_REJECTED_comment_value": 0.78,
        "u_0.8000_ns_1.00": 0.80,
        "u_0.9583_design_hull_top": DESIGN_MAX_U,
        "u_0.9900_ns_1.0475": 0.99,
    }
    return dict(
        distribution="Beta(8,8) exactly (8th order statistic of 15 iid Uniform(0,1))",
        mean=float(b.mean()), sd=float(b.std()),
        n_campaign_initializations=int(n_init),
        tail_prob={k: float(b.sf(v)) for k, v in thresholds.items()},
        expected_count_over_campaign={k: float(n_init * b.sf(v))
                                      for k, v in thresholds.items()},
        conclusion=("The deployed initialization CANNOT place a chain in the upper n_s "
                    "support region: P(init theta_unit > design hull) = 4.5e-8, so the "
                    "expected number over all 210 campaign initializations is ~1e-5. The "
                    "'initializations enter the upper n_s region' mechanism is excluded by "
                    "the init strategy itself."),
        limitation=("This constrains the INITIALIZATION only. It does NOT exclude a NUTS "
                    "trajectory reaching the upper region during warmup or sampling and "
                    "diverging there. That variant is what the primary test addresses."))


def u_of_phys(ns):
    return (np.asarray(ns) - NS_LO_PHYS) / (NS_HI_PHYS - NS_LO_PHYS)


def phys_of_u(u):
    return NS_LO_PHYS + (NS_HI_PHYS - NS_LO_PHYS) * np.asarray(u)


def d_ns_of_u(u):
    """The EXACT quantity the implemented KS edge term uses (data_likelihood.py:868)."""
    ns = phys_of_u(u)
    return np.maximum(np.maximum(ns - NS_BOX[1], NS_BOX[0] - ns), 0.0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_arm(arm, root=SCRATCH):
    """Per-mock truth + posterior summaries for one arm. Returns dict m -> record."""
    d, n = ARMS[arm]
    out = {}
    for m in range(n):
        p = os.path.join(root, d, f"mock_{m:04d}.pkl")
        with open(p, "rb") as fh:
            z = pickle.load(fh)
        names = list(z["names"])
        i_ns = names.index("ns")
        truth = np.asarray(z["truth_vec"], float)
        draws = np.asarray(z["draws"], float)
        dns = draws[:, i_ns]
        extra = {k: float(v["truth"]) for k, v in z.get("sites_extra", {}).items()
                 if isinstance(v, dict) and v.get("truth") is not None}
        out[m] = dict(
            names=names, i_ns=i_ns, truth=truth, draws=draws, extra=extra,
            u=float(truth[i_ns]),
            n_div=int(z["n_div"]), L=int(z["L"]),
            post_mean=float(dns.mean()), post_sd=float(dns.std(ddof=1)),
            post_q95=float(np.quantile(dns, 0.95)),
            mass_above_hull=float((dns > DESIGN_MAX_U).mean()),
            mass_above_095=float((dns > 0.95).mean()),
            rank=float((dns < truth[i_ns]).mean()),
            pull=float((truth[i_ns] - dns.mean()) / dns.std(ddof=1)),
        )
    return out


# --------------------------------------------------------------------------- #
# The permutation machinery. The null permutes INDEX LABELS, jointly on the shared
# block and independently on the KS-only block. Formulating it on labels (rather than
# on truth values) makes it correct for parameters whose truths are NOT shared across
# arms -- which is what the S2 control family needs.
# --------------------------------------------------------------------------- #
def make_perms(rng, B, n_shared=N_SHARED, n_ksonly=48):
    """(B, n_shared) and (B, n_ksonly) label permutations."""
    P_sh = np.argsort(rng.random((B, n_shared)), axis=1)
    P_ks = np.argsort(rng.random((B, n_ksonly)), axis=1) + n_shared
    return P_sh, P_ks


def stat_mean_over_retried(score_by_arm, retried, P_sh=None, P_ks=None):
    """T = mean of `score` over retried execution units.

    score_by_arm: arm -> np.array indexed by mock.
    If P_sh/P_ks given (shape (B,*)), returns the (B,) permuted statistics; else a scalar.
    """
    if P_sh is None:
        tot, cnt = 0.0, 0
        for arm, ms in retried.items():
            s = score_by_arm[arm]
            for m in ms:
                tot += s[m]
                cnt += 1
        return tot / cnt
    B = P_sh.shape[0]
    tot = np.zeros(B)
    cnt = 0
    for arm, ms in retried.items():
        s = score_by_arm[arm]
        for m in ms:
            tot += s[P_sh[:, m]] if m < N_SHARED else s[P_ks[:, m - N_SHARED]]
            cnt += 1
    return tot / cnt


def perm_test(score_by_arm, retried, rng, B=B_PERM, side="upper", n_ksonly=48):
    obs = stat_mean_over_retried(score_by_arm, retried)
    P_sh, P_ks = make_perms(rng, B, n_ksonly=n_ksonly)
    null = stat_mean_over_retried(score_by_arm, retried, P_sh, P_ks)
    if side == "upper":
        r = int(np.sum(null >= obs))
    elif side == "lower":
        r = int(np.sum(null <= obs))
    else:
        r = int(np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())))
    p = (r + 1) / (B + 1)
    return dict(obs=float(obs), p=float(p), mc_se=float(np.sqrt(p * (1 - p) / B)),
                null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), B=int(B))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    run = 0.0
    for j, i in enumerate(idx):
        run = max(run, (n - j) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(root=SCRATCH, B=B_PERM, out_json=None):
    rng = np.random.default_rng(SEED)
    data = {a: load_arm(a, root=root) for a in ARMS}
    res = {"_meta": dict(B=B, alpha=ALPHA, seed=SEED,
                         coordinate="theta_unit[0] (n_s), Uniform(0,1) by design",
                         boundary="theta_unit=1.0 (n_s=1.05), hard prior support truncation",
                         design_max_u=DESIGN_MAX_U, ns_box=list(NS_BOX),
                         retried=RETRIED)}

    # --- structural checks that the design assumptions actually hold ------------- #
    u = {a: np.array([data[a][m]["u"] for m in range(ARMS[a][1])]) for a in ARMS}
    shared_ok = (np.allclose(u["A1c_eBOSS"][:N_SHARED], u["A2c_DESI"][:N_SHARED]) and
                 np.allclose(u["A1c_eBOSS"][:N_SHARED], u["A3c_KS"][:N_SHARED]))
    res["_meta"]["shared_truth_verified"] = bool(shared_ok)
    if not shared_ok:
        raise SystemExit("REFUSE: truths are NOT identical across arms on 0..47; the "
                         "joint-permutation null in the prereg is not applicable.")
    res["_meta"]["all_final_n_div_zero"] = bool(
        all(data[a][m]["n_div"] == 0 for a in ARMS for m in range(ARMS[a][1])))

    # --- initialization analysis (PI items 3 + 11), analytic, seed-free ----------- #
    res["init_strategy"] = init_dist_analysis()

    # --- PRIMARY ----------------------------------------------------------------- #
    res["primary"] = perm_test(u, RETRIED, rng, B=B, side="upper")
    res["primary"]["note"] = ("T = mean theta_unit[0] over the 17 retried units; one-sided "
                              "upper; joint index permutation on {0..47}, independent on "
                              "{48..95}. E[T|H0] = 0.5.")
    res["primary"]["reject_at_alpha"] = bool(res["primary"]["p"] <= ALPHA)

    # --- S1 per-arm --------------------------------------------------------------- #
    res["S1_per_arm"] = {}
    for a in ARMS:
        n = ARMS[a][1]
        sub = {a: u[a]}
        subr = {a: RETRIED[a]}
        r2 = np.random.default_rng(SEED + 1)
        if n == N_SHARED:                    # arm lives entirely in the shared block
            res["S1_per_arm"][a] = perm_test(sub, subr, r2, B=B, side="upper", n_ksonly=0)
        else:                                # A3c: permute all 96 within-arm
            P = np.argsort(r2.random((B, n)), axis=1)
            obs = float(np.mean([u[a][m] for m in RETRIED[a]]))
            null = np.mean([u[a][P[:, m]] for m in RETRIED[a]], axis=0)
            p = (int(np.sum(null >= obs)) + 1) / (B + 1)
            res["S1_per_arm"][a] = dict(obs=obs, p=float(p),
                                        mc_se=float(np.sqrt(p * (1 - p) / B)),
                                        null_mean=float(null.mean()), B=int(B))

    # --- S2 control family over every other sampled parameter --------------------- #
    # PACKED names (25) are common to all three arms -> pooled test.
    # sites_extra sets DIFFER per leg (A1c 11, A2c 15, A3c 10; A3c uses a different HCD
    # parameterization entirely). Only the four sites present in ALL arms can be pooled;
    # the rest are tested per-arm. Prereg v2 amendment, 2026-08-08, pre-inspection.
    names = data["A1c_eBOSS"][0]["names"]
    ctrl, pv, keys = {}, [], []
    Bc = max(B // 10, 20000)
    for j, nm in enumerate(names):
        sc = {a: np.array([data[a][m]["truth"][j] for m in range(ARMS[a][1])]) for a in ARMS}
        t = perm_test(sc, RETRIED, np.random.default_rng(SEED + 100 + j), B=Bc, side="two")
        ctrl[nm] = t
        pv.append(t["p"])
        keys.append(nm)
    common_extra = sorted(set.intersection(*[
        set(data[a][0]["extra"].keys()) for a in ARMS]))
    for j, nm in enumerate(common_extra):
        sc = {a: np.array([data[a][m]["extra"][nm] for m in range(ARMS[a][1])]) for a in ARMS}
        t = perm_test(sc, RETRIED, np.random.default_rng(SEED + 300 + j), B=Bc, side="two")
        ctrl["extra:" + nm] = t
        pv.append(t["p"])
        keys.append("extra:" + nm)
    adj = holm(np.array(pv))
    for k, nm in enumerate(keys):
        ctrl[nm]["p_holm_within_S2"] = float(adj[k])
    res["S2_control_family"] = ctrl
    res["S2_common_extra_sites"] = common_extra
    res["S2_note"] = ("Two-sided, since a control has no pre-declared direction. Holm applied "
                      "WITHIN S2 only; S2 can never establish the primary claim. Family = 25 "
                      "packed parameters + the %d sites_extra common to all arms."
                      % len(common_extra))

    # --- S2b arm-specific extra sites (per-arm nulls; NOT in the pooled family) ---- #
    res["S2b_arm_specific_extra"] = {}
    for a in ARMS:
        n = ARMS[a][1]
        own = sorted(set(data[a][0]["extra"].keys()) - set(common_extra))
        rows, pvs = {}, []
        r2 = np.random.default_rng(SEED + 400)
        P = np.argsort(r2.random((Bc, n)), axis=1)
        for nm in own:
            v = np.array([data[a][m]["extra"][nm] for m in range(n)])
            obs = float(np.mean([v[m] for m in RETRIED[a]]))
            null = np.mean([v[P[:, m]] for m in RETRIED[a]], axis=0)
            p = (int(np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) + 1) / (Bc + 1)
            rows[nm] = dict(obs=obs, p=float(p), null_mean=float(null.mean()))
            pvs.append(p)
        if pvs:
            adj2 = holm(np.array(pvs))
            for k, nm in enumerate(own):
                rows[nm]["p_holm_within_arm"] = float(adj2[k])
        res["S2b_arm_specific_extra"][a] = rows

    # --- S3 two-sided and distance-to-nearer-boundary ------------------------------ #
    res["S3_two_sided"] = perm_test(u, RETRIED, np.random.default_rng(SEED + 2), B=B,
                                    side="two")
    dist = {a: np.abs(u[a] - 0.5) for a in ARMS}
    res["S3_nearer_boundary"] = perm_test(dist, RETRIED, np.random.default_rng(SEED + 3),
                                          B=B, side="upper")

    # --- S4 KS-specific kink test (the ONLY use of the 0.98 box) ------------------- #
    a = "A3c_KS"
    dns = d_ns_of_u(u[a])
    r4 = np.random.default_rng(SEED + 4)
    P = np.argsort(r4.random((B, 96)), axis=1)
    obs = float(np.mean([dns[m] for m in RETRIED[a]]))
    null = np.mean([dns[P[:, m]] for m in RETRIED[a]], axis=0)
    p4 = (int(np.sum(null >= obs)) + 1) / (B + 1)
    res["S4_KS_kink"] = dict(obs=obs, p=float(p4), null_mean=float(null.mean()),
                             n_above_box=int((dns > 0).sum()), B=int(B),
                             note="score = d_ns = max(ns-0.98, 0.86-ns, 0), the exact "
                                  "quantity data_likelihood.py:868 uses. A3c only "
                                  "(mf_floor_on true on KS, false on DESI/eBOSS).")

    # --- S5 calibration behavior, retried vs non-retried --------------------------- #
    res["S5_calibration"] = {}
    for key in ("rank", "pull", "post_sd"):
        sc = {a: np.array([data[a][m][key] for m in range(ARMS[a][1])]) for a in ARMS}
        res["S5_calibration"][key] = perm_test(
            sc, RETRIED, np.random.default_rng(SEED + 10), B=B, side="two")
        if key == "pull":
            sc_abs = {a: np.abs(v) for a, v in sc.items()}
            res["S5_calibration"]["abs_pull"] = perm_test(
                sc_abs, RETRIED, np.random.default_rng(SEED + 11), B=B, side="upper")
    # partial: residualize the calibration score on truth u, then re-test (PI item 10)
    res["S5_partial_controlling_truth"] = {}
    for key in ("rank", "pull"):
        sc = {}
        for a in ARMS:
            y = np.array([data[a][m][key] for m in range(ARMS[a][1])])
            x = u[a]
            A = np.vstack([np.ones_like(x), x, x ** 2]).T
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            sc[a] = y - A @ beta
        res["S5_partial_controlling_truth"][key] = perm_test(
            sc, RETRIED, np.random.default_rng(SEED + 12), B=B, side="two")
    res["S5_partial_note"] = ("Residual of the calibration score on a quadratic in truth "
                              "theta_unit[0], per arm; the permutation null is unchanged.")

    # --- S6 posterior boundary behavior -------------------------------------------- #
    res["S6_boundary"] = {}
    for key in ("post_mean", "post_q95", "mass_above_hull", "mass_above_095"):
        sc = {a: np.array([data[a][m][key] for m in range(ARMS[a][1])]) for a in ARMS}
        res["S6_boundary"][key] = perm_test(
            sc, RETRIED, np.random.default_rng(SEED + 20), B=B, side="two")
        # residualized on truth (PI item 12: "beyond what truth location predicts")
        scr = {}
        for a in ARMS:
            y, x = sc[a], u[a]
            A = np.vstack([np.ones_like(x), x, x ** 2]).T
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            scr[a] = y - A @ beta
        res["S6_boundary"][key + "_resid_on_truth"] = perm_test(
            scr, RETRIED, np.random.default_rng(SEED + 21), B=B, side="two")
    res["S6_limitation"] = ("BINDING: the discarded chains do not exist. S6 compares "
                            "retained-retried against retained-non-retried and CANNOT show "
                            "that a retained chain avoids mass its own discarded predecessor "
                            "visited. No causal claim of selection is made from S6.")

    # --- S7 cross-arm concordance (descriptive only) -------------------------------- #
    sets = {a: set(m for m in RETRIED[a] if m < N_SHARED) for a in ARMS}
    pairs = {}
    for x in ARMS:
        for y in ARMS:
            if x < y:
                pairs[f"{x}|{y}"] = sorted(sets[x] & sets[y])
    r7 = np.random.default_rng(SEED + 30)
    obs_ov = sum(len(v) for v in pairs.values())
    null_ov = np.zeros(B)
    ks = {a: len(sets[a]) for a in ARMS}
    for b in range(B):
        draw = {a: set(r7.choice(N_SHARED, size=ks[a], replace=False)) for a in ARMS}
        null_ov[b] = sum(len(draw[x] & draw[y]) for x in ARMS for y in ARMS if x < y)
    res["S7_concordance"] = dict(
        pairwise_overlap=pairs, total_overlap=int(obs_ov),
        p=float((int(np.sum(null_ov >= obs_ov)) + 1) / (B + 1)),
        null_mean=float(null_ov.mean()),
        note="DESCRIPTIVE ONLY. Truth and base_seed are perfectly confounded across arms at "
             "matched index, so concordance cannot separate 'same hard truth' from 'same "
             "unlucky seed'. Never used to support Case B.")

    # --- per-unit table ------------------------------------------------------------- #
    res["per_unit"] = {
        a: [dict(mock=m, retried=bool(m in RETRIED[a]), u=data[a][m]["u"],
                 ns_phys=float(phys_of_u(data[a][m]["u"])),
                 rank=data[a][m]["rank"], pull=data[a][m]["pull"],
                 post_mean=data[a][m]["post_mean"], post_sd=data[a][m]["post_sd"],
                 post_q95=data[a][m]["post_q95"],
                 mass_above_hull=data[a][m]["mass_above_hull"], L=data[a][m]["L"])
            for m in range(ARMS[a][1])]
        for a in ARMS}

    if out_json:
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=False)
    return res


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    r = run(out_json=out)
    p = r["primary"]
    print(f"PRIMARY  T={p['obs']:.4f}  (E[T|H0]={p['null_mean']:.4f})  "
          f"p={p['p']:.5f} +- {p['mc_se']:.5f}  reject={p['reject_at_alpha']}")
    for a, v in r["S1_per_arm"].items():
        print(f"  S1 {a:10s} T={v['obs']:.4f}  p={v['p']:.5f}")
    print(f"  S4 KS kink  d_ns={r['S4_KS_kink']['obs']:.5f}  p={r['S4_KS_kink']['p']:.5f}")
