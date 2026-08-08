#!/usr/bin/env python
"""PI #17 §12 BOUNDED n_s-EDGE DIAGNOSTIC.

Preregistered in `2026-08-08-NS-EDGE-DIAGNOSTIC-PREREG.md` (v3, both referee passes applied).
Reads ONLY committed per-mock pkls, committed logs, and frozen artifacts. Runs NO fit, NO
mock, NO posterior sampling.

THE QUESTION (PI #17 §12): did the divergence-triggered retry merely recover from
initialization-dependent numerical failure, or did it systematically select against a
scientifically supported high-curvature posterior region near the upper n_s boundary?

COORDINATE. Everything is in `theta_unit[0]`, the unit coordinate NUTS samples, in which the
n_s prior is EXACTLY Uniform(0,1) (`closure_legb.py:2650` + `data.py:32,:56`, both
[0.8, 1.05], so n_s occupies the full unit interval). Physical n_s = 0.8 + 0.25*theta_unit.
The unit coordinate is used because it is leg-invariant and INVARIANT TO THE BLIND
(`blinding.py:75-88` applies a post-hoc +-0.2165 offset to physical n_s only).

PRIMARY (m=1, threshold-free, one-sided upper):
    T = mean of u over the 17 retried execution units,   u = theta_unit[0] of the truth.
NULL: permute mock-index labels jointly over the shared block {0..47} across all three arms,
independently over {48..95}. Design-exact: mocks 0..47 carry byte-identical truths across arms
AND share their NUTS base_seed, so an independent-per-arm permutation would understate the null
variance (Sum w = 17 vs the true Sum w^2 = 23) and be ANTI-CONSERVATIVE.

n_eff = 17^2/23 = 12.57 over 14 distinct truths; exact null sd of the mean = 0.08144.

CASE RULE (prereg v4.2): a closed, ordered 7-member trigger family; Case B iff any raw
p <= 0.05, with Holm over the family also reported; Case A iff none fires; Case C iff the
step-0 validity gate fails. Measured P(Case B | global null) = 0.151 raw / 0.031 Holm (1500 replicates).
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/scratch/cavestru_root/cavestru1/mfho/cert_2026-07"

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS. Each is traceable to code or a frozen artifact; none is taken
# from a comment, from memory, or from post-hoc scanning (PI #17 §12).
# --------------------------------------------------------------------------- #
ARMS = {
    "A1c_eBOSS": ("armp_eBOSS_corrected_v2", 48),
    "A2c_DESI": ("armp_DESI_corrected_v1", 48),
    "A3c_KS": ("armp_KS_corrected_v1", 96),
}
RETRIED = {                                  # complete, from all 6701 SLURM logs
    "A1c_eBOSS": [1, 4, 7, 10, 15, 37, 39, 41],
    "A2c_DESI": [4, 5, 7, 8, 14],
    "A3c_KS": [5, 25, 49, 73],
}
# attempt-0 divergence counts (S10). SELECTION-FREE: attempt 0 ran unconditionally for all 192.
NDIV0 = {
    "A1c_eBOSS": {1: 1, 4: 2, 7: 1, 10: 1, 15: 3, 37: 10, 39: 1, 41: 1},
    "A2c_DESI": {4: 2, 5: 1, 7: 6, 8: 1, 14: 1},
    "A3c_KS": {5: 1, 25: 6, 49: 2, 73: 1},
}
# retained target_accept per unit (prereg v3.5 C1: PERFECTLY confounded with retry status)
TA_RETAINED_DEFAULT = 0.90
TA_RETAINED = {"A1c_eBOSS": {m: 0.95 for m in RETRIED["A1c_eBOSS"]},
               "A2c_DESI": {m: 0.95 for m in RETRIED["A2c_DESI"]},
               "A3c_KS": {5: 0.95, 25: 0.99, 49: 0.95, 73: 0.95}}
JOB_BLOCK = {                                # S9 execution-environment (prereg v3.5 C2)
    "A1c_eBOSS": [(0, 48, "56498747")],
    "A2c_DESI": [(0, 12, "56612160_pilot"), (12, 48, "56636183_array")],
    "A3c_KS": [(0, 48, "56590250_t1"), (48, 96, "56605660_t2")],
}
N_SHARED = 48
NS_LO_PHYS, NS_HI_PHYS = 0.8, 1.05
DESIGN_MAX_NS_PHYS = 1.0395833                # 60-point design hull, sha256-pinned LF cache
DESIGN_MAX_U = (DESIGN_MAX_NS_PHYS - NS_LO_PHYS) / (NS_HI_PHYS - NS_LO_PHYS)
NS_BOX = (0.86, 0.98)                         # frozen mf_cemu_floor.npz `ns_box`; KS-only
B_PERM = 200_000
ALPHA = 0.05
SEED = 20260808
KS_GATE_P = 0.001                             # prereg v3.6 validity gate


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def phys_of_u(u):
    return NS_LO_PHYS + (NS_HI_PHYS - NS_LO_PHYS) * np.asarray(u)


def u_of_phys(ns):
    return (np.asarray(ns) - NS_LO_PHYS) / (NS_HI_PHYS - NS_LO_PHYS)


def d_ns_of_u(u):
    """The EXACT quantity data_likelihood.py:868 uses. Symmetric about the box."""
    ns = phys_of_u(u)
    return np.maximum(np.maximum(ns - NS_BOX[1], NS_BOX[0] - ns), 0.0)


def score_mean(u):
    return np.asarray(u)


def score_tail(u):
    """S8 tail-weighted edge score. Sensitive to PROXIMITY, not just a mean shift."""
    return -np.log1p(-np.clip(np.asarray(u), 0.0, 1 - 1e-12))


def score_hull(u):
    """S_HULL (physics referee M5). d_hull = max(u - 0.9583, 0): the exact structural
    analogue of S4, for the 60-point training-design hull. Promoted from "out of scope":
    C_emu is BLIND to design distance on A1c/A2c, so above the hull the likelihood is as
    confident as anywhere while the emulator is unconstrained. One-sided upper and all-arm,
    i.e. this is the mechanism the primary is actually powered against."""
    return np.maximum(np.asarray(u) - DESIGN_MAX_U, 0.0)


def score_absdev(u):
    """S3 distance to the NEARER boundary. The primary's blind spot is S3's strong suit."""
    return np.abs(np.asarray(u) - 0.5)


def init_dist_analysis():
    """PI §12 items 3 and 11, ANALYTICALLY -- no replay, no sampling (prereg v3.4).

    `_run_nuts_legb:3307` sets `strat = init_to_median` when none is passed, and `run_legb`
    never passes one. NOTE: `init_to_median` is THIS WRAPPER's default; NumPyro's own default
    is `init_to_uniform`. On the `theta_unit` site it takes the componentwise median of 15
    prior draws under a monotone bijector, so theta_unit[0] at init ~ Beta(8,8) EXACTLY --
    closed form, seed-independent, so the never-logged per-attempt inits are not needed.
    """
    from scipy import stats
    b = stats.beta(8, 8)
    n_init = 192 + 18
    thr = {"u_0.7200_ns_0.98_KS_ns_box_UPPER": 0.72,
           "u_0.7800_ns_0.995_REJECTED_comment_value": 0.78,
           "u_0.8000_ns_1.00": 0.80,
           "u_0.9583_design_hull_top": DESIGN_MAX_U,
           "u_0.9900_ns_1.0475": 0.99}
    return dict(
        distribution="Beta(8,8) exactly (8th order statistic of 15 iid Uniform(0,1))",
        mean=float(b.mean()), sd=float(b.std()), n_campaign_initializations=int(n_init),
        tail_prob={k: float(b.sf(v)) for k, v in thr.items()},
        expected_count={k: float(n_init * b.sf(v)) for k, v in thr.items()},
        below_ks_box_lower=dict(u=0.24, prob=float(b.cdf(0.24)),
                                expected_count=float(n_init * b.cdf(0.24))),
        conclusion=("The 'initialization enters the upper n_s region' form of H_edge is "
                    "ANALYTICALLY REFUTED at the design hull and the hard prior boundary "
                    "(P ~ 4.5e-8 per attempt, ~1e-5 expected over the whole campaign). It "
                    "remains live ONLY near the KS ns_box edges, where ~10 of 210 attempts "
                    "land outside [0.24, 0.72]."),
        limitation=("Constrains the INITIALIZATION only. Does NOT exclude a NUTS trajectory "
                    "reaching the upper region during warmup/sampling and diverging there. "
                    "That variant is what the primary and S8 address."))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_arm(arm, root=SCRATCH):
    d, n = ARMS[arm]
    out = {}
    for m in range(n):
        with open(os.path.join(root, d, f"mock_{m:04d}.pkl"), "rb") as fh:
            z = pickle.load(fh)
        names = list(z["names"])
        i_ns = names.index("ns")
        truth = np.asarray(z["truth_vec"], float)
        draws = np.asarray(z["draws"], float)
        dns = draws[:, i_ns]
        extra = {k: float(v["truth"]) for k, v in z.get("sites_extra", {}).items()
                 if isinstance(v, dict) and v.get("truth") is not None}
        out[m] = dict(
            names=names, i_ns=i_ns, truth=truth, extra=extra, u=float(truth[i_ns]),
            n_div=int(z["n_div"]), L=int(z["L"]),
            post_mean=float(dns.mean()), post_sd=float(dns.std(ddof=1)),
            post_q95=float(np.quantile(dns, 0.95)),
            mass_above_hull=float((dns > DESIGN_MAX_U).mean()),
            mass_above_095=float((dns > 0.95).mean()),
            rank=float((dns < truth[i_ns]).mean()),
            pull=float((truth[i_ns] - dns.mean()) / dns.std(ddof=1)))
    return out


# --------------------------------------------------------------------------- #
# permutation machinery -- permutes INDEX LABELS, so it is correct for parameters
# whose truths are NOT shared across arms (the S2 control family needs that).
# --------------------------------------------------------------------------- #
def make_perms(rng, B, n_ksonly=48):
    P_sh = np.argsort(rng.random((B, N_SHARED)), axis=1)
    P_ks = (np.argsort(rng.random((B, n_ksonly)), axis=1) + N_SHARED
            if n_ksonly else np.zeros((B, 0), int))
    return P_sh, P_ks


def stat_mean_over_retried(score_by_arm, retried, P_sh=None, P_ks=None):
    if P_sh is None:
        tot, cnt = 0.0, 0
        for arm, ms in retried.items():
            for m in ms:
                tot += score_by_arm[arm][m]
                cnt += 1
        return tot / cnt
    tot = np.zeros(P_sh.shape[0])
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


def arm_perm_test(vals, retried_idx, rng, B=B_PERM, side="upper"):
    """Within-arm null over the arm's FULL index range (prereg v3.10)."""
    n = len(vals)
    P = np.argsort(rng.random((B, n)), axis=1)
    obs = float(np.mean([vals[m] for m in retried_idx]))
    null = np.mean([vals[P[:, m]] for m in retried_idx], axis=0)
    if side == "upper":
        r = int(np.sum(null >= obs))
    else:
        r = int(np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())))
    p = (r + 1) / (B + 1)
    return dict(obs=obs, p=float(p), mc_se=float(np.sqrt(p * (1 - p) / B)),
                null_mean=float(null.mean()), B=int(B))


def holm(pvals):
    p = np.asarray(pvals, float)
    idx = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    run = 0.0
    for j, i in enumerate(idx):
        run = max(run, (n - j) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def pit_rank(v):
    """Common-scale score for the S2 family (prereg v3.7): within-arm PIT via rank."""
    v = np.asarray(v, float)
    order = v.argsort().argsort()
    return (order + 0.5) / len(v)


# --------------------------------------------------------------------------- #
# step 0 -- validity gate (prereg v3.6). REFUSE-ON-FAIL.
# --------------------------------------------------------------------------- #
def validity_gate(data):
    from scipy import stats
    u = {a: np.array([data[a][m]["u"] for m in range(ARMS[a][1])]) for a in ARMS}
    shared_ok = bool(np.array_equal(u["A1c_eBOSS"][:N_SHARED], u["A2c_DESI"][:N_SHARED]) and
                     np.array_equal(u["A1c_eBOSS"][:N_SHARED], u["A3c_KS"][:N_SHARED]))
    u96 = np.concatenate([u["A1c_eBOSS"][:N_SHARED], u["A3c_KS"][N_SHARED:]])
    in_box = bool(np.all((u96 > 0.0) & (u96 < 1.0)))
    ks = stats.kstest(u96, "uniform")
    g = dict(n_distinct_truths=int(u96.size), all_in_unit_interval=in_box,
             shared_truths_identical_across_arms=shared_ok,
             ks_vs_uniform_stat=float(ks.statistic), ks_vs_uniform_p=float(ks.pvalue),
             ks_threshold=KS_GATE_P,
             passed=bool(in_box and shared_ok and ks.pvalue >= KS_GATE_P),
             note=("Exactness of the permutation test comes from the RANDOMIZATION, not from "
                   "uniformity of u. A marginal KS bears on interpretation of the score, not "
                   "on validity of the p-value."))
    return g, u, u96


def verbose_check():
    """Prereg v3.10: the retry lines survive only because verbose printed them."""
    out = {}
    for job in ("56498747", "56590250", "56605660", "56612160", "56636183"):
        try:
            n = subprocess.run(
                f"grep -l 'divergence(s) at ta=\\|wrote ->' {REPO}/logs/armp_{job}_*.out "
                f"2>/dev/null | wc -l", shell=True, capture_output=True, text=True).stdout
            out[job] = int(n.strip())
        except Exception:
            out[job] = None
    return dict(logs_with_verbose_output=out,
                note="A job that ran quiet would hide its retries; all five printed.")


# --------------------------------------------------------------------------- #
def run(root=SCRATCH, B=B_PERM, out_json=None):
    rng = np.random.default_rng(SEED)
    data = {a: load_arm(a, root=root) for a in ARMS}
    res = {"_meta": dict(
        B=B, alpha=ALPHA, seed=SEED, prereg="2026-08-08-NS-EDGE-DIAGNOSTIC-PREREG.md v3",
        coordinate="theta_unit[0] (n_s), Uniform(0,1) by design",
        boundary="theta_unit=1.0 (n_s=1.05), hard prior support truncation",
        design_max_u=DESIGN_MAX_U, ns_box=list(NS_BOX), retried=RETRIED,
        n_eff=17 ** 2 / 23.0, null_sd_mean_stat=float(np.sqrt(23 / 12) / 17),
        n_distinct_truths_behind_17_units=14)}

    # ---- STEP 0: validity gate ------------------------------------------------ #
    gate, u, u96 = validity_gate(data)
    res["step0_validity_gate"] = gate
    res["init_strategy"] = init_dist_analysis()
    res["verbose_check"] = verbose_check()
    if not gate["passed"]:
        res["CASE"] = "C"
        res["case_reason"] = "step-0 validity gate FAILED; nothing further computed."
        if out_json:
            json.dump(res, open(out_json, "w"), indent=1)
        return res
    res["_meta"]["all_final_n_div_zero"] = bool(
        all(data[a][m]["n_div"] == 0 for a in ARMS for m in range(ARMS[a][1])))

    # ---- PRIMARY + S8 + S3 (pooled, one-sided upper) -------------------------- #
    res["primary"] = perm_test({a: score_mean(u[a]) for a in ARMS}, RETRIED,
                               np.random.default_rng(SEED), B=B, side="upper")
    res["primary"]["note"] = ("T = mean theta_unit[0] over the 17 retried units; one-sided "
                              "upper; joint index permutation. E[T|H0] = 0.5. Power 0.87 vs a "
                              "one-sided all-arm alternative, but only 0.09 vs a SYMMETRIC "
                              "edge alternative and 0.45 vs a KS-only effect (prereg v3.1).")
    res["S_HULL_design_hull"] = perm_test({a: score_hull(u[a]) for a in ARMS}, RETRIED,
                                          np.random.default_rng(SEED + 5), B=B, side="upper")
    from scipy import stats as _st
    _n_above = int((u96 > DESIGN_MAX_U).sum())
    res["S_HULL_design_hull"].update(
        n_truths_above_hull=_n_above, n_distinct_truths=int(u96.size),
        expected_above_among_14_retried_truths=float(14 * _n_above / u96.size),
        p_none_above_given_H0=float(_st.hypergeom.pmf(0, u96.size, _n_above, 14)),
        POWER_CAVEAT=("With only %d of %d truths above the hull, the expected count among the "
                      "14 distinct retried truths is %.2f, and P(none above | H0) = %.3f. A "
                      "statistic pinned at its floor gives a DEGENERATE permutation p of 1.0. "
                      "S_HULL is therefore UNINFORMATIVE here, NOT a refutation of the hull "
                      "mechanism. This is the opportunity ceiling the prereg warned about."
                      % (_n_above, u96.size, 14 * _n_above / u96.size,
                         _st.hypergeom.pmf(0, u96.size, _n_above, 14))))
    res["S8_tail_edge_score"] = perm_test({a: score_tail(u[a]) for a in ARMS}, RETRIED,
                                          np.random.default_rng(SEED + 8), B=B, side="upper")
    res["S3_nearer_boundary"] = perm_test({a: score_absdev(u[a]) for a in ARMS}, RETRIED,
                                          np.random.default_rng(SEED + 3), B=B, side="upper")
    res["S3_two_sided_mean"] = perm_test({a: score_mean(u[a]) for a in ARMS}, RETRIED,
                                         np.random.default_rng(SEED + 2), B=B, side="two")

    # ---- S1 per-arm ----------------------------------------------------------- #
    res["S1_per_arm"] = {a: arm_perm_test(u[a], RETRIED[a],
                                          np.random.default_rng(SEED + 1), B=B, side="upper")
                         for a in ARMS}
    s1_holm = holm([res["S1_per_arm"][a]["p"] for a in ARMS])
    for k, a in enumerate(ARMS):
        res["S1_per_arm"][a]["p_holm_within_S1"] = float(s1_holm[k])
    res["S1_holm_min"] = float(min(s1_holm))

    # ---- S4 KS-only kink test (the ONLY use of the 0.98 box) ------------------- #
    a = "A3c_KS"
    res["S4_KS_kink"] = arm_perm_test(d_ns_of_u(u[a]), RETRIED[a],
                                      np.random.default_rng(SEED + 4), B=B, side="upper")
    res["S4_KS_kink"].update(
        n_truths_outside_box=int((d_ns_of_u(u[a]) > 0).sum()),
        note=("score = d_ns = max(ns-0.98, 0.86-ns, 0), exactly data_likelihood.py:868. A3c "
              "only: mf_floor_on true on KS, false on DESI/eBOSS. d_ns is SYMMETRIC about "
              "the box, which is why S3 is also a trigger."))

    # ---- S5 calibration, retried vs non-retried ------------------------------- #
    res["S5_calibration"] = {}
    for key in ("rank", "pull", "post_sd"):
        sc = {a: np.array([data[a][m][key] for m in range(ARMS[a][1])]) for a in ARMS}
        res["S5_calibration"][key] = perm_test(sc, RETRIED,
                                               np.random.default_rng(SEED + 10), B=B,
                                               side="two")
    s5_holm = holm([res["S5_calibration"][k]["p"] for k in ("rank", "pull", "post_sd")])
    for k, key in enumerate(("rank", "pull", "post_sd")):
        res["S5_calibration"][key]["p_holm_within_S5"] = float(s5_holm[k])
    res["S5_holm_min"] = float(min(s5_holm))
    res["S5_partial_controlling_truth"] = {}
    for key in ("rank", "pull"):
        sc = {}
        for a in ARMS:
            y = np.array([data[a][m][key] for m in range(ARMS[a][1])])
            A = np.vstack([np.ones_like(u[a]), u[a], u[a] ** 2]).T
            sc[a] = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        res["S5_partial_controlling_truth"][key] = perm_test(
            sc, RETRIED, np.random.default_rng(SEED + 12), B=B, side="two")
    res["S5_LIMITATION"] = (
        "BINDING (prereg v3.5 C1): target_accept is PERFECTLY confounded with retry status in "
        "the retained sample -- every retained retried chain ran at ta=0.95 (0.99 for A3c mock "
        "25), every retained non-retried chain at ta=0.90. Any S5/S6 difference is attributable "
        "to selection OR to the sampler setting and the two CANNOT be separated from what was "
        "persisted. Direction note: higher target_accept should if anything explore difficult "
        "geometry BETTER, i.e. opposite to the selection hypothesis.")

    # ---- S6 posterior boundary behaviour --------------------------------------- #
    res["S6_boundary"] = {}
    for key in ("post_mean", "post_q95", "mass_above_hull", "mass_above_095"):
        sc = {a: np.array([data[a][m][key] for m in range(ARMS[a][1])]) for a in ARMS}
        res["S6_boundary"][key] = perm_test(sc, RETRIED, np.random.default_rng(SEED + 20),
                                            B=B, side="two")
        scr = {}
        for a in ARMS:
            A = np.vstack([np.ones_like(u[a]), u[a], u[a] ** 2]).T
            scr[a] = sc[a] - A @ np.linalg.lstsq(A, sc[a], rcond=None)[0]
        res["S6_boundary"][key + "_resid_on_truth"] = perm_test(
            scr, RETRIED, np.random.default_rng(SEED + 21), B=B, side="two")
    res["S6_LIMITATION"] = (
        "BINDING: the discarded chains do not exist. S6 compares retained-retried against "
        "retained-non-retried and CANNOT show that a retained chain avoids mass its own "
        "discarded predecessor visited. No causal claim of selection is made from S6. The "
        "target_accept confound in S5_LIMITATION applies here too.")

    # ---- S2 control family (repaired, prereg v3.7) ----------------------------- #
    names = data["A1c_eBOSS"][0]["names"]
    keep = [j for j, nm in enumerate(names) if not nm.startswith("tau0_z")]
    common_extra = sorted(set.intersection(*[set(data[a][0]["extra"].keys()) for a in ARMS]))
    Bc = max(B // 10, 20000)
    ctrl, pv, keys = {}, [], []
    for j in keep:
        sc = {a: pit_rank([data[a][m]["truth"][j] for m in range(ARMS[a][1])]) for a in ARMS}
        t = perm_test(sc, RETRIED, np.random.default_rng(SEED + 100 + j), B=Bc, side="two")
        ctrl[names[j]] = t
        pv.append(t["p"])
        keys.append(names[j])
    for j, nm in enumerate(common_extra):
        sc = {a: pit_rank([data[a][m]["extra"][nm] for m in range(ARMS[a][1])]) for a in ARMS}
        t = perm_test(sc, RETRIED, np.random.default_rng(SEED + 300 + j), B=Bc, side="two")
        ctrl["extra:" + nm] = t
        pv.append(t["p"])
        keys.append("extra:" + nm)
    adj = holm(np.array(pv))
    for k, nm in enumerate(keys):
        ctrl[nm]["p_holm_within_S2"] = float(adj[k])
    # "distinguished" decision rule: rank of the primary's p among the family
    fam_p = [res["primary"]["p"]] + pv
    ns_rank = int(np.sum(np.array(fam_p) <= res["primary"]["p"]))
    res["S2_control_family"] = ctrl
    res["S2_membership"] = dict(packed_kept=[names[j] for j in keep],
                                dropped_tau0_rungs=[names[j] for j in range(len(names))
                                                    if names[j].startswith("tau0_z")],
                                common_extra=common_extra)
    res["S2_distinguished"] = dict(
        m=len(fam_p), ns_p=res["primary"]["p"], ns_rank=ns_rank,
        exact_null_prob=float(ns_rank / len(fam_p)),
        note=("Rank of n_s's p among the m = 1 + |S2| p-values; exact null P(rank <= r) = r/m "
              "under prior independence of the sampled sites. The 13 tau0 rungs are EXCLUDED "
              "per PI #17 §3.2 (rank-2 derived coordinate) and replaced by tau0_amp/dtau0. "
              "All members scored by within-arm PIT so the family is on a common scale."))

    # ---- S_MEANFLUX + S_HCDFUNNEL: degeneracy / funnel directions (referee M7) --- #
    mf = {}
    for nm in ("tau0_amp", "dtau0"):
        sc = {a: pit_rank([data[a][m]["extra"][nm] for m in range(ARMS[a][1])]) for a in ARMS}
        mf[nm] = perm_test(sc, RETRIED, np.random.default_rng(SEED + 50), B=B, side="two")
    sc = {a: pit_rank([np.hypot(data[a][m]["extra"]["tau0_amp"] - 1.0,
                                data[a][m]["extra"]["dtau0"]) for m in range(ARMS[a][1])])
          for a in ARMS}
    mf["joint_radius"] = perm_test(sc, RETRIED, np.random.default_rng(SEED + 51), B=B,
                                   side="upper")
    mf["_note"] = ("Mean-flux sector. Referee M7: the largest IMPLEMENTED C1 violations are "
                   "here -- jnp.interp on a SAMPLED abscissa with only 4 knots "
                   "(likelihood.py:121,:149) plus the 20-rung MF ladder "
                   "(multifidelity.py:463,:471,:476), ~50 gradient-discontinuity surfaces in "
                   "the (tau0_amp, dtau0) plane, ACTIVE ON ALL THREE ARMS. Also the arm's own "
                   "frozen Row-3 trigger was the rotated tau0_amp rank limb (p=0.0449).")
    res["S_MEANFLUX"] = mf
    ia = [data["A1c_eBOSS"][0]["names"].index(x)
          for x in ("alpha_lls", "alpha_subdla", "alpha_dla")]
    sc = {}
    for a in ARMS:
        M = np.array([[data[a][m]["truth"][j] for j in ia] for m in range(ARMS[a][1])])
        sd = M.std(axis=0, ddof=1)
        sc[a] = np.min(M / np.maximum(sd, 1e-30), axis=1)
    res["S_HCDFUNNEL"] = perm_test(sc, RETRIED, np.random.default_rng(SEED + 52), B=B,
                                   side="lower")
    res["S_HCDFUNNEL"]["note"] = (
        "min_c[alpha_c / sd_c] over the three HCD pivots, ONE-SIDED LOWER (the funnel NECK is "
        "at small alpha). Referee M7: the HCD block is a multiplicative amplitude x "
        "((1+z)/4)^slope construction with the slope marginalised -- a Neal-type funnel -- and "
        "on DESI/eBOSS Sum(alpha) is unconstrained so the clean-sightline coefficient can go "
        "negative. A far more standard divergence generator than any n_s-boundary effect, and "
        "arm-common.")

    # ---- S9 execution-environment association (prereg v3.5 C2) ----------------- #
    res["S9_execution_environment"] = {}
    for a, blocks in JOB_BLOCK.items():
        res["S9_execution_environment"][a] = [
            dict(job=lab, lo=lo, hi=hi, n=hi - lo,
                 n_retried=int(sum(lo <= m < hi for m in RETRIED[a])),
                 rate=float(sum(lo <= m < hi for m in RETRIED[a]) / (hi - lo)))
            for lo, hi, lab in blocks]
    fe = {}
    for lab, tab in (("A2c_pilot_vs_array", [[4, 8], [1, 35]]),
                     ("A1c_vs_A3c", [[8, 40], [4, 92]]),
                     ("A1c_vs_A2c", [[8, 40], [5, 43]]),
                     ("A2c_vs_A3c", [[5, 43], [4, 92]])):
        fe[lab] = float(_st.fisher_exact(tab)[1])
    res["S9_fisher_two_sided"] = fe
    res["S9_note"] = ("DESCRIPTIVE. Job assignment is a function of mock index and index is "
                      "independent of truth, so this cannot threaten the primary's validity -- "
                      "but it is a live competing explanation Case A would otherwise absorb.")

    # ---- S10 attempt-0 divergence count, SELECTION-FREE ------------------------ #
    sc = {a: np.array([NDIV0[a].get(m, 0) for m in range(ARMS[a][1])], float) for a in ARMS}
    allu = {a: u[a] for a in ARMS}
    tot = sum(v.sum() for v in sc.values())
    wsum = 0.0
    for a in ARMS:
        wsum += float(np.dot(sc[a], allu[a]))
    obs = wsum / tot
    P_sh, P_ks = make_perms(np.random.default_rng(SEED + 40), B)
    null = np.zeros(B)
    for a in ARMS:
        s = allu[a]
        for m in range(ARMS[a][1]):
            if sc[a][m]:
                null += sc[a][m] * (s[P_sh[:, m]] if m < N_SHARED
                                    else s[P_ks[:, m - N_SHARED]])
    null /= tot
    res["S10_attempt0_ndiv_weighted"] = dict(
        obs=float(obs), p=float((int(np.sum(null >= obs)) + 1) / (B + 1)),
        null_mean=float(null.mean()), total_ndiv0=float(tot), B=int(B),
        note=("Divergence-count-weighted mean of u. SELECTION-FREE: attempt 0 ran "
              "unconditionally for all 192 units. Secondary, NOT a Case-B trigger -- mass "
              "concentrates on A1c mock 37 (10 div) and A2c mock 7 (6 div)."))

    # ---- S7 cross-arm concordance (descriptive only) --------------------------- #
    sets = {a: set(m for m in RETRIED[a] if m < N_SHARED) for a in ARMS}
    pairs = {f"{x}|{y}": sorted(sets[x] & sets[y])
             for x in ARMS for y in ARMS if x < y}
    obs_ov = sum(len(v) for v in pairs.values())
    r7 = np.random.default_rng(SEED + 30)
    ks_n = {a: len(sets[a]) for a in ARMS}
    null_ov = np.array([
        sum(len(d[x] & d[y]) for x in ARMS for y in ARMS if x < y)
        for d in ({a: set(r7.choice(N_SHARED, size=ks_n[a], replace=False)) for a in ARMS}
                  for _ in range(20000))])
    res["S7_concordance"] = dict(
        pairwise_overlap=pairs, total_overlap=int(obs_ov),
        p=float((int(np.sum(null_ov >= obs_ov)) + 1) / 20001), null_mean=float(null_ov.mean()),
        note=("DESCRIPTIVE ONLY. Cannot separate 'same hard truth' from 'same unlucky seed'. "
              "NOTE (prereg v3.8): within a unit k_truth and k_nuts are INDEPENDENT split "
              "outputs of fold_in(key0,m) (closure_legb.py:3811), so u_m is independent of "
              "base_seed_m. The confound is a cross-arm attribution problem for S7 ONLY; it "
              "cannot contaminate the primary."))

    # ---- A2c worst-case influence bound (descriptive only) --------------------- #
    a = "A2c_DESI"
    pulls = np.array([data[a][m]["pull"] for m in range(48)])
    keep_m = [m for m in range(48) if m not in RETRIED[a]]
    res["A2c_influence_bound"] = dict(
        pull_sd_all48=float(pulls.std(ddof=1)),
        pull_sd_omit_retried=float(pulls[keep_m].std(ddof=1)),
        pull_mean_all48=float(pulls.mean()),
        sum_sq_pull=float((pulls ** 2).sum()), expected_sum_sq=48.0,
        excess_sum_sq=float((pulls ** 2).sum() - 48.0),
        mean_abs_pull_retried=float(np.abs(pulls[RETRIED[a]]).mean()),
        mean_abs_pull_needed_to_carry_all_excess=float(
            np.sqrt(max((pulls ** 2).sum() - 48.0, 0.0) / len(RETRIED[a]))),
        n_omitted=len(RETRIED[a]),
        note=("DESCRIPTIVE ONLY. Per this repo's standing convention, omission statistics are "
              "influence diagnostics and license NO exclusion. Bounds how much of the frozen "
              "1.2699 the retry could possibly explain; it does not attribute."))

    # ---- CASE ASSIGNMENT (prereg v3.2) ----------------------------------------- #
    trig = [("primary", res["primary"]["p"]),
            ("S8_tail", res["S8_tail_edge_score"]["p"]),
            ("S3_absdev", res["S3_nearer_boundary"]["p"]),
            ("S1_holm_min", res["S1_holm_min"]),
            ("S4_KS_kink", res["S4_KS_kink"]["p"]),
            ("S_HULL", res["S_HULL_design_hull"]["p"]),
            ("S5_holm_min", res["S5_holm_min"])]
    tp = np.array([p for _, p in trig])
    tadj = holm(tp)
    fired = [(n, float(p), float(q)) for (n, p), q in zip(trig, tadj) if p <= ALPHA]
    res["case_rule"] = dict(
        family=[n for n, _ in trig],
        raw_p={n: float(p) for n, p in trig},
        holm_p={n: float(q) for (n, _), q in zip(trig, tadj)},
        fired_raw=[n for n, _, _ in fired],
        fired_holm=[n for n, _, q in fired if q <= ALPHA],
        size_under_global_null=dict(raw_any_of_7=0.151, holm_any_of_7=0.031,
                                    source="prereg v3.2, 1500 replicates"))
    if not fired:
        res["CASE"] = "A"
        res["case_reason"] = ("No Case-B trigger fired at raw p <= 0.05. Given the v3.1 power "
                              "table this is a substantive null against a one-sided all-arm "
                              "alternative, and NOT informative against a symmetric edge "
                              "alternative (power 0.09) or a KS-only effect (0.45).")
    else:
        weak = all(q > ALPHA for _, _, q in fired)
        res["CASE"] = "B"
        res["case_reason"] = (("Case B (weak trigger, family-uncorrected): "
                               if weak else "Case B: ")
                              + ", ".join(f"{n} raw p={p:.5f} holm={q:.5f}"
                                          for n, p, q in fired))

    # ---- per-unit table --------------------------------------------------------- #
    res["per_unit"] = {a: [dict(
        mock=m, retried=bool(m in RETRIED[a]), ndiv_attempt0=NDIV0[a].get(m, 0),
        ta_retained=TA_RETAINED[a].get(m, TA_RETAINED_DEFAULT),
        u=data[a][m]["u"], ns_phys=float(phys_of_u(data[a][m]["u"])),
        rank=data[a][m]["rank"], pull=data[a][m]["pull"],
        post_mean=data[a][m]["post_mean"], post_sd=data[a][m]["post_sd"],
        post_q95=data[a][m]["post_q95"], mass_above_hull=data[a][m]["mass_above_hull"],
        L=data[a][m]["L"]) for m in range(ARMS[a][1])] for a in ARMS}

    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    return res


if __name__ == "__main__":
    r = run(out_json=sys.argv[1] if len(sys.argv) > 1 else None)
    g = r["step0_validity_gate"]
    print(f"STEP0 gate passed={g['passed']} (in-box={g['all_in_unit_interval']}, "
          f"shared={g['shared_truths_identical_across_arms']}, KS p={g['ks_vs_uniform_p']:.4f})")
    if not g["passed"]:
        print("CASE C -- gate failed"); sys.exit(0)
    for k, lab in (("primary", "PRIMARY mean"), ("S8_tail_edge_score", "S8 tail"),
                   ("S3_nearer_boundary", "S3 |u-.5|"), ("S4_KS_kink", "S4 KS d_ns")):
        v = r[k]
        print(f"  {lab:14s} obs={v['obs']:.5f} null={v['null_mean']:.5f} p={v['p']:.5f}")
    for a, v in r["S1_per_arm"].items():
        print(f"  S1 {a:11s} obs={v['obs']:.4f} p={v['p']:.5f} (holm {v['p_holm_within_S1']:.4f})")
    print(f"  S5 holm-min={r['S5_holm_min']:.5f}   S10 p={r['S10_attempt0_ndiv_weighted']['p']:.5f}")
    print(f"\n  ==> CASE {r['CASE']}: {r['case_reason']}")
