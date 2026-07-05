#!/usr/bin/env python3
"""NORC res_corr INJECTION-RECOVERY gate — the decisive Phase-2 n_s-safety verdict (spec §4.2).

NORC (Gate-A, DEPLOYED) drops the mean-flux res_corr particle-convergence correction ENTIRELY and
PINS the alpha_res sites, so there is NO nuisance to absorb a res_corr misspecification. This gate
measures the paired Delta(n_s, A_p) bias that the NORC forward incurs from being BLIND to the res_corr
the real universe carries: the CLEAN arm is a self-consistent NORC self-draw (truth AND forward both
drop res_corr → ~0 bias), the INJECTED arm adds the ACTUAL anchored res_corr (b = log(anchored
res_corr) on each leg's k-grid — the correction NORC drops) to the mock TRUTH ONLY. This script reads
the PAIRED RCINJ checkpoints (CLEAN + INJECTED at the SAME (survey, sim, fold, seed) so the two mocks
share byte-identical base truth + cosmic noise and differ ONLY by ``exp(b)`` on the truth) and gates
the per-mock PAIRED shift.

The pure core (unit-tested in tests/test_res_corr_injection_gate.py)::

    paired_injection_gate(clean_means, inj_means, sigma_ref) -> dict

forms, for paired mock i (shared seed/noise),

    Delta_i    = inj_means[i] - clean_means[i]          # paired -> the shared cosmic noise CANCELS
    delta_mean = mean(Delta)
    delta_se   = std(Delta, ddof=1) / sqrt(N)           # the WITHIN-PAIR Delta SD / sqrt(N)
    stat       = |delta_mean| + 2 * delta_se            # the confidence-bound gate statistic
    passed     = stat < 0.3 * sigma_ref                 # strict <

TWO load-bearing design choices (mirrors analyze_dnuis_bias.py's |mean|+2*SE structure, but in
FIXED-REFERENCE units instead of the per-record bias_z):

  1. FIXED-REFERENCE units (the alpha-inflation masking trap). The statistic is built from RAW
     posterior means + an EXTERNALLY-supplied, FIXED ``sigma_ref`` (the anchored, alpha-FIXED n_s/A_p
     posterior sigma -- NOT each fit's own alpha-inflated post_sd). With alpha_res free the per-fit
     post_sd inflates ~1.3-1.4x, so a bias_z = Delta/post_sd statistic would mechanically DEFLATE
     toward PASS without de-biasing. Scaling sigma_ref moves ONLY the threshold, never the statistic.

  2. PAIRED SE = within-pair Delta SD / sqrt(N). The shared cosmic noise cancels in Delta_i, so the
     paired SE is ~24x smaller than the naive unpaired between-arm SE; that is what makes N>=8 paired
     mocks enough to certify < 0.3 * sigma_ref.

VERDICT: PASS on BOTH A_p and n_s for ALL gate surveys (DESI, KS, eBOSS) => the deployed NORC
forward's blindness to the real res_corr does NOT leak into cosmology. Any FAIL => the NORC drop is
NOT cosmology-safe on that leg and a z-resolved res_corr treatment is needed there. eBOSS is INCLUDED
under NORC (its k_max sits near the 5x k_box anchor so the expected Delta is small — we MEASURE it
per leg rather than assume it away). tau0_mean / tau0_tilt paired shifts are also REPORTED (mean flux
is the suspected n_s driver) but are NOT gated.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/analyze_res_corr_injection.py
"""
import argparse
import functools
import glob
import json
import os

import numpy as np

print = functools.partial(print, flush=True)

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/stepA"
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"

# The two blinded cosmology params this gate protects. n_s is index 0, A_p index 1 of the packed
# draws (the theta9 block leads _draws_matrix), looked up BY NAME from each checkpoint's `names`.
PARAMS = ("ns", "Ap")
GATE_FRAC = 0.30                  # spec §4.2: stat < 0.30 * sigma_ref

# The alpha-inflation factor. Under NORC alpha_res is PINNED (fix_alpha_res), so the clean-arm post
# sigma is NOT alpha-inflated and needs NO deflation → the default is 1.0 (sigma_ref = median(clean_sd)
# directly). Overridable via --alpha-inflation for a non-NORC re-analysis where alpha_res is free.
ALPHA_INFLATION = 1.0

# The three PER-LEG gate surveys. Under NORC eBOSS is INCLUDED (measured, not assumed away). Each
# leg gets 2 interior sims x >=8 seeds = >=16 paired mocks. The tags mirror run_stepA's RCINJ block.
GATE_SURVEYS = ("DESI", "KS", "eBOSS")
SURVEY_TAG = {"DESI": "D", "KS": "K", "eBOSS": "E"}
# Report-only paired quantities (mean flux — the suspected n_s driver); NOT gated (spec §4.2).
REPORT_PARAMS = ("tau0_mean", "tau0_tilt")


# --------------------------------------------------------------------------------------------- #
#  THE PURE GATE CORE (unit-tested). Param-agnostic: call once per (survey, param).
# --------------------------------------------------------------------------------------------- #
def paired_injection_gate(clean_means, inj_means, sigma_ref):
    """Paired clean-vs-injected gate in FIXED-REFERENCE units.

    Parameters
    ----------
    clean_means, inj_means : array-like
        Posterior MEANS (one per paired mock, SAME order/seed) for one (survey, param). Paired:
        ``inj_means[i]`` and ``clean_means[i]`` share the mock seed/noise so Delta_i cancels it.
    sigma_ref : float
        The FIXED-reference (anchored, alpha-FIXED) n_s/A_p posterior sigma -- supplied EXTERNALLY,
        NOT recomputed from the alpha-inflated per-fit post_sd (the masking-trap guard).

    Returns
    -------
    dict(delta_mean, delta_se, stat, sigma_ref, passed)
        ``Delta_i = inj_means[i] - clean_means[i]``; ``delta_mean = mean(Delta)``;
        ``delta_se = std(Delta, ddof=1)/sqrt(N)`` (within-pair Delta SD / sqrt(N), N>1; for N==1 the
        single |Delta| stands in for the SE, mirroring analyze_dnuis_bias.py);
        ``stat = |delta_mean| + 2*delta_se``; ``passed = bool(stat < 0.30 * sigma_ref)`` (strict <).
    """
    clean = np.asarray(clean_means, dtype=float)
    inj = np.asarray(inj_means, dtype=float)
    if clean.shape != inj.shape:
        raise ValueError(f"clean_means {clean.shape} and inj_means {inj.shape} must be the same "
                         f"shape (paired one-to-one).")
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("clean_means/inj_means must be non-empty 1-D arrays of posterior means.")

    delta = inj - clean                              # paired -> shared cosmic noise cancels
    n = delta.size
    delta_mean = float(delta.mean())
    if n > 1:
        delta_se = float(delta.std(ddof=1) / np.sqrt(n))
    else:
        # Single pair: no within-pair SD; the lone |Delta| stands in (the analyze_dnuis_bias.py
        # convention). N>=8 in the real gate, so this is only a degenerate-input guard.
        delta_se = float(abs(delta[0]))
    sigma_ref = float(sigma_ref)
    stat = abs(delta_mean) + 2.0 * delta_se
    passed = bool(stat < GATE_FRAC * sigma_ref)
    return dict(delta_mean=delta_mean, delta_se=delta_se, stat=stat,
                sigma_ref=sigma_ref, passed=passed)


# --------------------------------------------------------------------------------------------- #
#  CHECKPOINT LOADING + POSITIONAL PAIRING by (survey, sim, fold, seed).
# --------------------------------------------------------------------------------------------- #
def _is_injected(z):
    """True if this checkpoint's inject_res_corr provenance is a real spec (the INJECTED arm).
    run_stepA stores json.dumps(spec or None); '' / 'null' / None == the CLEAN arm."""
    raw = z.get("inject_res_corr")
    if raw is None:
        return False
    s = str(raw)
    if s in ("", "null", "None"):
        return False
    try:
        return json.loads(s) is not None
    except (ValueError, TypeError):
        return s not in ("", "null", "None")


def _post_mean_sd(z, param):
    """(posterior mean, posterior sd) for `param`, pooled over a checkpoint's stacked draws."""
    names = [str(x) for x in z["names"]]
    i = names.index(param)
    col = np.asarray(z["packed"])[:, i]
    return float(col.mean()), float(col.std())


def _seed_key(z):
    """A stable per-mock pairing identity. The RCINJ pairs share (survey, sim, fold, seed); the
    clean and injected arms differ ONLY by the injection. We pair on (survey, sim, fold, seed)."""
    return (str(z.get("survey", "")), str(z.get("sim", "")),
            int(z.get("fold", -1)), int(z.get("seed", 0)))


def _pool_mock(paths):
    """Pool a mock's chain checkpoints (RCINJ*_c0/_c1/... ) -> one record with the post mean/sd per
    param over the STACKED draws + its (survey, sim, fold, seed) pairing key + divergence count."""
    Z = [np.load(p, allow_pickle=True) for p in paths]
    z0 = Z[0]
    packed = np.concatenate([np.asarray(z["packed"]) for z in Z], axis=0)
    names = [str(x) for x in z0["names"]]
    rec = {"key": _seed_key(z0), "injected": _is_injected(z0),
           "n_chains": len(Z), "n_draws": int(packed.shape[0]),
           "div": int(sum(int(z["divergences"]) for z in Z)),
           "sim": str(z0.get("sim", "")), "survey": str(z0.get("survey", "")),
           "fold": int(z0.get("fold", -1)),
           "seed": int(z0.get("seed", 0))}
    for p in PARAMS:
        i = names.index(p)
        rec[p] = (float(packed[:, i].mean()), float(packed[:, i].std()))
    # REPORT-ONLY tau0 aggregates over the packed tau0_z* ladder cols (mean flux — the suspected n_s
    # driver): tau0_mean = per-draw mean over z; tau0_tilt = highest-z minus lowest-z tau0_z. Post-
    # mean/sd of each per-draw scalar. NaN if the checkpoint has no tau0_z cols (guard).
    tau0_cols = [j for j, nm in enumerate(names) if nm.startswith("tau0_z")]
    if tau0_cols:
        tblock = packed[:, tau0_cols]                       # (N, nz), z-ascending (kept_global order)
        tmean = tblock.mean(axis=1)                         # per-draw mean tau0(z)
        ttilt = tblock[:, -1] - tblock[:, 0]                # high-z minus low-z tau0_z
        rec["tau0_mean"] = (float(tmean.mean()), float(tmean.std()))
        rec["tau0_tilt"] = (float(ttilt.mean()), float(ttilt.std()))
    else:
        rec["tau0_mean"] = (float("nan"), float("nan"))
        rec["tau0_tilt"] = (float("nan"), float("nan"))
    return rec


def _mock_ids_for_survey(survey, ckpt_dir):
    """All RCINJ mock-id stems for a survey (clean + injected), grouped from the checkpoint dir.
    A mock-id = the filename stem minus the `_c<chain>` suffix; e.g. RCINJD_clean972_c0 -> the mock
    RCINJD_clean972. Returns {mock_id: [chain_path, ...]} for this survey's tag."""
    tag = SURVEY_TAG[survey]
    out = {}
    for p in sorted(glob.glob(f"{ckpt_dir}/RCINJ{tag}_*.npz")):
        base = os.path.basename(p)[:-4]                       # strip .npz
        if "_c" not in base:
            continue
        mock_id = base.rsplit("_c", 1)[0]
        out.setdefault(mock_id, []).append(p)
    return out


def load_pairs(survey, ckpt_dir=CKPT):
    """Load + POSITIONALLY PAIR a survey's RCINJ clean/injected mocks by (sim, fold, seed).

    Returns (clean_means, inj_means) dicts keyed by param, plus diagnostics:
      {param: (clean_arr, inj_arr)}, n_pairs, n_div, clean_sd_by_param, unpaired_keys.
    Each (clean_arr[i], inj_arr[i]) is one paired mock (the clean & injected arm sharing the seed)."""
    by_id = _mock_ids_for_survey(survey, ckpt_dir)
    clean_recs, inj_recs = {}, {}
    n_div = 0
    for mock_id, paths in by_id.items():
        rec = _pool_mock(sorted(paths))
        n_div += rec["div"]
        (inj_recs if rec["injected"] else clean_recs)[rec["key"]] = rec
    # Pair on the shared (survey, sim, fold, seed) key.
    keys = sorted(set(clean_recs) & set(inj_recs))
    unpaired = sorted((set(clean_recs) ^ set(inj_recs)))
    all_params = PARAMS + REPORT_PARAMS                       # gated + report-only, paired identically
    means = {p: ([], []) for p in all_params}
    clean_sd = {p: [] for p in PARAMS}
    for k in keys:
        rc, ri = clean_recs[k], inj_recs[k]
        for p in all_params:
            means[p][0].append(rc[p][0])                      # clean post mean
            means[p][1].append(ri[p][0])                      # injected post mean
        for p in PARAMS:
            clean_sd[p].append(rc[p][1])                      # clean-arm post sd (alpha-free)
    means = {p: (np.asarray(c), np.asarray(i)) for p, (c, i) in means.items()}
    clean_sd = {p: np.asarray(v) for p, v in clean_sd.items()}
    return means, len(keys), n_div, clean_sd, unpaired


def sigma_ref_for(survey, param, clean_sd, explicit, alpha_inflation=ALPHA_INFLATION):
    """The FIXED-reference sigma_ref for (survey, param).

    Precedence:
      1. an EXPLICIT --sigma-ref-<survey>-<param> CLI value (the honest reference sigma from the
         Phase-0 Fisher or a clean reference fit) -- ALWAYS preferred;
      2. else DERIVE it as median(clean post sd) / alpha_inflation. Under NORC alpha_res is PINNED so
         the clean-arm post_sd is NOT alpha-inflated → alpha_inflation defaults to 1.0 and sigma_ref
         = median(clean post sd) directly. (For a non-NORC re-analysis with alpha_res free, pass
         --alpha-inflation ~1.4 to deflate back to the alpha-FIXED yardstick.)

    Returns (sigma_ref, source_str)."""
    key = f"{survey}:{param}"
    if explicit and key in explicit:
        return float(explicit[key]), "explicit(ref)"
    sd = np.asarray(clean_sd.get(param, []))
    sd = sd[np.isfinite(sd) & (sd > 0)]
    if sd.size == 0:
        return float("nan"), "unavailable"
    src = "median(clean_sd)" if alpha_inflation == 1.0 else f"clean_sd/{alpha_inflation:g}"
    return float(np.median(sd) / alpha_inflation), src


# --------------------------------------------------------------------------------------------- #
def _parse_explicit(args):
    """Collect any explicit --sigma-ref-<SURVEY>-<param> CLI overrides into {'SURVEY:param': val}."""
    explicit = {}
    for survey in GATE_SURVEYS:
        for param in PARAMS:
            v = getattr(args, f"sigma_ref_{survey.lower()}_{param.lower()}", None)
            if v is not None:
                explicit[f"{survey}:{param}"] = float(v)
    return explicit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default=CKPT, help="dir with RCINJ*_c*.npz checkpoints")
    for survey in GATE_SURVEYS:
        for param in PARAMS:
            ap.add_argument(f"--sigma-ref-{survey.lower()}-{param.lower()}", type=float, default=None,
                            help=f"explicit reference sigma_ref for {survey} {param} "
                                 f"(else median(clean post_sd) / --alpha-inflation)")
    ap.add_argument("--alpha-inflation", type=float, default=ALPHA_INFLATION,
                    help="deflate the derived sigma_ref by this (NORC pins alpha_res -> default 1.0; "
                         "pass ~1.4 for a non-NORC alpha-free re-analysis)")
    ap.add_argument("--no-fig", action="store_true", help="skip writing the figure")
    a = ap.parse_args()
    ckpt_dir = a.ckpt_dir
    explicit = _parse_explicit(a)
    alpha_inflation = float(a.alpha_inflation)

    print("\n=== NORC res_corr INJECTION-RECOVERY gate — paired clean-vs-injected, per leg ===")
    print("    Delta_i = post_mean(inj_i) - post_mean(clean_i)   (paired -> shared noise cancels)")
    print(f"    GATE: |mean Delta| + 2*SE < {GATE_FRAC:.2f} * sigma_ref   (sigma_ref = reference yardstick)")
    print("    NORC: no alpha_res (pinned). Measures the paired Delta(n_s,A_p) bias from the self-")
    print("    consistent NORC self-draw being BLIND to the res_corr the real universe carries.")
    print("    PASS on A_p AND n_s for ALL legs (DESI, KS, eBOSS) => the NORC drop is cosmology-safe.")
    print("    tau0_mean / tau0_tilt paired shifts are REPORTED (mean flux) but NOT gated.\n")

    hdr = (f"  {'survey':<6} {'param':<4} {'N':>3} {'div':>4} "
           f"{'mean_Delta':>11} {'SE':>9} {'|mean|+2SE':>11} {'sigma_ref':>10} "
           f"{'0.3*sref':>9} {'sref_src':>16}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    any_fail = False
    any_present = False
    fig_rows = []                      # (survey, param, stat, thr, passed) for the figure
    for survey in GATE_SURVEYS:
        means, n_pairs, n_div, clean_sd, unpaired = load_pairs(survey, ckpt_dir)
        if n_pairs == 0:
            print(f"  {survey:<6} {'--':<4} {'(pending — no paired RCINJ checkpoints yet)'}")
            continue
        any_present = True
        if unpaired:
            print(f"  {survey:<6} [warn] {len(unpaired)} unpaired arm(s) (clean XOR injected, no "
                  f"matching seed) DROPPED: {unpaired}")
        for param in PARAMS:
            clean_arr, inj_arr = means[param]
            sref, src = sigma_ref_for(survey, param, clean_sd, explicit, alpha_inflation)
            out = paired_injection_gate(clean_arr, inj_arr, sref)
            thr = GATE_FRAC * sref
            verdict = "PASS" if out["passed"] else "FAIL"
            if not out["passed"]:
                any_fail = True
            print(f"  {survey:<6} {param:<4} {n_pairs:>3} {n_div:>4} "
                  f"{out['delta_mean']:>+11.4f} {out['delta_se']:>9.4f} {out['stat']:>11.4f} "
                  f"{sref:>10.4f} {thr:>9.4f} {src:>16}  {verdict}")
            fig_rows.append((survey, param, out["stat"], thr, out["passed"], n_pairs))
        # REPORT-ONLY paired shifts (tau0 mean flux — the suspected n_s driver): print the paired
        # |mean Delta| + 2*SE but DO NOT gate (no sigma_ref / no verdict). Reuses the pure core.
        for param in REPORT_PARAMS:
            clean_arr, inj_arr = means[param]
            if not np.all(np.isfinite(clean_arr)) or not np.all(np.isfinite(inj_arr)):
                continue
            out = paired_injection_gate(clean_arr, inj_arr, sigma_ref=1.0)  # sref unused (report-only)
            print(f"  {survey:<6} {param:<4} {n_pairs:>3} {n_div:>4} "
                  f"{out['delta_mean']:>+11.4f} {out['delta_se']:>9.4f} {out['stat']:>11.4f} "
                  f"{'--':>10} {'--':>9} {'report-only':>16}  (not gated)")

    print("  " + "-" * (len(hdr) - 2))
    if not any_present:
        print("\n  OVERALL: (pending) — no RCINJ checkpoints found. Launch the gate arm (compute-gated, "
              "PI sign-off) then re-run this script.")
        return
    if any_fail:
        print("\n  OVERALL: FAIL — at least one (leg, param) exceeds 0.3*sigma_ref. The NORC forward's "
              "blindness to the real res_corr LEAKS into cosmology on that leg.")
        print("           ACTION: a z-resolved res_corr treatment (or a per-leg res_corr nuisance) is "
              "needed there; re-run the gate.")
    else:
        print("\n  OVERALL: PASS — A_p & n_s within 0.3*sigma_ref on ALL legs (DESI, KS, eBOSS). The "
              "deployed NORC drop is cosmology-safe: dropping res_corr does not leak into n_s/A_p.")

    if not a.no_fig and fig_rows:
        _write_figure(fig_rows, any_fail)


def _write_figure(fig_rows, any_fail):
    """Bar of stat vs 0.3*sigma_ref per (survey, param); bar GREEN if PASS, RED if FAIL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(NOTES_FIG, exist_ok=True)
    labels = [f"{s}\n{p}" for (s, p, *_rest) in fig_rows]
    stats = [r[2] for r in fig_rows]
    thrs = [r[3] for r in fig_rows]
    passed = [r[4] for r in fig_rows]
    npairs = [r[5] for r in fig_rows]
    x = np.arange(len(fig_rows))

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(fig_rows) + 3), 4.8))
    colors = ["#2e7d32" if pz else "#c62828" for pz in passed]
    ax.bar(x, stats, width=0.6, color=colors, alpha=0.85,
           label="|mean Δ| + 2·SE (the gate statistic)")
    # the per-cell 0.3*sigma_ref threshold (varies by survey/param) as red caps.
    for xi, t in zip(x, thrs):
        ax.plot([xi - 0.32, xi + 0.32], [t, t], color="k", lw=2.0, zorder=5)
    ax.plot([], [], color="k", lw=2.0, label=r"$0.3\,\sigma_{\rm ref}$ threshold")
    for xi, st, t, npz in zip(x, stats, thrs, npairs):
        ax.annotate(f"{st:.3f}\n(N={npz})", (xi, st), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$|\overline{\Delta}| + 2\,{\rm SE}$  (fixed-reference units)")
    verdict = "FAIL" if any_fail else "PASS"
    ax.set_title("NORC res_corr injection-recovery gate — paired clean-vs-injected\n"
                 f"OVERALL {verdict}  (bar < cap ⇒ PASS; per leg DESI/KS/eBOSS)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max([*stats, *thrs, 1e-6]) * 1.35)
    fig.tight_layout()
    out = f"{NOTES_FIG}/res_corr_injection_gate.png"
    fig.savefig(out, dpi=130)
    print(f"\n  wrote figure -> {out}")


if __name__ == "__main__":
    main()
