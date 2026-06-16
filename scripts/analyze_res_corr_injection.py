#!/usr/bin/env python3
"""res_corr INJECTION-RECOVERY gate — the decisive Phase-2 n_s-safety verdict (spec §4.2, Task 2.1).

The marginalized res_corr amplitude ``alpha_res`` (Task 1.3) is supposed to absorb the worst
OUT-OF-SPAN, z>=2.8-localized res_corr misspecification WITHOUT leaking into cosmology (A_p, n_s).
This script reads the PAIRED RCINJ checkpoints (a CLEAN control + an INJECTED arm at the SAME
(survey, sim, fold, seed) so the two mocks share byte-identical base truth + cosmic noise and differ
ONLY by the injected ``exp(b1)`` res_corr on the z>=2.8 truth) and gates the per-mock PAIRED shift.

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

VERDICT: PASS on BOTH A_p and n_s for ALL gate surveys (DESI, KS) => alpha_res absorbs the worst
out-of-span res_corr misspecification without leaking into cosmology. Any FAIL => trigger the spec
§3.4 / plan §2.1 z>=2.8 z-resolved cut and re-run. eBOSS is EXCLUDED from this gate by design
(documented below): eBOSS k_max 0.0195 s/km sits essentially inside the 5x k_box(z=3) ~ 0.019 anchor
so res_corr has minimal high-k leverage there; eBOSS is covered by the separate eBOSS MF-anchored
re-cert (spec §4.2 gate 4, plan Task 2.3).

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

# The alpha-inflation factor (spec §4.2 / Fisher Phase-0): with alpha_res free the per-fit posterior
# sigma inflates ~1.3-1.4x vs the alpha-FIXED reference. We DEFLATE the (alpha-free) clean-arm post
# sigma by this to recover the alpha-FIXED sigma_ref yardstick when no explicit sigma_ref is supplied.
ALPHA_INFLATION = 1.4

# The two gate surveys (eBOSS is EXCLUDED by design -- see the module docstring). KS gets >=16 paired
# mocks (2 worst-tilt sims x >=8 seeds); DESI >=16 likewise. The tags mirror run_stepA's RCINJ block.
GATE_SURVEYS = ("DESI", "KS")
SURVEY_TAG = {"DESI": "D", "KS": "K", "eBOSS": "E"}


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
    means = {p: ([], []) for p in PARAMS}
    clean_sd = {p: [] for p in PARAMS}
    for k in keys:
        rc, ri = clean_recs[k], inj_recs[k]
        for p in PARAMS:
            means[p][0].append(rc[p][0])                      # clean post mean
            means[p][1].append(ri[p][0])                      # injected post mean
            clean_sd[p].append(rc[p][1])                      # clean-arm post sd (alpha-free)
    means = {p: (np.asarray(c), np.asarray(i)) for p, (c, i) in means.items()}
    clean_sd = {p: np.asarray(v) for p, v in clean_sd.items()}
    return means, len(keys), n_div, clean_sd, unpaired


def sigma_ref_for(survey, param, clean_sd, explicit):
    """The FIXED-reference (anchored, alpha-FIXED) sigma_ref for (survey, param).

    Precedence:
      1. an EXPLICIT --sigma-ref-<survey>-<param> CLI value (the honest alpha-FIXED / eBOSS-anchor
         sigma from the Phase-0 Fisher or a clean alpha-fixed reference fit) -- ALWAYS preferred;
      2. else DERIVE it from the clean-arm posterior sigma deflated by the documented ALPHA_INFLATION
         (~1.4x): sigma_ref ~= median(clean post sd) / 1.4. The clean arm has alpha FREE, so its raw
         post_sd is alpha-INFLATED; deflating recovers the alpha-FIXED yardstick. (This is a
         documented stand-in; supply an explicit alpha-fixed sigma_ref for the production verdict.)

    Returns (sigma_ref, source_str)."""
    key = f"{survey}:{param}"
    if explicit and key in explicit:
        return float(explicit[key]), "explicit(alpha-fixed)"
    sd = np.asarray(clean_sd.get(param, []))
    sd = sd[np.isfinite(sd) & (sd > 0)]
    if sd.size == 0:
        return float("nan"), "unavailable"
    return float(np.median(sd) / ALPHA_INFLATION), f"clean_sd/{ALPHA_INFLATION:g}"


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
                            help=f"explicit alpha-FIXED sigma_ref for {survey} {param} "
                                 f"(else derived from clean post_sd / {ALPHA_INFLATION:g})")
    ap.add_argument("--no-fig", action="store_true", help="skip writing the figure")
    a = ap.parse_args()
    ckpt_dir = a.ckpt_dir
    explicit = _parse_explicit(a)

    print("\n=== res_corr INJECTION-RECOVERY gate — paired clean-vs-injected, FIXED-reference units ===")
    print("    Delta_i = post_mean(inj_i) - post_mean(clean_i)   (paired -> shared noise cancels)")
    print(f"    GATE: |mean Delta| + 2*SE < {GATE_FRAC:.2f} * sigma_ref   (sigma_ref = alpha-FIXED yardstick)")
    print("    PASS on A_p AND n_s for ALL gate surveys => alpha_res absorbs the misspecification.")
    print("    eBOSS EXCLUDED by design (k_max 0.0195 inside the 5x k_box anchor; covered by the eBOSS re-cert).\n")

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
            sref, src = sigma_ref_for(survey, param, clean_sd, explicit)
            out = paired_injection_gate(clean_arr, inj_arr, sref)
            thr = GATE_FRAC * sref
            verdict = "PASS" if out["passed"] else "FAIL"
            if not out["passed"]:
                any_fail = True
            print(f"  {survey:<6} {param:<4} {n_pairs:>3} {n_div:>4} "
                  f"{out['delta_mean']:>+11.4f} {out['delta_se']:>9.4f} {out['stat']:>11.4f} "
                  f"{sref:>10.4f} {thr:>9.4f} {src:>16}  {verdict}")
            fig_rows.append((survey, param, out["stat"], thr, out["passed"], n_pairs))

    print("  " + "-" * (len(hdr) - 2))
    if not any_present:
        print("\n  OVERALL: (pending) — no RCINJ checkpoints found. Launch the gate arm (compute-gated, "
              "PI sign-off) then re-run this script.")
        return
    if any_fail:
        print("\n  OVERALL: FAIL — at least one (survey, param) exceeds 0.3*sigma_ref. alpha_res does "
              "NOT fully absorb the out-of-span res_corr misspecification.")
        print("           ACTION: trigger the spec §3.4 / plan §2.1 z>=2.8 z-resolved res_corr cut "
              "and re-run the gate.")
    else:
        print("\n  OVERALL: PASS — A_p & n_s within 0.3*sigma_ref on ALL gate surveys. alpha_res absorbs "
              "the worst out-of-span res_corr misspecification without leaking into cosmology.")

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
    ax.set_title("res_corr injection-recovery gate — paired clean-vs-injected\n"
                 f"OVERALL {verdict}  (bar < cap ⇒ PASS; eBOSS excluded by design)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max([*stats, *thrs, 1e-6]) * 1.35)
    fig.tight_layout()
    out = f"{NOTES_FIG}/res_corr_injection_gate.png"
    fig.savefig(out, dpi=130)
    print(f"\n  wrote figure -> {out}")


if __name__ == "__main__":
    main()
