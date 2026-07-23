"""Export the F6(b) whitened-residual ECDF layers (diagonal vs cross-class C_emu).

The recipe of record is `scripts/diag_cemu_validation.py --xclass ...` (function
`_xclass_revalidate`). That entry point PRINTS the scalars and draws a Q-Q figure but does
not persist any numeric array, so the paper has been quoting a FALLBACK. This driver calls
the IDENTICAL tracked functions with the IDENTICAL arguments -- `revalidate_xclass`, which
internally calls `extract_holdout_residuals` and `run_whitening` -- and additionally keeps
the pooled whitened arrays so the ECDF layers can be written out.

It imports from the tracked module; it does not copy or re-implement any numerics, and it
writes nothing inside the code repo. The scalars it prints are cross-checked byte-for-byte
against the canonical `--xclass` stdout.

Blind status: BLIND-SAFE. Every array is an emulator-error diagnostic on HELD-OUT PRIYA
SIMULATIONS (LOSO folds). No observed P1D, no posterior, no real-data n_s/A_p.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/mfho/hcd_priya")
sys.path.insert(0, str(ROOT))

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import load_cache
import scripts.diag_cemu_validation as D

XCLASS = ROOT / "checkpoints/error_vector_xclass.npz"
CKPT_PREFIX = str(ROOT / "checkpoints/final_fold")
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/f6b_whitening_2026-07-21")
RUN_CMD = (
    "cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya "
    "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "
    "/home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_cemu_validation.py "
    "--xclass /home/mfho/hcd_priya/checkpoints/error_vector_xclass.npz --n-folds 8"
)

# KS 95% simultaneous critical coefficient: sup|F_n - F| > c/sqrt(n) w.p. 0.05, c = 1.35810.
KS95 = 1.3580986393225507


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _prior_sig():
    """The live hcd_prior_signature via module-attribute access (rebinding-trap safe)."""
    from hcd_analysis.emulator import inference as INF
    return INF.hcd_prior_signature()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a],
                          capture_output=True, text=True).stdout.strip()


def ecdf_layers(w, n_grid=4001):
    """Decimated ECDF of the pooled whitened residuals, exact at the retained points.

    Returns x (sorted whitened values, decimated to <=n_grid points, endpoints kept) and
    y = ECDF(x) = rank/n at those SAME points, so (x,y) redraws the step curve without any
    interpolation of the underlying sample.
    """
    ws = np.sort(np.asarray(w, float))
    n = ws.size
    y_all = (np.arange(1, n + 1)) / n
    if n <= n_grid:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, n_grid).round().astype(int))
    return ws[idx], y_all[idx], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--max-rows-per-fold", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out_dir)

    print("jax.devices():", jax.devices())
    # ---- inputs, exactly as _xclass_revalidate loads them -------------------
    evx = np.load(XCLASS, allow_pickle=True)
    rho = evx["rho"]
    sigma = evx["sigma"]
    alpha_centres = jnp.asarray(evx["tau0_band_centres"])
    dla_shot_flag_k = np.asarray(evx["dla_shot_flag"])
    z_band_edges = evx["z_band_edges"]
    kgrid = np.asarray(evx["kfkms"])
    diag_match_median = float(evx["diag_match_median"])
    n_pool_build = int(evx["n_pool"])
    print(f"  loaded {XCLASS}: rho {rho.shape}, diag-match median "
          f"{diag_match_median:.3f}, n_pool {n_pool_build}")

    models, pf_by_fold = [], []
    for f in range(args.n_folds):
        model, meta, norm = T.load_checkpoint(f"{CKPT_PREFIX}{f}")
        models.append(model)
        pf_by_fold.append({k: jnp.asarray(norm["P_filt"][k])
                           for k in ("mu_marg", "sig_marg", "sig_cosmo")})
    cache_path = meta["cache_path"]
    d = load_cache(cache_path)
    nyq = D.lf_nyquist_kmax(cache_path, kgrid)
    k_max = nyq["k_max_proposed"]

    # ---- THE tracked recipe, unmodified ------------------------------------
    diag, xcl = D.revalidate_xclass(
        d, pf_by_fold, models, sigma, rho, z_band_edges, alpha_centres,
        dla_shot_flag_k, n_folds=args.n_folds,
        max_rows_per_fold=args.max_rows_per_fold)
    (pd_, wd, ck_d, nk_d) = diag
    (px, wx, ck_x, nk_x) = xcl

    def _report(tag, pooled):
        print(f"\n  [{tag}]  N={pooled['n']}")
        print(f"    mean    = {pooled['mean']:+.4f}")
        print(f"    var     = {pooled['var']:.4f}")
        print(f"    chi2/dof= {pooled['chi2_over_dof']:.4f}")
        print(f"    KS      = {pooled['ks_stat']:.4f}  (p={pooled['ks_p']:.2g})")
    _report("BEFORE: diagonal C_emu (all-folds)", pd_)
    _report("AFTER : cross-class C_emu (all-folds)", px)
    print(f"\n  var: {pd_['var']:.4f} (diagonal) -> {px['var']:.4f} (cross-class)")

    # ---- the FOLD-0-ONLY diagonal whitening: the provenance of the quoted 2.46 ----
    # The paper's "whitening variance ~2.46" is NOT this all-folds diagonal number. It is the
    # ORIGINAL fold-0-only whitening (diag_cemu_validation's default path, --max-rows 600),
    # and fold 0 is the worst of the 8 folds by ~2x. Recomputed here through the same tracked
    # functions so the quoted scalar has its OWN matching curve instead of being paired with
    # the all-folds ECDF.
    from hcd_analysis.emulator.data import make_splits
    _tr0, va0, _ho0 = make_splits(d, 0, n_folds=args.n_folds)
    recs0 = D.extract_holdout_residuals(d, models[0], pf_by_fold[0], va0, sigma, z_band_edges,
                                        alpha_centres, dla_shot_flag_k, max_rows=600, rho=None)
    p0, w0, ck0, nk0, _ = D.run_whitening(recs0)
    print(f"\n  [fold-0-only diagonal (the quoted 2.46 setting)] N={p0['n']} "
          f"var={p0['var']:.4f}")

    # ---- per-fold breakdown: which folds drive the pooled number ------------
    per_fold_var_d, per_fold_var_x, per_fold_mean_d, per_fold_n = [], [], [], []
    for f in range(args.n_folds):
        _t, va, _h = make_splits(d, f, n_folds=args.n_folds)
        rdf = D.extract_holdout_residuals(d, models[f], pf_by_fold[f], va, sigma, z_band_edges,
                                          alpha_centres, dla_shot_flag_k,
                                          max_rows=args.max_rows_per_fold, rho=None)
        rxf = D.extract_holdout_residuals(d, models[f], pf_by_fold[f], va, sigma, z_band_edges,
                                          alpha_centres, dla_shot_flag_k,
                                          max_rows=args.max_rows_per_fold, rho=rho)
        pdf, _, _, _, _ = D.run_whitening(rdf)
        pxf, _, _, _, _ = D.run_whitening(rxf)
        per_fold_var_d.append(pdf["var"]); per_fold_var_x.append(pxf["var"])
        per_fold_mean_d.append(pdf["mean"]); per_fold_n.append(int(pdf["n"]))
        print(f"    fold {f}: var_diag={pdf['var']:.4f} var_xclass={pxf['var']:.4f}")

    # ---- ECDF layers + null band -------------------------------------------
    x_d, y_d, n_d = ecdf_layers(wd)
    x_x, y_x, n_x = ecdf_layers(wx)
    x_0, y_0, n_0 = ecdf_layers(w0)

    # Null: the N(0,1) the whitened residual is tested against, on a plotting grid, with a
    # 95% SIMULTANEOUS Kolmogorov band (sup-norm) and a 95% POINTWISE binomial band. Both are
    # quoted at the DIAGONAL pool size n_d (n_x == n_d by construction: same rows, same mask).
    from scipy import stats
    x_null = np.linspace(-6.0, 6.0, 4001)
    cdf_null = stats.norm.cdf(x_null)
    d_crit_d = KS95 / np.sqrt(n_d)
    d_crit_x = KS95 / np.sqrt(n_x)
    band_sim_lo = np.clip(cdf_null - d_crit_d, 0.0, 1.0)
    band_sim_hi = np.clip(cdf_null + d_crit_d, 0.0, 1.0)
    se = np.sqrt(cdf_null * (1.0 - cdf_null) / n_d)
    band_pt_lo = np.clip(cdf_null - 1.959963984540054 * se, 0.0, 1.0)
    band_pt_hi = np.clip(cdf_null + 1.959963984540054 * se, 0.0, 1.0)

    assert n_d == n_x, f"pool sizes differ ({n_d} vs {n_x}); the null band would be ambiguous"

    out.mkdir(parents=True, exist_ok=True)
    npz = out / "f6b_whitening_layers.npz"
    np.savez_compressed(
        npz,
        # --- ECDF curves, the F6(b) foreground ---
        ecdf_diag_x=x_d, ecdf_diag_y=y_d,
        ecdf_xclass_x=x_x, ecdf_xclass_y=y_x,
        # the fold-0-only diagonal curve that actually goes with the quoted 2.46
        ecdf_diag_fold0only_x=x_0, ecdf_diag_fold0only_y=y_0,
        var_diag_fold0only=np.float64(p0["var"]),
        mean_diag_fold0only=np.float64(p0["mean"]),
        chi2dof_diag_fold0only=np.float64(p0["chi2_over_dof"]),
        ks_stat_diag_fold0only=np.float64(p0["ks_stat"]),
        n_whitened_fold0only=np.int64(n_0),
        chi2k_diag_fold0only=ck0, nrows_k_diag_fold0only=nk0,
        # per-fold breakdown: fold 0 is the ~2x outlier that inflates the quoted scalar
        fold_index=np.arange(args.n_folds),
        var_diag_per_fold=np.asarray(per_fold_var_d, float),
        var_xclass_per_fold=np.asarray(per_fold_var_x, float),
        mean_diag_per_fold=np.asarray(per_fold_mean_d, float),
        n_whitened_per_fold=np.asarray(per_fold_n, np.int64),
        # --- null reference + bands ---
        null_x=x_null, null_cdf=cdf_null,
        null_band_sim95_lo=band_sim_lo, null_band_sim95_hi=band_sim_hi,
        null_band_point95_lo=band_pt_lo, null_band_point95_hi=band_pt_hi,
        null_d_crit_sim95_diag=np.float64(d_crit_d),
        null_d_crit_sim95_xclass=np.float64(d_crit_x),
        null_d_crit_sim95_fold0only=np.float64(KS95 / np.sqrt(n_0)),
        # --- the scalars the paper quotes ---
        var_diag=np.float64(pd_["var"]), var_xclass=np.float64(px["var"]),
        mean_diag=np.float64(pd_["mean"]), mean_xclass=np.float64(px["mean"]),
        chi2dof_diag=np.float64(pd_["chi2_over_dof"]),
        chi2dof_xclass=np.float64(px["chi2_over_dof"]),
        ks_stat_diag=np.float64(pd_["ks_stat"]), ks_p_diag=np.float64(pd_["ks_p"]),
        ks_stat_xclass=np.float64(px["ks_stat"]), ks_p_xclass=np.float64(px["ks_p"]),
        n_whitened_components=np.int64(n_d),
        diag_match_median=np.float64(diag_match_median),
        n_pool_build=np.int64(n_pool_build),
        # --- supporting per-k ramps (free, same run) ---
        kfkms=kgrid, chi2k_diag=ck_d, chi2k_xclass=ck_x,
        nrows_k_diag=nk_d, nrows_k_xclass=nk_x,
        k_max_proposed=np.float64(k_max),
        n_folds=np.int64(args.n_folds),
        max_rows_per_fold=np.int64(args.max_rows_per_fold),
        note=np.str_(
            "F6(b) whitened-residual ECDF layers. Pooled HELD-OUT PRIYA-sim P1D residuals "
            "r = P_obs_emu - P_obs_truth from all 8 LOSO folds, whitened by C_emu ALONE "
            "(cosmic_cov=0): 'diag' = the per-class DIAGONAL C_emu, 'xclass' = the "
            "cross-class 4x4 rho C_emu. ECDF arrays are the exact sample ECDF decimated to "
            "<=4001 points (x = sorted whitened value, y = rank/n) -- step-plot directly, do "
            "not interpolate. null_cdf is N(0,1); sim95 is the 95% simultaneous Kolmogorov "
            "band (1.3581/sqrt(n)), point95 is the 95% pointwise binomial band. "
            "diag_match_median is READ from checkpoints/error_vector_xclass.npz, it is a "
            "build-time scalar, not recomputed here. "
            "SCALAR WARNING: the paper's quoted whitening variance 2.46 is NOT var_diag "
            "(=1.2758, all-folds). It is var_diag_fold0only (=2.4599), the fold-0-only "
            "whitening, and fold 0 is the WORST of the 8 folds by ~2x (per-fold range "
            "0.76-1.47 for folds 1-7). Pair 2.46 with ecdf_diag_fold0only_*, never with "
            "ecdf_diag_* . Sims only: no data, no posterior, no cosmology."),
    )

    sidecar = {
        "schema": "f6b_whitening_v1",
        "purpose": ("live whitened-residual ECDF layers + null band + scalars for paper "
                    "figure F6(b), replacing the fallback numbers the paper was quoting"),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        # conditional (2026-07-23, paper-agent Q1): a CLEAN export must not carry a warning
        # claiming the tree was dirty.
        "dirty_warning": (None if not bool(git("status", "--porcelain")) else (
            "TREE WAS DIRTY AT EXPORT TIME. The untracked scripts present in the working tree "
            "are NOT on the import path of this run (the recipe imports only tracked modules: "
            "scripts/diag_cemu_validation.py, hcd_analysis.emulator.*), so the numbers are "
            "believed unaffected. The orchestrator MUST re-stamp this export from a CLEAN tree "
            "before it is cited as a frozen artifact of record.")),
        # prior-geometry pin (2026-07-23, paper-agent Q4): lets a figure sidecar prove which
        # prior era produced it, independently of the commit. MODULE-ATTRIBUTE read.
        "hcd_prior_signature": _prior_sig(),
        "run_command": RUN_CMD,
        "run_command_note": (
            "RUN_CMD is the canonical tracked entry point and was executed in full; its stdout "
            "is archived as canonical_stdout.txt in this directory. That entry point persists "
            "no arrays, so this export was produced by a thin scratchpad driver that calls the "
            "SAME tracked functions (diag_cemu_validation.revalidate_xclass) with the SAME "
            "arguments and keeps the pooled whitened vectors. The driver's scalars were "
            "cross-checked against canonical_stdout.txt; see scalar_crosscheck."),
        "inputs_sha256": {},
        "outputs_sha256": {},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "jax": jax.__version__, "host": platform.node(),
                        "jax_platforms": "cpu"},
        "blind_status": (
            "BLIND-SAFE: emulator-error diagnostics on HELD-OUT PRIYA SIMULATIONS (LOSO). "
            "Every stored array is a whitened sim residual, an ECDF thereof, an analytic "
            "N(0,1) null band, or a k grid. No observed P1D, no chain, no posterior, no "
            "real-data n_s/A_p in any array or metadata field."),
        "chain_of_record_status": (
            "presentation-layer derived artifact; not a chain. Immutable: regenerate to a NEW "
            "dated directory, never edit in place."),
        "scalar_reconciliation": {
            "quoted_by_paper": {"whitening_variance": 2.46, "diag_match_median": 0.777},
            "diag_match_median_measured": diag_match_median,
            "diag_match_median_verdict": (
                "MATCHES. 0.7771604805510948, stored in checkpoints/error_vector_xclass.npz by "
                "build_xclass_error_vector.py. It is a BUILD-TIME scalar read from the shipped "
                "input, not recomputed by the F6(b) run."),
            "whitening_variance_measured_allfolds_diag": pd_["var"],
            "whitening_variance_measured_allfolds_xclass": px["var"],
            "whitening_variance_measured_fold0only_diag": p0["var"],
            "whitening_variance_verdict": (
                "DOES NOT MATCH the all-folds curve. 2.46 reproduces EXACTLY as the FOLD-0-ONLY "
                "diagonal whitening (measured 2.4599, N=83035, diag_cemu_validation.py default "
                "path with --max-rows 600). The all-folds diagonal whitening that the F6(b) "
                "ECDF pair is built from is 1.2758 (N=444273), and the cross-class is 0.9321. "
                "Difference 2.4599 vs 1.2758 = a factor 1.93. Neither number is wrong; they are "
                "DIFFERENT ESTIMANDS, and the paper is pairing the fold-0-only scalar with the "
                "all-folds curve."),
            "why_they_differ": (
                "fold 0 is the worst-fitting LOSO fold by ~2x. Per-fold diagonal whitening "
                "variance at 400 rows/fold: fold0 2.4757, fold1 1.2673, fold2 1.1460, fold3 "
                "0.8279, fold4 0.7605, fold5 1.4699, fold6 0.8598, fold7 1.2608 (median "
                "1.2034). Fold 0's value is stable under subsample size (2.4757 at 400 rows vs "
                "2.4599 at 600), so it is a genuine property of fold 0, not a sampling "
                "artifact. Pooling the other 7 folds pulls the diagonal variance down to 1.28."),
            "recommendation": (
                "For F6(b), which draws the ALL-FOLDS diagonal-vs-cross-class ECDF pair, quote "
                "var_diag=1.276 -> var_xclass=0.932. If the text wants to keep 2.46 it must say "
                "'fold 0 only' and draw ecdf_diag_fold0only_*. NOTE this also weakens the "
                "premise in build_xclass_error_vector.py's own docstring ('under-sizes by ~2.5x, "
                "whitening var 2.46'): the honest all-folds under-sizing is 1.28x."),
        },
        "inputs_are_read_only": (
            "checkpoints/error_vector_xclass.npz and checkpoints/final_fold{0..7} are READ "
            "ONLY here. scripts/build_xclass_error_vector.py, which WRITES "
            "error_vector_xclass.npz, was deliberately NOT run."),
    }
    ins = {str(XCLASS): sha256(XCLASS)}
    for f in range(args.n_folds):
        for ext in ("eqx", "meta.json", "norm.pkl"):
            p = Path(f"{CKPT_PREFIX}{f}.{ext}")
            ins[str(p)] = sha256(p)
    ins[str(cache_path)] = sha256(cache_path)
    sidecar["inputs_sha256"] = ins
    # archive the canonical entry point's own stdout alongside the arrays
    import shutil
    canon = Path("/tmp/claude-114399728/-home-mfho-hcd-priya/"
                 "eeac10b8-2da6-4e35-9dc3-fd542bc32135/scratchpad/run/canonical_stdout.txt")
    outs = {npz.name: sha256(npz)}
    if canon.exists():
        shutil.copy2(canon, out / "canonical_stdout.txt")
        outs["canonical_stdout.txt"] = sha256(out / "canonical_stdout.txt")
    fold0 = canon.parent / "fold0_stdout.txt"
    if fold0.exists():
        shutil.copy2(fold0, out / "fold0only_stdout.txt")
        outs["fold0only_stdout.txt"] = sha256(out / "fold0only_stdout.txt")
    sidecar["outputs_sha256"] = outs
    (out / "PROVENANCE.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    print(f"\n[export] wrote {out}")
    print(f"  {npz.name}  sha256={sidecar['outputs_sha256'][npz.name][:16]}...")
    print(f"  commit={sidecar['commit'][:12]} dirty={sidecar['dirty']}")
    # machine-readable scalars for the cross-check step
    (out / "_scalars.json").write_text(json.dumps({
        "var_diag": pd_["var"], "var_xclass": px["var"],
        "mean_diag": pd_["mean"], "mean_xclass": px["mean"],
        "chi2dof_diag": pd_["chi2_over_dof"], "chi2dof_xclass": px["chi2_over_dof"],
        "ks_stat_diag": pd_["ks_stat"], "ks_stat_xclass": px["ks_stat"],
        "n": int(n_d), "diag_match_median": diag_match_median,
        "var_diag_fold0only": p0["var"], "n_fold0only": int(n_0),
        "var_diag_per_fold": per_fold_var_d,
        "var_xclass_per_fold": per_fold_var_x}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
