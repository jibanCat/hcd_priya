#!/usr/bin/env python3
"""Build the two Phase-2b emulator notebooks (01 training, 02 plotting) via nbformat.

This is the SOURCE that generates ``notebooks/01_emulator_training.ipynb`` and
``notebooks/02_emulator_plotting.ipynb``. The notebooks reuse the PRODUCTION
emulator API (``hcd_analysis.emulator.*``) and the keeper diagnostic plotting logic
(``scripts/plot_performance_walkthrough.py``); they do NOT reimplement it.

Regenerate (then execute):
  CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \\
    /home/mfho/.conda/envs/emu-jax/bin/python3 notebooks/_build_emulator_notebooks.py
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_DIR = Path(__file__).resolve().parent
KERNEL = {"name": "emu-jax", "display_name": "emu-jax", "language": "python"}


def md(*lines):
    return new_markdown_cell("\n".join(lines))


def code(*lines):
    return new_code_cell("\n".join(lines))


def finalize(nb):
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ============================================================================ #
# NOTEBOOK 1 — TRAINING
# ============================================================================ #
def build_training_nb():
    cells = []

    cells.append(md(
        "# Phase-2b Lyα P1D emulator — Notebook 1: how to train",
        "",
        "This notebook walks through **training** the deployed Phase-2b emulator",
        "from the production cache, using the *production API* directly",
        "(`hcd_analysis.emulator.{data,train,model}`) — no logic is reimplemented",
        "here. It trains a **short, illustrative fold-0** so the whole notebook runs",
        "in a couple of minutes on CPU; the production recipe runs ~150–250 epochs",
        "across **8 LOSO folds** (see the closing section).",
        "",
        "**Companion:** `02_emulator_plotting.ipynb` loads the *finalized* checkpoints",
        "and produces the full performance walkthrough.",
        "",
        "---",
        "## 1. Architecture (what we are training)",
        "",
        "The emulator (`hcd_analysis.emulator.model.Emulator`) is a shared **encoder**",
        "feeding three heads. Its central design move is a **baseline + residual**",
        "split of the per-class filtered P1D, `P_filt`:",
        "",
        "- **θ-blind baseline** `m̂(z, τ₀)` — a deep MLP (`BaselineHead`) that sees",
        "  *only* `(z, τ₀)`, never the 9 cosmology/IGM params. It predicts the",
        "  `(z,τ₀)`-conditional **mean** spectrum (the dominant ~99.5% of the per-k",
        "  log-variance). Because θ never enters, `∂m̂/∂θ = 0` exactly.",
        "- **cosmology residual** `r̂(θ, z, τ₀)` — `HeadB` predicts the per-k",
        "  **σ_cosmo-whitened residual** (the ~0.4% cosmology signal). Full log-P1D is",
        "  `logP̂ = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂`, so `∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ`.",
        "- **per-k conditional σ_cosmo whitening** — the residual target is whitened",
        "  per `(class,k)` by `σ_cosmo` = the within-cell cosmology scale, so the",
        "  residual head learns a ~unit-variance signal.",
        "- **SVD basis + differentiable spline** — both P_filt paths decode through a",
        "  low-rank, SVD-warm-started learned basis (`n_basis=24`); the residual is a",
        "  smooth, differentiable function of k (HMC-usable downstream).",
        "- **Head A** — `dN/dX` + CDDF (`f_nhi`); τ₀-invariant (reads only the latent).",
        "- **single-α_c HCD** — the 3 HCD-class `Δ_c` templates re-weight the",
        "  structural total `P_tier_p = Σ_c w_c·P_filt`.",
        "- **MF (ρ-only + res_corr)** — the multi-fidelity LF→HF layer",
        "  (`hcd_analysis.emulator.multifidelity`); covered at the end + in NB 2.",
    ))

    cells.append(md(
        "## 2. The `FINAL_RECIPE` knobs",
        "",
        "The production recipe lives in `scripts/run_loso_sweep.py::FINAL_RECIPE`. The",
        "load-bearing knobs (and *why* each exists):",
        "",
        "- **deep frozen pre-fit baseline** — the θ-blind baseline is *pre-fit* to its",
        "  `(z,τ₀)`-cell-mean floor (`prefit_baseline_epochs`), then **frozen** during",
        "  the joint loop so it cannot drift off that floor (the joint early-stop, driven",
        "  by the larger residual term, would otherwise cut it off too early).",
        "- **weighted-mean loss** — `masked_mse` normalizes by `Σ(weight·mask)`, not the",
        "  raw count, so per-class `inv_nc` weights set *relative* attention without",
        "  globally shrinking the gradient scale.",
        "- **`w_coh` coherent de-bias** (=80) — a FLAT-in-k regularizer on the per-cell",
        "  θ-mean of the residual error; flattens the coherent low-k tilt the per-sim",
        "  MSE leaves free (drives gate failures 3/8→1/8).",
        "- **edge k-weight** (`edge_gain=3`, `lowk_extra=2`) — U-shaped per-k emphasis on",
        "  the band edges (low-k is where A_p pivots).",
        "- **data-range soft down-weight** — out-of-range bins (z∉[2.2,4.6] or k<1e-3)",
        "  get ~0.2× weight, focusing capacity on the modes DESI constrains.",
        "- **`term_w[p_resid]=8`** — up-weights the cosmology term (uniform default would",
        "  dilute it 1:5).",
        "- **early-stop on resid+coh** — the inference-relevant metric.",
    ))

    cells.append(code(
        "# Environment is set by the `emu-jax` kernelspec:",
        "#   PYTHONNOUSERSITE=1, PYTHONPATH=/home/mfho/hcd_priya, CPU-only JAX.",
        "import os",
        "os.chdir('/home/mfho/hcd_priya')   # so relative repo paths resolve from notebooks/",
        "import time, json",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import jax, jax.numpy as jnp",
        "",
        "# importing the package enables jax_enable_x64 (the structural identities need it)",
        "import hcd_analysis.emulator  # noqa: F401",
        "from hcd_analysis.emulator.data import (",
        "    load_cache, make_splits, make_batch, reconstruct_P_filt, safe_log,",
        "    cell_id, edge_emphasis_k_weight, datarange_mask, COARSE_NAMES, DATA_RANGE,",
        ")",
        "from hcd_analysis.emulator.train import train_fold, load_checkpoint, save_checkpoint",
        "from hcd_analysis.emulator.model import Emulator",
        "from scripts.run_loso_sweep import FINAL_RECIPE",
        "",
        "print('jax devices:', jax.devices())",
        "print('x64 enabled:', jax.config.jax_enable_x64)",
        "print('FINAL_RECIPE:', FINAL_RECIPE)",
    ))

    cells.append(md(
        "## 3. Load the production LF cache",
        "",
        "`load_cache` reads the v3.3 HDF5 cache, collapses the 15 fine HCD classes to 4",
        "coarse ones (`clean, LLS, subDLA, DLA`), builds the τ₀ coordinate, the",
        "unit-cube encoder input `x = [params_unit(9), z_unit(1)]`, the analytic",
        "`inv_nc` sample-variance weights, and the Nyquist masks.",
    ))

    cells.append(code(
        "LF_CACHE = '/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5'",
        "t0 = time.time()",
        "d = load_cache(LF_CACHE)",
        "n_k = d['P_tier_p'].shape[1]",
        "kf = d['kfkms'][0]               # shared k-grid (s/km)",
        "print(f'loaded in {time.time()-t0:.1f}s')",
        "print(f\"rows={d['P_tier_p'].shape[0]}  n_k={n_k}  \"",
        "      f\"sims={len(set(d['sim_name']))}  classes={COARSE_NAMES}\")",
        "print('k range [s/km]:', f'{kf.min():.2e} .. {kf.max():.2e}')",
        "print('z grid:', np.unique(np.round(d['z_grid'], 2)))",
        "print('key cache contents (shape):')",
        "for key in ['x', 'tau0', 'P_filt', 'delta', 'snap_dNdX', 'snap_f_nhi',",
        "            'inv_nc', 'w_c_cache', 'mask']:",
        "    print(f'  {key:14s} {np.asarray(d[key]).shape}')",
    ))

    cells.append(md(
        "### LOSO split for fold 0",
        "",
        "`make_splits` composes a **τ₀-edge holdout** (the extrapolation probe, excluded",
        "from both train and val) with a **leave-one-sim-out** (LOSO) partition. Fold 0",
        "holds out a disjoint group of *sims* for validation — so val accuracy measures",
        "generalization to **unseen cosmologies**.",
    ))

    cells.append(code(
        "FOLD = 0",
        "train_idx, val_idx, holdout_idx = make_splits(d, FOLD, n_folds=8, holdout_frac=0.15)",
        "print(f'fold {FOLD}:  train={len(train_idx)}  val={len(val_idx)}  '",
        "      f'holdout={len(holdout_idx)} (τ₀-edge, unseen)')",
        "val_sims = sorted(set(d['sim_name'][val_idx]))",
        "train_sims = sorted(set(d['sim_name'][train_idx]))",
        "print(f'held-out val sims ({len(val_sims)}):', val_sims)",
        "assert not (set(val_sims) & set(train_sims)), 'LOSO: val sims must be disjoint from train'",
        "print('LOSO no-straddle: OK (val sims disjoint from train)')",
    ))

    cells.append(md(
        "## 4. Train a short, illustrative fold-0",
        "",
        "> **Illustrative only.** We run ~35 epochs with a reduced baseline pre-fit",
        "> (~2000 steps) to keep the notebook to a couple of minutes. **Production uses",
        "> ~150–250 epochs over all 8 LOSO folds** with `prefit_baseline_epochs=8000`",
        "> (the default in `train_fold`).",
        "",
        "We feed `train_fold` the `FINAL_RECIPE` knobs verbatim: the edge k-weight, the",
        "`p_resid` up-weight via `term_w`, the FLAT coherent de-bias `w_coh`, the",
        "data-range down-weight, weight decay, and `n_basis=24` (SVD-warm-started).",
    ))

    cells.append(code(
        "# Build the FINAL_RECIPE knobs the train_fold call consumes.",
        "k_weight = edge_emphasis_k_weight(",
        "    kf, edge_gain=FINAL_RECIPE['edge_gain'], lowk_extra=FINAL_RECIPE['lowk_extra'])",
        "term_w = {'f_nhi': 1.0, 'dndx': 1.0, 'p_base': 1.0,",
        "          'p_resid': FINAL_RECIPE['p_resid_w'], 'delta': 1.0}",
        "",
        "# notebook-scale overrides (clearly illustrative; production = the comments)",
        "NB_EPOCHS = 35          # production: 180 (FINAL_RECIPE['epochs'])",
        "NB_PREFIT = 2000        # production: 8000 (train_fold default)",
        "",
        "t0 = time.time()",
        "model, norm_stats, history = train_fold(",
        "    d, train_idx, val_idx,",
        "    n_basis=FINAL_RECIPE['n_basis'], lr=FINAL_RECIPE['lr'],",
        "    epochs=NB_EPOCHS, batch_size=FINAL_RECIPE['batch'], seed=0,",
        "    key=jax.random.PRNGKey(0), patience=FINAL_RECIPE['patience'], n_k=n_k,",
        "    term_w=term_w, k_weight=k_weight, w_coh=FINAL_RECIPE['w_coh'],",
        "    datarange=FINAL_RECIPE['datarange'], weight_decay=FINAL_RECIPE['weight_decay'],",
        "    prefit_baseline_epochs=NB_PREFIT,",
        ")",
        "n_ep = len(history['train_loss'])",
        "print(f'trained {n_ep} epochs in {time.time()-t0:.1f}s '",
        "      f'(includes the {NB_PREFIT}-step baseline pre-fit)')",
    ))

    cells.append(md(
        "### Train / val loss curves",
        "",
        "**How to read:** the joint train (solid) and val (dashed) loss should both fall",
        "and plateau; the val **residual** loss (the cosmology term, dotted) is the",
        "inference-relevant early-stop metric. The vertical line marks the restored-best",
        "epoch. The baseline is *pre-fit then frozen*, so the joint loop is mostly",
        "training the θ-dependent residual + Head A.",
    ))

    cells.append(code(
        "ep = np.arange(1, len(history['train_loss']) + 1)",
        "fig, ax = plt.subplots(figsize=(8, 5))",
        "ax.semilogy(ep, history['train_loss'], '-', color='C0', lw=1.6, label='train (joint)')",
        "ax.semilogy(ep, history['val_loss'], '--', color='C1', lw=1.6, label='val (joint)')",
        "vr = np.asarray(history['val_resid_loss'])",
        "if np.isfinite(vr).any():",
        "    ax.semilogy(ep, vr, ':', color='C3', lw=1.4, label='val residual (cosmology)')",
        "# early-stop (restore-best): argmin of resid+coh when coh present, else resid",
        "vc = np.asarray(history['val_coh_loss'])",
        "metric = vr + vc if np.isfinite(vc).all() and vc.size else vr",
        "es = int(np.nanargmin(metric)) + 1 if np.isfinite(metric).any() else n_ep",
        "ax.axvline(es, color='k', ls='-.', lw=1.0, label=f'early-stop ep {es}')",
        "ax.set_xlabel('epoch'); ax.set_ylabel('loss (log)')",
        "ax.set_title('Fold-0 (illustrative): train/val loss\\n'",
        "             'FINAL_RECIPE knobs: w_coh=80, edge k-weight, p_resid=8, data-range')",
        "ax.legend(); ax.grid(alpha=0.3, which='both')",
        "fig.tight_layout(); plt.show()",
    ))

    cells.append(md(
        "## 5. Headline numbers for this short fold-0",
        "",
        "Three honest, inference-relevant diagnostics, reusing the *same logic* as the",
        "keeper diagnostics (`plot_performance_walkthrough._honest_decomp` and",
        "`diag_ap_fisher_bias.fisher_bias`):",
        "",
        "1. **honest θ-tracking corr** — correlation between the *true* and *predicted*",
        "   log-P1D deviation from the deployed baseline `m̂` (the cosmology signal). 1.0",
        "   = perfect tracking of the cosmology-driven departure.",
        "2. **deployed median |frac err|** (in-range) — `|P̂/P − 1|` over held-out rows,",
        "   restricted to the DESI data range.",
        "3. **A_p / n_s Fisher-bias quick-check** — the in-range deployed bias in σ units",
        "   (the inference gate is |bias| < 0.2σ).",
    ))

    cells.append(code(
        "# 1) honest θ-tracking corr (logic from plot_performance_walkthrough._honest_decomp)",
        "pf = norm_stats['P_filt']",
        "sig_marg, mu_marg, sig_cosmo = pf['sig_marg'], pf['mu_marg'], pf['sig_cosmo']",
        "b = make_batch(d, val_idx, norm_stats)",
        "pred = jax.vmap(model)(jnp.asarray(b['x']), jnp.asarray(b['tau0']))",
        "base = np.asarray(pred['P_filt_base']); resid = np.asarray(pred['P_filt_resid'])",
        "m_hat = base * sig_marg + mu_marg                 # deployed baseline logP",
        "lt = safe_log(d['P_filt'][val_idx])               # true logP",
        "lp = m_hat + sig_cosmo * resid                    # deployed pred logP",
        "dev_t = (lt - m_hat).ravel(); dev_p = (lp - m_hat).ravel()",
        "g = np.isfinite(dev_t) & np.isfinite(dev_p)",
        "theta_corr = float(np.corrcoef(dev_t[g], dev_p[g])[0, 1])",
        "",
        "# 2) deployed median |frac err| in the DESI data range",
        "P_pred = reconstruct_P_filt(base, resid, pf)",
        "P_true = d['P_filt'][val_idx]",
        "mk = d['mask'][val_idx]; ir = datarange_mask(d)[val_idx]",
        "keep = (np.isfinite(P_pred) & np.isfinite(P_true) & (P_true > 0)",
        "        & mk[:, None, :] & ir[:, None, :])",
        "with np.errstate(invalid='ignore', divide='ignore'):",
        "    frac = np.abs(P_pred / P_true - 1.0)",
        "frac = np.where(keep, frac, np.nan)",
        "median_frac = float(np.nanmedian(frac))",
        "",
        "print(f'honest θ-tracking corr        : {theta_corr:.3f}  (1.0 = perfect)')",
        "print(f'deployed median |frac err|    : {median_frac*100:.2f}%  (in-range, held-out)')",
    ))

    cells.append(code(
        "# 3) A_p / n_s Fisher-bias quick-check — port of diag_ap_fisher_bias.fisher_bias",
        "#    (9-param Fisher at the z=3 fiducial slice, in-range modes, σ units).",
        "PARAM_NAMES = ['ns', 'Ap', 'herei', 'heref', 'alphaq',",
        "               'hub', 'omegamh2', 'hireionz', 'bhfeedback']",
        "SIGMA_FRAC = {'clean': 0.03, 'LLS': 0.15, 'subDLA': 0.15, 'DLA': 0.15}  # DESI-DR1-like",
        "sc = pf['sig_cosmo']",
        "z_grid = d['z_grid']",
        "zsel = np.isclose(z_grid, 3.0, atol=0.05)",
        "tau0_fid = float(np.median(d['tau0'][zsel]))",
        "z_unit_fid = float(np.median(d['x'][zsel, 9]))",
        "vz = np.isclose(d['z_grid'][val_idx], 3.0, atol=0.05)",
        "mask_va = np.isfinite(safe_log(d['P_filt'][val_idx]))",
        "kvalid = (mask_va[vz].mean(0) > 0.5)",
        "z_j = jnp.asarray(z_unit_fid)",
        "",
        "def _rhat(theta9):",
        "    x = jnp.concatenate([theta9, z_j[None]])",
        "    return model(x, jnp.asarray(tau0_fid))['P_filt_resid']",
        "p0 = jnp.asarray(np.full(9, 0.5), dtype=jnp.float64)",
        "Jr = np.asarray(jax.jacfwd(_rhat)(p0))            # (4,n_k,9)",
        "J_logP = Jr * sc[:, :, None]",
        "",
        "x_va = jnp.asarray(d['x'][val_idx][vz]); tau_va = jnp.asarray(d['tau0'][val_idx][vz])",
        "prv = jax.vmap(model)(x_va, tau_va)",
        "bb = np.asarray(prv['P_filt_base']) * pf['sig_marg'] + pf['mu_marg']",
        "logP_hat = bb + sc * np.asarray(prv['P_filt_resid'])",
        "logP_true = safe_log(d['P_filt'][val_idx][vz]); ms = np.isfinite(logP_true)",
        "with np.errstate(invalid='ignore'):",
        "    delta = np.array([[np.nanmean(np.where(ms[:, ci, j],",
        "                       (logP_hat - logP_true)[:, ci, j], np.nan))",
        "                       for j in range(n_k)] for ci in range(4)])",
        "rJ, rd, rC = [], [], []",
        "for ci, nm in enumerate(COARSE_NAMES):",
        "    for j in range(n_k):",
        "        if (not kvalid[ci, j] or not np.all(np.isfinite(J_logP[ci, j]))",
        "                or not np.isfinite(delta[ci, j])):",
        "            continue",
        "        rJ.append(J_logP[ci, j]); rd.append(delta[ci, j]); rC.append(SIGMA_FRAC[nm] ** 2)",
        "J9 = np.array(rJ); dv = np.array(rd); Cinv = 1.0 / np.array(rC)",
        "F = (J9.T * Cinv) @ J9",
        "ridge = 1e-12 * np.trace(F) / 9.0",
        "Finv = np.linalg.inv(F + ridge * np.eye(9))",
        "sigma = np.sqrt(np.diag(Finv))",
        "bias = (Finv @ ((J9.T * Cinv) @ dv)) / sigma",
        "bias_d = {PARAM_NAMES[i]: float(bias[i]) for i in range(9)}",
        "print(f\"Fisher-bias A_p : {bias_d['Ap']:+.3f} σ   (gate |bias|<0.2σ)\")",
        "print(f\"Fisher-bias n_s : {bias_d['ns']:+.3f} σ\")",
        "print(f'(over {len(dv)} in-range modes at z=3)')",
    ))

    cells.append(md(
        "> **Note (illustrative caveat).** These come from a short ~35-epoch single-fold",
        "> run; the *finalized* 8-fold checkpoints (loaded in NB 2) report the production",
        "> numbers. The point here is that the recipe trains a sensible model fast: even",
        "> this short fold tracks the cosmology signal well and stays inside the Fisher",
        "> gate.",
    ))

    cells.append(md(
        "## 6. Save / load a checkpoint",
        "",
        "`save_checkpoint` writes three files: `.eqx` (the Equinox leaves), `.meta.json`",
        "(arch config completed from the model's static fields, the git SHA, the k-grid,",
        "and the FINAL_RECIPE), and `.norm.pkl` (the train-split normalization). The",
        "checkpoint is **self-describing** — `load_checkpoint` rebuilds the exact model.",
    ))

    cells.append(code(
        "ckpt = '/home/mfho/hcd_priya/checkpoints/walkthrough_nb_fold0'",
        "recipe = dict(",
        "    n_basis=FINAL_RECIPE['n_basis'], p_resid_w=FINAL_RECIPE['p_resid_w'],",
        "    edge_gain=FINAL_RECIPE['edge_gain'], lowk_extra=FINAL_RECIPE['lowk_extra'],",
        "    w_coh=FINAL_RECIPE['w_coh'], weight_decay=FINAL_RECIPE['weight_decay'],",
        "    datarange=FINAL_RECIPE['datarange'], epochs=NB_EPOCHS, term_w=term_w,",
        "    note='illustrative short notebook fold-0; NOT a production checkpoint')",
        "save_checkpoint(ckpt, model, {'in_dim': 10, 'n_k': n_k, 'n_basis': FINAL_RECIPE['n_basis']},",
        "                norm_stats, seed=0, kfkms=d['kfkms'], cache_path=LF_CACHE, recipe=recipe)",
        "print('wrote', ckpt + '.{eqx,meta.json,norm.pkl}')",
        "",
        "# round-trip: load_checkpoint rebuilds the model + meta + norm",
        "m2, meta2, norm2 = load_checkpoint(ckpt)",
        "print('reloaded arch_cfg:', meta2['arch_cfg'])",
        "print('reloaded git_sha :', meta2.get('git_sha'))",
        "# verify the reloaded model gives identical predictions on a few rows",
        "chk = val_idx[:8]",
        "p_a = jax.vmap(model)(jnp.asarray(d['x'][chk]), jnp.asarray(d['tau0'][chk]))['P_filt_resid']",
        "p_b = jax.vmap(m2)(jnp.asarray(d['x'][chk]), jnp.asarray(d['tau0'][chk]))['P_filt_resid']",
        "print('round-trip max |Δ residual|:', float(np.max(np.abs(np.asarray(p_a) - np.asarray(p_b)))))",
    ))

    cells.append(md(
        "## 7. The full 8-fold LOSO sweep + the multi-fidelity layer",
        "",
        "**Full LOSO sweep (production).** `scripts/run_loso_sweep.py` runs the recipe",
        "above across all 8 LOSO folds, saving `checkpoints/final_fold{0..7}.{eqx,...}`,",
        "the per-epoch histories, and the aggregated **error vector** (`error_vector.npz`,",
        "RMS-over-folds fractional P1D σ + DLA high-k shot-noise flag) that feeds the",
        "likelihood covariance. To reproduce:",
        "",
        "```bash",
        "PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \\",
        "  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_loso_sweep.py",
        "#   add --smoke for a fast 2-fold / few-epoch end-to-end check",
        "```",
        "",
        "**Multi-fidelity (LF→HF).** The frozen LF backbone (these checkpoints) is the",
        "input to the resolution-correction layer in",
        "`hcd_analysis.emulator.multifidelity`. The forward model is",
        "`P_MF = (ρ(k,z) · f_LF) · res_corr`, with the validated default head being",
        "ρ(k,z)-only (`FixedMeanHead`, no learned θ-correction — the learned-MLP head",
        "over-fits the θ-gradient on the 6 HF sims). Build it via:",
        "",
        "```python",
        "from hcd_analysis.emulator import multifidelity as MF",
        "lf_model, meta, lf_norm, lf_logk = MF.load_lf_backbone(fold=0)",
        "# measure targets, build the default head, then MF.build_multifidelity(...)",
        "# end-to-end driver: scripts/build_mf_delta.py",
        "```",
        "",
        "NB 2 shows the MF high-k performance figure (B6).",
    ))

    return finalize(new_notebook(cells=cells))


# ============================================================================ #
# NOTEBOOK 2 — PLOTTING / PERFORMANCE WALKTHROUGH
# ============================================================================ #
def build_plotting_nb():
    cells = []

    cells.append(md(
        "# Phase-2b Lyα P1D emulator — Notebook 2: performance walkthrough",
        "",
        "This notebook loads the **finalized LOSO checkpoints** and renders the key",
        "performance figures **inline**, reusing the figure logic from the keeper",
        "diagnostic `scripts/plot_performance_walkthrough.py` (imported, not copied) so",
        "the notebook and the production walkthrough can never disagree.",
        "",
        "All figures are evaluated on **held-out** sims (LOSO). It is **robust to missing",
        "checkpoints**: if `checkpoints/final_fold{0..7}` are absent it falls back to",
        "`walkthrough_fold0` (or any single fold) and prints a clear message rather than",
        "crashing.",
        "",
        "**Companion:** `01_emulator_training.ipynb` shows how these checkpoints are trained.",
    ))

    cells.append(code(
        "# The `emu-jax` kernelspec sets the environment (CPU JAX, PYTHONPATH).",
        "import os",
        "os.chdir('/home/mfho/hcd_priya')   # so relative repo paths resolve from notebooks/",
        "import json, warnings",
        "from pathlib import Path",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import jax, jax.numpy as jnp",
        "",
        "import hcd_analysis.emulator  # noqa: F401  (enables x64)",
        "from hcd_analysis.emulator.data import (",
        "    load_cache, make_splits, reconstruct_P_filt, COARSE_NAMES, DATA_RANGE,",
        ")",
        "from hcd_analysis.emulator import train as T",
        "# reuse the keeper walkthrough's figure functions directly",
        "import scripts.plot_performance_walkthrough as W",
        "",
        "print('jax devices:', jax.devices())",
        "CACHE = W.CACHE",
        "N_FOLDS = W.N_FOLDS",
        "PARAM_NAMES = W.PARAM_NAMES",
    ))

    cells.append(md(
        "## 1. Load the cache + the finalized checkpoints (robust)",
        "",
        "We load whichever finalized folds are on disk. If none of `final_fold{0..7}`",
        "exist we fall back to `walkthrough_fold0`. The single-fold figures use fold 0",
        "(or the fallback); the multi-fold figures (B5 Fisher spread, Head A) use",
        "whatever folds loaded.",
    ))

    cells.append(code(
        "d = load_cache(CACHE)",
        "kf = d['kfkms'][0]",
        "z_levels = np.unique(np.round(d['z_grid'], 1))",
        "print(f\"cache: {d['P_tier_p'].shape[0]} rows, n_k={len(kf)}, \"",
        "      f'z-levels={len(z_levels)}')",
        "",
        "CKPT = str(W.ROOT / 'checkpoints/final_fold{f}')",
        "",
        "def _exists(stem):",
        "    return Path(stem + '.eqx').exists() and Path(stem + '.meta.json').exists()",
        "",
        "models, norms, splits = {}, {}, {}",
        "avail = [f for f in range(N_FOLDS) if _exists(CKPT.format(f=f))]",
        "if avail:",
        "    for f in avail:",
        "        m, meta, nrm = T.load_checkpoint(CKPT.format(f=f))",
        "        models[f], norms[f] = m, nrm",
        "        splits[f] = make_splits(d, f, n_folds=N_FOLDS, holdout_frac=0.15)",
        "    print(f'loaded finalized folds: {avail}')",
        "else:",
        "    fb = str(W.ROOT / 'checkpoints/walkthrough_fold0')",
        "    if _exists(fb):",
        "        m, meta, nrm = T.load_checkpoint(fb)",
        "        models[0], norms[0] = m, nrm",
        "        splits[0] = make_splits(d, 0, n_folds=N_FOLDS, holdout_frac=0.15)",
        "        avail = [0]",
        "        print('FALLBACK: final_fold* missing; using checkpoints/walkthrough_fold0 as fold 0')",
        "    else:",
        "        raise FileNotFoundError(",
        "            'No finalized or walkthrough checkpoints found. '",
        "            'Run scripts/run_loso_sweep.py first (see NB 1).')",
        "",
        "# the walkthrough figure fns reference a single (model,norm,split); fold 0 here.",
        "F0 = avail[0]",
        "m0, n0, s0 = {0: models[F0]}, {0: norms[F0]}, {0: splits[F0]}",
        "n_loaded = len(avail)",
        "print(f'{n_loaded} fold(s) available for the multi-fold figures')",
    ))

    cells.append(md(
        "## HEAD B — P1D / cosmology",
        "",
        "### B1 — Predicted vs true per-class P1D (+ fractional residual)",
        "",
        "**What:** deployed `P̂` vs truth for the 3 best-sampled held-out rows per class,",
        "log-log, with the fractional residual underneath. **How to read:** dashed (pred)",
        "should overlay solid (true); the residual subpanel should sit near 0 (±). The",
        "grey band is below the DESI `k_min`; the right-edge turnover is the per-row",
        "Nyquist. **Good:** tight overlay, residual within a few %. **Bad:** systematic",
        "tilt or offset in the residual.",
    ))
    cells.append(code(
        "p, fr = W.figB1_pred_vs_true_p1d(m0, n0, s0, d, kf)",
        "print('B1 pred/true frac-resid RMS per class:',",
        "      {k: round(v, 4) for k, v in fr.items()})",
        "from IPython.display import Image, display",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### B2 — Deployed fractional P1D error vs k (in-range, all folds) + CV floor",
        "",
        "**What:** median `|P̂/P − 1|` vs k per class (IQR shaded), over **all** loaded",
        "folds' held-out rows, restricted to the data range; dashed black = the",
        "cosmic-variance (CV) floor — the irreducible sampling scatter. **How to read:**",
        "the error should approach (not beat) the CV floor; clean is tightest, HCD",
        "classes are noisier (fewer absorbers). **Good:** error near the CV floor across",
        "k. **Bad:** error rising well above the floor, especially a coherent low-k rise.",
    ))
    cells.append(code(
        "cv_floor = W.load_cv_floor()",
        "frac, kf2 = W._deployed_frac_in_range(models, norms, splits, d)",
        "p, rms = W.figB2_deployed_frac_err_vs_k(frac, kf2, cv_floor)",
        "print('B2 deployed frac-err RMS per class (in-range):',",
        "      {k: round(v, 4) for k, v in rms.items()})",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### B3 — Learned cosmology response ∂lnP/∂θ vs k (jacfwd)",
        "",
        "**What:** `σ_cosmo·∂r̂/∂θ` (= `∂lnP̂/∂θ`) vs k for all 9 params, by class, via",
        "`jax.jacfwd` at a fiducial held-out point. **How to read:** larger |response| =",
        "the P1D constrains that param more; sign-flips across k = shape/tilt sensitivity.",
        "**Good:** A_p shows a coherent low-k amplitude response, n_s a tilt; IGM params",
        "respond where physically expected. **Bad:** a param with ~zero response",
        "everywhere is unconstrained by the P1D alone.",
    ))
    cells.append(code(
        "p, constr = W.figB3_cosmology_response(m0, n0, s0, d, kf)",
        "print('B3 response amplitude |∂lnP/∂θ|_rms per param:',",
        "      {k: round(v, 3) for k, v in constr.items()})",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### B4 — Within-cell θ-tracking + honest (a)/(b) error decomposition",
        "",
        "**What:** *left* — true vs predicted log-P1D deviation from the deployed",
        "baseline `m̂` (the cosmology signal), pooled over classes; *right* — the",
        "whitened error split into **(a)** residual-head fit error and **(b)** baseline",
        "mis-fit (σ_cosmo-amplified), per class. **How to read:** left hexbin should hug",
        "the `y=x` line with corr→1 and spread-ratio→1; right bars should be a small",
        "fraction of `σ_signal`. **Good:** high corr, small (a)+(b). **Bad:** corr well",
        "below 1, or a large (b) (baseline drifted off its cell-mean floor).",
    ))
    cells.append(code(
        "p, tt = W.figB4_theta_tracking(m0, n0, s0, d)",
        "print(f\"B4 honest θ-tracking corr = {tt['honest_corr']:.3f}, \"",
        "      f\"spread-ratio = {tt['spread_ratio']:.3f}\")",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### B5 — Per-fold A_p & n_s Fisher-bias spread",
        "",
        "**What:** the deployed in-range Fisher-bias of A_p and n_s for each loaded LOSO",
        "fold (DESI-DR1-like diagonal covariance, z=3). **How to read:** every bar should",
        "sit inside the ±0.2σ green gate; the dotted line is the RMS-over-folds. **Good:**",
        "all folds in the gate, small RMS. **Bad:** a fold spiking past the gate (a",
        "cosmology the model extrapolates poorly to).",
        "",
        "> With a single fold loaded this shows just that fold (the spread needs all 8).",
    ))
    cells.append(code(
        "p, fb = W.figB5_fisher_bias_perfold(models, norms, splits, d)",
        "print(f\"B5 A_p RMS = {fb['Ap_rms_sigma']:.3f}σ (max |{fb['Ap_max_abs']:.3f}|);  \"",
        "      f\"n_s RMS = {fb['ns_rms_sigma']:.3f}σ (max |{fb['ns_max_abs']:.3f}|)\")",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### B6 — Multi-fidelity high-k: MF vs LF-extrapolated vs HF-standalone",
        "",
        "**What:** RMS fractional P1D error vs k in the KODIAQ high-k band for the",
        "validated ρ(k,z)-only MF model, the LF-extrapolated-alone backbone, and an",
        "HF-standalone fit, aggregated over HF-LOSO folds. **How to read:** above the LF",
        "Nyquist (dash-dot ~0.069 s/km) the LF extrapolation degrades; the MF (blue)",
        "should track best across the shaded KODIAQ band. **Good:** MF below both",
        "alternatives at high k. Needs the HR cache + res_corr table; degrades gracefully",
        "if absent.",
    ))
    cells.append(code(
        "try:",
        "    p, mfsum = W.figB6_mf_high_k(d)",
        "    if p is None:",
        "        print('B6 skipped:', mfsum)",
        "    else:",
        "        print('B6 KODIAQ-band RMS frac err per class:')",
        "        for nm, v in mfsum.items():",
        "            print(f\"  {nm:7s} MF={v['mf_kodiaq']:.3f}  \"",
        "                  f\"LF-extrap={v['lf_kodiaq']:.3f}  HF-standalone={v['hf_kodiaq']:.3f}\")",
        "        display(Image(filename=str(p)))",
        "except Exception as e:",
        "    print('B6 unavailable (MF setup failed); continuing:', repr(e))",
    ))

    cells.append(md(
        "## HEAD A — dN/dX + CDDF",
        "",
        "We build the Head-A held-out stacks over all loaded folds once, then render its",
        "figures. (`_head_a_stacks` evaluates Head A on every fold's held-out rows.)",
    ))
    cells.append(code(
        "S = W._head_a_stacks(models, norms, splits, d)",
        "print('Head-A held-out stack built:',",
        "      {k: np.asarray(v).shape for k, v in S.items()",
        "       if k in ('dndx_fe', 'fnhi_fe', 'wc_emu', 'ptp_fe')})",
    ))

    cells.append(md(
        "### A1 — dN/dX fractional accuracy vs z (per class)",
        "",
        "**What:** median & p95 `|frac err|` of the predicted line-density `dN/dX` vs z,",
        "for LLS / subDLA / DLA. **How to read:** both curves should sit under the 2.5%",
        "target (red dotted); DLA is noisiest (rarest). **Good:** median well under 2.5%",
        "across z. **Bad:** a class/z rising above target.",
    ))
    cells.append(code(
        "p, a1 = W.figA1_dndx_pred_vs_true(S, z_levels)",
        "print('A1 dN/dX median |frac err| per class:',",
        "      {k: round(v['median_pct'], 3) for k, v in a1.items()})",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### A2 — CDDF (f_NHI) accuracy vs logN_HI",
        "",
        "**What:** median & p95 `|frac err|` of the CDDF per N_HI bin; vertical lines mark",
        "the class boundaries (17.2 / 19.0 / 20.3), the shaded band is the shot-noise tail",
        "(logN_HI ≥ 21.5), and the grey curve (right axis) is the valid-row fraction.",
        "**How to read:** error should be low and flat across the populated range and is",
        "*expected* to blow up in the shot-noise tail (few absorbers). **Good:** sub-few-%",
        "below 21.5. **Bad:** large error *below* 21.5 where statistics are ample.",
    ))
    cells.append(code(
        "p, a2 = W.figA2_cddf_pred_vs_true(S)",
        "print(f\"A2 CDDF median |frac err| = {a2['median_pct']:.3f}%  (p95 {a2['p95_pct']:.3f}%)\")",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### A3 — w_c → P_tier_p coupling",
        "",
        "**What:** *left* — the class weights `w_c` from emulated vs true dN/dX",
        "(round-trip); *middle* — the `|w_c(emu)−w_c(true)|` error distribution; *right*",
        "— the resulting fractional error on the **structural total** `P_tier_p = Σ_c",
        "w_c·P_filt`. **How to read:** the scatter should hug `y=x`; the P_tier_p error",
        "should sit inside the ±2.5% target band. **Good:** tight w_c round-trip, P_tier_p",
        "p95 ≲ 2.5%. **Bad:** biased w_c that propagates into a P_tier_p offset.",
    ))
    cells.append(code(
        "p, a3 = W.figA3_wc_ptierp_coupling(S)",
        "print(f\"A3 P_tier_p frac err: median {a3['ptierp_median_pct']:.3f}%  \"",
        "      f\"p95 {a3['ptierp_p95_pct']:.3f}%\")",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### A4 — Head-A dN/dX error heatmap (class × z × fold)",
        "",
        "**What:** median `|frac err|` of dN/dX as a heatmap over (fold, z) for each HCD",
        "class. **How to read:** uniform color = well-balanced across folds and z; a hot",
        "row flags a weak fold, a hot column a weak z. **Good:** even, cool map. **Bad:** a",
        "persistent hot stripe.",
    ))
    cells.append(code(
        "p = W.figA4_head_a_error_heatmap(S, z_levels)",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### A5 — dN/dX → α_c prior-center sanity",
        "",
        "**What:** the emulated `w_c` (which sets the α_c prior center) vs the cache's",
        "empirical CDDF-integral `w_c`. **How to read:** the scatter should hug `y=x` and",
        "the deviation histogram should peak well inside the 2.5% mark. **Good:** the",
        "emulated prior center matches the cache CDDF. **Bad:** a systematic offset",
        "(the prior would be mis-centered).",
    ))
    cells.append(code(
        "p, a5 = W.figA5_alpha_prior_sanity(S)",
        "print('A5 |w_c(emu)−w_c(cache)| median per class:',",
        "      {k: round(v['median'], 4) for k, v in a5.items()})",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "### B7 — Δ_c HCD class templates",
        "",
        "**What:** the deployed per-HCD-class `Δ_c` template (HeadB delta output) vs k at",
        "several held-out z slices (symlog-y). **How to read:** `Δ_c` is the shape that",
        "re-weights the structural `P_tier_p`; the low-k sign-flip is physical. **Good:**",
        "smooth, z-ordered templates. **Bad:** noisy, non-monotone-in-z templates.",
    ))
    cells.append(code(
        "p = W.figB7_delta_c_templates(m0, n0, s0, d, kf)",
        "display(Image(filename=str(p)))",
    ))

    cells.append(md(
        "## Summary",
        "",
        "All figures above are written to",
        "`figures/analysis/06_performance_walkthrough/` (and rendered inline) by the same",
        "functions the production walkthrough script uses. To regenerate the entire set",
        "outside the notebook (incl. the headline-numbers JSON):",
        "",
        "```bash",
        "PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \\",
        "  /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_performance_walkthrough.py",
        "#   add --no-mf to skip B6 if the HR cache / res_corr table is unavailable",
        "```",
    ))

    return finalize(new_notebook(cells=cells))


def main():
    nb1 = build_training_nb()
    nb2 = build_plotting_nb()
    p1 = NB_DIR / "01_emulator_training.ipynb"
    p2 = NB_DIR / "02_emulator_plotting.ipynb"
    nbf.write(nb1, str(p1))
    nbf.write(nb2, str(p2))
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
