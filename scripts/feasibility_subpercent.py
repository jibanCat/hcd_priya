"""Phase-2b sub-percent feasibility study (REAL experiments on the τ₀ cache).

Question (design doc §8): is SUB-PERCENT absolute |P̂/P−1| per class achievable
with the baseline+residual normalization design, and what limits it?

Five experiments on the fold-0 LOSO split of ``observables_tau0_lf.h5``:

  1. SVD basis-rank ceiling per class — the representation floor. SVD-reconstruct
     logP_filt at n_basis∈{8,12,16,24,32,48}; per-class frac-RMS in P. Does rank-12
     cap any class above 1%?
  2. Baseline-head trainability with the inv_nc FIX (the decisive test for term (b)).
     The θ-blind baseline target is the (z,τ₀) cell-mean. Train a baseline head with
     the loss normalized by Σ(weight·mask) (NOT inv_nc-shrunk), sweeping width/depth/
     epochs. Measure RMS(m̂ − cell_mean) in σ_cosmo units → does (b) collapse from
     0.8·σ_cosmo to <<σ_cosmo? Also the irreducible finite-sim baseline floor.
  3. Residual-head ceiling (term (a)). With a PERFECT (empirical) baseline, train the
     encoder + residual head hard; achievable (a) per class, and the finite-60-sim
     floor via the train-vs-val gap.
  4. Combined deployed accuracy with BOTH fixes — train baseline (fixed loss, deep) +
     residual (hard) and report per-class deployed |P̂/P−1| (median/p95) on held-out
     sims.
  5. Within-data-k-range check — DESI (~1e-3..2e-2 s/km) and KODIAQ (~3e-3..0.1)
     k-windows, excluding k<data-k_min where the known low-k spike lives.

Env (MANDATORY): PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/feasibility_subpercent.py
Writes figures to figures/analysis/04_emulator/ and a JSON summary.
"""
from __future__ import annotations

import json
import time

import hcd_analysis.emulator  # noqa: F401 -- enables jax_enable_x64 on import
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import (load_cache, make_splits, cell_id, safe_log,
                                        fit_baseline_residual_norm)
from hcd_analysis.emulator.model import Encoder, HeadB, BaselineHead, svd_basis_init

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "figures/analysis/04_emulator"
CLS = ("clean", "LLS", "subDLA", "DLA")
NK = 172
RANKS = (8, 12, 16, 24, 32, 48)
# Survey k-windows (s/km). DESI DR1 P1D and KODIAQ-SQUAD approximate ranges.
DESI_K = (1e-3, 2e-2)
KODIAQ_K = (3e-3, 0.1)


def _fracP_rms(recon_log, true_log, mask):
    """frac-RMS of P̂/P−1 over masked entries (recon/true in log space)."""
    frac = np.exp(recon_log - true_log) - 1.0
    return float(np.sqrt(np.mean(frac[mask] ** 2)))


# --------------------------------------------------------------------------- #
# EXPERIMENT 1 — SVD basis-rank ceiling per class (representation floor)        #
# --------------------------------------------------------------------------- #
def exp1_svd_rank(d, tr, va):
    print("\n=== EXP 1: SVD basis-rank reconstruction ceiling per class ===")
    logP = safe_log(d["P_filt"])
    res = {nm: {} for nm in CLS}
    sval = {}
    for ci, nm in enumerate(CLS):
        Xtr = logP[tr, ci, :]
        Xva = logP[va, ci, :]
        gtr = np.all(np.isfinite(Xtr), 1)
        gva = np.all(np.isfinite(Xva), 1)
        Xtr, Xva = Xtr[gtr], Xva[gva]
        mu = Xtr.mean(0)
        U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        sval[nm] = S
        for r in RANKS:
            B = Vt[:r]
            recon = (Xva - mu) @ B.T @ B + mu          # held-out-sim recon
            frac = np.exp(recon - Xva) - 1.0
            res[nm][r] = float(np.sqrt(np.mean(frac ** 2)))
        line = " ".join(f"r{r}:{res[nm][r]*100:.3f}%" for r in RANKS)
        print(f"  {nm:7}: {line}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for nm in CLS:
        ax[0].plot(RANKS, [res[nm][r] * 100 for r in RANKS], "o-", label=nm)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8, label="1% line")
    ax[0].set_xlabel("SVD n_basis"); ax[0].set_ylabel("held-out recon frac-RMS in P [%]")
    ax[0].set_title("Exp 1: SVD rank ceiling per class"); ax[0].set_yscale("log")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    for nm in CLS:
        s = sval[nm]; ax[1].plot(np.arange(1, len(s) + 1), s / s[0], lw=1.2, label=nm)
    ax[1].axvline(12, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("singular index"); ax[1].set_ylabel("σ_i/σ_0")
    ax[1].set_title("logP_filt singular spectrum"); ax[1].set_yscale("log")
    ax[1].set_xlim(0, 60); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/feas_exp1_svd_rank.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}/feas_exp1_svd_rank.png")
    return res


# --------------------------------------------------------------------------- #
# EXPERIMENT 2 — baseline-head trainability with the inv_nc FIX (term b)        #
# --------------------------------------------------------------------------- #
class DeepBaseline(eqx.Module):
    """θ-blind (z,τ₀)→(4,n_k) baseline, depth-configurable, low-rank decode.

    Generalizes the production BaselineHead (which is 1 hidden layer); depth is
    the variable Exp 2 sweeps to separate the optimization floor from the
    representation floor."""
    layers: list
    basis: jax.Array
    nb: int = eqx.field(static=True)

    def __init__(self, nb, width, depth, binit, key):
        ks = jax.random.split(key, depth + 1)
        dims = [2] + [width] * depth
        self.layers = ([eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i]) for i in range(depth)]
                       + [eqx.nn.Linear(width, 4 * nb, key=ks[depth])])
        self.basis = jnp.asarray(binit)
        self.nb = nb

    def __call__(self, z, tau0):
        x = jnp.stack([jnp.atleast_1d(z)[0], jnp.atleast_1d(tau0)[0]])
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        c = self.layers[-1](x).reshape(4, self.nb)
        return c @ self.basis


def _cell_table(d, idx, cm, mu, sm):
    """Per distinct (z,τ₀)-cell: input (z_unit, τ₀) + σ_marg-standardized cell-mean."""
    c = cell_id(d, idx)
    z_unit = d["x"][:, 9]
    uc = np.unique(c)
    cz, ct, cy = [], [], []
    for cc in uc:
        rows = idx[c == cc]
        cz.append(z_unit[rows][0]); ct.append(d["tau0"][rows][0]); cy.append(cm[int(cc)])
    cz, ct, cy = np.array(cz), np.array(ct), np.stack(cy)
    tb = (cy - mu) / sm
    return cz, ct, tb, np.isfinite(tb), uc


def _train_baseline(cz, ct, tb, mask, nb, width, depth, epochs, lr, seed=0):
    M = tb.reshape(-1, NK); M = M[np.all(np.isfinite(M), 1)]
    binit = np.asarray(svd_basis_init(jnp.asarray(M), nb))
    head = DeepBaseline(nb, width, depth, binit, jax.random.PRNGKey(seed))
    Z, T = jnp.asarray(cz), jnp.asarray(ct)
    Y = jnp.asarray(np.where(mask, tb, 0.0)); Mk = jnp.asarray(mask.astype(jnp.float64))

    def loss(h):                                   # inv_nc FIX: uniform Σ(mask) norm
        pred = jax.vmap(h)(Z, T)
        diff = jnp.where(Mk > 0, pred - Y, 0.0)
        return jnp.sum(diff ** 2) / jnp.maximum(jnp.sum(Mk), 1.0)

    opt = optax.adamw(optax.cosine_decay_schedule(lr, epochs), weight_decay=1e-7)
    st = opt.init(eqx.filter(head, eqx.is_array))

    @eqx.filter_jit
    def step(h, st):
        l, g = eqx.filter_value_and_grad(loss)(h)
        u, st = opt.update(g, st, eqx.filter(h, eqx.is_array))
        return eqx.apply_updates(h, u), st, l
    for _ in range(epochs):
        head, st, _ = step(head, st)
    return head


def exp2_baseline(d, tr, va, pf, sc_over_sm):
    print("\n=== EXP 2: baseline-head trainability (inv_nc FIX) — term (b) ===")
    mu, sm, sc, cm = pf["mu_marg"], pf["sig_marg"], pf["sig_cosmo"], pf["cell_mean"]
    cz, ct, tb, mask, _ = _cell_table(d, tr, cm, mu, sm)
    print(f"  {len(cz)} distinct (z,τ₀) cells (train==val under LOSO)")

    # (2a) representation floor: SVD reconstruction of the cell-means themselves.
    print("  -- representation floor: SVD recon of the 306 cell-means (σ_cosmo units)")
    rep_floor = {}
    for ci, nm in enumerate(CLS):
        X = tb[:, ci, :]; g = np.all(np.isfinite(X), 1); X = X[g]
        mc = X.mean(0); _, _, Vt = np.linalg.svd(X - mc, full_matrices=False)
        row = {}
        for r in (8, 12, 24, 48):
            rec = (X - mc) @ Vt[:r].T @ Vt[:r] + mc
            row[r] = float(np.sqrt(np.mean((rec - X) ** 2)) / sc_over_sm[ci])
        rep_floor[nm] = row
        print(f"     {nm:7}: " + " ".join(f"r{r}:{row[r]:.4f}sc" for r in (8, 12, 24, 48)))

    # (2b) achievable head fit — sweep depth/width/epochs.
    print("  -- achievable head fit RMS(m̂ − cell_mean_train) [σ_cosmo units]")
    sweep = [("d1_w64", 12, 64, 1, 4000, 3e-3),     # ≈ production BaselineHead
             ("d1_w128", 24, 128, 1, 5000, 3e-3),
             ("d3_w256_nb24", 24, 256, 3, 12000, 2e-3),
             ("d3_w256_nb48", 48, 256, 3, 15000, 2e-3)]
    fit_res = {}
    for tag, nb, w, dp, ep, lr in sweep:
        t0 = time.time()
        head = _train_baseline(cz, ct, tb, mask, nb, w, dp, ep, lr)
        pred = np.asarray(jax.vmap(head)(jnp.asarray(cz), jnp.asarray(ct)))
        pc = [float(np.sqrt(np.mean((pred[:, ci, :] - tb[:, ci, :])[mask[:, ci, :]] ** 2))
                    / sc_over_sm[ci]) for ci in range(4)]
        overall = float(np.sqrt(np.mean((pred - tb)[mask] ** 2)) / np.mean(sc_over_sm))
        fit_res[tag] = {"perclass_sc": pc, "overall_sc": overall,
                        "cfg": dict(nb=nb, width=w, depth=dp, epochs=ep)}
        print(f"     {tag:14}: overall={overall:.4f}sc | perclass " +
              " ".join(f"{x:.4f}" for x in pc) + f"  [{time.time()-t0:.0f}s]")

    # (2c) irreducible finite-sim baseline floor: train cell-mean vs full-data cell-mean.
    logP = safe_log(d["P_filt"])
    call = cell_id(d)
    full_cm = {int(c): np.nanmean(logP[call == c], 0) for c in np.unique(call)}
    cva = cell_id(d, va)
    fin = {ci: [] for ci in range(4)}
    for c in np.unique(cva):
        if int(c) not in cm:
            continue
        diff = (cm[int(c)] - full_cm[int(c)]) / sc
        for ci in range(4):
            v = diff[ci][np.isfinite(diff[ci])]; fin[ci].extend(v.tolist())
    fin_floor = {CLS[ci]: float(np.sqrt(np.mean(np.array(fin[ci]) ** 2))) for ci in range(4)}
    print("  -- finite-sim baseline floor (train cell-mean vs 60-sim cell-mean, σ_cosmo):")
    print("     " + " ".join(f"{nm}:{fin_floor[nm]:.4f}" for nm in CLS))

    # figure: achievable head fit vs config, with the finite-sim floor + the (current ~0.8) line.
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    tags = [s[0] for s in sweep]
    width_bar = 0.18
    xs = np.arange(len(CLS))
    for j, tag in enumerate(tags):
        ax.bar(xs + (j - 1.5) * width_bar, fit_res[tag]["perclass_sc"], width_bar, label=tag)
    ax.plot(xs, [fin_floor[nm] for nm in CLS], "k*", ms=14, label="finite-sim floor")
    ax.axhline(0.84, color="r", ls="--", lw=0.9, label="current (undertrained) ≈0.84")
    ax.set_xticks(xs); ax.set_xticklabels(CLS)
    ax.set_ylabel("baseline misfit (b) [σ_cosmo]")
    ax.set_title("Exp 2: baseline-head term (b) with inv_nc fix + capacity sweep")
    ax.set_yscale("log"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/feas_exp2_baseline_floor.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}/feas_exp2_baseline_floor.png")
    return {"rep_floor": rep_floor, "fit": fit_res, "finite_sim_floor": fin_floor}


# --------------------------------------------------------------------------- #
# EXPERIMENT 3 — residual-head ceiling (term a) with a perfect baseline         #
# --------------------------------------------------------------------------- #
class ResidModel(eqx.Module):
    enc: Encoder
    head: HeadB

    def __init__(self, nb, binit, key):
        k1, k2 = jax.random.split(key)
        self.enc = Encoder(in_dim=10, key=k1)
        self.head = HeadB(latent=64, n_k=NK, n_basis=nb, p_filt_basis_init=binit, key=k2)

    def __call__(self, x, tau0):
        return self.head(self.enc(x), tau0)["P_filt_resid"]


def _resid_target(d, idx, cm, sc):
    logP = safe_log(d["P_filt"])
    c = cell_id(d, idx)
    T = np.empty((len(idx), 4, NK))
    for i, (r, ci) in enumerate(zip(idx, c)):
        T[i] = (logP[r] - cm[int(ci)]) / sc
    return T


def _train_resid(d, tr, Ttr, mtr, nb, epochs, lr, bs=512, seed=0, wd=1e-5):
    M = Ttr.reshape(-1, NK); M = M[np.all(np.isfinite(M), 1)]
    binit = np.asarray(svd_basis_init(jnp.asarray(M), nb))
    m = ResidModel(nb, binit, jax.random.PRNGKey(seed))
    Xt = jnp.asarray(d["x"][tr]); Tt = jnp.asarray(d["tau0"][tr])
    Yt = jnp.asarray(np.where(mtr, Ttr, 0.0)); Mt = jnp.asarray(mtr.astype(jnp.float64))
    n = len(tr); spe = int(np.ceil(n / bs)); tot = spe * epochs
    opt = optax.adamw(optax.cosine_decay_schedule(lr, tot), weight_decay=wd)
    st = opt.init(eqx.filter(m, eqx.is_array))

    def loss(m, xb, tb, yb, mb):
        pred = jax.vmap(m)(xb, tb)
        diff = jnp.where(mb > 0, pred - yb, 0.0)
        return jnp.sum(diff ** 2) / jnp.maximum(jnp.sum(mb), 1.0)

    @eqx.filter_jit
    def step(m, st, xb, tb, yb, mb):
        l, g = eqx.filter_value_and_grad(loss)(m, xb, tb, yb, mb)
        u, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, u), st, l

    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < bs:
                b = np.concatenate([b, perm[:bs - len(b)]])
            m, st, _ = step(m, st, Xt[b], Tt[b], Yt[b], Mt[b])
    return m


def exp3_residual(d, tr, va, pf):
    print("\n=== EXP 3: residual-head ceiling (term a), perfect baseline ===")
    mu, sm, sc, cm = pf["mu_marg"], pf["sig_marg"], pf["sig_cosmo"], pf["cell_mean"]
    Ttr = _resid_target(d, tr, cm, sc); Tva = _resid_target(d, va, cm, sc)
    mtr, mva = np.isfinite(Ttr), np.isfinite(Tva)
    Xva = jnp.asarray(d["x"][va]); tva = jnp.asarray(d["tau0"][va])
    Xt = jnp.asarray(d["x"][tr]); tt = jnp.asarray(d["tau0"][tr])

    res = {}
    sweep = [("nb12_e150", 12, 150, 1e-3), ("nb24_e250", 24, 250, 1e-3)]
    for tag, nb, ep, lr in sweep:
        t0 = time.time()
        m = _train_resid(d, tr, Ttr, mtr, nb, ep, lr)
        pv = np.asarray(jax.vmap(m)(Xva, tva))
        pt = np.asarray(jax.vmap(m)(Xt, tt))
        a_val = [float(np.sqrt(np.mean((pv[:, ci, :] - Tva[:, ci, :])[mva[:, ci, :]] ** 2)))
                 for ci in range(4)]
        a_tr = [float(np.sqrt(np.mean((pt[:, ci, :] - Ttr[:, ci, :])[mtr[:, ci, :]] ** 2)))
                for ci in range(4)]
        res[tag] = {"a_val_sc": a_val, "a_train_sc": a_tr,
                    "cfg": dict(nb=nb, epochs=ep)}
        print(f"  {tag:12}: (a)_val(sc) " + " ".join(f"{x:.3f}" for x in a_val)
              + " | (a)_train " + " ".join(f"{x:.3f}" for x in a_tr)
              + f"  [{time.time()-t0:.0f}s]")
    print("  -- train-val gap => finite-60-sim irreducible (a); val>train means")
    print("     the residual head extrapolates to held-out cosmologies imperfectly.")

    best = res[sweep[-1][0]]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(CLS))
    ax.bar(xs - 0.2, best["a_train_sc"], 0.4, label="(a) train")
    ax.bar(xs + 0.2, best["a_val_sc"], 0.4, label="(a) val (deployed)")
    ax.axhline(0.20, color="r", ls="--", lw=0.9, label="current (a)≈0.20")
    ax.axhline(0.08, color="g", ls=":", lw=0.9, label="(a)=0.08 sub-% clean target")
    ax.set_xticks(xs); ax.set_xticklabels(CLS)
    ax.set_ylabel("residual-head fit (a) [σ_cosmo]")
    ax.set_title("Exp 3: residual-head ceiling (perfect baseline)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/feas_exp3_residual_floor.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}/feas_exp3_residual_floor.png")
    return res


# --------------------------------------------------------------------------- #
# EXPERIMENT 4 — combined deployed accuracy with BOTH fixes                     #
#   and EXPERIMENT 5 — within-data-k-range check                                #
# --------------------------------------------------------------------------- #
def exp4_5_combined(d, tr, va, pf, sc_over_sm, baseline_cfg, resid_cfg):
    print("\n=== EXP 4: combined deployed |P̂/P−1| with BOTH fixes ===")
    mu, sm, sc, cm = pf["mu_marg"], pf["sig_marg"], pf["sig_cosmo"], pf["cell_mean"]
    logP = safe_log(d["P_filt"])

    # train the deep baseline (fixed loss) on the train cells
    cz, ct, tb, bmask, _ = _cell_table(d, tr, cm, mu, sm)
    nb_b, w_b, dp_b, ep_b = baseline_cfg
    head_b = _train_baseline(cz, ct, tb, bmask, nb_b, w_b, dp_b, ep_b, 2e-3)

    # train the residual head hard
    Ttr = _resid_target(d, tr, cm, sc)
    mtr = np.isfinite(Ttr)
    nb_r, ep_r = resid_cfg
    m_r = _train_resid(d, tr, Ttr, mtr, nb_r, ep_r, 1e-3)

    # deployed reconstruction on val rows
    z_unit_va = jnp.asarray(d["x"][va, 9]); tau_va = jnp.asarray(d["tau0"][va])
    Xva = jnp.asarray(d["x"][va])
    m_hat = np.asarray(jax.vmap(head_b)(z_unit_va, tau_va)) * sm + mu      # (n,4,K) logP
    r_hat = np.asarray(jax.vmap(m_r)(Xva, tau_va))                         # (n,4,K)
    logP_hat = m_hat + sc * r_hat
    logP_true = logP[va]
    mask_va = np.isfinite(logP_true)
    kf = d["kfkms"][0]

    def frac_stats(mask):
        frac = np.exp(logP_hat - logP_true) - 1.0
        out = {}
        for ci, nm in enumerate(CLS):
            mm = mask[:, ci, :]
            f = frac[:, ci, :][mm]
            out[nm] = dict(median=float(np.median(np.abs(f))),
                           rms=float(np.sqrt(np.mean(f ** 2))),
                           p95=float(np.percentile(np.abs(f), 95)))
        return out, frac

    full_stats, frac = frac_stats(mask_va)
    print("  full sim k-range (LF 4e-4..0.069 s/km), deployed |P̂/P−1|:")
    for nm in CLS:
        s = full_stats[nm]
        print(f"     {nm:7}: median={s['median']*100:.3f}%  rms={s['rms']*100:.3f}%  p95={s['p95']*100:.3f}%")

    # EXP 5 — survey k-windows
    print("\n=== EXP 5: within-data-k-range (DESI / KODIAQ) ===")
    win_stats = {}
    for wname, (klo, khi) in (("DESI", DESI_K), ("KODIAQ", KODIAQ_K)):
        kmask = (kf >= klo) & (kf <= khi)
        m = mask_va & kmask[None, None, :]
        st, _ = frac_stats(m)
        win_stats[wname] = st
        print(f"  {wname} ({klo:.0e}..{khi:.0e} s/km, {int(kmask.sum())} k-bins):")
        for nm in CLS:
            s = st[nm]
            print(f"     {nm:7}: median={s['median']*100:.3f}%  rms={s['rms']*100:.3f}%  p95={s['p95']*100:.3f}%")

    # figure: deployed frac error vs k per class + window stats
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    frac_abs = np.abs(frac)
    for ci, nm in enumerate(CLS):
        with np.errstate(invalid="ignore"):
            prof = np.nanmedian(np.where(mask_va[:, ci, :], frac_abs[:, ci, :], np.nan), 0)
        ax[0].plot(kf, prof * 100, lw=1.3, label=nm)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[0].axvspan(DESI_K[0], DESI_K[1], color="C0", alpha=0.08)
    ax[0].axvspan(KODIAQ_K[0], KODIAQ_K[1], color="C1", alpha=0.06)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("median |P̂/P−1| [%]")
    ax[0].set_title("Exp 4/5: deployed error vs k (both fixes)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    xs = np.arange(len(CLS)); bw = 0.25
    labels = [("full", full_stats), ("DESI", win_stats["DESI"]), ("KODIAQ", win_stats["KODIAQ"])]
    for j, (lab, st) in enumerate(labels):
        ax[1].bar(xs + (j - 1) * bw, [st[nm]["rms"] * 100 for nm in CLS], bw, label=lab)
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="1%")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(CLS)
    ax[1].set_ylabel("deployed frac-RMS |P̂/P−1| [%]")
    ax[1].set_title("Exp 4/5: deployed RMS per class by k-window")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/feas_exp4_5_deployed.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}/feas_exp4_5_deployed.png")
    return {"full": full_stats, "windows": win_stats}


def main():
    t0 = time.time()
    d = load_cache(CACHE)
    tr, va, ho = make_splits(d, 0, 8)
    pf = fit_baseline_residual_norm(d, tr)
    sc_over_sm = pf["sig_cosmo"].mean(1) / pf["sig_marg"].mean(1)   # per class (4,)
    print(f"fold-0: train={len(tr)} val={len(va)} holdout={len(ho)} | "
          f"σ_cosmo/σ_marg per class = " + " ".join(f"{x:.4f}" for x in sc_over_sm))

    summary = {"fold": 0, "n_train": len(tr), "n_val": len(va),
               "sc_over_sm_perclass": {CLS[i]: float(sc_over_sm[i]) for i in range(4)}}
    summary["exp1_svd_rank"] = exp1_svd_rank(d, tr, va)
    summary["exp2_baseline"] = exp2_baseline(d, tr, va, pf, sc_over_sm)
    summary["exp3_residual"] = exp3_residual(d, tr, va, pf)
    # use the best baseline/residual configs from the sweeps for the combined run
    summary["exp4_5"] = exp4_5_combined(
        d, tr, va, pf, sc_over_sm,
        baseline_cfg=(48, 256, 3, 15000), resid_cfg=(24, 250))

    with open(f"{OUT}/feasibility_subpercent.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}/feasibility_subpercent.json  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
