"""Emulator dN/dX head: PREDICTED vs TRUTH, honest all-folds LOSO.

For each fold f, the held-out emulator final_fold{f} (which never saw fold f's sims) predicts the
per-class HCD incidence dN/dX on its held-out sims; truth = the cache snap_dNdX. Accumulated over
all 8 folds -> every one of the 60 PRIYA sims is predicted out-of-sample. The HCD incidence PRIOR
centers on (lit/sim)*w_c with w_c from PRIYA's sim dN/dX, so this checks the emulator faithfully
reproduces that sim incidence (top row), and shows where PRIYA's sim sits vs the literature the
per-survey pin is anchored to (bottom row).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_dndx_pred_vs_truth.py
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp

from hcd_analysis.emulator.data import load_cache, invert_norm
from hcd_analysis.emulator.train import load_checkpoint
from hcd_analysis.emulator import closure_legb as C

CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT = "figures/analysis/05_likelihood/dndx_pred_vs_truth.png"
CLS = ["LLS", "subDLA", "DLA"]; COL = {"LLS": "#2ca02c", "subDLA": "#ff7f0e", "DLA": "#d62728"}

# literature dN/dX(z) (same tables as plot_dndx_vs_literature.py)
LIT = {"LLS": ([2.4,2.8,3.35,3.47,3.58,3.74,3.97,4.23],[0.29,0.33,0.35,0.57,0.41,0.52,0.72,0.78]),
       "subDLA": ([2.27,2.73,3.25,3.77,4.20],[0.07,0.06,0.08,0.10,0.10]),
       "DLA": ([2.31,2.57,2.86,3.22,3.70,4.39],[0.048,0.055,0.067,0.084,0.075,0.106])}

d = load_cache(CACHE)
sim_name = np.asarray(d["sim_name"]); gidx = np.asarray(d["snap_group_idx"])
zrow = np.asarray(d["z_grid"]); X = np.asarray(d["x"]); truth = np.asarray(d["snap_dNdX"])  # (Ng,3)

# accumulate held-out predictions: one prediction per (group) using the fold that held its sim out
pred = np.full_like(truth, np.nan)       # (Ng,3) predicted dN/dX, out-of-sample
zg_of = np.full(truth.shape[0], np.nan)  # z per group
for f in range(8):
    model, meta, norm = load_checkpoint(f"checkpoints/final_fold{f}")
    ho = C.held_out_sims(d, fold=f)
    ho_names = set(np.asarray(ho[0]).tolist()) if isinstance(ho, tuple) else set(np.asarray(ho).tolist())
    sel = np.array([s in ho_names for s in sim_name])
    if sel.sum() == 0:
        continue
    # one representative row per held-out group (dN/dX is tau0-invariant)
    groups = {}
    for ri in np.where(sel)[0]:
        g = int(gidx[ri]); groups.setdefault(g, ri)
    g_ids = np.array(sorted(groups)); rows = np.array([groups[g] for g in g_ids])
    xr = jnp.asarray(X[rows])
    fwd = jax.vmap(lambda x: model(x, jnp.asarray(1.0))["dndx"])   # HeadA is tau0-invariant
    dn_norm = np.asarray(fwd(xr))                                  # (n,3) normalized
    dn = np.exp(invert_norm(dn_norm, norm["dndx"]))                # -> physical dN/dX
    pred[g_ids] = dn
    zg_of[g_ids] = zrow[rows]

ok = np.isfinite(pred[:, 0]) & np.isfinite(zg_of)
print(f"held-out groups predicted: {ok.sum()} / {truth.shape[0]}")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for j, c in enumerate(CLS):
    t = truth[ok, j]; p = pred[ok, j]; z = zg_of[ok]
    good = (t > 0) & (p > 0)
    t, p, z = t[good], p[good], z[good]
    # row 1: predicted vs truth scatter (log-log) + 1:1
    a = ax[0, j]
    sc = a.scatter(t, p, c=z, s=10, cmap="viridis", alpha=0.6)
    lim = [min(t.min(), p.min()) * 0.8, max(t.max(), p.max()) * 1.2]
    a.plot(lim, lim, "k--", lw=1, label="1:1")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
    fe = np.median(np.abs(p / t - 1.0)) * 100
    a.set_title(f"{c}: emulator dN/dX  (median |frac err| {fe:.1f}%)", fontsize=10)
    a.set_xlabel("truth dN/dX (PRIYA sim)"); a.set_ylabel("emulator-predicted dN/dX")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    if j == 2: fig.colorbar(sc, ax=a, label="z")
    # row 2: dN/dX(z) -- truth + predicted (held-out) + literature
    b = ax[1, j]
    # bin by z for a clean mean+/-std of truth and predicted
    zb = np.unique(np.round(z, 1))
    tm, ts, pm, zc = [], [], [], []
    for zz in zb:
        m = np.isclose(z, zz, atol=0.05)
        if m.sum() >= 2:
            zc.append(zz); tm.append(np.nanmean(t[m])); ts.append(np.nanstd(t[m])); pm.append(np.nanmean(p[m]))
    zc, tm, ts, pm = map(np.array, (zc, tm, ts, pm))
    b.fill_between(zc, tm - ts, tm + ts, color=COL[c], alpha=0.18)
    b.plot(zc, tm, "-", color=COL[c], lw=2, label="PRIYA sim (truth)")
    b.plot(zc, pm, "o--", color="k", ms=4, lw=1.2, label="emulator (held-out)")
    zl, vl = LIT[c]
    b.plot(zl, vl, "s", color="tab:red", ms=6, label="observed (literature)")
    b.set_yscale("log"); b.set_xlabel("z"); b.set_ylabel("dN/dX")
    b.set_title(f"{c}: sim vs emulator vs literature", fontsize=10)
    b.legend(fontsize=8); b.grid(alpha=0.3, which="both")

fig.suptitle("Emulator dN/dX head — PREDICTED vs TRUTH (all-folds held-out) and vs the literature pin",
             fontsize=13, y=1.0)
fig.text(0.5, -0.02, "Top: out-of-sample emulator fidelity on PRIYA's sim incidence (the prior's w_c). "
         "Bottom: PRIYA sim (band) vs emulator (held-out) vs observed literature — the gap the per-survey "
         "(lit/sim) pin corrects.", ha="center", fontsize=9, style="italic")
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
