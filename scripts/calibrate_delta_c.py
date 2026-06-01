"""Calibrate the diagonal-Poisson w_c correction delta_c(z) from the production cache.

The diagonal telescoping-Poisson map M0 (hcd_analysis/emulator/dndx_wc.py) predicts
the per-sightline class fractions w_c from the per-class incidence dN/dX, assuming the
N_HI classes are independent Poisson processes. They are not (real cross-class
clustering), so M0 is biased by <=2.5%, growing toward low z. delta_c(z) is the small
empirical correction:  w_c_true = w_c_M0 * (1 + delta_c(z)), renormalised to sum 1.

This calibrates delta_c(z) DIRECTLY from the production cache -- no fake_spectra, no
merge needed -- because the cache already stores both sides:
  - counted w_c  : tier_c_counts -> coarse_counts -> w_counted = counts / N_sl
  - M0 inputs    : snap_dNdX_{LLS,subDLA,DLA} + snap_total_path_dX (Xbar = path / N_sl)

We aggregate per snap-block over ALL LF shards (every sim, every z), fit a deg-2
polynomial delta_c(z) per class (the mean z-trend), and report the per-class residual
STD as the w_c prior width -- the residual is the irreducible cosmology dependence at
fixed z (a z-only polynomial cannot remove it; carried as a prior width per the
2026-05-29 w_c<->dN/dX coupling note).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/calibrate_delta_c.py [--shards GLOB] [--deg 2]
Outputs: docs/superpowers/2026-06-01-delta_c-coeffs.md  and  .../delta_c-coeffs.npz
"""
from __future__ import annotations
import argparse
import glob
import numpy as np
import jax.numpy as jnp

from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.dndx_wc import w_c_from_mu, mu_from_dndx
from hcd_analysis.emulator import CLASS_NAMES

_DEFAULT_GLOB = "/scratch/cavestru_root/cavestru0/mfho/tau0_shards/observables_tau0_lf.shard*.h5"


def collect(shard_glob):
    """Aggregate per snap-block (z, w_counted[4], w_M0[4], sim) over all shards."""
    z, wc, wm, sims = [], [], [], []
    shards = sorted(glob.glob(shard_glob))
    if not shards:
        raise SystemExit(f"no shards matched {shard_glob!r}")
    for sh in shards:
        d = load_cache(sh)
        g = d["snap_group_idx"]
        for gi in np.unique(g):
            r = int(np.where(g == gi)[0][0])          # counts are tau0-invariant -> one rep row
            cc = d["coarse_counts"][r].astype(float)  # (4,) clean,LLS,subDLA,DLA
            n_sl = cc.sum()
            if n_sl <= 0:
                continue
            w_counted = cc / n_sl
            dndx = d["snap_dNdX"][gi]                  # (3,) LLS,subDLA,DLA
            xbar = d["snap_total_path_dX"][gi] / n_sl
            w_m0 = np.asarray(w_c_from_mu(mu_from_dndx(jnp.asarray(dndx), jnp.asarray(xbar))))
            z.append(float(d["z_grid"][r])); wc.append(w_counted); wm.append(w_m0)
            sims.append(d["sim_name"][r])
    return (np.array(z), np.array(wc), np.array(wm), np.array(sims, dtype=object), len(shards))


def fit(z, wc, wm, deg=2):
    """Return coeffs[4, deg+1] (np.polyfit order) and per-class residual std (prior width)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(wm > 0, wc / wm - 1.0, 0.0)   # (N,4)
    coeffs = np.zeros((4, deg + 1))
    resid_std = np.zeros(4)
    resid_max = np.zeros(4)
    for ci in range(4):
        c = np.polyfit(z, delta[:, ci], deg)
        coeffs[ci] = c
        resid = delta[:, ci] - np.polyval(c, z)
        resid_std[ci] = resid.std()
        resid_max[ci] = np.abs(resid).max()
    return coeffs, delta, resid_std, resid_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=_DEFAULT_GLOB)
    ap.add_argument("--deg", type=int, default=2)
    args = ap.parse_args()

    z, wc, wm, sims, n_sh = collect(args.shards)
    coeffs, delta, resid_std, resid_max = fit(z, wc, wm, deg=args.deg)
    n_sims = len(set(map(str, sims)))

    print(f"calibrated delta_c(z) from {n_sh} shards, {len(z)} snap-blocks, {n_sims} sims, "
          f"z {z.min():.1f}-{z.max():.1f}, deg={args.deg}")
    print(f"{'class':8s} {'mean δ':>9s} {'fit-resid std':>14s} {'resid max':>11s}")
    for ci, name in enumerate(CLASS_NAMES):
        print(f"{name:8s} {delta[:,ci].mean():+9.4f} {resid_std[ci]:14.4f} {resid_max[ci]:11.4f}")

    out_npz = "docs/superpowers/2026-06-01-delta_c-coeffs.npz"
    np.savez(out_npz, coeffs=coeffs, resid_std=resid_std, resid_max=resid_max,
             class_names=np.array(CLASS_NAMES), deg=args.deg, n_blocks=len(z), n_sims=n_sims)

    # Markdown record (the frozen coeffs to paste into dndx_wc._DELTA_C_COEFFS).
    lines = [
        "# delta_c(z) calibration — frozen coefficients",
        "",
        f"Measured from {n_sh} LF shards ({len(z)} snap-blocks, {n_sims} sims, "
        f"z={z.min():.1f}-{z.max():.1f}) by `scripts/calibrate_delta_c.py`. NOT invented.",
        "",
        "`delta_c(z)` = deg-%d polynomial; w_c_corrected = w_M0*(1+delta_c) renormalised." % args.deg,
        "The fit removes the mean z-trend; the residual **std** is the irreducible",
        "cosmology dependence at fixed z, carried as the per-class w_c **prior width**.",
        "",
        "Coeffs are np.polyval order (highest power first), class order "
        f"{list(CLASS_NAMES)}:",
        "",
        "```python",
        "_DELTA_C_COEFFS = [  # (clean, LLS, subDLA, DLA) x (deg+1); np.polyval order",
    ]
    for ci, name in enumerate(CLASS_NAMES):
        lines.append(f"    {[float(x) for x in coeffs[ci]]},  # {name}")
    lines += ["]", "```", "",
              "| class | mean δ_c | fit-residual std (prior width) | residual max |",
              "|---|---|---|---|"]
    for ci, name in enumerate(CLASS_NAMES):
        lines.append(f"| {name} | {delta[:,ci].mean():+.4f} | {resid_std[ci]:.4f} | {resid_max[ci]:.4f} |")
    lines.append("")
    with open("docs/superpowers/2026-06-01-delta_c-coeffs.md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote docs/superpowers/2026-06-01-delta_c-coeffs.md and {out_npz}")


if __name__ == "__main__":
    main()
