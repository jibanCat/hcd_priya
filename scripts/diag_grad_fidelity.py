"""Phase-C Task 1 — ∂P/∂θ gradient-correctness GATE on the DEPLOYED emulator.

The HARD pre-gate to the closure/SBC test: HMC/NUTS needs the assembled forward
P_obs(θ,τ₀,α) (hcd_analysis.emulator.predict) to be smooth and its JAX autodiff
Jacobian to be CORRECT end-to-end (encoder -> SVD-basis HeadB -> exp reconstruction
-> structural Tier-P -> HCD add-back). This loads a trained LOSO fold and checks
jacfwd vs central finite differences at a fiducial point AND across the τ₀ ladder
(incl. both edges, where the τ₀×cosmology interaction is hardest), plus a finiteness
sweep over the unit cube.

GATE: median rel-err < 1e-4 at every τ₀ position; zero non-finite P_obs/gradient.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_grad_fidelity.py [--ckpt ...]
Writes figures/analysis/05_likelihood/grad_fidelity.png + grad_fidelity.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.predict import predict_P_obs

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
OUTDIR = ROOT / "figures/analysis/05_likelihood"
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
GATE_MEDIAN = 1e-4
GATE_MAX = 1e-2


def central_fd_jac(f, x, h=1e-3):
    """Central finite-diff Jacobian of f: R^n -> R^m, returns (m, n)."""
    x = np.asarray(x, float)
    cols = []
    for i in range(x.size):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        cols.append((np.asarray(f(jnp.asarray(xp))) - np.asarray(f(jnp.asarray(xm)))) / (2 * h))
    return np.stack(cols, axis=-1)  # (m, n)


def rel_err(J_ad, J_fd):
    """Per-element gradient error relative to the PEAK gradient magnitude.

    The denominator is floored at 1e-3·max|J_fd| so that derivative ZERO-CROSSINGS
    (|J_fd|→0, a pure finite-diff artifact) don't dominate the metric — the standard
    choice for an autodiff-vs-FD check. Where the gradient is sizeable this reduces to
    the ordinary relative error.
    """
    denom = np.abs(J_fd) + 1e-3 * np.max(np.abs(J_fd))
    return np.abs(J_ad - J_fd) / denom


def cov2corr(C):
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def fig_perparam_overlay(kf, J_ad, J_fd, per_param_med, path):
    """One panel per parameter: autodiff (line) vs central-FD (×) of ∂P_obs/∂θ_i vs k.
    The visual proof that EVERY parameter's response is captured, not just the median."""
    g = np.isfinite(kf)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(kf[g], J_ad[g, i], lw=1.8, color="tab:blue", label="autodiff")
        ax.plot(kf[g], J_fd[g, i], "x", ms=4, color="tab:red", label="finite-diff")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"∂P_obs/∂{PARAMS[i]}   (median rel-err {per_param_med[i]:.1e})",
                     fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
        if i >= 6:
            ax.set_xlabel("k [s/km]")
    fig.suptitle("Per-parameter ∂P_obs/∂θ — emulator autodiff vs finite-diff (mid-ladder τ₀)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def fig_convergence_fisher_tau0(hs, med_rel_h, rel_all, F_corr, kf, J_ad_pos, path):
    """2×2 reading panel:
       (a) FD step-size convergence (median rel-err vs h) — the V: truncation ∝h²
           decreasing then roundoff rising; the autodiff sits at the floor (rigorous proof).
       (b) per-element rel-err histogram (bulk near machine eps).
       (c) emulator-implied Fisher CORRELATION (9×9; DESI-like diagonal cov, ILLUSTRATIVE)
           — the parameter degeneracy structure the gradients imply.
       (d) τ₀-dependence of ∂logP/∂A_p across the ladder (the τ₀×cosmology interaction
           the τ₀-aware C_emu is built to carry)."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    # (a) convergence V
    ax[0, 0].loglog(hs, med_rel_h, "o-", color="tab:blue")
    ax[0, 0].set_xlabel("finite-diff step h (unit-cube)")
    ax[0, 0].set_ylabel("median rel-err (autodiff vs FD)")
    ax[0, 0].set_title("(a) FD step-size convergence — autodiff at the floor")
    ax[0, 0].grid(alpha=0.3, which="both")
    # (b) histogram
    rl = rel_all[np.isfinite(rel_all) & (rel_all > 0)]
    ax[0, 1].hist(np.log10(rl), bins=60, color="tab:green", alpha=0.8)
    ax[0, 1].axvline(np.log10(np.median(rl)), color="k", ls="--",
                     label=f"median {np.median(rl):.1e}")
    ax[0, 1].set_xlabel("log10 per-element rel-err"); ax[0, 1].set_ylabel("count")
    ax[0, 1].set_title("(b) gradient-error distribution"); ax[0, 1].legend(fontsize=8)
    # (c) Fisher correlation
    im = ax[1, 0].imshow(F_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[1, 0].set_xticks(range(9)); ax[1, 0].set_xticklabels(PARAMS, rotation=90, fontsize=8)
    ax[1, 0].set_yticks(range(9)); ax[1, 0].set_yticklabels(PARAMS, fontsize=8)
    ax[1, 0].set_title("(c) emulator-implied Fisher correlation\n(DESI-like diag cov — ILLUSTRATIVE)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)
    # (d) τ₀-dependence of ∂logP/∂A_p (A_p column, summed |∂P| over classes already in P_obs)
    g = np.isfinite(kf)
    iAp = 1
    for name, J in J_ad_pos.items():
        ax[1, 1].semilogx(kf[g], J[g, iAp], lw=1.6, label=f"τ₀ {name}")
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set_xlabel("k [s/km]"); ax[1, 1].set_ylabel("∂P_obs/∂A_p")
    ax[1, 1].set_title("(d) τ₀-dependence of the A_p response\n(the interaction C_emu must carry)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
    fig.suptitle("Gradient diagnostics — convergence, error distribution, Fisher, τ₀-dependence",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/decomp_nb24_fold0"))
    ap.add_argument("--z-fid", type=float, default=3.0)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--n-sweep", type=int, default=80)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    d = load_cache(CACHE)
    model, meta, norm = T.load_checkpoint(args.ckpt)
    pf = norm["P_filt"]
    print(f"[gate] ckpt={args.ckpt} n_basis={meta['arch_cfg']['n_basis']}")

    # --- fiducial at z_fid: a representative row for w_c, Δ_c templates, z_unit ----
    z_grid = d["z_grid"]
    zsel = np.isclose(z_grid, args.z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z_grid, z_grid[np.argmin(np.abs(z_grid - args.z_fid))], atol=1e-6)
    rows = np.where(zsel)[0]
    tau0_z = d["tau0"][rows]
    r_mid = rows[np.argmin(np.abs(tau0_z - np.median(tau0_z)))]
    z_unit = float(d["x"][r_mid, 9])
    w_c = jnp.asarray(np.asarray(d["w_c_cache"][r_mid]).reshape(4))
    delta = jnp.asarray(d["delta"][r_mid])             # (3,172)
    alpha = jnp.asarray(np.full(3, 1.0))               # prior-centre amplitudes
    kf = np.asarray(d["kfkms"][r_mid])
    theta0 = jnp.full(9, 0.5)                          # cube-centre fiducial

    tau0_lo, tau0_mid, tau0_hi = (float(tau0_z.min()),
                                  float(np.median(tau0_z)),
                                  float(tau0_z.max()))
    positions = {"ladder_lo": tau0_lo, "mid": tau0_mid, "ladder_hi": tau0_hi}

    # --- gate at each τ₀ position --------------------------------------------------
    per_pos = {}
    rel_mid = None
    J_ad_pos = {}                                     # per-τ₀-position autodiff Jacobians
    for name, tau0 in positions.items():
        def f(theta9, _tau0=tau0):
            return predict_P_obs(model, theta9, z_unit, _tau0, alpha, w_c, pf, delta)
        J_ad = np.asarray(jax.jacfwd(f)(theta0))       # (K,9)
        J_fd = central_fd_jac(f, theta0, h=args.h)     # (K,9)
        r = rel_err(J_ad, J_fd)
        # document that the worst element is a derivative zero-crossing (|J_fd|->0):
        scale = np.max(np.abs(J_fd))
        iworst = np.unravel_index(np.argmax(r), r.shape)
        per_pos[name] = dict(tau0=tau0,
                             median=float(np.median(r)),
                             p99=float(np.percentile(r, 99)),
                             max=float(np.max(r)),
                             worst_Jfd_frac_of_peak=float(np.abs(J_fd[iworst]) / scale),
                             n_nonfinite=int((~np.isfinite(J_ad)).sum()))
        J_ad_pos[name] = J_ad
        if name == "mid":
            rel_mid, J_ad_mid, J_fd_mid = r, J_ad, J_fd
        print(f"  [{name:9s} τ₀={tau0:.4f}] median={per_pos[name]['median']:.2e} "
              f"p99={per_pos[name]['p99']:.2e} max={per_pos[name]['max']:.2e} "
              f"(worst @ |J_fd|={per_pos[name]['worst_Jfd_frac_of_peak']:.1e}×peak) "
              f"nonfinite={per_pos[name]['n_nonfinite']}")

    # --- finiteness sweep over the unit cube + τ₀ ---------------------------------
    rng = np.random.default_rng(0)
    grad_norm = jax.jit(lambda th, zu, t: jnp.linalg.norm(
        jax.jacfwd(lambda x: predict_P_obs(model, x, zu, t, alpha, w_c, pf, delta))(th)))
    n_nf_val = n_nf_grad = 0
    for _ in range(args.n_sweep):
        th = jnp.asarray(rng.uniform(0, 1, 9))
        zu = float(rng.uniform(0, 1)); t = float(rng.uniform(tau0_lo, tau0_hi))
        val = predict_P_obs(model, th, zu, t, alpha, w_c, pf, delta)
        n_nf_val += int((~np.isfinite(np.asarray(val))).any())
        n_nf_grad += int(not np.isfinite(float(grad_norm(th, zu, t))))
    print(f"  [sweep {args.n_sweep}] nonfinite P_obs={n_nf_val} grad={n_nf_grad}")

    # --- verdict -------------------------------------------------------------------
    worst_median = max(p["median"] for p in per_pos.values())
    worst_max = max(p["max"] for p in per_pos.values())
    total_nf = sum(p["n_nonfinite"] for p in per_pos.values()) + n_nf_val + n_nf_grad
    passed = (worst_median < GATE_MEDIAN and worst_max < GATE_MAX and total_nf == 0)
    print(f"\n  GATE: worst-median={worst_median:.2e} (<{GATE_MEDIAN}) "
          f"worst-max={worst_max:.2e} (<{GATE_MAX}) nonfinite={total_nf}  "
          f"-> {'PASS' if passed else 'FAIL'}")

    # --- figure --------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].imshow(np.log10(rel_mid.T + 1e-16), aspect="auto", cmap="viridis",
                      extent=[kf.min(), kf.max(), 8.5, -0.5])
    ax[0].set_yticks(range(9)); ax[0].set_yticklabels(PARAMS, fontsize=8)
    ax[0].set_xlabel("k [s/km]"); ax[0].set_title("log10 |J_ad−J_fd|/|J_fd|  (mid-ladder τ₀)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    names = list(per_pos); meds = [per_pos[n]["median"] for n in names]
    maxs = [per_pos[n]["max"] for n in names]
    ax[1].semilogy(names, meds, "o-", label="median"); ax[1].semilogy(names, maxs, "s--", label="max")
    ax[1].axhline(GATE_MEDIAN, color="r", ls=":", label=f"gate {GATE_MEDIAN}")
    ax[1].set_ylabel("rel-err"); ax[1].set_title("autodiff-vs-FD across the τ₀ ladder")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    iAp = 1
    g = np.isfinite(kf)
    ax[2].semilogx(kf[g], J_ad_mid[g, iAp], lw=2, label="autodiff ∂P_obs/∂A_p")
    ax[2].semilogx(kf[g], J_fd_mid[g, iAp], "x", ms=4, label="finite-diff")
    ax[2].set_xlabel("k [s/km]"); ax[2].set_ylabel("∂P_obs/∂A_p")
    ax[2].set_title("autodiff vs FD (A_p column, mid τ₀)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle(f"∂P/∂θ gradient-correctness gate — {'PASS' if passed else 'FAIL'} "
                 f"(z={args.z_fid}, ckpt={Path(args.ckpt).name})", fontsize=11)
    fig.tight_layout()
    fpath = OUTDIR / "grad_fidelity.png"
    fig.savefig(fpath, dpi=150); plt.close(fig)
    print(f"  wrote {fpath}")

    # ---- extra diagnostic READING plots -----------------------------------------
    per_param_med = np.median(rel_mid, axis=0)        # (9,) median rel-err per param
    print("  per-param median rel-err: " +
          "  ".join(f"{p}={m:.1e}" for p, m in zip(PARAMS, per_param_med)))

    # (1) per-parameter autodiff-vs-FD overlay (all 9)
    fig_perparam_overlay(kf, J_ad_mid, J_fd_mid, per_param_med,
                         OUTDIR / "grad_fidelity_perparam.png")

    # (2) FD step-size convergence V: median rel-err vs h (autodiff fixed = J_ad_mid)
    def f_mid(theta9):
        return predict_P_obs(model, theta9, z_unit, positions["mid"], alpha, w_c, pf, delta)
    hs = np.logspace(-6, -1, 11)
    med_rel_h = [float(np.median(rel_err(J_ad_mid, central_fd_jac(f_mid, theta0, h=h))))
                 for h in hs]
    print("  FD-convergence (h, median rel-err): " +
          "  ".join(f"{h:.0e}:{m:.1e}" for h, m in zip(hs, med_rel_h)))

    # (3) emulator-implied Fisher correlation (DESI-like 5% diagonal cov — ILLUSTRATIVE,
    #     the real C_emu+C_cosmic lands in Task 3; this shows the degeneracy structure)
    P_obs_mid = np.asarray(f_mid(theta0))             # (K,)
    kok = (np.isfinite(kf) & np.isfinite(P_obs_mid) & (P_obs_mid != 0.0)
           & np.isfinite(J_ad_mid).all(axis=1))
    J9 = J_ad_mid[kok]                                 # (M,9)
    Cinv = 1.0 / (0.05 * np.abs(P_obs_mid[kok])) ** 2  # DESI-like 5% per-mode
    F = (J9.T * Cinv) @ J9
    F += 1e-12 * np.trace(F) / 9 * np.eye(9)
    F_corr = cov2corr(np.linalg.inv(F))
    fig_convergence_fisher_tau0(hs, med_rel_h, rel_mid, F_corr, kf, J_ad_pos,
                                OUTDIR / "grad_fidelity_convergence_fisher.png")

    out = dict(ckpt=args.ckpt, z_fid=args.z_fid, h=args.h, z_unit=z_unit,
               gate_median=GATE_MEDIAN, gate_max=GATE_MAX,
               per_tau0_position=per_pos,
               per_param_median_rel_err={PARAMS[i]: float(per_param_med[i]) for i in range(9)},
               fd_convergence={f"{h:.0e}": m for h, m in zip(hs, med_rel_h)},
               fisher_corr_params=list(PARAMS),
               fisher_corr=F_corr.tolist(),
               sweep=dict(n=args.n_sweep, nonfinite_Pobs=n_nf_val, nonfinite_grad=n_nf_grad),
               passed=bool(passed))
    jpath = OUTDIR / "grad_fidelity.json"
    json.dump(out, open(jpath, "w"), indent=2)
    print(f"  wrote {jpath}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
