"""Phase-C — PHYSICS validation of the emulator ∂lnP1D/∂θ gradients (PI questions).

Companion to scripts/diag_grad_fidelity.py (which proved the gradients are NUMERICALLY
correct: autodiff==finite-diff to ~1e-7). This asks whether they are PHYSICALLY sane:

  Q1  k-bin jumping: real or artifact? Decompose ∂lnP/∂θ into a smooth trend + a
      high-frequency residual; quantify the roughness per param AND the CROSS-PARAM
      correlation of the residual (a shared low-rank SVD basis would make every
      param's wiggles line up at the same k -> basis ringing, not physics).
  Q2  ∂lnP/∂Ap (expect ~flat amplitude) and ∂lnP/∂ns (expect a TILT crossing zero
      near a pivot k); report the ns zero-crossing.
  Q3  ∂lnP/∂bhfeedback roughness (why the big cross-k oscillation?).
  Q4  ∂lnP/∂herei vs z (HeII reionization ~z3): is it small + z-localized? And
      ∂lnP/∂omegamh2, ∂lnP/∂hub shapes.
  Q5  emulator Fisher CORRELATION vs PRIYA's published posterior covmat correlation.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_grad_physics.py
Writes figures/analysis/05_likelihood/grad_physics.png + grad_physics.json.
"""
from __future__ import annotations

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
from hcd_analysis.emulator.predict import predict_P_tier_p

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
OUTDIR = ROOT / "figures/analysis/05_likelihood"
CKPT = str(ROOT / "checkpoints/decomp_nb24_fold0")
PARAMS = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
          "hireionz", "bhfeedback")
PRIYA_COVMAT = ("/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/chains/simdat/"
                "s-simdat-ind15-48-z2.6-4.2-loo-nodatacorr-noemuerror-optimiseGP-"
                "discardkbins-0.005-0.064/simdat-48-z2.6-4.2.covmat")


def cov2corr(C):
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def smooth(y, w=5):
    """centred moving average (edge-padded), the 'smooth physical trend'."""
    k = np.ones(w) / w
    return np.convolve(np.pad(y, w // 2, mode="edge"), k, mode="valid")[:len(y)]


def dlnP_dtheta(model, theta9, z_unit, tau0, w_c, pf):
    """∂ ln P_tier_p / ∂θ_unit, shape (K, 9)."""
    def flog(t):
        return jnp.log(predict_P_tier_p(model, t, z_unit, tau0, w_c, pf))
    return np.asarray(jax.jacfwd(flog)(jnp.asarray(theta9)))


def rep_row(d, z_fid):
    z = d["z_grid"]
    zsel = np.isclose(z, z_fid, atol=0.05)
    if not zsel.any():
        zsel = np.isclose(z, z[np.argmin(np.abs(z - z_fid))], atol=1e-6)
    rows = np.where(zsel)[0]
    tau0_z = d["tau0"][rows]
    r = rows[np.argmin(np.abs(tau0_z - np.median(tau0_z)))]
    return r, float(d["x"][r, 9]), float(d["tau0"][r])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = load_cache(CACHE)
    model, meta, norm = T.load_checkpoint(CKPT)
    pf = norm["P_filt"]
    theta0 = jnp.full(9, 0.5)

    r0, z_unit, tau0_mid = rep_row(d, 3.0)
    w_c = jnp.asarray(np.asarray(d["w_c_cache"][r0]).reshape(4))
    kf = np.asarray(d["kfkms"][r0]); g = np.isfinite(kf)

    J = dlnP_dtheta(model, theta0, z_unit, tau0_mid, w_c, pf)        # (K,9) ∂lnP/∂θ
    Jg = J[g]; kfg = kf[g]

    # response MAGNITUDE per param (RMS over k) — answers "why is herei/X small?"
    mag = np.sqrt(np.mean(Jg ** 2, 0))
    print("response magnitude RMS_k(∂lnP/∂θ) per param (z=3):")
    for p, v in zip(PARAMS, mag):
        print(f"    {p:11s} {v:.4f}")

    # --- Q1: roughness + cross-param residual correlation ------------------------
    Jsm = np.stack([smooth(Jg[:, i], 5) for i in range(9)], axis=1)
    resid = Jg - Jsm
    rough = np.sqrt(np.mean(resid ** 2, 0)) / (np.sqrt(np.mean(Jsm ** 2, 0)) + 1e-30)
    # cross-param correlation of the high-freq residual: high => shared-basis ringing
    rr = resid / (np.std(resid, 0, keepdims=True) + 1e-30)
    resid_corr = (rr.T @ rr) / rr.shape[0]
    offdiag = resid_corr[~np.eye(9, dtype=bool)]
    print("Q1 roughness (high-freq/smooth) per param:")
    for p, v in zip(PARAMS, rough):
        print(f"    {p:11s} {v:.3f}")
    print(f"Q1 cross-param residual |corr| median={np.median(np.abs(offdiag)):.2f} "
          f"(high => shared SVD-basis ringing, not per-param physics)")

    # --- Q2: ns pivot / Ap flatness ----------------------------------------------
    iAp, ins = 1, 0
    dAp, dns = Jg[:, iAp], Jg[:, ins]
    Ap_flatness = float(np.std(dAp) / (np.abs(np.mean(dAp)) + 1e-30))
    sgn = np.sign(smooth(dns, 7))
    cross = np.where(np.diff(sgn) != 0)[0]
    ns_cross_k = float(kfg[cross[0]]) if len(cross) else float("nan")
    print(f"Q2 ∂lnP/∂Ap: mean={np.mean(dAp):+.3f} flatness(std/|mean|)={Ap_flatness:.2f} "
          f"(expect ~flat>0); ∂lnP/∂ns zero-crossing k={ns_cross_k:.4f} s/km")

    # --- Q4: herei z-dependence + omegamh2/hub -----------------------------------
    herei_by_z = {}
    for zf in (2.4, 3.0, 3.6, 4.2):
        rz, zuz, t0z = rep_row(d, zf)
        wz = jnp.asarray(np.asarray(d["w_c_cache"][rz]).reshape(4))
        kfz = np.asarray(d["kfkms"][rz]); gz = np.isfinite(kfz)
        Jz = dlnP_dtheta(model, theta0, zuz, t0z, wz, pf)
        herei_by_z[zf] = (kfz[gz], Jz[gz, 2], float(np.sqrt(np.mean(Jz[gz, 2] ** 2))))
    print("Q4 herei RMS(∂lnP/∂herei) by z: " +
          "  ".join(f"z{zf}={v[2]:.3f}" for zf, v in herei_by_z.items()))

    # --- Q5: emulator Fisher corr vs PRIYA covmat corr ---------------------------
    sigfrac = 0.05
    F = (Jg.T / sigfrac**2) @ Jg
    F += 1e-10 * np.trace(F) / 9 * np.eye(9)
    emu_corr = cov2corr(np.linalg.inv(F))

    priya_corr = None
    if Path(PRIYA_COVMAT).exists():
        names = open(PRIYA_COVMAT).readline().lstrip("#").split()
        M = np.loadtxt(PRIYA_COVMAT)
        idx = [names.index(p) for p in PARAMS]      # our 9 in PRIYA's covmat
        priya_corr = cov2corr(M[np.ix_(idx, idx)])
        # sign agreement of off-diagonal correlations
        eo, po = emu_corr[~np.eye(9, dtype=bool)], priya_corr[~np.eye(9, dtype=bool)]
        sign_agree = float(np.mean(np.sign(eo) == np.sign(po)))
        print(f"Q5 emulator-Fisher vs PRIYA covmat off-diag sign agreement: "
              f"{sign_agree*100:.0f}%")

    # --- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    # Q1/Q2: Ap + ns with smooth overlay
    a = ax[0, 0]
    a.semilogx(kfg, dAp, lw=1, alpha=0.5, color="tab:blue")
    a.semilogx(kfg, smooth(dAp, 7), lw=2.2, color="tab:blue", label="∂lnP/∂Ap (smooth)")
    a.semilogx(kfg, dns, lw=1, alpha=0.5, color="tab:red")
    a.semilogx(kfg, smooth(dns, 7), lw=2.2, color="tab:red", label="∂lnP/∂ns (smooth)")
    if np.isfinite(ns_cross_k):
        a.axvline(ns_cross_k, color="tab:red", ls=":", lw=1.2,
                  label=f"ns pivot ~{ns_cross_k:.3f}")
    a.axhline(0, color="k", lw=0.6); a.set_title("Q2: Ap (flat?) & ns (tilt/pivot?)")
    a.set_xlabel("k [s/km]"); a.set_ylabel("∂lnP_tier_p/∂θ"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # Q1: roughness bars
    a = ax[0, 1]
    a.bar(range(9), rough, color="tab:purple"); a.set_xticks(range(9))
    a.set_xticklabels(PARAMS, rotation=90, fontsize=8)
    a.set_ylabel("roughness = RMS(high-freq)/RMS(smooth)")
    a.set_title(f"Q1/Q3: gradient roughness per param\n(cross-param |corr| median "
                f"{np.median(np.abs(offdiag)):.2f} => shared-basis)")
    a.grid(alpha=0.3, axis="y")
    # Q1: residual cross-correlation heatmap
    a = ax[0, 2]
    im = a.imshow(resid_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    a.set_xticks(range(9)); a.set_xticklabels(PARAMS, rotation=90, fontsize=7)
    a.set_yticks(range(9)); a.set_yticklabels(PARAMS, fontsize=7)
    a.set_title("Q1: high-freq residual cross-param corr\n(red everywhere => shared SVD ringing)")
    fig.colorbar(im, ax=a, fraction=0.046)
    # Q4: herei vs z
    a = ax[1, 0]
    for zf, (kk, jj, rms) in herei_by_z.items():
        a.semilogx(kk, smooth(jj, 7), lw=1.8, label=f"z={zf} (RMS {rms:.2f})")
    a.axhline(0, color="k", lw=0.6); a.set_title("Q4: ∂lnP/∂herei vs z (HeII reion ~z3)")
    a.set_xlabel("k [s/km]"); a.set_ylabel("∂lnP/∂herei"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # Q4: omegamh2 + hub
    a = ax[1, 1]
    for i, nm, c in [(6, "omegamh2", "tab:green"), (5, "hub", "tab:orange")]:
        a.semilogx(kfg, smooth(Jg[:, i], 7), lw=2, color=c, label=f"∂lnP/∂{nm}")
    a.axhline(0, color="k", lw=0.6); a.set_title("Q4: omegamh2 & hub response")
    a.set_xlabel("k [s/km]"); a.set_ylabel("∂lnP/∂θ"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # Q5: emulator vs PRIYA correlation (lower/upper triangle)
    a = ax[1, 2]
    if priya_corr is not None:
        combo = np.tril(emu_corr, -1) + np.triu(priya_corr, 1) + np.eye(9)
        im = a.imshow(combo, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_title("Q5: corr — lower=emulator Fisher, upper=PRIYA covmat")
    else:
        im = a.imshow(emu_corr, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_title("Q5: emulator Fisher correlation (PRIYA covmat not found)")
    a.set_xticks(range(9)); a.set_xticklabels(PARAMS, rotation=90, fontsize=7)
    a.set_yticks(range(9)); a.set_yticklabels(PARAMS, fontsize=7)
    fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Emulator ∂lnP1D/∂θ — physics validation (z=3 fiducial)", fontsize=13)
    fig.tight_layout()
    fpath = OUTDIR / "grad_physics.png"
    fig.savefig(fpath, dpi=150); plt.close(fig)
    print(f"wrote {fpath}")

    out = dict(
        roughness={p: float(v) for p, v in zip(PARAMS, rough)},
        cross_param_resid_corr_median=float(np.median(np.abs(offdiag))),
        Ap_mean=float(np.mean(dAp)), Ap_flatness=Ap_flatness, ns_cross_k=ns_cross_k,
        herei_rms_by_z={str(zf): v[2] for zf, v in herei_by_z.items()},
        emu_corr=emu_corr.tolist(),
        priya_corr=(priya_corr.tolist() if priya_corr is not None else None),
        params=list(PARAMS),
    )
    json.dump(out, open(OUTDIR / "grad_physics.json", "w"), indent=2)
    print(f"wrote {OUTDIR / 'grad_physics.json'}")


if __name__ == "__main__":
    main()
