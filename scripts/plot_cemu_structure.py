#!/usr/bin/env python3
"""Visualize the C_total decomposition on the DESI leg (Phase-5a explainer):
  C_data (survey, dense)  +  diag(C_emu cross-class, diagonal-in-k)  +  diag(old MF floor)
  +  C_shape (NEW shape-aware MF floor — the off-diagonal eps-tilt mode).

Row 1: CORRELATION matrices (cov normalized to unit diagonal) → shows STRUCTURE.
Row 2: per-row fractional sqrt(diag)/P_model → shows MAGNITUDE of each component.

Built at the ns0.972 HFLOSO config (fold6, MF + exclude_held). Writes to the NOTES repo.
"""
import os, numpy as np, jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.closure_legb import (build_legb_ctx, make_hr_truth_from_cache,
    make_legb_mock, _mock_core_per_leg)
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import multifidelity as MF

NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_multifidelity/cemu_structure_decomp.png"
INFL = 1.5
FOLD = 6


def main():
    hrn = sorted(set(s.decode() if isinstance(s, bytes) else s
                     for s in MF.load_cache(MF.HR_CACHE)["sim_name"]))
    sim972 = [s for s in hrn if "ns0.972" in s][0]

    # ctx with MF + shape floor on DESI (the HFLOSO972 config); keep DESI only.
    ctx, d = build_legb_ctx(with_mf=True, mf_fold=FOLD, mf_with_floor=True,
                            mf_exclude_held=True, mf_shape=True, mf_shape_infl=INFL,
                            mf_shape_legs=("DESI",))
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    leg = ctx.legs[0]
    truth = make_hr_truth_from_cache(sim972, ctx.cache_k, tau0_anchor="priya")
    mock_legs, tp, _ = make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = _mock_core_per_leg(ctx, truth)["DESI"]
    th, t0g, al = tp["theta9"], tp["tau0_global"], tp["alpha_hcd"]
    zg = np.asarray(ctx.z_global)
    sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
    tau0_vec = t0g[np.asarray(sel)]
    al_leg = al if np.ndim(al) == 1 else al[np.asarray(sel)]
    szb = ctx.sigma_zb_per_leg.get("DESI")
    rzb = ctx.rho_zb_per_leg.get("DESI") if ctx.rho_zb_per_leg else None
    Cs_frac = ctx.mf_shape_per_leg["DESI"]

    kw = dict(pf_stats=ctx.pf_stats, dla_core=core, cache_k=ctx.cache_k, leg=leg,
              sigma_zb=szb, alpha_centres=ctx.alpha_centres, rho_zb=rzb, mf=ctx.mf)
    # (i) C_emu only (no floor, no shape): C_total = C_data + diag(emu_var)
    P, C_emuonly = DL.predict_P_obs_on_leg(ctx.model, th, tau0_vec, al_leg, mf_floor=None, **kw)
    # (ii) + old diagonal MF floor (force it on by computing _mf_floor_var_on_k per z)
    P2, C_floor = DL.predict_P_obs_on_leg(ctx.model, th, tau0_vec, al_leg,
                                          mf_floor=ctx.mf_floor, **kw)  # DESI mf_floor_on=False → ==C_emuonly
    # (iii) + shape floor
    P3, C_shapef = DL.predict_P_obs_on_leg(ctx.model, th, tau0_vec, al_leg, mf_floor=ctx.mf_floor,
                                           mf_shape_cov=Cs_frac, mf_shape_infl=INFL, **kw)

    P = np.asarray(P); kr = np.isfinite(np.asarray(leg.P_data))
    Cdata = np.asarray(leg.C_data)[np.ix_(kr, kr)]
    Cemu_diag = (np.diag(np.asarray(C_emuonly)) - np.diag(np.asarray(leg.C_data)))[kr]
    # the OLD diagonal MF floor on DESI (compute directly, since it's off by default here):
    floor_diag = np.zeros(leg.k.shape[0])
    z_idx = np.asarray(leg.z_idx)
    nsphys = 0.8 + 0.25 * float(th[0])
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        fv = DL._mf_floor_var_on_k(ctx.mf_floor, float(leg.z[iz]),
                                   np.asarray(leg.k)[rows], P[rows], nsphys)
        floor_diag[rows] = np.asarray(fv)
    floor_diag = floor_diag[kr]
    # shape covariance contribution (full matrix incl off-diagonal). Use the FIXED fiducial
    # amplitude P_data (the θ-independent construction the likelihood uses), not the live model P.
    Pfid = np.nan_to_num(np.asarray(leg.P_data))[kr]
    Cshape = (INFL ** 2) * np.asarray(Cs_frac)[np.ix_(kr, kr)] * (Pfid[:, None] * Pfid[None, :])
    Pk = P[kr]

    def corr(C):
        d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        return C / (d[:, None] * d[None, :])

    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    # ---- row 1: correlation matrices ----
    for a, (C, ttl) in zip(ax[0], [
            (Cdata, "(a) C_data (DESI survey) — DENSE"),
            (np.diag(Cemu_diag) + 1e-300, "(b) C_emu (cross-class) — DIAGONAL in k"),
            (Cshape, f"(c) C_shape (NEW shape floor, infl {INFL}) — OFF-DIAGONAL tilt")]):
        im = a.imshow(corr(C), vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        a.set_title(ttl, fontsize=10); a.set_xlabel("flat row (z-major, k within z)")
        a.set_ylabel("flat row"); fig.colorbar(im, ax=a, fraction=0.046)
    # ---- row 2: fractional sqrt(diag)/P per component ----
    b = ax[1, 0]
    b.semilogy(100 * np.sqrt(np.diag(Cdata)) / Pk, ".", ms=3, label="C_data")
    b.semilogy(100 * np.sqrt(np.clip(Cemu_diag, 0, None)) / Pk, ".", ms=3, label="C_emu (xclass)")
    b.semilogy(100 * np.sqrt(np.clip(floor_diag, 0, None)) / Pk, ".", ms=3, label="old MF floor (diag)")
    b.semilogy(100 * np.sqrt(np.clip(np.diag(Cshape), 0, None)) / Pk, ".", ms=3, label="shape floor (diag)")
    b.set_xlabel("flat row (z-major)"); b.set_ylabel("fractional σ [% of P]")
    b.set_title("(d) per-row fractional σ by component"); b.legend(fontsize=8); b.grid(alpha=0.3)
    # (e) the shape-floor correlation alone (zoom on off-diagonal blocks)
    e = ax[1, 1]
    im = e.imshow(corr(Cshape), vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    e.set_title("(e) C_shape correlation (cross-k WITHIN z + cross-z blocks)", fontsize=10)
    e.set_xlabel("flat row"); e.set_ylabel("flat row"); fig.colorbar(im, ax=e, fraction=0.046)
    # (f) old C_emu correlation = identity (to make the 'diagonal' point unmistakable)
    f = ax[1, 2]
    im = f.imshow(corr(np.diag(Cemu_diag) + 1e-300), vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    f.set_title("(f) C_emu correlation = IDENTITY (no k/z off-diagonal)", fontsize=10)
    f.set_xlabel("flat row"); f.set_ylabel("flat row"); fig.colorbar(im, ax=f, fraction=0.046)

    fig.suptitle("C_total decomposition on the DESI leg (ns0.972 HFLOSO config): "
                 "C_data dense + diag(C_emu) + diag(old MF floor) + C_shape (new off-diagonal)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(NOTES_FIG), exist_ok=True)
    fig.savefig(NOTES_FIG, dpi=120)
    print("wrote", NOTES_FIG)
    # quick numeric summary
    print(f"DESI leg N(kept)={kr.sum()}")
    print(f"  C_data   frac σ median={100*np.median(np.sqrt(np.diag(Cdata))/Pk):.2f}%")
    print(f"  C_emu    frac σ median={100*np.median(np.sqrt(np.clip(Cemu_diag,0,None))/Pk):.2f}%")
    print(f"  oldfloor frac σ median={100*np.median(np.sqrt(np.clip(floor_diag,0,None))/Pk):.2f}% (off on DESI by default)")
    print(f"  shape    frac σ median={100*np.median(np.sqrt(np.clip(np.diag(Cshape),0,None))/Pk):.2f}%")
    offmax = np.max(np.abs(corr(Cshape) - np.eye(kr.sum())))
    print(f"  shape correlation max |off-diagonal| = {offmax:.2f}")


if __name__ == "__main__":
    main()
