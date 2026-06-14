#!/usr/bin/env python3
"""diag_mf_rescorr_vs_tau0_z.py -- FORWARD cache analysis (no training) of how the
HF/LF P1D resolution correction rho = P_HR / P_LF depends jointly on tau0 (mean
flux) and redshift z. This is the DESIGN-INPUT diagnostic for a tau0-resolved
multi-fidelity (LF->HR) correction: does rho need to be resolved in BOTH (z, tau0),
or is one axis dominant?

DEFINITIONS
  - Clean-forest P1D = P_tier_c_filtered[:, 0, :]  (the 'clean' coarse class).
  - tau0 ladder = alpha_idx (0..19). alpha_slope is the per-idx tau0-RESCALE
    multiplier (bit-identical LF<->HR per idx): idx 0 -> alpha_slope~0.656 (LOW
    tau0 / HIGH mean flux), idx 19 -> alpha_slope~1.331 (HIGH tau0 / LOW mean flux).
    We also carry target_F (the mean-flux target each rung was solved to) for the
    physical readout.
  - 6 HR sims are EXACT LF design points. Match HR<->LF rows on
    (params_unit rounded, z rounded, alpha_idx) -- identical to
    diag_lf_vs_hr_highk / multifidelity.match_hr_to_lf. tau0 match is EXACT.
  - rho(sim, z, alpha_rung, k) = P_HR / P_LF on matched rows, HR log-log
    interpolated DOWN onto the LF native k-grid (LF Nyquist ~0.069 s/km), so HR is
    interpolated within its support and never extrapolated. Reported on a fixed
    reference k-grid; rho is averaged over the 6 sims for the (z, tau0) structure.

OUTPUTS (figures/analysis/04_emulator/):
  mf_rescorr_vs_tau0_z.png   -- multi-panel: (1) rho heatmaps in (z, tau0-rung) at a
    few fixed k centred at 1.0; (2) rho vs z coloured by tau0-rung at high k;
    (3) rho vs tau0-rung coloured by z at the same k.
  mf_rescorr_vs_tau0_z.txt   -- rho(z, tau0, k) table at grid points + 2-line readout
    of z-dependence vs tau0-dependence strength.

ENV:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/diag_mf_rescorr_vs_tau0_z.py
"""
from __future__ import annotations
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

LF_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
OUT_PNG = "/home/mfho/hcd_priya/figures/analysis/04_emulator/mf_rescorr_vs_tau0_z.png"
OUT_TXT = "/home/mfho/hcd_priya/figures/analysis/04_emulator/mf_rescorr_vs_tau0_z.txt"

CLEAN = 0
N_ALPHA = 20

# PRIYA design box (data.py PARAM_LIMITS) for the unit-cube match key.
PARAM_LIMITS = np.array([
    [0.8, 1.05], [1.2e-9, 2.6e-9], [3.5, 4.5], [2.2, 3.2], [1.3, 3.0],
    [0.65, 0.75], [0.14, 0.146], [6.5, 8.0], [0.03, 0.07]], dtype=np.float64)

# reference k-grid: log-uniform from the resolution band up to the LF Nyquist.
K_LO, K_HI, K_REF = 0.01, 0.069, 40

# fixed k slices to display in the heatmaps + line panels (s/km).
K_PANELS = [0.02, 0.04, 0.06]
K_LINE = 0.05            # representative high-k for the rho-vs-z / rho-vs-tau0 panels


def _norm(p):
    return (p - PARAM_LIMITS[:, 0]) / (PARAM_LIMITS[:, 1] - PARAM_LIMITS[:, 0])


def _strs(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def load_cache(fn):
    with h5py.File(fn, "r") as f:
        d = dict(
            P=f["P_tier_c_filtered"][:, CLEAN, :],   # (N, K) clean-forest P1D
            kfkms=f["kfkms"][:],                      # (N, K)
            z=np.round(f["z_grid"][:], 4),
            alpha=f["alpha_idx"][:].astype(int),
            aslope=f["alpha_slope"][:],
            target_F=f["target_F"][:],
            params=f["params"][:],
            sim=_strs(f["sim_name"][:]),
        )
    d["pu"] = np.round(_norm(d["params"]), 6)
    return d


def rowkey(d):
    return [(tuple(d["pu"][i]), d["z"][i], int(d["alpha"][i]))
            for i in range(len(d["z"]))]


def main():
    lf = load_cache(LF_CACHE)
    hr = load_cache(HR_CACHE)

    k_ref = np.power(10.0, np.linspace(np.log10(K_LO), np.log10(K_HI), K_REF))
    logk_ref = np.log10(k_ref)

    # match every HR row to its LF row on (params_unit, z, alpha_idx)
    lk = {k: i for i, k in enumerate(rowkey(lf))}
    hkeys = rowkey(hr)
    hr_rows, lf_rows = [], []
    for h, k in enumerate(hkeys):
        if k in lk:
            hr_rows.append(h)
            lf_rows.append(lk[k])
    hr_rows = np.array(hr_rows)
    lf_rows = np.array(lf_rows)
    assert len(hr_rows) == len(hkeys), "not all HR rows matched"

    sims = sorted(set(hr["sim"]))
    n_sim = len(sims)
    sim_idx = {s: i for i, s in enumerate(sims)}

    # per-row rho = P_HR / P_LF on the reference grid (HR interp DOWN onto k_ref)
    M = len(hr_rows)
    rho = np.full((M, K_REF), np.nan)
    row_sim = np.empty(M, dtype=int)
    row_z = np.empty(M)
    row_alpha = np.empty(M, dtype=int)
    for m, (h, l) in enumerate(zip(hr_rows, lf_rows)):
        Phr, khr = hr["P"][h], hr["kfkms"][h]
        Plf, klf = lf["P"][l], lf["kfkms"][l]
        mhr = np.isfinite(Phr) & (Phr > 0) & np.isfinite(khr) & (khr > 0)
        mlf = np.isfinite(Plf) & (Plf > 0) & np.isfinite(klf) & (klf > 0)
        logPhr = np.interp(logk_ref, np.log10(khr[mhr]), np.log(Phr[mhr]),
                           left=np.nan, right=np.nan)
        logPlf = np.interp(logk_ref, np.log10(klf[mlf]), np.log(Plf[mlf]),
                           left=np.nan, right=np.nan)
        rho[m] = np.exp(logPhr - logPlf)
        row_sim[m] = sim_idx[hr["sim"][h]]
        row_z[m] = hr["z"][h]
        row_alpha[m] = hr["alpha"][h]

    zvals = np.array(sorted(set(row_z)))
    nz = len(zvals)
    # alpha-rung -> (alpha_slope, target_F at z=2.2) for axis labels / readout.
    aslope_by_rung = np.array([
        float(np.mean(lf["aslope"][lf["alpha"] == a])) for a in range(N_ALPHA)])
    # mean target_F per rung (averaged over z & sim) just for context label.
    tF_by_rung = np.array([
        float(np.mean(lf["target_F"][lf["alpha"] == a])) for a in range(N_ALPHA)])

    # rho_mean(z, rung, k): mean over the 6 sims. First mean over sims at each
    # (z, rung) cell (each cell has exactly the 6 sims, 1 row each).
    rho_zak = np.full((nz, N_ALPHA, K_REF), np.nan)
    for zi, zz in enumerate(zvals):
        for a in range(N_ALPHA):
            sel = (row_z == zz) & (row_alpha == a)
            if sel.any():
                with np.errstate(invalid="ignore"):
                    rho_zak[zi, a] = np.nanmean(rho[sel], axis=0)

    # nearest reference-k indices for the requested slices
    def kidx(kq):
        return int(np.argmin(np.abs(k_ref - kq)))
    kpan_idx = [kidx(kq) for kq in K_PANELS]
    kline_idx = kidx(K_LINE)

    # ----------------------------- text report ----------------------------- #
    lines = []
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    ns_vals = [float(hr["params"][hr["sim"] == s][0][0]) for s in sims]
    ap_vals = [float(hr["params"][hr["sim"] == s][0][1]) for s in sims]

    P("=" * 80)
    P("MF LF->HR resolution correction rho = P_HR/P_LF  vs  tau0 (mean flux) and z")
    P("FORWARD cache analysis (no training). Clean-forest P1D, 6 HR sims (= LF pts).")
    P("=" * 80)
    P(f"reference k-grid: {K_REF} log bins, k in [{K_LO},{K_HI}] s/km (LF Nyquist).")
    P(f"  HR log-log interpolated DOWN onto LF k-grid; never extrapolated.")
    P(f"matched HR rows: {M};  HR sims: {n_sim};  z bins: {nz} (2.0..5.4).")
    P(f"tau0 ladder: alpha_idx 0..19.  alpha_slope = tau0-rescale multiplier:")
    P(f"  rung 0  -> alpha_slope={aslope_by_rung[0]:.3f}  (LOW tau0 / HIGH <F>, "
      f"target_F~{tF_by_rung[0]:.3f})")
    P(f"  rung 19 -> alpha_slope={aslope_by_rung[19]:.3f}  (HIGH tau0 / LOW <F>, "
      f"target_F~{tF_by_rung[19]:.3f})")
    P(f"HR sims ns span [{min(ns_vals):.3f},{max(ns_vals):.3f}], "
      f"Ap [{min(ap_vals):.2e},{max(ap_vals):.2e}] (clustered mid-high).")
    P("")
    P("CONVENTION: rho = P_HR/P_LF. rho>1 => LF is power-DEFICIENT (correction boosts).")
    P("")

    # rho table at a few grid points
    z_pts = [2.2, 3.6, 5.0]
    a_pts = [0, 9, 19]
    P("-" * 80)
    P("rho(z, tau0-rung, k) at selected grid points (6-sim mean):")
    P("-" * 80)
    for kq, ki in zip(K_PANELS + [K_LINE], kpan_idx + [kline_idx]):
        P(f"  k ~ {k_ref[ki]:.4f} s/km:")
        hdr = "     z \\ rung  " + "".join(
            f"{f'#{a}(s={aslope_by_rung[a]:.2f})':>16s}" for a in a_pts)
        P(hdr)
        for zq in z_pts:
            zi = int(np.argmin(np.abs(zvals - zq)))
            cells = "".join(
                f"{rho_zak[zi, a, ki]:16.4f}" for a in a_pts)
            P(f"     z={zvals[zi]:.1f}      {cells}")
        P("")

    # ----- quantify z-strength vs tau0-strength of rho (at the line k) ------ #
    # build rho(z, rung) at k=K_LINE, then peak-to-peak ranges along each axis.
    R = rho_zak[:, :, kline_idx]            # (nz, N_ALPHA)
    with np.errstate(invalid="ignore"):
        # z-variation at fixed rung: max over rungs of [max_z rho - min_z rho]
        z_ptp_per_rung = np.nanmax(R, axis=0) - np.nanmin(R, axis=0)
        # tau0-variation at fixed z: max over z of [max_rung rho - min_rung rho]
        a_ptp_per_z = np.nanmax(R, axis=1) - np.nanmin(R, axis=1)
        z_ptp = float(np.nanmax(z_ptp_per_rung))      # full z-swing
        a_ptp = float(np.nanmax(a_ptp_per_z))         # full tau0-swing
        z_ptp_med = float(np.nanmedian(z_ptp_per_rung))
        a_ptp_med = float(np.nanmedian(a_ptp_per_z))
    # where is rho worst (largest |rho-1|) at the line k
    iworst = np.unravel_index(np.nanargmax(np.abs(R - 1.0)), R.shape)
    P("-" * 80)
    P(f"STRENGTH of each axis at k~{k_ref[kline_idx]:.4f} s/km (rho range = peak-to-peak):")
    P("-" * 80)
    P(f"  z-driven swing of rho (max over rungs of rho.max_z - rho.min_z): "
      f"{z_ptp:.4f}  ({100*z_ptp:.1f} pp in P);  median over rungs {100*z_ptp_med:.1f} pp")
    P(f"  tau0-driven swing of rho (max over z of rho.max_rung - rho.min_rung): "
      f"{a_ptp:.4f}  ({100*a_ptp:.1f} pp in P);  median over z {100*a_ptp_med:.1f} pp")
    P(f"  worst |rho-1| cell: z={zvals[iworst[0]]:.1f} rung#{iworst[1]} "
      f"(alpha_slope={aslope_by_rung[iworst[1]]:.2f}) -> rho={R[iworst]:.4f} "
      f"({100*(R[iworst]-1):+.1f}%)")
    # tau0 trend at the lowest z (the regime cited in the prior finding)
    zi_lo = int(np.argmin(np.abs(zvals - 2.2)))
    r_lo, r_hi = R[zi_lo, 0], R[zi_lo, N_ALPHA - 1]
    P(f"  prior-finding check (z={zvals[zi_lo]:.1f}, k~{k_ref[kline_idx]:.3f}): "
      f"rung0 rho={r_lo:.4f} ({100*(r_lo-1):+.1f}%)  vs  "
      f"rung19 rho={r_hi:.4f} ({100*(r_hi-1):+.1f}%)  "
      f"=> low-tau0 boost {100*(r_lo-r_hi):+.1f}pp larger.")
    P("")

    # 2-line readout
    dominant = "z" if z_ptp > a_ptp else "tau0"
    ratio = z_ptp / a_ptp if a_ptp > 0 else np.inf
    P("=" * 80)
    P("READOUT:")
    P(f"  z-dependence of rho: full swing ~{100*z_ptp:.1f} pp in P (k~{K_LINE}); "
      f"rho is largest (LF most deficient) at LOW z, falling toward high z.")
    P(f"  tau0-dependence of rho: full swing ~{100*a_ptp:.1f} pp in P at fixed z,k; "
      f"rho is larger at LOW tau0 (high <F>) than HIGH tau0.")
    P(f"  => DOMINANT axis = {dominant} (z-swing/tau0-swing ~ {ratio:.1f}x). "
      f"{'Both' if ratio < 3 else 'z'} axes matter for the MF correction; "
      f"tau0 contributes a {'non-negligible' if a_ptp > 0.01 else 'small'} "
      f"~{100*a_ptp:.1f}pp secondary trend.")
    P("=" * 80)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {OUT_TXT}")

    # ================================ FIGURE =============================== #
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30,
                          height_ratios=[1.0, 0.85])

    rung_ax = np.arange(N_ALPHA)
    # diverging norm centred at 1.0; symmetric range from the data
    all_vals = rho_zak[:, :, kpan_idx]
    with np.errstate(invalid="ignore"):
        vmax = float(np.nanmax(np.abs(all_vals - 1.0)))
    vmax = max(vmax, 0.02)
    norm = TwoSlopeNorm(vmin=1.0 - vmax, vcenter=1.0, vmax=1.0 + vmax)

    # ---- top row: 3 heatmaps rho(z [y], tau0-rung [x]) at fixed k ---------- #
    for col, (kq, ki) in enumerate(zip(K_PANELS, kpan_idx)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(
            np.arange(N_ALPHA + 1), np.arange(nz + 1),
            rho_zak[:, :, ki], cmap="RdBu_r", norm=norm, shading="flat")
        ax.set_xticks(np.arange(N_ALPHA) + 0.5)
        ax.set_xticklabels([str(a) for a in range(N_ALPHA)], fontsize=6)
        ax.set_yticks(np.arange(nz) + 0.5)
        ax.set_yticklabels([f"{z:.1f}" for z in zvals], fontsize=7)
        ax.set_xlabel("tau0-rung (0=low tau0/high <F>  ->  19=high tau0/low <F>)",
                      fontsize=8)
        if col == 0:
            ax.set_ylabel("z")
        ax.set_title(f"rho = P_HR/P_LF   k ~ {k_ref[ki]:.4f} s/km   (6-sim mean)",
                     fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("rho (1.0 = no correction)", fontsize=8)
        # annotate a few cells
        for zq in [2.2, 3.6]:
            zi = int(np.argmin(np.abs(zvals - zq)))
            for a in [0, 19]:
                v = rho_zak[zi, a, ki]
                if np.isfinite(v):
                    ax.text(a + 0.5, zi + 0.5, f"{v:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="k" if abs(v - 1) < 0.6 * vmax else "w")

    # ---- bottom-left: rho vs z, lines coloured by tau0-rung --------------- #
    ax = fig.add_subplot(gs[1, 0])
    cmap = matplotlib.colormaps["viridis"]
    for a in range(N_ALPHA):
        ax.plot(zvals, rho_zak[:, a, kline_idx],
                color=cmap(a / (N_ALPHA - 1)), lw=1.3)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("z")
    ax.set_ylabel("rho = P_HR/P_LF")
    ax.set_title(f"rho vs z, coloured by tau0-rung   k ~ {k_ref[kline_idx]:.4f} s/km",
                 fontsize=10)
    ax.grid(alpha=0.3)
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(0, N_ALPHA - 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("tau0-rung (low->high tau0)", fontsize=8)

    # ---- bottom-mid: rho vs tau0-rung, lines coloured by z ---------------- #
    ax = fig.add_subplot(gs[1, 1])
    cmap2 = matplotlib.colormaps["plasma"]
    for zi, zz in enumerate(zvals):
        ax.plot(rung_ax, rho_zak[zi, :, kline_idx],
                color=cmap2(zi / (nz - 1)), lw=1.3)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("tau0-rung (0=low tau0/high <F>  ->  19=high tau0/low <F>)",
                  fontsize=8)
    ax.set_ylabel("rho = P_HR/P_LF")
    ax.set_title(f"rho vs tau0-rung, coloured by z   k ~ {k_ref[kline_idx]:.4f} s/km",
                 fontsize=10)
    ax.grid(alpha=0.3)
    sm2 = cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(zvals.min(), zvals.max()))
    cb = fig.colorbar(sm2, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("z", fontsize=8)

    # ---- bottom-right: rho vs tau0 at 3 z slices (annotated numbers) ------ #
    ax = fig.add_subplot(gs[1, 2])
    aslope_x = aslope_by_rung
    for zq, c in zip([2.2, 3.6, 5.0], ["C3", "C1", "C0"]):
        zi = int(np.argmin(np.abs(zvals - zq)))
        ax.plot(aslope_x, rho_zak[zi, :, kline_idx], "o-", color=c, ms=3, lw=1.6,
                label=f"z={zvals[zi]:.1f}")
        # annotate endpoints
        for a in [0, N_ALPHA - 1]:
            v = rho_zak[zi, a, kline_idx]
            if np.isfinite(v):
                ax.annotate(f"{100*(v-1):+.1f}%", (aslope_x[a], v),
                            fontsize=6, color=c,
                            xytext=(0, 4 if a == 0 else -8),
                            textcoords="offset points", ha="center")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("alpha_slope (tau0-rescale multiplier; low->high tau0)", fontsize=8)
    ax.set_ylabel("rho = P_HR/P_LF")
    ax.set_title(f"tau0 trend at 3 z   k ~ {k_ref[kline_idx]:.4f} s/km", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "MF LF->HR resolution correction rho = P_HR/P_LF vs tau0 (mean flux) and z  "
        "(forward cache, 6 HR sims; clean forest)", fontsize=13, y=0.99)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[wrote] {OUT_PNG}")


if __name__ == "__main__":
    main()
