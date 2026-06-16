"""SCRATCH (read-only diagnostic): res_corr-amplitude degeneracy Fisher pre-check (v2).

Q: is an ANCHORED res_corr amplitude nuisance (log res_corr -> alpha*deltahat(k,z),
with deltahat forced to 1 below 5*k_box(z)) separable from n_s, or degenerate?
And how aligned is it with the OTHER marginalized directions (dtau0 the mean-flux
z-slope, a_HCD the LLS amplitude, and the emucoh coherent-residual mode)?

For each leg (DESI, KS, eBOSS) we compute the whitened (C_data^{-1} metric) cosine of
the anchored res_corr response deltahat against:
  (a) n_s     : r_ns   = d logP / d n_s         (PRODUCTION MF forward, jax.jacfwd + FD-checked)
  (b) dtau0   : r_dtau = d logP / d dtau0        (mean-flux z-slope; FD-checked)
  (c) a_HCD   : r_aHCD = d logP / d (alpha_LLS amplitude)  (LLS incidence; FD-checked)
  (d) emucoh  : the leading eigenvector of the FRACTIONAL covariance in
                _emulator_data/mf_cemu_emucoh.npz, bound to the leg (z,k) grid by interp
                (a fractional shape => directly comparable to d logP). ALSO restricted to
                z>=2.8 high-k cells (the He-II window) separately.

Plus the per-leg PRIOR-DOMINANCE fraction of sigma(n_s) for a trial sigma_a0 in {0.3,0.5,1.0}:
  dominance = 1 - sigma(n_s | alpha0 free w/ Gaussian prior sigma_a0) / sigma(n_s | alpha0 fixed)
via the 2-param (n_s, alpha0) Fisher (alpha0 = the res_corr amplitude, response = deltahat).

NB: r_ns/r_dtau/r_aHCD are taken d/d(unit or natural param); the constant param->phys scale
cancels in BOTH the cosine and the inflation factor (only the DIRECTION of each response matters).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/scratch_res_corr_ns_fisher.py
"""
from __future__ import annotations
import functools
print = functools.partial(print, flush=True)

import numpy as np
import hcd_analysis.emulator  # x64 before jax
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.closure_legb import build_legb_ctx
from hcd_analysis.emulator.data_likelihood import predict_P_obs_on_leg, _predict_P_obs_mf
from hcd_analysis.emulator.multifidelity import interp_res_corr
from hcd_analysis.emulator.predict import predict_P_filt, _excess_from_P_filt
from hcd_analysis.emulator.data import SAMPLING_LIMITS, PARAM_LIMITS, normalize_params, Z_LIMITS

# v2 figure (the v1 path is left for the original prior check).
FIG = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator/res_corr_fisher_v2.png"
EMUCOH_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/mf_cemu_emucoh.npz"

# --- k_box anchor (L15 box fundamental in s/km), prompt formula --------------- #
H0, OM, HUB = 70.0, 0.3, 0.7
def Hz(z):
    return H0 * np.sqrt(OM * (1.0 + z) ** 3 + (1.0 - OM))
def k_box(z):
    # (2pi/15 Mpc/h) mapped to velocity via Hubble flow -> s/km
    return (2.0 * np.pi / 15.0) * HUB * (1.0 + z) / Hz(z)

ANCHOR_MULT = 5.0          # v2: force res_corr->1 below 5 * k_box(z)   (was 3.0)
TAPER_WIDTH_DEX = 0.12     # tanh blend width in log10 k (dex)

# He-II window for the z-resolved emucoh restriction.
Z_HEII = 2.8

# trial Gaussian prior widths on the res_corr amplitude alpha0 (for prior-dominance).
SIGMA_A0_TRIALS = (0.3, 0.5, 1.0)


def anchor_taper(k, z):
    """Smooth tanh low-k taper: ~0 below 5*k_box(z), ~1 above, width ~0.12 dex.

    The anchored response deltahat = log(res_corr) * taper -> forced to 0 (res_corr->1)
    below 5*k_box(z), untouched above."""
    k = np.asarray(k, float)
    kc = ANCHOR_MULT * k_box(z)
    x = (np.log10(k) - np.log10(kc)) / TAPER_WIDTH_DEX
    return 0.5 * (1.0 + np.tanh(x))


def _load_emucoh_leading_mode():
    """Leading eigenvector of the emucoh FRACTIONAL covariance, on its (z,k) grid.

    Returns (z_emu (nz,), k_emu (nk,), mode_zk (nz,nk)) where mode is the unit-norm
    leading eigenvector reshaped to the (z-major, k) grid. f_shape is the (nz*nk)^2
    fractional covariance (dP/P units), so its eigenvectors are fractional shapes that
    compare directly to d logP responses."""
    d = np.load(EMUCOH_NPZ)
    z_emu = np.asarray(d["z"], float)            # (13,)
    k_emu = np.asarray(d["k"], float)            # (134,) s/km, angular
    F = np.asarray(d["f_shape"], float)          # (13*134, 13*134) fractional cov
    nz, nk = z_emu.size, k_emu.size
    assert F.shape == (nz * nk, nz * nk), F.shape
    w, V = np.linalg.eigh(F)
    mode = V[:, -1]                              # leading eigenvector (unit norm)
    # sign convention: make the largest-|amp| entry positive (cosmetic; cos uses |.|).
    if mode[np.argmax(np.abs(mode))] < 0:
        mode = -mode
    mode_zk = mode.reshape(nz, nk)               # z-major (matches f_shape build order)
    frac_top = float(w[-1] / w.sum())
    print(f"   [emucoh] grid z[{z_emu.min():.1f},{z_emu.max():.1f}] ({nz}) "
          f"k[{k_emu.min():.4f},{k_emu.max():.4f}] ({nk}); leading eig var-frac={frac_top:.3f}")
    return z_emu, k_emu, mode_zk


def _emucoh_on_leg(z_emu, k_emu, mode_zk, z_leg, z_idx, k_leg, N):
    """Bind the emucoh leading mode to a leg's flat (z,k) grid by interp.

    For each leg z-row, pick the nearest emucoh z then linearly-interp the mode in log10 k
    onto the leg k of that row. Outside the emucoh k-span -> 0 (the mode has no support
    there; res_corr/n_s/dtau0 responses also live mostly inside it). Returns (N,) flat."""
    out = np.zeros(N)
    logk_emu = np.log10(k_emu)
    Z_TOL = 0.1                                       # production mf_shape_cov_for_leg z_tol
    for iz in range(len(z_leg)):
        rsel = np.where(z_idx == iz)[0]
        if rsel.size == 0:
            continue
        zz = float(z_leg[iz])
        je = int(np.argmin(np.abs(z_emu - zz)))     # nearest emucoh z
        if abs(float(z_emu[je]) - zz) > Z_TOL:
            continue                                 # leg z outside table support -> zero (e.g. KS z=4.6)
        kk = np.asarray(k_leg[rsel], float)
        lk = np.log10(np.clip(kk, k_emu.min(), k_emu.max()))
        vals = np.interp(lk, logk_emu, mode_zk[je])
        # zero out leg-k strictly outside the emucoh k-span (no extrapolation).
        inside = (kk >= k_emu.min()) & (kk <= k_emu.max())
        out[rsel] = np.where(inside, vals, 0.0)
    return out


def _whitened_cos(Cinv):
    def wcos(a, b):
        num = a @ Cinv @ b
        den = np.sqrt((a @ Cinv @ a) * (b @ Cinv @ b))
        if den <= 0:
            return float("nan")
        return float(num / den)
    return wcos


def _prior_dominance_2param(r_ns, r_alpha, Cinv, sigma_a0):
    """2-param (n_s, alpha0) Gaussian-prior Fisher prior-dominance of sigma(n_s).

    Likelihood Fisher F = J^T Cinv J with J=[r_ns, r_alpha] (logP responses).  Add a
    Gaussian prior 1/sigma_a0^2 on alpha0 only.  Compare sigma(n_s):
      free  = sqrt([ (F + diag(0, 1/sigma_a0^2))^{-1} ]_{ns,ns})
      fixed = sqrt(1 / F_{ns,ns})       (alpha0 clamped)
    dominance = 1 - sigma_free_unit / sigma_fixed_unit.

    The n_s param scale cancels: r_ns is in unit-cube d/dns, so sigma(n_s) here is in the
    SAME unit-cube units for both 'free' and 'fixed'; the ratio is scale-free."""
    J = np.column_stack([r_ns, r_alpha])             # (N,2)
    F = J.T @ Cinv @ J                               # (2,2)
    sig_fixed = 1.0 / np.sqrt(F[0, 0])               # alpha0 clamped
    Fp = F.copy()
    Fp[1, 1] += 1.0 / (sigma_a0 ** 2)                # Gaussian prior on alpha0
    cov = np.linalg.inv(Fp)
    sig_free = np.sqrt(cov[0, 0])
    ratio = sig_free / sig_fixed                     # >= 1 (freeing alpha0 cannot shrink sigma(ns))
    dom_literal = 1.0 - ratio                        # the task's literal formula (<= 0)
    inflation = ratio - 1.0                          # the practical magnitude (how much sigma(ns) grows)
    return dom_literal, inflation


def _fd_check(f, x0, name, h=1e-3):
    """Central finite-difference of a vector-valued f at scalar x0, report vs an AD jac."""
    jac = np.asarray(jax.jacfwd(f)(x0))
    fp = np.asarray(f(x0 + h))
    fm = np.asarray(f(x0 - h))
    fd = (fp - fm) / (2.0 * h)
    num = float(np.max(np.abs(jac - fd)))
    den = float(np.max(np.abs(jac)) + 1e-30)
    rel = num / den
    flag = "OK" if rel < 5e-3 else "WARN"
    print(f"   [FD-check {name}] max|AD-FD|={num:.3e}  rel={rel:.2e}  ({flag})")
    return jac


def main():
    print("=== building production Leg-B ctx (MF forward, DESI+KS+eBOSS) ===")
    ctx, _dla_core_global = build_legb_ctx(with_mf=True, with_eboss=True)
    model = ctx.model
    pf_stats = ctx.pf_stats
    mf = ctx.mf
    cache_k = np.asarray(ctx.cache_k)
    z_pivot = float(ctx.tau0_pivot_z)

    # --- FIDUCIAL theta: SAMPLING prior CENTER in the unit cube ----------------
    lim = SAMPLING_LIMITS
    theta_phys_fid = 0.5 * (lim[:, 0] + lim[:, 1])              # physical
    theta_unit_fid = jnp.asarray(normalize_params(theta_phys_fid))  # (9,) unit cube
    ns_phys_fid = float(theta_phys_fid[0])
    print(f"fiducial theta (phys): ns={theta_phys_fid[0]:.4f} Ap={theta_phys_fid[1]:.3e} "
          f"herei={theta_phys_fid[2]:.3f} heref={theta_phys_fid[3]:.3f} alphaq={theta_phys_fid[4]:.3f} "
          f"hub={theta_phys_fid[5]:.3f} omh2={theta_phys_fid[6]:.4f} hreion={theta_phys_fid[7]:.3f} "
          f"bhfb={theta_phys_fid[8]:.4f}")
    print(f"fiducial theta_unit: {np.asarray(theta_unit_fid)}")
    print(f"tau0 pivot z_p = {z_pivot}")

    # tau0 ladder at the prior center (Becker13 anchor used by build_legb_ctx)
    z_global = np.asarray(ctx.z_global)
    tau0_mu = np.asarray(ctx.tau0_mu)                            # per global-z prior center
    alpha_hcd_fid = jnp.asarray(ctx.alpha_hcd_mu)               # (3,) HCD incidence center
    print(f"tau0 prior center (Becker13) over z_global[{z_global.min():.2f},{z_global.max():.2f}]: "
          f"mean {tau0_mu.mean():.3f}")
    print(f"alpha_hcd center (LLS,subDLA,DLA): {np.asarray(alpha_hcd_fid)}")
    print(f"ANCHOR_MULT = {ANCHOR_MULT} (res_corr forced ->1 below {ANCHOR_MULT}*k_box(z))")

    # res_corr table handles from the production mf
    z_rc, logk_rc, rc_vals = np.asarray(mf.z_rc), np.asarray(mf.logk_rc), np.asarray(mf.rc_vals)

    # emucoh leading coherent-residual mode (fractional shape on its (z,k) grid).
    z_emu, k_emu, mode_zk = _load_emucoh_leading_mode()

    rows = []
    dom_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
    for col, leg in enumerate(ctx.legs):
        ax = axes[0, col]
        name = leg.name
        dla_core = ctx.dla_core_leg[name]
        k_leg = np.asarray(leg.k)                                # (N,) flat z-major
        z_idx = np.asarray(leg.z_idx)
        z_leg = np.asarray(leg.z)                               # (n_z,)
        C_data = np.asarray(leg.C_data)
        N = k_leg.shape[0]

        # tau0 per leg-z from the global ladder (nearest z) -----------------
        tau0_fid_np = np.array([tau0_mu[int(np.argmin(np.abs(z_global - zz)))] for zz in z_leg])
        z_leg_j = jnp.asarray(z_leg)

        # ---- model logP through the PRODUCTION MF forward, as a fn of (ns_unit, dtau0, a_HCD)
        # P_model flat (N,), via _predict_P_obs_mf per z + interp to leg k. The fiducial tau0
        # curve is tau0_fid (Becker center); dtau0 tilts it as tau0(z)*((1+z)/(1+z_p))^dtau0,
        # a_HCD scales the LLS (index-0) incidence amplitude.
        rest0 = theta_unit_fid[1:]

        def _logP(ns_unit, dtau0, a_hcd):
            theta = jnp.concatenate([ns_unit[None], rest0])             # (9,) unit
            tilt = ((1.0 + z_leg_j) / (1.0 + z_pivot)) ** dtau0         # (n_z,) dtau0 z-slope
            tau0_z = jnp.asarray(tau0_fid_np) * tilt                    # (n_z,) tilted curve
            alpha_hcd = alpha_hcd_fid.at[0].multiply(a_hcd)            # scale LLS amplitude
            P = jnp.zeros(N)
            for iz in range(leg.n_z):
                rsel = np.where(z_idx == iz)[0]
                if rsel.size == 0:
                    continue
                z_unit = float(leg.z_unit[iz])
                P_cache = _predict_P_obs_mf(mf, model, theta, z_unit, tau0_z[iz],
                                            alpha_hcd, pf_stats, dla_core[iz])
                P_z = jnp.interp(jnp.asarray(k_leg[rsel]), jnp.asarray(cache_k), P_cache)
                P = P.at[jnp.asarray(rsel)].set(P_z)
            return jnp.log(P)

        ns0 = theta_unit_fid[0]
        dtau0_0 = jnp.asarray(0.0)
        ahcd_0 = jnp.asarray(1.0)
        logP0 = np.asarray(_logP(ns0, dtau0_0, ahcd_0))
        P0 = np.exp(logP0)

        print(f"\n[{name}] N={N}  k in [{k_leg.min():.4f},{k_leg.max():.4f}] s/km  "
              f"z in [{z_leg.min():.2f},{z_leg.max():.2f}]")
        print(f"   {ANCHOR_MULT}*k_box(z) anchor range: [{ANCHOR_MULT*k_box(z_leg.max()):.4f}, "
              f"{ANCHOR_MULT*k_box(z_leg.min()):.4f}] s/km")

        # ---- responses (AD + FD-checked) ----------------------------------
        r_ns = np.asarray(_fd_check(lambda x: _logP(x, dtau0_0, ahcd_0), ns0, "n_s"))
        r_dtau = np.asarray(_fd_check(lambda x: _logP(ns0, x, ahcd_0), dtau0_0, "dtau0"))
        r_aHCD = np.asarray(_fd_check(lambda x: _logP(ns0, dtau0_0, x), ahcd_0, "a_HCD"))

        # ---- anchored res_corr response deltahat = log res_corr * taper(5*kbox)
        r_raw = np.zeros(N)
        r_anc = np.zeros(N)
        for iz in range(leg.n_z):
            rsel = np.where(z_idx == iz)[0]
            if rsel.size == 0:
                continue
            zz = float(z_leg[iz])
            rc = np.asarray(interp_res_corr(z_rc, logk_rc, rc_vals, zz, jnp.asarray(k_leg[rsel])))
            dlog = np.log(rc)                                    # log res_corr deviation
            r_raw[rsel] = dlog
            r_anc[rsel] = dlog * anchor_taper(k_leg[rsel], zz)   # force ->0 below 5*kbox

        # ---- emucoh leading mode bound to the leg grid (fractional shape) --
        r_emucoh = _emucoh_on_leg(z_emu, k_emu, mode_zk, z_leg, z_idx, k_leg, N)

        # ---- whitened cosines in C_data^{-1} metric -----------------------
        Cinv = np.linalg.inv(C_data)
        wcos = _whitened_cos(Cinv)

        cos_ns_raw = wcos(r_anc, r_raw)  # sanity only (anchored vs raw res_corr)
        cos_a_ns = wcos(r_anc, r_ns)
        cos_a_dtau = wcos(r_anc, r_dtau)
        cos_a_aHCD = wcos(r_anc, r_aHCD)
        cos_a_emu_allz = wcos(r_anc, r_emucoh)
        # raw (un-anchored) vs n_s, to confirm anchoring did NOT change cos(alpha,n_s):
        cos_a_ns_RAWres = wcos(r_raw, r_ns)

        # emucoh restricted to z>=2.8 high-k cells (He-II window) -----------
        zrow = z_leg[z_idx]                              # (N,) z per flat cell
        hiZ = zrow >= Z_HEII
        if hiZ.sum() >= 2:
            sub = np.where(hiZ)[0]
            C_sub = C_data[np.ix_(sub, sub)]
            Cinv_sub = np.linalg.inv(C_sub)
            wcos_sub = _whitened_cos(Cinv_sub)
            cos_a_emu_hiZ = wcos_sub(r_anc[sub], r_emucoh[sub])
        else:
            cos_a_emu_hiZ = float("nan")

        infl_anc = 1.0 / np.sqrt(max(1.0 - cos_a_ns ** 2, 1e-30))
        # how much whitened res_corr signal SURVIVES the 5x anchor (vs the raw, un-anchored amp):
        amp_raw = float(np.sqrt(r_raw @ Cinv @ r_raw))
        amp_anc = float(np.sqrt(r_anc @ Cinv @ r_anc))
        surv = amp_anc / max(amp_raw, 1e-30)

        rows.append(dict(name=name, cos_a_ns=cos_a_ns, cos_a_ns_RAWres=cos_a_ns_RAWres,
                         cos_a_dtau=cos_a_dtau, cos_a_aHCD=cos_a_aHCD,
                         cos_a_emu_allz=cos_a_emu_allz, cos_a_emu_hiZ=cos_a_emu_hiZ,
                         infl_anc=infl_anc, amp_raw=amp_raw, amp_anc=amp_anc, surv=surv))

        print(f"   whitened res_corr amp: RAW={amp_raw:.3f} ANCHORED={amp_anc:.3f} "
              f"(survival {surv*100:.1f}% of raw signal past the 5x anchor)")
        print(f"   cos(alpha,n_s)   ANCHORED={cos_a_ns:+.4f}   (RAW res_corr={cos_a_ns_RAWres:+.4f})")
        print(f"   cos(alpha,dtau0)            ={cos_a_dtau:+.4f}")
        print(f"   cos(alpha,a_HCD)            ={cos_a_aHCD:+.4f}")
        print(f"   cos(alpha,emucoh) all-z     ={cos_a_emu_allz:+.4f}   z>={Z_HEII}: {cos_a_emu_hiZ:+.4f}")
        print(f"   sigma(n_s) inflation (anchored) = {infl_anc:.3f}x")

        # ---- prior-dominance of sigma(n_s) vs sigma_a0 (2-param Fisher) ----
        # The task's literal "dominance" = 1 - sig_free/sig_fixed is <=0 (freeing alpha0 can only
        # inflate sigma(n_s)); the GATE-relevant magnitude is the INFLATION fraction = sig_free/sig_fixed-1
        # (= |dominance|). "prior-dom < 0.35" => freeing alpha0 inflates sigma(n_s) by < 35%, i.e. the
        # prior+data still constrain n_s well (the leg stays informative).
        infl_frac = {}
        for s0 in SIGMA_A0_TRIALS:
            _dom_lit, infl = _prior_dominance_2param(r_ns, r_anc, Cinv, s0)
            infl_frac[s0] = infl
        dom_rows.append(dict(name=name, infl=infl_frac))
        print("   prior-dominance sigma(n_s) [inflation frac = sig_free/sig_fixed-1]: " +
              "  ".join(f"sigma_a0={s0}: {infl_frac[s0]:.3f}" for s0 in SIGMA_A0_TRIALS))

        # ---- figure (top row): response shapes vs k, per representative z ---
        scale = (np.max(np.abs(r_anc)) / max(np.max(np.abs(r_ns)), 1e-30))
        zsel = [z_leg[0], z_leg[len(z_leg) // 2], z_leg[-1]]
        alphas = [0.45, 0.7, 1.0]
        for zz, al in zip(zsel, alphas):
            iz = int(np.argmin(np.abs(z_leg - zz)))
            rs = np.where(z_idx == iz)[0]
            if rs.size == 0:
                continue
            o = rs[np.argsort(k_leg[rs])]
            last = (zz == zsel[-1])
            ax.plot(k_leg[o], r_anc[o], "-", color="C2", alpha=al, lw=1.6,
                    label=(r"$\hat\delta$ res_corr ANCHORED" if last else None))
            ax.plot(k_leg[o], (r_ns * scale)[o], "-", color="C0", alpha=al * 0.7, lw=1.0,
                    label=(r"$r_{n_s}$ (rescaled)" if last else None))
            sc_d = (np.max(np.abs(r_anc)) / max(np.max(np.abs(r_dtau)), 1e-30))
            ax.plot(k_leg[o], (r_dtau * sc_d)[o], ":", color="C3", alpha=al * 0.7, lw=1.0,
                    label=(r"$r_{d\tau_0}$ (rescaled)" if last else None))
            sc_e = (np.max(np.abs(r_anc)) / max(np.max(np.abs(r_emucoh)), 1e-30))
            ax.plot(k_leg[o], (r_emucoh * sc_e)[o], "-.", color="C4", alpha=al * 0.7, lw=1.0,
                    label=("emucoh mode (rescaled)" if last else None))
        kc_lo = ANCHOR_MULT * k_box(z_leg.max())
        kc_hi = ANCHOR_MULT * k_box(z_leg.min())
        ax.axvspan(kc_lo, kc_hi, color="grey", alpha=0.18,
                   label=f"{ANCHOR_MULT}·k_box(z)")
        ax.set_xscale("log")
        ax.set_title(f"{name}  cos(α,n_s)={cos_a_ns:+.2f}  infl={infl_anc:.2f}×\n"
                     f"cos(α,dτ₀)={cos_a_dtau:+.2f} cos(α,a_HCD)={cos_a_aHCD:+.2f}\n"
                     f"cos(α,emucoh) all-z={cos_a_emu_allz:+.2f} z≥{Z_HEII}={cos_a_emu_hiZ:+.2f}",
                     fontsize=8.5)
        ax.set_xlabel("k [s/km]")
        ax.axhline(0, color="k", lw=0.5)
        if col == 0:
            ax.set_ylabel("response (log res_corr units)")
        ax.legend(fontsize=6.5, loc="best")

    # ---- bottom row: cosine + prior-dominance tables ----------------------
    gs = axes[1, 0].get_gridspec()
    for a in axes[1, :]:
        a.remove()
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")
    hdr = ["leg", "cos(α,n_s)", "cos(α,dτ₀)", "cos(α,a_HCD)",
           "cos(α,emucoh)\nall-z", "cos(α,emucoh)\nz≥2.8",
           f"σ(n_s) inflation frac\n@σ_a0={SIGMA_A0_TRIALS}"]
    cells = []
    for r, dr in zip(rows, dom_rows):
        dstr = "/".join(f"{dr['infl'][s0]:.2f}" for s0 in SIGMA_A0_TRIALS)
        cells.append([r["name"], f"{r['cos_a_ns']:+.3f}", f"{r['cos_a_dtau']:+.3f}",
                      f"{r['cos_a_aHCD']:+.3f}", f"{r['cos_a_emu_allz']:+.3f}",
                      f"{r['cos_a_emu_hiZ']:+.3f}", dstr])
    tab = ax_tab.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.0)
    ax_tab.set_title("res_corr Fisher v2 — whitened (C_data⁻¹) cosines + 2-param prior-dominance "
                     f"(anchor {ANCHOR_MULT}×k_box; fiducial n_s={ns_phys_fid:.3f})",
                     fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"\nsaved figure -> {FIG}")

    # ---- printed summary tables -------------------------------------------
    print("\n================ SUMMARY: cosines (whitened C_data^-1) ================")
    print(f"{'leg':<7} {'cos(a,ns)':>10} {'cos(a,dtau0)':>13} {'cos(a,aHCD)':>12} "
          f"{'cos(a,emu)allz':>15} {'cos(a,emu)z>=2.8':>17} {'infl ns':>9}")
    for r in rows:
        print(f"{r['name']:<7} {r['cos_a_ns']:>+10.4f} {r['cos_a_dtau']:>+13.4f} "
              f"{r['cos_a_aHCD']:>+12.4f} {r['cos_a_emu_allz']:>+15.4f} "
              f"{r['cos_a_emu_hiZ']:>+17.4f} {r['infl_anc']:>9.3f}")
    print("\n  (anchoring check: cos(alpha,n_s) ANCHORED vs RAW res_corr)")
    for r in rows:
        print(f"   {r['name']:<7} anchored={r['cos_a_ns']:+.4f}  raw-res_corr={r['cos_a_ns_RAWres']:+.4f}  "
              f"delta={r['cos_a_ns']-r['cos_a_ns_RAWres']:+.4f}")

    print("\n========== SUMMARY: prior-dominance of sigma(n_s) [inflation frac sig_free/sig_fixed-1] ==========")
    print(f"{'leg':<7} " + " ".join(f"{'sa0='+str(s0):>10}" for s0 in SIGMA_A0_TRIALS))
    for dr in dom_rows:
        print(f"{dr['name']:<7} " + " ".join(f"{dr['infl'][s0]:>10.4f}" for s0 in SIGMA_A0_TRIALS))


if __name__ == "__main__":
    main()
