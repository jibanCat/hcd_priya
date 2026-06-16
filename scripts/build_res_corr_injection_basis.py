"""BUILD (scratch): the OUT-OF-SPAN res_corr injection basis for the Phase-2
res_corr injection-recovery gate.

WHY
---
We marginalize the res_corr amplitude with a 2-param nuisance
    alpha(z) = alpha0 * ((1+z)/(1+3))^s
applied to the ANCHORED log res_corr deltahat(k,z)
(deltahat = log(interp_res_corr(..., anchor_mult=5.0)) -> forced ->0 below
5*k_box(z); the committed default). The injection gate must inject a res_corr
MISSPECIFICATION that the 2-param alpha CANNOT represent (out-of-span) AND is
concentrated in the He-II window z>=2.8 -- otherwise the gate passes vacuously.

To first order, every alpha(z)=alpha0*((1+z)/4)^s perturbs the log res_corr by
    alpha(z) * deltahat(k,z)
    = alpha0 * deltahat(k,z) + alpha0*s * deltahat(k,z)*log((1+z)/4) + O(s^2),
so the 2-D linear span the nuisance can absorb (to first order in s about s=0,
alpha0 about 0) is
    v1 = deltahat(k,z)
    v2 = deltahat(k,z) * log((1+z)/(1+3))           (the z-slope direction).
Anything ORTHOGONAL to span{v1,v2} in the data metric is out-of-span: the
nuisance provably cannot reshape the forward into it. We build the top-2 such
directions, restricted to / localized in z>=2.8 high-k, scaled to the +-5%
He-II envelope (the real table deficit magnitude there), as the injection basis.

WHAT (per leg DESI/KS/eBOSS, in the C_data^-1 whitened metric)
--------------------------------------------------------------
1. alpha-SPAN  V = [v1, v2]  on the leg (z,k) grid.
2. R(k,z) = anchored log res_corr restricted to z>=2.8 high-k cells (He-II
   window) = the plausible-true misspecification raw material. (+ an optional
   class-differential DLA-only-high-k variant, reported, not the primary basis.)
3. ORTHOGONALIZE R against span{V} in <a,b> = a^T C^-1 b (project out v1, v2),
   restrict to z>=2.8, SVD -> top-2 directions b1,b2 = the injection basis.
4. SCALE each b_i so |exp(b)-1| ~ 0.05 at the high-k z~3-4 cells.
5. ASSERT cos(b_i, span{V}) < 0.8 (must be ~0 by construction); report
   cos(b_i, n_s-response) and pre-select the worst-n_s-projecting member.
6. SAVE per-leg b1,b2 (on the leg z,k grid) + metadata to
   _emulator_data/res_corr_injection_basis.npz.
7. FIGURE -> notes/.../res_corr_injection_basis.png.

Reuses the leg-loading + whitening helpers / conventions of
scripts/scratch_res_corr_ns_fisher.py (anchor_taper, _whitened_cos, the
PRODUCTION MF n_s response, the Z_HEII=2.8 window).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/build_res_corr_injection_basis.py
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
from hcd_analysis.emulator.data_likelihood import _predict_P_obs_mf
from hcd_analysis.emulator.multifidelity import interp_res_corr
from hcd_analysis.emulator.data import SAMPLING_LIMITS, normalize_params

OUT_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/res_corr_injection_basis.npz"
FIG = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator/res_corr_injection_basis.png"

# --- k_box anchor (L15 box fundamental in s/km) -- same as the Fisher pre-check & interp_res_corr
H0, OM, HUB = 70.0, 0.3, 0.7
def Hz(z):
    return H0 * np.sqrt(OM * (1.0 + z) ** 3 + (1.0 - OM))
def k_box(z):
    return (2.0 * np.pi / 15.0) * HUB * (1.0 + z) / Hz(z)

ANCHOR_MULT = 5.0          # committed default: res_corr ->1 below 5*k_box(z)
TAPER_WIDTH_DEX = 0.12
Z_HEII = 2.8               # He-II window lower edge
Z_PIVOT_SLOPE = 3.0        # the alpha(z) pivot (1+z)/(1+3)
ENVELOPE = 0.05            # +-5% He-II envelope target: |exp(b)-1| ~ 0.05 at hi-k z~3-4
COS_GATE = 0.8             # assert cos(b_i, span{alpha}) < this


def anchor_taper(k, z):
    """Smooth tanh low-k taper -> 0 below 5*k_box(z), 1 above (width 0.12 dex)."""
    k = np.asarray(k, float)
    kc = ANCHOR_MULT * k_box(z)
    x = (np.log10(k) - np.log10(kc)) / TAPER_WIDTH_DEX
    return 0.5 * (1.0 + np.tanh(x))


def _whitened_inner(Cinv):
    """Return <a,b> = a^T Cinv b and the induced norm/cosine closures."""
    def inner(a, b):
        return float(a @ Cinv @ b)
    def norm(a):
        return float(np.sqrt(max(a @ Cinv @ a, 0.0)))
    def cos(a, b):
        den = norm(a) * norm(b)
        return float(inner(a, b) / den) if den > 0 else float("nan")
    return inner, norm, cos


def _gram_schmidt_whitened(vecs, Cinv, tol=1e-12):
    """C^-1-orthonormal basis of span{vecs} (drops near-degenerate columns)."""
    inner, norm, _ = _whitened_inner(Cinv)
    basis = []
    for v in vecs:
        u = np.array(v, float).copy()
        for q in basis:
            u = u - inner(u, q) * q
        nu = norm(u)
        if nu > tol:
            basis.append(u / nu)
    return basis


def _project_out(R, ortho_basis, Cinv):
    """Remove the components of R along the C^-1-orthonormal ortho_basis."""
    inner, _, _ = _whitened_inner(Cinv)
    out = np.array(R, float).copy()
    for q in ortho_basis:
        out = out - inner(out, q) * q
    return out


def main():
    print("=== building production Leg-B ctx (MF forward, DESI+KS+eBOSS) ===")
    ctx, _ = build_legb_ctx(with_mf=True, with_eboss=True)
    model = ctx.model
    pf_stats = ctx.pf_stats
    mf = ctx.mf
    cache_k = np.asarray(ctx.cache_k)
    z_pivot_tau0 = float(ctx.tau0_pivot_z)

    # fiducial theta = sampling prior center in the unit cube (matches the Fisher pre-check)
    lim = SAMPLING_LIMITS
    theta_phys_fid = 0.5 * (lim[:, 0] + lim[:, 1])
    theta_unit_fid = jnp.asarray(normalize_params(theta_phys_fid))
    ns_phys_fid = float(theta_phys_fid[0])

    z_global = np.asarray(ctx.z_global)
    tau0_mu = np.asarray(ctx.tau0_mu)
    alpha_hcd_fid = jnp.asarray(ctx.alpha_hcd_mu)
    z_rc, logk_rc, rc_vals = np.asarray(mf.z_rc), np.asarray(mf.logk_rc), np.asarray(mf.rc_vals)

    print(f"fiducial n_s={ns_phys_fid:.4f}; ANCHOR_MULT={ANCHOR_MULT}; "
          f"Z_HEII={Z_HEII}; envelope=+-{ENVELOPE*100:.0f}%")

    out = {}                 # npz payload
    rows = []                # per-leg report rows
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2))

    for col, leg in enumerate(ctx.legs):
        name = leg.name
        dla_core = ctx.dla_core_leg[name]
        k_leg = np.asarray(leg.k)                  # (N,) flat z-major angular k
        z_idx = np.asarray(leg.z_idx)
        z_leg = np.asarray(leg.z)                   # (n_z,)
        C_data = np.asarray(leg.C_data)
        N = k_leg.shape[0]
        Cinv = np.linalg.inv(C_data)
        inner, norm, wcos = _whitened_inner(Cinv)

        # ---------- alpha-span on the leg grid: v1, v2 ----------
        # v1 = anchored log res_corr deltahat(k,z); v2 = v1 * log((1+z)/(1+3))
        deltahat = np.zeros(N)
        logfac = np.zeros(N)                          # log((1+z)/(1+3)) per flat cell
        for iz in range(leg.n_z):
            rsel = np.where(z_idx == iz)[0]
            if rsel.size == 0:
                continue
            zz = float(z_leg[iz])
            rc = np.asarray(interp_res_corr(z_rc, logk_rc, rc_vals, zz,
                                            jnp.asarray(k_leg[rsel]),
                                            anchor_mult=ANCHOR_MULT))
            deltahat[rsel] = np.log(rc)               # already anchored (rc->1 below 5kbox)
            logfac[rsel] = np.log((1.0 + zz) / (1.0 + Z_PIVOT_SLOPE))
        v1 = deltahat.copy()
        v2 = deltahat * logfac
        V = [v1, v2]

        # ---------- n_s production response (AD) ----------
        z_leg_j = jnp.asarray(z_leg)
        tau0_fid_np = np.array([tau0_mu[int(np.argmin(np.abs(z_global - zz)))] for zz in z_leg])
        rest0 = theta_unit_fid[1:]

        def _logP_ns(ns_unit):
            theta = jnp.concatenate([ns_unit[None], rest0])
            P = jnp.zeros(N)
            for iz in range(leg.n_z):
                rsel = np.where(z_idx == iz)[0]
                if rsel.size == 0:
                    continue
                z_unit = float(leg.z_unit[iz])
                P_cache = _predict_P_obs_mf(mf, model, theta, z_unit,
                                            jnp.asarray(tau0_fid_np[iz]),
                                            alpha_hcd_fid, pf_stats, dla_core[iz])
                P_z = jnp.interp(jnp.asarray(k_leg[rsel]), jnp.asarray(cache_k), P_cache)
                P = P.at[jnp.asarray(rsel)].set(P_z)
            return jnp.log(P)

        r_ns = np.asarray(jax.jacfwd(_logP_ns)(theta_unit_fid[0]))

        # ---------- He-II window mask (z>=2.8) ----------
        zrow = z_leg[z_idx]                           # (N,) z per flat cell
        hiZ = zrow >= Z_HEII

        # ---------- R(k,z): plausible-true misspecification raw material ----------
        # the anchored log res_corr restricted to the z>=2.8 high-k cells (already anchored,
        # so it's automatically ~0 in the low-k anchored region; the He-II window just zeroes
        # the z<2.8 rows). This is the table's REAL high-k deficit shape in the He-II band.
        R_global = np.where(hiZ, deltahat, 0.0)

        # --- WHY a 2-column [deltahat, deltahat*z-slope] raw set is NOT enough for rank 2 ---
        # alpha(z)=alpha0*((1+z)/4)^s rescales deltahat's FIXED k-shape by a smooth z-function
        # (alpha0 + alpha0*s*logfac + ...). So span{v1,v2} = deltahat * {1, logfac}. Any raw
        # material that is ALSO deltahat times a smooth-in-z weight (the class-diff dla_frac, the
        # raw table shape) collapses, after projecting out span{V}, onto a SINGLE out-of-span
        # direction (the z-residual of that weight) -- the 2nd singular value vanishes (verified:
        # eBOSS S=[2.8e-3, 5e-18]). The directions alpha PROVABLY cannot make are the ones that
        # change the per-cell SHAPE of the perturbation, not just its smooth-z amplitude:
        #   (a) a DIFFERENT high-k k-tilt than deltahat carries (steepen/flatten the He-II deficit
        #       toward k_max), and
        #   (b) z-CURVATURE of the deficit (a quadratic-in-logfac modulation -- alpha's
        #       (1+z)^s is, to the first order we span, only LINEAR in logfac).
        # We assemble a small dictionary of such He-II misspecification shapes, project ALL out of
        # span{V|hiZ}, SVD, and keep the top-2 with non-negligible singular value -> a genuine
        # rank-2 out-of-span basis.

        # k-tilt direction: deltahat reweighted by a rising-with-k centred ramp (changes the high-k
        # k-shape of the deficit; alpha cannot, it's k-independent).
        logk = np.log10(np.maximum(k_leg, 1e-12))
        lk_lo, lk_hi = logk[hiZ].min(), logk[hiZ].max()
        ktilt = (logk - 0.5 * (lk_lo + lk_hi)) / max(lk_hi - lk_lo, 1e-9)   # ~[-0.5,0.5] over hiZ k
        R_ktilt = np.where(hiZ, deltahat * ktilt, 0.0)

        # z-curvature direction: deltahat reweighted by (logfac^2 - <logfac^2>) so it is the part of
        # a quadratic z-modulation that the LINEAR logfac (v2) cannot reach.
        lf2 = logfac ** 2
        if hiZ.any():
            lf2 = lf2 - lf2[hiZ].mean()
        R_zcurv = np.where(hiZ, deltahat * lf2, 0.0)

        # class-differential variant (DLA-class high-k only): res_corr is a GLOBAL multiplicative
        # correction in the forward (not class-resolved), so a class-differential injection is
        # approximated by the FRACTION of P at z>=2.8 hi-k carried by the DLA class (dla_core/P),
        # weighting deltahat. (Reported; contributes to the dictionary.)
        dla_frac = np.zeros(N)
        P0 = np.exp(np.asarray(_logP_ns(theta_unit_fid[0])))
        for iz in range(leg.n_z):
            rsel = np.where(z_idx == iz)[0]
            if rsel.size == 0:
                continue
            dcore_k = np.asarray(jnp.interp(jnp.asarray(k_leg[rsel]),
                                            jnp.asarray(cache_k), dla_core[iz]))
            dla_frac[rsel] = dcore_k / np.maximum(P0[rsel], 1e-30)
        R_classdiff = np.where(hiZ, deltahat * dla_frac, 0.0)

        # The dictionary of plausible He-II misspecifications (raw, pre-projection). R_global goes
        # FIRST so its out-of-span part dominates the leading SVD direction (the most physical
        # "the real high-k deficit doesn't follow alpha" injection).
        R_dict = [R_global, R_ktilt, R_zcurv, R_classdiff]

        # ---------- orthogonalize + SVD ENTIRELY INSIDE the He-II (z>=2.8) subspace ----------
        # Localization (b lives in z>=2.8) and out-of-span (b _|_ span{V} in C^-1) only co-hold
        # EXACTLY if both are enforced in the SAME inner-product space. span{V} has support outside
        # z>=2.8, so projecting in the full-N metric and then masking to z>=2.8 RE-INTRODUCES an
        # in-span component (the masked tail no longer cancels). The clean construction (and what
        # step 3 literally asks -- "SVD of the residual-after-projection restricted to z>=2.8") is
        # to restrict to the He-II sub-block C_sub = C_data[hiZ,hiZ] and do projection + SVD there.
        # The result is a hiZ-supported vector that is C_sub^-1-orthogonal to span{V|hiZ}; embedded
        # back to full N (zeros off-hiZ) it is exactly orthogonal to span{V} in the same restricted
        # metric we report cos in (the reference Fisher script also reports the z>=2.8 cosines using
        # exactly this C_sub^-1 sub-metric, so the gate's "<0.8 in C_data^-1 whitened" is evaluated
        # consistently on the He-II window).
        sub = np.where(hiZ)[0]
        C_sub = C_data[np.ix_(sub, sub)]
        Cinv_sub = np.linalg.inv(C_sub)
        inner_s, norm_s, wcos_s = _whitened_inner(Cinv_sub)

        V_sub = [v1[sub], v2[sub]]
        ortho_V_sub = _gram_schmidt_whitened(V_sub, Cinv_sub)   # C_sub^-1-orthonormal span{V|hiZ}

        # project EACH dictionary shape out of span{V|hiZ}, stack the whitened columns, SVD.
        L_s = np.linalg.cholesky(Cinv_sub)           # Cinv_sub = L_s L_s^T
        cols = []
        for R in R_dict:
            Rp = _project_out(R[sub], ortho_V_sub, Cinv_sub)
            cols.append(L_s.T @ Rp)                  # whitened, out-of-span column
        M_white = np.column_stack(cols)              # (n_sub, n_dict) whitened
        U, S, Vt = np.linalg.svd(M_white, full_matrices=False)
        # keep the top-2 directions with non-negligible singular value (rank guard); both lie in the
        # column space of M_white => both are EXACTLY C_sub^-1-orthogonal to span{V|hiZ}.
        smax = S[0] if S.size else 0.0
        rank = int(np.sum(S > 1e-9 * max(smax, 1e-30)))
        if rank < 2:
            print(f"   [WARN] {name}: out-of-span residual rank={rank} (<2); "
                  f"S={np.array2string(S, precision=2)} -- b2 may be weakly determined")
        Linv_T_s = np.linalg.inv(L_s.T)
        raw_b = []
        for j in range(2):
            bj_sub = Linv_T_s @ U[:, j]               # data-space direction in the hiZ subspace
            bj = np.zeros(N)
            bj[sub] = bj_sub                          # embed back to full N (zero off-hiZ)
            raw_b.append(bj)
        print(f"   out-of-span SVD singular values (top {min(4,S.size)}): "
              f"{np.array2string(S[:4], precision=3)}  (rank={rank})")

        # ---------- scale each b_i to the +-5% He-II envelope ----------
        # target: max|b| over the z~3-4 high-k cells == ENVELOPE (so |exp(b)-1| ~ ENVELOPE there)
        z34 = (zrow >= 3.0) & (zrow <= 4.0)
        scaled_b = []
        for bj in raw_b:
            ref = bj[z34] if z34.any() else bj[hiZ]
            peak = np.max(np.abs(ref)) if ref.size else np.max(np.abs(bj))
            if peak <= 0:
                scaled_b.append(bj)
                continue
            # sign: make the largest-|amp| z34 cell NEGATIVE (a high-k DEFICIT, matching the real
            # table deficit there); cosines/orthogonality are sign-invariant.
            s = ENVELOPE / peak
            bj_s = bj * s
            imax = np.argmax(np.abs(bj_s * z34)) if z34.any() else np.argmax(np.abs(bj_s))
            if bj_s[imax] > 0:
                bj_s = -bj_s
            scaled_b.append(bj_s)
        b1, b2 = scaled_b

        # ---------- assertions + cosines (in the He-II z>=2.8 C_sub^-1 sub-metric) ----------
        # cos(b_i, span{V}) = norm of the in-span component of b_i over its norm, all on the
        # z>=2.8 sub-block (the window the basis lives in and the gate's "<0.8 whitened" applies
        # to -- the reference Fisher script reports its z>=2.8 cosines in exactly this sub-metric).
        def cos_to_span_sub(b):
            bs = b[sub]
            proj = np.zeros_like(bs)
            for q in ortho_V_sub:
                proj = proj + inner_s(bs, q) * q
            nb = norm_s(bs)
            return float(norm_s(proj) / nb) if nb > 0 else float("nan")

        cos_b1_span = cos_to_span_sub(b1)
        cos_b2_span = cos_to_span_sub(b2)
        cos_b1_ns = wcos_s(b1[sub], r_ns[sub])
        cos_b2_ns = wcos_s(b2[sub], r_ns[sub])

        # max realized envelope (|exp(b)-1|) over z34 hi-k, for the report
        env_b1 = float(np.max(np.abs(np.expm1(b1[z34]))) if z34.any() else np.max(np.abs(np.expm1(b1))))
        env_b2 = float(np.max(np.abs(np.expm1(b2[z34]))) if z34.any() else np.max(np.abs(np.expm1(b2))))

        worst = 1 if abs(cos_b1_ns) >= abs(cos_b2_ns) else 2     # 1-indexed worst-n_s member
        worst_cos = cos_b1_ns if worst == 1 else cos_b2_ns

        assert cos_b1_span < COS_GATE, f"[{name}] cos(b1,span)={cos_b1_span:.4f} !< {COS_GATE}"
        assert cos_b2_span < COS_GATE, f"[{name}] cos(b2,span)={cos_b2_span:.4f} !< {COS_GATE}"

        print(f"\n[{name}] N={N}  z>=2.8 cells={int(hiZ.sum())}  "
              f"k[{k_leg.min():.4f},{k_leg.max():.4f}] z[{z_leg.min():.2f},{z_leg.max():.2f}]")
        print(f"   cos(b1,span alpha)={cos_b1_span:+.4f}  cos(b2,span alpha)={cos_b2_span:+.4f}"
              f"   (gate < {COS_GATE})  -> OUT-OF-SPAN OK")
        print(f"   cos(b1,n_s)={cos_b1_ns:+.4f}  cos(b2,n_s)={cos_b2_ns:+.4f}"
              f"   -> worst-n_s member = b{worst} (|cos|={abs(worst_cos):.4f})")
        print(f"   realized He-II envelope |exp(b)-1| (z~3-4 hi-k): b1={env_b1*100:.2f}%  b2={env_b2*100:.2f}%")

        rows.append(dict(name=name, cos_b1_span=cos_b1_span, cos_b2_span=cos_b2_span,
                         cos_b1_ns=cos_b1_ns, cos_b2_ns=cos_b2_ns, worst=worst,
                         worst_cos=worst_cos, env_b1=env_b1, env_b2=env_b2,
                         n_heii=int(hiZ.sum())))

        # ---------- npz payload (per leg) ----------
        out[f"{name}_b1"] = b1.astype(float)
        out[f"{name}_b2"] = b2.astype(float)
        out[f"{name}_k"] = k_leg.astype(float)
        out[f"{name}_z_row"] = zrow.astype(float)
        out[f"{name}_z_idx"] = z_idx.astype(int)
        out[f"{name}_z"] = z_leg.astype(float)
        out[f"{name}_hiZ_mask"] = hiZ.astype(bool)
        out[f"{name}_v1_span"] = v1.astype(float)
        out[f"{name}_v2_span"] = v2.astype(float)
        out[f"{name}_r_ns"] = r_ns.astype(float)
        out[f"{name}_cos_b_span"] = np.array([cos_b1_span, cos_b2_span])
        out[f"{name}_cos_b_ns"] = np.array([cos_b1_ns, cos_b2_ns])
        out[f"{name}_worst_ns_member"] = int(worst)   # 1 or 2 (b1 or b2)
        out[f"{name}_env_realized"] = np.array([env_b1, env_b2])

        # ---------- figure: per-leg response shapes vs k at representative z ----------
        ax = axes[0, col]
        zsel = [z for z in [3.0, 3.4, 3.8] if z >= z_leg.min() - 0.2 and z <= z_leg.max() + 0.2]
        if not zsel:
            zsel = [z_leg[len(z_leg) // 2]]
        alphas = np.linspace(0.45, 1.0, len(zsel))
        # rescale span/ns to b1 envelope for visual comparison
        s_v1 = ENVELOPE / max(np.max(np.abs(v1)), 1e-30)
        s_v2 = ENVELOPE / max(np.max(np.abs(v2)), 1e-30)
        s_ns = ENVELOPE / max(np.max(np.abs(r_ns)), 1e-30)
        for zz, al in zip(zsel, alphas):
            iz = int(np.argmin(np.abs(z_leg - zz)))
            rs = np.where(z_idx == iz)[0]
            if rs.size == 0:
                continue
            o = rs[np.argsort(k_leg[rs])]
            last = (zz == zsel[-1])
            ax.plot(k_leg[o], (v1 * s_v1)[o], "-", color="0.55", alpha=al, lw=1.0,
                    label=("v1=deltahat (span, rescaled)" if last else None))
            ax.plot(k_leg[o], (v2 * s_v2)[o], "--", color="0.35", alpha=al, lw=1.0,
                    label=("v2=z-slope (span, rescaled)" if last else None))
            ax.plot(k_leg[o], (r_ns * s_ns)[o], "-", color="C0", alpha=al * 0.8, lw=1.0,
                    label=("r_ns (rescaled)" if last else None))
            ax.plot(k_leg[o], b1[o], "-", color="C3", alpha=al, lw=2.0,
                    label=("b1 injection" if last else None))
            ax.plot(k_leg[o], b2[o], "-", color="C2", alpha=al, lw=2.0,
                    label=("b2 injection" if last else None))
        kc_lo = ANCHOR_MULT * k_box(z_leg.max())
        kc_hi = ANCHOR_MULT * k_box(z_leg.min())
        ax.axvspan(kc_lo, kc_hi, color="grey", alpha=0.15, label=f"{ANCHOR_MULT}·k_box(z)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k [s/km]")
        if col == 0:
            ax.set_ylabel("log-res_corr perturbation")
        ax.set_title(f"{name}  cos(b1,span)={cos_b1_span:+.2f} cos(b2,span)={cos_b2_span:+.2f}\n"
                     f"cos(b1,n_s)={cos_b1_ns:+.2f} cos(b2,n_s)={cos_b2_ns:+.2f}  "
                     f"(worst-n_s=b{worst})", fontsize=8.5)
        ax.legend(fontsize=6.0, loc="best", ncol=1)

        # ---------- figure bottom: z-localization heat (b1 over the z>=2.8 cells) ----------
        ax2 = axes[1, col]
        # scatter b1 amplitude in (k, z) to show He-II localization
        sc = ax2.scatter(k_leg, zrow, c=b1, cmap="RdBu_r",
                         vmin=-ENVELOPE, vmax=ENVELOPE, s=14)
        ax2.axhline(Z_HEII, color="k", ls=":", lw=1.0, label=f"z={Z_HEII} (He-II edge)")
        ax2.set_xscale("log")
        ax2.set_xlabel("k [s/km]")
        if col == 0:
            ax2.set_ylabel("z")
        ax2.set_title(f"{name}  b1 amplitude in (k,z)  (localized z>={Z_HEII})", fontsize=8.5)
        ax2.legend(fontsize=7, loc="lower left")
        fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.02, label="b1 (log-res_corr)")

    fig.suptitle("OUT-OF-SPAN res_corr injection basis (whitened C_data⁻¹; "
                 f"anchor {ANCHOR_MULT}×k_box; He-II z≥{Z_HEII}; ±{ENVELOPE*100:.0f}% envelope; "
                 f"fiducial n_s={ns_phys_fid:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG, dpi=130)
    print(f"\nsaved figure -> {FIG}")

    # ---------- save npz ----------
    out["_meta_anchor_mult"] = ANCHOR_MULT
    out["_meta_z_heii"] = Z_HEII
    out["_meta_z_pivot_slope"] = Z_PIVOT_SLOPE
    out["_meta_envelope"] = ENVELOPE
    out["_meta_cos_gate"] = COS_GATE
    out["_meta_legs"] = np.array([leg.name for leg in ctx.legs])
    out["_meta_doc"] = np.array(
        "Per-leg out-of-span res_corr injection basis. b1,b2 are multiplicative "
        "log-res_corr perturbations on the leg (z,k) grid (apply as P_truth*exp(b)). "
        "Built C_data^-1-orthogonal to span{v1=deltahat, v2=deltahat*log((1+z)/4)} "
        "(the 2-param alpha(z) nuisance span), localized to z>=2.8, scaled to +-5% at "
        "z~3-4 hi-k. worst_ns_member (1/2) = the b the gate should inject (max |cos(b,n_s)|)."
    )
    np.savez(OUT_NPZ, **out)
    print(f"saved basis -> {OUT_NPZ}")

    # ---------- printed report table ----------
    print("\n================ PER-LEG INJECTION-BASIS REPORT ================")
    print(f"{'leg':<7} {'cos(b1,span)':>12} {'cos(b2,span)':>12} "
          f"{'cos(b1,ns)':>11} {'cos(b2,ns)':>11} {'worst-ns':>9} {'env b1/b2 %':>14}")
    for r in rows:
        print(f"{r['name']:<7} {r['cos_b1_span']:>+12.4f} {r['cos_b2_span']:>+12.4f} "
              f"{r['cos_b1_ns']:>+11.4f} {r['cos_b2_ns']:>+11.4f} "
              f"{'b'+str(r['worst']):>9} {r['env_b1']*100:>6.1f}/{r['env_b2']*100:<6.1f}")
    allok = all(r["cos_b1_span"] < COS_GATE and r["cos_b2_span"] < COS_GATE for r in rows)
    print(f"\nASSERT cos(injection, span alpha) < {COS_GATE} for ALL b_i, ALL legs: "
          f"{'PASS' if allok else 'FAIL'}")
    print("Pre-selected gate injection (worst-n_s-projecting member) per leg:")
    for r in rows:
        print(f"   {r['name']:<7} -> b{r['worst']}  (cos(b{r['worst']},n_s)={r['worst_cos']:+.4f})")


if __name__ == "__main__":
    main()
