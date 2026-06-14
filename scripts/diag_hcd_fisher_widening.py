#!/usr/bin/env python3
"""Gate B of the HCD-class coherent C_emu build (#9): does a candidate coherent term widen
σ_subDLA by the RIGHT SIZE to explain the −1 to −3σ under-coverage? — GLS-Fisher, NO NUTS.

⚠️ CORRECTION (referee panel wf_f0b4dcfe-16c, 2026-06-13): this single-fiducial Fisher
OVER-ESTIMATES σ_like(α_subDLA) by ~1.8× — it reports σ_like≈0.051 (DESI) and concludes
"prior-dominated", but the ACTUAL NUTS chains (checkpoints/stepA/emucoh_validation.json) show
α_subDLA posterior off_sd≈0.021 < prior_sd 0.036, i.e. α_subDLA is mildly LIKELIHOOD-INFORMED.
The real defect is a posterior-MEAN bias (subDLA↔DLA degeneracy), not prior-projection. The
"prior-dominated" verdict branch below is therefore UNRELIABLE — trust the NUTS posterior sd,
not this Fisher's σ_like. The do-not-build conclusion still holds (a covariance term only widens,
can't move a biased mean; and post_sd<prior_sd caps the widening). Lesson: cross-check a Fisher
forecast against existing NUTS chains before drawing a prior-vs-likelihood conclusion.


Gate A confirmed the subDLA-class emulator residual IS k-coherent (cross-k |corr|≈0.55, one
mode = 60% of variance) — structure the diagonal-in-k C_emu misses. Gate B closes the loop:
at a fiducial truth on the KS leg, build the GLS Fisher F = Jᵀ C⁻¹ J over the active params
(J = ∂P_obs/∂θ by autodiff) and read σ_subDLA = sqrt[(F⁻¹)_subDLA] with C_emu DIAGONAL-only
(OFF) vs DIAGONAL + the candidate COHERENT term (ON). The candidate term is the cache-grid
global 4-class Gram (Gate A's pooled per-(sim,z) coherent residual) bound to the leg and scaled
by the fiducial per-class power coef_c·coef_c'·P_c⊗P_c' (the fixed-α Path-A form).

A fixed PSD add can ONLY widen σ_subDLA (PSD-monotone). The decisive question is the MAGNITUDE:
the prior closure had |bias_z(subDLA)| ≈ 1–3σ (RMS ~2). If the diagonal model under-estimates
σ_subDLA by a factor f = σ_on/σ_off, then the TRUE bias_z is ~bias_z_diag / f. So to bring a
2σ miscoverage down to ≲1σ we need f ≳ 2. PASS iff σ_on/σ_off is in/above the range implied by
the observed miscoverage (≳1.5–2×); a token widening (f≈1.05) would mean the term is real but
NOT the dominant cause → escalate.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_hcd_fisher_widening.py
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "/home/mfho/hcd_priya")
import hcd_analysis.emulator  # noqa: F401  x64 before jax
import jax
import jax.numpy as jnp
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator.predict import predict_P_filt

GATEA_NPZ = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator/hcd_emu_kcoherence.npz"
CLASSES = ["clean", "LLS", "subDLA", "DLA"]


def binder_S(leg, zc, kc, z_tol=0.1):
    """S (N, Nz_c·Nk_c): nearest-cache-z × log-k-interp binder (the mf_shape_cov_for_leg internals)."""
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]
    k_row = np.asarray(leg.k)
    N = len(k_row); Nz_c, Nk_c = len(zc), len(kc)
    S = np.zeros((N, Nz_c * Nk_c))
    lk = np.log(np.asarray(kc))
    for i in range(N):
        zi = int(np.argmin(np.abs(zc - z_row[i])))
        if abs(float(zc[zi]) - float(z_row[i])) > z_tol:
            continue
        x = np.log(max(k_row[i], 1e-12))
        if x <= lk[0]:
            w = np.zeros(Nk_c); w[0] = 1.0
        elif x >= lk[-1]:
            w = np.zeros(Nk_c); w[-1] = 1.0
        else:
            j = int(np.searchsorted(lk, x) - 1)
            t = (x - lk[j]) / (lk[j + 1] - lk[j])
            w = np.zeros(Nk_c); w[j] = 1 - t; w[j + 1] = t
        S[i, zi * Nk_c:(zi + 1) * Nk_c] = w
    return S


def main():
    # --- Gate A pool → global 4-class fractional Gram on the (zu,kb) cache grid ---
    g = np.load(GATEA_NPZ, allow_pickle=True)
    coh = np.asarray(g["coh"])               # (4, n_sim, nz, nk) per-(sim,z) coherent residual
    kb = np.asarray(g["kb"]); zu = np.asarray(g["zu"])
    n_cls, n_sim, nz, nk = coh.shape
    # stacked per-sim vector V[sim] ∈ R^{4·nz·nk}  (class-major, then z, then k)
    V = np.nan_to_num(coh).transpose(1, 0, 2, 3).reshape(n_sim, n_cls * nz * nk)
    F_full = (V.T @ V) / n_sim               # (M,M) PSD global second moment (uncentered, emucoh-style)
    F_full = 0.5 * (F_full + F_full.T)
    M = nz * nk
    # per-(c,c') block F_cc'  (nz*nk square)
    Fb = F_full.reshape(n_cls, M, n_cls, M).transpose(0, 2, 1, 3)   # (4,4,M,M)
    print(f"global Gram M={n_cls*M} (4×{M}); min-eig={np.linalg.eigvalsh(F_full).min():.2e} (PSD)")

    # --- legs + fiducial forward (held-out fold-6 sim truth, PRIYA τ₀, like the closure) ---
    LEG_NAME = sys.argv[1] if len(sys.argv) > 1 else "DESI"   # the emucoh miscoverage was on DESI arms
    ctx, d = C.build_legb_ctx(ckpt=f"{C.REPO}/checkpoints/final_fold6")
    # the closure α_subDLA prior σ (native units) — for the prior-vs-likelihood comparison
    import numpy as _np
    from hcd_analysis.emulator.inference import hcd_incidence_prior
    wc_med = _np.nanmedian(d["w_c_cache"][:, 1:], axis=0)
    amu, asd = hcd_incidence_prior(jnp.asarray(wc_med), z=3.0)
    prior_sigma = float(_np.asarray(asd)[1])                  # α_subDLA prior σ (native units)
    prior_mu = float(_np.asarray(amu)[1])
    leg = next(l for l in ctx.legs if l.name == LEG_NAME)
    sim = C.held_out_sims(d, fold=6)[0][0]
    truth = C.make_truth_from_sim(d, sim, fold=6, tau0_anchor="priya")
    theta9 = jnp.asarray(truth["params_unit"])
    # τ₀ on the leg z (nearest sim z) + α at the closure prior center (sim-mean w_c)
    z_sim = np.asarray(truth["z"]); tau0_sim = np.asarray(truth["tau0"])
    tau0_leg = jnp.asarray([tau0_sim[int(np.argmin(np.abs(z_sim - zz)))] for zz in np.asarray(leg.z)])
    alpha0 = jnp.asarray(np.nanmedian(d["w_c_cache"][:, 1:], axis=0))    # (3,) sim-mean incidence
    cache_k = jnp.asarray(ctx.cache_k)
    pf = ctx.pf_stats
    rho_leg = ctx.rho_zb_per_leg[LEG_NAME]
    alpha_centres = ctx.alpha_centres
    _core = np.asarray(ctx.dla_core_leg[LEG_NAME])           # (n_z,Kc) or (Kc,)
    dla_core = jnp.asarray(_core.mean(axis=0) if _core.ndim > 1 else _core)   # (Kc,) z-mean core

    # forward → (P_model, C_total) with the PRODUCTION diagonal cross-class C_emu (rho_zb)
    def fwd(th9, al):
        return DL.predict_P_obs_on_leg(
            ctx.model, th9, tau0_leg, al, pf_stats=pf, dla_core=dla_core,
            cache_k=cache_k, leg=leg, alpha_centres=alpha_centres, rho_zb=rho_leg)
    P_model, C_off = fwd(theta9, alpha0)
    P_model = np.asarray(P_model); C_off = np.asarray(C_off)
    finite = np.isfinite(P_model) & np.isfinite(np.diag(C_off))
    idx = np.where(finite)[0]
    N = len(idx)
    print(f"{LEG_NAME} leg: {leg.P_data.shape[0]} rows, {N} finite; sim={sim}")

    # --- Jacobian J = ∂P_obs/∂[θ9(9), α(3)] (autodiff) on the finite rows ---
    def Ponly(params):
        th9 = params[:9]; al = params[9:]
        P, _ = fwd(th9, al)
        return P[jnp.asarray(idx)]
    params0 = jnp.concatenate([theta9, alpha0])
    J = np.asarray(jax.jacrev(Ponly)(params0))          # (N, 12)  ns=theta9[0], Ap=theta9[1]

    # --- per-class power on the leg (for the coherent amplitude P_c⊗P_c') ---
    P_cls_leg = np.zeros((4, leg.P_data.shape[0]))
    for zi in range(leg.n_z):
        zu_leg = float(leg.z[zi]); z_unit = (zu_leg - 2.0) / 3.4
        Pf = np.asarray(predict_P_filt(ctx.model, theta9, jnp.asarray(z_unit), tau0_leg[zi], pf))  # (4,Kc)
        rows = np.where(np.asarray(leg.z_idx) == zi)[0]
        ksub = np.asarray(leg.k)[rows]
        Pcls = np.stack([Pf[0], Pf[1], Pf[2], Pf[3] + np.asarray(dla_core)])      # (4,Kc)
        for c in range(4):
            P_cls_leg[c, rows] = np.interp(ksub, np.asarray(cache_k), Pcls[c])

    # --- bind the global Gram to the leg and assemble the fixed-α coherent term ---
    # ⚠️ AMPLITUDE-BASIS CAVEAT (code review 2026-06-13): this scales each coherent block by the
    # EMULATED per-class power P_cls_leg (at the fixed fiducial θ), NOT the production θ-INDEPENDENT
    # leg.P_data (data_likelihood.py:841, whose CRITICAL note flags live-P scaling as a pathology).
    # At a FIXED fiducial P_cls_leg is a constant matrix, so it does NOT corrupt this gate's verdict —
    # which rests on the OFF-only σ_like-vs-σ_prior comparison, independent of the coherent amplitude.
    # But the f_post MAGNITUDE depends on this scaling, so DO NOT reuse this assembly to SIZE a real
    # production C_emu term; a real term must use leg.P_data (and the corrected mechanism: a covariance
    # term can't fix the posterior-MEAN bias anyway — see the header correction note).
    S = binder_S(leg, zu, kb)                                     # (Ntot, nz*nk)
    coef = np.concatenate([[1.0 - float(np.sum(alpha0))], np.asarray(alpha0)])   # (4,)
    Ntot = leg.P_data.shape[0]
    C_coh_full = np.zeros((Ntot, Ntot))
    for a in range(4):
        for b in range(4):
            block = S @ Fb[a, b] @ S.T                            # (Ntot,Ntot) fractional cov of (ε_a,ε_b)
            amp = coef[a] * coef[b]
            C_coh_full += amp * block * (P_cls_leg[a][:, None] * P_cls_leg[b][None, :])
    C_coh = C_coh_full[np.ix_(idx, idx)]
    C_coh = 0.5 * (C_coh + C_coh.T)
    # PSD check
    emin = np.linalg.eigvalsh(C_coh).min()
    print(f"coherent term on leg: min-eig={emin:.2e} (PSD by construction); "
          f"frac diag added (median) = {np.median(np.diag(C_coh)/np.clip(np.diag(C_off),1e-300,None)):.3f}")

    C_off_f = C_off[np.ix_(idx, idx)]
    C_on_f = C_off_f + C_coh

    # --- GLS Fisher σ for each param, OFF vs ON ---
    def sigmas(Ctot):
        Cinv = np.linalg.inv(Ctot)
        Fmat = J.T @ Cinv @ J
        Finv = np.linalg.inv(Fmat + 1e-12 * np.eye(Fmat.shape[0]))
        return np.sqrt(np.clip(np.diag(Finv), 0, None))
    s_off = sigmas(C_off_f)
    s_on = sigmas(C_on_f)
    names = [f"theta9_{i}" for i in range(9)] + ["alpha_LLS", "alpha_subDLA", "alpha_DLA"]
    i_sub = names.index("alpha_subDLA")
    i_ns = 0
    print("\n=== GLS-Fisher σ widening (OFF=diagonal C_emu → ON=+coherent) ===")
    for nm, so, sn in zip(names, s_off, s_on):
        star = "  <<<" if nm == "alpha_subDLA" else ("   <ns" if nm == "theta9_0" else "")
        print(f"  {nm:14s}: σ_off={so:.4e}  σ_on={sn:.4e}  ratio={sn/so:6.3f}{star}")

    # likelihood σ_subDLA OFF/ON, and the POSTERIOR σ (likelihood ⊕ Normal prior) — the closure's
    # actual σ. If the data barely constrains α_subDLA (σ_like >> σ_prior), the posterior is
    # prior-dominated and NO covariance term can move it → the miscoverage is prior-projection.
    sl_off, sl_on = s_off[i_sub], s_on[i_sub]
    def post_sigma(sl):
        return 1.0 / np.sqrt(1.0 / sl**2 + 1.0 / prior_sigma**2)
    sp_off, sp_on = post_sigma(sl_off), post_sigma(sl_on)
    f_like = sl_on / sl_off
    f_post = sp_on / sp_off
    f_ns = s_on[i_ns] / s_off[i_ns]
    print(f"\n=== GATE B VERDICT  (leg={LEG_NAME}) ===")
    print(f"  α_subDLA: prior σ={prior_sigma:.4f} (μ={prior_mu:.4f}), likelihood σ_off={sl_off:.4f}")
    print(f"    → data {'CONSTRAINS' if sl_off < prior_sigma else 'does NOT constrain'} α_subDLA "
          f"(σ_like/σ_prior={sl_off/prior_sigma:.1f})")
    print(f"  likelihood widening f_like={f_like:.3f};  POSTERIOR σ {sp_off:.4f}→{sp_on:.4f} "
          f"(f_post={f_post:.3f})")
    print(f"  → a posterior bias_z of ~2σ becomes ~{2/f_post:.2f}σ under the coherent C_emu")
    print(f"  n_s likelihood widening = {f_ns:.2f} (should be MODEST)")
    if sl_off >= prior_sigma:
        print("  ❌ PRIOR-DOMINATED: the data barely constrains α_subDLA → its posterior IS the prior.")
        print("     The miscoverage is PRIOR-PROJECTION (prior center offset from the per-sim truth),")
        print("     NOT diagonal-C_emu overconfidence. A coherent covariance term CANNOT fix it →")
        print("     escalate to a PRIOR/parameterization fix; do NOT spend the NUTS sweep on the term.")
    elif f_post >= 1.5:
        print("  ✅ PASS: data constrains α_subDLA AND the coherent term widens the POSTERIOR enough")
        print("     to explain the 1–3σ miscoverage → BUILD the production term + sweep.")
    elif f_post >= 1.15:
        print("  ⚠️  PARTIAL: real but sub-dominant posterior widening — helps but won't fully cover.")
    else:
        print("  ❌ WEAK: σ_subDLA barely moves → not diagonal-C_emu overconfidence; escalate to a")
        print("     prior/identifiability fix, do NOT spend the NUTS sweep on the term.")


if __name__ == "__main__":
    main()
