"""T3 ADOPTION GATE — all-folds n_s/A_p EMU bias THROUGH the MultiFidelity forward.

Clone of scripts/diag_emu_bias_allfolds_zlo24.py (the SHIPPED KS z_lo=2.4, KS<0.06 LF
baseline: n_s +0.030σ, A_p +0.034σ). The ONLY load-bearing change: the per-class P_filt
forward goes THROUGH the MF correction `g(z,τ₀,k) + log res_corr(z,k)` (resolved
separable + rank-1 FixedMeanHead + the fixed particle-convergence res_corr) instead of
the raw LF `predict_P_filt`. The LF backbone is FROZEN (only the fixed MF correction is
added). This is the contract of docs/superpowers/onboarding/2026-06-08-mf-gate-spec.md.

FAITHFUL through-MF wiring (spec §1.1): the MF eval grid is set EQUAL to the LF native
cache grid (`eval_logk = lf_logk`, 172 bins up to 0.0694 s/km), so the MF correction
`mf.logP_mf - mf.lf_logP == g + log res_corr` is evaluated EXACTLY on the cache k-grid,
then applied per class to P_filt at the cache-k level BEFORE the leg interp — i.e. the
correction enters the forward at the same place predict_P_obs reconstructs P_obs from
P_filt. KS<0.06 < 0.0694, so the kept KS rows interpolate inside the cache grid (no LF
tail extrapolation is hit for this band; the cache k_max IS the LF Nyquist 0.0694).

HF-LOSO (spec §1.1): when a fold's held-out sim is one of the 6 HR sims, the resolved
head is fit EXCLUDING that HR sim's rows (the LF backbone is already honestly held out).
Most folds hold out NO HR sim → the correction is in-HR-sample for those folds (only the
LF backbone is held out). Reported per-fold (the HR-LOSO generalization number is T4, not
this gate; THIS gate measures the n_s response of the PRODUCTION MF forward).

Two passes (spec §3.1, the primary aliasing probe): the rank-1 interaction term ON
(full g) and OFF (a_k≡0, pure separable). z-resolved n_s bias accumulated per z-bin.

Output: figures/analysis/04_emulator/emu_bias_allfolds_mf.txt (+ _mf.npz). Does NOT
overwrite emu_bias_allfolds.txt (z_lo=2.0 LF) or _zlo24.txt (z_lo=2.4 LF shipped).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_emu_bias_allfolds_mf.py
"""
import numpy as np
from pathlib import Path

import hcd_analysis.emulator  # x64 BEFORE jax
import jax, jax.numpy as jnp
import equinox as eqx

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg,
)
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator.predict import reconstruct_P_filt_jax, _excess_from_P_filt
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT
from hcd_analysis.emulator.data import Z_LIMITS

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/emu_bias_allfolds_mf.txt"
NPZ = f"{FIGDIR}/emu_bias_allfolds_mf.npz"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
KS_KMAX = 0.06   # the LOCKED KS cap
KS_ZLO = 2.4     # the SHIPPED z_lo default (matches emu_bias_allfolds_zlo24.txt)

# z-attribution bins (spec G3: the z=2.8-3.4 high-k contribution is the target band).
Z_BINS = [(2.0, 2.8), (2.8, 3.4), (3.4, 6.0)]

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
AP_I = int(np.where(np.array(PARAM_NAMES) == "Ap")[0][0])


# --------------------------------------------------------------------------- #
# Build the MF correction (resolved separable+rank-1 head) ONCE per fold, on the
# LF native cache grid. Returns a MultiFidelity whose eval grid == cache_k.
# --------------------------------------------------------------------------- #
def build_mf_for_fold(lf_cache, hr_cache, pairs, lf_model, lf_norm, lf_logk,
                      *, exclude_hr_sim=None, rank1=True):
    """A MultiFidelity on the LF native grid (eval_logk == lf_logk).

    The resolved FixedMeanHead is fit on the matched HR rows, EXCLUDING the
    held-out HR sim's rows (HF-LOSO) if ``exclude_hr_sim`` is one of the 6 HR sims.
    ``rank1=False`` zeroes the rank-1 interaction term (pure separable; the OFF arm)."""
    eval_logk = np.asarray(lf_logk)
    # measure g = logP_HF - logP_hat_LF on EVERY matched HR row, on the eval (=cache) grid.
    tg = MF.measure_delta_targets(lf_cache, hr_cache, lf_model, lf_norm, lf_logk,
                                  eval_logk, pairs)
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    # HF-LOSO row mask: drop the held-out HR sim's matched rows from the head fit.
    hr_sim_of_row = np.asarray(
        [hr_cache["sim_name"][h] for h, _ in pairs])
    hr_sim_of_row = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in hr_sim_of_row])
    if exclude_hr_sim is not None:
        train_rows = np.where(hr_sim_of_row != exclude_hr_sim)[0]
        loso = (exclude_hr_sim in set(hr_sim_of_row))
    else:
        train_rows = np.arange(len(pairs))
        loso = False
    comp = MF.fixed_mean_table_resolved(tg, log_rho, train_mask_rows=train_rows)
    a_k = comp["a_k"] if rank1 else np.zeros_like(comp["a_k"])
    head = MF.FixedMeanHead(
        comp["gbar_z_tab"], comp["z_tab"], resolved=True,
        gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
        a_k=a_k, u_z=comp["u_z"], u_tau=comp["u_tau"])
    mf = MF.build_multifidelity(lf_model, lf_norm, lf_logk, head,
                                eval_logk=eval_logk, log_rho=log_rho, delta_mode="none")
    return mf, dict(loso=loso, sing=float(np.sqrt(np.mean(comp["a_k"] ** 2))),
                    rank1=rank1)


def mf_corr_on_cache(mf, theta9, z_unit, tau0):
    """Per-class MF log-correction (4, Kc) on the cache grid: g + log res_corr.

    ``mf.eval_logk == cache_logk`` so this is exactly the production MF correction
    on the cache k-grid. g reads cond[9]=z_unit, cond[10]=tau0 (theta-blind by
    construction). res_corr is the fixed particle-convergence factor (z, k)."""
    x = jnp.concatenate([jnp.asarray(theta9), jnp.atleast_1d(z_unit)])  # (10,)
    g = mf.g(x, tau0)                                          # (4, Kc) = log_rho + head
    z_phys = z_unit * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
    log_rc = jnp.log(mf.res_corr(z_phys))[None, :]            # (1, Kc) broadcast over class
    return g + log_rc                                          # (4, Kc) additive in log


def predict_P_obs_mf(mf, model, theta9, z_unit, tau0, alpha_hcd, pf_stats, dla_core):
    """P_obs (Kc,) through the MF forward: LF P_filt * exp(g + log res_corr) per class,
    then the SAME clean+excess HCD combination as predict.predict_P_obs.

    The LF backbone P_filt is frozen; the MF correction is the fixed (theta-blind)
    per-class multiplicative resolution factor. Differentiable in (theta9, tau0, alpha)."""
    from hcd_analysis.emulator.predict import predict_P_filt
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)   # (4,Kc) LF
    corr = jnp.exp(mf_corr_on_cache(mf, theta9, z_unit, tau0))       # (4,Kc) MF factor
    P_filt_mf = P_filt * corr                                        # corrected per class
    P_clean = P_filt_mf[0]
    excess = _excess_from_P_filt(P_filt_mf, dla_core)               # (3,Kc)
    return P_clean + jnp.einsum("c,ck->k", jnp.asarray(alpha_hcd), excess)


def predict_P_obs_on_leg_mf(mf, model, theta9, tau0_vec, alpha_hcd, *, pf_stats,
                            dla_core, cache_k, leg):
    """Mirror of DL.predict_P_obs_on_leg's P_model branch, but P_obs goes through the
    MF forward. Returns ONLY the flat P_model (C_total is taken from the UNCHANGED LF
    path — spec: keep the same C_total). Differentiable in (theta9, tau0_vec, alpha)."""
    cache_k = jnp.asarray(cache_k)
    k_leg = jnp.asarray(leg.k)
    z_idx = np.asarray(leg.z_idx)
    tau0_vec = jnp.asarray(tau0_vec)
    alpha_hcd = jnp.asarray(alpha_hcd)
    N = k_leg.shape[0]
    P_model = jnp.zeros(N)
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        z_unit = float(leg.z_unit[iz])
        tau0 = tau0_vec[iz]
        k_sub = k_leg[jnp.asarray(rows)]
        alpha_z = alpha_hcd if alpha_hcd.ndim == 1 else alpha_hcd[iz]
        P_cache = predict_P_obs_mf(mf, model, theta9, z_unit, tau0, alpha_z,
                                   pf_stats, dla_core)
        P_z = jnp.interp(k_sub, cache_k, P_cache)
        P_model = P_model.at[jnp.asarray(rows)].set(P_z)
    return P_model


def emu_bias_one_sim(ctx, d, sim, fold, mf):
    """EMU-only A_p/n_s bias (in σ) for one held-out sim, summed over legs, THROUGH MF.
    Forward at the sim's EXACT per-z w_c (DLA-masked); ΔP = P_truth_masked − P_MF_forward
    (pure MF emulator error). z-resolved n_s/A_p attribution accumulated per z-bin.

    Returns (bAp, bNs, ns_u, ap_u, gz_ns(nP,nbin), gz_ap, F, Pp, info_perz)."""
    t = make_truth_from_sim(d, sim, fold)
    th_truth = jnp.asarray(t["params_unit"])
    wc = np.asarray(d["w_c_cache"])[np.asarray(t["rows"]), 1:].copy(); wc[:, 2] = 0.0  # mask DLA
    z_sim = np.asarray(t["z"])
    P_filt = np.asarray(d["P_filt"]); rows = np.asarray(t["rows"]); K = P_filt.shape[-1]
    # tau0 of each truth row (the selected ladder row's cache tau0) for the MF correction.
    tau0_sim = np.asarray(t["tau0"])
    z_unit_sim = (z_sim - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    P_masked = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        a = wc[i]; coef = np.concatenate([[1.0 - a.sum()], a])
        # The CLOSURE TRUTH is the LF-resolution sim. The PRODUCTION likelihood is the MF
        # forward (HR-resolution-target). For a PURE-emulator-error ΔP the truth must be at
        # the SAME (MF) resolution as the forward (spec §3.1 rationale: a theta-INDEPENDENT
        # correction applied to ALL 60 closure sims must NOT change the closure n_s bias by
        # construction — which only holds if the correction is applied to BOTH the forward
        # AND the truth so it cancels in ΔP up to the theta-dependence, which is zero). So we
        # apply the IDENTICAL per-class MF factor exp(g + log res_corr) to the truth P_filt
        # at the truth's (z, tau0) — the mock is then an MF-resolution draw, matching T5's
        # "re-run the closure through the MF forward". ΔP is then the genuine MF emulator
        # error (LF backbone error reshaped by the fixed correction), not a forward-vs-truth
        # RESOLUTION mismatch (which would be the LF->HR correction itself, ~+27% at k~0.06).
        corr = np.asarray(jnp.exp(mf_corr_on_cache(
            mf, th_truth, jnp.asarray(float(z_unit_sim[i])), jnp.asarray(float(tau0_sim[i])))))  # (4,K)
        pf_mf = np.stack([P_filt[r, 0] * corr[0], P_filt[r, 1] * corr[1],
                          P_filt[r, 2] * corr[2], P_filt[r, 3] * corr[3]])
        P_masked[i] = np.einsum("c,ck->k", coef, pf_mf)
    mock_legs, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"]); core = _mock_core_per_leg(ctx, t)
    zg = np.asarray(ctx.z_global); nZg = zg.size; nP = 9 + nZg + 3; AMP = np.arange(9 + nZg, 9 + nZg + 3)
    cache_k = np.asarray(ctx.cache_k)
    F = np.zeros((nP, nP)); g = np.zeros(nP)
    # per-z gradient blocks for the z-resolved attribution (g_z = Jᵀ C⁻¹ ΔP over that z's rows)
    gz_blocks = []  # list of (z_value, g_block (nP,))
    for leg in ctx.legs:
        z_leg = np.asarray(leg.z); k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
        sel = np.array([int(np.argmin(np.abs(zg - z))) for z in z_leg])
        ss = np.array([int(np.argmin(np.abs(z_sim - z))) for z in z_leg])
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        gfix = wc[ss] / np.clip(wc[ss][int(np.argmin(np.abs(z_sim[ss] - HCD_Z_PIVOT)))], 1e-12, None)
        apv0 = jnp.asarray(wc[ss][int(np.argmin(np.abs(z_sim[ss] - HCD_Z_PIVOT)))])
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None

        def fwd(th, t0g, apv, _leg=leg, _sel=jnp.asarray(sel), _keep=jnp.asarray(keep),
                _g=jnp.asarray(gfix)):
            al = apv[None, :] * _g
            P = predict_P_obs_on_leg_mf(mf, ctx.model, th, t0g[_sel], al,
                pf_stats=ctx.pf_stats, dla_core=core[_leg.name], cache_k=ctx.cache_k,
                leg=_leg)
            return P[_keep]
        J = jax.jacrev(fwd, argnums=(0, 1, 2))(th_truth, t0_truth, apv0)
        Jl = np.asarray(jnp.concatenate([J[0], J[1], J[2]], axis=1))
        P0 = np.asarray(fwd(th_truth, t0_truth, apv0)); Pfull = np.zeros(k.shape[0])
        for iz in range(leg.n_z):
            rr = np.where(z_idx == iz)[0]; j = int(np.argmin(np.abs(z_sim - float(z_leg[iz]))))
            Pfull[rr] = np.asarray(jnp.interp(jnp.asarray(k[rr]), jnp.asarray(cache_k),
                jnp.asarray(np.interp(cache_k, cache_k, P_masked[j]))))
        dP = Pfull[keep] - P0
        # C_total from the UNCHANGED LF path (spec: keep the same C_total).
        _, Ctot = DL.predict_P_obs_on_leg(ctx.model, th_truth, t0_truth[sel],
            apv0[None, :] * jnp.asarray(gfix), pf_stats=ctx.pf_stats,
            dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        F += Jl.T @ Cinv @ Jl; g += Jl.T @ Cinv @ dP
        # z-resolved attribution (EXACT, matches diag_emu_lowk_investigation.py:163):
        # g = Jᵀ Cinv dP = Σ_z Jᵀ Cinv (dP·m_z) with m_z the z-bin mask on the FLAT keep
        # rows. The FULL Cinv is used and dP is masked (NOT a per-z sub-inverse), so the
        # cross-z covariance coupling is kept and Σ_z g_z == g to machine precision.
        keep_z = z_idx[keep]                       # z-bin index of each kept row
        for iz in range(leg.n_z):
            m = (keep_z == iz)
            if not m.any():
                continue
            gz_blocks.append((float(z_leg[iz]), Jl.T @ Cinv @ (dP * m)))
    Pp = np.zeros(nP); Pp[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Cpost = np.linalg.inv(F + np.diag(Pp) + 1e-15 * np.mean(np.diag(F + np.diag(Pp))) * np.eye(nP))
    dx = Cpost @ g
    sNs = np.sqrt(Cpost[NS_I, NS_I]); sAp = np.sqrt(Cpost[AP_I, AP_I])
    # z-resolved bias: bias_z = (g_z · C_post[:,ns]) / σ_ns  (Σ_z == pooled bias).
    bias_ns_z = {lab: 0.0 for lab in range(len(Z_BINS))}
    bias_ap_z = {lab: 0.0 for lab in range(len(Z_BINS))}
    for zval, gblk in gz_blocks:
        b = next((bi for bi, (lo, hi) in enumerate(Z_BINS) if lo <= zval < hi), len(Z_BINS) - 1)
        bias_ns_z[b] += float((gblk @ Cpost[:, NS_I]) / sNs)
        bias_ap_z[b] += float((gblk @ Cpost[:, AP_I]) / sAp)
    return (dx[AP_I] / sAp, dx[NS_I] / sNs,
            float(t["params_unit"][NS_I]), float(t["params_unit"][AP_I]),
            bias_ns_z, bias_ap_z)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
emit("# T3 GATE — EMU bias THROUGH the MF forward (resolved separable+rank-1 g + res_corr).")
emit("# all 8 folds, each fold's OWN held-out LF backbone; HF-LOSO head when an HR sim is held out.")
emit("# KS z_lo=2.4 (SHIPPED), KS<0.06, same C_total as LF. forward-only Fisher.")
emit("# LF baseline (emu_bias_allfolds_zlo24.txt): n_s +0.030σ, A_p +0.034σ.")
emit("# u_tau FIXED to the physical mean-flux axis (Step-0 Bayesian rec).")
emit("# fold sim                            ns_u  Ap_u  bias(A_p)/σ bias(n_s)/σ  HRloso a_rms")

lf_cache = MF.load_cache(MF.LF_CACHE)
hr_cache = MF.load_cache(MF.HR_CACHE)
pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
hr_sims = sorted(set(s.decode() if isinstance(s, bytes) else s
                     for s in hr_cache["sim_name"]))

# accumulate ON and OFF passes
results = {"ON": dict(Ap=[], Ns=[], ns_u=[], bns_z=[], bap_z=[]),
           "OFF": dict(Ap=[], Ns=[], ns_u=[], bns_z=[], bap_z=[])}

for fold in range(8):
    ckpt = f"{REPO}/checkpoints/final_fold{fold}"
    ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0,
                            ks_kwargs={"k_max": KS_KMAX, "z_lo": KS_ZLO})
    lf_model, meta, lf_norm = ctx.model, None, ctx.pf_stats
    # the frozen backbone + norm for THIS fold (matches the ctx.model checkpoint).
    fold_model, fold_meta, fold_norm, lf_logk = MF.load_lf_backbone(fold)
    sims, _ = held_out_sims(d, fold)
    hr_held = [s for s in sims if s in hr_sims]
    excl = hr_held[0] if hr_held else None   # at most relevant HR sim(s); fit excludes ALL held HR
    # build the head excluding ALL held-out HR sims of this fold (HF-LOSO).
    hr_row_sim = np.array([s.decode() if isinstance(s, bytes) else s
                           for s, _ in [(hr_cache["sim_name"][h], l) for h, l in pairs]])
    held_set = set(hr_held)
    train_rows = np.where(~np.isin(hr_row_sim, list(held_set)))[0] if held_set else np.arange(len(pairs))
    # measure targets + build both heads (ON / OFF) sharing the SAME LOSO fit.
    tg = MF.measure_delta_targets(lf_cache, hr_cache, fold_model, fold_norm, lf_logk,
                                  np.asarray(lf_logk), pairs)
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, np.asarray(lf_logk)), nan=0.0)
    comp = MF.fixed_mean_table_resolved(tg, log_rho, train_mask_rows=train_rows)
    a_rms = float(np.sqrt(np.mean(comp["a_k"] ** 2)))
    for arm, rank1 in (("ON", True), ("OFF", False)):
        a_k = comp["a_k"] if rank1 else np.zeros_like(comp["a_k"])
        head = MF.FixedMeanHead(
            comp["gbar_z_tab"], comp["z_tab"], resolved=True,
            gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
            a_k=a_k, u_z=comp["u_z"], u_tau=comp["u_tau"])
        mf = MF.build_multifidelity(fold_model, fold_norm, lf_logk, head,
                                    eval_logk=np.asarray(lf_logk), log_rho=log_rho,
                                    delta_mode="none")
        for sim in sims:
            bAp, bNs, ns_u, ap_u, bns_z, bap_z = emu_bias_one_sim(ctx, d, sim, fold, mf)
            R = results[arm]
            R["Ap"].append(bAp); R["Ns"].append(bNs); R["ns_u"].append(ns_u)
            R["bns_z"].append(bns_z); R["bap_z"].append(bap_z)
            if arm == "ON":
                loso = "Y" if sim in held_set else "-"
                emit(f"  {fold}   {sim[:30]:30s} {ns_u:.3f} {ap_u:.3f}  {bAp:+7.3f}    {bNs:+7.3f}    {loso}    {a_rms:.4f}")

def boot(x, n=10000):
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

summary = {}
for arm in ("ON", "OFF"):
    R = results[arm]
    Ap = np.array(R["Ap"]); Ns = np.array(R["Ns"]); ns_u = np.array(R["ns_u"])
    emit(f"\n# ===== POOLED over ALL {len(Ap)} held-out sims — MF {arm} (rank-1 {'on' if arm=='ON' else 'OFF'}) =====")
    sm = {}
    for nm, arr in (("A_p", Ap), ("n_s", Ns)):
        m, lo, hi = boot(arr)
        se = arr.std(ddof=1) / np.sqrt(len(arr)); tt = arr.mean() / se
        npos = int((arr > 0).sum()); nneg = int((arr < 0).sum())
        emit(f"  {nm}: mean {arr.mean():+.4f}σ  bootstrap95% [{lo:+.3f},{hi:+.3f}]  t={tt:+.2f}  "
             f"signs {npos}+/{nneg}-  max|{np.abs(arr).max():.2f}|")
        sm[nm] = float(arr.mean())
    # z-resolved n_s & A_p attribution (mean over sims of the per-z bias contribution)
    bns_z = np.array([[d[b] for b in range(len(Z_BINS))] for d in R["bns_z"]])  # (Nsim, nbin)
    bap_z = np.array([[d[b] for b in range(len(Z_BINS))] for d in R["bap_z"]])
    emit("  z-resolved n_s bias contribution (mean over sims), per z-bin:")
    for bi, (zlo, zhi) in enumerate(Z_BINS):
        emit(f"    z∈[{zlo},{zhi}): n_s {bns_z[:,bi].mean():+.4f}σ   A_p {bap_z[:,bi].mean():+.4f}σ")
    chk = bns_z.sum(1).mean()
    emit(f"  [exactness] Σ_z n_s bias = {chk:+.4f}σ  (must == pooled n_s {sm['n_s']:+.4f}σ; "
         f"|Δ|={abs(chk-sm['n_s']):.2e})")
    sm["bns_z"] = bns_z.mean(0); sm["bap_z"] = bap_z.mean(0)
    sm["Ap_arr"] = Ap; sm["Ns_arr"] = Ns; sm["ns_u"] = ns_u
    summary[arm] = sm

# ---- G4.1 aliasing: rank-1 ON vs OFF ----
emit("\n# ===== G4.1 ALIASING: rank-1 ON vs OFF =====")
dns = summary["ON"]["n_s"] - summary["OFF"]["n_s"]
dap = summary["ON"]["A_p"] - summary["OFF"]["A_p"]
emit(f"  Δn_s (ON-OFF) = {dns:+.4f}σ   (PASS if |Δn_s| ≤ 0.10σ)")
emit(f"  ΔA_p (ON-OFF) = {dap:+.4f}σ")
emit("  Δ per z-bin (n_s):")
for bi, (zlo, zhi) in enumerate(Z_BINS):
    d_b = summary["ON"]["bns_z"][bi] - summary["OFF"]["bns_z"][bi]
    emit(f"    z∈[{zlo},{zhi}): Δn_s {d_b:+.4f}σ")
g41 = abs(dns) <= 0.10
emit(f"  G4.1: {'PASS' if g41 else 'FAIL'}  (Δn_s {'not ' if g41 else ''}> 0.10σ)")

np.savez(NPZ,
         ON_Ap=summary["ON"]["Ap_arr"], ON_Ns=summary["ON"]["Ns_arr"],
         OFF_Ap=summary["OFF"]["Ap_arr"], OFF_Ns=summary["OFF"]["Ns_arr"],
         ns_u=summary["ON"]["ns_u"],
         ON_bns_z=summary["ON"]["bns_z"], ON_bap_z=summary["ON"]["bap_z"],
         OFF_bns_z=summary["OFF"]["bns_z"], OFF_bap_z=summary["OFF"]["bap_z"],
         z_bins=np.array(Z_BINS))
emit(f"\n[npz] {NPZ}")
emit("[done]")
_rf.close()
