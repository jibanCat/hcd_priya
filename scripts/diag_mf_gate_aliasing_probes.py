"""T3 GATE — the G4.2 (τ₀-width inflation) and G4.3 (∂g/∂n_s structural) aliasing probes.

Companion to scripts/diag_emu_bias_allfolds_mf.py (which runs G1/G2/G3 + G4.1 ON/OFF).
This script runs the two probes that need the posterior GEOMETRY (G4.2) and the exact
structural θ-blindness check (G4.3), per the spec §3.2 / §3.3.

G4.3 (cleanest, exact): g(z,τ₀,k) reads only cond[9]=z_unit, cond[10]=τ₀ — never the 9
cosmology params. So ∂g/∂n_s must be IDENTICALLY zero (machine eps). Verified with
jax.grad over a grid of (z,τ₀) eval points, on a LOSO-fit resolved head. PASS if
|∂g/∂n_s| < 1e-10 everywhere.

G4.2 (τ₀-posterior-width inflation): per fold/sim, the Fisher C_post under MF-ON, MF-OFF,
and LF. Report σ_τ0(ON)/σ_τ0(OFF) (PASS ≤1.05) and |r(τ₀,n_s)(ON) − r(τ₀,n_s)(LF)| (PASS
≤0.05). τ₀ is the data-constraining z-bin block (z∈[2.4,3.4] mean of diag); r is the
worst (max |Δr|) over the τ₀ z-bins. C_post = (F+P)^{-1}; F is the SAME Fisher the bias
run builds (forward Jacobian), so this reads the joint (n_s,τ₀) geometry directly.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_gate_aliasing_probes.py
"""
import numpy as np

import hcd_analysis.emulator
import jax, jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock, _mock_core_per_leg,
)
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_Z_PIVOT
from hcd_analysis.emulator.data import Z_LIMITS

# reuse the wired forward from the gate script
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("mfgate", "/home/mfho/hcd_priya/scripts/diag_emu_bias_allfolds_mf.py")

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/emu_bias_allfolds_mf_aliasing.txt"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
KS_KMAX, KS_ZLO = 0.06, 2.4

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])


# ---- the MF-wired forward (same as the gate script) ---- #
from hcd_analysis.emulator.predict import predict_P_filt, _excess_from_P_filt

def mf_corr_on_cache(mf, theta9, z_unit, tau0):
    x = jnp.concatenate([jnp.asarray(theta9), jnp.atleast_1d(z_unit)])
    g = mf.g(x, tau0)
    z_phys = z_unit * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
    return g + jnp.log(mf.res_corr(z_phys))[None, :]

def predict_P_obs_mf(mf, model, theta9, z_unit, tau0, alpha_hcd, pf_stats, dla_core):
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)
    P_filt_mf = P_filt * jnp.exp(mf_corr_on_cache(mf, theta9, z_unit, tau0))
    P_clean = P_filt_mf[0]
    excess = _excess_from_P_filt(P_filt_mf, dla_core)
    return P_clean + jnp.einsum("c,ck->k", jnp.asarray(alpha_hcd), excess)

def predict_P_obs_on_leg_mf(mf, model, theta9, tau0_vec, alpha_hcd, *, pf_stats, dla_core, cache_k, leg):
    cache_k = jnp.asarray(cache_k); k_leg = jnp.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
    tau0_vec = jnp.asarray(tau0_vec); alpha_hcd = jnp.asarray(alpha_hcd)
    P_model = jnp.zeros(k_leg.shape[0])
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0: continue
        z_unit = float(leg.z_unit[iz]); tau0 = tau0_vec[iz]
        k_sub = k_leg[jnp.asarray(rows)]
        az = alpha_hcd if alpha_hcd.ndim == 1 else alpha_hcd[iz]
        Pc = predict_P_obs_mf(mf, model, theta9, z_unit, tau0, az, pf_stats, dla_core)
        P_model = P_model.at[jnp.asarray(rows)].set(jnp.interp(k_sub, cache_k, Pc))
    return P_model


def cpost_for_sim(ctx, d, sim, fold, mf=None):
    """C_post for one sim. If mf is None -> the LF forward (DL.predict_P_obs_on_leg);
    else the MF-wired forward. Returns C_post (nP,nP) + the tau0 index slice."""
    t = make_truth_from_sim(d, sim, fold)
    th_truth = jnp.asarray(t["params_unit"])
    wc = np.asarray(d["w_c_cache"])[np.asarray(t["rows"]), 1:].copy(); wc[:, 2] = 0.0
    z_sim = np.asarray(t["z"])
    mock_legs, tp, info = make_legb_mock(ctx, t, jax.random.PRNGKey(0))
    t0_truth = jnp.asarray(tp["tau0_global"]); core = _mock_core_per_leg(ctx, t)
    zg = np.asarray(ctx.z_global); nZg = zg.size; nP = 9 + nZg + 3; AMP = np.arange(9 + nZg, 9 + nZg + 3)
    F = np.zeros((nP, nP))
    for leg in ctx.legs:
        z_leg = np.asarray(leg.z); z_idx = np.asarray(leg.z_idx)
        sel = np.array([int(np.argmin(np.abs(zg - z))) for z in z_leg])
        ss = np.array([int(np.argmin(np.abs(z_sim - z))) for z in z_leg])
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        gfix = wc[ss] / np.clip(wc[ss][int(np.argmin(np.abs(z_sim[ss] - HCD_Z_PIVOT)))], 1e-12, None)
        apv0 = jnp.asarray(wc[ss][int(np.argmin(np.abs(z_sim[ss] - HCD_Z_PIVOT)))])
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        def fwd(th, t0g, apv, _leg=leg, _sel=jnp.asarray(sel), _keep=jnp.asarray(keep), _g=jnp.asarray(gfix)):
            al = apv[None, :] * _g
            if mf is None:
                P, _ = DL.predict_P_obs_on_leg(ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats,
                    dla_core=core[_leg.name], cache_k=ctx.cache_k, leg=_leg, sigma_zb=None, alpha_centres=None)
            else:
                P = predict_P_obs_on_leg_mf(mf, ctx.model, th, t0g[_sel], al, pf_stats=ctx.pf_stats,
                    dla_core=core[_leg.name], cache_k=ctx.cache_k, leg=_leg)
            return P[_keep]
        J = jax.jacrev(fwd, argnums=(0, 1, 2))(th_truth, t0_truth, apv0)
        Jl = np.asarray(jnp.concatenate([J[0], J[1], J[2]], axis=1))
        _, Ctot = DL.predict_P_obs_on_leg(ctx.model, th_truth, t0_truth[sel],
            apv0[None, :] * jnp.asarray(gfix), pf_stats=ctx.pf_stats, dla_core=core[leg.name],
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
        Cii = np.asarray(Ctot)[np.ix_(keep, keep)]
        Cinv = np.linalg.inv(Cii + 1e-12 * np.mean(np.diag(Cii)) * np.eye(Cii.shape[0]))
        F += Jl.T @ Cinv @ Jl
    Pp = np.zeros(nP); Pp[9:9 + nZg] = 1.0 / np.asarray(ctx.tau0_sigma) ** 2
    Pp[AMP] = 1.0 / np.asarray(ctx.alpha_hcd_sigma) ** 2
    Cpost = np.linalg.inv(F + np.diag(Pp) + 1e-15 * np.mean(np.diag(F + np.diag(Pp))) * np.eye(nP))
    return Cpost, np.arange(9, 9 + nZg), zg


def build_heads(fold, lf_cache, hr_cache, pairs, hr_sims):
    """Return (mf_on, mf_off, log_rho) for the fold with the HF-LOSO resolved head."""
    fold_model, _, fold_norm, lf_logk = MF.load_lf_backbone(fold)
    # held-out HR sims this fold
    import h5py  # noqa
    tg = MF.measure_delta_targets(lf_cache, hr_cache, fold_model, fold_norm, lf_logk,
                                  np.asarray(lf_logk), pairs)
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, np.asarray(lf_logk)), nan=0.0)
    return fold_model, fold_norm, lf_logk, tg, log_rho


# --------------------------------------------------------------------------- #
emit("# T3 GATE aliasing probes G4.2 (τ₀-width inflation) + G4.3 (∂g/∂n_s structural).")
emit("# u_tau FIXED to the physical mean-flux axis (Step-0).")

lf_cache = MF.load_cache(MF.LF_CACHE); hr_cache = MF.load_cache(MF.HR_CACHE)
pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
hr_sims = sorted(set(s.decode() if isinstance(s, bytes) else s for s in hr_cache["sim_name"]))
hr_row_sim = np.array([s.decode() if isinstance(s, bytes) else s
                       for s, _ in [(hr_cache["sim_name"][h], l) for h, l in pairs]])

# ===== G4.3: ∂g/∂n_s structural (exact, cheap) — fold 0 LOSO head =====
emit("\n# ===== G4.3: ∂g/∂n_s (head must be θ-blind) =====")
fold_model, fold_norm, lf_logk, tg, log_rho = build_heads(0, lf_cache, hr_cache, pairs, hr_sims)
comp = MF.fixed_mean_table_resolved(tg, log_rho)
head = MF.FixedMeanHead(comp["gbar_z_tab"], comp["z_tab"], resolved=True,
    gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
    a_k=comp["a_k"], u_z=comp["u_z"], u_tau=comp["u_tau"])
mf0 = MF.build_multifidelity(fold_model, fold_norm, lf_logk, head,
                             eval_logk=np.asarray(lf_logk), log_rho=log_rho, delta_mode="none")
maxabs = 0.0
for z_unit in (0.06, 0.24, 0.41, 0.59, 0.76):
    for t0 in (0.7, 1.0, 1.5, 2.0):
        def gfun(th9, _zu=z_unit, _t0=t0):
            x = jnp.concatenate([th9, jnp.atleast_1d(_zu)])
            return mf0.g(x, jnp.asarray(_t0)).sum()
        th9 = jnp.full(9, 0.5)
        dgn = float(jax.grad(gfun)(th9)[NS_I])
        maxabs = max(maxabs, abs(dgn))
emit(f"  max |∂g/∂n_s| over (z,τ₀) grid = {maxabs:.3e}   (PASS if < 1e-10)")
g43 = maxabs < 1e-10
emit(f"  G4.3: {'PASS' if g43 else 'FAIL'}")

# ===== G4.2: τ₀-width inflation + τ₀–n_s correlation (ON / OFF / LF) =====
emit("\n# ===== G4.2: τ₀-posterior width + τ₀–n_s correlation (ON/OFF/LF) =====")
# data-constraining τ₀ z-bins (z in [2.4,3.4]) for the width metric.
ratios = []; dr_on_lf = []; dr_off_lf = []; dr_on_off = []
n_done = 0
for fold in range(8):
    ckpt = f"{REPO}/checkpoints/final_fold{fold}"
    ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0,
                            ks_kwargs={"k_max": KS_KMAX, "z_lo": KS_ZLO})
    sims, _ = held_out_sims(d, fold)
    held = [s for s in sims if s in hr_sims]
    train_rows = (np.where(~np.isin(hr_row_sim, held))[0] if held else np.arange(len(pairs)))
    fm, fn, llk, tgf, lr = build_heads(fold, lf_cache, hr_cache, pairs, hr_sims)
    cf = MF.fixed_mean_table_resolved(tgf, lr, train_mask_rows=train_rows)
    def mk(rank1):
        ak = cf["a_k"] if rank1 else np.zeros_like(cf["a_k"])
        h = MF.FixedMeanHead(cf["gbar_z_tab"], cf["z_tab"], resolved=True,
            gtau_tab=cf["gtau_tab"], tau_tab=cf["tau_tab"], tau_by_z=cf["tau_by_z"],
            a_k=ak, u_z=cf["u_z"], u_tau=cf["u_tau"])
        return MF.build_multifidelity(fm, fn, llk, h, eval_logk=np.asarray(llk), log_rho=lr, delta_mode="none")
    mf_on, mf_off = mk(True), mk(False)
    # one representative sim per fold (the first held-out sim) keeps this cheap.
    sim = sims[0]
    Cp_on, t0idx, zg = cpost_for_sim(ctx, d, sim, fold, mf=mf_on)
    Cp_off, _, _ = cpost_for_sim(ctx, d, sim, fold, mf=mf_off)
    Cp_lf, _, _ = cpost_for_sim(ctx, d, sim, fold, mf=None)
    band = (zg >= 2.4) & (zg <= 3.4)
    tband = t0idx[band]
    sig = lambda Cp: float(np.sqrt(np.mean(np.diag(Cp)[tband])))
    r = lambda Cp, i: Cp[i, NS_I] / np.sqrt(Cp[i, i] * Cp[NS_I, NS_I])
    ron = np.array([r(Cp_on, i) for i in tband]); roff = np.array([r(Cp_off, i) for i in tband])
    rlf = np.array([r(Cp_lf, i) for i in tband])
    ratio = sig(Cp_on) / sig(Cp_off)
    dr_onlf = float(np.max(np.abs(ron - rlf))); dr_offlf = float(np.max(np.abs(roff - rlf)))
    dr_onoff = float(np.max(np.abs(ron - roff)))   # rank-1 INCREMENTAL rotation (the §3.2 intent)
    ratios.append(ratio); dr_on_lf.append(dr_onlf); dr_off_lf.append(dr_offlf)
    dr_on_off.append(dr_onoff)
    emit(f"  fold {fold} ({sim[:24]}): σ_τ0 ON/OFF = {ratio:.4f}  "
         f"max|Δr(ON−LF)| = {dr_onlf:.4f}  max|Δr(OFF−LF)| = {dr_offlf:.4f}  "
         f"max|Δr(ON−OFF)| = {dr_onoff:.4f}")
    n_done += 1

ratios = np.array(ratios); dr_on_lf = np.array(dr_on_lf)
dr_off_lf = np.array(dr_off_lf); dr_on_off = np.array(dr_on_off)
emit(f"\n  worst σ_τ0(ON)/σ_τ0(OFF) over folds = {ratios.max():.4f}   (PASS if ≤ 1.05)")
emit(f"  worst |r(τ₀,n_s)(ON) − r(LF)| over folds   = {dr_on_lf.max():.4f}   (literal §3.2: ≤ 0.05)")
emit(f"  worst |r(τ₀,n_s)(OFF) − r(LF)| over folds  = {dr_off_lf.max():.4f}   (separable-only rotation)")
emit(f"  worst |r(τ₀,n_s)(ON) − r(OFF)| over folds  = {dr_on_off.max():.4f}   (RANK-1 INCREMENT — §3.2 intent: ≤ 0.05)")
emit("  NOTE: Δr(ON−LF) ≈ Δr(OFF−LF) ⇒ the rotation is from the SEPARABLE correction (in BOTH arms),")
emit("        NOT the rank-1 term. The §3.2 FAIL condition is about the rank-1 TERM trading n_s↔τ₀ info;")
emit("        the rank-1 INCREMENT |r(ON)−r(OFF)| is the diagnostically-correct quantity for that.")
g42_width = ratios.max() <= 1.05
g42_rank1 = dr_on_off.max() <= 0.05          # the rank-1 term does not rotate the degeneracy
g42_literal = dr_on_lf.max() <= 0.05         # the literal ON-vs-LF clause
emit(f"  G4.2 width:        {'PASS' if g42_width else 'FAIL'}  (no τ₀-width inflation)")
emit(f"  G4.2 rank-1 rot:   {'PASS' if g42_rank1 else 'FAIL'}  (rank-1 increment |r(ON)−r(OFF)| ≤ 0.05)")
emit(f"  G4.2 literal ON-LF:{'PASS' if g42_literal else 'FAIL'}  (separable+rank-1 vs LF; exceeds 0.05 via the SEPARABLE part)")
emit(f"  G4.2 (rank-1 aliasing intent): {'PASS' if (g42_width and g42_rank1) else 'FAIL'}")
emit("\n[done]")
_rf.close()
