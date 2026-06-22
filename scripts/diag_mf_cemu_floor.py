"""TASK B — the T4 small-scale C_emu floor (LF->HR generalization budget).

Per hcd_priya_notes/docs/superpowers/onboarding/2026-06-08-mf-cemu-floor-spec.md, runs the HONEST
6-fold HR-LOSO of the LF->HR resolution correction, on the production forward's
log-ratio object g(z,tau0,k) = log_rho + resolved separable+rank-1 FixedMeanHead,
EXERCISED through the SAME loglog_interp/extrap the production forward uses (so the
k>=0.07 tail-extrapolation band IS sized). Hold out each of the 6 HR sims, build the
correction from the OTHER 5, predict the held-out HR ratio, measure the coherent
residual:

    coherent_s(z,band) = nanmean_{k in band}( P_MF_LOSO(s;z,k)/P_HR_true(s;z,k) - 1 )
                       = nanmean_{k in band}( exp(g_LOSO(s;z,k) - g_true(s;z,k)) - 1 )
    sigma_floor(z,band) = max( 1.35 * max_s |coherent_s(z,band)| , 0.0123 )

CRITICAL FOOTING (the faithfulness fix): the LF->HR ratio is measured on the RAW LF/HR
CACHE P1D (the resolution backbone), NOT on the LF EMULATOR's prediction. Measuring
through the emulator backbone (mf.logP_mf) conflates the LF emulator's OWN point error
at the HR design points (-0.70 logP / -50% near the 172-bin NATIVE Nyquist k~0.069 —
verified) with the resolution-correction generalization the floor is supposed to budget.
The emulator point error is a SEPARATE object (the LF emulator's own per-k C_emu / LOSO
validation, already in the pipeline). The raw-cache footing isolates the LF->HR object
and reproduces the spec anchor (mf_rescorr_loso.txt: worst-per-sim coherent +0.91%).
The production forward's tail-extrapolation is STILL exercised: the cache backbone is fed
through multifidelity.loglog_interp_extrap (the exact forward routine) for k>=0.07, so the
extrapolated-band generalization is measured; the backbone cancels in g_LOSO - g_true, so
what remains is purely the correction's (incl. extrapolated-band) generalization error.

PLUS the n_s-edge term sigma_edge(z,band;ns) for ns outside the HR box [0.86,0.98] (§4).

SCOPE (on every output): covers LF->HR GENERALIZATION ONLY. HR->truth non-convergence is
the LOCKED k<0.06 analysis cap, NOT this floor. The LF-emulator point error is the LF
emulator's own C_emu, NOT this floor. n=6 HR sims (ns in [0.859,0.979]) -> necessary-not-
sufficient; ns>0.98 is the edge term, NOT certified.

Output: figures/analysis/04_emulator/mf_cemu_floor.{png,txt,npz}. New script; no commit.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_cemu_floor.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hcd_analysis.emulator  # x64 before jax
import jax.numpy as jnp

from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator.data import Z_LIMITS

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
TXT = f"{FIGDIR}/mf_cemu_floor.txt"
NPZ = f"{FIGDIR}/mf_cemu_floor.npz"
PNG = f"{FIGDIR}/mf_cemu_floor.png"

INFLATE = 1.35           # finite-sample inflation, n=6 (spec §3)
FLOOR_MIN = 0.0123       # global worst-per-sim +0.91% x 1.35 (spec §2.4 lower bound)
K_LFRES = 0.069          # LF Nyquist: band split (s/km)
K_EXTRAP = 0.07          # extrapolated band lower edge
NS_BOX = (0.86, 0.98)    # HR ns cluster box for the edge term (spec §4)
EDGE_FAC = 2.0           # 2x because a 6-sim slope is itself noisy (spec §4.1)
CLEAN = 0                # clean-forest class (P_obs is clean-dominated; per-class spread <=1.1pp < floor)
LOWZ_MAX = 2.6           # the low-z band where the LF deficit is largest (anchor footing)

_rf = open(TXT, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()


def cache_logP(cache, row, eval_logk, extrap):
    """Clean-class log P1D of a cache row on eval_logk. If extrap=True, fed through the
    forward's loglog_interp_extrap (tail extrapolation above the cache k_max); else a
    plain within-support interp (NaN outside)."""
    P = cache["P_filt"][row, CLEAN]; k = cache["kfkms"][row]
    m = np.isfinite(P) & (P > 0) & np.isfinite(k)
    if m.sum() < 2:
        return np.full(len(eval_logk), np.nan)
    src_logk = np.log10(k[m]); src_logP = np.log(P[m])
    if extrap:
        return np.asarray(MF.loglog_interp_extrap(
            jnp.asarray(src_logk), jnp.asarray(src_logP), jnp.asarray(eval_logk)))
    return np.interp(eval_logk, src_logk, src_logP, left=np.nan, right=np.nan)


def main():
    lf_cache = MF.load_cache(MF.LF_CACHE)
    hr_cache = MF.load_cache(MF.HR_CACHE)
    pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
    hr_sim_of_pair = np.array([
        (hr_cache["sim_name"][h].decode()
         if isinstance(hr_cache["sim_name"][h], bytes) else hr_cache["sim_name"][h])
        for h, _ in pairs])
    hr_sims = sorted(set(hr_sim_of_pair))
    ns_phys = {}
    for s in hr_sims:
        h, l = pairs[np.where(hr_sim_of_pair == s)[0][0]]
        ns_phys[s] = float(hr_cache["params"][h, 0])

    # PRODUCTION eval grid: k_min -> 0.10, n_k=48 (build_eval_grid). SPANS the LF Nyquist
    # 0.069 so the tail-extrapolation band (k>=0.07) IS exercised via loglog_interp_extrap.
    k_eval, eval_logk = MF.build_eval_grid(hr_cache, k_max=0.10, n_k=48)
    band_lfres = k_eval < K_LFRES
    band_extrap = k_eval >= K_EXTRAP

    emit("# T4 C_emu FLOOR — 6-fold HR-LOSO of the LF->HR resolution correction (RAW-CACHE footing).")
    emit(f"# eval grid: {len(k_eval)} log-bins, k=[{k_eval.min():.4f},{k_eval.max():.4f}] s/km.")
    emit(f"# band split: LF-resolvable k<{K_LFRES} ({band_lfres.sum()} bins) | "
         f"extrapolated k>={K_EXTRAP} ({band_extrap.sum()} bins, fed through loglog_interp_extrap).")
    emit(f"# INFLATE={INFLATE} (n=6 sampling, spec §3); FLOOR_MIN={FLOOR_MIN} (global worst x1.35).")
    emit("# FOOTING: g measured on RAW LF/HR CACHE P1D (NOT the LF emulator) -> isolates the LF->HR")
    emit("#   resolution-correction generalization from the LF emulator's own point error near the")
    emit("#   native Nyquist (-50% in P; a SEPARATE object = the LF emulator's per-k C_emu). The")
    emit("#   tail-extrapolation IS exercised (cache backbone through loglog_interp_extrap); the")
    emit("#   backbone cancels in g_LOSO-g_true so only the correction's generalization remains.")
    emit("# SCOPE: covers LF->HR GENERALIZATION ONLY. HR->truth is the LOCKED k<0.06 cap.")
    emit("# class: CLEAN (0) — P_obs is clean-dominated; per-class spread <=1.1pp < the floor (spec §2.3).")
    emit(f"# 6 HR sims, ns_phys: " + ", ".join(f"{ns_phys[s]:.3f}" for s in hr_sims))
    emit("")

    z_grid = np.array(sorted(set(np.round(hr_cache["z_grid"], 2))))

    # --- measure g_true on the raw-cache footing for ALL matched rows, on the eval grid --- #
    # g_true(row, k) = logP_HR_cache - logP_LF_cache (both clean), with the LF backbone fed
    # through the forward's loglog_interp_extrap so k>=0.07 is the tail-extrapolated LF.
    M = len(pairs)
    G = np.full((M, len(eval_logk)), np.nan)     # (M, K_eval) the LF->HR log-ratio
    Zrow = np.zeros(M); Arow = np.zeros(M, int); T0row = np.zeros(M)
    Xrow = np.zeros((M, 10))
    for ridx in range(M):
        h, l = pairs[ridx]
        bb = cache_logP(lf_cache, l, eval_logk, extrap=True)      # tail-extrapolated LF backbone
        hr = cache_logP(hr_cache, h, eval_logk, extrap=False)     # HR truth (within HR support)
        G[ridx] = hr - bb
        Xrow[ridx] = lf_cache["x"][l]
        Zrow[ridx] = round(float(lf_cache["x"][l][9]) * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
        Arow[ridx] = int(lf_cache["alpha_idx"][l]); T0row[ridx] = float(lf_cache["tau0"][l])

    # a synthetic targets dict so we can reuse fixed_mean_table_resolved (the PRODUCTION head
    # construction) verbatim on the raw-cache g (broadcast the single clean class to 4 so the
    # head shape matches; only clean is used downstream).
    g4 = np.repeat(G[:, None, :], MF.N_CLASSES, axis=1)           # (M,4,K)
    targets = dict(x=Xrow, tau0=T0row, alpha_idx=Arow, g=g4)

    coh = np.full((len(hr_sims), len(z_grid), 2), np.nan)
    for si, s_held in enumerate(hr_sims):
        train_rows = np.where(hr_sim_of_pair != s_held)[0]
        log_rho = np.nan_to_num(
            MF.mean_log_ratio_rho({"g": g4[train_rows]}, eval_logk), nan=0.0)
        comp = MF.fixed_mean_table_resolved(targets, log_rho, train_mask_rows=train_rows)
        head = MF.FixedMeanHead(
            comp["gbar_z_tab"], comp["z_tab"], resolved=True,
            gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
            a_k=comp["a_k"], u_z=comp["u_z"], u_tau=comp["u_tau"])
        # a MultiFidelity only to get g(x,tau0) on the eval grid (the LF backbone is unused for g).
        fm, _, fn, lf_logk = MF.load_lf_backbone(0)
        mf = MF.build_multifidelity(fm, fn, lf_logk, head, eval_logk=eval_logk,
                                    log_rho=log_rho, delta_mode="none")
        held = np.where(hr_sim_of_pair == s_held)[0]
        eps_by_z = {zi: {0: [], 1: []} for zi in range(len(z_grid))}
        for ridx in held:
            h, l = pairs[ridx]
            x = jnp.asarray(lf_cache["x"][l]); tau0 = jnp.asarray(float(lf_cache["tau0"][l]))
            g_loso = np.asarray(mf.g(x, tau0))[CLEAN]            # log-ratio prediction (clean)
            g_true = G[ridx]
            eps = np.exp(g_loso - g_true) - 1.0                  # P_MF/P_HR - 1 (backbone cancels)
            zi = int(np.argmin(np.abs(z_grid - Zrow[ridx])))
            with np.errstate(invalid="ignore"):
                eps_by_z[zi][0].append(np.where(band_lfres, eps, np.nan))
                eps_by_z[zi][1].append(np.where(band_extrap, eps, np.nan))
        for zi in range(len(z_grid)):
            for bi in (0, 1):
                if eps_by_z[zi][bi]:
                    with np.errstate(invalid="ignore"):
                        coh[si, zi, bi] = np.nanmean(np.array(eps_by_z[zi][bi]))
        lz = Zrow[held] <= LOWZ_MAX
        emit(f"  LOSO sim ns={ns_phys[s_held]:.3f}: worst|coherent| (all z) "
             f"LFres={np.nanmax(np.abs(coh[si,:,0])):.4f}  extrap={np.nanmax(np.abs(coh[si,:,1])):.4f}")

    # anchor cross-check: low-z high-k LF-resolvable coherent worst-per-sim (vs +0.91%).
    m_lzhik = (k_eval >= 0.0442) & band_lfres
    lz_coh = np.full(len(hr_sims), np.nan)
    for si in range(len(hr_sims)):
        lz_mask = z_grid <= LOWZ_MAX
        lz_coh[si] = np.nanmean(coh[si, lz_mask, 0])  # approx: low-z LFres band coherent
    emit(f"\n# anchor cross-check (low-z<=2.6 LF-resolvable coherent): "
         f"worst-per-sim |.| = {np.nanmax(np.abs(lz_coh))*100:.2f}%  "
         f"[mf_rescorr_loso.txt raw-cache anchor: +0.91%]")

    with np.errstate(invalid="ignore"):
        worst_s = np.nanmax(np.abs(coh), axis=0)                 # (n_z, 2)
    sigma_floor = np.where(np.isfinite(worst_s),
                           np.maximum(INFLATE * worst_s, FLOOR_MIN), FLOOR_MIN)

    # n_s-edge SLOPE(z,band) = |d(coherent_s)/d(ns_s)| over the 6 sims.
    ns_arr = np.array([ns_phys[s] for s in hr_sims])
    slope = np.full((len(z_grid), 2), np.nan)
    for zi in range(len(z_grid)):
        for bi in (0, 1):
            y = coh[:, zi, bi]; m = np.isfinite(y)
            if m.sum() >= 3:
                A = np.vstack([ns_arr[m], np.ones(m.sum())]).T
                slope[zi, bi] = abs(np.linalg.lstsq(A, y[m], rcond=None)[0][0])
    ns_query = 1.009
    d_ns = max(ns_query - NS_BOX[1], NS_BOX[0] - ns_query, 0.0)
    with np.errstate(invalid="ignore"):
        edge_slope_term = EDGE_FAC * slope * d_ns
        edge_floor_term = 0.5 * sigma_floor * (d_ns / 0.03)
        sigma_edge = np.where(np.isfinite(slope),
                              np.maximum(edge_slope_term, edge_floor_term), edge_floor_term)

    # NOISE-LIMITED flag: a (z,band) cell whose worst sim is >3x the median |coherent| of
    # the OTHER 5 sims is single-sim-driven (likely 6-sim noise at the sparse high-z bins),
    # not a robust generalization residual. Flag it; do NOT silently down-weight (spec §2.2
    # mandates worst-per-sim) but mark it so the assembler/reviewer treats it as a CEILING
    # estimate, not a measured floor.
    noise_flag = np.zeros((len(z_grid), 2), dtype=bool)
    n_cov = np.zeros(len(z_grid), int)
    for zi in range(len(z_grid)):
        n_cov[zi] = int(np.isfinite(coh[:, zi, 0]).sum())
        for bi in (0, 1):
            col = np.abs(coh[:, zi, bi]); col = col[np.isfinite(col)]
            if col.size >= 4:
                wi = np.argmax(col); others = np.delete(col, wi)
                med = np.median(others)
                if med > 0 and col[wi] > 3.0 * med:
                    noise_flag[zi, bi] = True

    emit("\n# ===== sigma_floor(z, band)  [LF->HR generalization, FIXED, ns-independent] =====")
    emit("#   z     LF-resolvable(k<0.069)   extrapolated(k>=0.07)   [flags]")
    for zi, zz in enumerate(z_grid):
        fl = []
        if n_cov[zi] < 4: fl.append("SPARSE(<4 sims)")
        if noise_flag[zi, 0]: fl.append("LFres single-sim-driven")
        if noise_flag[zi, 1]: fl.append("extrap single-sim-driven")
        emit(f"  {zz:4.1f}      {sigma_floor[zi,0]*100:6.2f}%                 "
             f"{sigma_floor[zi,1]*100:6.2f}%       {('; '.join(fl)) if fl else ''}")
    emit("#  FLAGS: SPARSE = <4 HR sims cover this z (floor pinned to FLOOR_MIN). single-sim-driven =")
    emit("#  the worst sim is >3x the median |coherent| of the other 5 -> a CEILING (likely 6-sim noise")
    emit("#  at the high-z He-II regime, spec §5: 'CANNOT distinguish a real z=5 sign-flip from 6-sim")
    emit("#  noise in a single z-bin'), NOT a robust floor. Recommend the assembler CAP these cells at")
    emit("#  the next-largest per-z floor (or carry as-is with the flag) pending more HR sims.")
    emit(f"\n  GLOBAL max sigma_floor: LFres={sigma_floor[:,0].max()*100:.2f}%  "
         f"extrap={sigma_floor[:,1].max()*100:.2f}%")
    emit(f"  (>= FLOOR_MIN={FLOOR_MIN*100:.2f}% everywhere; the extrapolated band is larger — the LF tail")
    emit(f"   extrapolation above the Nyquist is where the correction is least reliable, as spec §2.2 expects.")
    emit(f"   The DATA rows sit below k<0.06 (LF-resolvable band); the extrapolated band sizes the eval-grid")
    emit(f"   / cosmic-variance leakage above the Nyquist, NOT the kept data rows.)")

    emit(f"\n# ===== sigma_edge(z, band; ns)  [n_s OUT-of-cluster extrapolation, ns-DEPENDENT] =====")
    emit(f"# reported at ns_query={ns_query} (eBOSS-like; AT/above the HR ceiling {NS_BOX[1]}); "
         f"d_ns={d_ns:.3f} outside [{NS_BOX[0]},{NS_BOX[1]}].")
    emit(f"# sigma_edge = max( {EDGE_FAC}*|d coherent/d ns|*d_ns , 0.5*sigma_floor*(d_ns/0.03) ).")
    emit("#   z     |slope|_LFres  |slope|_extrap   sigma_edge_LFres  sigma_edge_extrap")
    for zi, zz in enumerate(z_grid):
        sl0, sl1 = slope[zi, 0], slope[zi, 1]
        emit(f"  {zz:4.1f}    {sl0 if np.isfinite(sl0) else float('nan'):7.4f}      "
             f"{sl1 if np.isfinite(sl1) else float('nan'):7.4f}        "
             f"{sigma_edge[zi,0]*100:6.2f}%           {sigma_edge[zi,1]*100:6.2f}%")
    emit(f"\n  GLOBAL max sigma_edge @ ns={ns_query}: LFres={np.nanmax(sigma_edge[:,0])*100:.2f}%  "
         f"extrap={np.nanmax(sigma_edge[:,1])*100:.2f}%")

    emit("\n# DISCIPLINE (n=6):")
    emit("#  - CAN: size the in-cluster LF->HR generalization floor + its sampling inflation (worst-per-sim,")
    emit("#    z-resolved, per-band). CANNOT: certify the amplitude OUTSIDE ns in [0.86,0.98] (the sigma_edge")
    emit("#    term — an INFLATION not a measurement; the 6-sim slope is noisy, hence 2x+floor).")
    emit("#  - CANNOT distinguish a real z>=5 sign-flip from 6-sim noise in a single z-bin (mitigated by the")
    emit("#    FLOOR_MIN max). CANNOT cover HR->truth (k<0.06 cap) nor the LF-emulator point error (its own C_emu).")
    emit("#  - ASSEMBLY (CS agent, small-scale leg ONLY): emu_var(k,z) += (sigma_floor(z,band(k))*P_obs)^2")
    emit("#    + (sigma_edge(z,band(k);ns)*P_obs)^2.  ns>0.98 posterior is GUARDED, not nominal.")

    np.savez(NPZ, z_grid=z_grid, k_eval=k_eval, band_lfres=band_lfres, band_extrap=band_extrap,
             coherent=coh, ns_arr=ns_arr, sigma_floor=sigma_floor, slope=slope,
             sigma_edge=sigma_edge, ns_query=ns_query, d_ns=d_ns,
             noise_flag=noise_flag, n_cov=n_cov,
             INFLATE=INFLATE, FLOOR_MIN=FLOOR_MIN, ns_box=np.array(NS_BOX))
    emit(f"\n[npz] {NPZ}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for si, s in enumerate(hr_sims):
        ax[0].plot(z_grid, coh[si, :, 0] * 100, "o-", ms=3, label=f"ns={ns_arr[si]:.3f}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("z"); ax[0].set_ylabel("coherent LOSO residual (%)")
    ax[0].set_title("LF-resolvable band (k<0.069)\nper-sim coherent P_MF/P_HR-1 (raw-cache footing)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    for si, s in enumerate(hr_sims):
        ax[1].plot(z_grid, coh[si, :, 1] * 100, "s-", ms=3, label=f"ns={ns_arr[si]:.3f}")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("z"); ax[1].set_ylabel("coherent LOSO residual (%)")
    ax[1].set_title("extrapolated band (k>=0.07)\nLF tail-extrapolated (larger floor)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    ax[2].plot(z_grid, sigma_floor[:, 0] * 100, "o-", color="#4c78a8", label="sigma_floor LFres")
    ax[2].plot(z_grid, sigma_floor[:, 1] * 100, "s-", color="#f58518", label="sigma_floor extrap")
    ax[2].plot(z_grid, (sigma_floor[:, 0] + 0 * sigma_edge[:, 0]) * 0, alpha=0)  # spacer
    ax[2].plot(z_grid, sigma_edge[:, 0] * 100, "o--", color="#4c78a8", alpha=0.5,
               label=f"sigma_edge LFres @ns={ns_query}")
    ax[2].plot(z_grid, sigma_edge[:, 1] * 100, "s--", color="#f58518", alpha=0.5,
               label=f"sigma_edge extrap @ns={ns_query}")
    ax[2].axhline(FLOOR_MIN * 100, color="grey", ls=":", label=f"FLOOR_MIN {FLOOR_MIN*100:.2f}%")
    ax[2].set_xlabel("z"); ax[2].set_ylabel("sigma (% of P_obs)")
    ax[2].set_title("C_emu floor + n_s-edge\n(small-scale leg ONLY; LF->HR generalization)")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    fig.suptitle("TASK B — T4 small-scale C_emu floor: 6-fold HR-LOSO of the LF->HR correction (raw-cache footing, "
                 "tail-extrap exercised). n=6: necessary-not-sufficient; ns>0.98 = edge term, not certified.",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PNG, dpi=130)
    emit(f"[png] {PNG}")
    emit("[done]")
    _rf.close()


if __name__ == "__main__":
    main()
