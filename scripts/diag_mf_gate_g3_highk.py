"""T3 GATE — G3: the z=2.8-3.4 high-k (k≥0.0442 s/km) signed-coherent logP residual,
LF-only forward vs the MF forward. The MF correction's JOB is to REDUCE this deficit.

The LF->HR deficit at each matched HR row is the measured log-ratio g = logP_HR - logP_LF
(measure_delta_targets). The LF-only forward leaves the FULL deficit (residual = g). The MF
forward applies the fixed-mean correction g_corr = mf.g (HF-LOSO), so its residual is
g - g_corr. G3 PASSES if |mean(g - g_corr)| < |mean(g)| in z∈[2.8,3.4], k≥0.0442, over the
6 HR sims (signed coherent mean — the band the real fit's high-k systematic lives in).

This is the through-MF analog of mf_rescorr_loso (the worst-per-sim coherent residual that
sizes the T4 floor), restricted to the G3 target band. Per HR sim, the correction is fit
EXCLUDING that sim (HF-LOSO) so the residual is the honest LF->HR generalization.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_gate_g3_highk.py
"""
import numpy as np
import hcd_analysis.emulator
import jax.numpy as jnp
from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator.data import Z_LIMITS

REPO = "/home/mfho/hcd_priya"
RESULTS = f"{REPO}/figures/analysis/04_emulator/emu_bias_allfolds_mf_g3.txt"
K_HIGH = 0.0442   # s/km, the G3 high-k floor
Z_LO, Z_HI = 2.8, 3.4

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()

lf_cache = MF.load_cache(MF.LF_CACHE); hr_cache = MF.load_cache(MF.HR_CACHE)
pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
hr_row_sim = np.array([s.decode() if isinstance(s, bytes) else s
                       for s, _ in [(hr_cache["sim_name"][h], l) for h, l in pairs]])
hr_sims = sorted(set(hr_row_sim))

# fold-0 LF backbone for the measured g (g is the LF-emulator->HR-truth log-ratio; the
# backbone only enters logP_LF; for the deficit MAGNITUDE any final fold is representative.
fm, fmeta, fn, lf_logk = MF.load_lf_backbone(0)
ck = 10 ** np.asarray(lf_logk)
tg = MF.measure_delta_targets(lf_cache, hr_cache, fm, fn, lf_logk, np.asarray(lf_logk), pairs)
g = tg["g"]                                    # (M,4,K) measured LF->HR log-ratio
z_row = np.round(tg["x"][:, 9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0], 2)
log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, np.asarray(lf_logk)), nan=0.0)

# band masks
kmask = ck >= K_HIGH
zband = (z_row >= Z_LO) & (z_row <= Z_HI)
basis = MF.smooth_k_basis(jnp.asarray(lf_logk), 4)

emit(f"# G3: z∈[{Z_LO},{Z_HI}], k≥{K_HIGH} s/km signed-coherent logP residual, class 0 (clean).")
emit(f"#   LF-forward residual = g (full deficit); MF-forward residual = g - g_corr (HF-LOSO).")
emit(f"#   n high-k bins (k≥{K_HIGH}): {int(kmask.sum())}/{len(ck)};  rows in z-band: {int(zband.sum())}")

def head_g_for_row(comp, lr, x_row, tau0_row):
    head = MF.FixedMeanHead(comp["gbar_z_tab"], comp["z_tab"], resolved=True,
        gtau_tab=comp["gtau_tab"], tau_tab=comp["tau_tab"], tau_by_z=comp["tau_by_z"],
        a_k=comp["a_k"], u_z=comp["u_z"], u_tau=comp["u_tau"])
    cond = MF.make_cond(jnp.asarray(x_row), jnp.asarray(float(tau0_row)))
    return np.asarray(lr[None, :] + head(cond, basis))    # (4,K) = g_corr

lf_resid_all, mf_resid_all = [], []
emit("# sim                         LF |mean resid|   MF |mean resid|   reduced?")
for s in hr_sims:
    sel = zband & (hr_row_sim == s)
    if not sel.any():
        continue
    # HF-LOSO correction: fit excluding THIS sim's rows.
    train = np.where(hr_row_sim != s)[0]
    comp = MF.fixed_mean_table_resolved(tg, log_rho, train_mask_rows=train)
    rows = np.where(sel)[0]
    lf_r, mf_r = [], []
    for ri in rows:
        gc = head_g_for_row(comp, log_rho, tg["x"][ri], tg["tau0"][ri])[0]   # class 0 (clean)
        gm = g[ri, 0]                                          # measured deficit (class 0)
        fin = np.isfinite(gm) & kmask
        lf_r.append(gm[fin]); mf_r.append((gm - gc)[fin])
    lf_r = np.concatenate(lf_r); mf_r = np.concatenate(mf_r)
    lf_resid_all.append(lf_r); mf_resid_all.append(mf_r)
    lf_m = abs(np.mean(lf_r)); mf_m = abs(np.mean(mf_r))
    emit(f"  {s[:26]:26s}  {lf_m:.4f}          {mf_m:.4f}         {'YES' if mf_m < lf_m else 'NO'}")

lf_all = np.concatenate(lf_resid_all); mf_all = np.concatenate(mf_resid_all)
lf_mean = np.mean(lf_all); mf_mean = np.mean(mf_all)
emit(f"\n# POOLED (6-sim HF-LOSO) signed coherent mean in the G3 band:")
emit(f"  LF-forward |mean residual| = {abs(lf_mean):.4f}  (raw deficit, signed mean {lf_mean:+.4f})")
emit(f"  MF-forward |mean residual| = {abs(mf_mean):.4f}  (after correction, signed mean {mf_mean:+.4f})")
emit(f"  reduction factor = {abs(lf_mean)/max(abs(mf_mean),1e-9):.1f}x")
g3 = abs(mf_mean) < abs(lf_mean)
emit(f"  G3: {'PASS' if g3 else 'FAIL'}  (MF |mean| {'<' if g3 else '≥'} LF |mean|)")
emit("[done]")
_rf.close()
