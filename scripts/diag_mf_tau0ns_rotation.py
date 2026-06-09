"""TASK A — quantify the τ₀–n_s correlation-rotation flag the T3 gate raised.

The gate (emu_bias_allfolds_mf_aliasing.txt) found |r(τ₀,n_s)(ON) − r(LF)| worst
= 0.129 (> the 0.05 §3.2 literal clause), but traced it to the SEPARABLE correction
(|r(OFF−LF)| ≈ |r(ON−LF)| every fold), NOT the rank-1 term (|r(ON−OFF)| worst 0.014).
G4.1 closure n_s is +0.026σ (unbiased). The OPEN question this script answers: does
the ~0.13 rotation WIDEN or TIGHTEN the marginal σ(n_s)/σ(τ₀)? A widening is
conservative (cannot under-cover); a tightening would be the dangerous case.

This reuses scripts/diag_mf_gate_aliasing_probes.py::cpost_for_sim (the SAME Fisher
C_post = (F+P)^{-1} the gate built) under three forwards — LF, MF-OFF (pure separable),
MF-ON (separable+rank-1) — and reports the ABSOLUTE marginal widths + the r(τ₀,n_s),
per fold, that the aliasing txt did not print. Forward-only Fisher; no NUTS; no commit.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_mf_tau0ns_rotation.py
"""
import importlib.util as _ilu
import numpy as np

import hcd_analysis.emulator  # x64 before jax
import jax  # noqa: F401

from hcd_analysis.emulator.closure_legb import build_legb_ctx, held_out_sims
from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator.inference import PARAM_NAMES

# Import the gate's wired forward + cpost_for_sim verbatim (do NOT re-implement).
_spec = _ilu.spec_from_file_location(
    "mfprobe", "/home/mfho/hcd_priya/scripts/diag_mf_gate_aliasing_probes.py")
# the probe module runs a full analysis on import; instead load only the funcs we need
# by importing the gate-bias module's forward and re-using cpost_for_sim's definition.
# cpost_for_sim lives in the probe module but that module executes on import. To avoid
# re-running it, we replicate the tiny import surface here by exec'ing only the defs.
import types
_probe_src = open(
    "/home/mfho/hcd_priya/scripts/diag_mf_gate_aliasing_probes.py").read()
# cut the script at the first top-level emit() driver call so only defs execute.
_cut = _probe_src.index('emit("# T3 GATE aliasing probes')
_mod = types.ModuleType("mfprobe_defs")
_mod.__dict__["__name__"] = "mfprobe_defs"
exec(compile(_probe_src[:_cut], "diag_mf_gate_aliasing_probes.py(defs)", "exec"),
     _mod.__dict__)
cpost_for_sim = _mod.cpost_for_sim
build_heads = _mod.build_heads

REPO = "/home/mfho/hcd_priya"
FIGDIR = f"{REPO}/figures/analysis/04_emulator"
RESULTS = f"{FIGDIR}/mf_tau0ns_rotation.txt"
NPZ = f"{FIGDIR}/mf_tau0ns_rotation.npz"
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
KS_KMAX, KS_ZLO = 0.06, 2.4
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])

_rf = open(RESULTS, "w")
def emit(s): print(s, flush=True); _rf.write(s + "\n"); _rf.flush()


emit("# TASK A — τ₀–n_s rotation: ABSOLUTE marginal σ(n_s)/σ(τ₀) + r under LF/OFF/ON.")
emit("# C_post = (F+P)^{-1}, SAME Fisher as the gate. τ₀-block = z∈[2.4,3.4] diag mean.")
emit("# Widening (σ_MF/σ_LF ≥ 1) is conservative; tightening (<1) is the dangerous case.")
emit("# one representative held-out sim per fold (matches the gate aliasing probe).")
emit("#")
emit("# fold sim                       σ(n_s)  σ(n_s) σ(n_s)  | σ(τ0) σ(τ0) σ(τ0) | r(τ0,n_s)")
emit("#                                  LF    OFF    ON      |  LF    OFF    ON   | LF   OFF   ON")

lf_cache = MF.load_cache(MF.LF_CACHE); hr_cache = MF.load_cache(MF.HR_CACHE)
pairs = MF.match_hr_to_lf(lf_cache, hr_cache)
hr_sims = sorted(set(s.decode() if isinstance(s, bytes) else s
                     for s in hr_cache["sim_name"]))
hr_row_sim = np.array([s.decode() if isinstance(s, bytes) else s
                       for s, _ in [(hr_cache["sim_name"][h], l) for h, l in pairs]])

rows = []  # (fold, sNs_lf, sNs_off, sNs_on, sT_lf, sT_off, sT_on, r_lf, r_off, r_on)
for fold in range(8):
    ckpt = f"{REPO}/checkpoints/final_fold{fold}"
    ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0,
                            ks_kwargs={"k_max": KS_KMAX, "z_lo": KS_ZLO})
    sims, _ = held_out_sims(d, fold)
    held = [s for s in sims if s in hr_sims]
    train_rows = (np.where(~np.isin(hr_row_sim, held))[0]
                  if held else np.arange(len(pairs)))
    fm, fn, llk, tgf, lr = build_heads(fold, lf_cache, hr_cache, pairs, hr_sims)
    cf = MF.fixed_mean_table_resolved(tgf, lr, train_mask_rows=train_rows)

    def mk(rank1):
        ak = cf["a_k"] if rank1 else np.zeros_like(cf["a_k"])
        h = MF.FixedMeanHead(cf["gbar_z_tab"], cf["z_tab"], resolved=True,
            gtau_tab=cf["gtau_tab"], tau_tab=cf["tau_tab"], tau_by_z=cf["tau_by_z"],
            a_k=ak, u_z=cf["u_z"], u_tau=cf["u_tau"])
        return MF.build_multifidelity(fm, fn, llk, h, eval_logk=np.asarray(llk),
                                      log_rho=lr, delta_mode="none")
    mf_on, mf_off = mk(True), mk(False)
    sim = sims[0]
    Cp_on, t0idx, zg = cpost_for_sim(ctx, d, sim, fold, mf=mf_on)
    Cp_off, _, _ = cpost_for_sim(ctx, d, sim, fold, mf=mf_off)
    Cp_lf, _, _ = cpost_for_sim(ctx, d, sim, fold, mf=None)

    band = (zg >= 2.4) & (zg <= 3.4)
    tband = t0idx[band]
    # marginal n_s width
    sNs = lambda Cp: float(np.sqrt(Cp[NS_I, NS_I]))
    # τ₀-block marginal width (RMS of the per-z τ₀ diag in the data band — matches the
    # gate aliasing probe's σ_τ0 definition exactly).
    sT = lambda Cp: float(np.sqrt(np.mean(np.diag(Cp)[tband])))
    # representative τ₀–n_s correlation: the worst-|·| over the band z-bins, signed
    # by the value AT that worst bin (so the rotation direction is reported, not just |·|).
    def rval(Cp):
        rr = np.array([Cp[i, NS_I] / np.sqrt(Cp[i, i] * Cp[NS_I, NS_I]) for i in tband])
        return float(rr[np.argmax(np.abs(rr))])
    row = (fold, sNs(Cp_lf), sNs(Cp_off), sNs(Cp_on),
           sT(Cp_lf), sT(Cp_off), sT(Cp_on),
           rval(Cp_lf), rval(Cp_off), rval(Cp_on))
    rows.append(row)
    emit(f"  {fold}  {sim[:26]:26s} "
         f"{row[1]:.4f} {row[2]:.4f} {row[3]:.4f} | "
         f"{row[4]:.4f} {row[5]:.4f} {row[6]:.4f} | "
         f"{row[7]:+.3f} {row[8]:+.3f} {row[9]:+.3f}")

R = np.array([r[1:] for r in rows])  # (8,9)
sNs_lf, sNs_off, sNs_on = R[:, 0], R[:, 1], R[:, 2]
sT_lf, sT_off, sT_on = R[:, 3], R[:, 4], R[:, 5]
r_lf, r_off, r_on = R[:, 6], R[:, 7], R[:, 8]

emit("\n# ===== SUMMARY (ratio MF/LF; >1 = WIDENING = conservative) =====")
emit(f"  σ(n_s) ratio ON/LF : mean {np.mean(sNs_on/sNs_lf):.4f}  "
     f"range [{np.min(sNs_on/sNs_lf):.4f}, {np.max(sNs_on/sNs_lf):.4f}]")
emit(f"  σ(n_s) ratio OFF/LF: mean {np.mean(sNs_off/sNs_lf):.4f}  "
     f"range [{np.min(sNs_off/sNs_lf):.4f}, {np.max(sNs_off/sNs_lf):.4f}]")
emit(f"  σ(n_s) ratio ON/OFF: mean {np.mean(sNs_on/sNs_off):.4f}  "
     f"range [{np.min(sNs_on/sNs_off):.4f}, {np.max(sNs_on/sNs_off):.4f}]")
emit(f"  σ(τ0)  ratio ON/LF : mean {np.mean(sT_on/sT_lf):.4f}  "
     f"range [{np.min(sT_on/sT_lf):.4f}, {np.max(sT_on/sT_lf):.4f}]")
emit(f"  σ(τ0)  ratio OFF/LF: mean {np.mean(sT_off/sT_lf):.4f}  "
     f"range [{np.min(sT_off/sT_lf):.4f}, {np.max(sT_off/sT_lf):.4f}]")
emit(f"  σ(τ0)  ratio ON/OFF: mean {np.mean(sT_on/sT_off):.4f}  "
     f"range [{np.min(sT_on/sT_off):.4f}, {np.max(sT_on/sT_off):.4f}]")
emit("")
emit(f"  Δr(ON−LF)  : mean {np.mean(r_on-r_lf):+.4f}  worst|·| {np.max(np.abs(r_on-r_lf)):.4f}")
emit(f"  Δr(OFF−LF) : mean {np.mean(r_off-r_lf):+.4f}  worst|·| {np.max(np.abs(r_off-r_lf)):.4f}")
emit(f"  Δr(ON−OFF) : mean {np.mean(r_on-r_off):+.4f}  worst|·| {np.max(np.abs(r_on-r_off)):.4f}")
emit("")
# verdict logic
wid_ns = float(np.min(sNs_on / sNs_lf))   # the smallest n_s ratio -> if ≥ ~1 it never tightens
wid_t = float(np.min(sT_on / sT_lf))
emit(f"  VERDICT inputs: min σ(n_s) ON/LF = {wid_ns:.4f}; min σ(τ0) ON/LF = {wid_t:.4f}")
emit(f"  (ratio ≥ 1.0 in EVERY fold ⇒ the rotation only WIDENS — conservative, cannot under-cover)")

np.savez(NPZ, rows=R, fold=np.array([r[0] for r in rows]),
         cols=np.array(["sNs_lf", "sNs_off", "sNs_on", "sT_lf", "sT_off", "sT_on",
                        "r_lf", "r_off", "r_on"]))
emit(f"\n[npz] {NPZ}")
emit("[done]")
_rf.close()
