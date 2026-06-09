"""LYA-LENS root-cause probe (FAST): is the coherent n_s-tilt EMU bias tau0-asymmetric?

Feedback #1: two-stage head accurate at HIGH tau0 but not LOW tau0 -> via tau0<->n_s
degeneracy a NEGATIVE n_s bias. Project the per-row (emu-truth) clean-logP residual onto
the emulator's own d logP_clean/d n_s direction; the coeff c = n_s-equivalent mis-tilt
(c<0 => emulator under-tilts => its n_s too low). Bin c by tau0 tercile. If LOW-tau0 c is
materially more negative than HIGH-tau0 c -> #1 supported.

FAST: one jitted per-row function (theta,z_u,t0)->(logP_clean, u_ns) reused across rows
(no per-row recompile); subsample sims/fold; probe z in {2.4,3.0,3.6,4.2}.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_lya_ns_tau0_split.py
"""
import numpy as np
import hcd_analysis.emulator  # x64 before jax
import jax, jax.numpy as jnp
import equinox as eqx
from functools import partial
from hcd_analysis.emulator.closure_legb import build_legb_ctx, held_out_sims
from hcd_analysis.emulator import predict as P
from hcd_analysis.emulator.data import safe_log
from hcd_analysis.emulator.inference import PARAM_NAMES

REPO = "/home/mfho/hcd_priya"
NS_I = int(np.where(np.array(PARAM_NAMES) == "ns")[0][0])
HOLD0 = f"{REPO}/checkpoints/error_vector_xclass_holdout0.npz"
KLO, KHI = 1e-3, 0.06
Z_PROBE = [2.4, 3.0, 3.6, 4.2]
N_SIM_PER_FOLD = 4    # subsample held-out sims per fold for speed


def make_fn(model, pf):
    @eqx.filter_jit
    def fn(theta, z_u, t0):
        def logclean(th):
            Pf = P.predict_P_filt(model, th, z_u, t0, pf)  # (4,K)
            return jnp.log(Pf[0])
        lp = logclean(theta)
        u = jax.jacrev(lambda nsv: logclean(theta.at[NS_I].set(nsv)))(theta[NS_I])
        return lp, u
    return fn


def run():
    rows_c, rows_t, rows_z = [], [], []
    for fold in range(8):
        ckpt = f"{REPO}/checkpoints/final_fold{fold}"
        ctx, d = build_legb_ctx(ckpt=ckpt, xclass_error_vector=HOLD0)
        fn = make_fn(ctx.model, ctx.pf_stats)
        kf = np.asarray(d["kfkms"]); kf0 = kf[0] if kf.ndim == 2 else kf
        sims, _ = held_out_sims(d, fold)
        sims = sims[:N_SIM_PER_FOLD]
        simset = set(sims)
        names = d["sim_name"]; params_u = d["params_unit"]; z_grid = d["z_grid"]
        tau0 = d["tau0"]; Pfilt = d["P_filt"]; x = d["x"]
        idx = np.where([nm in simset for nm in names])[0]
        for r in idx:
            z = float(z_grid[r])
            if not any(abs(z - zp) < 0.05 for zp in Z_PROBE):
                continue
            kk = kf[r] if kf.ndim == 2 else kf0
            km = (kk >= KLO) & (kk <= KHI) & np.isfinite(Pfilt[r, 0])
            if km.sum() < 5:
                continue
            lp, u = fn(jnp.asarray(params_u[r]), jnp.asarray(float(x[r, 9])),
                       jnp.asarray(float(tau0[r])))
            logP_emu = np.asarray(lp); u_ns = np.asarray(u)
            resid = logP_emu - safe_log(Pfilt[r, 0])
            uu = u_ns[km]; rr = resid[km]
            denom = float(uu @ uu)
            if denom <= 0:
                continue
            rows_c.append(float(uu @ rr) / denom)
            rows_t.append(float(tau0[r])); rows_z.append(z)
    return np.array(rows_c), np.array(rows_t), np.array(rows_z)


c, t0, z = run()
print(f"# n rows projected: {len(c)}")
print("# c = n_s-equivalent mis-tilt of (emu-truth) clean logP, unit-cube n_s; c<0 => emu under-tilts")
print()
qs = np.quantile(t0, [1/3, 2/3])
lo = t0 <= qs[0]; mid = (t0 > qs[0]) & (t0 <= qs[1]); hi = t0 > qs[1]


def stat(m, lab):
    x = c[m]
    se = x.std(ddof=1) / np.sqrt(max(len(x), 1)) if len(x) > 1 else float("nan")
    print(f"  {lab:22s} n={len(x):4d}  <c>={x.mean():+.4f} +/- {se:.4f}  median={np.median(x):+.4f}")


print("=== tau0 tercile split (tau0=-log<F>; LOW tau0 = high mean flux) ===")
stat(lo, f"LOW tau0 (<{qs[0]:.2f})"); stat(mid, "MID tau0"); stat(hi, f"HIGH tau0 (>{qs[1]:.2f})")
print("\n=== pooled ===")
stat(np.ones(len(c), bool), "ALL")
print("\n=== by z (all tau0) ===")
for zp in Z_PROBE:
    m = np.abs(z - zp) < 0.05
    if m.sum() > 0:
        stat(m, f"z={zp}")
dlh = c[lo].mean() - c[hi].mean()
print(f"\n# DISCRIMINATOR <c>_low_tau0 - <c>_high_tau0 = {dlh:+.4f}")
print("#   << 0 => feedback #1 SUPPORTED (low-tau0 tilt deficit); ~0 => #1 weak (tau0-symmetric)")
