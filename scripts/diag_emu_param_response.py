"""Emulator P1D response d ln P_clean / d theta (unit cube) across k, via autodiff.

Shows WHICH parameters move the P1D and with WHAT k-shape. Motivation (2026-06-11): hub has the
LARGEST response of any param (RMS ~0.2-0.3 across its prior range, > A_p) but as a COHERENT
scale-dependent tilt/swap (the velocity-rescaling 'bin swap') that is DEGENERATE with A_p/n_s -> it
is weakly *constrained* though strongly *felt*; the shared-latent neural emulator renders it smoothly
(a per-bin GP without cross-bin C_emu cannot). bhfeedback response ~0 = a null direction (unconstrained
by physics, not emulation). Output: figures/analysis/05_likelihood/emu_hub_bhfb_response.png.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_emu_param_response.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hcd_analysis.emulator  # x64
import jax, jax.numpy as jnp
from hcd_analysis.emulator.closure_legb import build_legb_ctx
from hcd_analysis.emulator.predict import predict_P_filt
from hcd_analysis.emulator.meanflux_prior import becker13_tau0
from hcd_analysis.emulator.data import Z_LIMITS

ctx, d = build_legb_ctx()
model, pf = ctx.model, ctx.pf_stats
k = np.asarray(ctx.cache_k)
inb = (k >= 1e-3) & (k <= 0.08)
names = ["ns","Ap","herei","heref","alphaq","hub","omegamh2","hireionz","bhfeedback"]
th0 = jnp.full(9, 0.5)                      # box-centre fiducial (unit cube)
SHOW = {"ns":"#000000","Ap":"#1f77b4","hub":"#d62728","bhfeedback":"#2ca02c"}
zs = [2.4, 3.0, 3.8]

fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for iz, z in enumerate(zs):
    zu = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    tau0 = float(becker13_tau0(jnp.asarray(z)))
    def lnPclean(th):
        return jnp.log(predict_P_filt(model, th, jnp.asarray(zu), jnp.asarray(tau0), pf)[0])
    J = np.asarray(jax.jacfwd(lnPclean)(th0))          # (K,9) d lnP_clean / d theta_unit
    a = ax[iz]
    for nm, c in SHOW.items():
        a.plot(k[inb], J[inb, names.index(nm)], "-", color=c, lw=2,
                label=nm if iz == 0 else None)
    a.axhline(0, color="gray", lw=0.6)
    a.set_xscale("log"); a.set_xlabel("k [s/km]"); a.set_title(f"z = {z}", fontsize=12)
    a.grid(alpha=0.25, which="both")
ax[0].set_ylabel(r"$\partial\,\ln P_{\rm clean}/\partial\theta_{\rm unit}$  (response across the full prior range)")
ax[0].legend(fontsize=10, title="param (unit)")
fig.suptitle("Emulator P1D response across the FULL prior range of each parameter (shared-latent ⇒ smooth in k)",
             fontsize=13, y=1.0)
fig.text(0.5, -0.03, "h (red): a COHERENT, low-k-weighted response (the velocity-rescaling 'bin swap') — smooth across k, "
         "captured by the shared latent. bhfeedback (green): ~flat/tiny → why it's prior-dominated. "
         "n_s/A_p shown for scale.", ha="center", fontsize=9, style="italic")
fig.tight_layout()
out = "figures/analysis/05_likelihood/emu_hub_bhfb_response.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
# magnitudes: RMS response over in-band k per param, per z
for z in zs:
    zu=(z-Z_LIMITS[0])/(Z_LIMITS[1]-Z_LIMITS[0]); tau0=float(becker13_tau0(jnp.asarray(z)))
    J=np.asarray(jax.jacfwd(lambda th: jnp.log(predict_P_filt(model,th,jnp.asarray(zu),jnp.asarray(tau0),pf)[0]))(th0))
    rms={nm:float(np.sqrt(np.nanmean(J[inb,names.index(nm)]**2))) for nm in SHOW}
    print(f"z={z}: RMS dlnP/dunit  " + "  ".join(f"{nm}={rms[nm]:.3f}" for nm in SHOW))
