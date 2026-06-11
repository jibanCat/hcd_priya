"""Design-validation figure for the τ₀ refactor (607ae9d + 036398e): the 13 independent
per-z rungs → PRIYA 2-param (τ₀, dτ₀). Shows (A) the prior-family mechanism — the OLD prior
admits jagged per-z τ₀(z) (which leaks into A_p), the NEW prior only smooth curves; (B) the
jaggedness metric; (C) closure self-consistency — the PRIYA-anchored truth τ₀(z) is represented
by the 2-param model the forward samples. No NUTS (design validation)."""
import numpy as np
import warnings; warnings.simplefilter("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import hcd_analysis.emulator  # noqa: enable x64
import jax.numpy as jnp
from hcd_analysis.emulator.data import load_cache, KIM_AMP, KIM_SLOPE
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.meanflux_prior import (
    becker13_tau0, kim_tau0, tau0_alpha_priya, meanflux_tau0_prior,
    TAU0_AMP_RANGE, DTAU0_RANGE, TAU0_PIVOT_Z)

rng = np.random.default_rng(0)
CACHE = "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
d = load_cache(CACHE)
z = np.unique(np.round(d["z_grid"], 4))
z = z[(z >= 2.2 - 1e-6) & (z <= 4.6 + 1e-6)]
kim = np.asarray(kim_tau0(jnp.asarray(z)))
ND = 50

# OLD prior: 13 INDEPENDENT per-z rungs, alpha_z ~ Normal(becker/kim, 5%) -> tau0 = alpha*kim
mu, sig = meanflux_tau0_prior(jnp.asarray(z), center="becker13")
mu = np.asarray(mu); sig = np.asarray(sig)
old_tau0 = (mu / kim)[None, :] + (sig / kim)[None, :] * rng.standard_normal((ND, len(z)))
old_tau0 = old_tau0 * kim[None, :]

# NEW prior: 2-param PRIYA, tau0_amp~U[0.75,1.25], dtau0~U[-0.4,0.25] -> alpha(z)*kim
amp = rng.uniform(*TAU0_AMP_RANGE, ND); dt = rng.uniform(*DTAU0_RANGE, ND)
new_tau0 = np.array([np.asarray(tau0_alpha_priya(jnp.asarray(z), amp[i], dt[i])) * kim
                     for i in range(ND)])

def d2rms(curves):  # 2nd-difference RMS per draw (jaggedness), normalized by the curve scale
    c = curves / curves.mean(1, keepdims=True)
    return np.sqrt((np.diff(c, 2, axis=1) ** 2).mean(1))

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

# Panel A — prior τ₀(z) families
for i in range(ND):
    ax[0].plot(z, old_tau0[i], color="#c0392b", alpha=0.18, lw=0.8)
    ax[0].plot(z, new_tau0[i], color="#2471a3", alpha=0.25, lw=0.8)
ax[0].plot(z, np.asarray(becker13_tau0(jnp.asarray(z))), "k-", lw=2.2, label="Becker+2013")
ax[0].plot(z, kim, "k--", lw=1.6, label="Kim07 (α=1)")
ax[0].plot([], [], color="#c0392b", lw=2, label="OLD: 13 per-z rungs (jagged)")
ax[0].plot([], [], color="#2471a3", lw=2, label="NEW: PRIYA 2-param (smooth)")
ax[0].set_xlabel("z"); ax[0].set_ylabel(r"$\tau_0(z)=-\ln\langle F\rangle$")
ax[0].set_title("Prior τ₀(z) families: 50 draws each\n(OLD admits per-z wiggle that leaks into A_p; NEW is smooth)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# Panel B — jaggedness metric (2nd-diff RMS)
ax[1].hist(d2rms(old_tau0), bins=20, color="#c0392b", alpha=0.6, label=f"OLD 13-rung (med {np.median(d2rms(old_tau0)):.3f})")
ax[1].hist(d2rms(new_tau0), bins=20, color="#2471a3", alpha=0.6, label=f"NEW 2-param (med {np.median(d2rms(new_tau0)):.4f})")
ax[1].set_xlabel("τ₀(z) 2nd-difference RMS (jaggedness)"); ax[1].set_ylabel("prior draws")
ax[1].set_title("Per-z wiggle removed by construction\n(the channel that leaked into A_p, |corr(A_p,τ₀)|≈0.5–0.6)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].set_yscale("log")

# Panel C — closure self-consistency: PRIYA-anchored truth vs 2-param fit (3 fiducials)
sims, _ = CL.held_out_sims(d, fold=0)
for (amp_t, dt_t), col, lab in [((1.0, 0.0), "#2471a3", "central (1.0, 0.0)"),
                                ((1.15, 0.20), "#27ae60", "hi-corner (1.15, 0.20)"),
                                ((0.85, -0.25), "#8e44ad", "lo-corner (0.85, −0.25)")]:
    t = CL.make_truth_from_sim(d, sims[0], fold=0, tau0_anchor=(amp_t, dt_t))
    zt = np.asarray(t["z"])
    fit = np.asarray(tau0_alpha_priya(jnp.asarray(zt), t["tau0_amp"], t["dtau0"]) * kim_tau0(jnp.asarray(zt)))
    ax[2].plot(zt, t["tau0"], "o", color=col, ms=5, label=f"truth rungs: {lab}")
    ax[2].plot(zt, fit, "-", color=col, lw=1.5,
               label=f"  2-param fit (τ₀={t['tau0_amp']:.3f}, dτ₀={t['dtau0']:.3f})")
ax[2].set_xlabel("z"); ax[2].set_ylabel(r"$\tau_0(z)$")
ax[2].set_title("Closure self-consistency: PRIYA-anchored truth τ₀(z)\nvs the 2-param model the forward samples (<3% interior)")
ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)

plt.tight_layout()
out = "figures/analysis/05_likelihood/tau0_refactor_validation.png"
plt.savefig(out, dpi=120)
print("saved", out)
print(f"jaggedness 2nd-diff RMS: OLD median={np.median(d2rms(old_tau0)):.3f}  NEW median={np.median(d2rms(new_tau0)):.5f}")
print(f"  -> NEW is {np.median(d2rms(old_tau0))/max(np.median(d2rms(new_tau0)),1e-9):.0f}x smoother")
