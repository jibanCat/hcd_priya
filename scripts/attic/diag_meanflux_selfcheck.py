# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# mean-flux forward-response self-check; self-described throwaway BUT it is the instrument behind the committed notes doc 2026-06-18-meanflux-forward-response-selfcheck.md, so ATTIC not DELETE (triage-table deviation, conservative direction, recorded in the freeze execution log).
"""DIAGNOSTIC (throwaway, uncommitted): does the production emulator/forward reproduce the
mean flux <F>(z) self-consistently at low z? Forward-only / profile-likelihood, no NUTS.

Build the production ctx exactly as run_prod_sbc_shard.py. For each held-out (fold-0) sim:
  - make_truth_from_sim(..., mf=ctx.mf, tau0_anchor="priya")  -> truth = REAL cache P1D + MF corr
  - make_legb_mock -> NOISELESS truth-on-leg (we use info["truth_on_leg"], not the noisy P_data,
    to isolate the SYSTEMATIC: the profile peak then is not moved by one cosmic-noise draw)
  - TEST (i) INJECTION-RECOVERY: profile the multi-leg loglik in tau0_amp ONLY (theta9, dtau0,
    alpha_hcd_z, a_SiIII, alpha_res ALL fixed at truth) on a fine grid -> argmax = recovered
    tau0_amp. Bias = recovered - truth. (everything-else-fixed => a non-zero bias is a pure
    forward/emulator mean-flux RESPONSE mismatch, not a degeneracy / nuisance absorption.)
  - TEST (ii) SELF-CONSISTENCY: at tau0_amp = truth, compare the mock GENERATING <F>(z)=exp(-tau_eff)
    (from the selected cache ladder row) to the forward emulator's effective <F>(z). Decompose by z.

Output: per-z and overall tau0_amp bias, in sigma (prior sd of the tau0_amp posterior ~ from
the curvature), and the per-z <F> offset.  Saves a figure.
"""
import glob, os, sys
import numpy as np

import hcd_analysis.emulator  # x64 before jax
import jax, jax.numpy as jnp
from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock,
    _mock_core_per_leg, _data_loglik_legcore, _kim, tau0_alpha_priya)
from hcd_analysis.emulator import data_likelihood as DL

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"
FIGDIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
os.makedirs(FIGDIR, exist_ok=True)

N_SIMS = int(os.environ.get("DIAG_NSIMS", "8"))
GRID_N = 121
AMP_LO, AMP_HI = 0.85, 1.15   # fine grid inside the [0.75,1.25] prior, around 1.0

members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
print(f"[build] {len(members)} ensemble members")
ctx, d = build_legb_ctx(
    ensemble_ckpts=members, use_xclass=True, with_mf=True, mf_with_floor=True,
    mf_emucoh=True, mf_emucoh_offdiag_only=True,
    with_eboss=True, metals_on=True, sample_metals=True, hierarchical_hcd=False)
zg = np.asarray(ctx.z_global)
kim_g = np.asarray(_kim(jnp.asarray(zg)))
sims, _ = held_out_sims(d, fold=0)
print(f"[build] legs={[l.name for l in ctx.legs]} n_held_out={len(sims)} grid=[{AMP_LO},{AMP_HI}]x{GRID_N}")


def make_loglik_fn(ctx, mock_legs, core_per_leg, truth_pack, dtau0):
    """Return a jitted scalar->scalar loglik(tau0_amp): theta9 / alpha_hcd / a_SiIII / alpha_res at
    TRUTH, only tau0_amp varied. Traced ONCE per sim, then evaluated cheaply over the grid.
    Mirrors _legb_model's forward: tau0_global = tau0_amp*((1+z)/(1+zp))^dtau0 * Kim."""
    theta9 = jnp.asarray(truth_pack["theta9"])
    alpha_hcd = jnp.asarray(truth_pack["alpha_hcd_z"])   # Z-RESOLVED truth incidence (n_zg,3), FIXED at
    #                    truth. Was the z-flat pivot (3,) -> broadcast to a spurious z-ramp vs the z-resolved
    #                    mock, which biased the recovered tau0 (this self-check was itself contaminated).

    def _ll(tau0_amp):
        alpha_z = tau0_alpha_priya(jnp.asarray(zg), tau0_amp, dtau0, z_pivot=ctx.tau0_pivot_z)
        tau0_global = alpha_z * jnp.asarray(kim_g)
        return _data_loglik_legcore(
            ctx, theta9, tau0_global, alpha_hcd, mock_legs, core_per_leg,
            a_siiii=0.0, alpha_res=None)
    return jax.jit(_ll)


def profile_one(sim, m):
    truth_sim = make_truth_from_sim(d, sim, fold=0, mf=ctx.mf, tau0_anchor="priya")
    k_mock = jax.random.fold_in(jax.random.PRNGKey(0), m)
    mock_legs, truth_pack, info = make_legb_mock(ctx, truth_sim, k_mock)
    core_per_leg = _mock_core_per_leg(ctx, truth_sim)
    amp_true = float(truth_sim["tau0_amp"]); dtau0_true = float(truth_sim["dtau0"])

    # NOISELESS profile: replace each leg's P_data by the noiseless truth-on-leg so the profile peak
    # is the SYSTEMATIC (no single-draw noise scatter). Keep the NaN pattern (dropped z stay NaN).
    noiseless_legs = []
    for leg in mock_legs:
        tol = np.asarray(info["truth_on_leg"][leg.name])
        Pd = np.array(leg.P_data, float)
        keep = np.isfinite(Pd)
        Pn = np.full_like(Pd, np.nan)
        Pn[keep] = tol[keep]
        noiseless_legs.append(leg._replace(P_data=Pn))

    amps = np.linspace(AMP_LO, AMP_HI, GRID_N)
    ll_fn = make_loglik_fn(ctx, noiseless_legs, core_per_leg, truth_pack, dtau0_true)
    lls = np.array([float(ll_fn(jnp.asarray(float(a)))) for a in amps])
    # parabolic fit near the peak -> MAP amp + curvature -> sigma_amp
    i0 = int(np.argmax(lls))
    lo, hi = max(0, i0 - 8), min(len(amps), i0 + 9)
    c2, c1, c0 = np.polyfit(amps[lo:hi], lls[lo:hi], 2)
    amp_map = -c1 / (2 * c2)
    sigma_amp = float(np.sqrt(-1.0 / (2 * c2))) if c2 < 0 else np.nan
    bias = amp_map - amp_true
    bias_sigma = bias / sigma_amp if np.isfinite(sigma_amp) else np.nan

    # --- self-consistency: per-z <F> at amp=truth.  Mock generating tau_eff = selected cache row
    #     (truth_sim["tau0"] at truth_sim["z"]); forward tau_eff = amp_true*...*Kim at z_global.
    z_sim = np.asarray(truth_sim["z"]); tau_sim = np.asarray(truth_sim["tau0"])
    F_gen = np.exp(-tau_sim)                                          # mock's generating <F>(z)
    alpha_fwd = float(amp_true) * ((1.0 + z_sim) / (1.0 + ctx.tau0_pivot_z)) ** dtau0_true
    tau_fwd = alpha_fwd * np.asarray(_kim(jnp.asarray(z_sim)))
    F_fwd = np.exp(-tau_fwd)                                          # forward param-curve <F>(z)
    dF_over_F = (F_fwd - F_gen) / F_gen                              # generating-vs-param-curve
    return dict(sim=sim, m=m, amp_true=amp_true, dtau0_true=dtau0_true,
                amp_map=amp_map, sigma_amp=sigma_amp, bias=bias, bias_sigma=bias_sigma,
                amps=amps, lls=lls, z_sim=z_sim, F_gen=F_gen, F_fwd=F_fwd, dF=dF_over_F,
                ns_true=float(truth_sim["params_unit"][0]))


recs = []
for m in range(min(N_SIMS, len(sims))):
    r = profile_one(sims[m], m)
    recs.append(r)
    zr = np.round(r["z_sim"], 2)
    print(f"[sim {m}] amp_true={r['amp_true']:.4f} dtau0={r['dtau0_true']:+.3f} "
          f"amp_MAP={r['amp_map']:.4f}  bias={r['bias']:+.4f} ({r['bias_sigma']:+.2f}sig "
          f"sd={r['sigma_amp']:.4f})  <F>param-vs-gen dF/F[lowz]="
          f"{np.array2string(r['dF'][:4]*100, precision=2)}%")

# aggregate
bias = np.array([r["bias"] for r in recs])
bsig = np.array([r["bias_sigma"] for r in recs])
amp_true = np.array([r["amp_true"] for r in recs])
# dln<F>/<F> implied by the tau0_amp bias at each z (the bias is amp -> tau scaling): dF/F = -dtau ~ -kim*alpha*dln(amp)
# overall: the absorbed mean-flux offset = - <kim*alpha> * bias  (per z); report the fractional <F> shift the
# forward 'wants' relative to truth, by z.
print("\n==== AGGREGATE (held-out-sim closure, tau0_amp profile, everything else at truth) ====")
print(f"  N sims                 : {len(recs)}")
print(f"  mean tau0_amp bias     : {bias.mean():+.4f}  (median {np.median(bias):+.4f})")
print(f"  mean bias in sigma     : {bsig.mean():+.3f}  (median {np.median(bsig):+.3f})")
print(f"  frac of <F> the fwd-amp implies (mean dlnF = -mean(amp*kim)*bias): see per-z below")

# per-z absorbed <F> offset: at the MAP amp the forward predicts <F> shifted vs truth by
#   dlnF(z) = -(tau_eff_map - tau_eff_truth) = -(amp_map-amp_true)*((1+z)/(1+zp))^dtau0 * Kim(z)
# average across sims at the GLOBAL z grid (interp each sim's contribution by nearest z).
zg_lo = zg
acc = {zz: [] for zz in zg_lo}
for r in recs:
    a_map, a_tru, dt = r["amp_map"], r["amp_true"], r["dtau0_true"]
    for zz in r["z_sim"]:
        alpha = a_tru * ((1.0 + zz) / (1.0 + ctx.tau0_pivot_z)) ** dt
        kimz = float(_kim(jnp.asarray(float(zz))))
        dtau = (a_map - a_tru) * ((1.0 + zz) / (1.0 + ctx.tau0_pivot_z)) ** dt * kimz
        dlnF = -dtau    # <F> shift the forward 'wants' (MAP) vs truth, fractional
        j = int(np.argmin(np.abs(zg_lo - zz)))
        acc[zg_lo[j]].append(dlnF * 100.0)
print("\n  per-z forward <F> offset the tau0_amp-MAP implies (%, =-dtau_eff):")
zflat = []
for zz in zg_lo:
    if acc[zz]:
        v = np.array(acc[zz]); zflat.append(v.mean())
        print(f"    z={zz:.1f}: dF/F = {v.mean():+.3f}%  (n={len(v)}, sd={v.std():.3f})")
zflat = np.array(zflat)
print(f"\n  z-FLAT? mean over z = {zflat.mean():+.3f}%, spread(sd over z) = {zflat.std():.3f}%")

# --- figure ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
# (1) profiles
for r in recs:
    ax[0].plot(r["amps"], r["lls"] - r["lls"].max(), lw=0.9, alpha=0.7)
    ax[0].axvline(r["amp_true"], color="k", lw=0.4, alpha=0.3)
ax[0].axvline(1.0, color="r", ls="--", lw=1, label="amp=1 (Kim)")
ax[0].set_xlabel("tau0_amp"); ax[0].set_ylabel("Δ loglik (per sim)")
ax[0].set_title("Profile loglik in tau0_amp\n(others at truth, noiseless)"); ax[0].set_ylim(-8, 0.5)
ax[0].legend(fontsize=8)
# (2) bias scatter
ax[1].axhline(0, color="k", lw=0.6)
ax[1].scatter(amp_true, bias * 100, c="C0")
ax[1].set_xlabel("tau0_amp truth"); ax[1].set_ylabel("tau0_amp bias (MAP-truth) [%]")
ax[1].set_title(f"tau0_amp recovery bias\nmean={bias.mean()*100:+.2f}% ({bsig.mean():+.2f}sig)")
# (3) per-z forward <F> offset
zz_have = [zz for zz in zg_lo if acc[zz]]
ax[2].axhline(0, color="k", lw=0.6)
ax[2].plot(zz_have, zflat, "o-", color="C2")
ax[2].set_xlabel("z"); ax[2].set_ylabel("forward <F> offset at MAP [%]")
ax[2].set_title(f"per-z <F> offset (z-flat?)\nmean={zflat.mean():+.3f}% sd={zflat.std():.3f}%")
fig.tight_layout()
out = os.path.join(FIGDIR, "meanflux_selfcheck_tau0amp_profile.png")
fig.savefig(out, dpi=110); print(f"\n[fig] {out}")
np.savez(os.path.join(FIGDIR, "meanflux_selfcheck_tau0amp_profile.npz"),
         bias=bias, bias_sigma=bsig, amp_true=amp_true,
         z=np.array(zz_have), dF_perz=zflat)
print("[npz] saved")
