#!/usr/bin/env python3
"""Does the MF (LF→HR resolution) correction reach the eBOSS low-k band and tilt n_s?

The MF correction is a TILT (the LF box is power-deficient at high-k vs HR; ~+4% low-k → −6% high-k).
eBOSS is low-k (k∈[0.0011,0.0195] s/km), so the question (PI 2026-06-13; Fernandez+2024 ran MF) is
whether the LOW-k end of that tilt is large enough at eBOSS k to shift the recovered n_s.

This is a FORWARD check (no NUTS): at a fixed (truth) cosmology, compare the eBOSS-leg forward power
with the MF correction ON vs OFF, per (z,k). Report the fractional imprint P_MF/P_LF−1 at the eBOSS
k-extremes (the tilt) and the implied Δn_s ≈ d ln(P_MF/P_LF)/d ln(k) over the eBOSS band (a tilt in
the multiplicative ratio aliases directly into the power-law tilt n_s).
"""
import sys
import numpy as np

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import hcd_analysis.emulator  # noqa: F401
import jax.numpy as jnp
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.closure_legb import build_legb_ctx, make_truth_from_sim, load_cache, CACHE_PATH


def main():
    # fold6 (Planck-ish) MF correction + the eBOSS leg.
    ctx, d = build_legb_ctx(ckpt="/home/mfho/hcd_priya/checkpoints/final_fold6", with_eboss=True,
                            with_mf=True, mf_fold=6, mf_with_floor=False)
    leg = next(l for l in ctx.legs if l.name == "eBOSS")
    # truth θ for the Planck-ish held-out sim (E_f6 uses the fold6 sim closest to n_s 0.966).
    from hcd_analysis.emulator.closure_legb import held_out_sims
    from hcd_analysis.emulator.data import PARAM_LIMITS
    sims, _ = held_out_sims(d, fold=6)
    # pick the sim closest to ns 0.966
    def _ns(s):
        import numpy as _np
        sn = _np.asarray(d["sim_name"]); pu = d["params_unit"]; lo, hi = PARAM_LIMITS[0]
        return lo + pu[_np.where(sn == s)[0][0], 0] * (hi - lo)
    sim = min(sims, key=lambda s: abs(_ns(s) - 0.966))
    truth = make_truth_from_sim(d, sim, fold=6, tau0_anchor="priya", mf=None)
    th = jnp.asarray(truth["params_unit"])
    tau0 = jnp.asarray(truth["tau0"][:leg.n_z]) if len(truth["tau0"]) >= leg.n_z else jnp.full(leg.n_z, 0.9)
    tau0 = jnp.full(leg.n_z, float(jnp.mean(tau0)))     # flat fiducial τ₀ for the comparison
    alpha = jnp.asarray(truth["w_c"])

    kw = dict(pf_stats=ctx.pf_stats, dla_core=ctx.dla_core_per_leg["eBOSS"] if hasattr(ctx, "dla_core_per_leg") else
              np.zeros(np.asarray(ctx.cache_k).shape[0]),
              cache_k=ctx.cache_k, leg=leg, alpha_centres=ctx.alpha_centres)
    # forward LF (mf=None) vs MF (mf=ctx.mf), same θ.
    P_lf, _ = DL.predict_P_obs_on_leg(ctx.model, th, tau0, alpha, mf=None, **kw)
    P_mf, _ = DL.predict_P_obs_on_leg(ctx.model, th, tau0, alpha, mf=ctx.mf, mf_floor=None, **kw)
    P_lf = np.asarray(P_lf); P_mf = np.asarray(P_mf)
    k = np.asarray(leg.k); z = np.asarray(leg.z_row)
    ratio = P_mf / P_lf - 1.0                            # fractional MF imprint per (z,k)

    print("=== MF imprint on the eBOSS forward (fold6, Planck-ish truth); P_MF/P_LF − 1 ===")
    print(f"  eBOSS k band: [{k.min():.4f}, {k.max():.4f}] s/km")
    # per-z tilt: imprint at the lowest vs highest eBOSS k, and the d ln-ratio / d ln-k slope (≈ Δn_s)
    dns = []
    for zi in np.unique(z):
        m = z == zi
        ks = k[m]; rr = ratio[m]
        order = np.argsort(ks); ks, rr = ks[order], rr[order]
        lo, hi = rr[0], rr[-1]
        # Δn_s ≈ slope of ln(1+ratio) vs ln(k) (a multiplicative tilt aliases into the spectral tilt)
        slope = np.polyfit(np.log(ks), np.log1p(rr), 1)[0]
        dns.append(slope)
        if zi in (z.min(), np.median(np.unique(z)), z.max()) or abs(zi - 3.0) < 0.11:
            print(f"  z={zi:.1f}: imprint @k_lo {100*lo:+.2f}%  @k_hi {100*hi:+.2f}%  tilt(Δn_s)≈{slope:+.4f}")
    print(f"\n  MEAN over z: imprint @k_lo {100*np.mean([ratio[z==zi][np.argmin(k[z==zi])] for zi in np.unique(z)]):+.2f}%, "
          f"Δn_s(tilt) ≈ {np.mean(dns):+.4f}  (rms {np.std(dns):.4f})")
    print("  Read: a non-zero Δn_s means the MF tilt reaches eBOSS k and WOULD shift n_s on real data")
    print("        (LF-only would be biased by that amount; the gate-invariant mf=True closure cancels it).")


if __name__ == "__main__":
    main()
