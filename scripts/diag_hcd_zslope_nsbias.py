"""Does the HCD z-slope mis-specification bias n_s? — a NUTS-free (n_s, slope) loglik profile.

The forward evolves the HCD incidence as α_c(z)=α_pivot·((1+z)/(1+z_p))^(B+δs_c) with the z-slope
B centered at the lit/sim-RATIO slope HCD_LIT_OVER_SIM_SLOPE[0] (0.95 when this diagnostic was
run, 2026-06-14; 0.764 since the 2026-07-18 corrected-law swap). But the mock truth's native
w_c(z) evolves at ~2.4 (the FULL incidence slope). This profiles the data-loglik at the NOISELESS
mock truth over (n_s, B) on the JOINT DESI+KS leg to answer:
  (1) what slope B do the data prefer? (argmax_B at n_s=truth)
  (2) does a WRONG fixed slope bias n_s? (argmax_{n_s} at B=0.95 vs B=2.4=truth vs the marginalized B)
  (3) the (n_s, B) degeneracy direction (does moving B drag n_s?).
Noiseless → the loglik peak IS the truth (no noise confound); any n_s shift with B is pure
slope→n_s leakage. Uses the 2D ctx's δs_c; bypasses NUTS (a 2D grid of the closure loglik).

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_hcd_zslope_nsbias.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/mfho/hcd_priya")
import hcd_analysis.emulator  # noqa: F401  x64
import jax, jax.numpy as jnp
from hcd_analysis.emulator import closure_legb as C

NS_LO, NS_HI = 0.8, 1.05    # the n_s unit→physical box


def main():
    ctx, d = C.build_legb_ctx(ckpt=f"{C.REPO}/checkpoints/final_fold6",
                              hierarchical_hcd=True, hcd_2d_tilt=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name in ("DESI", "KS")])
    sim = C.held_out_sims(d, fold=6)[0][0]
    truth = C.make_truth_from_sim(d, sim, fold=6, tau0_anchor="priya")
    mock_legs, tp, info = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    # NOISELESS mock: replace each leg's P_data with the noiseless truth-on-leg
    nl_legs = []
    for leg in mock_legs:
        t = np.asarray(info["truth_on_leg"][leg.name])
        nl_legs.append(leg._replace(P_data=t))

    th0 = np.asarray(tp["theta9"]); tau0g = jnp.asarray(tp["tau0_global"])
    apiv = jnp.asarray(tp["alpha_hcd"])                      # (3,) pivot α truth
    zg = jnp.asarray(ctx.z_global); ds = jnp.asarray(ctx.hcd_dslope)
    ns_truth_phys = NS_LO + float(th0[0]) * (NS_HI - NS_LO)
    print(f"sim={sim}\n n_s truth (phys)={ns_truth_phys:.4f}  δs_c={np.asarray(ds)}  "
          f"B prior center={float(ctx.hcd_btilt_mu):.2f}±{float(ctx.hcd_btilt_sigma):.2f}")

    def loglik(ns_unit, B):
        th = jnp.asarray(th0).at[0].set(ns_unit)
        shape = ((1.0 + zg)[:, None] / (1.0 + C.HCD_Z_PIVOT)) ** (B + ds)   # (nzg,3)
        ah = apiv[None, :] * shape
        return float(C._data_loglik_legcore(ctx, th, tau0g, ah, nl_legs, core))

    Bgrid = np.array([0.95, 1.5, 2.0, 2.37, 2.9])
    ns_grid = np.linspace(NS_LO, NS_HI, 51)
    ns_unit = (ns_grid - NS_LO) / (NS_HI - NS_LO)
    print("\n=== (2)+(3) n_s that MAXIMIZES the (noiseless) loglik, per FIXED forward slope B ===")
    print(f"   {'B (slope)':>10s} {'argmax n_s':>11s} {'Δn_s vs truth':>14s}   lnL@argmax")
    for B in Bgrid:
        lls = np.array([loglik(u, B) for u in ns_unit])
        j = int(np.argmax(lls)); ns_hat = ns_grid[j]
        flag = "  <- truth slope" if abs(B - 2.37) < 0.1 else ("  <- prior center" if abs(B - 0.95) < 0.06 else "")
        print(f"   {B:>10.2f} {ns_hat:>11.4f} {ns_hat - ns_truth_phys:>+14.4f}   {lls[j]:.2f}{flag}")

    print("\n=== (1) data-preferred slope B at n_s=truth (argmax_B) ===")
    Bfine = np.linspace(0.5, 3.5, 31)
    llb = np.array([loglik(float(th0[0]), B) for B in Bfine])
    jb = int(np.argmax(llb))
    print(f"   argmax_B lnL = {Bfine[jb]:.2f}  (lnL={llb[jb]:.2f}); "
          f"ΔlnL(B=0.95 → argmax) = {llb[jb] - loglik(float(th0[0]), 0.95):.2f}")
    # marginalized slope estimate: prior N(0.95,σ) × the data curve (Laplace)
    pri = -0.5 * ((Bfine - float(ctx.hcd_btilt_mu)) / float(ctx.hcd_btilt_sigma)) ** 2
    post = llb + pri; jpm = int(np.argmax(post))
    print(f"   marginalized (prior N(0.95,{float(ctx.hcd_btilt_sigma):.2f}) ⊕ data) MAP slope ≈ {Bfine[jpm]:.2f}")
    # the n_s argmax at the marginalized slope
    Bm = Bfine[jpm]; lls = np.array([loglik(u, Bm) for u in ns_unit]); jm = int(np.argmax(lls))
    print(f"   => n_s argmax at the MARGINALIZED slope {Bm:.2f}: {ns_grid[jm]:.4f} "
          f"(Δ vs truth {ns_grid[jm]-ns_truth_phys:+.4f})")
    print("\n=== VERDICT ===")
    dns_wrong = None
    lls095 = np.array([loglik(u, 0.95) for u in ns_unit]); dns_wrong = ns_grid[int(np.argmax(lls095))] - ns_truth_phys
    print(f"   n_s bias at the WRONG fixed slope 0.95: {dns_wrong:+.4f}  "
          f"({'SIGNIFICANT' if abs(dns_wrong) > 0.005 else 'small'})")
    print(f"   the data prefer slope ~{Bfine[jb]:.1f} (truth ~2.4); the marginalized MAP ~{Bm:.1f} "
          f"=> residual n_s bias {ns_grid[jm]-ns_truth_phys:+.4f}")


if __name__ == "__main__":
    main()
