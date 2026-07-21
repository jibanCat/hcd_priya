"""DIAGNOSTIC (a): whiten the per-k COHERENT held-out emulator residual r(k,z) by the DEPLOYED
sqrt(diag C_total) the production SBC actually uses, over the FULL DESI/eBOSS k-grid INCLUDING
k<0.0102, and project onto the autodiff n_s response u = dlnP/dn_s.

Goal: confirm the whitened n_s pull BLOWS UP below k=0.0102 (the C_emu emucoh hole, which covers
only k∈[0.01016, 0.06892]) and is small above; report the proj magnitude vs the per-mock posterior
n_s sigma, and the k where it blows up.

Two projection forms (both reported):
  (i) SIMPLE per-row (the task's spec):  pull_row = u_row * (r_row / sigma_row), and
      proj = sum_row u_row*(r_row/sigma_row); split low-k (<0.0102) vs high-k (>=0.0102).
  (ii) GLS / Fisher (apples-to-apples vs posterior sigma_ns):  for the DESI leg with its FULL
      C_total (incl. the off-diagonal coherent terms), the linear n_s shift a coherent residual r
      induces is  dns = (u^T C^-1 r)/(u^T C^-1 u),  pull_sigma = (u^T C^-1 r)/sqrt(u^T C^-1 u).
      Done for the WHOLE leg, the LOW-k-only rows, and the HIGH-k-only rows (block C^-1).

r(k,z) is the COHERENT FRACTIONAL held-out construction residual (truth-fwd)/fwd from
figures/analysis/04_emulator/embias_arch_closure_resid.npz (fmean / fmean_ks), which is
C-INDEPENDENT (so we may re-whiten with the DEPLOYED C). sigma_row = sqrt(diag C_total)/P_fid is
the fractional deployed per-k sigma, so r_row/sigma_row is the per-k whitened pull in sigma units.

=== ATTIC (2026-07-21 freeze triage, PI-approved). NOT IN THE FROZEN FORWARD. ===
Nothing imports this; its only artifacts are the four diag_cemu_lowk_whiten_*.npz now tracked
in the NOTES repo, read by nothing in the code repo. The low-k C_emu fix it evaluated was NOT
deployed.
KEPT rather than deleted because it is the INSTRUMENT OF RECORD behind a live NEGATIVE
result: the low-k C_emu fix was refuted on its GLS number and a 320-640 CPU-h SBC was
deliberately not spent. The GLS half is NOT recoverable from the tracked npz (they carry
k, r_frac, sigma_frac, u, wres, P_model but NOT C_total), so deleting this would leave a
standing decision with no reproducible instrument.

NOT THE PAPER'S F6 RECIPE (triage-doc label corrected 2026-07-21). F6 wants the CROSS-CLASS
whitening-variance 2.46 -> ~1 / diag_match_median 0.777 decomposition, which is already
COMMITTED in scripts/build_xclass_error_vector.py (docstring line 5; diag_match_median at
:290) plus scripts/diag_cemu_validation.py (run_whitening / revalidate_xclass /
fig_whitening_before_after), run as
  diag_cemu_validation.py --xclass checkpoints/error_vector_xclass.npz --n-folds 8
Both inputs are present. This file computes neither 2.46 nor 0.777 nor any cross-class rho;
it shares only the word "whiten". F6 is blocked on a ~10-30 min RUN, not a missing recipe.

STALE vs the DEPLOYED forward: build_ctx below never passes res_corr_on, so it inherits the
build_legb_ctx default res_corr_on=True, i.e. the PRE-NORC forward. The claim immediately
below about "the EXACT production-SBC flags" is therefore no longer true. Re-running this
does NOT reproduce the deployed forward unless res_corr_on=False / fix_alpha_res=True are
passed (same hazard logged for its siblings in notes norc-refactor-artifacts/caveats.md:49).

CONSTANTS AUDIT (2026-07-21): no novel science choice. HOLE_K=0.0102 is documented in-file
and verified against the emucoh floor's k span [0.01016, 0.06892] (reporting split only).
The z-match tolerance 0.15 is the committed library convention (closure_legb.py:1631,
"nearest-z map tolerance, cache dz=0.2"). tau0_anchor="becker13" overrides the library
default "priya" and is the one real science choice: justified in the library itself at
closure_legb.py:1313 as closest to the production observational anchor.

The DEPLOYED C_total is built with the EXACT production-SBC flags from run_prod_sbc_shard.py:
  build_legb_ctx(ensemble_ckpts=ALL, use_xclass=True, with_mf=True, mf_with_floor=True,
                 mf_emucoh=True, mf_emucoh_offdiag_only=True, with_eboss=True, metals_on=True,
                 sample_metals=True, hierarchical_hcd=False)
plus an OPTIONAL --fixed arm that ALSO turns on mf_shape=True and points emucoh at a LOW-k npz
(the post-fix C) so the SAME projection can be re-run after the fix.

Env: PYTHONPATH=/home/mfho/hcd_priya, import hcd_analysis.emulator FIRST (x64).
"""
from __future__ import annotations
import argparse, os
import numpy as np

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax, jax.numpy as jnp

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.closure_legb_figs import _truth_alpha_zresolved_on_leg
from hcd_analysis.emulator import data_likelihood as DL

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"
HOLE_K = 0.0102        # the emucoh floor's low-k edge (covers [0.01016, 0.06892])


def build_ctx(fixed: bool, emucoh_npz: str | None):
    import glob
    members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
    kw = dict(
        ensemble_ckpts=members, use_xclass=True,
        with_mf=True, mf_with_floor=True,
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=True, metals_on=True, sample_metals=True,
        hierarchical_hcd=False)
    if fixed:
        kw["mf_shape"] = True                          # MODE 2 ON
    if emucoh_npz:
        kw["mf_emucoh_npz"] = emucoh_npz               # MODE 1: the low-k-extended floor
    ctx, d = CL.build_legb_ctx(**kw)
    return ctx, d, members


def deployed_P_and_C(ctx, leg, truth):
    """Return (P_model, C_total, u_frac) on this leg at the sim TRUTH theta/tau0/alpha, where
    u_frac = dlnP/dn_s (the autodiff fractional n_s response per row).  C_total is the FULL
    deployed covariance (incl. the coherent off-diagonal shape/emucoh terms)."""
    cache_k = np.asarray(ctx.cache_k)
    z_sim = np.asarray(truth["z"]); P_sim = np.asarray(truth["P_obs_true"])
    tau0_sim = np.asarray(truth["tau0"])
    theta9 = jnp.asarray(truth["params_unit"])
    alpha_hcd = _truth_alpha_zresolved_on_leg(truth, leg)   # z-RESOLVED (was z-flat truth["w_c"] -> the +5.5sigma artifact)
    core = CL._mock_core_per_leg(ctx, truth)[leg.name]
    N = leg.k.shape[0]
    tau0_vec = np.zeros(leg.n_z)
    for iz in range(leg.n_z):
        zz = float(leg.z[iz]); j = int(np.argmin(np.abs(z_sim - zz)))
        if abs(z_sim[j] - zz) <= 0.15:
            tau0_vec[iz] = tau0_sim[j]
        else:
            tau0_vec[iz] = float(np.interp(zz, z_sim, tau0_sim))
    szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
    msc = (ctx.mf_shape_per_leg.get(leg.name)
           if getattr(ctx, "mf_shape_per_leg", None) is not None else None)
    mec = (ctx.mf_emucoh_per_leg.get(leg.name)
           if getattr(ctx, "mf_emucoh_per_leg", None) is not None else None)

    def predict(th):
        Pm, Ct = DL.predict_P_obs_on_leg(
            ctx.model, th, jnp.asarray(tau0_vec), alpha_hcd, pf_stats=ctx.pf_stats,
            dla_core=core, cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
            alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb,
            mf=ctx.mf, mf_floor=ctx.mf_floor, mf_shape_cov=msc,
            mf_shape_infl=getattr(ctx, "mf_shape_infl", 1.0), mf_emucoh_cov=mec,
            mf_emucoh_infl=getattr(ctx, "mf_emucoh_infl", 1.0),
            mf_emucoh_offdiag_only=getattr(ctx, "mf_emucoh_offdiag_only", False))
        return Pm, Ct

    P_model, C_total = predict(theta9)
    # dlnP/dn_s : n_s is theta9[0]; jacobian of log P_model wrt theta9[0]
    def logP_of_ns(ns0):
        th = theta9.at[0].set(ns0)
        Pm, _ = predict(th)
        return jnp.log(jnp.clip(Pm, 1e-30, None))
    dlogP = jax.jacobian(logP_of_ns)(theta9[0])     # (N,) dlnP/dtheta9[0]
    return (np.asarray(P_model), np.asarray(C_total), np.asarray(dlogP))


def coherent_frac_resid_on_leg(ctx, d, leg, sims):
    """The COHERENT (mean-over-held-out-sims) FRACTIONAL residual r(k) = mean_sim (truth_on_leg -
    fwd)/fwd on THIS leg's exact rows — the honest construction residual, computed live (no stale
    grid). truth_on_leg = the sim's MEASURED P_obs interp'd to the leg k; fwd = the deployed forward
    (predict_P_obs_on_leg, the SBC forward). Averaged over all fold-0 held-out sims."""
    cache_k = np.asarray(ctx.cache_k)
    N = leg.k.shape[0]
    fsum = np.zeros(N); fcnt = np.zeros(N)
    szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
    mec = (ctx.mf_emucoh_per_leg.get(leg.name)
           if getattr(ctx, "mf_emucoh_per_leg", None) is not None else None)
    msc = (ctx.mf_shape_per_leg.get(leg.name)
           if getattr(ctx, "mf_shape_per_leg", None) is not None else None)
    for sname in sims:
        truth = CL.make_truth_from_sim(d, sname, fold=0, tau0_anchor="becker13", mf=ctx.mf)
        z_sim = np.asarray(truth["z"]); P_sim = np.asarray(truth["P_obs_true"])
        tau0_sim = np.asarray(truth["tau0"])
        theta9 = jnp.asarray(truth["params_unit"])
        alpha = _truth_alpha_zresolved_on_leg(truth, leg)   # z-RESOLVED (was z-flat truth["w_c"])
        core = CL._mock_core_per_leg(ctx, truth)[leg.name]
        P_truth = np.full(N, np.nan); keep = np.zeros(N, bool); tau0_vec = np.zeros(leg.n_z)
        for iz in range(leg.n_z):
            zz = float(leg.z[iz]); j = int(np.argmin(np.abs(z_sim - zz)))
            if abs(z_sim[j] - zz) > 0.15:
                tau0_vec[iz] = float(np.interp(zz, z_sim, tau0_sim)); continue
            rows = np.where(np.asarray(leg.z_idx) == iz)[0]
            ksub = np.asarray(leg.k)[rows]
            P_truth[rows] = np.asarray(jnp.interp(jnp.asarray(ksub), jnp.asarray(cache_k),
                                                  jnp.asarray(P_sim[j])))
            keep[rows] = True; tau0_vec[iz] = float(tau0_sim[j])
        P_model, _ = DL.predict_P_obs_on_leg(
            ctx.model, theta9, jnp.asarray(tau0_vec), alpha, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor,
            mf_emucoh_cov=mec, mf_emucoh_infl=getattr(ctx, "mf_emucoh_infl", 1.0),
            mf_emucoh_offdiag_only=getattr(ctx, "mf_emucoh_offdiag_only", False),
            mf_shape_cov=msc, mf_shape_infl=getattr(ctx, "mf_shape_infl", 1.0))
        P_model = np.asarray(P_model)
        for ridx in np.where(keep)[0]:
            if P_model[ridx] != 0:
                fsum[ridx] += (P_truth[ridx] - P_model[ridx]) / P_model[ridx]; fcnt[ridx] += 1
    r = np.where(fcnt > 0, fsum / np.maximum(fcnt, 1), np.nan)
    return r, fcnt > 0


def project_leg(name, k, P_model, C_total, dlogP, r_frac, keep, post_sigma_ns_unit):
    """Whiten + project. r_frac is the FRACTIONAL coherent residual on the SAME rows.
    keep masks rows with a valid residual + positive diag.  Reports simple + GLS forms."""
    diagC = np.diag(C_total)
    sigma_frac = np.where(P_model > 0, np.sqrt(np.clip(diagC, 0, None)) / np.maximum(P_model, 1e-30), np.nan)
    u = dlogP                                        # dlnP/dn_s (theta-unit); fractional response
    ok = keep & np.isfinite(r_frac) & np.isfinite(sigma_frac) & (sigma_frac > 0)
    lo = ok & (k < HOLE_K); hi = ok & (k >= HOLE_K)

    # (i) SIMPLE per-row whitened pull  u * (r/sigma)
    pull = np.where(ok, u * (r_frac / np.where(sigma_frac > 0, sigma_frac, 1.0)), 0.0)
    proj_all = pull[ok].sum(); proj_lo = pull[lo].sum(); proj_hi = pull[hi].sum()
    # the per-row whitened residual magnitude (r/sigma), to see WHERE it blows up
    wres = np.where(ok, r_frac / np.where(sigma_frac > 0, sigma_frac, 1.0), np.nan)

    print(f"\n===== {name} =====")
    print(f"  rows used: {int(ok.sum())}  (low-k<{HOLE_K}: {int(lo.sum())}, high-k: {int(hi.sum())})")
    print(f"  [SIMPLE u*(r/sigma)] proj_all = {proj_all:+.3f}   proj_LOW-k = {proj_lo:+.3f}   "
          f"proj_HIGH-k = {proj_hi:+.3f}")
    print(f"     => low-k share of |proj|: {100*abs(proj_lo)/max(abs(proj_lo)+abs(proj_hi),1e-30):.0f}%")
    # mean whitened residual magnitude (the 'pull blows up' check), low vs high
    print(f"  mean |r/sigma|  LOW-k = {np.nanmean(np.abs(wres[lo])):.3f} sigma   "
          f"HIGH-k = {np.nanmean(np.abs(wres[hi])):.3f} sigma   "
          f"(ratio low/high = {np.nanmean(np.abs(wres[lo]))/max(np.nanmean(np.abs(wres[hi])),1e-9):.1f}x)")
    # the k where |r/sigma| peaks
    if ok.any():
        ipk = np.nanargmax(np.where(ok, np.abs(wres), -np.inf))
        print(f"  PEAK |r/sigma| = {abs(wres[ipk]):.2f} sigma at k = {k[ipk]:.5f}  "
              f"(r={100*r_frac[ipk]:+.2f}%, sigma_dep={100*sigma_frac[ipk]:.2f}%)")

    # (ii) GLS/Fisher n_s pull, WHOLE leg + low-k block + high-k block (block C^-1)
    def gls(mask):
        idx = np.where(mask)[0]
        if idx.size < 2:
            return np.nan, np.nan, np.nan
        Csub = C_total[np.ix_(idx, idx)]
        Csub = Csub + 1e-12 * np.mean(np.diag(Csub)) * np.eye(len(idx))
        # r in raw units = r_frac * P_model ; u in raw units = dlnP/dns * P_model
        r_raw = (r_frac[idx] * P_model[idx])
        u_raw = (u[idx] * P_model[idx])
        Cinv_r = np.linalg.solve(Csub, r_raw)
        Cinv_u = np.linalg.solve(Csub, u_raw)
        F = float(u_raw @ Cinv_u)                     # Fisher info on theta-unit ns
        num = float(u_raw @ Cinv_r)                   # u^T C^-1 r
        dns_unit = num / F if F > 0 else np.nan       # induced n_s shift (theta-unit)
        pull_sig = num / np.sqrt(F) if F > 0 else np.nan   # in units of this block's own sigma_ns
        return dns_unit, pull_sig, F
    dns_all, pull_all, F_all = gls(ok)
    dns_lo, pull_lo, F_lo = gls(lo)
    dns_hi, pull_hi, F_hi = gls(hi)
    sig_ns_leg = 1.0 / np.sqrt(F_all) if (F_all and F_all > 0) else np.nan   # this leg's own ns sigma (unit)
    print(f"  [GLS  Fisher]  whole-leg  dns_unit = {dns_all:+.4f}  ( = {dns_all/post_sigma_ns_unit:+.2f} "
          f"x post-sigma_ns; leg-only sigma_ns_unit={sig_ns_leg:.4f})")
    print(f"                 LOW-k only dns_unit = {dns_lo:+.4f}   HIGH-k only dns_unit = {dns_hi:+.4f}")
    print(f"                 (post-mock sigma_ns_unit ~ {post_sigma_ns_unit:.4f}; "
          f"dns/post = low {dns_lo/post_sigma_ns_unit:+.2f} / high {dns_hi/post_sigma_ns_unit:+.2f})")
    return dict(name=name, k=k, r_frac=r_frac, sigma_frac=sigma_frac, u=u, wres=wres, ok=ok,
                lo=lo, hi=hi, proj_all=proj_all, proj_lo=proj_lo, proj_hi=proj_hi,
                dns_all=dns_all, dns_lo=dns_lo, dns_hi=dns_hi, P_model=P_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true", help="turn ON mf_shape (MODE 2) for the post-fix arm")
    ap.add_argument("--emucoh-npz", default=None, help="point the emucoh floor at a low-k npz (post-fix MODE 1)")
    ap.add_argument("--mock", type=int, default=0, help="held-out sim index (fold 0) used as the truth")
    ap.add_argument("--post-sigma-ns-unit", type=float, default=0.030,
                    help="representative per-mock posterior n_s sigma (theta-unit); ~0.019-0.040 measured")
    ap.add_argument("--save", default=None, help="npz to save the per-row arrays")
    a = ap.parse_args()

    tag = "FIXED" if (a.fixed or a.emucoh_npz) else "DEPLOYED(current)"
    print(f"[diag] building {tag} production-SBC ctx "
          f"(fixed={a.fixed}, emucoh_npz={a.emucoh_npz})...")
    ctx, d, members = build_ctx(a.fixed, a.emucoh_npz)
    print(f"[diag] ensemble members={len(members)}  legs={[l.name for l in ctx.legs]}  "
          f"mf_shape_per_leg={'set' if getattr(ctx,'mf_shape_per_leg',None) else 'None'}  "
          f"mf_emucoh_per_leg={'set' if getattr(ctx,'mf_emucoh_per_leg',None) else 'None'}")

    sims, _ = CL.held_out_sims(d, fold=0)
    # The COHERENT residual is computed LIVE on the leg (truth-vs-forward, averaged over ALL
    # held-out sims at the n_s response of one representative truth) — NOT remapped from the stale
    # RESID_NPZ grid (whose k-grid differs from the live leg, which corrupts the low-k rows).
    rep_sim = sims[a.mock % len(sims)]
    truth_rep = CL.make_truth_from_sim(d, rep_sim, fold=0, tau0_anchor="becker13", mf=ctx.mf)
    print(f"[diag] n response sim = {rep_sim}  ;  residual averaged over {len(sims)} held-out sims")

    out = {}
    for leg in ctx.legs:
        nm = leg.name
        # the deployed P/C/u at the representative truth (u = dlnP/dn_s; C = the deployed total)
        P_model, C_total, dlogP = deployed_P_and_C(ctx, leg, truth_rep)
        # LIVE coherent fractional residual r(k) = mean over held-out sims of (truth_on_leg - fwd)/fwd
        r_leg, keep = coherent_frac_resid_on_leg(ctx, d, leg, sims)
        if nm.lower().startswith("eboss"):
            # eBOSS shares DESI's low-k structure; report its residual+coverage but it is a
            # secondary low-k leg (DESI binds). Still projected (it has k<0.0102 rows).
            pass
        res = project_leg(nm, np.asarray(leg.k), P_model, C_total, dlogP, r_leg, keep,
                          a.post_sigma_ns_unit)
        out[nm] = res

    if a.save:
        np.savez(a.save, **{f"{nm}_{key}": out[nm][key]
                            for nm in out for key in ("k", "r_frac", "sigma_frac", "u", "wres", "ok", "lo", "hi", "P_model")})
        print(f"\n[diag] saved per-row arrays -> {a.save}")


if __name__ == "__main__":
    main()
