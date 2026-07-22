"""DESI HCD-prior sensitivity: fit a DESI closure mock WITH vs WITHOUT the HCD incidence
prior and measure the leverage on (A_p, n_s) + the recovered per-class dN/dX(z).

THE QUESTION (PI, 2026-06-21). On the DESI leg the subDLA<->n_s / LLS<->A_p degeneracy is LIVE
(low-k damping-wing power mimics a forest amplitude/tilt change). The HCD incidence prior holds
the per-class amplitudes near the observed dN/dX; removing it lets the HCD float along that
degeneracy and shift A_p/n_s. This runner measures that leverage by running NUTS on the SAME
DESI closure mock under two priors:
  WITH prior    : HCD_PRIOR_FRAC_SIGMA = (0.15, 0.40, 0.50)   (the deployed cosmic-average widths)
  WITHOUT prior : HCD_PRIOR_FRAC_SIGMA = (5.0, 5.0, 5.0)      (sigma ~= 5*mu, ~flat within low=0)

MATCHED A/B (cleanest design). The mock is a HELD-OUT DESI SIM (leg_a=False): the truth is the
sim's MEASURED contaminated power through the production MF correction -- INDEPENDENT of the HCD
prior width. So the with/without arms over the SAME (seed, mock) draw BYTE-IDENTICAL data and ONLY
the likelihood's HCD prior differs (a guaranteed-matched A/B). (A self-draw leg_a=True mock would
draw the truth alpha FROM the prior, so a wider prior => different data => not matched.)

PRIOR OVERRIDE = a RUNTIME monkeypatch of inference.HCD_PRIOR_FRAC_SIGMA in THIS process, BEFORE
build_legb_ctx -- exactly the idiom run_prod_sbc_shard.py uses for SBC_SUBDLA_AMP_SIGMA. The
closure path builds the ctx with survey=None, so all three class widths flow through
hcd_incidence_prior, which reads inference.HCD_PRIOR_FRAC_SIGMA at call time => the patch takes
effect. We ASSERT it propagated onto ctx.alpha_hcd_sigma (no silent no-op).

SBC SAFETY: this is a NEW standalone runner. It edits NOTHING. It imports build_legb_ctx / run_legb
(read-only) and patches the module attribute in its OWN process only. The production SBC arms run in
separate processes with their own module state and are unaffected.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle

print = functools.partial(print, flush=True)

import numpy as np

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import hcd_analysis.emulator.inference as _I
from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, run_legb, load_cache, CACHE_PATH, hcd_pivot_wc_and_xbar,
    HCD_INCIDENCE_SLOPE, HCD_Z_PIVOT)
from hcd_analysis.emulator.inference import PARAM_NAMES
from hcd_analysis.emulator.prod_ensemble import production_member_paths
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
import jax.numpy as jnp

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"

# The two prior arms. WITH = the deployed cosmic-average widths; WITHOUT = ~flat (sigma ~= 5*mu),
# still truncated at low=0 (a physical incidence is non-negative). Override exactly here; the rest
# of the prior (centers mu_c = observed dN/dX, z-slope s_c) is UNCHANGED so the A/B isolates the
# AMPLITUDE width as the handoff mandates.
PRIOR_WIDTHS = {
    "on":  (0.15, 0.40, 0.50),
    "off": (5.0,  5.0,  5.0),
}


def _build_xbar_fn(d):
    """Xbar(z) deg-2 fit (mean path length per sightline), the dN/dX readout's Xbar(z).
    MIRRORS scripts/plot_hcd_prior_dndx_overlay_v3_production.build_xbar_fn_and_sim exactly."""
    gid = np.asarray(d["snap_group_idx"]); zrow = np.asarray(d["z_grid"])
    Xtot = np.asarray(d["snap_total_path_dX"]); wc = np.asarray(d["w_c_cache"])
    dndx = np.asarray(d["snap_dNdX"]); Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    ok = np.isfinite(Xbar) & (zg >= 2.1) & (zg <= 4.7)
    cf = np.polyfit(zg[ok], Xbar[ok], 2)
    return lambda z: np.polyval(cf, np.asarray(z))


def _dndx_from_pivot_draws(alpha_pivot_draws, z_global, xbar_fn):
    """Map per-draw PIVOT alpha (L,3) -> per-class dN/dX(z) (L, nZ, 3) on z_global.

    CLOSURE path (survey=None): the forward z-slope is the SIM incidence slope HCD_INCIDENCE_SLOPE
    (~2.4 -- the slope the held-out-sim mock truth carries; the _zslope_sites FIXED branch / the
    _legb_reconstruct_deterministics fixed-slope fallback). alpha_c(z) = alpha_pivot * ((1+z)/(1+z_p))^s_c,
    then dN/dX_c(z) = alpha_to_dndx(alpha_c(z), Xbar(z), z). Same construction as
    plot_hcd_prior_dndx_overlay_v3_production.to_dndx_curve."""
    a = np.asarray(alpha_pivot_draws)                       # (L,3)
    zg = np.asarray(z_global, float)                        # (nZ,)
    s_c = np.asarray(HCD_INCIDENCE_SLOPE, float)            # (3,) closure/SBC slope
    Xb = np.asarray(xbar_fn(zg), float)                     # (nZ,)
    shape = ((1.0 + zg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** s_c[None, :]   # (nZ,3)
    out = np.empty((a.shape[0], len(zg), 3))
    for j in range(len(zg)):
        az = a * shape[j][None, :]                          # (L,3) alpha_c at z_j
        out[:, j, :] = np.asarray(alpha_to_dndx(
            jnp.asarray(az), jnp.asarray(float(Xb[j])), jnp.asarray(float(zg[j]))))
    return out, zg, Xb


def _col(names, draws, key):
    """Column of the (L,P) draws matrix by site name (returns None if absent)."""
    names = list(names)
    return np.asarray(draws)[:, names.index(key)] if key in names else None


def _alpha_pivot_block(rec):
    """Extract the (L,3) PIVOT alpha [lls,subdla,dla] + (3,) truth from a run_legb per-mock record,
    by NAME (robust to the appended a_SiIII / hierarchical columns -- the positive-index fix the
    SBC aggregator uses)."""
    names = list(rec["names"]); draws = np.asarray(rec["draws"]); tv = np.asarray(rec["truth_vec"])
    cols = [names.index(k) for k in ("alpha_lls", "alpha_subdla", "alpha_dla")]
    return draws[:, cols], tv[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["on", "off"], required=True,
                    help="HCD incidence prior: 'on' = deployed (0.15,0.40,0.50); 'off' = ~flat (5,5,5)")
    ap.add_argument("--mock", type=int, required=True,
                    help="held-out-sim mock index m -> held_out_sims(fold)[m %% n_sims]")
    ap.add_argument("--fold", type=int, default=0,
                    help="LOSO fold for held_out_sims (n_s-sorted: 0=low edge, 4=mid, 7=high). "
                         "fold0 (default) = all low-n_s. Mid-box demo: --fold 4.")
    ap.add_argument("--n-mocks", type=int, default=8,
                    help="for fold_in(seed,m) reproducibility (so a subset matches a full run)")
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--single-member", action="store_true",
                    help="final_prod_seed0 only (cheap de-risk; NOT the production object)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # SKIP-IF-EXISTS (resume safety, mirrors batch_sbc_heldout): the per-mock pkl is the unit.
    path = os.path.join(a.out_dir, f"mock_{a.mock:04d}.pkl")
    if os.path.exists(path):
        print(f"=== mock {a.mock} pkl exists at {path} -- SKIP ===")
        return

    # ---- RUNTIME PRIOR OVERRIDE (the SBC_SUBDLA_AMP_SIGMA idiom) -----------------------------
    widths = PRIOR_WIDTHS[a.prior]
    _pf_before = _I.HCD_PRIOR_FRAC_SIGMA
    _I.HCD_PRIOR_FRAC_SIGMA = tuple(float(w) for w in widths)
    print(f"[prior-{a.prior}] HCD_PRIOR_FRAC_SIGMA {_pf_before} -> {_I.HCD_PRIOR_FRAC_SIGMA}")

    # PINNED members (freeze decision 6): manifest-verified (sha256 + exact pairing + count +
    # stray-member tripwire) via checkpoints/production_ensemble_manifest.json, NOT a glob.
    members = production_member_paths(checkpoints_dir=os.path.dirname(PROD_PREFIX))
    ens = [members[0]] if a.single_member else members
    print(f"[ensemble] {len(ens)} member(s): {[os.path.basename(m) for m in ens]}")

    # DESI leg only, with the DR1 metal model (metals_on + sample_metals) -- the DEPLOYED DESI fit,
    # built EXACTLY as run_prod_sbc_shard.py's --leg DESI path (production MF + emucoh, current C_emu,
    # hierarchical_hcd=False). survey=None (closure) so the per-class widths come from the patched
    # HCD_PRIOR_FRAC_SIGMA via hcd_incidence_prior.
    ctx, d = build_legb_ctx(
        use_xclass=True,
        with_mf=True, mf_with_floor=True,
        mf_emucoh=True, mf_emucoh_offdiag_only=True, mf_emucoh_npz=None,
        mf_shape=False,
        with_eboss=False, metals_on=True, sample_metals=True,
        hierarchical_hcd=False, ensemble_ckpts=ens)
    # restrict to the single DESI leg (slice ctx.legs, the run_prod_sbc_shard --leg pattern).
    _pre = [l.name for l in ctx.legs]
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name.upper().startswith("DESI")])
    assert len(ctx.legs) == 1, f"--leg DESI: expected 1 leg, got {[l.name for l in ctx.legs]} (pre={_pre})"
    print(f"[per-leg] DESI: legs={[l.name for l in ctx.legs]} metals_on=True sample_metals=True")

    # ---- PROPAGATION ASSERT (the referee's #1 hazard: a silent no-op) -------------------------
    # The closure widths flow alpha_sd = HCD_PRIOR_FRAC_SIGMA * alpha_mu, so ctx ratio = the patched
    # width per class. Verify ALL THREE classes match the requested widths (no silent no-op).
    ratio = np.asarray(ctx.alpha_hcd_sigma) / np.asarray(ctx.alpha_hcd_mu)
    want = np.asarray(widths, float)
    # DLA carries a z-dependent inflate above z=3.5 in the prior, but at the z=3 pivot mu/sigma the
    # base ratio is the requested width; check LLS+subDLA exactly, DLA within the >=1 inflate.
    assert abs(ratio[0] - want[0]) < 1e-3, f"LLS width NO-OP: ctx {ratio[0]:.4f} != {want[0]}"
    assert abs(ratio[1] - want[1]) < 1e-3, f"subDLA width NO-OP: ctx {ratio[1]:.4f} != {want[1]}"
    assert ratio[2] >= want[2] - 1e-3, f"DLA width NO-OP: ctx {ratio[2]:.4f} < {want[2]}"
    print(f"[prior-{a.prior}] VERIFIED ctx.alpha_hcd_sigma/mu = {np.round(ratio,4).tolist()} "
          f"(requested {want.tolist()})")

    # ---- the matched held-out-sim mock + NUTS (deployed settings) -----------------------------
    # leg_a=False => held-out-sim mock (truth = sim measured power thru the production MF), PRIOR-
    # INDEPENDENT => the with/without arms over the same (seed,m) share identical data. fold=0.
    records = run_legb(ctx, d, n_mocks=a.n_mocks, mock_indices=[int(a.mock)], return_per_mock=True,
                       leg_a=False, fold=int(a.fold), n_warmup=a.n_warmup, n_samples=a.n_samples,
                       seed=a.seed, dense_mass=True, max_tree_depth=a.max_tree_depth, verbose=True)
    assert len(records) == 1, f"expected 1 record for mock {a.mock}, got {len(records)}"
    rec = records[0]

    # ---- post-process: cosmology draws + recovered per-class dN/dX(z) -------------------------
    names = list(rec["names"]); draws = np.asarray(rec["draws"]); tv = np.asarray(rec["truth_vec"])
    n_s_draws = _col(names, draws, "ns"); A_p_draws = _col(names, draws, "Ap")
    alpha_pivot_draws, alpha_pivot_truth = _alpha_pivot_block(rec)   # (L,3),(3,)
    n_s_truth = float(tv[names.index("ns")]); A_p_truth = float(tv[names.index("Ap")])

    xbar_fn = _build_xbar_fn(d)
    dndx_draws, z_dndx, Xbar_z = _dndx_from_pivot_draws(alpha_pivot_draws, ctx.z_global, xbar_fn)
    dndx_truth, _, _ = _dndx_from_pivot_draws(alpha_pivot_truth[None, :], ctx.z_global, xbar_fn)
    dndx_truth = dndx_truth[0]                                       # (nZ,3)

    out = dict(
        prior=a.prior, widths=tuple(float(w) for w in widths),
        mock=int(a.mock), sim=rec.get("sim"), seed=int(a.seed),
        names=names, draws=draws, truth_vec=tv,
        # convenience cosmology + alpha-pivot blocks (also fully recoverable from draws/names)
        ns_draws=n_s_draws, Ap_draws=A_p_draws,
        alpha_pivot_draws=alpha_pivot_draws, alpha_pivot_truth=alpha_pivot_truth,
        ns_truth=n_s_truth, Ap_truth=A_p_truth,
        # recovered per-class dN/dX(z): (L, nZ, 3) draws + (nZ,3) truth, class order [LLS,subDLA,DLA]
        dndx_z=z_dndx, dndx_Xbar=Xbar_z, dndx_draws=dndx_draws, dndx_truth=dndx_truth,
        dndx_class_order=("LLS", "subDLA", "DLA"),
        # prior centers (observed dN/dX pin) for the dN/dX-vs-prior-center panel
        alpha_hcd_mu=np.asarray(ctx.alpha_hcd_mu), alpha_hcd_sigma=np.asarray(ctx.alpha_hcd_sigma),
        n_div=int(rec.get("n_div", 0)), L=int(rec.get("L", draws.shape[0])),
        config=dict(leg="DESI", metals_on=True, leg_a=False, fold=int(a.fold),
                    ensemble=[os.path.basename(m) for m in ens],
                    n_warmup=a.n_warmup, n_samples=a.n_samples,
                    max_tree_depth=a.max_tree_depth, dense_mass=True, target_accept=0.9,
                    hcd_prior_frac_sigma=tuple(float(w) for w in widths)),
    )
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    print(f"[prior-{a.prior}] mock {a.mock} sim={str(rec.get('sim'))[:24]} "
          f"n_div={out['n_div']} L={out['L']} -> {path}")
    print(f"  n_s truth={n_s_truth:.4f} post-mean={float(np.mean(n_s_draws)):.4f} "
          f"A_p truth={A_p_truth:.3e} post-mean={float(np.mean(A_p_draws)):.3e}")


if __name__ == "__main__":
    main()
