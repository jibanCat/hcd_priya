"""DESI HCD-bias 3-rung MCMC: fit the SAME DESI closure mock under three HCD configurations
that differ ONLY by which per-class α the fit floats, and measure the leverage on (A_p, n_s) +
the recovered per-class dN/dX(z) / τ₀. A near-clone of scripts/desi_hcd_prior_sensitivity.py
(the validated prior-sensitivity machinery) — same DESI-only ctx slice, same deployed ensemble,
same DR1 metals, same matched held-out-sim mock, same per-mock record schema.

THE DEPLOYED FORWARD (data_likelihood.py:502): P_model = P_clean + Σ_c α_c·excess_c. The three
rungs differ ONLY in the HCD prior the fit uses, set by a RUNTIME override (host-side ctx field
swap of alpha_hcd_mu/alpha_hcd_sigma — exactly the prior-sensitivity idiom). The legacy HCD sites
(ctx.hierarchical_hcd=False) are:
    alpha_lls    ~ TruncatedNormal(alpha_hcd_mu[0], alpha_hcd_sigma[0], low=0)
    alpha_subdla ~ TruncatedNormal(alpha_hcd_mu[1], alpha_hcd_sigma[1], low=0)
    alpha_dla    = softplus( Normal(_dla_raw_mu(alpha_hcd_mu[2]), 1.0) )
NOTE the DLA latent SCALE is hardcoded 1.0 (NOT alpha_hcd_sigma[2]) — its prior is controlled
ONLY by alpha_hcd_mu[2] (via _dla_raw_mu = softplus^{-1}). So to PIN a class to ≈0:
  - LLS / subDLA : set mu_c ≈ 1e-9 AND sigma_c ≈ 1e-9 (TruncNormal half-normal mean ≈ 1.3e-9).
  - DLA          : set mu_DLA ≈ 1e-6  -> _dla_raw_mu ≈ -13.8 -> softplus(raw + N(0,1)) mean ≈ 1.6e-6
                   (the softplus floor does NOT keep it above ~1e-3; verified numerically).

THE THREE RUNGS (--rung):
  clean : DELTA-PIN ALL THREE α ≈ 0 (σ→tiny) -> pure clean forest (P_model ≈ P_clean).
  tierp : DELTA-PIN α_LLS + α_subDLA at their DEPLOYED PIN CENTERS (ctx.alpha_hcd_mu[0:2] = the
          observed-dN/dX literature pin), σ→tiny so they are NOT marginalized; pin α_DLA ≈ 0.
          This applies a FIXED, un-marginalized HCD correction that is WRONG vs THIS closure mock's
          sim-truth HCD (the closure truth α ≠ the literature pin). Only cosmology+τ₀+metals float.
  marg  : DEPLOYED prior on all three, FLOATED (0.15, 0.40, 0.50 frac widths; the "on" arm).

So clean vs tierp differ ONLY in WHERE LLS+subDLA are delta-pinned (0 vs the deployed pin center);
BOTH are delta-pinned (σ→0, NOT marginalized). marg is the ONLY rung that floats the HCD. The ladder
shows: clean (no correction) = worst; tierp (fixed WRONG correction, no marginalization) =
intermediate-but-still-biased; marg (floated w/ prior) = unbiased -- the value of marginalizing.

MATCHED DATA (the cleanest design). The mock is a HELD-OUT DESI SIM (leg_a=False): the truth is
the sim's MEASURED contaminated power through the production MF correction — INDEPENDENT of the HCD
prior. So all three rungs over the SAME (seed, fold, mock_index) draw BYTE-IDENTICAL data; ONLY the
likelihood's HCD prior (which α are floated) differs. A guaranteed-matched 3-way A/B/C.

SBC SAFETY: this is a NEW standalone runner. It EDITS NOTHING. It imports build_legb_ctx / run_legb
(read-only) and applies the rung override host-side on its OWN ctx (and, for the deployed widths it
reads, the module attribute inference.HCD_PRIOR_FRAC_SIGMA) IN THIS PROCESS ONLY. The live Gate-B
SBC array runs in separate processes with their own module state and is unaffected. Isolated OUTDIR
+ logs.

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
    build_legb_ctx, run_legb, HCD_INCIDENCE_SLOPE, HCD_Z_PIVOT)
from hcd_analysis.emulator.prod_ensemble import production_member_paths
from hcd_analysis.emulator.sampler_numpyro import _dla_raw_mu
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"

# Deployed cosmic-average fractional widths (the "marg"/"on" arm). Read live from the module so a
# future change to the deployed prior flows through to tierp's LLS/subDLA widths automatically.
DEPLOYED_FRAC = tuple(float(w) for w in _I.HCD_PRIOR_FRAC_SIGMA)   # (0.15, 0.40, 0.50)

# Per-class PIN targets (≈0). LLS/subDLA: TruncatedNormal(mu, sigma, low=0) -> half-normal mean
# ≈ 1.3e-9. DLA: mu_DLA -> _dla_raw_mu -> softplus(Normal) mean ≈ 1.6e-6 (the scale is hardcoded
# 1.0, so mu is the ONLY DLA lever). All three land well below the 1e-3 gate.
PIN_MU = 1e-9          # LLS / subDLA center pin
PIN_SIGMA = 1e-9       # LLS / subDLA width pin
PIN_MU_DLA = 1e-6      # DLA center pin (the softplus floor is ~1.6e-6, < 1e-3)


def _apply_rung(ctx, rung):
    """Return (ctx_overridden, info). Host-side swap of alpha_hcd_mu / alpha_hcd_sigma per rung.
    DEPLOYED centers/widths are taken from the ctx that build_legb_ctx already populated (closure
    path, survey=None: mu = observed dN/dX literature pin, sigma = DEPLOYED_FRAC * mu).

    DELTA-PIN convention: a class is DELTA-PINNED at value v by mu=v, sigma=tiny (a TruncatedNormal
    collapsing to v). For DLA the latent scale is hardcoded 1.0, so DLA is delta-pinned via mu_DLA
    tiny (-> _dla_raw_mu very negative -> softplus ~ 0). The sg[2] we set is INERT but kept tidy."""
    mu = np.asarray(ctx.alpha_hcd_mu, float).copy()        # (3,) [LLS, subDLA, DLA] deployed pin centers
    sg = np.asarray(ctx.alpha_hcd_sigma, float).copy()     # (3,) deployed sigmas
    mu0, sg0 = mu.copy(), sg.copy()
    # tiny fractional width for a delta-pin of LLS/subDLA at a NONZERO center (so sigma scales with
    # the center). frac 1e-4 -> sigma ~1e-5 of the center: a true delta, not a marginalization.
    PIN_FRAC = 1e-4
    if rung == "marg":
        pass                                               # deployed prior on all three (no-op) -- FLOATED
    elif rung == "tierp":
        # DELTA-PIN LLS + subDLA at their DEPLOYED PIN CENTERS (mu unchanged), sigma->tiny so they are
        # NOT marginalized (a fixed, un-marginalized correction WRONG vs the sim-truth). DLA pinned -> 0.
        sg[0] = PIN_FRAC * mu0[0]
        sg[1] = PIN_FRAC * mu0[1]
        mu[2] = PIN_MU_DLA
        sg[2] = PIN_FRAC * PIN_MU_DLA                      # inert (DLA scale hardcoded 1.0) but kept tidy
    elif rung == "clean":
        # DELTA-PIN ALL THREE -> 0. LLS/subDLA via (mu, sigma) ~ 0; DLA via mu_DLA tiny.
        mu[:] = [PIN_MU, PIN_MU, PIN_MU_DLA]
        sg[:] = [PIN_SIGMA, PIN_SIGMA, PIN_FRAC * PIN_MU_DLA]
    else:
        raise ValueError(f"unknown rung {rung!r}")
    ctx = ctx._replace(alpha_hcd_mu=jnp.asarray(mu), alpha_hcd_sigma=jnp.asarray(sg))
    return ctx, dict(mu_before=mu0, sg_before=sg0, mu_after=mu, sg_after=sg)


def _prior_draw_means(ctx, n=200000, key=0):
    """Quick prior-predictive means of (alpha_lls, alpha_subdla, alpha_dla) under the rung's ctx —
    EXACTLY the legacy _hcd_sites sampling (TruncatedNormal low=0 for LLS/subDLA; softplus(Normal)
    for DLA). The validation gate's empirical check."""
    mu = np.asarray(ctx.alpha_hcd_mu, float); sg = np.asarray(ctx.alpha_hcd_sigma, float)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(int(key)), 3)
    a_lls = np.asarray(dist.TruncatedNormal(mu[0], sg[0], low=0.0).sample(k1, (n,)))
    a_sub = np.asarray(dist.TruncatedNormal(mu[1], sg[1], low=0.0).sample(k2, (n,)))
    raw = np.asarray(dist.Normal(float(_dla_raw_mu(mu[2])), 1.0).sample(k3, (n,)))
    a_dla = np.asarray(jax.nn.softplus(jnp.asarray(raw)))
    return float(a_lls.mean()), float(a_sub.mean()), float(a_dla.mean())


def _validate_rung(ctx, rung):
    """STEP-3 validation gate: PRINT the effective per-class (mu, sigma) + prior-draw means and
    ASSERT the rung's pin/float contract. Returns the DLA-floor value for the report."""
    mu = np.asarray(ctx.alpha_hcd_mu, float); sg = np.asarray(ctx.alpha_hcd_sigma, float)
    m_lls, m_sub, m_dla = _prior_draw_means(ctx)
    print(f"[rung-{rung}] effective alpha_hcd_mu    = {np.array2string(mu, precision=6)}")
    print(f"[rung-{rung}] effective alpha_hcd_sigma = {np.array2string(sg, precision=6)}")
    # ratio = sigma/mu (the deployed fractional width) for the floated classes.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sg / mu
    print(f"[rung-{rung}] sigma/mu (LLS,subDLA,DLA) = {np.array2string(ratio, precision=4)} "
          f"(deployed frac {DEPLOYED_FRAC})")
    print(f"[rung-{rung}] prior-draw MEANS alpha_lls={m_lls:.3e} alpha_subdla={m_sub:.3e} "
          f"alpha_dla={m_dla:.3e}")
    TOL = 1e-3
    # a delta-pin shows as sigma/mu ~ 0 (NOT marginalized) and a spread (std) collapsed to ~0.
    DELTA_FRAC_TOL = 1e-3            # ratio[i] below this == delta-pinned (not the 0.15/0.40 deployed)
    m_lls_c, m_sub_c, m_dla_c = float(mu[0]), float(mu[1]), float(mu[2])  # delta-pin targets (= centers)
    if rung == "clean":
        # all three DELTA-PINNED at 0: draw means <= TOL and the ABSOLUTE widths ~ 0 (not marginalized).
        # NB the sigma/mu RATIO is degenerate at a zero center (mu=sigma=1e-9 => ratio 1); check the
        # absolute sigma instead -- a delta at 0 is sigma << TOL.
        assert m_lls <= TOL, f"clean: alpha_lls draw-mean {m_lls:.3e} > {TOL}"
        assert m_sub <= TOL, f"clean: alpha_subdla draw-mean {m_sub:.3e} > {TOL}"
        assert m_dla <= TOL, f"clean: alpha_dla draw-mean {m_dla:.3e} > {TOL}"
        assert sg[0] <= TOL and sg[1] <= TOL, \
            f"clean: LLS/subDLA NOT delta-pinned (abs sigma {sg[0]:.2e},{sg[1]:.2e})"
    elif rung == "tierp":
        # LLS+subDLA DELTA-PINNED at the DEPLOYED CENTERS (mu[0:2]), sigma->tiny => NOT marginalized.
        assert m_dla <= TOL, f"tierp: alpha_dla draw-mean {m_dla:.3e} > {TOL} (DLA NOT pinned ~0)"
        assert ratio[0] <= DELTA_FRAC_TOL, \
            f"tierp: LLS sigma/mu {ratio[0]:.2e} > {DELTA_FRAC_TOL} (LLS IS marginalized, should be delta-pinned)"
        assert ratio[1] <= DELTA_FRAC_TOL, \
            f"tierp: subDLA sigma/mu {ratio[1]:.2e} > {DELTA_FRAC_TOL} (subDLA IS marginalized, should be delta-pinned)"
        # the draw means must sit AT the deployed pin centers (delta), NOT at 0.
        assert abs(m_lls - m_lls_c) <= max(1e-4, 0.02 * m_lls_c), \
            f"tierp: alpha_lls draw-mean {m_lls:.4e} != pin center {m_lls_c:.4e}"
        assert abs(m_sub - m_sub_c) <= max(1e-4, 0.02 * m_sub_c), \
            f"tierp: alpha_subdla draw-mean {m_sub:.4e} != pin center {m_sub_c:.4e}"
        assert m_lls_c > TOL and m_sub_c > TOL, \
            f"tierp: pin centers should be the NONZERO deployed pin ({m_lls_c:.3e},{m_sub_c:.3e})"
    elif rung == "marg":
        # all three FLOATED at the deployed widths.
        assert abs(ratio[0] - DEPLOYED_FRAC[0]) < 1e-3, \
            f"marg: LLS width {ratio[0]:.4f} != deployed {DEPLOYED_FRAC[0]}"
        assert abs(ratio[1] - DEPLOYED_FRAC[1]) < 1e-3, \
            f"marg: subDLA width {ratio[1]:.4f} != deployed {DEPLOYED_FRAC[1]}"
        assert m_lls > TOL and m_sub > TOL and m_dla > TOL, \
            f"marg: a class is pinned ({m_lls:.3e},{m_sub:.3e},{m_dla:.3e}) -- should all float"
    print(f"[rung-{rung}] VALIDATION GATE PASSED (DLA-floor draw-mean = {m_dla:.3e})")
    return m_dla


# ---- dN/dX post-processing (verbatim from desi_hcd_prior_sensitivity.py) --------------------------
def _build_xbar_fn(d):
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
    # EXACT inverse in mode="mask" (readout defect B, 2026-07-22): posterior draws from the
    # pre-2026-07-22 unbounded alpha prior can legitimately leave the occupancy simplex after
    # z-scaling; such entries come back NaN (counted + printed) instead of the old silent
    # saturation at dN/dX = 27.631021/Xbar.
    a = np.asarray(alpha_pivot_draws)
    zg = np.asarray(z_global, float)
    s_c = np.asarray(HCD_INCIDENCE_SLOPE, float)
    Xb = np.asarray(xbar_fn(zg), float)
    shape = ((1.0 + zg)[:, None] / (1.0 + HCD_Z_PIVOT)) ** s_c[None, :]
    out = np.empty((a.shape[0], len(zg), 3))
    n_invalid = 0
    for j in range(len(zg)):
        az = a * shape[j][None, :]
        out[:, j, :], ok = alpha_to_dndx_exact(az, float(Xb[j]), float(zg[j]), mode="mask")
        n_invalid += int(np.size(ok) - np.count_nonzero(ok))
    if n_invalid:
        print(f"[dndx-readout] {n_invalid}/{a.shape[0] * len(zg)} draw-z entries outside the "
              f"exact-inverse domain (negative alpha or sum(alpha) >= 1) -> NaN "
              f"(pre-2026-07-22 code silently saturated these at 27.631021/Xbar)")
    return out, zg, Xb


def _col(names, draws, key):
    names = list(names)
    return np.asarray(draws)[:, names.index(key)] if key in names else None


def _alpha_pivot_block(rec):
    names = list(rec["names"]); draws = np.asarray(rec["draws"]); tv = np.asarray(rec["truth_vec"])
    cols = [names.index(k) for k in ("alpha_lls", "alpha_subdla", "alpha_dla")]
    return draws[:, cols], tv[cols]


def _tau0_block(rec):
    """The per-z tau0 draws + truth (tau0_z* columns) — so the tau0/dtau0 bias is recoverable from
    the pkl (the report-tau0-dtau0-bias note: mean flux is the suspected n_s driver)."""
    names = list(rec["names"]); draws = np.asarray(rec["draws"]); tv = np.asarray(rec["truth_vec"])
    cols = [i for i, n in enumerate(names) if n.startswith("tau0_z")]
    if not cols:
        return None, None
    return draws[:, cols], tv[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=["clean", "tierp", "marg"], required=True,
                    help="clean=pin all 3 alpha~0; tierp=float LLS+subDLA, pin DLA~0; "
                         "marg=deployed prior on all 3")
    ap.add_argument("--mock", type=int, default=0,
                    help="held-out-sim mock index m -> held_out_sims(fold)[m %% n_sims]")
    ap.add_argument("--fold", type=int, default=0,
                    help="LOSO fold for held_out_sims (n_s-sorted: 0=low edge default)")
    ap.add_argument("--n-mocks", type=int, default=8,
                    help="for fold_in(seed,m) reproducibility (matched across rungs)")
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--single-member", action="store_true",
                    help="final_prod_seed0 only (cheap de-risk; NOT the production object)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast pipeline check: n_warmup=20, n_samples=30 (overrides --n-*)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.smoke:
        a.n_warmup, a.n_samples = 20, 30
        print(f"[smoke] n_warmup={a.n_warmup} n_samples={a.n_samples}")
    os.makedirs(a.out_dir, exist_ok=True)

    # SKIP-IF-EXISTS (resume safety): the per-mock pkl is the unit.
    path = os.path.join(a.out_dir, f"mock_{a.mock:04d}.pkl")
    if os.path.exists(path):
        print(f"=== rung {a.rung} mock {a.mock} pkl exists at {path} -- SKIP ===")
        return

    # PINNED members (freeze decision 6): manifest-verified (sha256 + exact pairing + count +
    # stray-member tripwire) via checkpoints/production_ensemble_manifest.json, NOT a glob.
    members = production_member_paths(checkpoints_dir=os.path.dirname(PROD_PREFIX))
    ens = [members[0]] if a.single_member else members
    print(f"[ensemble] {len(ens)} member(s): {[os.path.basename(m) for m in ens]}")

    # DESI leg only, DR1 metals (metals_on + sample_metals) — the DEPLOYED DESI fit, built EXACTLY
    # as run_prod_sbc_shard.py's --leg DESI path (production MF + emucoh, current C_emu,
    # hierarchical_hcd=False, survey=None closure).
    ctx, d = build_legb_ctx(
        use_xclass=True,
        with_mf=True, mf_with_floor=True,
        mf_emucoh=True, mf_emucoh_offdiag_only=True, mf_emucoh_npz=None,
        mf_shape=False,
        with_eboss=False, metals_on=True, sample_metals=True,
        hierarchical_hcd=False, ensemble_ckpts=ens)
    _pre = [l.name for l in ctx.legs]
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name.upper().startswith("DESI")])
    assert len(ctx.legs) == 1, f"--leg DESI: expected 1 leg, got {[l.name for l in ctx.legs]} (pre={_pre})"
    print(f"[per-leg] DESI: legs={[l.name for l in ctx.legs]} metals_on=True sample_metals=True")
    assert not getattr(ctx, "hierarchical_hcd", False), "expected legacy HCD sites (hierarchical_hcd=False)"

    # ---- APPLY THE RUNG OVERRIDE (host-side ctx field swap) -----------------------------------
    deployed_mu = np.asarray(ctx.alpha_hcd_mu, float).copy()
    deployed_sg = np.asarray(ctx.alpha_hcd_sigma, float).copy()
    print(f"[deployed] alpha_hcd_mu={np.array2string(deployed_mu, precision=6)} "
          f"alpha_hcd_sigma={np.array2string(deployed_sg, precision=6)}")
    ctx, ov = _apply_rung(ctx, a.rung)
    if a.rung == "tierp":
        print(f"[rung-tierp] DELTA-PIN centers (deployed dN/dX pin): "
              f"alpha_LLS={deployed_mu[0]:.6f} alpha_subDLA={deployed_mu[1]:.6f} "
              f"(alpha_DLA pinned ~0); these are the FIXED, un-marginalized correction values.")

    # ---- STEP-3 VALIDATION GATE (print + assert the pin/float contract) -----------------------
    dla_floor = _validate_rung(ctx, a.rung)

    # ---- the matched held-out-sim mock + NUTS (deployed settings) -----------------------------
    # leg_a=False => held-out-sim mock (truth = sim measured power thru the production MF), HCD-
    # PRIOR-INDEPENDENT => the three rungs over the same (seed, fold, m) share IDENTICAL data.
    records = run_legb(ctx, d, n_mocks=a.n_mocks, mock_indices=[int(a.mock)], return_per_mock=True,
                       leg_a=False, fold=int(a.fold), n_warmup=a.n_warmup, n_samples=a.n_samples,
                       seed=a.seed, dense_mass=True, max_tree_depth=a.max_tree_depth, verbose=True)
    assert len(records) == 1, f"expected 1 record for mock {a.mock}, got {len(records)}"
    rec = records[0]

    # ---- post-process: cosmology + per-class dN/dX(z) + tau0 (same as prior-sensitivity) -------
    names = list(rec["names"]); draws = np.asarray(rec["draws"]); tv = np.asarray(rec["truth_vec"])
    n_s_draws = _col(names, draws, "ns"); A_p_draws = _col(names, draws, "Ap")
    alpha_pivot_draws, alpha_pivot_truth = _alpha_pivot_block(rec)
    n_s_truth = float(tv[names.index("ns")]); A_p_truth = float(tv[names.index("Ap")])
    tau0_draws, tau0_truth = _tau0_block(rec)

    # ---- "WRONG"-CHECK: the closure mock's SIM-TRUTH alpha vs the tierp delta-pin centers ------
    # alpha_pivot_truth is THIS held-out sim's measured pivot HCD amplitude (the closure truth). For
    # tierp the fit is delta-pinned at deployed_mu[0:2] (the literature pin). Confirm truth != pin so
    # the fixed correction is genuinely WRONG (the whole point of the rung).
    t_lls, t_sub, t_dla = float(alpha_pivot_truth[0]), float(alpha_pivot_truth[1]), float(alpha_pivot_truth[2])
    print(f"[mock] SIM-TRUTH alpha_pivot LLS={t_lls:.6f} subDLA={t_sub:.6f} DLA={t_dla:.6f}")
    if a.rung == "tierp":
        d_lls = t_lls - deployed_mu[0]; d_sub = t_sub - deployed_mu[1]
        print(f"[rung-tierp] SIM-TRUTH vs DELTA-PIN: "
              f"LLS truth={t_lls:.6f} pin={deployed_mu[0]:.6f} (Δ={d_lls:+.6f}); "
              f"subDLA truth={t_sub:.6f} pin={deployed_mu[1]:.6f} (Δ={d_sub:+.6f})")
        if abs(d_lls) < 1e-6 and abs(d_sub) < 1e-6:
            print("[rung-tierp] WARNING: sim-truth ~= the delta-pin -> 'wrong' correction is NOT real "
                  "for this mock (clean vs tierp would coincide).")
        else:
            print("[rung-tierp] CONFIRMED: sim-truth DIFFERS from the delta-pin -> the fixed "
                  "correction is genuinely WRONG (clean<tierp<marg ladder is meaningful).")

    xbar_fn = _build_xbar_fn(d)
    dndx_draws, z_dndx, Xbar_z = _dndx_from_pivot_draws(alpha_pivot_draws, ctx.z_global, xbar_fn)
    dndx_truth, _, _ = _dndx_from_pivot_draws(alpha_pivot_truth[None, :], ctx.z_global, xbar_fn)
    dndx_truth = dndx_truth[0]

    out = dict(
        rung=a.rung, widths=tuple(float(w) for w in DEPLOYED_FRAC),
        mock=int(a.mock), sim=rec.get("sim"), seed=int(a.seed),
        names=names, draws=draws, truth_vec=tv,
        ns_draws=n_s_draws, Ap_draws=A_p_draws,
        alpha_pivot_draws=alpha_pivot_draws, alpha_pivot_truth=alpha_pivot_truth,
        ns_truth=n_s_truth, Ap_truth=A_p_truth,
        # per-z tau0 (mean-flux) draws + truth (report-tau0-dtau0-bias: n_s driver suspect)
        tau0_draws=tau0_draws, tau0_truth=tau0_truth,
        dndx_z=z_dndx, dndx_Xbar=Xbar_z, dndx_draws=dndx_draws, dndx_truth=dndx_truth,
        dndx_class_order=("LLS", "subDLA", "DLA"),
        # effective (rung) + deployed prior centers/widths for the dN/dX-vs-center panel
        alpha_hcd_mu=np.asarray(ctx.alpha_hcd_mu), alpha_hcd_sigma=np.asarray(ctx.alpha_hcd_sigma),
        deployed_alpha_hcd_mu=deployed_mu, deployed_alpha_hcd_sigma=deployed_sg,
        # tierp delta-pin centers (= deployed LLS/subDLA pin; DLA~0) + the sim-truth alpha they miss
        tierp_pin_centers=np.array([deployed_mu[0], deployed_mu[1], 0.0]),
        sim_truth_alpha_pivot=np.asarray(alpha_pivot_truth),
        dla_floor_draw_mean=float(dla_floor),
        n_div=int(rec.get("n_div", 0)), L=int(rec.get("L", draws.shape[0])),
        config=dict(leg="DESI", metals_on=True, leg_a=False, fold=int(a.fold),
                    rung=a.rung, ensemble=[os.path.basename(m) for m in ens],
                    n_warmup=a.n_warmup, n_samples=a.n_samples,
                    max_tree_depth=a.max_tree_depth, dense_mass=True, target_accept=0.9,
                    hcd_prior_frac_sigma=tuple(float(w) for w in DEPLOYED_FRAC),
                    smoke=bool(a.smoke)),
    )
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    print(f"[rung-{a.rung}] mock {a.mock} sim={str(rec.get('sim'))[:24]} "
          f"n_div={out['n_div']} L={out['L']} -> {path}")
    print(f"  n_s truth={n_s_truth:.4f} post-mean={float(np.mean(n_s_draws)):.4f} "
          f"A_p truth={A_p_truth:.3e} post-mean={float(np.mean(A_p_draws)):.3e}")


if __name__ == "__main__":
    main()
