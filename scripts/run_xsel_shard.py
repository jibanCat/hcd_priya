"""KS X-battery shard runner (campaign X-battery-2; PROPOSAL-extreme-battery-v2 as adopted by
PI decisions of record #7 + execution annex, 2026-07-24). LAUNCH TOOLING ONLY: zero edits to
the frozen forward/prior/lock/registry; every truth path routes through frozen callables.

One arm x one mock of the K0-paired battery. Truth construction per arm kind (registry
scripts/ks_xsel_arms.py):

  mixture_corner (X2/X3): the frozen run_legb truth_fn hook draws the clean truth
      (draw_leg_a_leg_truth on the DEPLOYED mapped KS ctx) and the driver overrides the
      alpha pivot triple + every alpha_hcd_z row to the pure corner (alpha_cls = 1, others 0;
      host-side dict edit, apply_class_truth_boost_profile-style copy semantics). The mock
      data, covariance and readout then flow through run_legb UNCHANGED.
  mixture_profile (X4): same hook; alpha_LLS(z) = f_sel(z) (peak-normalized inverted-U),
      other classes 0.
  dndx_displaced (K8a/b/c): same hook; the drawn dN/dX sites are displaced by exactly
      +/-1 DEPLOYED prior sigma and re-mapped through the frozen occupancy map
      (dndx_wc.w_c_corrected, READ-ONLY import; the _ks_dndx_sites deterministic math is
      mirrored host-side and a per-draw ROUND-TRIP assert against the drawn alpha rows
      guards the mirror at 1e-9). In-simplex for every draw by construction.
  data_swap (X1): NO frozen hook exists for data-side P1D substitution, so this driver
      reimplements run_legb's per-mock loop FROM FROZEN CALLABLES (draw_leg_a_leg_truth,
      make_leg_a_legmock, _run_nuts_legb + the per-mock readout assembly, all read-only
      imports; proposal Sec 8). The noise draw is recovered EXACTLY as
      eps = P_data - info["truth_on_leg"], the truth vector is swapped to
      P_swap = truth_on_leg_clean(draw) * ratio_rows_X1_dla100 (the sha-pinned
      dilution-CORRECTED table curve transported to the drawn mean-flux/cosmology truth,
      contract scripts/xsel_truth_contract.md), and the mock is rebuilt as
      leg._replace(P_data = P_swap + eps). Drift risk is controlled by the PRE-REGISTERED
      swap-off equivalence gate (--swap-off-check): with the swap disabled the per-mock
      records must be BYTE-IDENTICAL to run_legb's at the same seeds (round-2 revision 6;
      per-mock records, not file bytes, because shard pkls embed wall-clock meta fields).
      This gate must PASS before any X1 fit counts.

inject_spec is NEVER passed (no double application). Truth-admissibility tripwire fires per
mock. Every pkl stamps: the X registry signature (truth-table sha INCLUDED), arm, truth-table
sha + convention flags, the prior-constants tripwire block (byte-equal to the reused K0
shards so the analyzer's cross-shard equality holds), forward_stamp, and the span log.

PAIRING SEED: 20260615 (ks_xsel_arms.PAIR_SEED), the reused R4/R5 K0 shards' stamped seed.
Round-2 revision 4b requires X-vs-K0 pairs to SHARE truth-theta and noise keys, which is
only realizable at the K0 seed; the build brief's "fresh 20260724 + fold_in tag" wording is
NOT adopted for the truth/noise keys (deviation recorded; r6x at 20260724 is domain-separated
automatically because the seeds differ). --allow-nonpair-seed exists for diagnostics only and
poisons the pkl name guard (analyzer refuses on run_kw seed mismatch anyway).

Pre-registered caveats (registry docstring): the emulator-span WARNING fires BY DESIGN at the
corners; alpha ranks are degenerate there (truth above every draw) and rank uniformity is NOT
expected on displaced-truth arms; the corner/profile arms' sites_extra dN/dX-site truths are
the UNDERLYING CLEAN DRAW's sites (the corner alpha is not the map image of any site vector).
At the X1 corner the mock covariance keeps clean-composition C_emu weights (disclosed, not
repaired). The gate is n_s/A_p bias, never alpha recovery.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import json
import os
import pickle
import time

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401 (x64)
import jax
import jax.numpy as jnp
import hcd_analysis.emulator.closure_legb as CL
import hcd_analysis.emulator.inference as INF
from hcd_analysis.emulator.dndx_wc import w_c_corrected   # READ-ONLY frozen map import (K8)
from scripts.run_dnuis_bias_shard import build_arm_ctx
from scripts.run_ks_selboost_shard import assert_ks_forward, assert_prior_state
import scripts.ks_xsel_arms as XA
from scripts.ks_xsel_arms import batch_cells  # noqa: F401 (re-export for the batch script)


# ---------------------------------------------------------------------------------------- #
#  Truth builders (host-side, copy semantics; frozen callables only).
# ---------------------------------------------------------------------------------------- #
def build_corner_truth(truth_pack, cls_idx, n_zg):
    """X2/X3 corner: COPY of truth_pack with alpha pivot + all rows at the pure corner."""
    out = dict(truth_pack)
    a = np.zeros(3, float)
    a[cls_idx] = 1.0
    out["alpha_hcd"] = a
    out["alpha_hcd_z"] = XA.corner_alpha_rows(n_zg, cls_idx)
    assert XA.truth_admissible(out["alpha_hcd_z"]), "corner truth left the mixture simplex (bug)"
    return out


def build_profile_truth(truth_pack, cls_idx, z_global):
    """X4: COPY with alpha_cls(z) = f_sel(z), other classes 0; pivot at exactly z = Z_PIVOT."""
    out = dict(truth_pack)
    a = np.zeros(3, float)
    a[cls_idx] = float(XA.f_sel(float(XA.Z_PIVOT)))
    out["alpha_hcd"] = a
    out["alpha_hcd_z"] = XA.profile_alpha_rows(np.asarray(z_global, float), cls_idx)
    assert XA.truth_admissible(out["alpha_hcd_z"]), "profile truth left the simplex (bug)"
    return out


def k8_alpha_from_raw(raw, ks_dndx_ref, ks_xbar_z, z_global, ks_dndx_ref_pivot, ks_xbar_pivot,
                      *, d_eps=0.0, d_kappa=0.0):
    """Host-side mirror of the frozen _ks_dndx_sites deterministic forward (closure_legb
    lines 2715-2736), with optional eps_lls/kappa_lls displacements added PRE-map. Returns
    (alpha_rows (nZg,3), alpha_pivot (3,)). Uses the frozen w_c_corrected + _simplex_tie_norm
    read-only; the caller MUST round-trip the undisplaced output against the drawn truth."""
    zg = jnp.asarray(z_global)
    ratio = (1.0 + zg) / (1.0 + float(XA.Z_PIVOT))
    eps = float(raw["eps_lls"]) + float(d_eps)
    kap = float(raw["kappa_lls"]) + float(d_kappa)
    m_sub, t_sub = float(raw["m_sub"]), float(raw["t_sub"])
    dla_raw, t_dla = float(raw["dla_raw"]), float(raw["t_dla"])
    dla_amp = jax.nn.softplus(dla_raw) / jax.nn.softplus(float(INF.KS_DNDX_DLA_RAW_MU0))
    fac = jnp.stack([jnp.exp(eps) * ratio ** kap,
                     jnp.exp(m_sub) * ratio ** t_sub,
                     dla_amp * ratio ** t_dla], axis=-1)
    alpha_z = CL._simplex_tie_norm(
        w_c_corrected(jnp.asarray(ks_dndx_ref) * fac, jnp.asarray(ks_xbar_z), zg)[..., 1:])
    fac_piv = jnp.stack([jnp.exp(eps), jnp.exp(m_sub), dla_amp])
    a_piv = CL._simplex_tie_norm(
        w_c_corrected(jnp.asarray(ks_dndx_ref_pivot) * fac_piv,
                      jnp.asarray(float(ks_xbar_pivot)),
                      jnp.asarray(float(XA.Z_PIVOT)))[..., 1:])
    return np.asarray(alpha_z, float), np.asarray(a_piv, float)


def build_k8_truth(truth_pack, ctx, site, n_sigma, sigma_expect):
    """K8: COPY with the drawn dN/dX site displaced by n_sigma DEPLOYED prior sigma, alpha
    rebuilt through the frozen map. Round-trip guard: the UNDISPLACED mirror must reproduce
    the drawn alpha rows/pivot to 1e-9 (protects against mirror-vs-frozen-prior drift)."""
    assert site in ("eps_lls", "kappa_lls"), site
    sigma = float({"eps_lls": INF.KS_DNDX_SIGMA_EPS, "kappa_lls": INF.KS_DNDX_SIGMA_KAPPA}[site])
    assert sigma == float(sigma_expect), (
        f"deployed {site} sigma {sigma} != registry pin {sigma_expect}: the prior this arm "
        f"was designed against has drifted; refusing to run")
    raw = truth_pack["raw"]
    args = (raw, ctx.ks_dndx_ref, ctx.ks_xbar_z, np.asarray(ctx.z_global, float),
            ctx.ks_dndx_ref_pivot, ctx.ks_xbar_pivot)
    base_rows, base_piv = k8_alpha_from_raw(*args)
    assert np.allclose(base_rows, np.asarray(truth_pack["alpha_hcd_z"], float),
                       rtol=0, atol=1e-9), \
        "K8 mirror round-trip FAILED on alpha rows (host mirror drifted from _ks_dndx_sites)"
    assert np.allclose(base_piv, np.asarray(truth_pack["alpha_hcd"], float),
                       rtol=0, atol=1e-9), \
        "K8 mirror round-trip FAILED on the pivot triple"
    d_eps = float(n_sigma) * sigma if site == "eps_lls" else 0.0
    d_kap = float(n_sigma) * sigma if site == "kappa_lls" else 0.0
    rows, piv = k8_alpha_from_raw(*args, d_eps=d_eps, d_kappa=d_kap)
    assert XA.truth_admissible(rows, strict_interior=True), \
        "K8 displaced truth left the strict simplex interior (map recompute broke)"
    out = dict(truth_pack)
    out["alpha_hcd"] = piv
    out["alpha_hcd_z"] = rows
    disp = float(raw[site]) + (d_eps if site == "eps_lls" else d_kap)
    out[site] = disp                                   # sites_extra truth = the DISPLACED truth
    out["raw"] = dict(raw, **{site: disp})
    return out


def make_truth_fn(ctx, arm_id):
    """The run_legb truth_fn (key -> truth_pack) for a mixture-flavor arm."""
    e = XA.ARMS[arm_id]
    n_zg = len(np.asarray(ctx.z_global))

    def truth_fn(k):
        tp = CL.draw_leg_a_leg_truth(ctx, k)
        if e["kind"] == "mixture_corner":
            return build_corner_truth(tp, int(e["cls"]), n_zg)
        if e["kind"] == "mixture_profile":
            return build_profile_truth(tp, int(e["cls"]), np.asarray(ctx.z_global, float))
        if e["kind"] == "dndx_displaced":
            return build_k8_truth(tp, ctx, e["site"], e["n_sigma"], e["sigma_expect"])
        raise ValueError(f"truth_fn does not handle kind {e['kind']!r} (X1 is data_swap)")

    return truth_fn


# ---------------------------------------------------------------------------------------- #
#  X1 data-side loop: run_legb's leg-A per-mock loop reimplemented from frozen callables
#  (closure_legb.run_legb lines 3661-3805 mirrored; the ONLY functional insertion is the
#  data-side swap between mock construction and NUTS). ratio_rows=None disables the swap,
#  which is the pre-registered byte-identity branch.
# ---------------------------------------------------------------------------------------- #
def x1_per_mock(ctx, d, idxs, *, seed, n_warmup, n_samples, max_tree_depth,
                dense_mass=True, ratio_rows=None, verbose=True):
    assert not getattr(ctx, "hcd_2d_tilt", False) and not getattr(ctx, "hierarchical_hcd", False)
    assert not getattr(ctx, "sample_metals", False), "KS is metal-free; X1 loop assumes it"
    CL._check_single_instrument_for_res(ctx.legs, getattr(ctx, "sample_res", False))
    key0 = jax.random.PRNGKey(int(seed))
    fid_core = {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
                for name, v in CL._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}
    per_mock = []
    for m in idxs:
        k_truth, k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, m), 3)
        truth_pack = CL.draw_leg_a_leg_truth(ctx, k_truth)
        mock_legs, info = CL.make_leg_a_legmock(ctx, fid_core, truth_pack, k_mock)
        core_per_leg = fid_core
        sim = "leg_a_prior"
        swap_stamp = None
        if ratio_rows is not None:
            assert len(mock_legs) == 1, "X1 swap assumes the single restricted KS leg"
            leg = mock_legs[0]
            P_clean = np.asarray(info["truth_on_leg"][leg.name], float)
            eps = np.asarray(leg.P_data, float) - P_clean       # EXACT noise recovery
            P_swap = P_clean * np.asarray(ratio_rows, float)
            assert P_swap.shape == P_clean.shape and np.all(np.isfinite(P_swap)) \
                and np.all(P_swap > 0), "X1 swapped truth non-finite/non-positive (tripwire)"
            mock_legs = [leg._replace(P_data=P_swap + eps)]
            swap_stamp = dict(P_clean_truth=P_clean, P_swap_truth=P_swap.copy(), eps=eps,
                              fork="dilution_corrected")

        base_seed = int(jax.random.randint(k_nuts, (), 0, 2 ** 31 - 1))
        ta_sched = (0.9,) + tuple(CL.DIVERGENCE_RETRY_TARGET_ACCEPT)
        samples = n_div = None
        for attempt, ta in enumerate(ta_sched):
            samples, n_div = CL._run_nuts_legb(
                ctx, mock_legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
                seed=base_seed + attempt, target_accept=ta, dense_mass=dense_mass,
                max_tree_depth=max_tree_depth)
            if n_div == 0:
                break
            if verbose:
                print(f"  [mock {m}] sim={sim[:20]}... {n_div} divergence(s) at ta={ta}"
                      + (" -> retry" if attempt < len(ta_sched) - 1 else ""))

        kept_global = truth_pack["kept_global_z"]
        draws = CL._draws_matrix(samples, kept_global)
        draws_t, step, ess_min = CL.thin_to_ess(draws)
        L = draws_t.shape[0]
        truth_vec = np.concatenate([
            truth_pack["theta9"],
            truth_pack["tau0_global"][kept_global],
            truth_pack["alpha_hcd"]])
        _a_si_true = float(truth_pack.get("a_siiii", 0.0))
        _a_si2_true = float(truth_pack.get("a_siii", 0.0))
        ll_true = float(CL._data_loglik_legcore(
            ctx, jnp.asarray(truth_pack["theta9"]),
            jnp.asarray(truth_pack["tau0_global"]),
            jnp.asarray(truth_pack["alpha_hcd_z"]), mock_legs, core_per_leg,
            a_siiii=_a_si_true, a_siii=_a_si2_true, require_zresolved=True))
        ll_draws = CL._loglik_of_draws(ctx, mock_legs, core_per_leg, samples, kept_global)
        ll_draws_t = ll_draws[::step][:L]
        sites_extra = {}
        _truth_raw = truth_pack.get("raw") or {}
        for nm in CL.SELF_DRAWN_EXTRA_SITES:
            if nm in samples:
                dr = np.asarray(samples[nm])[::step][:L]
                sites_extra[nm] = dict(
                    draws=dr, truth=float(truth_pack.get(nm, _truth_raw.get(nm, np.nan))))
        sites_extra.update(CL._metal_node_sites_extra(samples, step, L, None, ctx, True))
        sites_extra.update(CL._resolution_sites_extra(samples, step, L, None, True))
        rec = dict(sim=sim, truth_vec=truth_vec, draws=draws_t, L=L,
                   ll_true=ll_true, ll_draws=ll_draws_t,
                   names=CL._packed_names_for(samples, kept_global),
                   kept_global=kept_global, dropped=info["dropped"],
                   sites_extra=sites_extra, n_div=n_div,
                   truth_alpha_hcd_z=np.array(truth_pack["alpha_hcd_z"], float))
        if swap_stamp is not None:
            rec["xsel_swap"] = swap_stamp        # ADDITIVE key, swap-on only (byte-identity)
        per_mock.append(rec)
        if verbose:
            print(f"  [mock {m}] sim={sim[:24]}... L={L} (step {step}, ess {ess_min:.0f}) "
                  f"nKeptZ={int(kept_global.sum())} div={n_div}")
    return per_mock


def swap_off_check(ctx, d, idxs, *, seed, n_warmup, n_samples, max_tree_depth):
    """The pre-registered X1 equivalence gate (round-2 revision 6): swap-off driver records
    vs run_legb records at the same seeds, compared byte-level per mock. Returns (ok, report)."""
    mine = x1_per_mock(ctx, d, idxs, seed=seed, n_warmup=n_warmup, n_samples=n_samples,
                       max_tree_depth=max_tree_depth, ratio_rows=None)
    ref = CL.run_legb(ctx, d, n_mocks=max(idxs) + 1, mock_indices=idxs,
                      return_per_mock=True, leg_a=True, n_warmup=n_warmup,
                      n_samples=n_samples, seed=seed, dense_mass=True,
                      max_tree_depth=max_tree_depth, verbose=True)
    report = {}
    ok = True
    for m, ra, rb in zip(idxs, mine, ref):
        mism = XA.compare_per_mock_records(ra, rb)
        report[int(m)] = mism
        ok &= not mism
    return ok, report


# ---------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-id", required=True, choices=XA.arm_ids())
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=None,
                    help="default: the registry N for --arm-id")
    ap.add_argument("--seed", type=int, default=XA.PAIR_SEED)
    ap.add_argument("--allow-nonpair-seed", action="store_true",
                    help="diagnostics only: run at a non-K0-pairing seed (the analyzer will "
                         "refuse such pkls on the run_kw seed homogeneity check)")
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    # PRIOR-STATE TRIPWIRE (selboost spec 5c convention): REQUIRED.
    ap.add_argument("--expect-lls-boost", type=float, required=True)
    ap.add_argument("--expect-lls-frac-sigma", type=float, required=True)
    ap.add_argument("--truth-table", default=None,
                    help="override the registry truth-table path (sha pin still enforced)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--swap-off-check", action="store_true",
                    help="X1 only: run the pre-registered swap-off byte-identity gate "
                         "instead of a campaign fit (writes a certificate json, no shard pkl)")
    a = ap.parse_args()

    e = XA.ARMS[a.arm_id]
    if a.swap_off_check:
        assert e["kind"] == "data_swap", "--swap-off-check is the X1 equivalence gate only"
    assert a.seed == XA.PAIR_SEED or a.allow_nonpair_seed, (
        f"--seed {a.seed} != PAIR_SEED {XA.PAIR_SEED}: X-vs-K0 pair identity (round-2 "
        f"revision 4b / annex OQ5) requires the reused K0 shards' seed. Pass "
        f"--allow-nonpair-seed only for diagnostics.")
    if a.n_mocks is None:
        a.n_mocks = int(e["n_mocks"])
    if a.smoke:
        a.n_mocks = 1
        a.n_warmup = min(a.n_warmup, 20)
        a.n_samples = min(a.n_samples, 30)

    # TRUTH TABLE FIRST (fail loud before any compute): the registry signature includes the
    # sha, so NO arm can run before the stage-V tables are emitted, reviewed and pinned.
    tt = XA.load_truth_tables(a.truth_table)
    reg_sig = XA.registry_signature(a.truth_table)

    # prior-state tripwire (cheap, before ctx build), then the DEPLOYED KS forward.
    prior_stamp = assert_prior_state(a.expect_lls_boost, a.expect_lls_frac_sigma)
    ctx, d, _ = build_arm_ctx("lls_excess", "ks", True, use_prod_forward=True)
    assert len(ctx.legs) == 1, f"expected the single restricted KS leg, got {len(ctx.legs)}"
    L = ctx.legs[0]
    assert_ks_forward(ctx, L)
    assert bool(getattr(ctx, "ks_dndx_mapped", False)) is True, \
        "X-battery requires the DEPLOYED mapped KS prior (ctx.ks_dndx_mapped)"
    # built-prior band + width asserts and the mapped-era stamps: byte-follow the selboost
    # runner (run_ks_selboost_shard.py main) so prior_constants equal the reused K0 shards'.
    alpha_mu0 = float(np.asarray(ctx.alpha_hcd_mu)[0])
    alpha_sd0 = float(np.asarray(ctx.alpha_hcd_sigma)[0])
    INF.assert_hcd_pivot_z3(alpha_mu0, z=INF.HCD_Z_PIVOT, where="run_xsel_shard",
                            boost=prior_stamp["lls_survey_boost_ks"])
    assert abs(alpha_sd0 / alpha_mu0 - prior_stamp["lls_frac_sigma_ks"]) < 1e-9
    prior_stamp["alpha_lls_center_built"] = alpha_mu0
    prior_stamp["alpha_lls_sigma_built"] = alpha_sd0
    prior_stamp["hcd_parameterization"] = "dndx_mapped_v2"
    from hcd_analysis.emulator.dndx_wc import w_c_corrected as _wcc
    prior_stamp["alpha_lls_center_built_semantics"] = "LEGACY-AUDIT-ONLY (dormant vector)"
    prior_stamp["alpha_lls_mapped_pivot_center"] = float(np.asarray(
        _wcc(jnp.asarray(ctx.ks_dndx_ref_pivot), jnp.asarray(float(ctx.ks_xbar_pivot)),
             jnp.asarray(3.0)))[1])
    prior_stamp["ks_dndx_ref_z"] = np.asarray(ctx.ks_dndx_ref, float).tolist()
    prior_stamp["ks_xbar_z"] = np.asarray(ctx.ks_xbar_z, float).tolist()
    prior_stamp["ks_dndx_sigma_eps"] = float(INF.KS_DNDX_SIGMA_EPS)
    prior_stamp["ks_dndx_sigma_kappa"] = float(INF.KS_DNDX_SIGMA_KAPPA)

    # LEG-GRID contract vs the pinned table (no interpolation anywhere: refuse on mismatch).
    leg_k = np.asarray(L.k, float)
    leg_z_rows = np.asarray(L.z, float)[np.asarray(L.z_idx, int)]
    assert tt["leg_k"].shape == leg_k.shape and np.allclose(tt["leg_k"], leg_k, rtol=1e-10), \
        "truth-table leg_k != the deployed KS leg k rows (contract violation)"
    assert np.allclose(tt["leg_z"], leg_z_rows, rtol=1e-10), \
        "truth-table leg_z != the deployed KS leg z rows (contract violation)"

    z_global = np.asarray(ctx.z_global, float)
    print(f"[forward] DEPLOYED KS mapped: res_corr_on={bool(ctx.res_corr_on)} "
          f"sample_res={bool(ctx.sample_res)} metals={bool(L.metals_on)} "
          f"k_max={leg_k.max():.4f} dla_forward_frac={L.dla_forward_frac} | "
          f"arm={a.arm_id} kind={e['kind']} n_mocks={a.n_mocks} seed={a.seed} | "
          f"truth-table sha {tt['sha256'][:12]}... reg-sig {reg_sig[:12]}...")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    assert idxs, (f"shard {a.shard}/{a.n_shards} selects no mock at n_mocks={a.n_mocks} "
                  f"(smoke forces n_mocks=1: only shard 0 is a valid smoke cell)")
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)

    if a.swap_off_check:
        print(f"[x1 swap-off gate] {len(idxs)} mock(s), pre-registered byte-identity check")
        t0 = time.time()
        ok, report = swap_off_check(ctx, d, idxs, seed=a.seed, n_warmup=a.n_warmup,
                                    n_samples=a.n_samples, max_tree_depth=a.max_tree_depth)
        os.makedirs(a.out_dir, exist_ok=True)
        cert = os.path.join(a.out_dir, f"xsel_x1_swapoff_cert_shard_{a.shard:03d}"
                                       f"{'.smoke' if a.smoke else ''}.json")
        with open(cert, "w") as fh:
            json.dump(dict(ok=bool(ok), idxs=idxs, seed=a.seed, n_warmup=a.n_warmup,
                           n_samples=a.n_samples, max_tree_depth=a.max_tree_depth,
                           smoke=bool(a.smoke), wall_s=time.time() - t0,
                           forward_signature=CL.forward_stamp(ctx, L)["forward_signature"],
                           mismatches={str(k): v for k, v in report.items()}), fh, indent=1)
        print(f"[x1 swap-off gate] {'PASS' if ok else 'FAIL'} -> {cert}")
        if not ok:
            for m, mism in report.items():
                for line in mism[:20]:
                    print(f"   mock {m}: {line}")
            raise SystemExit(1)
        return

    t0 = time.time()
    if e["kind"] == "data_swap":
        ratio_rows = tt["ratios"][e["table_key"]]
        per_mock = x1_per_mock(ctx, d, idxs, seed=a.seed, n_warmup=a.n_warmup,
                               n_samples=a.n_samples, max_tree_depth=a.max_tree_depth,
                               ratio_rows=ratio_rows)
        mode = "swap"
    else:
        per_mock = CL.run_legb(ctx, d, inject_spec=None, truth_fn=make_truth_fn(ctx, a.arm_id),
                               **run_kw)
        mode = "mixture"
    wall = time.time() - t0

    # POST-HOC truth-admissibility tripwire on the RECORDED rows (belt and braces on top of
    # the in-truth_fn asserts; must be all-pass, K6-redesign Sec 6).
    for m, rec in zip(idxs, per_mock):
        rows = np.asarray(rec["truth_alpha_hcd_z"], float)
        assert XA.truth_admissible(rows, strict_interior=(e["kind"] == "dndx_displaced")), \
            f"mock {m}: recorded truth rows fail the admissibility tripwire"

    # emulator-span logging (fires BY DESIGN at the corners; readout caveat, not a failure).
    wc = np.asarray(d["w_c_cache"], float)
    cache_max = np.nanmax(wc[:, 1:], axis=0)
    truth_max = np.nanmax(np.stack([np.asarray(r["truth_alpha_hcd_z"], float)
                                    for r in per_mock]), axis=(0, 1))
    oos_class = [c for j, c in enumerate(("lls", "subdla", "dla")) if truth_max[j] > cache_max[j]]
    span = dict(cache_alpha_max=cache_max.tolist(), truth_alpha_max=truth_max.tolist(),
                out_of_span_classes=oos_class)
    if oos_class:
        print(f"[span] WARNING (pre-registered, by design at the corners): truth alpha "
              f"exceeds the training-cache grid for {oos_class} "
              f"(truth_max {truth_max} vs cache_max {cache_max})")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, XA.shard_pkl_name(a.arm_id, a.shard, smoke=a.smoke))
    meta = dict(vars(a), mode=mode, paired="cross-pkl vs reused K0 (annex OQ5)", wall_s=wall,
                per_fit_wall_s=wall / max(len(per_mock), 1), run_kw=run_kw)
    meta["forward"] = CL.forward_stamp(ctx, L)
    meta["prior_constants"] = prior_stamp
    meta["arm_stamp"] = dict(
        arm_id=a.arm_id, kind=e["kind"], cls=int(e["cls"]), part1=bool(e["part1"]),
        campaign=XA.CAMPAIGN, registry_signature=reg_sig,
        truth_table=dict(path=tt["path"], sha256=tt["sha256"]),
        convention=tt["convention"],
        expected_direction=XA.EXPECTED_DIRECTIONS[a.arm_id],
        seed_convention=f"PAIR_SEED {XA.PAIR_SEED} (K0 reuse, annex OQ5)",
        f_sel_z_global=(XA.f_sel(z_global).tolist() if e["kind"] == "mixture_profile"
                        else None),
        displacement=(dict(site=e["site"], n_sigma=float(e["n_sigma"]),
                           sigma_deployed=float(e["sigma_expect"]))
                      if e["kind"] == "dndx_displaced" else None),
        x1_fork=("dilution_corrected" if e["kind"] == "data_swap" else None),
        z_global=z_global.tolist())
    meta["span"] = span
    # atomic write: the batch skip-guard is existence-only, so a preempted task must
    # never leave a truncated pkl that "skips as done" (gate-revision memo, CS lens)
    tmp = str(out) + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(dict(arm=a.arm_id, survey="ks", mode=mode, idxs=idxs,
                         per_mock=per_mock, meta=meta), fh)
    os.replace(tmp, out)
    nd = sum(int(r.get("n_div", 0) > 0) for r in per_mock)
    print(f"[ks xsel {a.arm_id} shard {a.shard}] wrote {len(per_mock)} {mode} mock(s) "
          f"({nd} div) -> {out} wall={wall:.1f}s")


if __name__ == "__main__":
    main()
