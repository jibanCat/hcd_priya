#!/usr/bin/env python3
"""JOINT multi-leg REAL-DATA fit driver (per-leg HCD alpha sites; POST-UNBLIND machinery).

THIN wrapper over ``closure_legb.build_legb_joint_ctx`` + ``_run_nuts_legb`` + the
``run_real_fit`` export helpers: each requested leg carries its REAL P1D measurement and its
OWN deployed single-leg HCD alpha prior (``alpha_{lls,subdla,dla_raw}_{leg}`` sites; spec
2026-07-20-per-leg-alpha-sites-spec.md). Chains go out in cobaya/GetDist format (the standing
convention) with the A_p/n_s parameter blind applied by DEFAULT, plus a ``.joint_stamp.json``
audit record (per-leg survey/boost/width/prior + embedded per-leg forward stamps with BOTH
freeze signatures; mixed-signature legs are rejected at stamp time).

SCOPE / STANDING DECISIONS:
  * Joint fits are POST-UNBLIND products (the per-leg blind eBOSS→KS→DESI sequence comes
    first); this driver lands with the dormant capability so the lock attests the exact code.
  * v1 samples NO f_res (sample_res=False): per-leg f_res sites are NOT built and
    build_legb_joint_ctx raises NotImplementedError on >1 resolution-floated leg (PI C4).
  * z-slopes SHARED by default (PI C1 carried default); --per-leg-zslope flips the plumbing.
  * tau0 stays GLOBAL (field convention, shared across legs); cosmology shared; metal sites
    are already per-leg (flatlog2node nodes on metals_on legs only).
  * DESI results privacy: any joint fit including DESI routes to results_local/ (gitignored).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_joint_fit.py --legs DESI,KS
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import zlib

print = functools.partial(print, flush=True)

import numpy as np

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
import jax
import jax.numpy as jnp
from numpyro.infer import init_to_sample

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import blinding as BL
from hcd_analysis.emulator.prod_ensemble import production_member_paths
from hcd_analysis.emulator.seeding import nuts_fold_int
from hcd_analysis.emulator.closure_legb import (
    build_legb_joint_ctx, joint_stamp, _run_nuts_legb, _draws_matrix, _packed_names_for,
    convergence_battery, _ebfmi1)
# reuse the single-leg driver's export machinery (cobaya/GetDist + blinding + nuisance export).
from scripts.run_real_fit import (_assert_norc_ks_cap,
                                  export_getdist, _nuisance_export_keys, _reconstruct_nuisance,
                                  PUBLIC_DIR, PRIVATE_DIR, PROD_PREFIX)

REPO = "/home/mfho/hcd_priya"


def build_joint_ctx(leg_names, *, per_leg_zslope=False, single_member=False,
                    ensemble_glob=None):
    """The PRODUCTION joint ctx: per-leg alpha sites at each leg's deployed single-leg survey
    prior (survey key == leg name), the NORC forward, MF + floor + emucoh (off-diag-only),
    cross-class C_emu, flatlog2node metals on the metal legs.

    DELIBERATE v1 DEVIATIONS from the certified per-leg single-leg forwards (panel
    required-fix 1a; all stamped per leg in joint_stamp as k_max_effective /
    resolution_float):
      * sample_res=False on EVERY leg (PI C4: one shared f_res pair cannot serve the three
        certified instrument widths 0.02/0.05/0.15; per-leg f_res sites are the named
        follow-up and the re-enable condition).
      * KS ks_kwargs OMITTED: without the echelle R_z float the NORC KS cap applies, so the
        joint KS leg runs at k<=0.045 with the historical conservative covariance — NOT the
        certified single-leg 0.065-with-R_z-float band. Joint-KS vs single-leg-KS posterior
        comparisons are therefore at DIFFERENT k_max until per-leg f_res lands (carried in
        the joint-campaign design notes).
    Everything else mirrors run_real_fit.build_real_ctx knob-for-knob."""
    if ensemble_glob is None:
        # PINNED path (freeze decision 6): committed-manifest members (sha256 + pairing +
        # count/order verified, stray-member tripwire), NOT a permissive glob.
        members = production_member_paths(checkpoints_dir=os.path.dirname(PROD_PREFIX))
    else:
        # DIAGNOSTIC override (explicit --ensemble-glob ONLY): NOT the pinned production
        # ensemble. main() stamps the export meta ensemble_pinned=False + the glob used.
        members = sorted(p[:-4] for p in glob.glob(ensemble_glob + ".eqx"))
        if not members:
            raise SystemExit(f"no ensemble checkpoints match --ensemble-glob {ensemble_glob}*.eqx")
        print(f"!!! WARNING: --ensemble-glob override in effect -- this run does NOT use the "
              f"manifest-pinned production ensemble (glob={ensemble_glob}*.eqx -> "
              f"{len(members)} member(s)); meta is stamped ensemble_pinned=False !!!")
    ens = [members[0]] if single_member else members
    norc = CL.prod_norc_forward()
    metals = any(CL.prod_forward_config(n)["metals"] for n in leg_names)
    survey_by_leg = {n: n for n in leg_names}        # per-leg survey key == leg name
    ctx, d = build_legb_joint_ctx(
        survey_by_leg, per_leg_zslope=per_leg_zslope,
        ensemble_ckpts=ens, use_xclass=True,
        with_mf=True, mf_with_floor=True,
        res_corr_on=norc["res_corr_on"],
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        metals_on=metals, sample_metals=metals,
        metal_prior=("flatlog2node" if metals else "uniform"),
        sample_res=False)                            # v1: no shared f_res across instruments (C4)
    if norc["fix_alpha_res"]:
        ctx = ctx._replace(fix_alpha_res=True)
    assert ctx.res_corr_on == norc["res_corr_on"] and ctx.fix_alpha_res == norc["fix_alpha_res"], \
        "prod NORC forward not applied to the joint ctx"
    # the two build_real_ctx fail-loud tripwires, replicated (panel required-fix 1c):
    _assert_norc_ks_cap(ctx)
    from hcd_analysis.emulator import data_likelihood as _DL
    for _leg in ctx.legs:
        if _leg.name == "DESI":
            assert bool(_leg.dla_cov_reduced) == bool(_DL.DESI_DLA_COV_REDUCE), \
                "DESI leg dla_cov_reduced disagrees with the DESI_DLA_COV_REDUCE authority"
    return ctx, d, ens


def _joint_loglik_chain(ctx, core_per_leg, samples):
    """Per-draw joint -lnL re-score (data term) with the PER-LEG alpha dict rebuilt from the
    suffixed alpha_hcd_z_{leg} deterministics — mirrors _legb_model's dict threading."""
    theta = jnp.asarray(samples["theta_unit"])                 # (L,9)
    tau0 = jnp.asarray(samples["tau0_vec"])                    # (L,nZg)
    a_z = {leg.name: jnp.asarray(samples[f"alpha_hcd_z_{leg.name}"]) for leg in ctx.legs}
    nkeys = list(_nuisance_export_keys(samples))
    if not getattr(ctx, "fix_alpha_res", False):
        nkeys += [k for k in ("alpha_res", "alpha_res_slope") if k in samples]
    nkeys += [k for k in ("a_SiIII", "a_SiII") if k in samples]
    nuis = {k: jnp.asarray(samples[k]) for k in nkeys}

    def one(th, t0, al_dict, draw):
        metal_nodes, b_res, alpha_res, a_siiii, a_siii = _reconstruct_nuisance(ctx, draw)
        return CL._data_loglik_legcore(
            ctx, th, t0, al_dict, ctx.legs, core_per_leg, a_siiii=a_siiii, a_siii=a_siii,
            metal_nodes=metal_nodes, alpha_res=alpha_res, b_res_global=b_res,
            require_zresolved=True)
    return jax.vmap(one)(theta, tau0, a_z, nuis)


def run_joint_fit(leg_names, *, n_chains=4, n_warmup=250, n_samples=600, max_tree_depth=10,
                  seed=20260720, per_leg_zslope=False, single_member=False, ensemble_glob=None,
                  verbose=True):
    """Multi-chain dispersed NUTS on the REAL multi-leg data (NO mock anywhere)."""
    ctx, d, members = build_joint_ctx(leg_names, per_leg_zslope=per_leg_zslope,
                                      single_member=single_member, ensemble_glob=ensemble_glob)
    stamp = joint_stamp(ctx)                                   # also rejects mixed signatures
    n_real = int(sum(np.isfinite(np.asarray(leg.P_data)).sum() for leg in ctx.legs))
    # fiducial (mean held-out) z-mean DLA core per leg — the real-fit convention.
    core_per_leg = {}
    for leg in ctx.legs:
        fid = np.asarray(ctx.dla_core_leg[leg.name])
        core_per_leg[leg.name] = jnp.asarray(fid.mean(axis=0) if fid.ndim == 2 else fid)

    zg = np.asarray(ctx.z_global)
    kept_global = np.array([any(np.any(np.isclose(leg.z, zz, atol=1e-3)) for leg in ctx.legs)
                            for zz in zg])

    key0 = jax.random.PRNGKey(int(seed))
    # DETERMINISTIC leg-set fold (panel required-fix 3): python hash() is SipHash-salted per
    # process, so the recorded seed would NOT reproduce the chains. crc32 is stable across
    # processes/machines; the derivation is recorded in the export meta. (The hash() idiom
    # at run_real_fit.py remains as a PI-flagged pre-existing follow-up: changing it there
    # would alter future single-leg chain streams.)
    # P0 2026-07-21: routed through the SHARED helper (hcd_analysis.emulator.seeding) so the two
    # drivers cannot drift. Byte-identical to the previous inline crc32 for every label, so the
    # joint driver's chain stream is UNCHANGED (pinned by tests/test_seed_determinism.py).
    k_nuts = jax.random.fold_in(key0, nuts_fold_int("+".join(leg_names)))
    packed_chains, energies, num_steps_all, per_chain_div, ll_chains = [], [], [], [], []
    nuisance_chains, names = [], None
    for cid in range(int(n_chains)):
        chain_key = jax.random.fold_in(k_nuts, int(cid))
        samples, n_div, extra = _run_nuts_legb(
            ctx, ctx.legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
            seed=chain_key, target_accept=0.9, dense_mass=True,
            max_tree_depth=max_tree_depth, init_strategy=init_to_sample, return_extra=True)
        nuisance_chains.append({k: np.asarray(samples[k]) for k in _nuisance_export_keys(samples)})
        draws = _draws_matrix(samples, kept_global)            # per-leg alpha columns, legs order
        packed_chains.append(draws)
        if names is None:
            names = _packed_names_for(samples, kept_global)
        energies.append(extra["energy"]); num_steps_all.append(extra["num_steps"])
        per_chain_div.append(int(n_div))
        ll_chains.append(np.asarray(_joint_loglik_chain(ctx, core_per_leg, samples)))
        if verbose:
            print(f"  [chain {cid}] legs={'+'.join(leg_names)} draws={draws.shape[0]} "
                  f"div={n_div} E-BFMI={_ebfmi1(extra['energy']):.2f}")

    packed = np.stack(packed_chains, axis=0)
    battery = convergence_battery(
        packed, names, energy=np.stack(energies), num_steps=np.stack(num_steps_all),
        max_tree_depth=max_tree_depth, n_div=int(sum(per_chain_div)))
    nuisance_bounds = dict(f=(float(ctx.metal_fnode_lo), float(ctx.metal_fnode_hi)),
                           k=(float(ctx.metal_knode_lo), float(ctx.metal_knode_hi)))
    return dict(packed=packed, names=names, battery=battery, per_chain_div=per_chain_div,
                members=members, leg_name="+".join(l.name for l in ctx.legs),
                n_real_rows=n_real, ll_chains=ll_chains, kept_global=kept_global,
                nuisance_chains=nuisance_chains, nuisance_bounds=nuisance_bounds,
                joint_stamp=stamp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", default="DESI,KS",
                    help="comma-separated joint legs (DESI,KS[,eBOSS]); survey key == leg name")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--per-leg-zslope", action="store_true",
                    help="per-leg HCD z-slope sites (C1 plumbing; default SHARED slopes). "
                         "NOTE: per-leg slopes decouple the legs' power-law z-evolution; "
                         "they do NOT model the Ho+2025 non-monotonic KS selection "
                         "z-profile — that physics is owned by the KS-challenge "
                         "S-criterion machinery.")
    ap.add_argument("--single-member", action="store_true",
                    help="final_prod_seed0 only (cheap de-risk; NOT the production ensemble)")
    ap.add_argument("--ensemble-glob", default=None,
                    help="DANGER (diagnostic ONLY): override the manifest-pinned production "
                         "ensemble with a checkpoint-prefix glob. Prominent warning + meta "
                         "stamped ensemble_pinned=False; NEVER for a production fit.")
    ap.add_argument("--blind-lock", default=f"{REPO}/blind.lock")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-blind", dest="blind", action="store_false",
                    help="DANGER: export UNBLINDED. Only after freeze + the authorized unblind.")
    ap.add_argument("--allow-env-data-flags", action="store_true",
                    help="DANGER: permit the env data-selection flags (deliberate "
                         "non-baseline arm ONLY; F2 tripwire).")
    ap.set_defaults(blind=True)
    a = ap.parse_args()

    # F2 ENV DATA-FLAG TRIPWIRE (same contract as run_real_fit): refuse at entry unless the
    # override is explicit.
    from hcd_analysis.emulator import data_likelihood as _DLF
    _DLF.assert_env_data_flags_unset("run_joint_fit", allow=a.allow_env_data_flags)

    leg_names = [s.strip() for s in a.legs.split(",") if s.strip()]
    if a.blind and not os.path.exists(a.blind_lock):
        raise SystemExit(f"blind.lock not found at {a.blind_lock}; joint fits stay "
                         f"parameter-blind until the authorized unblind (--no-blind).")
    offset = (BL.offset_from_lock(a.blind_lock) if a.blind
              else {p: 0.0 for p in BL.BLIND_PARAMS})

    private = "DESI" in leg_names                     # DESI results privacy (memory)
    out_dir = a.out_dir or (PRIVATE_DIR if private else PUBLIC_DIR)
    root = "joint_" + "_".join(n.lower() for n in leg_names)
    print(f"=== JOINT REAL-DATA fit legs={'+'.join(leg_names)} blind={a.blind} "
          f"private={private} out={out_dir} per_leg_zslope={a.per_leg_zslope} ===")

    result = run_joint_fit(
        leg_names, n_chains=a.n_chains, n_warmup=a.n_warmup, n_samples=a.n_samples,
        max_tree_depth=a.max_tree_depth, seed=a.seed, per_leg_zslope=a.per_leg_zslope,
        single_member=a.single_member, ensemble_glob=a.ensemble_glob)

    bat = result["battery"]
    print(f"--- sampler health (UNBLINDED) legs={'+'.join(leg_names)} ---")
    print(f"    R-hat max {bat['rhat_max']:.4f}  div {bat['n_divergent']} "
          f"per-chain={result['per_chain_div']}  E-BFMI min {bat['ebfmi_min']:.3f}")

    chain_files, rec = export_getdist(
        result, out_dir, root, offset=offset, blind=a.blind, survey="+".join(leg_names),
        meta=dict(blind_lock=os.path.abspath(a.blind_lock) if a.blind else None,
                  seed=a.seed, joint_legs=leg_names, per_leg_zslope=a.per_leg_zslope,
                  # freeze decision 6: pinned-ensemble self-declaration (see run_real_fit).
                  ensemble_pinned=(a.ensemble_glob is None and not a.single_member),
                  ensemble_glob=a.ensemble_glob,
                  seed_derivation="fold_in(PRNGKey(seed), crc32('+'.join(legs)) & 0x7fffffff)"
                                  " then fold_in(chain_id)",
                  alpha_mode="per_leg"))
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{root}.joint_stamp.json", "w") as f:
        json.dump(result["joint_stamp"], f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"=== wrote {len(chain_files)} cobaya/GetDist chains -> {out_dir}/{root}.*.txt "
          f"(+ .paramnames .yaml .health.json .joint_stamp.json) | BLINDED={a.blind} ===")
    if private:
        print("    NOTE: DESI cosmology is PRIVATE — results_local/ is gitignored. Do NOT commit.")


if __name__ == "__main__":
    main()
