#!/usr/bin/env python3
"""REAL-DATA BLIND production fit — the actual Lyα-P1D cosmology measurement.

Fits the PRODUCTION likelihood (``closure_legb.build_legb_ctx`` + the N=5 ensemble forward) to a
survey's REAL P1D measurement (``leg.P_data`` as loaded by ``data_likelihood.load_*_leg`` — this
is the ACTUAL data, NOT a mock: there is NO ``make_*_mock`` call anywhere in this driver). Runs
4 dispersed NUTS chains, writes GetDist/cobaya chains, and applies the PARAMETER-BLIND offset to
the exported A_p / n_s so the default artifact is BLIND. Sampler health (R̂, divergences, ESS) is
printed and stored UNBLINDED — only the A_p / n_s VALUES are hidden.

PRODUCTION BASELINE (the locked decisions; mirrors scripts/run_prod_sbc_shard.py):
  - 2-param PRIYA τ₀ (amplitude × slope, uniform priors);
  - lit-pinned per-class HCD incidence (hierarchical_hcd=False — the referee baseline). The LLS
    forward z-slope is the litWLS γ_LLS=2.127 (PI re-determination 2026-06-17): build_legb_ctx(survey=…)
    plumbs ctx.zslope_mu=(2.127, sim_subDLA, sim_DLA) on this REAL-FIT path (survey != None), so the
    LLS center tracks the literature dN/dX(z) — NOT the sim incidence slope 2.465 (closure/SBC only).
    σ_LLS = 0.15 (1× lit measurement error, PI WIDTH RULE; 2× cosmic-variance hedge available);
  - emucoh (off-diagonal-only) + cross-class C_emu;
  - MF P1D correction + the LF→HR / n_s-edge C_emu floor;
  - eBOSS+DESI metals: a SHARED a_SiIII nuisance (sample_metals=True when the survey is metal-bearing);
  - the N=5 production ensemble forward (ensemble_ckpts = final_prod_seed*).

SURVEYS (one --survey per invocation; the survey selects WHICH leg(s) carry the real data):
  eboss : eBOSS DR14 ONLY (low-k; metals_on + a_SiIII; committable).
  ks    : KODIAQ-SQUAD ONLY (high-res; no metals; committable).
  desi  : DESI DR1 ONLY (the headline; metals_on + a_SiIII; PRIVATE → results_local/).

WHY one leg per survey (a deliberate, review-flag-worthy choice): each public survey is reported
as an INDEPENDENT measurement (the cross-survey joint is a separate, later product). build_legb_ctx
always assembles DESI+KS(+eBOSS); we then RESTRICT ctx.legs to the requested survey's leg so the
likelihood factor sums over that leg ONLY (its real P_data), with that leg's own C_emu slice, MF
floor, DLA-forward fraction, and metal flag — byte-consistent with how the closure exercises a leg.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_real_fit.py --survey {eboss,ks,desi}
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from datetime import datetime, timezone

print = functools.partial(print, flush=True)

import numpy as np

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
import jax

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, _run_nuts_legb, _draws_matrix, _packed_names_for,
    convergence_battery, _ebfmi1)
from hcd_analysis.emulator.inference import PARAM_NAMES
from hcd_analysis.emulator.seeding import SEED_DERIVATION, nuts_fold_int
from hcd_analysis.emulator import blinding as BL
from hcd_analysis.emulator.data import PARAM_LIMITS
from numpyro.infer import init_to_sample

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"

# GetDist LaTeX labels (extends scripts/legb_chains_to_cobaya.LABELS with the nuisance latents).
LABELS = {
    "ns": "n_s", "Ap": "A_p", "herei": r"z_{\rm HeII,i}", "heref": r"z_{\rm HeII,f}",
    "alphaq": r"\alpha_q", "hub": "h", "omegamh2": r"\Omega_m h^2",
    "hireionz": r"z_{\rm HI,reion}", "bhfeedback": r"\epsilon_{\rm BH}",
    "alpha_lls": r"\alpha_{\rm LLS}", "alpha_subdla": r"\alpha_{\rm subDLA}",
    "alpha_dla": r"\alpha_{\rm DLA}", "a_SiIII": r"a_{\rm SiIII}",
}

# Survey → (leg name, metals?, the loader kwargs DESI-private routing). The leg's own load_*_leg
# defaults (k/z cuts, C_data, DLA-forward fraction) are AUTHORITATIVE — we do not override them.
SURVEY = {
    "eboss": dict(leg="eBOSS", metals=True, private=False),
    "ks":    dict(leg="KS",    metals=False, private=False),
    "desi":  dict(leg="DESI",  metals=True, private=True),
}

# Output routing (memory desi-results-privacy): DESI real cosmology stays LOCAL + gitignored.
PUBLIC_DIR = f"{REPO}/results/real_fit"                       # eBOSS / KS (committable)
PRIVATE_DIR = f"{REPO}/results_local/desi_production"         # DESI (gitignored)


def _denorm_theta(theta_unit):
    """Unit cube → physical θ9 via the emulator's WIDENED PARAM_LIMITS box (the inverse of
    data.normalize_params; the emulator normalizes with PARAM_LIMITS, so the inverse MUST use
    it). A_p / n_s rows of PARAM_LIMITS == SAMPLING_LIMITS, so the physical A_p / n_s and the
    blinding σ_prior (which uses SAMPLING_LIMITS) live in the SAME physical units."""
    lo = PARAM_LIMITS[:, 0]
    hi = PARAM_LIMITS[:, 1]
    return np.asarray(theta_unit) * (hi - lo) + lo


def _packed_to_physical(draws, names):
    """Map the θ9 block of a packed (L,P) draw matrix from unit cube to PHYSICAL θ; leave τ₀ /
    α / a_SiIII columns unchanged. Returns a COPY with the same column order/names."""
    out = np.array(draws, dtype=float, copy=True)
    th_unit = out[:, :len(PARAM_NAMES)]
    out[:, :len(PARAM_NAMES)] = _denorm_theta(th_unit)
    return out


# Gate-A NORC (2026-07-04): the deployed production forward DROPS the res_corr particle-convergence
# correction (res_corr_on=False), PINS alpha_res (fix_alpha_res=True), and caps KS at k<=0.045 (auto
# in build_legb_ctx on the res_corr_on=False path). Rationale: the 2026-06-16 NUTS result showed the
# anchored+alpha config AMPLIFIES the native n_s bias ~4x. The deployed NORC decision now lives in ONE
# place -- closure_legb.PROD_RES_CORR_ON (the SINGLE reversal knob for the 4-referee panel: flip it to
# True to restore the pre-NORC forward on EVERY deployed path at once). build_real_ctx consumes it via
# CL.prod_norc_forward(). Gated by the panel + the freeze + PI sign-off before any unblind.


def _assert_norc_ks_cap(ctx):
    """Under NORC the KS leg is capped at k<=0.045, EXCEPT when it floats echelle f_res
    (resolution_ready=True), which intentionally lifts the cap to the certified 0.065."""
    _ksleg = [l for l in ctx.legs if l.name == "KS"]
    if not _ksleg or getattr(_ksleg[0], "resolution_ready", False):
        return
    assert float(np.asarray(_ksleg[0].k).max()) <= 0.045 + 1e-9, \
        "NORC KS k_max cap (0.045) not applied"


def build_real_ctx(survey, *, single_member=False, ensemble_glob=None, ks_zlo=None):
    """The PRODUCTION ctx for a real-data fit, then RESTRICTED to the requested survey's leg.

    Baseline flags mirror run_prod_sbc_shard.build (use_xclass, MF + floor, emucoh off-diag-only,
    hierarchical_hcd=False). ``metals_on``/``sample_metals`` follow the survey (eBOSS/DESI bear a
    shared a_SiIII; KS does not). ``with_eboss`` is True only for the eboss survey (so the eBOSS
    leg is assembled at all). We then keep ONLY the requested leg in ctx.legs.

    ``ks_zlo`` (default None → the loader's authoritative z_lo=2.4 baseline) overrides ONLY the KS
    leg's low-z cut via ks_kwargs={"z_lo": ks_zlo}. The PI's z2.4=baseline / z2.8=diagnostic
    comparison (KS-author published cut is the more-conservative z<2.8; see load_ks_leg docstring).
    """
    import glob as _glob
    members = sorted(p[:-4] for p in _glob.glob((ensemble_glob or (PROD_PREFIX + "*")) + ".eqx"))
    if not members:
        raise SystemExit(f"no production ensemble checkpoints at {PROD_PREFIX}*.eqx")
    ens = [members[0]] if single_member else members

    info = SURVEY[survey]
    fc = CL.prod_forward_config(info["leg"])   # certified per-leg data-nuisance forward (SINGLE source == SBC)
    norc = CL.prod_norc_forward()              # the GLOBAL deployed sim-convergence (NORC) decision (SINGLE authority)
    metals = bool(fc["metals"])
    assert metals == bool(info["metals"]), \
        f"SURVEY[{survey!r}].metals={info['metals']} disagrees with prod_forward_config({info['leg']!r})"
    # KS ks_kwargs: merge the certified forward's dict (resolution_float/k_max; None for DESI/eBOSS)
    # with the diagnostic z_lo override -- neither must clobber the other. Empty dict -> None so
    # DESI/eBOSS with no ks_zlo stay byte-identical (ks_kwargs=None, as before Task 1A/1B).
    ks_kw = ({**(fc.get("ks_kwargs") or {}),
              **({"z_lo": float(ks_zlo)} if ks_zlo is not None else {})}) or None
    ctx, d = build_legb_ctx(
        ensemble_ckpts=ens, use_xclass=True,
        with_mf=True, mf_with_floor=True,
        res_corr_on=norc["res_corr_on"],  # NORC: drop res_corr (+ auto-cap KS k<=0.045 in build_legb_ctx)
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=(survey == "eboss"),
        ks_kwargs=ks_kw,                  # threads z_lo into load_ks_leg (default None → z_lo=2.4 baseline)
        metals_on=metals,                 # applies the SiIII/SiII forward term on metals_on legs
        sample_metals=metals,             # samples the metal nuisance (flatlog2node nodes / uniform a_SiIII)
        sample_res=fc["sample_res"],      # option-b f_res float (DESI 0.02 / eBOSS 0.05 / KS 0.15; task #5 DONE)
        f_res_amp_sigma=fc["f_res_amp_sigma"],  # its Normal(0,.) width (DESI 0.02 / eBOSS 0.05 / KS 0.15)
        metal_prior=fc["metal_prior"],    # flatlog2node (Gate-C Model C+) on metal legs; uniform on KS
        survey=info["leg"],               # PER-SURVEY LLS pin: DESI 1.0×/σ0.287 (2026-07-18 width), KS 2.5×/σ0.40 (eBOSS→cosmic-avg)
        hierarchical_hcd=False)           # the referee production baseline (per-class HCD)
    if norc["fix_alpha_res"]:
        ctx = ctx._replace(fix_alpha_res=True)   # NORC: also drop the 2 alpha_res sites (now inert)
    assert ctx.res_corr_on == norc["res_corr_on"] and ctx.fix_alpha_res == norc["fix_alpha_res"], \
        "prod NORC forward not applied"
    if not norc["res_corr_on"]:
        # KS-cap parity assert (referee M2), gated on resolution_ready. NORC-only: on the restore
        # flip (PROD_RES_CORR_ON=True) build_legb_ctx does not auto-cap KS, so the assert would
        # spuriously fire on the always-built proxy KS leg (k_max 0.069) before the leg restriction.
        _assert_norc_ks_cap(ctx)

    # REDUCED-COV tripwire (PI disposition 2026-07-17; consistency-review MINOR-2): the DESI leg
    # must carry the reduced covariance (syst_e_dla_completeness removed, cup1d "red") whenever the
    # DESI_DLA_COV_REDUCE authority says so — a future desi_kwargs default threading
    # dla_cov_reduce=False through a driver would otherwise run the FINAL fit un-reduced silently.
    from hcd_analysis.emulator import data_likelihood as _DL
    for _leg in ctx.legs:
        if _leg.name == "DESI":
            assert bool(_leg.dla_cov_reduced) == bool(_DL.DESI_DLA_COV_REDUCE), \
                "DESI leg dla_cov_reduced disagrees with the DESI_DLA_COV_REDUCE authority"

    # RESTRICT to the requested survey's leg (the real measurement for THIS survey only). The
    # per-leg C_emu / MF-floor / emucoh dicts are keyed by leg name, so dropping other legs leaves
    # this leg's covariance pieces intact; the likelihood factor then sums over this leg ALONE.
    want = info["leg"]
    legs = [leg for leg in ctx.legs if leg.name == want]
    if not legs:
        raise SystemExit(f"survey {survey!r} expects leg {want!r} but ctx.legs="
                         f"{[l.name for l in ctx.legs]}")
    ctx = ctx._replace(legs=legs)

    # SELF-CONSISTENCY (mirror the NORC block): the certified data-nuisance forward must be WIRED, not
    # silently regressed to the build_legb_ctx defaults (sample_res=False / metal_prior='uniform'). Asserted
    # on the single restricted leg so the real-fit forward provably == the SBC-certified forward.
    L = ctx.legs[0]
    assert bool(ctx.sample_res) == fc["sample_res"], "prod f_res float not wired into build_real_ctx"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"], "f_res prior width mismatch"
    assert ctx.metal_prior == fc["metal_prior"], "prod metal model (flatlog2node) not wired"
    assert tuple(ctx.metal_node_z) == (2.2, 4.2), "Gate-C metal_node_z drifted"
    if fc["metals"]:
        assert ctx.sample_metals is True and L.metals_on is True, "metal leg must sample+apply metals"
    else:
        assert not L.metals_on, "KS must be metals-off"
    if ctx.sample_res:                             # f_res is per-INSTRUMENT: single-leg + resolution_ready
        assert getattr(L, "resolution_ready", False), "f_res float on a resolution_ready=False leg"
        assert len({l.name for l in ctx.legs}) == 1, "f_res float requires a single-instrument leg"
    return ctx, d, members


def run_real_fit(survey, *, n_chains=4, n_warmup=250, n_samples=600, max_tree_depth=10,
                 seed=20260614, single_member=False, ks_zlo=None, verbose=True):
    """Multi-chain dispersed NUTS on the REAL leg.P_data (NO mock). Returns
    ``dict(packed_chains, names, battery, per_chain_div, members, leg_name, n_real_rows)``.

    DISPERSED inits (init_to_sample, per-chain fold_in seed) so split-R̂ is a valid convergence
    diagnostic — IDENTICAL to run_legb_convergence's chain seeding (the validated path), but the
    LIKELIHOOD points at the real data (ctx.legs already carry leg.P_data = the measurement; we
    do NOT overwrite it with a mock)."""
    ctx, d, members = build_real_ctx(survey, single_member=single_member, ks_zlo=ks_zlo)
    leg = ctx.legs[0]
    n_real = int(np.isfinite(np.asarray(leg.P_data)).sum())

    # the per-leg DLA core for the REAL forward = the fiducial (mean held-out) core build_legb_ctx
    # already assembled (ctx.dla_core_leg, shape (n_z, K)). For a real fit there is NO sim-truth
    # core; the fiducial cache DLA core is the production DLA-excess template the closure validated
    # against. _data_loglik_legcore wants ONE (K,) core per leg (z-mean — the documented MVP the
    # closure's _mock_core_per_leg also uses: the per-z core variation is tiny vs the P1D, and on
    # KS/eBOSS the forward DLA term is 0 anyway via dla_forward_frac).
    import jax.numpy as jnp
    fid = np.asarray(ctx.dla_core_leg[leg.name])               # (n_z, K) or (K,)
    core_k = jnp.asarray(fid.mean(axis=0) if fid.ndim == 2 else fid)
    core_per_leg = {leg.name: core_k}

    key0 = jax.random.PRNGKey(int(seed))
    # P0 (plan-to-unblind 3A): python hash() is SipHash-salted per process, so the recorded seed
    # would NOT reproduce the chain. nuts_fold_int is crc32, stable across processes/machines;
    # the derivation is recorded in the export meta. Shared with run_joint_fit so the two drivers
    # cannot drift. NOTE: this changes the chain stream vs the June eBOSS chains, which are
    # already marked chain_of_record: NO / superseded-forward.
    k_nuts = jax.random.fold_in(key0, nuts_fold_int(survey))

    packed_chains, energies, num_steps_all, per_chain_div, ll_chains = [], [], [], [], []
    nuisance_chains = []                                        # Fix 2: raw f_res / metal-node draws
    names = None
    for cid in range(int(n_chains)):
        chain_key = jax.random.fold_in(k_nuts, int(cid))
        samples, n_div, extra = _run_nuts_legb(
            ctx, ctx.legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
            seed=chain_key, target_accept=0.9, dense_mass=True,
            max_tree_depth=max_tree_depth, init_strategy=init_to_sample, return_extra=True)
        # Fix 2: keep the UNBLINDED data-nuisance posteriors (f_res + Model C+ metal f/k nodes) so we
        # can see if they RAIL at the real fit. These are NOT the blinded A_p/n_s (safe to export).
        nuisance_chains.append({k: np.asarray(samples[k]) for k in _nuisance_export_keys(samples)})
        # all real-leg rows are kept (no dropped-z in a real fit): kept_global = z_global mask of
        # the leg's z. We use the leg's z directly via _draws_matrix's kept_global contract: the
        # tau0_vec is on z_global; keep the z this leg actually has data at.
        zg = np.asarray(ctx.z_global)
        kept_global = np.array([np.any(np.isclose(leg.z, zz, atol=1e-3)) for zz in zg])
        draws = _draws_matrix(samples, kept_global)            # (L, P)
        packed_chains.append(draws)
        if names is None:
            names = _packed_names_for(samples, kept_global)
        energies.append(extra["energy"]); num_steps_all.append(extra["num_steps"])
        per_chain_div.append(int(n_div))
        ll_chains.append(np.asarray(_loglik_chain(ctx, core_per_leg, samples, kept_global)))
        if verbose:
            print(f"  [chain {cid}] survey={survey} draws={draws.shape[0]} div={n_div} "
                  f"E-BFMI={_ebfmi1(extra['energy']):.2f}")

    packed = np.stack(packed_chains, axis=0)                    # (C, N, P)
    battery = convergence_battery(
        packed, names, energy=np.stack(energies), num_steps=np.stack(num_steps_all),
        max_tree_depth=max_tree_depth, n_div=int(sum(per_chain_div)))
    # Fix 2: the LogUniform node brackets (for the rail-fraction summary) come straight off the ctx
    # (the SAME static values _metal_2node_sites samples within) so the artifact self-documents them.
    nuisance_bounds = dict(f=(float(ctx.metal_fnode_lo), float(ctx.metal_fnode_hi)),
                           k=(float(ctx.metal_knode_lo), float(ctx.metal_knode_hi)))
    return dict(packed=packed, names=names, battery=battery, per_chain_div=per_chain_div,
                members=members, leg_name=leg.name, n_real_rows=n_real,
                ll_chains=ll_chains, kept_global=kept_global,
                nuisance_chains=nuisance_chains, nuisance_bounds=nuisance_bounds)


# Nuisance-site prefixes for the Model C+ per-leg, per-ion metal f/k nodes. Kept in sync with
# closure_legb._metal_2node_sites / _legb_model (the deployed forward we MIRROR).
_METAL_NODE_PREFIXES = ("f_SiIII_", "f_SiII_", "k_SiIII_", "k_SiII_")


def _nuisance_export_keys(samples):
    """The data-nuisance posterior sites to export (Fix 2): the option-b f_res sites + every Model C+
    metal f/k node site present in ``samples``. NOT the blinded cosmology (theta_unit) nor the packed
    alpha/tau0 columns -- just the floated data-nuisance latents whose railing we want to see at the
    real fit. Empty when none were sampled (KS uniform) -> no artifact written."""
    keys = [k for k in ("f_res_amp", "f_res_slope") if k in samples]
    keys += [k for k in samples if k.startswith(_METAL_NODE_PREFIXES)]
    return keys


def _reconstruct_nuisance(ctx, draw):
    """PURE reconstruction of ``(metal_nodes, b_res_global, alpha_res, a_siiii, a_siii)`` for ONE
    draw -- the deployed data-nuisance forward terms ``_legb_model`` threads into
    ``_data_loglik_legcore`` (closure_legb.py:1819-1859). ``draw`` is a dict {site_name: per-draw
    value} (a vmap slice of ``samples``, or synthetic scalars in tests). MIRRORS the model's STATIC
    branches EXACTLY so the re-scored export loglik == the true model loglik under the Stage-1 config
    (flatlog2node + sample_res + fix_alpha_res). Under the uniform / no-f_res / not-fixed path it
    yields ``(None, None, None, a_siiii, 0.0)`` -- the byte-identical pre-fix a_siiii-only call.

      metals (closure_legb:1819-1830): metal_prior=='flatlog2node' -> metal_nodes =
        ``{leg.name: (f3, f2, k3, k2)}`` per metals_on leg IN ctx.legs ORDER (f2/k2 = the SiII nodes
        ONLY on ctx.metal_siII_legs), each node ``jnp.stack([z0, z1])`` of the sampled sites (byte-
        identical to _metal_2node_sites' single-trace stack); a_siiii=a_siii=0. Else metal_nodes=None
        and a_siiii/a_siii come from the scalar a_SiIII/a_SiII draws (0.0 if absent).
      f_res (closure_legb:1848-1855): sample_res and 'f_res_amp' present -> b_res_global =
        ``_bres_of_z(z_global, f_res_amp, f_res_slope)``; else None.
      alpha_res (closure_legb:1839-1843): fix_alpha_res -> the pinned no-op ``(1.0, 0.0)``; elif the
        'alpha_res' site is present -> ``(alpha_res, alpha_res_slope)``; else None (pre-alpha_res
        golden, which is byte-identical to (1.0,0.0): _mf_corr_on_cache multiplies log_rc by 1.0)."""
    import jax.numpy as jnp
    if getattr(ctx, "metal_prior", "uniform") == "flatlog2node":
        a_siiii = a_siii = 0.0                       # flatlog2node: scalar metals unused (mirror _legb_model)
        metal_nodes = {}
        # _metal_2node_sites emits {} (scalar path) when NOT sampling metals; mirror that guard so a
        # flatlog2node + sample_metals=False ctx does not KeyError on the (absent) node sites. Defense-
        # in-depth: production pairs flatlog2node with sample_metals=True (asserted in build_real_ctx).
        if getattr(ctx, "sample_metals", False):
            siII_legs = tuple(getattr(ctx, "metal_siII_legs", ("DESI",)))
            for leg in ctx.legs:                    # FIXED order == _metal_2node_sites
                if not getattr(leg, "metals_on", False):
                    continue
                f3 = jnp.stack([draw[f"f_SiIII_{leg.name}_z0"], draw[f"f_SiIII_{leg.name}_z1"]])
                k3 = jnp.stack([draw[f"k_SiIII_{leg.name}_z0"], draw[f"k_SiIII_{leg.name}_z1"]])
                f2 = k2 = None
                if leg.name in siII_legs:
                    f2 = jnp.stack([draw[f"f_SiII_{leg.name}_z0"], draw[f"f_SiII_{leg.name}_z1"]])
                    k2 = jnp.stack([draw[f"k_SiII_{leg.name}_z0"], draw[f"k_SiII_{leg.name}_z1"]])
                metal_nodes[leg.name] = (f3, f2, k3, k2)
    else:
        metal_nodes = None
        a_siiii = draw["a_SiIII"] if "a_SiIII" in draw else 0.0
        a_siii = draw["a_SiII"] if "a_SiII" in draw else 0.0
    if getattr(ctx, "sample_res", False) and "f_res_amp" in draw:
        b_res_global = CL._bres_of_z(jnp.asarray(ctx.z_global), draw["f_res_amp"], draw["f_res_slope"])
    else:
        b_res_global = None
    if getattr(ctx, "fix_alpha_res", False):
        alpha_res = (1.0, 0.0)                       # NORC pinned no-op (mirrors _legb_model)
    elif "alpha_res" in draw:
        alpha_res = (draw["alpha_res"], draw["alpha_res_slope"])
    else:
        alpha_res = None                             # pre-alpha_res golden (byte-identical)
    return metal_nodes, b_res_global, alpha_res, a_siiii, a_siii


def _loglik_chain(ctx, core_per_leg, samples, kept_global):
    """Per-draw log-likelihood of the REAL data (for the minusloglike column; -lnL, data term, no
    prior -- matches the export header). MIRRORS the deployed
    ``_legb_model`` call (closure_legb.py:1856-1859) EXACTLY: reconstructs the data-nuisance forward
    terms (metal_nodes / b_res / alpha_res / a_siiii / a_siii) per draw via ``_reconstruct_nuisance``,
    then threads ALL of them into ``_data_loglik_legcore``. Under the Stage-1 config the metal power
    lives in the per-leg 2-node metal sites and the f_res term in f_res_amp/slope, so the pre-fix
    a_siiii-only re-score OMITTED them and was NOT the true model loglik. vmapped over draws; the
    per-draw metal-node dict is rebuilt INSIDE the vmap from the per-draw scalar sites (byte-identical
    to the model's single-trace ``jnp.stack``), so no python draw-loop and no perf cost."""
    from hcd_analysis.emulator.closure_legb import _data_loglik_legcore
    import jax.numpy as jnp
    theta = jnp.asarray(samples["theta_unit"])                 # (L,9)
    tau0 = jnp.asarray(samples["tau0_vec"])                    # (L, nZg)
    a_z = jnp.asarray(samples["alpha_hcd_z"])                  # (L, nZg, 3) z-resolved
    # The per-draw nuisance sites present in `samples` (metal f/k nodes, f_res, scalar a_SiIII/a_SiII,
    # and the sampled alpha_res if NOT pinned). Passed as a dict pytree so vmap slices each leaf to a
    # per-draw scalar we reconstruct in `one`. Empty dict (KS uniform) is a no-op under vmap (the
    # batch axis comes from theta/tau0/a_z).
    nkeys = list(_nuisance_export_keys(samples))
    if not getattr(ctx, "fix_alpha_res", False):
        nkeys += [k for k in ("alpha_res", "alpha_res_slope") if k in samples]
    nkeys += [k for k in ("a_SiIII", "a_SiII") if k in samples]
    nuis = {k: jnp.asarray(samples[k]) for k in nkeys}

    def one(th, t0, al, draw):
        metal_nodes, b_res, alpha_res, a_siiii, a_siii = _reconstruct_nuisance(ctx, draw)
        return _data_loglik_legcore(
            ctx, th, t0, al, ctx.legs, core_per_leg, a_siiii=a_siiii, a_siii=a_siii,
            metal_nodes=metal_nodes, alpha_res=alpha_res, b_res_global=b_res,
            require_zresolved=True)
    return jax.vmap(one)(theta, tau0, a_z, nuis)


# --------------------------------------------------------------------------------------------- #
#  Data-nuisance posterior export (Fix 2) -- UNBLINDED (these are NOT the A_p / n_s cosmology).
# --------------------------------------------------------------------------------------------- #
# A draw is "near" a LogUniform bound if it sits within this fraction of the log10(hi/lo) span of it
# (a simple, prior-scale railing flag for the metal f/k nodes; 2% ~ the ceiling-check band).
RAIL_LOG_FRAC = 0.02


def _rail_fracs(vals, bounds):
    """(frac_near_lo, frac_near_hi) of ``vals`` against a LogUniform ``(lo, hi)`` prior, measured in
    log10 space; ``(None, None)`` when ``bounds`` is None (a non-LogUniform site, e.g. the Normal
    f_res)."""
    if bounds is None:
        return None, None
    lo, hi = float(bounds[0]), float(bounds[1])
    v = np.clip(np.asarray(vals, float), lo, hi)
    logv, logL, logH = np.log10(v), np.log10(lo), np.log10(hi)
    span = logH - logL
    near_lo = float(np.mean((logv - logL) <= RAIL_LOG_FRAC * span))
    near_hi = float(np.mean((logH - logv) <= RAIL_LOG_FRAC * span))
    return near_lo, near_hi


def _bounds_for_site(name, nb):
    """The LogUniform ``(lo, hi)`` bracket for a nuisance site name (f_-node -> f bracket, k_-node ->
    k bracket); None for f_res_amp/slope (a Normal prior, no hard rail)."""
    if name.startswith(("f_SiIII_", "f_SiII_")):
        return nb["f"]
    if name.startswith(("k_SiIII_", "k_SiII_")):
        return nb["k"]
    return None


def _export_nuisance(out_dir, root, result, survey):
    """Fix 2: write the data-nuisance posteriors (option-b f_res + Model C+ metal f/k nodes) NEXT TO
    the chains so railing is visible at the real fit. These are NUISANCE latents, NOT the blinded
    A_p / n_s, so they are exported UNBLINDED (like the health json). Writes:
      ``<root>.nuisance.npz``  -- per-chain raw draws, one (C, N) array per site (round-trips exact);
      ``<root>.nuisance.json`` -- per-site rail summary (mean/std/quantiles + frac_near_lo/hi vs the
                                  LogUniform bounds; null rail for the Normal f_res).
    No-op (nothing written, returns None) when no nuisance sites were sampled (KS uniform)."""
    chains = result.get("nuisance_chains") or []
    sites = sorted({k for ch in chains for k in ch})
    if not sites:
        return None
    nb = result.get("nuisance_bounds", dict(f=(0.003, 0.03), k=(1e-3, 0.1)))
    npz, summary = {}, {}
    for name in sites:
        per_chain = np.stack([np.asarray(ch[name], float) for ch in chains if name in ch])  # (C,N)
        npz[name] = per_chain
        flat = per_chain.reshape(-1)
        bnds = _bounds_for_site(name, nb)
        near_lo, near_hi = _rail_fracs(flat, bnds)
        q05, q50, q95 = (float(x) for x in np.quantile(flat, [0.05, 0.5, 0.95]))
        summary[name] = dict(
            mean=float(np.mean(flat)), std=float(np.std(flat)), q05=q05, q50=q50, q95=q95,
            n=int(flat.size), frac_near_lo=near_lo, frac_near_hi=near_hi,
            bounds=(list(bnds) if bnds is not None else None))
    np.savez(f"{out_dir}/{root}.nuisance.npz", **npz)
    rec = dict(survey=survey, leg=result.get("leg_name"), root=root, n_chains=len(chains),
               sites=summary, bounds=dict(f=list(nb["f"]), k=list(nb["k"])),
               rail_log_frac=RAIL_LOG_FRAC,
               note="UNBLINDED data-nuisance posteriors (f_res + Model C+ metal nodes); NOT A_p/n_s")
    with open(f"{out_dir}/{root}.nuisance.json", "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    return rec


# --------------------------------------------------------------------------------------------- #
#  GetDist / cobaya export (BLINDED on A_p / n_s by default).
# --------------------------------------------------------------------------------------------- #
def export_getdist(result, out_dir, root, *, offset, blind=True, survey="", meta=None):
    """Write per-chain GetDist/cobaya chains: <root>.{c}.txt + <root>.paramnames + <root>.yaml.

    Each row: ``weight  minusloglike  <physical θ9...>  <τ₀...>  <α...>  [a_SiIII]``.
    The θ9 block is in PHYSICAL units; if ``blind`` the A_p / n_s columns are SHIFTED by the
    hidden offset (θ_shown = θ_phys + δ) so the default artifact is BLIND. Sampler-health YAML is
    written UNBLINDED (R̂/divergences/ESS are not the cosmology values)."""
    os.makedirs(out_dir, exist_ok=True)
    names = list(result["names"])
    labels = [LABELS.get(n, (rf"\tau_0[{n.split('z')[-1]}]" if n.startswith("tau0_") else n))
              for n in names]
    packed = result["packed"]                                   # (C, N, P)
    C, N, P = packed.shape
    chain_files = []
    for c in range(C):
        draws = _packed_to_physical(packed[c], names)           # θ9 → physical
        if blind:
            draws = BL.apply_blind(draws, offset, columns=names)  # θ_shown on A_p/n_s ONLY
        minusloglike = -np.asarray(result["ll_chains"][c])[:N]   # -ln L (data term); see header note
        weight = np.ones(N)
        table = np.column_stack([weight, minusloglike, draws])
        assert np.isfinite(table).all(), f"non-finite value in chain {c} export"
        fn = f"{out_dir}/{root}.{c + 1}.txt"
        np.savetxt(fn, table, fmt=["%.8g"] * table.shape[1],
                   header="weight  minusloglike  " + "  ".join(names))
        chain_files.append(fn)

    with open(f"{out_dir}/{root}.paramnames", "w") as f:
        for n, lab in zip(names, labels):
            f.write(f"{n}\t{lab}\n")

    bat = result["battery"]
    rec = dict(
        survey=survey, leg=result["leg_name"], n_real_rows=int(result["n_real_rows"]),
        n_chains=int(C), n_draws=int(N), n_params=int(P),
        blinded=bool(blind), blind_params=list(BL.BLIND_PARAMS),
        members=[os.path.basename(m) for m in result["members"]],
        rhat_max=float(bat["rhat_max"]), ess_bulk_min=float(bat["ess_bulk_min"]),
        ess_tail_min=float(bat["ess_tail_min"]), ebfmi_min=float(bat["ebfmi_min"]),
        n_divergent=int(bat["n_divergent"]), per_chain_div=list(result["per_chain_div"]),
        treedepth_sat_frac=float(bat["treedepth_sat_frac"]),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    if meta:
        rec.update(meta)
    with open(f"{out_dir}/{root}.yaml", "w") as f:
        f.write("# cobaya/GetDist metadata for a REAL-DATA blind NUTS fit\n")
        f.write(f"# survey: {survey}   leg: {result['leg_name']}   BLINDED(A_p,n_s): {blind}\n")
        f.write("sampler:\n  numpyro_nuts: {dense_mass: true, target_accept: 0.9}\n")
        f.write("params:\n")
        for n, lab in zip(names, labels):
            f.write(f"  {n}: {{latex: '{lab}'}}\n")
        f.write("health:\n")
        for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "ebfmi_min",
                  "n_divergent", "treedepth_sat_frac"):
            f.write(f"  {k}: {rec[k]}\n")
    with open(f"{out_dir}/{root}.health.json", "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    # Fix 2: the UNBLINDED data-nuisance posterior companion (f_res + metal nodes). Additive-only:
    # it does NOT touch the GetDist chain columns / the A_p/n_s blinding above, and is a no-op when
    # no nuisance sites were sampled (KS uniform).
    _export_nuisance(out_dir, root, result, survey)
    return chain_files, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", required=True, choices=("eboss", "ks", "desi"))
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--blind-lock", default=f"{REPO}/blind.lock",
                    help="path to the committed blind.lock (the SEED for the A_p/n_s offset)")
    ap.add_argument("--out-dir", default=None,
                    help="override the default routing (DESI->results_local, else results/real_fit)")
    ap.add_argument("--no-blind", dest="blind", action="store_false",
                    help="DANGER: export UNBLINDED. Only after freeze + the authorized unblind.")
    ap.add_argument("--single-member", action="store_true",
                    help="final_prod_seed0 only (cheap de-risk; NOT the production ensemble)")
    ap.add_argument("--ks-zlo", type=float, default=2.4,
                    help="KS leg low-z cut (only for --survey ks). 2.4 = PI BASELINE (default); "
                         "2.8 = the KS-author published conservative DIAGNOSTIC. Routes a "
                         "non-baseline value to a distinct root (real_ks_z<NN>) so it never "
                         "clobbers the z2.4 baseline.")
    ap.add_argument("--allow-env-data-flags", action="store_true",
                    help="DANGER: permit the env data-selection flags (HCD_DESI_SNR3 / "
                         "HCD_CV_FLOOR / HCD_CV_FLOOR_RANK1) to be set at driver entry — a "
                         "deliberate non-baseline arm ONLY. Default: refuse to start (F2 "
                         "tripwire; the resolved values are stamped on the DataLeg).")
    ap.set_defaults(blind=True)
    a = ap.parse_args()

    # ENV DATA-FLAG TRIPWIRE (adversarial backfill F2, 2026-07-19; companion to the
    # dla_cov_reduced authority assert in build_real_ctx): a stray exported HCD_DESI_SNR3 /
    # HCD_CV_FLOOR(_RANK1) would silently swap the DESI measurement / inflate the covariance
    # under the REAL fit. Refuse at entry unless the override is explicit.
    from hcd_analysis.emulator import data_likelihood as _DLF
    _DLF.assert_env_data_flags_unset("run_real_fit", allow=a.allow_env_data_flags)

    if a.blind and not os.path.exists(a.blind_lock):
        raise SystemExit(
            f"blind.lock not found at {a.blind_lock}. Create it FIRST (frozen seed) e.g.:\n"
            f"  python3 -c \"from hcd_analysis.emulator import blinding as B; "
            f"B.write_blind_lock('{a.blind_lock}', 'hcd_priya_real_fit_v1')\"\n"
            f"or pass --no-blind only post-freeze/unblind.")
    offset = BL.offset_from_lock(a.blind_lock) if a.blind else {p: 0.0 for p in BL.BLIND_PARAMS}

    info = SURVEY[a.survey]
    out_dir = a.out_dir or (PRIVATE_DIR if info["private"] else PUBLIC_DIR)
    root = f"real_{a.survey}"
    # KS z_lo routing: only KS honors --ks-zlo. A NON-baseline z_lo (≠2.4) is a DIAGNOSTIC and
    # routes to a distinct root (real_ks_z28 for z_lo=2.8) so it never clobbers the z2.4 baseline.
    ks_zlo = a.ks_zlo if a.survey == "ks" else None
    if a.survey == "ks" and abs(a.ks_zlo - 2.4) > 1e-6:
        root = f"real_ks_z{int(round(a.ks_zlo * 10)):02d}"   # e.g. z_lo=2.8 -> real_ks_z28

    print(f"=== REAL-DATA fit  survey={a.survey}  leg={info['leg']}  "
          f"blind={a.blind}  private={info['private']}  out={out_dir}"
          f"{f'  KS z_lo={a.ks_zlo}  root={root}' if a.survey == 'ks' else ''} ===")
    print(f"    NUTS: chains={a.n_chains} warmup={a.n_warmup} samples={a.n_samples} "
          f"mtd={a.max_tree_depth} dense-mass=True  (ensemble{'=single' if a.single_member else '=N'})")

    result = run_real_fit(
        a.survey, n_chains=a.n_chains, n_warmup=a.n_warmup, n_samples=a.n_samples,
        max_tree_depth=a.max_tree_depth, seed=a.seed, single_member=a.single_member,
        ks_zlo=ks_zlo)

    bat = result["battery"]
    print(f"--- sampler health (UNBLINDED) survey={a.survey} ---")
    print(f"    R-hat max      : {bat['rhat_max']:.4f}  (target < 1.01)")
    print(f"    ESS bulk min   : {bat['ess_bulk_min']:.1f}")
    print(f"    ESS tail min   : {bat['ess_tail_min']:.1f}")
    print(f"    E-BFMI min     : {bat['ebfmi_min']:.3f}  (healthy >= 0.3)")
    print(f"    divergences    : {bat['n_divergent']}  per-chain={result['per_chain_div']}")
    print(f"    treedepth sat  : {bat['treedepth_sat_frac']:.3f}")
    print(f"    real data rows : {result['n_real_rows']}  (leg {result['leg_name']})")

    chain_files, rec = export_getdist(
        result, out_dir, root, offset=offset, blind=a.blind, survey=a.survey,
        meta=dict(blind_lock=os.path.abspath(a.blind_lock) if a.blind else None,
                  seed=a.seed, ks_zlo=(a.ks_zlo if a.survey == "ks" else None),
                  # P0: record the DERIVATION alongside the seed, and the resolved integer, so a
                  # reader can re-derive the chain stream from the export alone (mirrors
                  # run_joint_fit). A seed without its derivation is ambiguous.
                  seed_derivation=SEED_DERIVATION,
                  seed_fold_label=a.survey,
                  seed_fold_int=int(nuts_fold_int(a.survey))))
    print(f"=== wrote {len(chain_files)} chains -> {out_dir}/{root}.*.txt "
          f"(+ .paramnames .yaml .health.json) | A_p/n_s BLINDED={a.blind} ===")
    if info["private"]:
        print("    NOTE: DESI cosmology is PRIVATE — results_local/ is gitignored. Do NOT commit.")


if __name__ == "__main__":
    main()
