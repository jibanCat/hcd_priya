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
    metals = bool(info["metals"])
    ks_kw = {"z_lo": float(ks_zlo)} if ks_zlo is not None else None  # KS low-z cut override (diagnostic)
    ctx, d = build_legb_ctx(
        ensemble_ckpts=ens, use_xclass=True,
        with_mf=True, mf_with_floor=True,
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=(survey == "eboss"),
        ks_kwargs=ks_kw,                  # threads z_lo into load_ks_leg (default None → z_lo=2.4 baseline)
        metals_on=metals,                 # applies the SiIII/SiII forward term on metals_on legs
        sample_metals=metals,             # samples the shared a_SiIII nuisance (Uniform[0, a_max])
        survey=info["leg"],               # PER-SURVEY LLS pin: DESI 1.0×/σ0.30, KS 2.5×/σ0.40 (eBOSS→cosmic-avg)
        hierarchical_hcd=False)           # the referee production baseline (per-class HCD)

    # RESTRICT to the requested survey's leg (the real measurement for THIS survey only). The
    # per-leg C_emu / MF-floor / emucoh dicts are keyed by leg name, so dropping other legs leaves
    # this leg's covariance pieces intact; the likelihood factor then sums over this leg ALONE.
    want = info["leg"]
    legs = [leg for leg in ctx.legs if leg.name == want]
    if not legs:
        raise SystemExit(f"survey {survey!r} expects leg {want!r} but ctx.legs="
                         f"{[l.name for l in ctx.legs]}")
    ctx = ctx._replace(legs=legs)
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
    k_nuts = jax.random.fold_in(key0, hash(survey) & 0x7fffffff)

    packed_chains, energies, num_steps_all, per_chain_div, ll_chains = [], [], [], [], []
    names = None
    for cid in range(int(n_chains)):
        chain_key = jax.random.fold_in(k_nuts, int(cid))
        samples, n_div, extra = _run_nuts_legb(
            ctx, ctx.legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
            seed=chain_key, target_accept=0.9, dense_mass=True,
            max_tree_depth=max_tree_depth, init_strategy=init_to_sample, return_extra=True)
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
    return dict(packed=packed, names=names, battery=battery, per_chain_div=per_chain_div,
                members=members, leg_name=leg.name, n_real_rows=n_real,
                ll_chains=ll_chains, kept_global=kept_global)


def _loglik_chain(ctx, core_per_leg, samples, kept_global):
    """Per-draw log-likelihood of the REAL data (for the minuslogpost column). Reuses the same
    per-leg-core loglik the model uses; vmapped over draws."""
    from hcd_analysis.emulator.closure_legb import _data_loglik_legcore
    import jax.numpy as jnp
    zg = np.asarray(ctx.z_global)
    theta = jnp.asarray(samples["theta_unit"])                 # (L,9)
    tau0 = jnp.asarray(samples["tau0_vec"])                    # (L, nZg)
    a_z = jnp.asarray(samples["alpha_hcd_z"])                  # (L, nZg, 3)
    a_si = (jnp.asarray(samples["a_SiIII"]) if "a_SiIII" in samples
            else jnp.zeros(theta.shape[0]))

    def one(th, t0, al, asi):
        return _data_loglik_legcore(ctx, th, t0, al, ctx.legs, core_per_leg, a_siiii=asi)
    return jax.vmap(one)(theta, tau0, a_z, a_si)


# --------------------------------------------------------------------------------------------- #
#  GetDist / cobaya export (BLINDED on A_p / n_s by default).
# --------------------------------------------------------------------------------------------- #
def export_getdist(result, out_dir, root, *, offset, blind=True, survey="", meta=None):
    """Write per-chain GetDist/cobaya chains: <root>.{c}.txt + <root>.paramnames + <root>.yaml.

    Each row: ``weight  minuslogpost  <physical θ9...>  <τ₀...>  <α...>  [a_SiIII]``.
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
    ap.set_defaults(blind=True)
    a = ap.parse_args()

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
                  seed=a.seed, ks_zlo=(a.ks_zlo if a.survey == "ks" else None)))
    print(f"=== wrote {len(chain_files)} chains -> {out_dir}/{root}.*.txt "
          f"(+ .paramnames .yaml .health.json) | A_p/n_s BLINDED={a.blind} ===")
    if info["private"]:
        print("    NOTE: DESI cosmology is PRIVATE — results_local/ is gitignored. Do NOT commit.")


if __name__ == "__main__":
    main()
