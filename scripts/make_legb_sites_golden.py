#!/usr/bin/env python3
"""Generate the PRE-FEATURE byte-identity goldens for the per-leg HCD alpha sites build.

Two npz references (consumed by tests/test_legb_per_leg_alpha.py T1/T2), BOTH generated at
the PRE-FEATURE tree state (choreography: run this on the UNTOUCHED tree BEFORE any source
edit; the generating commit hash is recorded inside each npz):

  tests/golden/legb_sites_golden.npz
      Seeded-trace site NAME ORDER + bitwise VALUES for the deployed-DESI config
      (build_trace_ctx below: build_legb_ctx(survey="DESI") + the certified per-leg
      data-nuisance forward prod_forward_config("DESI") + the NORC decision
      prod_norc_forward(), single final_fold0 checkpoint standing in for the N=5
      ensemble, legs restricted to DESI — the run_real_fit.build_real_ctx shape), for
      BOTH twins (_legb_model with the loglik factor value, and _legb_priors_only).

  tests/golden/legb_shortnuts_golden.npz
      A short-NUTS bitwise reference: _run_nuts_legb 5 warmup / 5 samples,
      max_tree_depth=5, diag mass, fixed PRNGKey(20260720), on a small synthetic
      mock (narrow-z DESI-only ctx, build_nuts_ctx below) — the FULL samples dict
      (catches mass-ordering / flatten-order effects a single trace cannot).

REGENERATION PROTOCOL (T2 is env-pinned): regenerate ONLY with an explicit, recorded
reason (a deliberate model change or a jax/numpyro env migration), via
    PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
      scripts/make_legb_sites_golden.py
then commit the new npz together with the reason; the generating commit + jax/numpyro
versions are stamped inside the npz for the audit trail.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

import numpy as np

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
import jax
import numpyro
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as CL

REPO = "/home/mfho/hcd_priya"
GOLDEN_DIR = f"{REPO}/tests/golden"
TRACE_SEED = 7
NUTS_SEED = 20260720
MOCK_KEY = 0


def build_trace_ctx():
    """The deployed-DESI single-leg config (run_real_fit.build_real_ctx shape, single
    final_fold0 checkpoint standing in for the ensemble): certified per-leg forward
    (prod_forward_config) + NORC (prod_norc_forward) + survey='DESI' prior pin + MF +
    floor + emucoh(off-diag) + cross-class C_emu, legs restricted to DESI."""
    fc = CL.prod_forward_config("DESI")
    norc = CL.prod_norc_forward()
    ctx, d = CL.build_legb_ctx(
        use_xclass=True, with_mf=True, mf_with_floor=True,
        res_corr_on=norc["res_corr_on"],
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        metals_on=True, sample_metals=True,
        sample_res=fc["sample_res"], f_res_amp_sigma=fc["f_res_amp_sigma"],
        metal_prior=fc["metal_prior"], survey="DESI")
    if norc["fix_alpha_res"]:
        ctx = ctx._replace(fix_alpha_res=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    return ctx, d


def build_nuts_ctx():
    """A narrow-z DESI-only ctx (z<2.6 → the full real-grid path in ~1 min; the
    test_legb_hier_hcd _build_small idiom) with the deployed data-nuisance SITE SET
    (flatlog2node metals + f_res float + the alpha_res pair sampled) and the
    survey='DESI' prior pin — small enough for a 5+5 NUTS golden, rich enough to
    exercise every deployed site family's ordering."""
    ctx, d = CL.build_legb_ctx(
        desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
        metals_on=True, sample_metals=True,
        sample_res=True, f_res_amp_sigma=0.02,
        metal_prior="flatlog2node", survey="DESI")
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    return ctx, d


def mock_for(ctx, d, *, key=MOCK_KEY):
    """The deterministic held-out-sim mock (fixed sim, fixed noise key) + matched cores."""
    sims, _ = CL.held_out_sims(d, fold=0)
    truth = CL.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = CL.make_legb_mock(ctx, truth, jax.random.PRNGKey(key))
    core = CL._mock_core_per_leg(ctx, truth)
    return mock_legs, core


def trace_records(ctx, mock_legs, core, *, seed=TRACE_SEED):
    """Ordered (name, kind, value) records for both twins at one seeded trace.

    kinds: 'sample' / 'deterministic'; the _legb_model loglik factor value is returned
    separately (numpyro.factor emits an observed Unit site; its log_factor is the
    likelihood value at the traced prior draw)."""
    tr_m = handlers.trace(handlers.seed(
        lambda: CL._legb_model(ctx, mock_legs, core), jax.random.PRNGKey(seed))).get_trace()
    tr_p = handlers.trace(handlers.seed(
        lambda: CL._legb_priors_only(ctx), jax.random.PRNGKey(seed))).get_trace()

    def _records(tr):
        rec = []
        factor = None
        for name, site in tr.items():
            if site["type"] == "deterministic":
                rec.append((name, "deterministic", np.asarray(site["value"])))
            elif site["type"] == "sample":
                if site.get("is_observed"):
                    # the numpyro.factor site: pin its log_factor (the loglik value).
                    factor = np.asarray(site["fn"].log_factor)
                else:
                    rec.append((name, "sample", np.asarray(site["value"])))
        return rec, factor

    model_rec, model_factor = _records(tr_m)
    prior_rec, prior_factor = _records(tr_p)
    assert prior_factor is None, "_legb_priors_only must carry NO factor site"
    assert model_factor is not None, "_legb_model must carry the loglik factor site"
    return model_rec, model_factor, prior_rec


def _provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    return dict(git_commit=commit,
                generated_utc=datetime.now(timezone.utc).isoformat(),
                jax_version=jax.__version__, numpyro_version=numpyro.__version__,
                numpy_version=np.__version__)


def make_sites_golden(path):
    ctx, d = build_trace_ctx()
    mock_legs, core = mock_for(ctx, d)
    model_rec, model_factor, prior_rec = trace_records(ctx, mock_legs, core)
    payload = dict(
        model_names=np.asarray([n for n, _, _ in model_rec]),
        model_kinds=np.asarray([k for _, k, _ in model_rec]),
        prior_names=np.asarray([n for n, _, _ in prior_rec]),
        prior_kinds=np.asarray([k for _, k, _ in prior_rec]),
        loglik_factor=np.asarray(model_factor),
        trace_seed=np.asarray(TRACE_SEED), mock_key=np.asarray(MOCK_KEY),
        **{f"model::{n}": v for n, _, v in model_rec},
        **{f"prior::{n}": v for n, _, v in prior_rec},
        **{k: np.asarray(v) for k, v in _provenance().items()},
    )
    np.savez(path, **payload)
    print(f"[sites-golden] wrote {path}: {len(model_rec)} model sites "
          f"({sum(1 for _, k, _ in model_rec if k == 'sample')} sample), "
          f"{len(prior_rec)} priors-only sites, loglik={float(model_factor):.6f}")


def run_short_nuts(ctx, mock_legs, core):
    """The pinned short-NUTS config: 5 warmup / 5 samples, mtd 5, diag mass, fixed key,
    fast_postprocess (deployed default). Returns (samples dict, n_div)."""
    return CL._run_nuts_legb(ctx, mock_legs, core, n_warmup=5, n_samples=5,
                             seed=NUTS_SEED, target_accept=0.9, dense_mass=False,
                             max_tree_depth=5)


def make_shortnuts_golden(path):
    ctx, d = build_nuts_ctx()
    mock_legs, core = mock_for(ctx, d)
    samples, n_div = run_short_nuts(ctx, mock_legs, core)
    keys = list(samples.keys())
    payload = dict(
        sample_keys=np.asarray(keys), n_div=np.asarray(n_div),
        nuts_seed=np.asarray(NUTS_SEED), mock_key=np.asarray(MOCK_KEY),
        **{f"nuts::{k}": np.asarray(samples[k]) for k in keys},
        **{k: np.asarray(v) for k, v in _provenance().items()},
    )
    np.savez(path, **payload)
    print(f"[shortnuts-golden] wrote {path}: {len(keys)} sample-dict keys, n_div={int(n_div)}")


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    make_sites_golden(os.path.join(GOLDEN_DIR, "legb_sites_golden.npz"))
    make_shortnuts_golden(os.path.join(GOLDEN_DIR, "legb_shortnuts_golden.npz"))


if __name__ == "__main__":
    main()
