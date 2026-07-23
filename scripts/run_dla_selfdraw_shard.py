"""Displaced-truth PRIYA DLA-completeness closure arm (PI disposition 2026-07-17, Workstream A).

The SELF-CONSISTENT test the e_dla envelope campaign was not: mock AND inference use the SAME
deployed PRIYA R_DLA response (same filtering, dla_core add-back, normalization, redshift law,
dla_forward_frac), but the injected truth alpha_DLA is DISPLACED from the prior center by
--boost (default 1.5: the prior-drawn alpha_DLA scaled so its distribution median lands at
0.15 x (lit/sim) x w_DLA — the intended ~15% residual of the OBSERVED DLA incidence; the prior
center is 0.10). Paired clean-vs-boosted per mock: same seed => same truth draw AND the same
standard-normal noise vector g; the realized noise eps = chol(C_total(truth)) @ g differs between
arms at the sub-0.1% level (the boost perturbs C_total through the alpha-weighted C_emu
coefficients — the SBC-consistent construction; NOT byte-identical noise like the e_dla campaign's
post-noise additive injection), so realization noise and the A3 metal-floor baseline cancel in the
paired Delta to that precision. NO e_dla mean template, NO monkeypatch — the boost rides the
run_legb inject_spec contract ({"dla_truth_boost": boost} -> apply_dla_truth_boost).

Scientific question: does a 15%-level truth drawn from the deployed PRIYA DLA response recover
without material cosmology leakage (alpha_dla / tau0_amp / A_p / n_s) under the final production
likelihood?

Both arms fit under the REDUCED DESI covariance (DESI_DLA_COV_REDUCE: syst_e_dla_completeness
removed per-z because alpha_DLA floats — the cup1d "red" convention; asserted below), i.e. the
final production likelihood, NOT the stale pre-2026-07-17 covariance.

STAMP PROVENANCE (adversarial backfill F5, 2026-07-19): meta["forward"] now comes from the
closure_legb.forward_stamp single authority, which adds hcd_prior_signature (the prior-constants
freeze hash) + the env data-selection stamps. Pre-2026-07-19 campaign pkls (the 53851306 array)
LACK the hcd_prior_signature stamp; they were certified against the deployed prior by
latent reconstruction (the adversarial review record) — do not treat the missing key in
those pkls as a drifted prior.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse, functools, os, pickle, time
import numpy as np
print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401 (x64)
import hcd_analysis.emulator.closure_legb as CL
from hcd_analysis.emulator.closure_legb import run_legb
from scripts.run_dnuis_bias_shard import build_arm_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--boost", type=float, default=1.5,
                    help="truth alpha_DLA multiplier (pivot + z-resolved column). 1.5 puts the "
                         "prior-drawn truth median at 0.15 x (lit/sim) x w_DLA = the 15%% residual "
                         "of the OBSERVED DLA incidence (prior center = 0.10). Stamped into meta.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1; a.n_warmup = min(a.n_warmup, 20); a.n_samples = min(a.n_samples, 30)

    # DEPLOYED forward (single source: prod_forward_config + NORC via build_arm_ctx use_prod_forward),
    # mirroring run_dla_completeness_shard.py's parity asserts — fail LOUD before any NUTS.
    ctx, d, _ = build_arm_ctx("metal_misspec", "desi", True, use_prod_forward=True)
    fc = CL.prod_forward_config("DESI"); norc = CL.prod_norc_forward(); L = ctx.legs[0]
    assert bool(ctx.res_corr_on) == norc["res_corr_on"], "selfdraw forward != deployed NORC res_corr"
    assert bool(ctx.fix_alpha_res) == norc["fix_alpha_res"], "selfdraw fix_alpha_res != deployed NORC"
    assert bool(ctx.sample_res) == fc["sample_res"], "prod f_res float not wired into the selfdraw arm"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"], "f_res prior width mismatch vs deployed DESI"
    assert ctx.metal_prior == fc["metal_prior"], "prod metal model (flatlog2node) not wired"
    assert tuple(ctx.metal_node_z) == (2.2, 4.2), "Gate-C metal_node_z drifted"
    assert ctx.sample_metals is True and L.metals_on is True, "DESI leg must sample+apply metals"
    # PI requirement 5: BOTH arms fit under the final REDUCED covariance (never the stale fid cov).
    assert getattr(L, "dla_cov_reduced", False) is True, (
        "DESI leg C_data is NOT the reduced covariance (syst_e_dla_completeness still inside); "
        "the selfdraw closure must run on the final production likelihood — check "
        "data_likelihood.DESI_DLA_COV_REDUCE")
    assert float(L.dla_forward_frac) == 1.0, "DESI alpha_DLA mean model must be live (frac 1.0)"
    print(f"[forward] DEPLOYED DESI: res_corr_on={bool(ctx.res_corr_on)} fix_alpha_res={bool(ctx.fix_alpha_res)} "
          f"sample_res={bool(ctx.sample_res)} f_res_amp_sigma={ctx.f_res_amp_sigma} "
          f"metal_prior={ctx.metal_prior!r} dla_cov_reduced={L.dla_cov_reduced} "
          f"dla_forward_frac={L.dla_forward_frac} boost={a.boost} "
          f"(== prod_forward_config('DESI') + NORC + reduced cov)")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[dla selfdraw shard {a.shard}] CLEAN run (boost=1, reduced cov) ...")
    clean = run_legb(ctx, d, inject_spec=None, **run_kw)
    print(f"[dla selfdraw shard {a.shard}] BOOSTED run (dla_truth_boost={a.boost}) ...")
    boosted = run_legb(ctx, d, inject_spec={"dla_truth_boost": float(a.boost)}, **run_kw)
    wall = time.time() - t0
    np_ = max(min(len(clean), len(boosted)), 1)
    os.makedirs(a.out_dir, exist_ok=True)
    # F6a: smoke writes a SUFFIXED pkl (width-study mirror) so it can never be pooled as a real
    # shard (the analyzer additionally filters *.smoke.* basenames at ingest, belt-and-braces).
    suffix = ".smoke" if a.smoke else ""
    out = os.path.join(a.out_dir, f"dla_selfdraw_desi_shard_{a.shard:03d}{suffix}.pkl")
    meta = dict(vars(a), paired=True, wall_s=wall, per_fit_wall_s=wall / (2 * np_))
    # F5: the closure_legb.forward_stamp SINGLE AUTHORITY (adds hcd_prior_signature + the env
    # data-selection stamps on top of the old inline dict — see the docstring provenance note).
    meta["forward"] = CL.forward_stamp(ctx, L)
    pickle.dump(dict(arm="dla_selfdraw", survey="desi", idxs=idxs, clean_per_mock=clean,
                     boost_per_mock=boosted, meta=meta), open(out, "wb"))
    nd = sum(int(r.get("n_div", 0) > 0) for r in clean) + sum(int(r.get("n_div", 0) > 0) for r in boosted)
    print(f"[dla selfdraw shard {a.shard}] wrote {len(boosted)} paired mocks ({nd} div) -> {out} "
          f"wall={wall:.1f}s")


if __name__ == "__main__":
    main()
