"""NUTS-confirm the Stage-B DLA-completeness bias (Fisher said n_s -0.63, A_p -0.47 for a 1-sigma mode).

Inject a +1-sigma COHERENT realization of the DESI-provided syst_e_dla_completeness mode (mapped onto the
DESI leg's (z,k) rows) ADDITIVELY into the mock truth, then fit with the DEPLOYED forward (which floats
alpha_dla + the DLA core template, so this tests whether the masking-COMPLETENESS residual leaks BEYOND
our DLA-INCIDENCE model). Paired clean-vs-injected; the injected arm = clean P_data + e_dla (same truth +
same noise, so the noise cancels in the paired Delta).

SBC-SAFE: process-local monkeypatch of make_leg_a_legmock (on-disk closure_legb unchanged; running jobs
in other processes untouched). The deployed forward is NOT modified.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse, functools, os, pickle, time
import numpy as np
print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401 (x64)
import hcd_analysis.emulator.closure_legb as CL
from hcd_analysis.emulator.closure_legb import run_legb
from scripts.run_dnuis_bias_shard import build_arm_ctx

_ORIG_MOCK = CL.make_leg_a_legmock
_STATE = {"on": False, "e": {}, "strength": 1.0}                 # toggled between clean and injected runs;
#                                                                 strength scales the injected mode (+/-1sigma)


def _mock_with_dla(ctx, dla_core_per_leg, truth_pack, key, **kw):
    mock_legs, info = _ORIG_MOCK(ctx, dla_core_per_leg, truth_pack, key, **kw)
    if _STATE["on"]:
        s = float(_STATE.get("strength", 1.0))                   # +1.0 == the shipped +1sigma arm (byte-identical)
        out = []
        for leg in mock_legs:
            e = _STATE["e"].get(leg.name)
            if e is None:
                out.append(leg)
            else:
                P = np.asarray(leg.P_data, float).copy()
                fin = np.isfinite(P)
                P[fin] = P[fin] + s * e[fin]                      # additive strength*sigma DLA-completeness mode
                out.append(leg._replace(P_data=P))
        mock_legs = out
    return mock_legs, info


def build_e_dla(ctx):
    """Map syst_e_dla_completeness (DESI npz, 1020 rows) onto each leg's (z,k) rows (nearest)."""
    npz = np.load("/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz")
    nz, nk, edla = npz["z"], npz["k"], npz["syst_e_dla_completeness"]
    e = {}
    for leg in ctx.legs:
        k = np.asarray(leg.k); zrow = np.asarray(leg.z)[np.asarray(leg.z_idx)]
        idx = np.array([np.argmin((nz - zz) ** 2 / 0.01 + (nk - kk) ** 2) for zz, kk in zip(zrow, k)])
        v = np.asarray(edla)[idx]
        v[~np.isfinite(np.asarray(leg.P_data, float))] = 0.0
        e[leg.name] = v
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="scale the injected DLA-completeness mode for the +/-1sigma bracket (default 1.0 = "
                         "the shipped +1sigma arm, byte-identical; use -1.0 for the -1sigma arm). Stamped into meta.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1; a.n_warmup = min(a.n_warmup, 20); a.n_samples = min(a.n_samples, 30)
    _STATE["strength"] = float(a.strength)                        # thread into the monkeypatch (see _mock_with_dla)

    # The DLA re-run fits with the DEPLOYED DESI forward (sample_res=True / f_res_amp_sigma=0.02 / flatlog2node)
    # pulled from prod_forward_config -- the SINGLE source build_real_ctx consumes (run_real_fit.py) -- for
    # deployment-consistency: a diagnostic on a DIFFERENT forward than the deployed inference manufactures phantom
    # systematics. The injection logic (make_leg_a_legmock monkeypatch) is unchanged; only the fit forward moves.
    ctx, d, _ = build_arm_ctx("metal_misspec", "desi", True, use_prod_forward=True)
    # Load-bearing runtime parity assert (mirror run_real_fit.py:186-193) on the real built ctx (single restricted
    # DESI leg): fail LOUD so a future refactor cannot silently regress this re-run to the thin build_legb_ctx
    # defaults (sample_res=False / metal_prior='uniform'). Fires at run start, BEFORE any NUTS.
    fc = CL.prod_forward_config("DESI"); norc = CL.prod_norc_forward(); L = ctx.legs[0]
    assert bool(ctx.res_corr_on) == norc["res_corr_on"], "DLA re-run forward != deployed NORC res_corr"
    assert bool(ctx.fix_alpha_res) == norc["fix_alpha_res"], "DLA re-run fix_alpha_res != deployed NORC"
    assert bool(ctx.sample_res) == fc["sample_res"], "prod f_res float not wired into the DLA re-run"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"], "f_res prior width mismatch vs deployed DESI"
    assert ctx.metal_prior == fc["metal_prior"], "prod metal model (flatlog2node) not wired"
    assert tuple(ctx.metal_node_z) == (2.2, 4.2), "Gate-C metal_node_z drifted"
    assert ctx.sample_metals is True and L.metals_on is True, "DESI leg must sample+apply metals"
    print(f"[forward] DEPLOYED DESI: res_corr_on={bool(ctx.res_corr_on)} fix_alpha_res={bool(ctx.fix_alpha_res)} "
          f"sample_res={bool(ctx.sample_res)} f_res_amp_sigma={ctx.f_res_amp_sigma} "
          f"metal_prior={ctx.metal_prior!r} sample_metals={ctx.sample_metals} metals_on={L.metals_on} "
          f"(== prod_forward_config('DESI') + NORC)")
    _STATE["e"] = build_e_dla(ctx)
    CL.make_leg_a_legmock = _mock_with_dla
    edesi = _STATE["e"]["DESI"]
    print(f"[patch] make_leg_a_legmock -> +syst_e_dla_completeness on DESI; "
          f"|e_dla|/median: median frac of injected rows ~ {np.median(np.abs(edesi[edesi!=0])):.3e} (abs P units)")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    _STATE["on"] = False
    print(f"[dla shard {a.shard}] CLEAN run ...")
    clean = run_legb(ctx, d, inject_spec=None, **run_kw)
    _STATE["on"] = True
    print(f"[dla shard {a.shard}] INJECTED run (+1sigma DLA-completeness) ...")
    inj = run_legb(ctx, d, inject_spec=None, **run_kw)            # inject_spec=None; the monkeypatch adds e_dla
    _STATE["on"] = False
    wall = time.time() - t0
    np_ = max(min(len(clean), len(inj)), 1)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"dla_completeness_desi_shard_{a.shard:03d}.pkl")
    # Stamp the RESOLVED deployed forward from the built ctx (anti-pool + audit ingest guard): the analyzer
    # can then assert at ingest that the analyzed pkls were produced on the NORC deployed forward, so a
    # pre-fix thin-forward pkl (res_corr_on=True / metal_prior='uniform') is rejectable. strength is already
    # in vars(a) (the +/-1sigma bracket tag).
    meta = dict(vars(a), paired=True, wall_s=wall, per_fit_wall_s=wall / (2 * np_))
    # F2 (2026-07-19): the closure_legb.forward_stamp SINGLE AUTHORITY — a superset of the old
    # inline dict (adds dla_cov_reduced/dla_forward_frac, the env data-selection stamps, and both
    # freeze signatures). The analyzer's forward-stamp pooling key therefore separates post-F2
    # pkls from the archived campaign's (intended: different stamp coverage != poolable).
    meta["forward"] = CL.forward_stamp(ctx, L)
    pickle.dump(dict(arm="dla_completeness", survey="desi", idxs=idxs, clean_per_mock=clean,
                     inj_per_mock=inj, meta=meta), open(out, "wb"))
    nd = sum(int(r.get("n_div", 0) > 0) for r in clean) + sum(int(r.get("n_div", 0) > 0) for r in inj)
    print(f"[dla shard {a.shard}] wrote {len(inj)} paired mocks ({nd} div) -> {out} wall={wall:.1f}s")


if __name__ == "__main__":
    main()
