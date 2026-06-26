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
_STATE = {"on": False, "e": {}}                                  # toggled between clean and injected runs


def _mock_with_dla(ctx, dla_core_per_leg, truth_pack, key, **kw):
    mock_legs, info = _ORIG_MOCK(ctx, dla_core_per_leg, truth_pack, key, **kw)
    if _STATE["on"]:
        out = []
        for leg in mock_legs:
            e = _STATE["e"].get(leg.name)
            if e is None:
                out.append(leg)
            else:
                P = np.asarray(leg.P_data, float).copy()
                fin = np.isfinite(P)
                P[fin] = P[fin] + e[fin]                          # additive +1-sigma DLA-completeness mode
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1; a.n_warmup = min(a.n_warmup, 20); a.n_samples = min(a.n_samples, 30)

    ctx, d, _ = build_arm_ctx("metal_misspec", "desi", True)      # DESI-only deployed ctx (metals on)
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
    pickle.dump(dict(arm="dla_completeness", survey="desi", idxs=idxs, clean_per_mock=clean,
                     inj_per_mock=inj, meta=dict(vars(a), paired=True, wall_s=wall,
                     per_fit_wall_s=wall/(2*np_))), open(out, "wb"))
    nd = sum(int(r.get("n_div", 0) > 0) for r in clean) + sum(int(r.get("n_div", 0) > 0) for r in inj)
    print(f"[dla shard {a.shard}] wrote {len(inj)} paired mocks ({nd} div) -> {out} wall={wall:.1f}s")


if __name__ == "__main__":
    main()
