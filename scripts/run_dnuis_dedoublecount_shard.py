"""De-double-counted metal_misspec:desi re-measure -- the HONEST metal n_s bias.

The Gate B metal_misspec arm DOUBLE-COUNTS SiIII: make_leg_a_legmock builds the truth P_model WITH
the drawn truth a_SiIII (closure_legb.py:1225) and then metal_inject re-adds its OWN SiIII oscillation
(:1241), so the forward's single a_SiIII rails +41.9sigma chasing two copies and the -0.96 n_s is a
double-count-inflated CEILING. The 4-lens decisions panel (2026-06-25) requires fixing this and
re-measuring by NUTS BEFORE sizing any metal nuisance.

FIX: monkeypatch closure_legb.draw_leg_a_leg_truth to force truth a_siiii=0, so metal_inject's SiIII is
the ONLY SiIII content. Then the forward fits a_SiIII once (absorbing the injected SiIII) and the
measured paired bias is the HONEST unfittable residual (the SiII doublet + additive SiII-SiII). Both
the clean and injected arms use truth a_siiii=0 so the pairing stays clean.

SBC-SAFE: process-local monkeypatch only (on-disk closure_legb unchanged; the running 3-rung job +
golden test are separate processes/untouched). Same seed (20260615) + same N=8 mocks as the original
metal_misspec:desi cell -> directly comparable (-0.96 inflated vs the honest number measured here).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import os
import pickle
import time

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import hcd_analysis.emulator.closure_legb as CL
from hcd_analysis.emulator.closure_legb import run_legb
from scripts.run_dnuis_bias_shard import build_arm_ctx          # reuse the metal_misspec:desi ctx build

_ORIG_DRAW = CL.draw_leg_a_leg_truth


def _draw_zero_asiiii(ctx, key):
    """draw_leg_a_leg_truth but with the truth SiIII zeroed (removes the double-count)."""
    tp = dict(_ORIG_DRAW(ctx, key))
    tp["a_siiii"] = 0.0
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260615)         # SAME as the original cell
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--float-a-siii", action="store_true",
                    help="ALSO float a_SiII in the forward (Stage C NUTS-confirm: does the floated "
                         "SiII doublet absorb the honest -0.69 metal bias?)")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1
        a.n_warmup = min(a.n_warmup, 20)
        a.n_samples = min(a.n_samples, 30)

    CL.draw_leg_a_leg_truth = _draw_zero_asiiii                   # PROCESS-LOCAL monkeypatch
    print("[patch] draw_leg_a_leg_truth -> truth a_siiii=0 (de-double-count; on-disk unchanged)")

    ctx, d, inject_spec = build_arm_ctx("metal_misspec", "desi", True)
    if a.float_a_siii:
        ctx = ctx._replace(sample_a_siii=True)               # forward also floats the SiII doublet
        print("[forward] sample_a_siii=True -> floating a_SiII (the doublet fix for the -0.69)")
    # self-check: the patched truth has a_siiii==0 (and the unpatched would not, generically)
    tp = CL.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(0))
    assert float(tp["a_siiii"]) == 0.0, f"patch failed: truth a_siiii={tp['a_siiii']}"
    tp0 = _ORIG_DRAW(ctx, jax.random.PRNGKey(0))
    print(f"[selfcheck] patched truth a_siiii={tp['a_siiii']} (==0 OK); "
          f"unpatched draw would be a_siiii={float(tp0['a_siiii']):.4f}")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[dedbl metal_misspec/desi shard {a.shard}/{a.n_shards}] mocks={idxs} "
          f"inject_spec={inject_spec} PAIRED truth-a_siiii=0 "
          f"(warmup={a.n_warmup} samples={a.n_samples} seed={a.seed})")

    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[dedbl shard {a.shard}] CLEAN run ...")
    clean_per_mock = run_legb(ctx, d, inject_spec=None, **run_kw)
    print(f"[dedbl shard {a.shard}] INJECTED run (de-double-counted desi_full) ...")
    inj_per_mock = run_legb(ctx, d, inject_spec=inject_spec, **run_kw)
    wall = time.time() - t0
    n_pairs = max(min(len(clean_per_mock), len(inj_per_mock)), 1)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"metal_dedbl_desi_shard_{a.shard:03d}.pkl")
    meta = dict(vars(a))
    meta.update(inject_spec=inject_spec, paired=True, truth_a_siiii=0.0,
                legs=[l.name for l in ctx.legs], wall_s=wall, per_fit_wall_s=wall / (2.0 * n_pairs))
    with open(out, "wb") as f:
        pickle.dump(dict(arm="metal_dedbl", survey="desi", idxs=idxs,
                         clean_per_mock=clean_per_mock, inj_per_mock=inj_per_mock, meta=meta), f)
    n_div = (sum(int(r.get("n_div", 0) > 0) for r in clean_per_mock)
             + sum(int(r.get("n_div", 0) > 0) for r in inj_per_mock))
    print(f"[dedbl shard {a.shard}] wrote {len(inj_per_mock)} paired mocks ({n_div} div) -> {out} "
          f"wall={wall:.1f}s per-fit={wall/(2*n_pairs):.1f}s")


if __name__ == "__main__":
    main()
