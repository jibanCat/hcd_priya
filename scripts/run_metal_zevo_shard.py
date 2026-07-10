#!/usr/bin/env python3
"""Model C+ per-leg metal-injection BIAS test (paired clean-vs-injected, flatlog2node forward).

Injects a METAL_ZEVO arm as the SOLE metal truth and fits with the Model C+ forward (floated
f-nodes + per-z k_SiIII/k_SiII decorrelation + the UNDAMPED SiIII-SiII cross). Measures the
recovered A_p / n_s (+ tau0 / dtau0) bias per survey. Under metal_prior="flatlog2node" the
leg-A truth-draw carries NO scalar SiIII (truth a_siiii=0), so the injected arm is the SOLE metal
signal -- there is NO double-count (a self-check asserts it). eBOSS is SiIII-only (the SiII doublet
+ cross auto-vanish, metal_siII_legs=("DESI",)); DESI carries SiIII+SiII+cross. KS is metals_off
(no metal test) and is intentionally unsupported.

Reuses build_arm_ctx (the production N=5 single-survey ctx) then flips metal_prior -> flatlog2node
and swaps the additive metal_misspec injection for a METAL_ZEVO arm. Paired: same (seed, mock idx)
run twice -- CLEAN (inject_spec=None) and INJECTED -- so truth theta + cosmic noise cancel in the
per-mock delta-bias. Output pkl mirrors run_dnuis_dedoublecount_shard.py (clean_per_mock/inj_per_mock
+ meta), analyzable with scripts/analyze_dnuis_bias.py.

  python scripts/run_metal_zevo_shard.py --survey desi --arm arm2_increasing \
      --shard 0 --n-shards 4 --n-mocks 8 --out-dir <scratch>/metal_zevo
  SMOKE: --smoke ;  ctx-only: --dry-run
"""
import argparse
import os
import pickle
import time

import jax

import hcd_analysis.emulator.closure_legb as C
from hcd_analysis.emulator.closure_legb import run_legb
from scripts.run_dnuis_bias_shard import build_arm_ctx           # production single-survey ctx build

ARMS = ("arm1_decreasing", "arm2_increasing", "arm3_ma2025", "arm4_decreasing_ooc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True, choices=["desi", "eboss"],
                    help="metal leg (KS is metals_off -> no metal test, unsupported)")
    ap.add_argument("--arm", required=True, choices=list(ARMS),
                    help="METAL_ZEVO arm injected as the sole metal truth (in-class arm1/arm2, "
                         "out-of-class arm3 ma2025 / arm4 gauss)")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260615)         # SAME seed family as the metal cells
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny warmup/samples/mocks for an end-to-end shakedown")
    ap.add_argument("--dry-run", action="store_true",
                    help="build ctx + assert flatlog2node/no-double-count, then exit BEFORE NUTS")
    a = ap.parse_args()
    if a.smoke:
        a.n_warmup = min(a.n_warmup, 20)
        a.n_samples = min(a.n_samples, 20)
        a.n_mocks = min(a.n_mocks, 2)

    # Production single-survey ctx, then engage Model C+ (flatlog2node: floated f-nodes + k-nodes +
    # the undamped SiIII-SiII cross). The additive inject_spec from build_arm_ctx is DISCARDED; we
    # inject a METAL_ZEVO arm instead.
    ctx, d, _ = build_arm_ctx("metal_misspec", a.survey, with_mf=True)
    ctx = ctx._replace(metal_prior="flatlog2node")
    inject_spec = {"metal_misspec": dict(C.METAL_ZEVO_ARMS[a.arm])}

    # self-check: under flatlog2node the clean truth carries no scalar SiIII -> no double-count.
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(0))
    assert float(tp["a_siiii"]) == 0.0, (
        f"flatlog2node truth must have a_siiii=0 (no double-count); got {tp['a_siiii']}")

    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[metal_zevo {a.arm}/{a.survey} shard {a.shard}/{a.n_shards}] mocks={idxs} "
          f"metal_prior={ctx.metal_prior} inject={inject_spec} PAIRED "
          f"(warmup={a.n_warmup} samples={a.n_samples} seed={a.seed})")

    if a.dry_run:
        print(f"[dry-run] survey={a.survey} metal_prior={ctx.metal_prior} "
              f"knode=[{getattr(ctx, 'metal_knode_lo', None)},{getattr(ctx, 'metal_knode_hi', None)}] "
              f"fnode=[{getattr(ctx, 'metal_fnode_lo', None)},{getattr(ctx, 'metal_fnode_hi', None)}] "
              f"siII_legs={getattr(ctx, 'metal_siII_legs', None)} "
              f"legs={[l.name for l in ctx.legs]} arm={a.arm} mocks={idxs}")
        print("[dry-run] OK: ctx builds, flatlog2node wired, truth a_siiii=0 (no double-count). "
              "Exit before NUTS.")
        return

    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[metal_zevo shard {a.shard}] CLEAN run (inject_spec=None) ...")
    clean_per_mock = run_legb(ctx, d, inject_spec=None, **run_kw)
    print(f"[metal_zevo shard {a.shard}] INJECTED run (arm {a.arm}) ...")
    inj_per_mock = run_legb(ctx, d, inject_spec=inject_spec, **run_kw)
    wall = time.time() - t0
    n_pairs = max(min(len(clean_per_mock), len(inj_per_mock)), 1)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"metal_zevo_{a.arm}_{a.survey}_shard_{a.shard:03d}.pkl")
    meta = dict(vars(a))
    meta.update(inject_spec=inject_spec, paired=True, truth_a_siiii=0.0,
                metal_prior="flatlog2node", legs=[l.name for l in ctx.legs],
                wall_s=wall, per_fit_wall_s=wall / (2.0 * n_pairs))
    with open(out, "wb") as f:
        pickle.dump(dict(arm=a.arm, survey=a.survey, idxs=idxs,
                         clean_per_mock=clean_per_mock, inj_per_mock=inj_per_mock, meta=meta), f)
    n_div = (sum(int(r.get("n_div", 0) > 0) for r in clean_per_mock)
             + sum(int(r.get("n_div", 0) > 0) for r in inj_per_mock))
    print(f"[metal_zevo shard {a.shard}] wrote {len(inj_per_mock)} paired mocks ({n_div} div) -> {out} "
          f"wall={wall:.1f}s per-fit={wall / (2 * n_pairs):.1f}s")


if __name__ == "__main__":
    main()
