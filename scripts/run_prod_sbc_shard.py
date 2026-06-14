"""Run ONE shard of the PRODUCTION-ensemble Leg-A SBC pilot: the mocks m with
m % n_shards == shard.

This is the §6 inference-calibration gate (validation Doc B): draw θ from the inference
prior, simulate a matched-C mock with the PRODUCTION N=5 ensemble forward (mean of P_filt
over members), run NUTS, and (after the cross-shard merge) check rank uniformity. Per-mock
RNG is fold_in(seed, m) so shards are disjoint + reproducible and the merged set equals a
single full run. Writes the raw per-mock records to {out_dir}/shard_{shard}.pkl for
merge_prod_sbc_shards.py.

COVARIANCE: the production cross-class C_emu (error_vector_xclass.npz). NOTE: the emucoh
k-coherent term is wired in the LEG path (build_legb_ctx), NOT this cache-grid build_ctx —
emucoh-on-cache is the one production-covariance gap to close before the FULL run (the pilot
certifies the ensemble forward + sampler calibration with the cross-class C_emu).

NUTS: run_leg_a_sbc uses sampler_numpyro.run_nuts (make_nuts defaults for the mass matrix /
tree depth). Whether to match the production leg-path config (dense mass, mtd=8/10) is a
pilot-review item — measure cost + ESS here first.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
from hcd_analysis.emulator.closure_sbc import build_ctx, run_leg_a_sbc

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"
XCLASS_EV = f"{REPO}/checkpoints/error_vector_xclass.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--n-z", type=int, default=3)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=600,
                    help="≥600 targets L_eff≥99 after thinning (NUTS ESS≈0.18/sample)")
    ap.add_argument("--prob", type=float, default=0.95)
    ap.add_argument("--xclass-ev", default=XCLASS_EV)
    ap.add_argument("--cemu-inflate", type=float, default=1.0)
    ap.add_argument("--single-member", action="store_true",
                    help="run on final_prod_seed0 only (cheap de-risk; NOT the production object)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
    if not members:
        raise SystemExit(f"no production ensemble checkpoints at {PROD_PREFIX}*.eqx")
    ens = [members[0]] if a.single_member else members
    ctx = build_ctx(n_z=a.n_z, seed=0, ensemble_ckpts=ens,
                    xclass_error_vector=a.xclass_ev, cemu_inflate=a.cemu_inflate)
    n_members = len(getattr(ctx.model, "members", [None]))
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[shard {a.shard}/{a.n_shards}] mocks={idxs}  members={n_members}  n_z={a.n_z}  "
          f"rho={os.path.basename(a.xclass_ev)}  (warmup={a.n_warmup} samples={a.n_samples})")

    records = run_leg_a_sbc(ctx, n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True,
                            n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed,
                            prob=a.prob, verbose=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"shard_{a.shard:03d}.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(idxs=idxs, n_z=a.n_z, per_mock=records, meta=vars(a)), f)
    n_div = sum(int(r.get("n_div", 0) > 0) for r in records)
    print(f"[shard {a.shard}] wrote {len(records)} mocks ({n_div} divergent) -> {out}")


if __name__ == "__main__":
    main()
