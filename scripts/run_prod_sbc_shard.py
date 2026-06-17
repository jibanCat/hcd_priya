"""Run ONE shard of the PRODUCTION-ensemble Leg-A SBC pilot ON THE LEG GRIDS (path B).

The §6 inference-calibration gate on the ACTUAL production likelihood: build_legb_ctx with the
production BASELINE (2-param τ₀, lit-pinned HCD, emucoh + cross-class C_emu, MF P1D+dN/dX with the
off-diagonal covariance, eBOSS metals; hierarchical_hcd=False per the referee), the N=5 ENSEMBLE
forward. Per mock: draw θ from the prior → matched-C self-draw on the DESI(+KS+eBOSS) legs (C_mock
≡ C_like) → NUTS → rank. Per-mock RNG is fold_in(seed, m) so a shard subset reproduces the mocks a
single full run would draw.

PER-MOCK CHECKPOINT (OOM / 24h-wall fix, 2026-06-17). The OOM (job 51884006, MaxRSS≈16.77 GB on
``--mem=16g``) and the N=5 wall both came from running every mock of a shard in ONE ``run_legb``
call that accumulates every mock's draws in memory and writes only at the very end — so an OOM/wall
lost the WHOLE shard. We now run ONE mock at a time: ``_run_mock`` writes ``{out}/mock_{m:04d}.pkl``
atomically (tmp + os.replace) after each mock and SKIPS at the loop top if that pkl already exists,
freeing each mock's NUTS arrays before the next. A wall/OOM now loses ≤1 mock and a resubmit picks
up where it stopped. ``merge_prod_sbc_shards.py`` globs ``mock_*.pkl`` (the mock index is parsed
from the FILENAME — the per-mock record dict carries no ``m`` key). The end-of-run ``shard_*.pkl``
is still written for back-compat with already-merged shards.

FORWARD (2026-06-17): the CORRECTED HCD z-slope (re-centered on HCD_INCIDENCE_SLOPE ~2.4, commits
3603522/6358742/bad8f15, with the runtime forward-exponent guard ACTIVE) + the 1D power-law
incidence (NOT the 2D-tilt; hcd_2d_tilt defaults False) + the production MF correction. NOTE on
res_corr: the ``res_corr_on=False`` (NORC) forward flag is NOT yet wired into build_legb_ctx /
LegBCtx / data_likelihood (that is Group 1 of the SBC plan, a load-bearing forward change that
needs the golden-test + 4-referee gate). Until it lands this runner uses the EXISTING default MF
forward (res_corr anchored below 5× the box fundamental, alpha_res marginalized forward-only) — so
``--no-res-corr-on`` is accepted but is currently a NO-OP placeholder that only records the intent
in the meta. The mock TRUTH and the likelihood go through the SAME forward, so C_mock ≡ C_like and
the rank-uniformity null is exact regardless.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
from hcd_analysis.emulator.closure_legb import build_legb_ctx, run_legb

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"


def _mock_path(out_dir, m):
    return os.path.join(out_dir, f"mock_{int(m):04d}.pkl")


def _run_mock(ctx, d, m, out_dir, *, n_mocks, n_warmup, n_samples, max_tree_depth, seed,
              dense_mass=True, verbose=True):
    """Run (or load) ONE mock and persist it to ``{out_dir}/mock_{m:04d}.pkl``.

    SKIP-IF-EXISTS: if the per-mock pkl is already on disk (a previous task / attempt finished it)
    just load + return it — no re-NUTS. Otherwise call ``run_legb`` for the SINGLE mock index ``m``
    (its RNG is ``fold_in(seed, m)``, the SAME mock a full run would draw), then ATOMICALLY write
    the one record (tmp + os.replace) so a crash mid-write cannot leave a truncated pkl. Returns the
    per-mock record dict (the same dict ``run_legb(return_per_mock=True)`` puts in its list)."""
    os.makedirs(out_dir, exist_ok=True)
    path = _mock_path(out_dir, m)
    if os.path.exists(path):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        if verbose:
            print(f"  [mock {m}] SKIP (exists) -> {path}")
        return rec

    records = run_legb(ctx, d, n_mocks=n_mocks, mock_indices=[int(m)], return_per_mock=True,
                       leg_a=True, n_warmup=n_warmup, n_samples=n_samples, seed=seed,
                       dense_mass=dense_mass, max_tree_depth=max_tree_depth, verbose=verbose)
    assert len(records) == 1, f"expected 1 record for mock {m}, got {len(records)}"
    rec = records[0]
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, path)        # atomic on POSIX (same dir)
    if verbose:
        print(f"  [mock {m}] wrote -> {path} (n_div={rec.get('n_div', 0)})")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=600,
                    help="≥600 targets L_eff≥99 after thinning (NUTS ESS≈0.18/sample)")
    ap.add_argument("--max-tree-depth", type=int, default=10)   # match the production leg fit
    ap.add_argument("--no-mf", dest="with_mf", action="store_false")
    ap.add_argument("--no-eboss", dest="with_eboss", action="store_false")
    ap.add_argument("--res-corr-on", dest="res_corr_on", action="store_true",
                    help="(placeholder) keep res_corr ON; the NORC flag is not yet wired in the "
                         "forward — see the module docstring. Recorded in meta only.")
    ap.add_argument("--no-res-corr-on", dest="res_corr_on", action="store_false",
                    help="(placeholder) request NORC (res_corr OFF). NOT yet wired in the forward; "
                         "records the intent in meta. Default.")
    ap.add_argument("--single-member", action="store_true",
                    help="run on final_prod_seed0 only (cheap de-risk; NOT the production object)")
    ap.add_argument("--no-shard-pkl", dest="write_shard_pkl", action="store_false",
                    help="skip the back-compat end-of-run shard_*.pkl (per-mock pkls are the unit)")
    ap.add_argument("--out-dir", required=True)
    ap.set_defaults(with_mf=True, with_eboss=True, res_corr_on=False, write_shard_pkl=True)
    a = ap.parse_args()

    members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
    if not members:
        raise SystemExit(f"no production ensemble checkpoints at {PROD_PREFIX}*.eqx")
    ens = [members[0]] if a.single_member else members
    # the PRODUCTION baseline (referee: standard per-class HCD, hierarchical_hcd=False).
    ctx, d = build_legb_ctx(
        ensemble_ckpts=ens, use_xclass=True,
        with_mf=a.with_mf, mf_with_floor=a.with_mf,
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        with_eboss=a.with_eboss, metals_on=a.with_eboss, sample_metals=a.with_eboss,
        hierarchical_hcd=False)
    n_members = len(getattr(ctx.model, "members", [None]))
    n_z = len(ctx.z_global)
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[shard {a.shard}/{a.n_shards}] mocks={idxs}  members={n_members}  "
          f"legs={[l.name for l in ctx.legs]}  n_z={n_z}  mf={a.with_mf} eboss={a.with_eboss}  "
          f"res_corr_on={a.res_corr_on}(flag-not-wired; default-forward) "
          f"(warmup={a.n_warmup} samples={a.n_samples} mtd={a.max_tree_depth})")

    # PER-MOCK loop: one mock at a time, checkpoint + skip after each (bounds RSS, ≤1 mock lost
    # per OOM/wall, resumable). The end-of-run shard pkl is still written for back-compat.
    records = []
    for m in idxs:
        rec = _run_mock(ctx, d, m, a.out_dir, n_mocks=a.n_mocks, n_warmup=a.n_warmup,
                        n_samples=a.n_samples, max_tree_depth=a.max_tree_depth, seed=a.seed,
                        dense_mass=True, verbose=True)
        records.append(rec)

    if a.write_shard_pkl:
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, f"shard_{a.shard:03d}.pkl")
        with open(out, "wb") as f:
            pickle.dump(dict(idxs=idxs, n_z=n_z, per_mock=records, meta=vars(a)), f)
        n_div = sum(int(r.get("n_div", 0) > 0) for r in records)
        print(f"[shard {a.shard}] wrote {len(records)} mocks ({n_div} divergent) -> {out}")
    else:
        print(f"[shard {a.shard}] {len(records)} per-mock pkls in {a.out_dir} (no shard pkl)")


if __name__ == "__main__":
    main()
