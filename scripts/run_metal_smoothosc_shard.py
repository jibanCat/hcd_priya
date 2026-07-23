"""DECISIVE smooth-vs-oscillatory metal injection test (the mechanism arbiter for Gate B n_s).

The Gate B 4-lens binning triple-check concluded the metal n_s leak rides the SMOOTH residual, NOT
the oscillatory SiII-doublet zigzag — but two linear Fishers DISAGREED in sign (real-emulator +0.10
vs proxy −0.6) and neither matched NUTS (−0.96). This runner settles it with NUTS: inject ONLY the
smooth pieces of the desi_full metal model vs ONLY the oscillatory pieces, paired clean-vs-injected,
and read the (A_p, n_s) bias. PREDICTION: smooth-only ≈ the full metal_misspec:desi (−0.96), osc-only
≈ 0 (the zigzag is n_s-inert). smooth + osc per mock ≈ the full metal_misspec residual.

SBC-SAFE: this is a NEW file and it MONKEYPATCHES closure_legb.metal_inject IN THIS PROCESS ONLY
(the on-disk module is byte-unchanged; the running 3-rung job, the golden test, and the production
forward are untouched). The patch is component-aware: component="full" reproduces the ORIGINAL
metal_inject(form="desi_full") to machine precision (asserted at startup); "smooth"/"osc" return only
the a²-constants+additive-SiII-SiII piece / the cos cross-terms.

SAME SEED (20260615) and fold_in(seed, m) as the Gate B metal_misspec:desi cell, so mocks 0..N-1 are
the IDENTICAL truths+noise → the smooth/osc cells are directly comparable to the full cell, and the
smooth/osc Δbias should add to the full Δbias per mock.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle
import time

import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import hcd_analysis.emulator.closure_legb as CL
import hcd_analysis.emulator.data_likelihood as DL
from hcd_analysis.emulator.closure_legb import build_legb_ctx, run_legb, _LAMBDA_SiIIb
from hcd_analysis.emulator.prod_ensemble import production_member_paths

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"

# the ORIGINAL metal_inject (captured before patching, for the full-component self-check).
_ORIG_METAL_INJECT = CL.metal_inject


def metal_inject_component(P, k, mean_flux, *, form="desi_full", component="full",
                           f_SiIII=0.009, f_SiII=0.004, f_SiII_SiII=0.002,
                           r_doublet=0.5, k_damp=0.05, k_decorr=0.05):
    """desi_full metal contamination split into SMOOTH (a² constants + additive SiII-SiII non-cos)
    and OSC (the cos cross-terms). component in {full, smooth, osc}. full == ORIGINAL metal_inject.
    (Only desi_full is split here; eboss is not needed for this test.)"""
    assert form == "desi_full", f"smooth/osc split only defined for desi_full (got {form!r})"
    k = np.asarray(k, float)
    omF = max(1.0 - float(mean_flux), 1e-3)
    A3 = f_SiIII / omF
    A2 = f_SiII / omF
    dvA = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)      # Lyα–SiIII
    dva = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiII)       # Lyα–SiII a
    dvb = DL.C_KMS * np.log(DL.LAMBDA_LYA / _LAMBDA_SiIIb)        # Lyα–SiII b
    dvd = DL.C_KMS * np.log(_LAMBDA_SiIIb / DL.LAMBDA_SiII)       # SiII intra-doublet
    D = 2.0 - 2.0 / (1.0 + np.exp(-k / k_decorr))
    gdamp = np.exp(-(k / k_damp) ** 2)
    # SMOOTH = a² DC constants + the NON-cos part of the additive same-ion SiII-SiII term
    C_smooth = (A3 ** 2 + A2 ** 2 * (1.0 + r_doublet ** 2)) + f_SiII_SiII * (1.0 + r_doublet ** 2) * gdamp
    # OSC = the cos cross-terms (SiIII + SiII doublet) + the cos part of the additive term
    C_osc = (2.0 * A3 * np.cos(dvA * k) * D
             + 2.0 * A2 * (np.cos(dvb * k) + r_doublet * np.cos(dva * k)) * D
             + f_SiII_SiII * (2.0 * r_doublet * np.cos(dvd * k)) * gdamp)
    C = {"smooth": C_smooth, "osc": C_osc, "full": C_smooth + C_osc}[component]
    return P * (1.0 + C)


def _selfcheck_full_matches_original():
    """component='full' MUST equal the original metal_inject(form='desi_full') to machine precision."""
    rng = np.linspace(1e-4, 0.06, 257)
    P = 1.0 + 0.3 * np.cos(rng * 1234.0)                          # arbitrary nonconstant P
    for F in (0.85, 0.70, 0.49):
        a = np.asarray(_ORIG_METAL_INJECT(P, rng, F, form="desi_full"))
        b = np.asarray(metal_inject_component(P, rng, F, form="desi_full", component="full"))
        err = float(np.max(np.abs(a - b)))
        assert err < 1e-12, f"full-component mismatch at <F>={F}: max|diff|={err:.2e}"
        # smooth + osc must also reconstruct full
        s = np.asarray(metal_inject_component(P, rng, F, component="smooth"))
        o = np.asarray(metal_inject_component(P, rng, F, component="osc"))
        rec = float(np.max(np.abs((s - P) + (o - P) - (b - P))))  # (1+Cs)+(1+Co)-(1+Cf) on P
        assert rec < 1e-12, f"smooth+osc != full at <F>={F}: {rec:.2e}"
    print("[selfcheck] component='full' == original metal_inject; smooth+osc==full (machine precision)")


def build_desi_ctx():
    # PINNED members (freeze decision 6): manifest-verified (sha256 + exact pairing + count +
    # stray-member tripwire) via checkpoints/production_ensemble_manifest.json, NOT a glob.
    members = production_member_paths(checkpoints_dir=os.path.dirname(PROD_PREFIX))
    ctx, d = build_legb_ctx(
        ensemble_ckpts=members, use_xclass=True, with_mf=True, mf_with_floor=True,
        mf_emucoh=True, mf_emucoh_offdiag_only=True, with_eboss=False,
        metals_on=True, sample_metals=True, hierarchical_hcd=False, survey="DESI")
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    if not ctx.legs:
        raise SystemExit("no DESI leg in ctx")
    return ctx, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["metal_smooth", "metal_osc"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-mocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260615)        # SAME as the Gate B metal_misspec cell
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1
        a.n_warmup = min(a.n_warmup, 20)
        a.n_samples = min(a.n_samples, 30)

    _selfcheck_full_matches_original()
    CL.metal_inject = metal_inject_component                     # PROCESS-LOCAL monkeypatch
    print("[patch] closure_legb.metal_inject -> component-aware (process-local; on-disk unchanged)")

    component = "smooth" if a.arm == "metal_smooth" else "osc"
    inject_spec = {"metal_misspec": {"form": "desi_full", "component": component}}

    ctx, d = build_desi_ctx()
    n_members = len(getattr(ctx.model, "members", [None]))
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    print(f"[smoothosc {a.arm} shard {a.shard}/{a.n_shards}] mocks={idxs} members={n_members} "
          f"legs={[l.name for l in ctx.legs]} inject_spec={inject_spec} PAIRED "
          f"(warmup={a.n_warmup} samples={a.n_samples} mtd={a.max_tree_depth} seed={a.seed})")

    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[smoothosc {a.arm} shard {a.shard}] CLEAN run (inject_spec=None) ...")
    clean_per_mock = run_legb(ctx, d, inject_spec=None, **run_kw)
    print(f"[smoothosc {a.arm} shard {a.shard}] INJECTED run ({component}) ...")
    inj_per_mock = run_legb(ctx, d, inject_spec=inject_spec, **run_kw)
    wall = time.time() - t0
    n_pairs = max(min(len(clean_per_mock), len(inj_per_mock)), 1)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.arm}_desi_shard_{a.shard:03d}.pkl")
    meta = dict(vars(a))
    meta.update(inject_spec=inject_spec, n_members=n_members, paired=True, component=component,
                legs=[l.name for l in ctx.legs], wall_s=wall, per_fit_wall_s=wall / (2.0 * n_pairs))
    with open(out, "wb") as f:
        pickle.dump(dict(arm=a.arm, survey="desi", idxs=idxs,
                         clean_per_mock=clean_per_mock, inj_per_mock=inj_per_mock, meta=meta), f)
    n_div = (sum(int(r.get("n_div", 0) > 0) for r in clean_per_mock)
             + sum(int(r.get("n_div", 0) > 0) for r in inj_per_mock))
    print(f"[smoothosc {a.arm} shard {a.shard}] wrote {len(inj_per_mock)} paired mocks "
          f"({n_div} divergent fits) -> {out}  wall={wall:.1f}s per-fit={wall/(2*n_pairs):.1f}s")


if __name__ == "__main__":
    main()
