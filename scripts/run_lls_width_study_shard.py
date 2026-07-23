"""LLS prior-width HEDGE-FORM cosmology-sensitivity study (PI decision 2026-07-18:
"Run a few tests on how much these two matters for the cosmology").

THE QUESTION. The corrected-law swap set the deployed DESI-family LLS width sigma/mu = 0.287
(inference.HCD_LLS_SURVEY_FRAC_SIGMA, the 1x lit-anchored primary). The DORMANT 2x hedge knob
(inference.hcd_incidence_prior use_lls_width_hedge2x, default False, never passed by
build_legb_ctx) has two candidate values:
  0.574 = the literal 2x cascade (currently in HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X), vs
  0.416 = the semantically composed alternative (double ONLY the measurement part:
          hypot(2*0.1745, 0.2273)).
If a hedge arm ever runs, how different are the DESI-leg cosmology posteriors (A_p, n_s, plus
tau0_amp/dtau0 per repo convention, and alpha_lls) between widths {0.287 baseline, 0.416, 0.574}?

DESIGN. Reuses the paired-selfdraw machinery pattern (scripts/run_dla_selfdraw_shard.py +
build_arm_ctx use_prod_forward=True: the DEPLOYED NORC forward + the REDUCED DESI covariance),
but runs the CLEAN ARM ONLY (run_legb inject_spec=None) at an OVERRIDDEN LLS width. The
BASELINE (width 0.287) is the clean arm of the spot-check campaign already on disk
(/scratch/cavestru_root/cavestru1/mfho/dla_selfdraw_spotchk, job 53937355, seed 20260615,
shards {0, 8, 15}) -- NOT re-run here, and this runner must NEVER write into that outdir.

WIDTH OVERRIDE = a RUNTIME prior override in THIS process only (the desi_hcd_prior_sensitivity
idiom): rebind inference.HCD_LLS_SURVEY_FRAC_SIGMA to a NEW dict (DESI-family keys replaced, KS
untouched) BEFORE build_arm_ctx. hcd_incidence_prior reads the module global at call time, so
the patch lands on ctx.alpha_hcd_sigma; we ASSERT propagation (no silent no-op) AND that the
ORIGINAL deployed dict object is unmutated (no constant leak). NO deployed file is edited. The
override value + hcd_prior_signature (before/after) + forward_signature are stamped into the
pkl meta so the analyzer cannot mix arms silently.

SELFDRAW CAVEAT (document once, use everywhere). leg_a=True draws the truth FROM the prior:
for the same (seed, mock) the theta9 / tau0 / subDLA / DLA truth draws are IDENTICAL across
widths (per-site keys are unchanged), but the alpha_lls TRUTH draw scales with the width (same
base draw, wider TruncatedNormal). So the arms share the cosmology truth but NOT the LLS truth;
the mock data differ only through the LLS-truth displacement. The readout must therefore use
per-mock BIAS (posterior mean - truth) for alpha_lls, while A_p/n_s/tau0_amp/dtau0 mean shifts
vs baseline are directly interpretable (same truth values across arms).

READOUT PLAN (run when jobs finish; per width w in {0.416, 0.574}, per mock m in {0, 8}):
  base = clean_per_mock[0] of dla_selfdraw_spotchk/dla_selfdraw_desi_shard_{m:03d}.pkl
  test = clean_per_mock[0] of lls_width_study/lls_width_{tag}_shard_{m:03d}.pkl
  For p in (Ap, ns, alpha_lls): col = draws[:, names.index(p)];
  for tau0_amp/dtau0: rec["sites_extra"][p]["draws"] (+ ["truth"]).
  Report per param:
    sd ratio        = sd_test / sd_base
    mean shift      = (mean_test - mean_base) / sd_base        [sigma_post units]
    bias (selfdraw) = (mean - truth) / sd, per arm             [alpha_lls: the meaningful stat]
  DELIVERABLE = how much the hedge FORM choice (0.574 vs 0.416) changes the A_p / n_s
  posteriors (their mean-shift difference and sd-ratio difference), to inform whether the
  hedge-form decision matters for cosmology at all.

Budget: 2 widths x 2 mocks = 4 clean fits, ~15 ks/fit on 4 CPUs => ~70 CPU-h (cavestru1).
Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse, functools, os, pickle, time
import numpy as np
print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401 (x64)
import hcd_analysis.emulator.inference as _I
import hcd_analysis.emulator.closure_legb as CL
from hcd_analysis.emulator.closure_legb import run_legb

# The DESI-family keys the deployed 1x/2x dicts move together; KS keeps its own selection-driven
# rule (0.40) in BOTH deployed dicts, so it is deliberately NOT overridden here.
DESI_FAMILY_KEYS = ("DESI", "eBOSS", "DESI+KS")


def override_lls_width(width):
    """RUNTIME LLS-width override (the desi_hcd_prior_sensitivity idiom, adapted to the
    per-survey dict): REBIND inference.HCD_LLS_SURVEY_FRAC_SIGMA to a NEW dict with the
    DESI-family entries set to ``width`` (KS untouched). The original dict OBJECT is never
    mutated (tripwire-testable). Returns (original_dict, sig_before, sig_after) where sig_* =
    inference.hcd_prior_signature() so the pkl can stamp the effective prior identity."""
    before = _I.HCD_LLS_SURVEY_FRAC_SIGMA        # keep the ORIGINAL object (mutation tripwire)
    sig_before = _I.hcd_prior_signature()
    new = {**before, **{k: float(width) for k in DESI_FAMILY_KEYS}}
    # FROM-IMPORT REBINDING TRAP (job 53945595 failure): closure_legb imports the dict BY NAME
    # at module load (closure_legb.py:57) and build_legb_ctx:718 reads THAT binding, so
    # rebinding inference's global alone is a silent no-op at the ctx. Rebind BOTH modules.
    import hcd_analysis.emulator.closure_legb as _CL
    _I.HCD_LLS_SURVEY_FRAC_SIGMA = new
    _CL.HCD_LLS_SURVEY_FRAC_SIGMA = new
    sig_after = _I.hcd_prior_signature()
    return before, sig_before, sig_after


def restore_lls_width(original):
    """Rebind the deployed dict back in BOTH modules (tests / defensive symmetry)."""
    import hcd_analysis.emulator.closure_legb as _CL
    _I.HCD_LLS_SURVEY_FRAC_SIGMA = original
    _CL.HCD_LLS_SURVEY_FRAC_SIGMA = original


def width_tag(width):
    """'0.416' -> '0p416' (pkl / filename tag)."""
    return f"{float(width):.3f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, required=True,
                    help="LLS sigma/mu override for the DESI-family keys (study arms: 0.416 = "
                         "hedge measurement-only-doubled, 0.574 = literal 2x cascade). The "
                         "0.287 baseline comes from the spot-check campaign -- do NOT re-run it.")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=16,
                    help="MUST match the spot-check (16) so shard m -> mock m matches its mocks")
    ap.add_argument("--n-mocks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--n-warmup", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_mocks = 1; a.n_warmup = min(a.n_warmup, 20); a.n_samples = min(a.n_samples, 30)
    assert "dla_selfdraw_spotchk" not in os.path.abspath(a.out_dir), (
        "refusing to write into the RUNNING spot-check baseline outdir")

    tag = width_tag(a.width)
    os.makedirs(a.out_dir, exist_ok=True)
    # smoke writes a SUFFIXED pkl so it can never trip the real run's skip-if-exists.
    suffix = ".smoke" if a.smoke else ""
    out = os.path.join(a.out_dir, f"lls_width_{tag}_shard_{a.shard:03d}{suffix}.pkl")
    if os.path.exists(out):
        print(f"=== {out} exists -- SKIP ==="); return

    # ---- RUNTIME WIDTH OVERRIDE (before any ctx build) -------------------------------------
    base_dict, sig_before, sig_after = override_lls_width(a.width)
    base_snapshot = dict(base_dict)              # value snapshot for the mutation tripwire below
    hedge_snapshot = dict(_I.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X)
    print(f"[width-override] HCD_LLS_SURVEY_FRAC_SIGMA {base_snapshot} -> "
          f"{_I.HCD_LLS_SURVEY_FRAC_SIGMA}")
    print(f"[width-override] hcd_prior_signature {sig_before[:12]} -> {sig_after[:12]}")

    # DEPLOYED forward (single source: prod_forward_config + NORC via build_arm_ctx
    # use_prod_forward), mirroring run_dla_selfdraw_shard.py's parity asserts -- fail LOUD
    # before any NUTS. Imported here so the override above is unambiguously first.
    from scripts.run_dnuis_bias_shard import build_arm_ctx
    ctx, d, _ = build_arm_ctx("metal_misspec", "desi", True, use_prod_forward=True)
    fc = CL.prod_forward_config("DESI"); norc = CL.prod_norc_forward(); L = ctx.legs[0]
    assert bool(ctx.res_corr_on) == norc["res_corr_on"], "study forward != deployed NORC res_corr"
    assert bool(ctx.fix_alpha_res) == norc["fix_alpha_res"], "study fix_alpha_res != deployed NORC"
    assert bool(ctx.sample_res) == fc["sample_res"], "prod f_res float not wired into the study arm"
    assert ctx.f_res_amp_sigma == fc["f_res_amp_sigma"], "f_res prior width mismatch vs deployed DESI"
    assert ctx.metal_prior == fc["metal_prior"], "prod metal model (flatlog2node) not wired"
    assert tuple(ctx.metal_node_z) == (2.2, 4.2), "Gate-C metal_node_z drifted"
    assert ctx.sample_metals is True and L.metals_on is True, "DESI leg must sample+apply metals"
    # Same likelihood the spot-check baseline fits under (comparability): REDUCED covariance.
    assert getattr(L, "dla_cov_reduced", False) is True, (
        "DESI leg C_data is NOT the reduced covariance -- the study must run on the SAME "
        "production likelihood as the dla_selfdraw_spotchk baseline (data_likelihood."
        "DESI_DLA_COV_REDUCE)")
    assert float(L.dla_forward_frac) == 1.0, "DESI alpha_DLA mean model must be live (frac 1.0)"

    # ---- PROPAGATION + NO-MUTATION ASSERTS (the silent-no-op / constant-leak hazards) --------
    ratio = np.asarray(ctx.alpha_hcd_sigma) / np.asarray(ctx.alpha_hcd_mu)
    assert abs(ratio[0] - a.width) < 1e-9, (
        f"LLS width override NO-OP: ctx sigma/mu = {ratio[0]:.6f} != requested {a.width}")
    assert abs(ratio[1] - _I.HCD_PRIOR_FRAC_SIGMA[1]) < 1e-9, (
        f"subDLA width DRIFTED: {ratio[1]:.6f} != {_I.HCD_PRIOR_FRAC_SIGMA[1]} (override must "
        f"touch ONLY the LLS class)")
    assert dict(base_dict) == base_snapshot, (
        "DEPLOYED constant dict MUTATED by the override -- it must be rebound, never edited")
    assert dict(_I.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X) == hedge_snapshot, (
        "HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X changed -- the study must not touch the deployed 2x dict")
    print(f"[width-override] VERIFIED ctx sigma/mu = {np.round(ratio, 4).tolist()} "
          f"(LLS = {a.width}, subDLA/DLA deployed); deployed dicts unmutated")
    print(f"[forward] DEPLOYED DESI: res_corr_on={bool(ctx.res_corr_on)} "
          f"fix_alpha_res={bool(ctx.fix_alpha_res)} sample_res={bool(ctx.sample_res)} "
          f"metal_prior={ctx.metal_prior!r} dla_cov_reduced={L.dla_cov_reduced}")

    # ---- CLEAN-ARM-ONLY selfdraw fit (identical run_kw to the spot-check) --------------------
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    run_kw = dict(n_mocks=a.n_mocks, mock_indices=idxs, return_per_mock=True, leg_a=True,
                  n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, dense_mass=True,
                  max_tree_depth=a.max_tree_depth, verbose=True)
    t0 = time.time()
    print(f"[lls_width {tag} shard {a.shard}] CLEAN run (inject_spec=None, width {a.width}) ...")
    clean = run_legb(ctx, d, inject_spec=None, **run_kw)
    wall = time.time() - t0

    meta = dict(vars(a), arm="lls_width_study", paired=False, wall_s=wall,
                per_fit_wall_s=wall / max(len(clean), 1),
                lls_width_override=float(a.width), width_tag=tag,
                lls_width_dict_effective=dict(_I.HCD_LLS_SURVEY_FRAC_SIGMA),
                lls_width_dict_baseline=base_snapshot,
                hcd_prior_signature_baseline=sig_before,
                hcd_prior_signature_effective=sig_after,
                baseline_campaign="dla_selfdraw_spotchk (job 53937355, width 0.287, clean arm)")
    meta["forward"] = dict(res_corr_on=bool(ctx.res_corr_on), fix_alpha_res=bool(ctx.fix_alpha_res),
                           sample_res=bool(ctx.sample_res), f_res_amp_sigma=float(ctx.f_res_amp_sigma),
                           metal_prior=str(ctx.metal_prior),
                           metal_node_z=tuple(float(z) for z in ctx.metal_node_z),
                           dla_cov_reduced=bool(L.dla_cov_reduced),
                           dla_forward_frac=float(L.dla_forward_frac),
                           forward_signature=CL.forward_signature())
    tmp = out + f".tmp.{os.getpid()}"
    pickle.dump(dict(arm="lls_width_study", survey="desi", idxs=idxs, clean_per_mock=clean,
                     meta=meta), open(tmp, "wb"))
    os.replace(tmp, out)
    nd = sum(int(r.get("n_div", 0) > 0) for r in clean)
    print(f"[lls_width {tag} shard {a.shard}] wrote {len(clean)} clean mocks ({nd} div) -> {out} "
          f"wall={wall:.1f}s")


if __name__ == "__main__":
    main()
