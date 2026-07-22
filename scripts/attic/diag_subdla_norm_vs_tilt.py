# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# subDLA DOF (norm vs tilt) diagnostic, documented.
"""Which subDLA DOF does the data pull/constrain — NORMALIZATION (alpha_subdla pivot amplitude) or
TILT (s_subdla z-slope)? — on the REAL 3-leg production NUTS path (referee pre-check #2).

The production SBC/DV outputs drop the marginalized z-slope sites, so s_subdla cannot be read there.
This reuses the EXACT production path (_run_nuts_legb on the production ctx) and keeps the RAW samples
dict (which includes s_subdla). SHARDED: one (mock, chain) per process, per-task npz, skip-if-exists,
so it survives the 24h wall. Env: DIAG_MOCK (held-out-sim index, fold 0), DIAG_CHAIN, DIAG_OUTDIR.

Analyze (scripts/analyze_subdla_norm_vs_tilt.py over the per-task npz):
  - corr(ns, alpha_subdla)  -- AMPLITUDE channel (validate ~+0.59..0.73 vs the SBC ground truth)
  - corr(ns, s_subdla)      -- TILT channel (referee: confirm WEAK on the 3-leg path, not DESI-only)
  - sd(s_subdla)/prior      -- does the DATA constrain the tilt (shrink<<1) or prior-dominated (~1)?
"""
import os, glob
import numpy as np
import hcd_analysis.emulator  # x64 before jax
import jax
from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock,
    _mock_core_per_leg, _run_nuts_legb, ZSLOPE_PRIOR_SIGMA, HCD_INCIDENCE_SLOPE)

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"
OUTDIR = os.environ.get("DIAG_OUTDIR", f"{REPO}/checkpoints/diag_normtilt")
MOCK = int(os.environ.get("DIAG_MOCK", "0"))
CHAIN = int(os.environ.get("DIAG_CHAIN", "0"))
N_WARMUP = int(os.environ.get("DIAG_NWARMUP", "250"))
N_SAMPLES = int(os.environ.get("DIAG_NSAMPLES", "400"))
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, f"diag_m{MOCK:02d}_c{CHAIN:02d}.npz")
if os.path.exists(out):
    print(f"SKIP (exists): {out}"); raise SystemExit(0)

members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
# PRODUCTION config (sigma_amp=0.40, the current state — validate the tilt verdict on the real path)
ctx, d = build_legb_ctx(
    ensemble_ckpts=members, use_xclass=True, with_mf=True, mf_with_floor=True,
    mf_emucoh=True, mf_emucoh_offdiag_only=True,
    with_eboss=True, metals_on=True, sample_metals=True, hierarchical_hcd=False)
print(f"[m{MOCK} c{CHAIN}] marginalize_zslope={getattr(ctx,'marginalize_zslope',None)} "
      f"legs={[l.name for l in ctx.legs]} s_subdla prior sd={float(ZSLOPE_PRIOR_SIGMA[1]):.3f}")

sims, _ = held_out_sims(d, fold=0)
sim = sims[MOCK % len(sims)]
truth = make_truth_from_sim(d, sim, fold=0, mf=ctx.mf)
key0 = jax.random.PRNGKey(0)
k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, MOCK), 2)
mock_legs, truth_pack, info = make_legb_mock(ctx, truth, k_mock)
core = _mock_core_per_leg(ctx, truth)
ck = jax.random.fold_in(k_nuts, CHAIN)
samples, n_div, extra = _run_nuts_legb(
    ctx, mock_legs, core, n_warmup=N_WARMUP, n_samples=N_SAMPLES, seed=ck,
    target_accept=0.9, dense_mass=True, max_tree_depth=10, return_extra=True)
th = np.asarray(samples["theta_unit"])      # ns=col0, Ap=col1 (unit space; corr-invariant)
rec = dict(
    sim=sim, mock=MOCK, chain=CHAIN, n_div=int(n_div),
    ns=th[:, 0], Ap=th[:, 1],
    alpha_subdla=np.asarray(samples["alpha_subdla"]),
    alpha_lls=np.asarray(samples["alpha_lls"]),
    s_subdla=np.asarray(samples["s_subdla"]),
    s_subdla_prior_sd=float(ZSLOPE_PRIOR_SIGMA[1]),
    s_subdla_truth=float(HCD_INCIDENCE_SLOPE[1]),
)
tmp = out + f".tmp.{os.getpid()}"
np.savez(tmp, **rec); os.replace(tmp, out)
ns, ssub = rec["ns"], rec["s_subdla"]
print(f"[m{MOCK} c{CHAIN}] div={n_div} corr(ns,s_subdla)={np.corrcoef(ns,ssub)[0,1]:+.2f} "
      f"corr(ns,alpha_subdla)={np.corrcoef(ns,rec['alpha_subdla'])[0,1]:+.2f} "
      f"sd(s_subdla)/prior={ssub.std()/rec['s_subdla_prior_sd']:.2f} -> {out}")
