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
res_corr: the ``res_corr_on`` (NORC) forward flag IS NOW WIRED (2026-07-04, Gate-A) through
build_legb_ctx -> build_mf_correction -> MultiFidelity.res_corr_on and, for NORC, this runner also
pins fix_alpha_res=True + caps KS at k<=0.045. DEFAULT is NORC (res_corr_on=False); pass
``--res-corr-on`` to restore the pre-NORC anchored+alpha forward. res_corr_on lives on the mf object
(the single chokepoint), so the mock TRUTH and the likelihood SHARE it -> C_mock ≡ C_like and the
rank-uniformity null is exact at either setting. (Gated by the 4-referee panel + freeze.)

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
"""
import argparse
import functools
import glob
import os
import pickle

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
from hcd_analysis.emulator.closure_legb import build_legb_ctx, run_legb, prod_forward_config

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"


def _mock_path(out_dir, m):
    return os.path.join(out_dir, f"mock_{int(m):04d}.pkl")


def _run_mock(ctx, d, m, out_dir, *, n_mocks, n_warmup, n_samples, max_tree_depth, seed,
              dense_mass=True, verbose=True, leg_a=True, run_cfg=None, fold=0, inject_spec=None):
    """Run (or load) ONE mock and persist it to ``{out_dir}/mock_{m:04d}.pkl``.

    SKIP-IF-EXISTS: if the per-mock pkl is already on disk (a previous task / attempt finished it)
    just load + return it — no re-NUTS. Otherwise call ``run_legb`` for the SINGLE mock index ``m``
    (its RNG is ``fold_in(seed, m)``, the SAME mock a full run would draw), then ATOMICALLY write
    the one record (tmp + os.replace) so a crash mid-write cannot leave a truncated pkl. Returns the
    per-mock record dict (the same dict ``run_legb(return_per_mock=True)`` puts in its list).

    CONFIG-KEY GUARD (2026-06-19): the per-mock pkl is keyed by INDEX ONLY (``mock_{m:04d}.pkl``),
    with NO leg_a / cemu_variant / amp_sigma discriminator. So a held-out (or cemu-variant /
    width-check) run pointed at an ``--out-dir`` that already holds self-draw (or differently-
    configured) pkls would SILENTLY ``SKIP`` and load the WRONG config — quietly mixing two
    different SBC populations in one directory. We now STAMP the run config (``run_cfg``: leg_a,
    cemu_variant, amp_sigma) into the record dict under ``run_cfg`` and, on the skip branch, ASSERT
    the existing pkl's stamp MATCHES the requested one — RAISING on a mismatch (so a config clash is
    a loud failure, not a silent wrong-config load). Pre-stamp pkls (no ``run_cfg`` key) are treated
    as a clash when a non-default config is requested (a conservative, fail-loud default).

    ``leg_a`` (default True = the SELF-DRAW production SBC): when False, ``run_legb`` builds the
    mock from a HELD-OUT SIM (make_legb_mock + held_out_sims, fold 0) so the mock TRUTH is the sim's
    MEASURED power and the likelihood forward is the EMULATOR prediction — the path that CONTAINS
    emulator error (the honest C_emu instrument; the self-draw cancels emu error by construction)."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(run_cfg) if run_cfg else {}
    path = _mock_path(out_dir, m)
    if os.path.exists(path):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        existing = rec.get("run_cfg")
        # The DEFAULT (pre-2026-06-19) config every un-stamped pkl was written under.
        default_cfg = dict(leg_a=True, cemu_variant="current", amp_sigma=0.0, leg="all", fold=0,
                           tau0_prior_sigma=0.0, subdla_truth_boost=1.0, res_corr_on=True,
                           sample_res=False, f_res_amp_sigma=None, metal_prior="uniform")
        # A PRE-STAMP pkl (no run_cfg) is treated as the default config — so resuming a DEFAULT run
        # over old pkls still works, but a non-default (held-out / cemu-variant / width-check) run
        # over those same old pkls correctly CLASHES (it would otherwise silently load self-draws).
        eff_existing = existing if existing is not None else default_cfg
        # BACK-COMPAT (2026-06-21): pkls written before the `leg` stamp was added lack that key. The
        # per-leg runs are leg-separated by --out-dir, so a leg-less pkl in a per-leg dir IS that leg —
        # don't CLASH on the missing `leg` key alone (but still enforce leg_a/cemu_variant/amp_sigma
        # exactly, so a self-draw pkl can never load into a held-out run). Prevents a halt when a
        # DESI/eBOSS straggler (launched before the stamp) is resubmitted.
        _req = dict(cfg)
        if "leg" not in eff_existing:
            _req.pop("leg", None)
        # BACK-COMPAT (2026-06-21, fold stamp): pkls written before the `fold` stamp lack that key and
        # were ALL the ensemble/fold-0 path (run_cfg.fold==0). Don't CLASH on the missing key alone
        # WHEN the current run is also fold 0 (the default ensemble path) — pop it so a fold-0 resume
        # over pre-stamp pkls still works. A --fold k>0 run carries fold=k in `cfg`, which then differs
        # from the popped (==0-equivalent) existing => it correctly CLASHES (a fold-7 single-net run
        # must never load an ensemble/fold-0 pkl: a different held-out population).
        if "fold" not in eff_existing and int(_req.get("fold", 0)) == 0:
            _req.pop("fold", None)
        # BACK-COMPAT (2026-06-21, τ₀-prior + subDLA-truth stamps): pkls written before these stamps
        # lack the keys and were ALL the DEFAULT (uniform τ₀ / un-displaced truth: 0.0 / 1.0). Don't
        # CLASH on a missing key alone WHEN the current run is at that default — pop it so a default
        # resume over pre-stamp pkls still works. A NON-default run (informative prior σ>0 or boost≠1)
        # carries a distinct value in `cfg`, which then differs from the popped (==default-equivalent)
        # existing => it correctly CLASHES (an informative/displaced run must never load a uniform/
        # un-displaced pkl: a different SBC population / mock truth).
        if "tau0_prior_sigma" not in eff_existing and float(_req.get("tau0_prior_sigma", 0.0)) == 0.0:
            _req.pop("tau0_prior_sigma", None)
        if "subdla_truth_boost" not in eff_existing and float(_req.get("subdla_truth_boost", 1.0)) == 1.0:
            _req.pop("subdla_truth_boost", None)
        # BACK-COMPAT (2026-07-04, NORC stamp): pre-stamp pkls lack res_corr_on and were ALL the pre-NORC
        # (res_corr ON) forward. Don't CLASH on the missing key alone WHEN the current run is ALSO
        # res_corr_on=True (a pre-NORC resume). A NORC run (res_corr_on=False) does NOT pop => it differs
        # from the res_corr-ON existing => it correctly CLASHES: a NORC SBC certificate must never pool a
        # pre-NORC anchored+alpha mock (ranks don't transfer across forwards).
        if "res_corr_on" not in eff_existing and bool(_req.get("res_corr_on", True)) is True:
            _req.pop("res_corr_on", None)
        # BACK-COMPAT (2026-07-07, data-nuisance forward stamps): pre-stamp pkls lack sample_res /
        # f_res_amp_sigma / metal_prior and were ALL the pre-wiring forward (no f_res float, uniform
        # scalar metals). Don't CLASH on a missing key alone WHEN the current run is ALSO at that
        # pre-wiring default. A WIRED run (sample_res=True or metal_prior=flatlog2node) does NOT pop =>
        # it differs from the pre-wiring existing => it correctly CLASHES: a wired SBC certificate must
        # never pool a pre-wiring uniform/no-f_res mock (ranks don't transfer across forwards).
        if "sample_res" not in eff_existing and bool(_req.get("sample_res", False)) is False:
            _req.pop("sample_res", None)
        if "f_res_amp_sigma" not in eff_existing and _req.get("f_res_amp_sigma", None) is None:
            _req.pop("f_res_amp_sigma", None)
        if "metal_prior" not in eff_existing and str(_req.get("metal_prior", "uniform")) == "uniform":
            _req.pop("metal_prior", None)
        if eff_existing != _req:
            raise RuntimeError(
                f"[mock {m}] config CLASH at {path}: existing pkl run_cfg={existing} "
                f"(effective {eff_existing}) != requested {cfg}. The per-mock pkl is keyed by index "
                f"only; a held-out / cemu-variant / width-check run must use a SEPARATE --out-dir "
                f"(or delete the stale pkl). Refusing to load the wrong-config mock.")
        if verbose:
            print(f"  [mock {m}] SKIP (exists, cfg match {cfg}) -> {path}")
        return rec

    # fold THREADS into run_legb's held-out path: held_out_sims(d, fold=fold) +
    # make_truth_from_sim(d, sim, fold=fold) draw THIS fold's EXCLUDED sims as the mock TRUTH (so a
    # --fold k run's truth sims match the fold-k single net the forward uses). fold=0 = back-compat.
    records = run_legb(ctx, d, n_mocks=n_mocks, mock_indices=[int(m)], return_per_mock=True,
                       leg_a=leg_a, fold=int(fold), n_warmup=n_warmup, n_samples=n_samples, seed=seed,
                       dense_mass=dense_mass, max_tree_depth=max_tree_depth, verbose=verbose,
                       inject_spec=inject_spec)
    assert len(records) == 1, f"expected 1 record for mock {m}, got {len(records)}"
    rec = records[0]
    rec["run_cfg"] = cfg            # STAMP the config so a later skip can verify it (the guard above)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, path)        # atomic on POSIX (same dir)
    if verbose:
        print(f"  [mock {m}] wrote -> {path} (n_div={rec.get('n_div', 0)}, cfg={cfg})")
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
    # PER-LEG SBC (2026-06-21): the deployed real analysis is PER-LEG, not joint. DESI and eBOSS share
    # many quasars (eBOSS QSOs are a large subset of DESI's) so a block-diagonal joint fit double-counts;
    # KODIAQ/KS is INDEPENDENT high-res echelle (no QSO overlap) — per-leg there is a deliberate
    # conservative choice forgoing joint constraining power, not a double-count fix. '--leg X' restricts
    # the likelihood to ONE survey leg; 'all' = the (validation-only) joint fit. Per-leg posteriors are
    # NOT naively combinable. Metals per leg: DESI metals_on (DR1 model); a_SiIII fit on DESI+eBOSS only.
    ap.add_argument("--leg", choices=["all", "DESI", "KS", "eBOSS"], default="all",
                    help="restrict the SBC likelihood to one survey leg (per-leg = the deployed analysis)")
    ap.add_argument("--res-corr-on", dest="res_corr_on", action="store_true",
                    help="restore the pre-NORC forward: res_corr ON + alpha_res marginalized + "
                         "KS k<=0.069 (the anchored+alpha baseline).")
    ap.add_argument("--no-res-corr-on", dest="res_corr_on", action="store_false",
                    help="NORC (DEFAULT): res_corr OFF + fix_alpha_res + KS k<=0.045 (Gate-A).")
    # (res_corr_on default False is set in the ap.set_defaults(...) below with the other run defaults)
    ap.add_argument("--single-member", action="store_true",
                    help="run on final_prod_seed0 only (cheap de-risk; NOT the production object)")
    ap.add_argument("--no-shard-pkl", dest="write_shard_pkl", action="store_false",
                    help="skip the back-compat end-of-run shard_*.pkl (per-mock pkls are the unit)")
    # HELD-OUT-SIM hook (2026-06-18): leg_a=True (default) is the SELF-DRAW production SBC (mock and
    # likelihood share the forward → emulator error CANCELS by construction). --held-out flips to
    # leg_a=False: the mock TRUTH is a held-out-sim's MEASURED power and the forward is the emulator
    # prediction → the path that CONTAINS emulator error (the honest C_emu instrument).
    ap.add_argument("--held-out", dest="leg_a", action="store_false",
                    help="leg_a=False: held-out-sim mocks (make_legb_mock + held_out_sims) — CONTAINS "
                         "emulator error (the C_emu fix gate). Default = leg_a=True self-draw.")
    # C_EMU-VARIANT hook (2026-06-18): which C_emu the FORWARD likelihood uses. Env SBC_CEMU_VARIANT
    # (or --cemu-variant) ∈ {current, fixed, oldc}:
    #   current : the deployed production C_emu (mf_emucoh ON @ the [0.0102,0.069] table, mf_shape OFF).
    #   fixed   : MODE-1 + MODE-2 fix — mf_emucoh @ the LOW-K table (covers [0.001,0.069]) AND mf_shape ON.
    #   oldc    : explicit control == 'current' (the matched OLD-C arm for the fixed-vs-control compare).
    # Threaded into build_legb_ctx and ASSERTED to propagate (no silent no-op) below.
    ap.add_argument("--cemu-variant", default=None,
                    choices=["current", "fixed", "oldc"],
                    help="C_emu the forward uses (overrides env SBC_CEMU_VARIANT). Default: env or 'current'.")
    # TRUE-LOSO HELD-OUT hook (2026-06-21): the default held-out path uses the PRODUCTION ENSEMBLE
    # (final_prod_seed*, which SAW ALL sims) on held_out_sims(fold=0) — every fold-0 sim is LOW n_s,
    # so that "held-out" run is BOTH not-truly-held-out (the net saw those sims in training) AND
    # n_s-BOUNDARY-CONFOUNDED (only the lowest-n_s edge is tested). --fold k flips to the SINGLE
    # per-fold LOSO net final_fold{k} (trained EXCLUDING fold k) tested on held_out_sims(fold=k) —
    # a TRUE held-out SBC, and sweeping k=0..7 covers the whole n_s box (the folds are n_s-sorted:
    # fold 0 = lowest n_s, fold 7 = highest). Default None = the back-compat ensemble/fold-0 path.
    ap.add_argument("--fold", type=int, choices=list(range(8)), default=None,
                    help="TRUE-LOSO held-out SBC: use the single per-fold net final_fold{k} (excl. "
                         "fold k) on held_out_sims(fold=k). Default None = production ensemble on "
                         "fold 0 (back-compat). Composes with --leg.")
    ap.add_argument("--prod-emu", action="store_true",
                    help="A/B for the LOSO-tilt diagnostic (2026-06-21): with --fold k, forward with the "
                         "PRODUCTION ENSEMBLE (saw all sims, in-sample) instead of final_fold{k}, but on "
                         "the SAME fold-k sims + SAME mf_fold=k. Only the emulator differs vs --fold k, so "
                         "the pull-vs-n_s tilt isolates the LOSO out-of-sample (extrapolation) effect.")
    ap.add_argument("--out-dir", required=True)
    ap.set_defaults(with_mf=True, with_eboss=True, res_corr_on=False, write_shard_pkl=True, leg_a=True)
    a = ap.parse_args()

    members = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
    if not members:
        raise SystemExit(f"no production ensemble checkpoints at {PROD_PREFIX}*.eqx")
    ens = [members[0]] if a.single_member else members
    # WIDTH-CHECK override (env SBC_SUBDLA_AMP_SIGMA>0): set the subDLA AMPLITUDE prior sigma/mu for the
    # referee-mandated 0.40-vs-0.20 over-dispersion pre-check. Default 0 => unchanged production prior.
    # The override threads via inference.HCD_PRIOR_FRAC_SIGMA -> build_legb_ctx (verified to propagate to
    # ctx.alpha_hcd_sigma[1]); asserted below so it can never silently no-op (the referee's #1 hazard).
    _amp_sig = float(os.environ.get("SBC_SUBDLA_AMP_SIGMA", "0"))
    if _amp_sig > 0:
        from hcd_analysis.emulator import inference as _I
        _pf = _I.HCD_PRIOR_FRAC_SIGMA
        _I.HCD_PRIOR_FRAC_SIGMA = (_pf[0], _amp_sig, _pf[2])
        print(f"[width-check] subDLA AMPLITUDE sigma/mu overridden {_pf[1]} -> {_amp_sig}")
    # INFORMATIVE-τ₀ PRIOR (env SBC_TAU0_PRIOR_SIGMA>0): replace the UNIFORM prior on the two mean-flux
    # sites (tau0_amp, dtau0) with a Kim-centered TruncatedNormal (amp center 1.0 = the injected closure
    # truth, dtau0 center 0.0), truncated to the same physical range. SBC_TAU0_DTAU0_SIGMA overrides the
    # dτ₀ width (default = the amp σ). Threaded onto the ctx via _replace (tau0_amp_gauss/dtau0_gauss)
    # at the per-leg finalization below, then ASSERTED so it can never silently no-op. Default 0 => off
    # (uniform, byte-identical).
    _tau0_prior_sig = float(os.environ.get("SBC_TAU0_PRIOR_SIGMA", "0"))
    _tau0_dtau0_sig = float(os.environ.get("SBC_TAU0_DTAU0_SIGMA", str(_tau0_prior_sig)))
    # subDLA TRUTH DISPLACEMENT (env SBC_SUBDLA_TRUTH_BOOST != 1.0): scale the mock-truth subDLA
    # incidence (alpha_hcd[1] / alpha_hcd_z[:,1]) by this factor so the truth carries a subDLA excess
    # the forward keeps at the prior pin — the subDLA sibling of the LLS-excess bias-gate arm. Threaded
    # into the inject_spec dict reaching run_legb. Default 1.0 => identity (no displacement).
    _subdla_truth_boost = float(os.environ.get("SBC_SUBDLA_TRUTH_BOOST", "1"))
    # C_EMU VARIANT resolution (CLI > env > 'current'). 'fixed' = MODE-1 (low-k emucoh table) +
    # MODE-2 (mf_shape ON); 'current'/'oldc' = the deployed C_emu. Asserted to propagate below.
    LOWK_EMUCOH_NPZ = f"{REPO}/hcd_analysis/_emulator_data/mf_cemu_emucoh_lowk.npz"
    variant = a.cemu_variant or os.environ.get("SBC_CEMU_VARIANT", "current")
    assert variant in ("current", "fixed", "oldc"), f"bad SBC_CEMU_VARIANT {variant!r}"
    a.cemu_variant = variant      # resolved value into the namespace so vars(a) records it in meta
    _fixed = (variant == "fixed")
    _emucoh_npz = LOWK_EMUCOH_NPZ if _fixed else None
    if _fixed and not os.path.exists(LOWK_EMUCOH_NPZ):
        raise SystemExit(f"--cemu-variant fixed needs {LOWK_EMUCOH_NPZ} (run build_mf_emucoh_floor "
                         f"with CEMU_OUT_NPZ pointing there)")
    print(f"[cemu-variant] {variant}  (mf_shape={'ON' if _fixed else 'off'}, "
          f"emucoh_npz={'LOWK' if _fixed else 'default[0.0102,0.069]'})  leg_a={a.leg_a}")
    # PER-LEG config (a.leg): DESI carries its DR1 metal model; each single leg fits a_SiIII; eBOSS is
    # only built when selected (or for the joint 'all'). build_legb_ctx always constructs DESI+KS, so
    # the requested single leg is sliced out of ctx.legs after the build.
    if a.leg == "all":
        _metals_on, _sample_metals, _build_eboss = a.with_eboss, a.with_eboss, a.with_eboss
        # VALIDATION-ONLY joint: f_res is a per-INSTRUMENT systematic (a shared global f_res across
        # instruments RAISES in _check_single_instrument_for_res), so it is OFF here; metals follow the
        # metal legs (flatlog2node when any metal leg is present, else uniform).
        _sample_res, _f_res_sigma = False, None
        _metal_prior = "flatlog2node" if a.with_eboss else "uniform"
    else:
        _build_eboss   = (a.leg == "eBOSS")
        _metals_on     = (a.leg == "DESI")     # metals_on param controls the DESI loader ONLY (eBOSS leg is
                                               # metals-on by its own loader default; KS metals-off)
        # Fit a_SiIII ONLY where the forward actually applies a metal factor (metals_on legs): DESI
        # (its DR1 model) and eBOSS (DR14 not SiIII-subtracted). KS=KODIAQ-HR runs metals_on=False
        # (metals live in its covariance / conservative mode) and run_real_fit deploys KS metals=False,
        # so sampling a_SiIII for KS would be an INERT, deployment-mismatched extra dim (panel 2026-06-21).
        _sample_metals = (a.leg in ("DESI", "eBOSS"))
        # The certified per-leg data-nuisance forward = the SINGLE source run_real_fit also consumes, so
        # the SBC self-draw forward provably == the real-fit forward (f_res float + flat-log 2-node metals).
        _fc = prod_forward_config(a.leg)
        _sample_res, _f_res_sigma, _metal_prior = (
            _fc["sample_res"], _fc["f_res_amp_sigma"], _fc["metal_prior"])
    # FOLD ROUTING (2026-06-21): --fold k selects the TRUE-LOSO single net final_fold{k} (the
    # else-branch of build_legb_ctx, ensemble_ckpts=None) with mf_fold=k so the MF backbone matches
    # the fold-k LF net, and held_out_sims(fold=k) supplies that fold's EXCLUDED sims (run_legb fold,
    # threaded below). --fold None keeps the production ENSEMBLE (ensemble_ckpts=ens) on mf_fold=0.
    _fold = a.fold                                    # None => ensemble/fold-0 back-compat
    if _fold is not None and a.prod_emu:
        # PROD-EMU A/B: production ensemble (saw all sims, in-sample) on fold-k sims, SAME mf_fold=k as
        # the LOSO run -> only the main emulator differs, so the tilt isolates the LOSO extrapolation.
        _build_kw = dict(ensemble_ckpts=ens, mf_fold=int(_fold))
        print(f"[fold] PROD-EMU A/B: production ENSEMBLE + mf_fold={_fold} on held_out_sims(fold={_fold}) "
              f"(in-sample emulator; vs the LOSO final_fold{_fold})")
    elif _fold is not None:
        _build_kw = dict(ckpt=f"{REPO}/checkpoints/final_fold{_fold}", mf_fold=int(_fold))
        print(f"[fold] TRUE-LOSO held-out: single net final_fold{_fold} + mf_fold={_fold} "
              f"(held_out_sims(fold={_fold}))")
    else:
        _build_kw = dict(ensemble_ckpts=ens)          # mf_fold defaults to 0 (unchanged)
    # the PRODUCTION baseline (referee: standard per-class HCD, hierarchical_hcd=False).
    ctx, d = build_legb_ctx(
        use_xclass=True,
        with_mf=a.with_mf, mf_with_floor=a.with_mf,
        res_corr_on=a.res_corr_on,                    # NORC default False (Gate-A): drop res_corr + cap KS
        mf_emucoh=True, mf_emucoh_offdiag_only=True,
        mf_emucoh_npz=_emucoh_npz,                    # None → default table; LOWK → the fix (MODE 1)
        mf_shape=_fixed,                              # MODE 2 (LF→HR resolution tilt) ON only when fixed
        with_eboss=_build_eboss, metals_on=_metals_on, sample_metals=_sample_metals,
        sample_res=_sample_res,                       # option-b f_res float (DESI/eBOSS; OFF on KS + joint)
        f_res_amp_sigma=_f_res_sigma,                 # its Normal(0,.) width (DESI 0.02 / eBOSS 0.05; None off)
        metal_prior=_metal_prior,                     # flatlog2node (Gate-C) on metal legs; uniform on KS
        hierarchical_hcd=False, **_build_kw)
    if not a.res_corr_on:
        ctx = ctx._replace(fix_alpha_res=True)        # NORC also pins the 2 alpha_res sites (now inert)
    assert ctx.res_corr_on == a.res_corr_on, "res_corr_on did not propagate to the ctx"
    assert ctx.fix_alpha_res == (not a.res_corr_on), "fix_alpha_res inconsistent with NORC state"
    print(f"[NORC] res_corr_on={ctx.res_corr_on} fix_alpha_res={ctx.fix_alpha_res} "
          f"KS_kmax={'0.045' if not a.res_corr_on else '0.069'}")
    if a.leg != "all":
        _pre = [l.name for l in ctx.legs]
        ctx = ctx._replace(legs=[l for l in ctx.legs if l.name.upper().startswith(a.leg.upper())])
        assert len(ctx.legs) == 1, f"--leg {a.leg}: expected 1 leg, got {[l.name for l in ctx.legs]} (pre={_pre})"
        print(f"[per-leg] restricted to {a.leg}: legs={[l.name for l in ctx.legs]}  "
              f"metals_on={_metals_on} sample_metals={_sample_metals}")
    # SELF-CONSISTENCY (mirror the NORC + run_real_fit asserts): the certified data-nuisance forward must
    # be WIRED so the SBC self-draw forward == the real-fit forward (not silently regressed to defaults).
    assert ctx.metal_prior == _metal_prior, "SBC metal model (flatlog2node) not wired"
    assert bool(ctx.sample_res) == bool(_sample_res), "SBC f_res float not wired"
    assert ctx.f_res_amp_sigma == _f_res_sigma, "SBC f_res prior width mismatch"
    assert tuple(ctx.metal_node_z) == (2.2, 4.2), "Gate-C metal_node_z drifted"
    if ctx.sample_res:                             # f_res is per-INSTRUMENT: single-leg + resolution_ready
        assert all(getattr(l, "resolution_ready", False) for l in ctx.legs), \
            "f_res float on a resolution_ready=False leg"
        assert len({l.name for l in ctx.legs}) == 1, "f_res float requires a single-instrument leg"
    if a.leg == "all":
        assert bool(ctx.sample_res) is False, "multi-instrument joint must not float f_res"
    # INFORMATIVE-τ₀ PRIOR ARM: when SBC_TAU0_PRIOR_SIGMA>0, swap the UNIFORM τ₀ sites for a
    # Kim-centered TruncatedNormal (center = the injected closure truth: amp 1.0, dτ₀ 0.0). Done
    # AFTER the per-leg filtering so it lands on the FINAL ctx, then ASSERTED so it can never
    # silently no-op (mirrors the SBC_SUBDLA_AMP_SIGMA assert pattern).
    if _tau0_prior_sig > 0:
        ctx = ctx._replace(tau0_amp_gauss=(1.0, _tau0_prior_sig),
                           dtau0_gauss=(0.0, _tau0_dtau0_sig))
        print(f"[tau0-prior] informative TruncatedNormal: amp N(1.0,{_tau0_prior_sig}) "
              f"dtau0 N(0.0,{_tau0_dtau0_sig}) (else uniform)")
        assert ctx.tau0_amp_gauss is not None, "SBC_TAU0_PRIOR_SIGMA NO-OP: ctx.tau0_amp_gauss is None"
        assert ctx.dtau0_gauss is not None, "SBC_TAU0_PRIOR_SIGMA NO-OP: ctx.dtau0_gauss is None"
    else:
        print(f"[tau0-prior] uniform (SBC_TAU0_PRIOR_SIGMA=0)")
    # subDLA TRUTH DISPLACEMENT ARM: when SBC_SUBDLA_TRUTH_BOOST != 1.0, thread it into the inject_spec
    # dict reaching run_legb (apply_subdla_truth_boost scales the mock-truth subDLA column). Default
    # 1.0 => no inject_spec key (byte-identical clean self-draw).
    _inject_spec = None
    if _subdla_truth_boost != 1.0:
        # NO-OP GUARD (referee MUST-FIX): the inject_spec hook in run_legb fires ONLY on the
        # leg_a=True (self-draw) branch; the held-out branch ignores inject_spec, so a displaced
        # truth under --held-out would silently run a CLEAN mock mislabeled as displaced. Forbid it.
        assert a.leg_a, ("SBC_SUBDLA_TRUTH_BOOST requires self-draw (leg_a=True); --held-out "
                         "ignores inject_spec and would silently no-op the displacement")
        _inject_spec = {"subdla_truth_boost": float(_subdla_truth_boost)}
        print(f"[subdla-truth] displacing mock-truth subDLA incidence by ×{_subdla_truth_boost}")
    else:
        print(f"[subdla-truth] no displacement (SBC_SUBDLA_TRUTH_BOOST=1.0)")
    # PROPAGATION ASSERTS (the referee's #1 hazard — a flag that silently no-ops). The emucoh/shape
    # binders live on DESI+KS; verify on DESI when it is present in the (possibly filtered) legs.
    import numpy as _np
    _desi_legs = [l for l in ctx.legs if l.name.upper().startswith("DESI")]
    if _desi_legs:
        _desi = _desi_legs[0]
        _ec = ctx.mf_emucoh_per_leg.get(_desi.name) if getattr(ctx, "mf_emucoh_per_leg", None) else None
        assert _ec is not None, "mf_emucoh_per_leg NOT set on DESI — emucoh term silently OFF"
        _msc = ctx.mf_shape_per_leg.get(_desi.name) if getattr(ctx, "mf_shape_per_leg", None) else None
        if _fixed:
            assert _msc is not None and _np.any(_np.asarray(_msc) != 0.0), \
                "fixed variant: mf_shape_per_leg NOT populated on DESI — MODE-2 silently no-op"
            # MODE-1: the low-k emucoh binder must carry coherent covariance on rows below k=0.0102.
            _klo = _np.asarray(_desi.k) < 0.0102
            _ecd = _np.diag(_np.asarray(_ec))
            assert _klo.sum() > 0 and _np.any(_ecd[_klo] > 0), \
                "fixed variant: low-k emucoh binder has NO covariance below k=0.0102 — MODE-1 silently no-op"
            print(f"[cemu-variant] VERIFIED fixed: mf_shape on DESI nonzero; emucoh covers "
                  f"{int((_ecd[_klo]>0).sum())}/{int(_klo.sum())} DESI rows below k=0.0102")
        else:
            assert getattr(ctx, "mf_shape_per_leg", None) is None, \
                "current variant: mf_shape_per_leg should be None (MODE-2 off)"
            print(f"[cemu-variant] VERIFIED current: mf_shape off; emucoh = default table")
    else:
        print(f"[cemu-variant] DESI not in legs (--leg {a.leg}); DESI-binder asserts skipped")
    if _amp_sig > 0:
        _r = float(_np.asarray(ctx.alpha_hcd_sigma)[1] / _np.asarray(ctx.alpha_hcd_mu)[1])
        assert abs(_r - _amp_sig) < 0.01, f"subDLA amp-width override NO-OP: ctx ratio {_r:.4f} != {_amp_sig}"
        print(f"[width-check] verified ctx.alpha_hcd_sigma[1]/mu[1] = {_r:.4f}")
    n_members = len(getattr(ctx.model, "members", [None]))
    n_z = len(ctx.z_global)
    idxs = [m for m in range(a.n_mocks) if m % a.n_shards == a.shard]
    _fold_s = "ensemble/fold0" if _fold is None else f"final_fold{_fold}/fold{_fold}"
    print(f"[shard {a.shard}/{a.n_shards}] mocks={idxs}  members={n_members}  fold={_fold_s}  "
          f"legs={[l.name for l in ctx.legs]}  n_z={n_z}  mf={a.with_mf} eboss={a.with_eboss}  "
          f"cemu_variant={variant} leg_a={a.leg_a}(held_out={not a.leg_a})  "
          f"res_corr_on={a.res_corr_on}(NORC={not a.res_corr_on}: res_corr off/fix_alpha_res/KS0.045) "
          f"(warmup={a.n_warmup} samples={a.n_samples} mtd={a.max_tree_depth})")

    # PER-MOCK loop: one mock at a time, checkpoint + skip after each (bounds RSS, ≤1 mock lost
    # per OOM/wall, resumable). The end-of-run shard pkl is still written for back-compat.
    # CONFIG STAMP (2026-06-19): the discriminators that change the SBC POPULATION but NOT the pkl
    # filename (index-only). _run_mock writes this into each record and asserts it matches on a skip,
    # so a held-out / cemu-variant / width-check run cannot silently load self-draw (or other-config)
    # pkls left in the same --out-dir.
    run_cfg = dict(leg_a=bool(a.leg_a), cemu_variant=str(variant), amp_sigma=float(_amp_sig),
                   leg=str(a.leg),    # per-leg discriminator (panel 2026-06-21): a KS pkl must not
                                      # silently load a DESI pkl if two legs ever share an --out-dir.
                   fold=int(a.fold) if a.fold is not None else 0,   # TRUE-LOSO discriminator
                                      # (2026-06-21): a fold-7 single-net pkl must not silently load
                                      # into a fold-0/ensemble run (different held-out population).
                   tau0_prior_sigma=float(_tau0_prior_sig),   # informative-τ₀ discriminator (2026-06-21):
                                      # an informative-prior pkl must never load into a uniform run (or
                                      # vice versa) — distinct SBC population.
                   subdla_truth_boost=float(_subdla_truth_boost),   # subDLA-displacement discriminator
                                      # (2026-06-21): a displaced-truth pkl must never load into an
                                      # un-displaced run (different mock truth).
                   res_corr_on=bool(a.res_corr_on),   # NORC discriminator (2026-07-04): a NORC pkl
                                      # (res_corr OFF + fix_alpha_res + KS 0.045) must NEVER pool with a
                                      # pre-NORC anchored+alpha pkl -- ranks don't transfer across forwards.
                   sample_res=bool(_sample_res),      # DATA-NUISANCE forward discriminators (2026-07-07,
                   f_res_amp_sigma=_f_res_sigma,      # task #4): the f_res float + flat-log 2-node metals
                   metal_prior=str(_metal_prior))     # change the SBC POPULATION, so a WIRED pkl must never
                                      # pool with a pre-wiring (uniform / no-f_res) pkl -- distinct forward.
    records = []
    for m in idxs:
        rec = _run_mock(ctx, d, m, a.out_dir, n_mocks=a.n_mocks, n_warmup=a.n_warmup,
                        n_samples=a.n_samples, max_tree_depth=a.max_tree_depth, seed=a.seed,
                        dense_mass=True, verbose=True, leg_a=a.leg_a, run_cfg=run_cfg,
                        fold=(a.fold if a.fold is not None else 0), inject_spec=_inject_spec)
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
