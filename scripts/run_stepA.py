#!/usr/bin/env python3
"""STEP-A closure-mock launcher — ROBUST, RESTARTABLE, single-node 16-core background runner.

Runs the STEP-A closure battery (23 mocks / 44 chains) as a POOL of single-thread worker
SUBPROCESSES (1 chain ≈ 1 core; ``OMP_NUM_THREADS=1`` + JAX intra/inter-op=1 +
``JAX_PLATFORMS=cpu`` — NO numpyro ``num_chains>1``, which serializes on CPU). Each chain is an
isolated subprocess (``--run-one <chain_id>``) so a crash takes down only that chain. Per-chain
checkpoints (``checkpoints/stepA/<chain_id>.npz``) make the run RESTARTABLE — a re-launch skips
any chain whose checkpoint already exists. A machine-readable health log
(``checkpoints/stepA/health.json`` + a human ``health.txt``) is updated atomically (temp+rename)
per chain transition so a watcher can poll it without ever reading a half-written file.

THE RUN (resolve exact sim names via ``held_out_sims``, closest to each target n_s):
  TIER 1 (LF, mf=False):
    L1a (convergence, 4 dispersed chains): fold0 n_s≈0.81; fold4 n_s≈0.92; fold7 n_s≈1.0  (12)
    L1b (bias, 1 chain): 2 held-out sims/fold × 8 folds spanning n_s                         (16)
  TIER 2 (MF, mf=True; matching fold's emulator + with_mf=True, mf_with_floor=True),
    4 dispersed chains each:
    M1 fold7 n_s≈1.019; M2 fold7 n_s≈1.040 (τ₀ anchor = ladder EXTREME); M3 = M1 truth with
    the HCD per-class z-slope MARGINALIZED; M4 HR-resolution truth (HR sim n_s≈0.979)         (16)

Usage:
  # print the resolved 44-chain table (no run)
  ... run_stepA.py --print-table
  # SMOKE (1 L1b single-chain + 1 fiducial's 2 chains, ~40/40): proves the launcher end-to-end
  ... run_stepA.py --smoke
  # FULL run, 14 workers, in the background (the orchestrator launches this; see bg wrapper)
  ... run_stepA.py --run --workers 14
  # one chain (used internally by the pool; also runnable by hand to debug a single chain)
  ... run_stepA.py --run-one <chain_id>

Env (MANDATORY for every process):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_stepA.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

REPO = "/home/mfho/hcd_priya"
PY = "/home/mfho/.conda/envs/emu-jax/bin/python3"
CKPT_DIR = f"{REPO}/checkpoints/stepA"
HEALTH_JSON = f"{CKPT_DIR}/health.json"
HEALTH_TXT = f"{CKPT_DIR}/health.txt"

# Production NUTS knobs (STEP-A fiducials): dense mass + mtd=10 + warmup 250 (the
# run_legb_convergence defaults). ESS target ~400 (pooled for the ≥4-chain fiducials; per-chain
# for the L1b single chains).
PROD = dict(n_warmup=250, n_samples=400, dense_mass=True, max_tree_depth=10, target_accept=0.9)
ESS_TARGET = 400
N_FOLDS = 8

# ----------------------------------------------------------------------------- #
#  Single-thread environment for every worker (1 chain ≈ 1 core).
# ----------------------------------------------------------------------------- #
def _force_single_thread_env():
    """Set the thread caps BEFORE importing jax/numpy-with-BLAS (must precede the heavy import).
    Returns the env dict (also exported into os.environ for the current process)."""
    # XLA CPU single-thread: disable the Eigen thread pool (the real intra-op knob for JAX-CPU)
    # AND force a single host device. With OMP/BLAS=1 this pins one chain ≈ one core (verified by
    # ~90% — not >100% — per-worker CPU; XLA still allocates an idle thread pool but does not run
    # it). xla_cpu_enable_fast_math left default (numerics unchanged).
    env = dict(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        XLA_FLAGS=("--xla_cpu_multi_thread_eigen=false "
                   "--xla_force_host_platform_device_count=1"),
        JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
        PYTHONNOUSERSITE="1", PYTHONPATH=REPO)
    os.environ.update(env)
    return env


# ----------------------------------------------------------------------------- #
#  CONFIG TABLE — resolve the 44 chains (sim names at build time).
# ----------------------------------------------------------------------------- #
def _ns_of_sim(d, sim, PARAM_LIMITS):
    names = np.asarray(d["sim_name"])
    rows = np.where(names == sim)[0]
    u = d["params_unit"][rows[0], 0]
    lo, hi = PARAM_LIMITS[0]
    return float(lo + u * (hi - lo))


def _closest_sim(sims_ns, target):
    """(ns, sim) pair whose ns is closest to target, from a list of (ns, sim)."""
    return min(sims_ns, key=lambda t: abs(t[0] - target))


def build_config(verbose=False):
    """Resolve the full 44-chain config (a list of dicts). Each chain dict:
      id, tier, fold, ckpt, sim, n_s, mf, z_slope_marginalized, hr_truth, tau0_extreme,
      chain_id, seed, n_chains (the fiducial pool size for the battery merge).
    Sim names are resolved here from the cache (closest held-out sim to each target n_s)."""
    import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
    from hcd_analysis.emulator.data import PARAM_LIMITS
    from hcd_analysis.emulator.closure_legb import held_out_sims, CACHE_PATH, load_cache

    d = load_cache(CACHE_PATH)
    # per-fold (ns, sim) sorted by ns.
    fold_sims = {}
    for f in range(N_FOLDS):
        sims, _ = held_out_sims(d, fold=f)
        fold_sims[f] = sorted([(_ns_of_sim(d, s, PARAM_LIMITS), s) for s in sims])

    def ckpt_of(fold):
        return f"{REPO}/checkpoints/final_fold{fold}"

    cfg = []

    def add_fiducial(mock_id, tier, fold, target_ns, *, mf, n_chains=4, sim=None,
                     z_slope_marginalized=False, hr_truth=False, tau0_extreme=False):
        if sim is None:
            ns, sim = _closest_sim(fold_sims[fold], target_ns)
        else:
            ns = _ns_of_sim(d, sim, PARAM_LIMITS)
        for c in range(n_chains):
            cfg.append(dict(
                id=f"{mock_id}_c{c}", mock_id=mock_id, tier=tier, fold=fold, ckpt=ckpt_of(fold),
                sim=sim, n_s=round(ns, 4), mf=mf, z_slope_marginalized=z_slope_marginalized,
                hr_truth=hr_truth, tau0_extreme=tau0_extreme, chain_id=c, n_chains=n_chains,
                seed=0))

    def add_single(mock_id, tier, fold, sim, *, mf=False):
        ns = _ns_of_sim(d, sim, PARAM_LIMITS)
        cfg.append(dict(
            id=mock_id, mock_id=mock_id, tier=tier, fold=fold, ckpt=ckpt_of(fold),
            sim=sim, n_s=round(ns, 4), mf=mf, z_slope_marginalized=False, hr_truth=False,
            tau0_extreme=False, chain_id=0, n_chains=1, seed=0))

    # ---- TIER 1 L1a: convergence fiducials, 4 dispersed chains each (12 chains) ----
    add_fiducial("L1a_fold0", "L1a", 0, 0.81, mf=False)
    add_fiducial("L1a_fold4", "L1a", 4, 0.92, mf=False)
    add_fiducial("L1a_fold7", "L1a", 7, 1.0, mf=False)

    # ---- TIER 1 L1b: bias, 2 held-out sims/fold × 8 folds spanning n_s, 1 chain (16 chains) --
    for f in range(N_FOLDS):
        lo_ns, lo_sim = fold_sims[f][0]
        hi_ns, hi_sim = fold_sims[f][-1]
        add_single(f"L1b_fold{f}_lo", "L1b", f, lo_sim)
        add_single(f"L1b_fold{f}_hi", "L1b", f, hi_sim)

    # ---- TIER 2 MF fiducials, 4 dispersed chains each (16 chains) ----
    # M1: fold7 n_s≈1.019 truth, through-MF forward + truth (the gate invariant).
    add_fiducial("M1", "M2tier", 7, 1.019, mf=True)
    # M2: fold7 n_s≈1.040, τ₀ anchor set to a ladder EXTREME (not the becker13 interior).
    add_fiducial("M2", "M2tier", 7, 1.040, mf=True, tau0_extreme=True)
    # M3: M1's truth (same fold7 n_s≈1.019 sim) BUT the HCD per-class z-slope MARGINALIZED.
    m1 = next(c for c in cfg if c["mock_id"] == "M1")
    add_fiducial("M3", "M2tier", 7, 1.019, mf=True, sim=m1["sim"], z_slope_marginalized=True)
    # M4: HR-resolution truth — the HR sim n_s≈0.979 (lives in fold6's held-out LF pool), the
    # make_truth_from_sim HR/MF path with the fold6 emulator + fold6 MF correction.
    hr_sim = _resolve_hr_sim(d, fold_sims, PARAM_LIMITS, target_ns=0.979)
    add_fiducial("M4", "M2tier", hr_sim["fold"], hr_sim["n_s"], mf=True, sim=hr_sim["sim"],
                 hr_truth=True)

    if verbose:
        print(f"[config] resolved {len(cfg)} chains; HR sim n_s={hr_sim['n_s']:.4f} "
              f"(fold {hr_sim['fold']})")
    return cfg


def _resolve_hr_sim(d, fold_sims, PARAM_LIMITS, target_ns=0.979):
    """The HR sim closest to target_ns (from the HR cache) + the LF fold whose held-out pool
    contains the SAME sim name (so make_truth_from_sim can read its LF cache rows for the
    MF-resolution truth). Returns dict(sim, fold, n_s)."""
    from hcd_analysis.emulator import multifidelity as MF
    hr = MF.load_cache(MF.HR_CACHE)
    hrs = np.array([s.decode() if isinstance(s, bytes) else s for s in hr["sim_name"]])
    if "params_unit" in hr:
        pu = hr["params_unit"]
    else:
        from hcd_analysis.emulator.data import normalize_params
        pu = normalize_params(hr["params"])
    lo, hi = PARAM_LIMITS[0]
    cand = []
    for s in sorted(set(hrs)):
        rows = np.where(hrs == s)[0]
        cand.append((float(lo + pu[rows[0], 0] * (hi - lo)), s))
    ns, sim = min(cand, key=lambda t: abs(t[0] - target_ns))
    # which LF fold holds this sim out?
    fold = None
    for f, lst in fold_sims.items():
        if sim in [ss for _, ss in lst]:
            fold = f
            break
    if fold is None:
        # fall back to the fold whose held-out ns is closest (the sim should be in some fold).
        raise ValueError(f"HR sim {sim!r} not found in any LF held-out fold")
    return dict(sim=sim, fold=fold, n_s=ns)


def print_table(cfg):
    print(f"\n=== STEP-A config: {len(cfg)} chains "
          f"({len(set(c['mock_id'] for c in cfg))} mocks) ===")
    hdr = (f"{'id':22s} {'tier':6s} {'fold':>4s} {'sim n_s':>8s} {'mf':>3s} "
           f"{'zslope':>6s} {'hr':>3s} {'τ0ext':>6s} {'cid':>3s}  sim")
    print(hdr)
    print("-" * len(hdr))
    for c in cfg:
        print(f"{c['id']:22s} {c['tier']:6s} {c['fold']:>4d} {c['n_s']:>8.4f} "
              f"{'Y' if c['mf'] else '.':>3s} {'Y' if c['z_slope_marginalized'] else '.':>6s} "
              f"{'Y' if c['hr_truth'] else '.':>3s} "
              f"{'Y' if c['tau0_extreme'] else '.':>6s} {c['chain_id']:>3d}  {c['sim'][:46]}")
    # per-tier counts
    from collections import Counter
    tc = Counter(c['tier'] for c in cfg)
    print(f"\nchain counts by tier: {dict(tc)}  TOTAL={len(cfg)}")


# ----------------------------------------------------------------------------- #
#  Health log — atomic (temp+rename) writes.
# ----------------------------------------------------------------------------- #
def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)        # atomic on POSIX same-fs rename


def _load_health():
    if os.path.exists(HEALTH_JSON):
        try:
            with open(HEALTH_JSON) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _render_txt(health):
    lines = [f"# STEP-A health  ({_now()})", ""]
    order = {"FAILED": 0, "running": 1, "queued": 2, "done": 3}
    chains = health.get("chains", {})
    for cid in sorted(chains, key=lambda k: (order.get(chains[k].get("status"), 9), k)):
        c = chains[cid]
        extra = ""
        if c.get("status") == "done":
            extra = (f" n={c.get('n_samples')} div={c.get('divergences')} "
                     f"mtdsat={c.get('treedepth_sat_frac')}")
        if c.get("status") == "FAILED":
            extra = f"  ERR: {str(c.get('error',''))[:120]}"
        lines.append(f"  {c.get('status','?'):8s} {cid:24s}{extra}")
    # fiducial batteries
    fids = health.get("fiducials", {})
    if fids:
        lines += ["", "# fiducial batteries (all-chains-in):"]
        for mid, b in sorted(fids.items()):
            lines.append(f"  {mid:12s} Rhat_max={b.get('rhat_max')} "
                         f"ESSbulk_min={b.get('ess_bulk_min')} "
                         f"ESStail_min={b.get('ess_tail_min')} "
                         f"ns_bias={b.get('bias_ns')} Ap_bias={b.get('bias_Ap')}")
    counts = health.get("counts", {})
    lines += ["", f"# counts: {counts}"]
    return "\n".join(lines) + "\n"


def update_health(updates_per_chain=None, fiducial=None, counts_recompute=True):
    """Merge per-chain status updates (and/or a fiducial battery) into health.json + .txt,
    ATOMICALLY. ``updates_per_chain`` = {chain_id: {field: val, ...}}. NOTE: this is called by
    the PARENT pool process serially (workers communicate via their checkpoint files + a tiny
    status sentinel) so there is a single writer — no cross-process race on health.json."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    h = _load_health()
    h.setdefault("chains", {})
    h.setdefault("fiducials", {})
    h["updated"] = _now()
    if updates_per_chain:
        for cid, fields in updates_per_chain.items():
            h["chains"].setdefault(cid, {})
            h["chains"][cid].update(fields)
    if fiducial:
        mid, battery = fiducial
        h["fiducials"][mid] = battery
    if counts_recompute:
        from collections import Counter
        cc = Counter(c.get("status", "?") for c in h["chains"].values())
        h["counts"] = dict(cc)
    _atomic_write(HEALTH_JSON, json.dumps(h, indent=2, default=_json_default))
    _atomic_write(HEALTH_TXT, _render_txt(h))
    return h


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------- #
#  RUN ONE CHAIN (the worker entrypoint; isolated subprocess, single-thread).
# ----------------------------------------------------------------------------- #
def run_one_chain(chain, *, n_warmup, n_samples, dense_mass, max_tree_depth, target_accept):
    """Build the chain's fold-matched ctx + run ONE NUTS chain reproducing the exact draw a
    monolithic run_legb_convergence would (same seed derivation), and WRITE the checkpoint.

    Reproduces run_legb_convergence's per-chain seed: key0=PRNGKey(seed);
    k_mock,k_nuts=split(fold_in(key0, mock_index)); chain_key=fold_in(k_nuts, chain_id). The mock
    NOISE key (k_mock) is shared across a fiducial's chains (same mock dataset; chains differ
    only in the NUTS seed) — exactly run_legb_convergence's contract."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import init_to_sample
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator import closure_legb as C
    from hcd_analysis.emulator.closure_legb import (
        build_legb_ctx, held_out_sims, make_truth_from_sim, make_legb_mock,
        _mock_core_per_leg, _run_nuts_legb, _draws_matrix, CACHE_PATH,
        ZSLOPE_PRIOR_SIGMA)
    from hcd_analysis.emulator.inference import PARAM_NAMES, HCD_LIT_OVER_SIM_SLOPE

    fold = chain["fold"]
    # MOCK INDEX: run_legb_convergence selects the sim by mock_index OR an explicit sim. We pass
    # sim explicitly, but the SEED stream uses mock_index — so derive a STABLE mock_index from the
    # sim's position in the fold's held-out list (so all of a mock's chains share one noise draw,
    # and distinct mocks get distinct noise). This makes the seed reproducible + per-mock unique.
    sims, _ = held_out_sims(C.load_cache(CACHE_PATH), fold=fold)
    try:
        mock_index = sims.index(chain["sim"])
    except ValueError:
        mock_index = 0

    # fold-matched emulator backbone; the error vector is the production (fold0) C_emu (the ONE
    # matched xclass pair; there is no per-fold error vector — see SESSION_HANDOVER §248).
    ctx, d = build_legb_ctx(
        ckpt=chain["ckpt"], with_mf=bool(chain["mf"]),
        mf_fold=fold, mf_with_floor=bool(chain["mf"]))

    if chain["z_slope_marginalized"]:
        ctx = ctx._replace(marginalize_zslope=True,
                           zslope_mu=jnp.asarray(HCD_LIT_OVER_SIM_SLOPE),
                           zslope_sigma=jnp.asarray(ZSLOPE_PRIOR_SIGMA))

    # TRUTH: MF-resolution (gate invariant) when mf is set; LF otherwise. τ₀ anchor = the
    # most-absorption ladder EXTREME for M2 (tau0_extreme; the rung where the τ₀×cosmology
    # interaction is hardest), else the becker13 interior anchor (the regime the data visits).
    tau0_anchor = "extreme_hi" if chain["tau0_extreme"] else "becker13"
    truth_sim = make_truth_from_sim(d, chain["sim"], fold=fold, tau0_anchor=tau0_anchor,
                                    mf=ctx.mf)

    key0 = jax.random.PRNGKey(int(chain["seed"]))
    k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, int(mock_index)), 2)
    mock_legs, truth_pack, info = make_legb_mock(ctx, truth_sim, k_mock)
    core_per_leg = _mock_core_per_leg(ctx, truth_sim)
    kept_global = truth_pack["kept_global_z"]

    chain_key = jax.random.fold_in(k_nuts, int(chain["chain_id"]))
    t0 = time.time()
    samples, n_div, extra = _run_nuts_legb(
        ctx, mock_legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
        seed=chain_key, target_accept=target_accept, dense_mass=dense_mass,
        max_tree_depth=max_tree_depth, init_strategy=init_to_sample, return_extra=True)
    wall = time.time() - t0

    draws = _draws_matrix(samples, kept_global)            # (N, P)
    tau0_names = [f"tau0_z{i}" for i in range(int(kept_global.sum()))]
    packed_names = list(PARAM_NAMES) + tau0_names + ["alpha_lls", "alpha_subdla", "alpha_dla"]
    truth_vec = np.concatenate([
        truth_pack["theta9"], truth_pack["tau0_global"][kept_global], truth_pack["alpha_hcd"]])

    num_steps = np.asarray(extra["num_steps"])
    mtd_sat = float(np.mean(num_steps >= (2 ** int(max_tree_depth) - 1))) if num_steps.size else float("nan")

    out = dict(
        chain_id=chain["id"], mock_id=chain["mock_id"], tier=chain["tier"], fold=fold,
        sim=chain["sim"], n_s=chain["n_s"], mf=bool(chain["mf"]),
        z_slope_marginalized=bool(chain["z_slope_marginalized"]),
        hr_truth=bool(chain["hr_truth"]), tau0_extreme=bool(chain["tau0_extreme"]),
        chain_index=int(chain["chain_id"]), n_chains_target=int(chain["n_chains"]),
        # battery inputs: the per-chain packed draws + the extra fields (energy/num_steps/diverg).
        packed=draws.astype(np.float64), names=np.array(packed_names),
        truth_vec=truth_vec.astype(np.float64), kept_global=kept_global,
        energy=np.asarray(extra["energy"]), num_steps=num_steps,
        diverging=np.asarray(extra["diverging"]),
        n_samples=int(draws.shape[0]), divergences=int(n_div),
        treedepth_sat_frac=mtd_sat, wall_s=float(wall),
        dropped=json.dumps(info["dropped"], default=str))
    ckpt_path = f"{CKPT_DIR}/{chain['id']}.npz"
    os.makedirs(CKPT_DIR, exist_ok=True)
    # ATOMIC write: np.savez_compressed APPENDS ".npz" if the name lacks it, so the temp name
    # MUST already end in ".npz" (else os.replace can't find what numpy actually wrote). Use a
    # ".tmp<pid>.npz" suffix → numpy writes exactly that → rename onto the final path.
    tmp = f"{CKPT_DIR}/{chain['id']}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, ckpt_path)
    return dict(n_samples=int(draws.shape[0]), divergences=int(n_div),
                treedepth_sat_frac=mtd_sat, wall_s=float(wall))


# ----------------------------------------------------------------------------- #
#  AGGREGATE a fiducial's battery once all its chains are in.
# ----------------------------------------------------------------------------- #
def aggregate_fiducial(mock_id, cfg):
    """If ALL chains of ``mock_id`` have checkpoints, load them, stack the per-chain draws +
    energy/num_steps, and compute the convergence battery + the n_s/A_p bias-z. Returns a JSON-
    serialisable battery summary dict, or None if not all chains are present yet."""
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator.closure_legb import convergence_battery
    from hcd_analysis.emulator.inference import PARAM_NAMES

    chain_ids = [c["id"] for c in cfg if c["mock_id"] == mock_id]
    paths = [f"{CKPT_DIR}/{cid}.npz" for cid in chain_ids]
    if not all(os.path.exists(p) for p in paths):
        return None
    return _battery_over_chains(chain_ids)


def _battery_over_chains(chain_ids):
    """Compute the convergence battery + n_s/A_p bias-z over an explicit list of chain ids whose
    checkpoints all exist. Shared by aggregate_fiducial (all 4 chains) + the smoke proof (2)."""
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator.closure_legb import convergence_battery
    paths = [f"{CKPT_DIR}/{cid}.npz" for cid in chain_ids]
    if not all(os.path.exists(p) for p in paths):
        return None
    packs, energies, num_steps_all, divs, names, truth_vec, kept = [], [], [], [], None, None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        packs.append(np.asarray(z["packed"]))
        energies.append(np.asarray(z["energy"]))
        num_steps_all.append(np.asarray(z["num_steps"]))
        divs.append(int(z["divergences"]))
        names = list(z["names"]) if names is None else names
        truth_vec = np.asarray(z["truth_vec"]) if truth_vec is None else truth_vec
        kept = z["kept_global"] if kept is None else kept
    # truncate to a common draw count (chains should match; guard anyway).
    nmin = min(p.shape[0] for p in packs)
    packed = np.stack([p[:nmin] for p in packs], axis=0)         # (C, N, P)
    mtd = int(PROD["max_tree_depth"])
    battery = convergence_battery(
        packed, names,
        energy=np.stack([e[:nmin] for e in energies]) if all(e.size for e in energies) else None,
        num_steps=np.stack([n[:nmin] for n in num_steps_all]) if all(n.size for n in num_steps_all) else None,
        max_tree_depth=mtd, n_div=int(sum(divs)))
    # n_s / A_p bias-z (truth − pooled-mean)/pooled-sd.
    pooled = packed.reshape(-1, packed.shape[-1])
    def bias_z(pname):
        if pname not in names:
            return None
        j = names.index(pname)
        sd = float(pooled[:, j].std())
        return float((truth_vec[j] - pooled[:, j].mean()) / sd) if sd > 0 else None
    summary = dict(
        n_chains=int(packed.shape[0]), n_draws=int(packed.shape[1]),
        rhat_max=_r4(battery["rhat_max"]), ess_bulk_min=_r1(battery["ess_bulk_min"]),
        ess_tail_min=_r1(battery["ess_tail_min"]), ebfmi_min=_r3(battery["ebfmi_min"]),
        treedepth_sat_frac=_r4(battery["treedepth_sat_frac"]), n_divergent=int(sum(divs)),
        bias_ns=_r3(bias_z("ns")), bias_Ap=_r3(bias_z("Ap")),
        ess_target=ESS_TARGET, completed=_now())
    return summary


def _r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
def _r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def _r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


# ----------------------------------------------------------------------------- #
#  POOL dispatcher (parent process; subprocess per chain; semaphore = workers).
# ----------------------------------------------------------------------------- #
def dispatch(cfg, *, workers, nuts_kwargs, smoke=False):
    """Run the chains as a pool of single-thread SUBPROCESSES (one per chain). RESTARTABLE: a
    chain whose checkpoint already exists is SKIPPED. Crash-isolated: a worker that exits non-zero
    is logged FAILED (with its stderr tail) and the pool continues. The PARENT is the SOLE writer
    of health.json (workers only write their own .npz + a stderr capture file)."""
    import subprocess

    os.makedirs(CKPT_DIR, exist_ok=True)
    # seed the health log: queued / done(skip).
    init = {}
    todo = []
    for c in cfg:
        ckpt = f"{CKPT_DIR}/{c['id']}.npz"
        if os.path.exists(ckpt):
            rec = dict(status="done", note="checkpoint exists (skipped)",
                       tier=c["tier"], mock_id=c["mock_id"])
            try:                                  # populate the human log from the existing ckpt
                z = np.load(ckpt, allow_pickle=True)
                rec.update(n_samples=int(z["n_samples"]), divergences=int(z["divergences"]),
                           treedepth_sat_frac=_r4(float(z["treedepth_sat_frac"])),
                           wall_s=round(float(z["wall_s"]), 1))
            except Exception:
                pass
            init[c["id"]] = rec
        else:
            init[c["id"]] = dict(status="queued", tier=c["tier"], mock_id=c["mock_id"])
            todo.append(c)
    update_health(updates_per_chain=init)
    print(f"[dispatch] {len(cfg)} chains total; {len(cfg)-len(todo)} already done (skipped); "
          f"{len(todo)} to run on {workers} workers", flush=True)

    base_env = dict(os.environ)
    base_env.update(_force_single_thread_env())
    nk = nuts_kwargs

    # CPU-AFFINITY pinning: env thread caps alone don't fully tame XLA/LAPACK's internal pool
    # (Cholesky leaks ~1.5 cores during dense-mass warmup), so PIN each worker to ONE core via
    # ``taskset`` — a hard guarantee of 1 chain ≈ 1 core, no oversubscription. We pin to cores
    # [0, workers) and leave the remaining cores as headroom. A small free-slot pool recycles a
    # core when its chain finishes. (If taskset is absent we fall back to env caps only.)
    have_taskset = _which("taskset")
    # PIN to the ALLOWED cpuset: a SLURM cgroup gives a NON-CONTIGUOUS set (e.g.
    # [1,2,5,6,9,10,14,18,...]); range(workers) would taskset out-of-set ids -> "Invalid
    # argument" and FAIL those chains. Use os.sched_getaffinity. (fixed 2026-06-09)
    _allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(os.cpu_count() or workers))
    free_cores = _allowed[:workers]

    def launch(chain, core):
        cmd = []
        if have_taskset and core is not None:
            cmd += ["taskset", "-c", str(core)]
        cmd += [PY, os.path.abspath(__file__), "--run-one", chain["id"],
                "--n-warmup", str(nk["n_warmup"]), "--n-samples", str(nk["n_samples"]),
                "--max-tree-depth", str(nk["max_tree_depth"]),
                "--target-accept", str(nk["target_accept"])]
        if not nk["dense_mass"]:
            cmd.append("--diag-mass")
        if smoke:
            cmd.append("--smoke-cfg")
        logf = open(f"{CKPT_DIR}/{chain['id']}.log", "wb")
        p = subprocess.Popen(cmd, env=base_env, stdout=logf, stderr=subprocess.STDOUT)
        p._logf = logf
        p._chain = chain
        p._core = core
        p._t0 = time.time()
        return p

    running = {}        # pid -> Popen
    queue = list(todo)
    completed_mocks = set()

    def on_finish(p):
        chain = p._chain
        p._logf.close()
        rc = p.returncode
        ckpt = f"{CKPT_DIR}/{chain['id']}.npz"
        if rc == 0 and os.path.exists(ckpt):
            z = np.load(ckpt, allow_pickle=True)
            update_health(updates_per_chain={chain["id"]: dict(
                status="done", tier=chain["tier"], mock_id=chain["mock_id"],
                end=_now(), wall_s=round(float(z["wall_s"]), 1),
                n_samples=int(z["n_samples"]), divergences=int(z["divergences"]),
                treedepth_sat_frac=_r4(float(z["treedepth_sat_frac"])))})
            print(f"[done] {chain['id']} ({float(z['wall_s']):.0f}s "
                  f"n={int(z['n_samples'])} div={int(z['divergences'])})", flush=True)
        else:
            tail = _tail_file(f"{CKPT_DIR}/{chain['id']}.log", 1500)
            update_health(updates_per_chain={chain["id"]: dict(
                status="FAILED", tier=chain["tier"], mock_id=chain["mock_id"],
                end=_now(), returncode=rc, error=tail)})
            print(f"[FAILED] {chain['id']} rc={rc}\n  {tail[-300:]}", flush=True)
        # if this chain's fiducial is now complete, compute + log the battery.
        mid = chain["mock_id"]
        if mid not in completed_mocks:
            summ = aggregate_fiducial(mid, cfg)
            if summ is not None:
                completed_mocks.add(mid)
                update_health(fiducial=(mid, summ))
                print(f"[battery] {mid}: Rhat_max={summ['rhat_max']} "
                      f"ESSbulk_min={summ['ess_bulk_min']} ESStail_min={summ['ess_tail_min']} "
                      f"ns_bias={summ['bias_ns']} Ap_bias={summ['bias_Ap']}", flush=True)

    while queue or running:
        while queue and len(running) < workers:
            c = queue.pop(0)
            core = free_cores.pop(0) if free_cores else None
            p = launch(c, core)
            running[p.pid] = p
            update_health(updates_per_chain={c["id"]: dict(
                status="running", tier=c["tier"], mock_id=c["mock_id"], start=_now(),
                core=core)})
            print(f"[launch] {c['id']} (pid {p.pid}, core {core}); "
                  f"running={len(running)} queued={len(queue)}", flush=True)
        # poll for any finished worker.
        done_pids = []
        for pid, p in list(running.items()):
            if p.poll() is not None:
                done_pids.append(pid)
        for pid in done_pids:
            p = running.pop(pid)
            if getattr(p, "_core", None) is not None:
                free_cores.append(p._core)          # recycle the core slot
            on_finish(p)
        if not done_pids:
            time.sleep(2.0)

    # final battery pass for any fiducial not yet aggregated (e.g. all chains were pre-existing).
    for mid in sorted(set(c["mock_id"] for c in cfg)):
        if mid in completed_mocks:
            continue
        summ = aggregate_fiducial(mid, cfg)
        if summ is not None:
            update_health(fiducial=(mid, summ))
    print("[dispatch] ALL CHAINS DONE.", flush=True)


def _which(name):
    from shutil import which
    return which(name) is not None


def _tail_file(path, n_bytes):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            f.seek(max(0, sz - n_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return "(no log)"


# ----------------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="STEP-A closure-mock launcher")
    ap.add_argument("--print-table", action="store_true", help="resolve + print the 44-chain table")
    ap.add_argument("--run", action="store_true", help="run the FULL battery (pool of workers)")
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE: 1 L1b single-chain + 1 fiducial's 2 chains @ ~40/40")
    ap.add_argument("--run-one", type=str, default=None,
                    help="(internal) run ONE chain by id + write its checkpoint")
    ap.add_argument("--workers", type=int, default=14, help="pool size (default 14; 2 cores headroom)")
    # NUTS knobs (the worker reads these; the pool forwards them).
    ap.add_argument("--n-warmup", type=int, default=PROD["n_warmup"])
    ap.add_argument("--n-samples", type=int, default=PROD["n_samples"])
    ap.add_argument("--max-tree-depth", type=int, default=PROD["max_tree_depth"])
    ap.add_argument("--target-accept", type=float, default=PROD["target_accept"])
    ap.add_argument("--diag-mass", action="store_true", help="diagonal NUTS mass (smoke/debug)")
    ap.add_argument("--smoke-cfg", action="store_true",
                    help="(internal) the worker is part of a SMOKE pool (uses the smoke subset)")
    args = ap.parse_args()

    if args.run_one is not None:
        # WORKER: single-thread env MUST be set before the heavy import (done at module import
        # via the wrapper env; re-assert here for direct invocation).
        _force_single_thread_env()
        cfg = _smoke_subset(build_config()) if args.smoke_cfg else build_config()
        chain = next((c for c in cfg if c["id"] == args.run_one), None)
        if chain is None:
            # the smoke subset may not contain it; fall back to the full config.
            chain = next((c for c in build_config() if c["id"] == args.run_one), None)
        if chain is None:
            print(f"[run-one] unknown chain id {args.run_one!r}", file=sys.stderr)
            sys.exit(2)
        try:
            r = run_one_chain(
                chain, n_warmup=args.n_warmup, n_samples=args.n_samples,
                dense_mass=not args.diag_mass, max_tree_depth=args.max_tree_depth,
                target_accept=args.target_accept)
            print(f"[run-one] {args.run_one} OK: {r}")
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return

    _force_single_thread_env()
    cfg = build_config(verbose=True)

    if args.print_table:
        print_table(cfg)
        return

    nuts_kwargs = dict(n_warmup=args.n_warmup, n_samples=args.n_samples,
                       dense_mass=not args.diag_mass, max_tree_depth=args.max_tree_depth,
                       target_accept=args.target_accept)

    if args.smoke:
        smoke_cfg = _smoke_subset(cfg)
        # smoke NUTS: tiny warmup/samples, diagonal mass + capped tree depth (path proof, fast).
        nuts_kwargs.update(n_warmup=40, n_samples=40, dense_mass=False, max_tree_depth=7)
        print("\n=== STEP-A SMOKE ===")
        print_table(smoke_cfg)
        print(f"\n[smoke] NUTS knobs: {nuts_kwargs}; workers={min(args.workers, len(smoke_cfg))}")
        dispatch(smoke_cfg, workers=min(args.workers, len(smoke_cfg)),
                 nuts_kwargs=nuts_kwargs, smoke=True)
        _print_smoke_report(smoke_cfg)
        return

    if args.run:
        print_table(cfg)
        dispatch(cfg, workers=args.workers, nuts_kwargs=nuts_kwargs)
        return

    ap.print_help()


def _smoke_subset(cfg):
    """1 L1b single-chain + 1 fiducial's first 2 chains (the cheapest end-to-end proof)."""
    l1b = next(c for c in cfg if c["tier"] == "L1b")
    fid = [c for c in cfg if c["mock_id"] == "L1a_fold0"][:2]   # 2 of the 4 chains
    return [l1b] + fid


def _print_smoke_report(smoke_cfg):
    print("\n========== SMOKE RESULT ==========")
    h = _load_health()
    ok_ckpt = []
    for c in smoke_cfg:
        p = f"{CKPT_DIR}/{c['id']}.npz"
        ok_ckpt.append((c["id"], os.path.exists(p)))
    print("checkpoints written:")
    for cid, ok in ok_ckpt:
        if ok:
            z = np.load(f"{CKPT_DIR}/{cid}.npz", allow_pickle=True)
            print(f"  {cid:24s} OK  (n={int(z['n_samples'])} div={int(z['divergences'])} "
                  f"wall={float(z['wall_s']):.0f}s packed={np.asarray(z['packed']).shape})")
        else:
            print(f"  {cid:24s} MISSING")
    print(f"\nhealth.json counts: {h.get('counts')}")
    print(f"health.json valid JSON: {os.path.exists(HEALTH_JSON)}  "
          f"(fiducials logged: {list(h.get('fiducials', {}).keys())})")

    # PROVE the battery path on the 2 smoke L1a chains (the full gate needs all 4; the smoke runs
    # 2, so aggregate them directly here as a path proof — same convergence_battery code).
    l1a = [c for c in smoke_cfg if c["mock_id"] == "L1a_fold0"]
    if len(l1a) >= 2 and all(os.path.exists(f"{CKPT_DIR}/{c['id']}.npz") for c in l1a):
        b = _battery_over_chains([c["id"] for c in l1a])
        if b is not None:
            print(f"\n[battery PROOF on {len(l1a)} smoke chains of L1a_fold0]:")
            print(f"  Rhat_max={b['rhat_max']}  ESSbulk_min={b['ess_bulk_min']}  "
                  f"ESStail_min={b['ess_tail_min']}  ns_bias={b['bias_ns']}σ  "
                  f"Ap_bias={b['bias_Ap']}σ  (smoke depth → values indicative only)")
    print("\nTo launch the FULL run (14 workers) in the BACKGROUND:")
    print(f"  bash {REPO}/scripts/run_stepA_bg.sh")
    print(f"  (or: nohup env PYTHONNOUSERSITE=1 PYTHONPATH={REPO} JAX_PLATFORMS=cpu "
          f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 {PY} "
          f"{REPO}/scripts/run_stepA.py --run --workers 14 "
          f"> {CKPT_DIR}/run_stepA.out 2>&1 &)")


if __name__ == "__main__":
    main()
