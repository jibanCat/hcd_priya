#!/usr/bin/env python3
"""desi_stage2_runner.py -- DESI mechanism follow-up, Stage 2: re-sample ONE stored A2c mock with a stronger sampler.

Preregistration: hcd_priya_notes/desi_mechanism_followup_2026-09/2026-09-24-DESI-STAGE2-PREREGISTRATION-v1.md (PI #25).

HOW THE MOCK IS REGENERATED IDENTICALLY. The frozen certification driver scripts/run_prod_sbc_shard.py is imported as a
module and its per-mock function ``_run_mock`` is replaced IN PROCESS by ``_stage2_mock`` below; ``main()`` is then
called with the exact A2c command line (``--shard m --n-shards 48 --n-mocks 48 --deployed-prior --leg DESI --no-shard-pkl
--metal-selfdraw --fres-selfdraw``). Everything up to the per-mock call (context, priors, NORC, Option-A truth flags,
per-leg restriction, ARM-P asserts, run_cfg stamp) is therefore the frozen code path, unmodified. Inside
``_stage2_mock`` the mock is rebuilt exactly as ``run_legb`` does for the self-draw branch (key split, truth draw,
mock legs, fiducial core) and CHECKED against the stored pkl (run_cfg equality; truth_vec, site truths,
truth_alpha_hcd_z, kept_global, dropped equality; the stored ll_true reproduced), refusing before any sampling on
any mismatch. Then:
  (R) REPLICA: the original single chain (init_to_median, base_seed from k_nuts, the divergence retry ladder,
      n_warmup 250, n_samples 600) is re-run and its thinned draws compared to the stored ones (reproduction test).
  (S) STRONG: n_chains dispersed chains (init_to_sample, chain_key = fold_in(k_nuts, chain_id) as in
      run_legb_convergence), n_warmup 1000 (4x), n_samples 1500 (PI #27; was 600 for the first pilot), dense mass, target_accept 0.9, max_tree_depth 10,
      RAW per-chain samples retained (every site, unthinned) with energy, num_steps and divergences; the convergence
      battery (rank R-hat, bulk and tail ESS, E-BFMI, tree-depth saturation) on the packed draws.
Outputs (atomic; refuses if present): <out>/stage2_mock_XXXX.pkl (raw), <out>/stage2_mock_XXXX.json (summary).
Frozen inputs are never written. Forward, priors, covariance, likelihood and the lock are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = "/home/mfho/hcd_priya"
sys.path.insert(0, REPO)
A2C_ARGV_TEMPLATE = ["--shard", "{m}", "--n-shards", "48", "--n-mocks", "48", "--deployed-prior", "--leg", "DESI",
                     "--no-shard-pkl", "--out-dir", "{scratch}", "--metal-selfdraw", "--fres-selfdraw"]
STRONG = dict(n_chains=4, n_warmup=1000, n_samples=1500, max_tree_depth=10, target_accept=0.9, dense_mass=True)   # PI #27: n_samples 600 -> 1500 (the ONLY change)
STORED = dict(n_warmup=250, n_samples=600, max_tree_depth=10, seed=20260614)
C4_SITES = ("tau0_amp", "dtau0", "k_SiIII_DESI_z1")
EXTRA_SITES = ("tau0_amp", "dtau0", "s_lls", "s_subdla", "s_dla", "f_res_amp", "f_res_slope", "f_SiIII_DESI_z0", "f_SiIII_DESI_z1", "f_SiII_DESI_z0", "f_SiII_DESI_z1", "k_SiIII_DESI_z0", "k_SiIII_DESI_z1", "k_SiII_DESI_z0", "k_SiII_DESI_z1")


class Stage2Refusal(RuntimeError):
    pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_shard_module():
    s = importlib.util.spec_from_file_location("run_prod_sbc_shard", os.path.join(REPO, "scripts", "run_prod_sbc_shard.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------------------------- #
#  identity checks (pure; testable on synthetic records)
# --------------------------------------------------------------------------------------------- #
def identity_checks(stored, run_cfg, truth_vec, site_truths, truth_alpha_hcd_z, kept_global, dropped, ll_true, rtol_ll=1e-9):
    """Compare the regenerated mock against the stored A2c record. Returns a dict of booleans; raises on any failure."""
    rep = {}
    s_cfg = dict(stored["run_cfg"]); rep["run_cfg_equal"] = (s_cfg == dict(run_cfg))
    rep["truth_vec_equal"] = bool(np.allclose(np.asarray(stored["truth_vec"], float), np.asarray(truth_vec, float), rtol=0, atol=1e-12))
    se = stored["sites_extra"]
    rep["site_truths_equal"] = bool(set(se) == set(site_truths) and all(np.isclose(float(se[k]["truth"]), float(site_truths[k]), rtol=0, atol=1e-12) for k in se))
    rep["truth_alpha_hcd_z_equal"] = bool(np.allclose(np.asarray(stored["truth_alpha_hcd_z"], float), np.asarray(truth_alpha_hcd_z, float), rtol=0, atol=1e-12))
    rep["kept_global_equal"] = bool(np.array_equal(np.asarray(stored["kept_global"]), np.asarray(kept_global)))
    rep["dropped_equal"] = (stored["dropped"] == dropped)
    rep["ll_true_equal"] = bool(abs(float(stored["ll_true"]) - float(ll_true)) <= rtol_ll * max(1.0, abs(float(stored["ll_true"]))))
    rep["ll_true_stored"] = float(stored["ll_true"]); rep["ll_true_regenerated"] = float(ll_true)
    bad = [k for k, v in rep.items() if isinstance(v, bool) and not v]
    if bad:
        raise Stage2Refusal(f"identity checks failed: {bad}")
    return rep


def compare_replica(stored_draws, replica_draws_thinned, L_stored):
    a = np.asarray(stored_draws, float); b = np.asarray(replica_draws_thinned, float)
    same_shape = (a.shape == b.shape)
    maxabs = float(np.max(np.abs(a - b))) if same_shape else None
    return dict(same_shape=same_shape, L_stored=int(L_stored), L_replica=int(b.shape[0]), max_abs_diff=maxabs,
                bit_identical=bool(same_shape and maxabs == 0.0), close_1e_8=bool(same_shape and maxabs is not None and maxabs < 1e-8))


def _vals(x):
    return np.asarray(list(x.values()) if isinstance(x, dict) else x, float)


def pilot_gate(battery, per_chain_div, n_chains=4, thresholds=dict(rhat=1.01, ess_bulk=400, ess_tail=400, ebfmi=0.3, treedepth=0.02),
               cpu_h=None, cost_cap_cpu_h=150.0, identity_ok=True, extra_sites=None):
    """Preregistered pilot gate (prereg v1.1 section 3). ``battery`` is CL.convergence_battery output (rhat / ess_bulk / ess_tail
    are DICTS keyed by parameter; ebfmi is a per-chain array). Every packed parameter must be finite and within the GREEN
    thresholds; zero divergences over all chains; E-BFMI per chain (count == n_chains). ``extra_sites`` (dict name -> dict(rhat, ess))
    adds the non-packed C4 sites (tau0_amp, dtau0, k_SiIII_DESI_z1) to the R-hat / ESS-bulk conditions (v1.1)."""
    rh, eb, et = (_vals(battery[k]) for k in ("rhat", "ess_bulk", "ess_tail"))
    bf = _vals(battery.get("ebfmi", []))
    td = float(battery.get("treedepth_sat_frac", np.nan)); nd = int(sum(int(x) for x in per_chain_div))
    finite = bool(rh.size and np.isfinite(rh).all() and np.isfinite(eb).all() and np.isfinite(et).all() and bf.size == n_chains and np.isfinite(bf).all() and np.isfinite(td))
    ex_rh = np.array([float(v["rhat"]) for v in (extra_sites or {}).values()], float); ex_es = np.array([float(v["ess"]) for v in (extra_sites or {}).values()], float)
    ex_ok = bool(np.isfinite(ex_rh).all() and np.isfinite(ex_es).all()) if ex_rh.size else True
    conds = dict(finite=finite, rhat=bool(finite and rh.max() < thresholds["rhat"]), ess_bulk=bool(finite and eb.min() >= thresholds["ess_bulk"]),
                 ess_tail=bool(finite and et.min() >= thresholds["ess_tail"]), divergences=(nd == 0), ebfmi=bool(finite and bf.min() >= thresholds["ebfmi"]),
                 treedepth=bool(finite and td < thresholds["treedepth"]),
                 c4_sites_rhat=bool(ex_ok and (ex_rh.max() < thresholds["rhat"] if ex_rh.size else True)),
                 c4_sites_ess=bool(ex_ok and (ex_es.min() >= thresholds["ess_bulk"] if ex_es.size else True)))
    sampler_ok = bool(all(conds.values()))
    cost_ok = (None if cpu_h is None else bool(cpu_h <= cost_cap_cpu_h))
    passed = bool(sampler_ok and bool(identity_ok) and (cost_ok is not False))
    return dict(passed_sampler_criteria=sampler_ok, cost_ok=cost_ok, cpu_h=cpu_h, cost_cap_cpu_h=cost_cap_cpu_h, identity_ok=bool(identity_ok),
                pilot_gate_passed=passed, conds=conds, rhat_max=(float(rh.max()) if rh.size else None), ess_bulk_min=(float(eb.min()) if eb.size else None),
                ess_tail_min=(float(et.min()) if et.size else None), ebfmi_min=(float(bf.min()) if bf.size else None), ebfmi_count=int(bf.size),
                treedepth_sat_frac=td, n_divergent=nd, thresholds=thresholds, n_chains=n_chains,
                c4_sites={k: dict(rhat=float(v["rhat"]), ess=float(v["ess"])) for k, v in (extra_sites or {}).items()})


def site_diagnostics(chains, names_extra):
    """Split-R-hat / ESS (numpyro.diagnostics, raw draws) for non-packed sampled sites across the strong chains."""
    import numpyro.diagnostics as npd
    out = {}
    for nm in names_extra:
        if all(nm in c["samples"] for c in chains):
            x = np.stack([np.asarray(c["samples"][nm], float).reshape(len(c["samples"][nm]), -1)[:, 0] for c in chains])
            out[nm] = dict(rhat=float(npd.split_gelman_rubin(x)), ess=float(npd.effective_sample_size(x)))
    return out


# --------------------------------------------------------------------------------------------- #
#  the per-mock replacement (runs inside the frozen driver's main())
# --------------------------------------------------------------------------------------------- #
def _partial(out_dir, m, tag, obj):
    """Atomic per-chain checkpoint (S1): a wall-time or OOM kill loses at most one chain."""
    d = os.path.join(out_dir, f"stage2_mock_{int(m):04d}.partial"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{tag}.pkl")
    with open(p + ".tmp", "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(p + ".tmp", p)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def make_stage2_mock(stored_pkl, out_dir, strong, do_replica=True, do_strong=True, dry_run=False, n_threads=None):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import init_to_sample
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator.closure_diagnostics import thin_to_ess

    with open(stored_pkl, "rb") as f:
        stored = pickle.load(f)

    def _stage2_mock(ctx, d, m, _out_dir, *, n_mocks, n_warmup, n_samples, max_tree_depth, seed, dense_mass=True, verbose=True,
                     leg_a=True, run_cfg=None, fold=0, inject_spec=None):
        t0 = time.time()
        if not leg_a:
            raise Stage2Refusal("Stage 2 is a self-draw (leg_a) protocol")
        if (int(n_warmup), int(n_samples), int(max_tree_depth), int(seed)) != (STORED["n_warmup"], STORED["n_samples"], STORED["max_tree_depth"], STORED["seed"]):
            raise Stage2Refusal("frozen driver settings differ from the stored A2c settings")
        # ---- rebuild the mock exactly as run_legb's self-draw branch ----
        fid_core = {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
                    for name, v in CL._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}
        key0 = jax.random.PRNGKey(int(seed))
        k_truth, k_mock, k_nuts = jax.random.split(jax.random.fold_in(key0, int(m)), 3)
        truth_pack = CL.draw_leg_a_leg_truth(ctx, k_truth)
        if inject_spec:
            truth_pack = CL._apply_truth_boosts(truth_pack, inject_spec, z=np.asarray(ctx.z_global, float))
            mock_legs, info = CL.make_leg_a_legmock(ctx, fid_core, truth_pack, k_mock, inject_metal_misspec=inject_spec.get("metal_misspec"),
                                                    inject_resolution=inject_spec.get("resolution"))
        else:
            mock_legs, info = CL.make_leg_a_legmock(ctx, fid_core, truth_pack, k_mock)
        core_per_leg = fid_core
        kept_global = truth_pack["kept_global_z"]
        truth_vec = np.concatenate([truth_pack["theta9"], truth_pack["tau0_global"][kept_global], truth_pack["alpha_hcd"]])
        _a_si_true = float(truth_pack.get("a_siiii", 0.0)); _a_si2_true = float(truth_pack.get("a_siii", 0.0))
        ll_true = float(CL._data_loglik_legcore(ctx, jnp.asarray(truth_pack["theta9"]), jnp.asarray(truth_pack["tau0_global"]),
                                                jnp.asarray(truth_pack["alpha_hcd_z"]), mock_legs, core_per_leg,
                                                a_siiii=_a_si_true, a_siii=_a_si2_true, require_zresolved=True))
        _truth_raw = truth_pack.get("raw") or {}
        site_truths = {nm: float(truth_pack.get(nm, _truth_raw.get(nm, np.nan))) for nm in stored["sites_extra"]}
        ident = identity_checks(stored, run_cfg, truth_vec, site_truths, np.array(truth_pack["alpha_hcd_z"], float), kept_global, info["dropped"], ll_true)
        fp = {}
        for leg in mock_legs:
            for attr in ("P_data", "C_data", "cov", "k", "z"):
                v = getattr(leg, attr, None)
                if v is not None:
                    fp[f"{getattr(leg, 'name', 'leg')}.{attr}"] = hashlib.sha256(np.ascontiguousarray(np.asarray(v, float)).tobytes()).hexdigest()
        ident["mock_arrays_sha256"] = fp
        summary = dict(mock=int(m), identity=ident, stored=dict(L=int(stored["L"]), n_div=int(stored["n_div"]), run_cfg=stored["run_cfg"]),
                       settings=dict(stored=STORED, strong=strong), timing={}, host=platform.node(), threads=n_threads,
                       utc_start=datetime.fromtimestamp(t0, timezone.utc).isoformat())
        if dry_run:
            summary["dry_run"] = True; summary["timing"]["identity_s"] = time.time() - t0
            return dict(summary=summary, raw=None)
        raw = dict(mock=int(m), names=None, truth_vec=truth_vec, kept_global=np.asarray(kept_global), sites_truth=site_truths)
        # ---- (R) replica of the stored single chain ----
        if do_replica:
            tr = time.time()
            base_seed = int(jax.random.randint(k_nuts, (), 0, 2 ** 31 - 1))
            ta_sched = (0.9,) + tuple(CL.DIVERGENCE_RETRY_TARGET_ACCEPT)
            samples = n_div = None; attempts = 0
            for attempt, ta in enumerate(ta_sched):
                samples, n_div = CL._run_nuts_legb(ctx, mock_legs, core_per_leg, n_warmup=n_warmup, n_samples=n_samples,
                                                   seed=base_seed + attempt, target_accept=ta, dense_mass=dense_mass, max_tree_depth=max_tree_depth)
                attempts = attempt + 1
                if n_div == 0:
                    break
            draws = CL._draws_matrix(samples, kept_global)
            draws_t, step, ess_min = thin_to_ess(draws)
            raw["replica"] = dict(samples={k: np.asarray(v) for k, v in samples.items()}, draws=draws, draws_thinned=draws_t, step=int(step),
                                  ess_min=float(ess_min), n_div=int(n_div), attempts=attempts, base_seed=base_seed)
            _partial(out_dir, m, "replica", raw["replica"])
            raw["names"] = CL._packed_names_for(samples, kept_global)
            summary["replica"] = dict(compare=compare_replica(stored["draws"], draws_t, stored["L"]), n_div=int(n_div), attempts=attempts,
                                      step=int(step), ess_min=float(ess_min), names_equal=(list(raw["names"]) == list(stored["names"])), base_seed=int(base_seed))
            summary["timing"]["replica_s"] = time.time() - tr
        # ---- (S) strong run ----
        if do_strong:
            ts = time.time()
            chains = []; energies = []; steps = []; divs = []; packed = []; init_vals = []; chain_keys = []
            for cid in range(int(strong["n_chains"])):
                chain_key = jax.random.fold_in(k_nuts, int(cid)); chain_keys.append(np.asarray(chain_key).tolist())
                samples, n_div, extra = CL._run_nuts_legb(ctx, mock_legs, core_per_leg, n_warmup=int(strong["n_warmup"]), n_samples=int(strong["n_samples"]),
                                                          seed=chain_key, target_accept=float(strong["target_accept"]), dense_mass=bool(strong["dense_mass"]),
                                                          max_tree_depth=int(strong["max_tree_depth"]), init_strategy=init_to_sample, return_extra=True)
                dr = CL._draws_matrix(samples, kept_global)
                rec_c = dict(chain=cid, samples={k: np.asarray(v) for k, v in samples.items()}, energy=np.asarray(extra["energy"]),
                             num_steps=np.asarray(extra["num_steps"]), n_div=int(n_div), diverging=np.asarray(extra.get("diverging", [])), draws=dr, chain_key=chain_keys[-1])
                chains.append(rec_c); _partial(out_dir, m, f"chain_{cid}", rec_c)
                energies.append(np.asarray(extra["energy"])); steps.append(np.asarray(extra["num_steps"])); divs.append(int(n_div)); packed.append(dr); init_vals.append(dr[0])
                if verbose:
                    print(f"  [stage2 mock {m} chain {cid}] draws={dr.shape[0]} div={n_div}", flush=True)
            packed = np.stack(packed, axis=0)
            names = CL._packed_names_for(samples, kept_global)
            battery = CL.convergence_battery(packed, names, energy=np.stack(energies), num_steps=np.stack(steps), max_tree_depth=int(strong["max_tree_depth"]), n_div=int(sum(divs)))
            init_arr = np.stack(init_vals); post_sd = packed.reshape(-1, packed.shape[-1]).std(axis=0)
            battery["init_spread"] = init_arr.std(axis=0); battery["post_sd"] = post_sd
            battery["init_spread_over_postsd"] = np.where(post_sd > 0, init_arr.std(axis=0) / post_sd, np.nan)
            raw["strong"] = dict(chains=chains, packed=packed, names=names, battery=battery, per_chain_div=divs, chain_keys=chain_keys)
            if raw["names"] is None:
                raw["names"] = names
            summary["strong"] = dict(names=list(names), per_chain_div=divs, n_chains=int(strong["n_chains"]), chain_keys=chain_keys)
            summary["timing"]["strong_s"] = time.time() - ts
        summary["timing"]["total_s"] = time.time() - t0; summary["utc_end"] = datetime.now(timezone.utc).isoformat()
        return dict(summary=summary, raw=raw)

    return _stage2_mock


def run_one(mock, stored_dir, sha_file, out_dir, do_replica=True, do_strong=True, dry_run=False):
    stored_pkl = os.path.join(stored_dir, f"mock_{int(mock):04d}.pkl")
    if not os.path.exists(stored_pkl):
        raise Stage2Refusal(f"stored pkl missing: {stored_pkl}")
    want = None
    with open(sha_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and os.path.basename(parts[1]) == os.path.basename(stored_pkl):
                want = parts[0]
    if want is None or sha256(stored_pkl) != want:
        raise Stage2Refusal("stored pkl sha256 mismatch against the manifest")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, f"stage2_mock_{int(mock):04d}.pkl"); out_json = os.path.join(out_dir, f"stage2_mock_{int(mock):04d}.json")
    if not dry_run and (os.path.exists(out_pkl) or os.path.exists(out_json)):
        raise Stage2Refusal(f"output exists: {out_pkl}")
    scratch = os.path.join(out_dir, "_driver_scratch"); os.makedirs(scratch, exist_ok=True)
    mod = _load_shard_module()
    holder = {}
    fn = make_stage2_mock(stored_pkl, out_dir, STRONG, do_replica=do_replica, do_strong=do_strong, dry_run=dry_run,
                          n_threads=os.environ.get("XLA_FLAGS"))
    def _capture(*a, **kw):
        res = fn(*a, **kw); holder["res"] = res
        return dict(names=[], draws=np.zeros((1, 1)), truth_vec=np.zeros(1), L=1, run_cfg=kw.get("run_cfg"))   # placeholder record for main()'s bookkeeping
    mod._run_mock = _capture
    argv = [x.format(m=int(mock), scratch=scratch) for x in A2C_ARGV_TEMPLATE]
    old = sys.argv; sys.argv = ["run_prod_sbc_shard.py"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old
    res = holder.get("res")
    if res is None:
        raise Stage2Refusal("the frozen driver did not reach the per-mock call")
    summ = res["summary"]; summ["stored_pkl"] = dict(path=stored_pkl, sha256=want); summ["driver_argv"] = argv
    if dry_run:
        print(json.dumps(summ["identity"], indent=1)); return summ
    with open(out_pkl + ".tmp", "wb") as f:
        pickle.dump(res["raw"], f, protocol=4)
    os.replace(out_pkl + ".tmp", out_pkl)
    summ["output_pkl_sha256"] = sha256(out_pkl)
    # derived summaries AFTER the raw chains are safely on disk (M2): any post-processing failure cannot discard them
    try:
        if res["raw"] is not None and "strong" in res["raw"]:
            st = res["raw"]["strong"]; bat = st["battery"]
            summ["strong"]["battery"] = _jsonable({k: (v if not isinstance(v, np.ndarray) else v) for k, v in bat.items()})
            extra = site_diagnostics(st["chains"], EXTRA_SITES)
            summ["strong"]["site_diagnostics_non_packed"] = extra
            cpus = float(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
            cpu_h = (summ["timing"].get("total_s", 0.0) * cpus / 3600.0) if cpus > 0 else None
            summ["strong"]["gate"] = pilot_gate(bat, st["per_chain_div"], n_chains=int(STRONG["n_chains"]), cpu_h=cpu_h,
                                                identity_ok=all(v for k, v in summ["identity"].items() if isinstance(v, bool)),
                                                extra_sites={k: extra[k] for k in C4_SITES if k in extra})
    except Exception as e:  # noqa: BLE001
        summ["strong"] = dict(summ.get("strong", {}), gate=dict(pilot_gate_passed=False, passed_sampler_criteria=False, error=repr(e)))
    with open(out_json + ".tmp", "w") as f:
        json.dump(_jsonable(summ), f, indent=1, sort_keys=True, default=str); f.write("\n")
    os.replace(out_json + ".tmp", out_json)
    print(f"stage2 mock {mock}: identity OK; replica {summ.get('replica', {}).get('compare')}; gate {summ.get('strong', {}).get('gate', {}).get('pilot_gate_passed')}")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", type=int, required=True)
    ap.add_argument("--stored-dir", default="/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/armp_DESI_corrected_v1")
    ap.add_argument("--sha-file", default="/home/mfho/hcd_priya_notes/docs/superpowers/a3c-artifacts/a2c_n48_sha256.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-replica", dest="replica", action="store_false")
    ap.add_argument("--no-strong", dest="strong", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="context build + mock regeneration + identity checks only (no sampling, no output)")
    a = ap.parse_args(argv)
    run_one(a.mock, a.stored_dir, a.sha_file, a.out_dir, do_replica=a.replica, do_strong=a.strong, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
