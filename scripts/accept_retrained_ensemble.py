#!/usr/bin/env python3
"""The acceptance of a retrained ensemble before it is exported: the pre-registered member checks.

For every member checkpoint prefix (``.eqx``, ``.meta.json``, ``.norm.pkl``, ``.hist.json``):
  * the meta records a full commit (``git_sha``, 40 hex), the same for every member, and ``code_dirty`` false;
  * ``cache_sha256`` is the release cache's (``--cache-sha256``) and ``cache_lineage_sha256`` the digest of the lineage
    record stored in that cache file (read from the file whose sha256 is checked; the record is parsed and checked);
  * ``cache_path`` is a file name, the seeds are distinct, the recipe and architecture are the same for every member,
    and every member is a production member by the metadata rule (``eligibility.production_recipe_reasons``: the
    recipe declares ``all_sims`` true, no reserved held-out or fold key anywhere, no research marking;
    ``trained_on_every_simulation``), declaring its split sizes (``training_set_declared_complete``);
  * the content of the declared full training set (``eligibility.production_content_check`` on the release cache and
    the w_c map ``--wc-map``, pinned by ``--wc-map-sha256``): the production split recomputed from THIS cache's rows
    gives the recipe's ``n_train`` / ``n_val`` and covers every simulation (``split_recomputed_equal``), every
    member's normaliser equals ``train._fit_norm`` on the recomputed training rows bitwise
    (``normaliser_fingerprint_equal``), and the w_c map equals its refit on the cache (``wc_map_refit_equal``); the
    environment of the check is recorded (``content_check_environment``) with the component outcomes
    (``content_checks``);
  * the basename and directory are not the historical ensembles' (``--historical-prefix``, repeatable);
  * the comparison is like for like: the old ensemble's recipe (in each old history) equals the new members' recipe,
    and the old training cache (``--old-source``, pinned by ``--old-source-sha256``) and the new cache hold the same
    rows in the same order (``sim_name``, ``z_grid`` and ``target_F``, hence tau0, equal row by row), so the fixed row
    split (``data.production_row_split``) selects the same validation rows; its sizes must be the recipe's
    ``n_train`` / ``n_val`` when the recipe states them;
  * training completed under the recipe's stopping rule, as ``train.train_fold`` applies it: finite losses, at most the
    recipe's ``epochs``, and either all of them or a stop exactly ``patience`` epochs after the restored epoch (the last
    improvement, by more than 1e-9, of the stop metric val_resid_loss + w_coh val_coh_loss, or val_resid_loss when
    w_coh is 0);
  * the validation error is not worse than the old ensemble's beyond its seed spread, every value taken at each
    member's restored epoch (the saved model; ``restored_epoch``), for the old and the new members alike: pass iff
    mean(new val_resid_loss) <= mean(old val_resid_loss) + std(old val_resid_loss, ddof 1), with the old ensemble's
    histories given by ``--old-hist`` (repeatable). val_loss, val_coh_loss and the stop metric at the restored epoch,
    and the old-to-new change, are reported only. The old threshold is printed.

Writes a JSON report; exit 0 when every check passes, 1 otherwise.

    python scripts/accept_retrained_ensemble.py --member DIR/NAME_seed0 [...] --cache CACHE.h5 --cache-sha256 HEX \\
        --wc-map WC_MAP.json --wc-map-sha256 HEX \\
        --old-hist OLD_seed0.hist.json [...] --old-source OLD_CACHE.h5 --old-source-sha256 HEX \\
        --historical-prefix OLD_DIR/prod_repaired --out REPORT.json

The report format is ``hcd_priya.ensemble_acceptance/2`` (``/1`` plus the content-check entries). The map's record
names the cache's FILE NAME and the map check compares it: ``--cache`` must carry that name (a renamed copy of the
right cache fails ``wc_map_refit_equal``).
"""

import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.catalog import RELEASE_FINDER_IDS  # noqa: E402
from hcd_analysis.emulator import eligibility as E  # noqa: E402
from hcd_analysis.emulator.data import production_row_split  # noqa: E402

FORMAT = "hcd_priya.ensemble_acceptance/2"

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def cache_record_digest(path, sha256):
    """The digest of the lineage record stored in the cache file whose sha256 is ``sha256`` (record parsed, checked
    and eligible; the payload itself was checked by the strict loader when the members were trained).
    """
    import h5py

    if L.sha256_file(path) != sha256:
        raise SystemExit(f"refusing: {path} does not hash to {sha256}")
    with h5py.File(path, "r") as f:
        rec = L.check_record(L.parse_record(f["lineage_json"].asstr()[()]), RELEASE_FINDER_IDS)
    if rec["kind"] != "tau0_cache" or not rec["release_eligible"]:
        raise SystemExit(f"refusing: {path} does not hold an eligible tau0_cache record")
    return L.record_sha256(rec)


def restored_epoch(hist, w_coh):
    """The epoch ``train.train_fold`` restores: the last strict improvement (by more than 1e-9) of its stop metric
    val_resid_loss + w_coh val_coh_loss (val_resid_loss when w_coh is 0). Returns ``(epoch or None, stop metric)``.
    """
    stop = np.asarray(hist.get("val_resid_loss", []), dtype=float)
    if w_coh > 0:
        stop = stop + float(w_coh) * np.asarray(hist.get("val_coh_loss", []), dtype=float)
    best_metric, r = np.inf, None
    for i, m in enumerate(stop):
        if m < best_metric - 1e-9:
            best_metric, r = m, i
    return r, stop


def stopped_by_the_rule(hist, recipe):
    """Whether the history ended as ``train_fold`` ends: finite losses, at most ``epochs``, and either all epochs or a
    stop ``patience`` epochs after the restored epoch."""
    try:
        epochs, patience, w = (
            int(recipe["epochs"]),
            int(recipe["patience"]),
            float(recipe.get("w_coh", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return False
    keys = ("train_loss", "val_loss", "val_resid_loss") + (("val_coh_loss",) if w > 0 else ())
    vals = [np.asarray(hist.get(k, []), dtype=float) for k in keys]
    n = vals[0].size
    if n == 0 or any(v.size != n or not np.all(np.isfinite(v)) for v in vals):
        return False
    r, _ = restored_epoch(hist, w)
    return r is not None and n <= epochs and (n == epochs or n - 1 - r == patience)


ROW_KEYS = ("sim_name", "z_grid", "target_F")


def same_rows(old_path, old_sha256, new_path):
    """``(ok, info)``: the old source (hashing to ``old_sha256``) and the new cache hold the same rows in the same order
    (``ROW_KEYS`` equal row by row, the floats bitwise)."""
    import h5py

    if L.sha256_file(old_path) != old_sha256:
        raise SystemExit(f"refusing: {old_path} does not hash to {old_sha256}")
    cols = []
    for path in (old_path, new_path):
        with h5py.File(path, "r") as f:
            cols.append(
                {k: (f[k].asstr()[...] if k == "sim_name" else f[k][...]) for k in ROW_KEYS}
            )
    o, n = cols
    differing = {
        k: (int(np.sum(o[k] != n[k])) if o[k].shape == n[k].shape else "shape") for k in ROW_KEYS
    }
    ok = all(v == 0 for v in differing.values())
    return ok, {
        "n_rows": [int(o["sim_name"].size), int(n["sim_name"].size)],
        "rows_differing": differing,
        "n_sim_name": n["sim_name"],
    }


DECIDING = "val_resid_loss"
REPORTED = ("val_loss", "val_coh_loss", "stop_metric")


def at_restore(hist, recipe):
    """``{epoch, val_resid_loss, val_loss, val_coh_loss, stop_metric}`` of the saved model (the restored epoch), or
    None when the history does not give one."""
    try:
        w = float(recipe.get("w_coh", 0.0))
    except (TypeError, ValueError, AttributeError):
        return None
    r, stop = restored_epoch(hist, w)
    if r is None:
        return None
    out = {"epoch": int(r), "stop_metric": float(stop[r])}
    for k in ("val_resid_loss", "val_loss", "val_coh_loss"):
        v = np.asarray(hist.get(k, []), dtype=float)
        out[k] = float(v[r]) if v.size > r else None
    return out


def trained_on_every_simulation(meta, sims=None):
    """True iff the member is a production member by the metadata rule (``eligibility.production_recipe_reasons``
    empty: the recipe declares ``all_sims`` true, no reserved held-out or fold key anywhere in the meta, no simulation
    subset named anywhere (``sims``: the cache's simulation names), no research marking). It takes the member's whole
    META (it used to take the recipe): anything without a ``recipe`` key is a TypeError, so an old-style call cannot
    silently return False."""
    if not isinstance(meta, dict) or "recipe" not in meta:
        raise TypeError(
            "trained_on_every_simulation takes a member meta (with its 'recipe'), not a recipe"
        )
    return not E.production_recipe_reasons(meta, sims=sims)


def declared_complete(meta):
    """The recipe declares the full training set in a form a recomputation can check: ``all_sims`` true and the split
    sizes ``n_train`` and ``n_val``."""
    r = meta.get("recipe") if isinstance(meta.get("recipe"), dict) else {}
    return (
        r.get("all_sims") is True
        and isinstance(r.get("n_train"), int)
        and isinstance(r.get("n_val"), int)
        and not isinstance(r.get("n_train"), bool)
        and not isinstance(r.get("n_val"), bool)
    )


def content_checks(members, a):
    """The production content rules on the members' checkpoints (``train.load_checkpoint``) against the cache and
    the w_c map (both hashed to their pinned digests)."""
    from hcd_analysis.emulator import train as T
    from hcd_analysis.emulator import wc_map as W

    if L.sha256_file(a.wc_map) != a.wc_map_sha256:
        raise SystemExit(f"refusing: {a.wc_map} does not hash to {a.wc_map_sha256}")
    wcm = W.read_wc_map(a.wc_map, expected_sha256=a.wc_map_sha256)
    loaded = [T.load_checkpoint(p) for p, _, _ in members]
    return E.production_content_check(
        [m for _, m, _ in loaded],
        [n for _, _, n in loaded],
        wcm,
        cache_path=a.cache,
        expected_cache_sha256=a.cache_sha256,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cache-sha256", required=True)
    ap.add_argument(
        "--wc-map",
        required=True,
        help="the w_c map the bundle will carry (scripts/refit_wc_map.py)",
    )
    ap.add_argument("--wc-map-sha256", required=True)
    ap.add_argument("--old-hist", action="append", required=True)
    ap.add_argument(
        "--old-source", required=True, help="the old ensemble's training cache (rows compared)"
    )
    ap.add_argument("--old-source-sha256", required=True)
    ap.add_argument("--historical-prefix", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing: {a.out} exists")
    rec_sha = cache_record_digest(a.cache, a.cache_sha256)
    checks, members = {}, []
    hist_names = {os.path.basename(p) for p in a.historical_prefix}
    hist_dirs = {os.path.realpath(os.path.dirname(p)) for p in a.historical_prefix}
    for p in a.member:
        meta = json.load(open(p + ".meta.json"))
        hist = json.load(open(p + ".hist.json"))
        members.append((p, meta, hist))
    metas = [m for _, m, _ in members]
    m0 = metas[0]
    checks["git_sha_full_and_one"] = (
        all(isinstance(m.get("git_sha"), str) and _HEX40.match(m["git_sha"]) for m in metas)
        and len({m["git_sha"] for m in metas}) == 1
    )
    checks["code_clean"] = all(m.get("code_dirty") is False for m in metas)
    checks["cache_sha256_is_the_release_cache"] = all(
        m.get("cache_sha256") == a.cache_sha256 for m in metas
    )
    checks["cache_lineage_record"] = all(m.get("cache_lineage_sha256") == rec_sha for m in metas)
    checks["cache_path_is_a_file_name"] = all(
        isinstance(m.get("cache_path"), str) and os.sep not in m["cache_path"] for m in metas
    )
    checks["seeds_distinct"] = len({m.get("seed") for m in metas}) == len(metas)
    checks["one_recipe_and_architecture"] = all(
        m.get("recipe") == m0.get("recipe") and m.get("arch_cfg") == m0.get("arch_cfg")
        for m in metas
    )
    stems = {re.sub(r"_seed\d+\Z", "", os.path.basename(p)) for p in a.member}
    checks["distinct_basename_and_directory"] = not (
        stems & {re.sub(r"_seed\d+\Z", "", n) for n in hist_names}
    ) and not ({os.path.realpath(os.path.dirname(p)) for p in a.member} & hist_dirs)
    checks["training_completed_under_the_stopping_rule"] = all(
        stopped_by_the_rule(hist, meta.get("recipe") or {}) for _, meta, hist in members
    )
    old = [json.load(open(h)) for h in a.old_hist]
    old_recipes = [h.get("recipe") for h in old]
    checks["recipe_equal_to_the_old_ensemble"] = all(
        r == m0.get("recipe") for r in old_recipes
    ) and all(m.get("recipe") == m0.get("recipe") for m in metas)
    rows_ok, rows = same_rows(a.old_source, a.old_source_sha256, a.cache)
    names = rows.pop("n_sim_name")
    sims = sorted(
        set(np.asarray(names).astype(str).tolist())
    )  # simulation_sets/1 with the cache's names
    checks["trained_on_every_simulation"] = all(
        trained_on_every_simulation(m, sims=sims) for m in metas
    )
    r0 = m0.get("recipe") or {}
    try:
        tr, va, _held = production_row_split(
            names, val_seed=int(r0["val_seed"]), val_frac=float(r0["val_frac"])
        )
        sizes_ok = all(
            r0.get(k) is None or int(r0[k]) == v
            for k, v in (("n_train", tr.size), ("n_val", va.size))
        )
        rows.update(n_train=int(tr.size), n_val=int(va.size))
    except (KeyError, TypeError, ValueError) as e:
        sizes_ok = False
        rows["split_error"] = f"{type(e).__name__}: {e}"
    checks["same_rows_and_validation_split_as_the_old_ensemble"] = bool(rows_ok and sizes_ok)
    new_at = [at_restore(h, meta.get("recipe") or {}) for _, meta, h in members]
    old_at = [at_restore(h, h.get("recipe") or {}) for h in old]
    usable = (
        None not in new_at
        and None not in old_at
        and len(old_at) > 1
        and all(np.isfinite(x[DECIDING]) for x in new_at + old_at if x[DECIDING] is not None)
        and all(x[DECIDING] is not None for x in new_at + old_at)
    )
    validation = {
        "rule": f"mean(new {DECIDING}) <= mean(old {DECIDING}) + std(old {DECIDING}, ddof 1), each at the "
        "member's restored epoch; the other metrics are reported only",
        "new_at_restore": new_at,
        "old_at_restore": old_at,
    }
    if usable:
        o = np.array([x[DECIDING] for x in old_at])
        n = np.array([x[DECIDING] for x in new_at])
        mo, so, mn = float(o.mean()), float(o.std(ddof=1)), float(n.mean())
        validation.update(
            old_mean=mo,
            old_std=so,
            old_threshold=mo + so,
            new_mean=mn,
            new_std=float(n.std(ddof=1)) if n.size > 1 else None,
            change_new_minus_old=mn - mo,
            change_in_old_std=(mn - mo) / so if so else None,
            ok=bool(mn <= mo + so),
        )
        validation["reported"] = {}
        for k in REPORTED:
            ov = [x[k] for x in old_at]
            nv = [x[k] for x in new_at]
            if None not in ov and None not in nv:
                validation["reported"][k] = {
                    "old_mean": float(np.mean(ov)),
                    "new_mean": float(np.mean(nv)),
                    "change_new_minus_old": float(np.mean(nv) - np.mean(ov)),
                }
    else:
        validation.update(
            ok=False,
            error="a history gives no restored epoch or no finite value, or the old ensemble "
            "has fewer than two members",
        )
    checks[f"validation_{DECIDING}_at_restore_within_old_seed_spread"] = bool(validation["ok"])
    checks["training_set_declared_complete"] = all(declared_complete(m) for m in metas)
    cc = content_checks(members, a)
    oc = cc.outcomes()
    checks["split_recomputed_equal"] = oc.get("cache") == E.PASS and oc.get("split") == E.PASS
    checks["normaliser_fingerprint_equal"] = oc.get("normaliser") == E.PASS
    checks["wc_map_refit_equal"] = oc.get("wc_map") == E.PASS
    passed = all(checks.values())
    report = {
        "format": FORMAT,
        "passed": passed,
        "checks": checks,
        "content_check_environment": cc["environment"],
        "content_checks": cc["components"],
        "members": [os.path.basename(p) for p in a.member],
        "cache_sha256": a.cache_sha256,
        "cache_lineage_sha256": rec_sha,
        "git_sha": m0.get("git_sha"),
        "validation": validation,
        "rows": rows,
        "old_source_sha256": a.old_source_sha256,
    }
    with open(a.out, "x") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    if "old_threshold" in validation:
        print(
            f"  {DECIDING} at the restored epoch: old mean {validation['old_mean']:.6f} + std "
            f"{validation['old_std']:.6f} = threshold {validation['old_threshold']:.6f}; new mean "
            f"{validation['new_mean']:.6f} (change {validation['change_new_minus_old']:+.6f})"
        )
        for k, v in validation["reported"].items():
            print(f"  (reported only) {k}: old {v['old_mean']:.6g}, new {v['new_mean']:.6g}")
    print(f"verdict: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
