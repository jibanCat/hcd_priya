#!/usr/bin/env python3
"""Export trained RESEARCH members (cross-validation fold models, full-simulation research models) to a digest-anchored
research bundle (format ``hcd_priya.h5/2`` with ``provenance.research``; docs/release_format_h5.md and
docs/tutorials/production_and_research_bundles.md). A research bundle is never a production model: every production
loader refuses it, and it loads only through ``safe_h5.load_research`` / ``predictor.load_research_bundle``.

    python scripts/export_research_bundle.py --out NEW_DIR --purpose "TEXT" --split-rule RULE \\
        [--held-out-sim NAME ...] [--fold K] [--folds-digest HEX] [--fold-file FILE --fold-file-sha256 HEX] \\
        [--cv-set-id ID] [--description TEXT] \\
        --member DIR/NAME_seed0 [--member ...] --sha256sums SUMS --sha256sums-sha256 HEX \\
        --cache CACHE.h5 --cache-sha256 HEX

The declared split is given explicitly (``--split-rule`` ``multi_holdout_row_split/1``, ``production_row_split/1`` or
``unverifiable``, the held-out simulations, the fold and the fold-file digest) and is checked against every member's
recipe and recomputed from the cache; it is never taken from a label alone. Before anything is written the export
refuses: a dirty tree or an unknown commit; member files that do not hash to the pinned SHA256SUMS (listing each file
by name; checked before anything is unpickled); a cache that does not hash to its digest or that the release reader
refuses; members not trained on that cache at one clean commit (the lineage ties); members the declared split does not
admit (``eligibility.research_member_reasons``: a fold member's held-out set, fold, fold digest and split sizes must be
the declared ones; a member that declares no exclusion must carry ``recipe.research`` naming the purpose); a content
check that is MISMATCH (``eligibility.research_content_check``: the split recomputed under its rule, every normaliser
tensor refitted on the training rows, and the w_c map refitted on the TRAINING simulations only, so the held-out
simulations never enter it). UNAVAILABLE is allowed and recorded.

The w_c map is refitted here (``wc_map.refit_subset``) on the cache minus the declared held-out simulations and written
with its lineage record (which names ``--cache``'s file name). Write order: members and map, then the content checks
again on the files read back, then the manifest (``provenance.research`` with the outcomes). After writing: the research
loader and the research predictor load the bundle anchored on its digest, the members' leaves, normaliser and meta equal
the checkpoints', the ensemble P_filt is bitwise the checkpoints', and every production loader refuses the bundle.
Evidence goes to ``<out>/evidence``. Exit status: 0 PASS, 1 FAIL, 2 refused (nothing written)."""
import argparse
import datetime
import json
import os
import platform
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.emulator import eligibility as E  # noqa: E402
from hcd_analysis.emulator import safe_h5 as H  # noqa: E402
from hcd_analysis.emulator import safe_io as S  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.data import Z_LIMITS  # noqa: E402
from hcd_analysis.emulator.ensemble import EnsembleEmulator  # noqa: E402
from hcd_analysis.emulator.predict import predict_P_filt  # noqa: E402

N_INPUTS, SEED = 16, 20261102
EXTS = (".eqx", ".norm.pkl", ".meta.json")


class Refused(RuntimeError):
    """An input the export refuses; nothing is written."""


def read_sums(path):
    """{file name: sha256} of a ``sha256sum`` listing (``<hex>  <name>`` or ``<hex> *<name>``)."""
    out = {}
    for i, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise Refused(f"{path}:{i}: not a sha256sum line")
        name = os.path.basename(parts[1].strip().lstrip("*"))
        if name in out:
            raise Refused(f"{path}: {name} is listed twice")
        out[name] = parts[0].lower()
    return out


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _tree_same(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(_tree_same(a[k], b[k])
                                                                                       for k in a)
    return _same(a, b)


def _restored(hist, recipe):
    """The epoch train.train_fold restores (the last improvement by more than 1e-9 of val_resid_loss + w_coh
    val_coh_loss) and the validation losses there; None when the history does not give one."""
    try:
        w = float((recipe or {}).get("w_coh", 0.0))
        stop = np.asarray(hist.get("val_resid_loss", []), dtype=float)
        if w > 0:
            stop = stop + w * np.asarray(hist.get("val_coh_loss", []), dtype=float)
    except (TypeError, ValueError):
        return None
    best, r = np.inf, None
    for i, m in enumerate(stop):
        if m < best - 1e-9:
            best, r = m, i
    if r is None or not np.isfinite(best):
        return None
    out = {"epoch": int(r), "stop_metric": float(stop[r])}
    for k in ("val_resid_loss", "val_loss"):
        v = np.asarray(hist.get(k, []), dtype=float)
        out[k] = float(v[r]) if v.size > r and np.isfinite(v[r]) else None
    return out


def declared_research(a):
    """The research marking the command line declares (before the content checks are added)."""
    held = sorted(set(a.held_out_sim or []))
    if a.split_rule == E.RULE_UNVERIFIABLE:
        split = {"held_out_sims": held, "description": a.description or ""}
    else:
        split = {"held_out_sims": held}
    return {"schema": E.RESEARCH_SCHEMA, "purpose": a.purpose, "split_rule": a.split_rule, "split": split,
            "folds_digest": a.folds_digest, "cv_set_id": a.cv_set_id, "content_checks": [],
            "wc_map": {"rule": E.RULE_WCMAP_SUBSET if held else E.RULE_WCMAP, "fitted_on_sims_sha256": None,
                       "n_sims": 1}}


def check_inputs(a):
    """Every check before anything is written. Returns (prefixes, checkpoints, metas, cache dict, cache identity,
    code, the member files' sha256, the research marking with its split filled in from the members)."""
    from hcd_analysis import tau0_cache as TC
    from hcd_analysis.emulator.data import load_cache
    code = L.code_identity()
    if not (isinstance(code.get("commit"), str) and len(code["commit"]) == 40) or code.get("dirty") is not False:
        raise Refused(f"the research export runs from a clean, committed checkout only (code identity {code})")
    if not a.purpose.strip() or a.purpose != a.purpose.strip():
        raise Refused("--purpose must be non-empty text without surrounding space")
    if a.split_rule == E.RULE_EXPLICIT_ROWS:
        raise Refused(f"{E.RULE_EXPLICIT_ROWS} needs trainers that record the row-index digests; not supported yet")
    if a.split_rule == E.RULE_UNVERIFIABLE and not (a.description or "").strip():
        raise Refused("an unverifiable split needs --description")
    if (a.fold_file is None) != (a.fold_file_sha256 is None):
        raise Refused("--fold-file and --fold-file-sha256 go together")
    fold_file_sha = None
    if a.fold_file is not None:
        fold_file_sha = L.sha256_file(a.fold_file)
        if fold_file_sha != a.fold_file_sha256:
            raise Refused(f"{a.fold_file} does not hash to {a.fold_file_sha256}")
        if a.folds_digest is not None and a.folds_digest != fold_file_sha:
            raise Refused("--folds-digest is not the fold file's sha256")
        a.folds_digest = fold_file_sha                       # the verified fold file gives the folds digest
    if L.sha256_file(a.sha256sums) != a.sha256sums_sha256:
        raise Refused(f"{a.sha256sums} does not hash to {a.sha256sums_sha256}; nothing was read")
    sums = read_sums(a.sha256sums)
    prefixes = [os.path.abspath(p) for p in a.member]
    if len(set(map(os.path.basename, prefixes))) != len(prefixes):
        raise Refused("two members share a name")
    files = {}
    for p in prefixes:
        for e in EXTS:
            name = os.path.basename(p) + e
            if name not in sums:
                raise Refused(f"{name} is not listed in {a.sha256sums}")
            got = L.sha256_file(p + e)
            if got != sums[name]:
                raise Refused(f"{p + e} does not match {a.sha256sums} (nothing unpickled)")
            files[name] = got
    metas = [json.load(open(p + ".meta.json")) for p in prefixes]
    ident = L.file_identity(a.cache)
    if ident["sha256"] != a.cache_sha256:
        raise Refused(f"{a.cache} has sha256 {ident['sha256']}, not {a.cache_sha256}")
    try:
        d = load_cache(a.cache)
    except TC.CacheLineageError as e:
        raise Refused(str(e)) from None
    if d["cache_sha256"] != ident["sha256"] or L.file_identity(a.cache) != ident:
        raise Refused(f"{a.cache} changed while it was read")
    m0 = metas[0]
    for i, m in enumerate(metas):
        if m.get("arch_cfg") != m0.get("arch_cfg") or m.get("recipe") != m0.get("recipe"):
            raise Refused(f"member {i} has another architecture or recipe than member 0 (one split per bundle)")
    if len({m.get("seed") for m in metas}) != len(metas):
        raise Refused("two members share a seed")
    rs = declared_research(a)
    r0 = m0.get("recipe") if isinstance(m0.get("recipe"), dict) else {}
    if a.split_rule != E.RULE_UNVERIFIABLE:
        rs["split"].update({k: r0.get(k) for k in ("val_seed", "val_frac", "n_train", "n_val")})
        if a.split_rule == E.RULE_MULTI_HOLDOUT:
            rs["split"]["fold"] = a.fold
    if fold_file_sha is not None:
        rs["fold_file_sha256"] = fold_file_sha
    try:
        E.check_research_marking(rs, check_recorded=False)   # the declaration; the checks are recorded later
    except E.EligibilityError as err:
        raise Refused(str(err)) from None
    why = E.research_member_reasons(metas, rs, sims=sorted(set(np.asarray(d["sim_name"]).astype(str).tolist())))
    if why:
        raise Refused("the declared split does not admit the members: " + "; ".join(why))
    checkpoints = [T.load_checkpoint(p) for p in prefixes]        # digests verified above
    if [json.loads(json.dumps(m)) for _, m, _ in checkpoints] != metas:
        raise Refused("a member's meta changed between the two reads")
    restored = {}
    for p, m in zip(prefixes, metas):
        if os.path.isfile(p + ".hist.json"):
            hist = json.load(open(p + ".hist.json"))
            restored[os.path.basename(p)] = dict(_restored(hist, m.get("recipe")) or {},
                                                 hist_sha256=L.sha256_file(p + ".hist.json"))
    rs["restored_epoch"] = restored or None
    return prefixes, checkpoints, metas, d, {"sha256": ident["sha256"], "size": ident["size"]}, code, files, rs


def _failed(check):
    return "; ".join(f"{c['component']} {c['outcome']} ({c['sub_reason']}"
                     + (f": {c['detail'].get('why')}" if c["detail"].get("why") else "") + ")"
                     for c in check["components"] if c["outcome"] != E.PASS)


def _inputs():
    r = np.random.default_rng(SEED)
    x = r.uniform(size=(N_INPUTS, 10))
    z = Z_LIMITS[0] + x[:, 9] * (Z_LIMITS[1] - Z_LIMITS[0])
    tau0 = r.uniform(0.7, 1.3, N_INPUTS) * 0.0023 * (1.0 + z) ** 3.65
    return x, tau0


def run(a):
    from hcd_analysis.emulator import wc_map as W
    try:
        prefixes, ckpts, metas, d, ident, code, files, rs = check_inputs(a)
        sims = sorted(set(np.asarray(d["sim_name"]).astype(str).tolist()))
        held = rs["split"]["held_out_sims"]
        if not set(held) <= set(sims):
            raise Refused(f"held-out simulations not in the cache: {sorted(set(held) - set(sims))}")
        train_sims = sorted(set(sims) - set(held))
        try:
            doc = W.refit_subset(d, train_sims) if held else W.refit(d)
        except W.WcMapError as e:
            raise Refused(f"the w_c map cannot be refitted on the training simulations: {e}") from None
        wrec = W.make_record(doc, d["lineage"], ident, code=code, cache_name=os.path.basename(a.cache))
        if not wrec["release_eligible"]:
            raise Refused("the refitted w_c map's record is not eligible: " + "; ".join(wrec["ineligible_reasons"]))
        rs["wc_map"] = {"rule": E.RULE_WCMAP_SUBSET if held else E.RULE_WCMAP,
                        "fitted_on_sims_sha256": E.sims_sha256(train_sims), "n_sims": doc["delta_c"]["n_sims"]}
        wraw = W.to_bytes(doc, wrec)
        wcm = W.parse_wc_map(wraw, "the refitted w_c map")
        before = E.research_content_check(metas, [n for _, _, n in ckpts], wcm, rs, cache=d, cache_path=a.cache,
                                          expected_cache_sha256=a.cache_sha256, cache_size=ident["size"])
        if not E.research_outcome_ok(before):
            raise Refused("a content check is MISMATCH: " + _failed(before))
        rs["content_checks"] = before["components"]
        block = H.make_lineage_block(cache_record=d["lineage"], cache_identity=ident, wc_map=wcm, member_metas=metas,
                                     export_code=code, research=rs)
        if H.lineage_block_reasons(block, H._release_finder_ids()):
            raise Refused("the bundle's lineage is not clean: " + "; ".join(
                H.lineage_block_reasons(block, H._release_finder_ids())))
        if H.member_tie_reasons(block, metas):
            raise Refused("the members are not tied to the cache and one clean commit: " + "; ".join(
                H.member_tie_reasons(block, metas)))
    except Refused as e:
        print(f"refusing: {e}")
        return 2
    bundle, evidence = os.path.join(a.out, "bundle"), os.path.join(a.out, "evidence")
    os.makedirs(bundle)
    os.makedirs(evidence)
    out_prefixes = []
    for p, (model, meta, norm) in zip(prefixes, ckpts):
        q = os.path.join(bundle, os.path.basename(p))
        H.export_member(q, model, meta, norm)
        out_prefixes.append(q)
    W.write_wc_map(os.path.join(bundle, H.WC_MAP_FILE), doc, wrec)
    back = [H.load_member(q, expected_sha256=S.sha256_file(q + H.EXT), expected_json_sha256=S.sha256_file(q + ".json"))
            for q in out_prefixes]
    wback = W.read_wc_map(os.path.join(bundle, H.WC_MAP_FILE))
    after = E.research_content_check([m for _, m, _ in back], [n for _, _, n in back], wback, rs, cache=d,
                                     cache_path=a.cache, expected_cache_sha256=a.cache_sha256,
                                     cache_size=ident["size"])
    if not E.research_outcome_ok(after) or E.outcomes_differ(before, after):
        print("FAIL: the content checks on the files read back differ or are MISMATCH (no manifest written, nothing "
              "loadable): " + _failed(after))
        return 1
    rs["content_checks"] = after["components"]               # the outcomes of the objects actually written
    del d
    import equinox
    import h5py
    man_path = H.write_manifest(bundle, out_prefixes, {
        "source_sha256sums": os.path.basename(a.sha256sums), "source_sha256sums_sha256": a.sha256sums_sha256,
        "source_files_sha256": files, "cache_file": os.path.basename(a.cache), "export_code": code,
        "versions": {"python": platform.python_version(), "jax": jax.__version__, "equinox": equinox.__version__,
                     "numpy": np.__version__, "h5py": h5py.__version__, "libhdf5": h5py.version.hdf5_version},
        "conversion": "research checkpoints read by train.load_checkpoint after their sha256 matched the pinned "
                      "SHA256SUMS; the cache read by data.load_cache; the w_c map refitted on the training "
                      "simulations (wc_map.refit_subset)",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "research": rs}, lineage=block)
    sha = S.sha256_file(man_path)
    checks = verify(bundle, sha, a.purpose, ckpts)
    ok = all(checks.values())
    summary = {"format": H.FORMAT, "kind": "research", "verdict": "PASS" if ok else "FAIL", "checks": checks,
               "bundle_manifest_sha256": sha, "research": rs, "n_members": len(prefixes), "n_inputs": N_INPUTS,
               "seed": SEED}
    with open(os.path.join(evidence, "research_export_summary.json"), "x") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True, allow_nan=False)
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print(f"verdict: {summary['verdict']}; research bundle manifest sha256 {sha}")
    return 0 if ok else 1


def verify(bundle, sha, purpose, ckpts):
    """After writing: the research loaders load the bundle, it equals the checkpoints bitwise (leaves, normaliser,
    meta, ensemble P_filt), and every production loader refuses it."""
    import hcd_analysis.predictor as hp
    checks = {}
    rb = H.load_research(bundle, expected_manifest_sha256=sha, purpose=purpose)
    checks["research_load"] = rb.kind == "research" and rb.release_verified is False and rb.metas.kind == "research"
    for i, ((m0, meta0, n0), m1, meta1) in enumerate(zip(ckpts, rb.ensemble.members, rb.metas)):
        l0, t0 = jax.tree_util.tree_flatten(m0)
        l1, t1 = jax.tree_util.tree_flatten(m1)
        checks[f"m{i}_leaves"] = t0 == t1 and all(_same(u, v) for u, v in zip(l0, l1))
        checks[f"m{i}_meta"] = json.loads(json.dumps(meta0)) == meta1
    checks["norm"] = _tree_same(S.norm_from_tensors(S.norm_tensors(ckpts[0][2])), rb.norm)
    x, tau0 = _inputs()
    ens0 = EnsembleEmulator([m for m, _, _ in ckpts])
    P0 = np.stack([np.asarray(predict_P_filt(ens0, jnp.asarray(xi[:9]), xi[9], ti, ckpts[0][2]["P_filt"]))
                   for xi, ti in zip(x, tau0)])
    P1 = np.stack([np.asarray(predict_P_filt(rb.ensemble, jnp.asarray(xi[:9]), xi[9], ti, rb.norm["P_filt"]))
                   for xi, ti in zip(x, tau0)])
    checks["ensemble_P_filt"] = _same(P0, P1)
    rp = hp.load_research_bundle(bundle, sha, purpose=purpose)
    checks["research_predictor"] = isinstance(rp, hp.ResearchPredictor) and not isinstance(rp, hp.Bundle)
    refused = []
    for name, f in (("load_release", lambda: H.load_release(bundle, expected_manifest_sha256=sha)),
                    ("load_ensemble", lambda: H.load_ensemble(bundle, expected_manifest_sha256=sha)),
                    ("load_bundle", lambda: H.load_bundle(bundle, expected_manifest_sha256=sha)),
                    ("predictor.load_bundle", lambda: hp.load_bundle(bundle, sha))):
        try:
            f()
        except H.SafeFormatError:
            refused.append(name)
    checks["production_loaders_refuse"] = len(refused) == 4
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="NEW directory (refused if it exists)")
    ap.add_argument("--purpose", required=True, help="what the research bundle is for (recorded)")
    ap.add_argument("--split-rule", required=True, choices=[*E.SPLIT_RULES, E.RULE_EXPLICIT_ROWS])
    ap.add_argument("--held-out-sim", action="append",
                    help="a held-out simulation (repeatable; none: nothing held out)")
    ap.add_argument("--fold", type=int, default=None, help="the fold index (multi_holdout_row_split/1)")
    ap.add_argument("--folds-digest", default=None, help="sha256 of the fold file the members record (folds_sha256)")
    ap.add_argument("--fold-file", default=None, help="the fold file itself (its digest recorded)")
    ap.add_argument("--fold-file-sha256", default=None)
    ap.add_argument("--cv-set-id", default=None, help="an identifier of the cross-validation set (recorded)")
    ap.add_argument("--description", default=None, help="what an unverifiable split held out")
    ap.add_argument("--member", action="append", required=True, help="a member's checkpoint prefix (repeatable)")
    ap.add_argument("--sha256sums", required=True, help="a SHA256SUMS listing every member file by name")
    ap.add_argument("--sha256sums-sha256", required=True)
    ap.add_argument("--cache", required=True, help="the release training cache (format 3.4)")
    ap.add_argument("--cache-sha256", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"refusing: {a.out} exists (the export never writes into an existing directory)")
        return 2
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
