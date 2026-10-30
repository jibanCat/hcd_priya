"""The R1 carry: a release training cache (format 3.4) whose per-class P1D rows are copied from a historical cache
(format 3.3) after a per-group validation against the rebuilt catalogues, with every other field rebuilt. This is the
only route by which historical P1D bytes enter a release-eligible cache.

Why it is sound: a cache row is a function of the raw optical depths, the per-sightline Tier-C labels and the
mean-flux rung only. The labels depend on each sightline's highest-N_HI system, which the box-edge fix did not
change. So rows built from the old catalogue equal rows built from the rebuilt one whenever the labels are equal, the
raw files are the same bytes, and the code that computes the rows is the same.

Checks, for every (simulation, snapshot) group of the historical cache:
  1. inputs: the rebuilt snapshot through tau0_cache.load_snapshot_inputs (a release-eligible catalogue made with the
     release finder settings from the whole raw file, its CDDF and meta.json, all matching their sha256), the old
     catalogue through load_catalog_historical, the old meta.json;
  2. the label tie, three ways: the per-sightline Tier-C labels of the old and the rebuilt catalogue are identical;
     the rebuilt labels' per-class sightline counts equal the stored tier_c_counts of every row; the stored
     snap_n_absorbers equal the old catalogue's histogram;
  3. raw identity: the raw file matches the rebuilt catalogue's record (sha256 and size; a matching mtime is accepted
     without rereading unless --rehash-raw); its mtime precedes the old catalogue's and the historical cache's
     creation; its header equals the facts the old meta.json recorded (the old products recorded no raw sha256 or
     mtime; what they did record is reported);
  4. the row metadata recomputed from the rebuilt inputs (params, z_meta, z_grid, dv_kms, nbins_native,
     alpha_slope, alpha_idx, sim_name, snap) equals the stored values exactly, and the catalogue-to-raw pairing holds;
  5. the per-row DLA core (the loader's delta[:, 2]) from the rebuilt counts equals the one from the stored counts;
  6. the snapshot block rebuilt from the rebuilt CDDF and meta.json: CDDF counts equal the rebuilt catalogue's
     histogram, meta.json counts its class counts, the CDDF bins and the absorption path equal the stored ones.
Once, first: the row-path code (ROW_PATH) at each builder commit of record equals the code at HEAD, by a
function-level diff (a non-empty diff passes only when its sha256 is accepted with --accept-row-path-diff, after
review); the historical cache's layout, recipe attributes, static tables and mean-flux ladder equal the builder's.
If one of these fails, no group is validated (the summary and the diffs are still written).

Output: a validation record per group (<validation-dir>/groups/), the row-path diffs (<validation-dir>/row_path/) and
<validation-dir>/summary.json. Only if every check passes, the cache: the CARRIED_FIELDS copied from the historical
cache (marked carried_validated in the lineage record, with its sha256), every other field rebuilt. On any failure no
cache is written and the exit status is 2.

Usage (from a clean, committed checkout; the interpreter of the cache build):
    python scripts/carry_validated_cache.py --historical-cache OLD.h5 --old-root OLD_SNAPSHOT_ROOT \\
        --rebuilt-root REBUILT_SNAPSHOT_ROOT --raw-root RAW_SPECTRA_ROOT --row-path-commit COMMIT [...] \\
        [--accept-row-path-diff COMMIT=SHA256 ...] [--rehash-raw] --validation-dir DIR --output NEW.h5
"""
from __future__ import annotations

import argparse
import ast
import datetime
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_emulator_cache as bec  # noqa: E402
import build_emulator_cache_tau0 as bt0  # noqa: E402
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402
from hcd_analysis.catalog import load_catalog_historical  # noqa: E402
from hcd_analysis.io import pixel_dv_kms, read_header  # noqa: E402
from hcd_analysis.tau0_rescale import make_alpha_grid_priya_aligned  # noqa: E402

# The code that computes the carried fields: (file, symbols) with None for the whole file. A symbol is a function,
# a class method ("Class.method") or a module-level assignment.
ROW_PATH = (
    ("scripts/build_emulator_cache_tau0.py", ("build_tau0_rows", "_read_tau", "_snap_z_to_priya_grid",
                                              "catalogue_z_label", "_PHASE1_Z_OVERRIDE")),
    ("hcd_analysis/priya_p1d.py", None),
    ("hcd_analysis/tau0_rescale.py", None),
    ("hcd_analysis/catalog.py", ("classify_system", "AbsorberCatalog.load_npz", "AbsorberCatalog._from_arrays")),
)
CARRY_RULE = ("per-class P1D rows copied from the historical cache only for groups whose per-sightline labels from "
              "the old and the rebuilt catalogue are identical, whose rebuilt label counts equal the stored "
              "tier_c_counts and whose stored counts equal the old catalogue; raw files matched to the rebuilt "
              "catalogues' sha256 and older than the old catalogues; row-path code compared with the builder "
              "commits of record; every count field rebuilt from the rebuilt catalogues")
SOURCE_ROLE = "historical_source"


class CarryRefused(RuntimeError):
    """A precondition of the carry does not hold (nothing is validated or written)."""


# ------------------------------------------------------------------------------------------- row-path code diff
def _symbol_source(src, symbol):
    """Source text of a function, a class method or a module-level assignment; None when absent."""
    tree = ast.parse(src)
    parts = symbol.split(".")
    nodes = tree.body
    for i, name in enumerate(parts):
        last = i == len(parts) - 1
        hit = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
                hit = n
            elif last and isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    hit = n
        if hit is None:
            return None
        if last:
            return ast.get_source_segment(src, hit)
        nodes = getattr(hit, "body", [])
    return None


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def row_path_diff(repo, old_commit, new_commit, spec=ROW_PATH):
    """Function-level diff of the row-path code between two commits: ``(result, diff_text)``. ``result`` holds the
    resolved commits, each item's status (identical, changed, absent at one side) and the diff's sha256, which
    depends on the code only (not on the commit hashes), so a reviewed diff keeps its digest; the diff is empty when
    every item is identical."""
    old = _git(repo, "rev-parse", "--verify", f"{old_commit}^{{commit}}").strip()
    new = _git(repo, "rev-parse", "--verify", f"{new_commit}^{{commit}}").strip()
    items, chunks = [], []
    for path, symbols in spec:
        texts = {}
        for side, c in (("old", old), ("new", new)):
            try:
                texts[side] = _git(repo, "show", f"{c}:{path}")
            except subprocess.CalledProcessError:
                texts[side] = None
        for sym in (symbols or (None,)):
            a, b = ((texts[s] if sym is None or texts[s] is None else _symbol_source(texts[s], sym))
                    for s in ("old", "new"))
            name = path if sym is None else f"{path}:{sym}"
            status = ("identical" if a == b and a is not None else "absent at both" if a is None and b is None
                      else "absent at old" if a is None else "absent at new" if b is None else "changed")
            items.append({"item": name, "status": status})
            if status != "identical":
                chunks.append("".join(difflib.unified_diff((a or "").splitlines(True), (b or "").splitlines(True),
                                                           f"old/{name}", f"new/{name}")))
    text = "\n".join(chunks)
    return ({"old_commit": old, "new_commit": new, "identical": not chunks, "items": items,
             "diff_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}, text)


# --------------------------------------------------------------------------------------------- per-group checks
def _check(checks, name, ok, **info):
    checks[name] = dict(ok=bool(ok), **info)


def _utc_ns(stamp):
    """Nanoseconds since the epoch of an ISO UTC stamp such as 2026-01-02T03:04:05Z."""
    t = datetime.datetime.strptime(str(stamp).rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    return int(t.timestamp()) * 1_000_000_000


def validate_group(src, b, rows, ctx):
    """Validate one group of the historical cache ``src`` (block ``b``, row indices ``rows``). Returns
    ``(record, rebuilt)``: the validation record, and on success the rebuilt per-row and snapshot values."""
    from hcd_analysis.priya_p1d import N_TIER_C_BINS, bin_sightlines_by_nhi
    sim, snap = str(src["snap_sim_name"][b]), int(src["snap_snap"][b])
    rec = {"sim": sim, "snap": snap, "z_grid": float(src["z_grid"][rows[0]]), "rows": [int(rows[0]), int(rows[-1])],
           "origin": ctx["origins"].get((sim, snap), "base"), "checks": {}}
    checks = rec["checks"]
    raw = bt0.locate_raw_tau_file(ctx["raw_root"], sim, snap)
    old_dir = Path(ctx["old_root"]) / sim / f"snap_{snap:03d}"
    try:
        if raw is None:
            raise FileNotFoundError(f"no raw spectra file for {sim} snap {snap} under the raw root")
        inp, entries = T.load_snapshot_inputs(Path(ctx["rebuilt_root"]) / sim / f"snap_{snap:03d}", raw, sim=sim,
                                              snap=snap, rehash_raw=ctx["rehash_raw"])
        old = load_catalog_historical(old_dir / "catalog.npz", reason="R1 carry: the label tie with the rebuilt "
                                                                      "catalogue")
        old_meta = json.loads((old_dir / "meta.json").read_text())
        hdr = read_header(raw)
    except Exception as e:                                   # any refusal of an input fails the group
        _check(checks, "inputs", False, error=f"{type(e).__name__}: {e}")
        rec["ok"] = False
        return rec, None
    _check(checks, "inputs", True, rebuilt={x["role"].split(":")[0]: x["sha256"] for x in entries})
    rec["entries"] = entries
    n_sk, nr = int(hdr.n_skewers), len(rows)

    # 2. the label tie, three ways
    lab_old, lab_new = bin_sightlines_by_nhi(old, n_sk), bin_sightlines_by_nhi(inp.catalog, n_sk)
    _check(checks, "labels_old_equal_rebuilt", np.array_equal(lab_old, lab_new), n_sightlines=n_sk,
           n_changed=int(np.sum(lab_old != lab_new)))
    counts = np.bincount(lab_new, minlength=N_TIER_C_BINS).astype(np.int64)
    stored = np.asarray(src["tier_c_counts"])[rows]
    _check(checks, "rebuilt_counts_equal_stored_tier_c_counts", counts.size == N_TIER_C_BINS
           and np.array_equal(stored, np.tile(counts, (nr, 1))), rebuilt=counts.tolist(),
           rows_differing=int(np.sum(np.any(stored != counts[None, :], axis=1))) if counts.size == stored.shape[1]
           else nr)
    old_hist = np.histogram(old.log_nhi_array(), bins=np.asarray(src["log_nhi_edges"]))[0]
    _check(checks, "stored_counts_equal_old_catalogue", np.array_equal(np.asarray(src["snap_n_absorbers"][b]),
                                                                        old_hist),
           n_old_entries=len(old.absorbers), stored_total=int(np.sum(src["snap_n_absorbers"][b])))

    # 3. raw identity
    raw_rec = L.find_input(inp.catalog.lineage, "raw_spectra")
    st, st_old = os.stat(raw), os.stat(old_dir / "catalog.npz")
    _check(checks, "raw_matches_rebuilt_record", True, sha256=raw_rec["sha256"], size=raw_rec["size"],   # step 1
           recorded_mtime_ns=raw_rec.get("mtime_ns"), mtime_ns=int(st.st_mtime_ns), rehashed=ctx["rehash_raw"],
           note="verified by the inputs step (the release catalogue loader with this raw file)")
    _check(checks, "raw_predates_old_catalogue_and_source", st.st_mtime_ns < st_old.st_mtime_ns
           and st.st_mtime_ns < ctx["source_created_ns"], raw_mtime_ns=int(st.st_mtime_ns),
           old_catalogue_mtime_ns=int(st_old.st_mtime_ns), source_created_ns=ctx["source_created_ns"])
    header = {"nbins": int(hdr.nbins), "n_skewers": n_sk, "box_kpc_h": float(hdr.box), "hubble": float(hdr.hubble),
              "dv_kms": float(pixel_dv_kms(hdr))}
    recorded = {k: old_meta.get(k) for k in ("nbins", "n_skewers", "box_kpc_h", "hubble", "dv_kms", "z")}
    _check(checks, "raw_header_equals_old_meta", all(recorded[k] == v for k, v in header.items()), header=header,
           old_meta_recorded=recorded, old_sha256_recorded=None)
    p1d_attrs = old_dir / "p1d_per_class.h5"
    if p1d_attrs.exists():
        with h5py.File(p1d_attrs, "r") as f:
            rec["old_p1d_per_class_attrs"] = {k: T._attr_value(v) for k, v in f.attrs.items()}

    # 4. row metadata, recomputed
    try:
        z_meta = bt0.catalogue_z_label(sim, snap, float(inp.meta["z"]), float(hdr.redshift))
        bt0.assert_label_matches_header(z_meta, float(hdr.redshift))
        bt0.assert_catalogue_built_on(inp.meta, bt0._catalogue_pix_max(inp.catalog), hdr)
        params = bt0._read_priya_params(raw)
        pairing = None
    except Exception as e:
        pairing = f"{type(e).__name__}: {e}"
    _check(checks, "pairing_and_labels", pairing is None, error=pairing)
    if pairing is not None:
        rec["ok"] = False
        return rec, None
    a_idx = np.asarray(src["alpha_idx"])[rows]
    meta_rows = {
        "sim_name": np.array([sim] * nr, dtype=object), "snap": np.full(nr, snap, dtype=np.int32),
        "alpha_idx": np.arange(nr, dtype=np.int32), "alpha_slope": ctx["alpha_grid"][a_idx],
        "z_meta": np.full(nr, z_meta), "z_grid": np.full(nr, float(bt0._snap_z_to_priya_grid(z_meta))),
        "dv_kms": np.full(nr, float(inp.meta["dv_kms"])), "nbins_native": np.full(nr, int(hdr.nbins), dtype=np.int32),
        "params": np.tile(params, (nr, 1)),
    }
    diff = sorted(k for k, v in meta_rows.items() if not np.array_equal(np.asarray(src[k])[rows], v))
    _check(checks, "row_metadata_recomputed_equal_stored", not diff, differing=diff)

    # 5. the per-row DLA core from the rebuilt counts
    pu, pf = np.asarray(src["P_tier_c"])[rows], np.asarray(src["P_tier_c_filtered"])[rows]
    new_core = T.dla_core_rows(pu, pf, np.tile(counts, (nr, 1)))
    _check(checks, "dla_core_rebuilt_equal_stored", np.array_equal(new_core, T.dla_core_rows(pu, pf, stored),
                                                                   equal_nan=True))

    # 6. the snapshot block, rebuilt
    cddf = inp.cddf
    new_hist = np.histogram(inp.catalog.log_nhi_array(), bins=np.asarray(cddf["log_nhi_edges"]))[0]
    by_class = {c: n for c, n in inp.catalog.summary().items()}
    _check(checks, "rebuilt_cddf_and_meta_counts_equal_rebuilt_catalogue",
           np.array_equal(np.asarray(cddf["n_absorbers"]), new_hist)
           and {k: int(v) for k, v in inp.meta["n_absorbers"].items()} == by_class)
    _check(checks, "cddf_bins_and_absorption_path_equal_stored",
           np.array_equal(np.asarray(cddf["log_nhi_edges"], float), np.asarray(src["log_nhi_edges"]))
           and np.array_equal(np.asarray(cddf["log_nhi_centres"], float), np.asarray(src["log_nhi_centres"]))
           and float(cddf["total_path"]) == float(src["snap_total_path_dX"][b]))
    dndx = bec.compute_dndx_per_class(inp.meta["n_absorbers"], float(cddf["total_path"]))
    block = {"snap_sim_name": sim, "snap_snap": snap, "snap_total_path_dX": float(cddf["total_path"]),
             "snap_f_nhi": np.asarray(cddf["f_nhi"], dtype=np.float64),
             "snap_n_absorbers": np.asarray(cddf["n_absorbers"], dtype=np.int64),
             **{"snap_" + k: v for k, v in dndx.items()}}
    rec["counts_old_minus_rebuilt_per_bin"] = (np.asarray(src["snap_n_absorbers"][b]) - block["snap_n_absorbers"]
                                               ).tolist()
    rec["old_catalogue"] = {k: v for k, v in L.file_identity(old_dir / "catalog.npz").items()}
    rec["ok"] = all(c["ok"] for c in checks.values())
    return rec, (dict(meta_rows, tier_c_counts=np.tile(counts, (nr, 1))), block, entries, raw)


# ------------------------------------------------------------------------------------------------------- the run
def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    return L.sha256_file(path)


def check_source(src, attrs, n_alpha_refine, fidelity):
    """Global checks of the historical cache: canonical layout, the builder's recipe and static tables, the ladder."""
    from hcd_analysis.priya_p1d import FINE_NHI_EDGES, tier_c_labels
    out = {}
    G = len(src["snap_snap"])
    gi, a = np.asarray(src["snap_group_idx"]), np.asarray(src["alpha_idx"])
    n_alpha = 10 * n_alpha_refine
    ids = list(zip(map(str, src["snap_sim_name"]), map(int, src["snap_snap"])))
    out["canonical_layout"] = bool(gi.size == G * n_alpha and np.array_equal(gi, np.repeat(np.arange(G), n_alpha))
                                   and np.array_equal(a, np.tile(np.arange(n_alpha), G)) and ids == sorted(ids)
                                   and len(set(ids)) == G)
    n_k = int(T._attr_value(attrs["n_k"]))
    out["n_k_equal_builder"] = n_k == bt0._N_K[fidelity] == np.asarray(src["kfkms"]).shape[1]
    want = bt0.cache_attrs(n_k, T._attr_value(attrs["alpha_range"]))        # n_k and alpha_range: checked apart
    out["recipe_attributes_equal_builder"] = all(np.array_equal(np.asarray(T._attr_value(attrs.get(k))),
                                                                np.asarray(T._attr_value(want[k])))
                                                 for k in bt0.RECIPE_ATTRS)
    grid = make_alpha_grid_priya_aligned(refine=n_alpha_refine)
    out["mean_flux_ladder_equal_builder"] = bool(np.array_equal(np.asarray(T._attr_value(attrs["alpha_range"])),
                                                                [grid[0], grid[-1]])
                                                 and np.array_equal(np.asarray(src["alpha_slope"]), grid[a]))
    out["static_tables_equal_builder"] = bool(list(src["tier_c_labels"]) == tier_c_labels()
                                              and np.array_equal(src["tier_c_nhi_edges"], FINE_NHI_EDGES)
                                              and list(src["param_names"]) == list(bec.PARAM_ORDER))
    return out, grid


def run(args):
    vdir, out_path = Path(args.validation_dir), Path(args.output)
    if out_path.exists():
        raise CarryRefused(f"{out_path} exists; a cache is never overwritten")
    if vdir.exists() and any(vdir.iterdir()):
        raise CarryRefused(f"{vdir} is not empty; every run writes its records into a new directory")
    code = L.code_identity()
    if not (code.get("commit") and code.get("dirty") is False):
        raise CarryRefused(f"the carry runs from a clean, committed checkout only (code identity {code})")
    accepted = {}
    for a in args.accept_row_path_diff or []:
        c, _, sha = a.partition("=")
        if len(sha) != 64:
            raise CarryRefused(f"--accept-row-path-diff {a!r}: expected COMMIT=SHA256")
        accepted[c] = sha

    # once: the code, then the source
    row_path = []
    for c in args.row_path_commit:
        res, text = row_path_diff(REPO_ROOT, c, code["commit"])
        res["accepted_sha256"] = accepted.get(c) or accepted.get(res["old_commit"])
        res["ok"] = res["identical"] or res["accepted_sha256"] == res["diff_sha256"]
        res["diff_file"] = f"row_path/{res['old_commit'][:12]}.diff"
        (vdir / "row_path").mkdir(parents=True, exist_ok=True)
        (vdir / res["diff_file"]).write_text(text)
        row_path.append(res)
    source_ident = L.file_identity(args.historical_cache)
    src, attrs, _ = T.read_cache_historical(args.historical_cache, reason="R1 carry: the source of the carried rows")
    version = str(T._attr_value(attrs.get("cache_version")))
    n_alpha = int(np.max(src["alpha_idx"])) + 1
    if n_alpha % 10:
        raise CarryRefused(f"the source holds {n_alpha} mean-flux rungs, not a multiple of the PRIYA ten")
    globals_, grid = check_source(src, attrs, n_alpha // 10, args.fidelity)
    globals_["source_is_version_3.3"] = version == "3.3"
    globals_["row_path_code"] = all(r["ok"] for r in row_path)
    origins = {}
    if "repair_record" in attrs:
        for sim, snap, _z in json.loads(T._attr_value(attrs["repair_record"])).get("added", []):
            origins[(str(sim), int(snap))] = "spliced"
    ctx = dict(raw_root=args.raw_root, old_root=args.old_root, rebuilt_root=args.rebuilt_root, origins=origins,
               rehash_raw=bool(args.rehash_raw), alpha_grid=grid,
               source_created_ns=_utc_ns(T._attr_value(attrs["created_utc"])))

    # every group (skipped when a global check already failed: no group could be carried)
    gi = np.asarray(src["snap_group_idx"])
    records, rebuilt, files = [], [], {}
    for b in range(len(src["snap_snap"]) if all(globals_.values()) else 0):
        rows = np.flatnonzero(gi == b)
        rec, reb = validate_group(src, b, rows, ctx)
        name = f"groups/{rec['sim']}__snap_{rec['snap']:03d}.json"
        files[name] = _write_json(vdir / name, {k: v for k, v in rec.items() if k != "entries"})
        records.append(rec)
        rebuilt.append(reb)
        if (b + 1) % 50 == 0:
            print(f"  validated {b + 1}/{len(src['snap_snap'])} groups ({sum(not r['ok'] for r in records)} failed)",
                  flush=True)
    failed = [f"{r['sim']} {r['snap']}: " + ", ".join(k for k, c in r["checks"].items() if not c["ok"])
              for r in records if not r["ok"]]
    passed = not failed and all(globals_.values()) and len(records) == len(src["snap_snap"])
    summary = {
        "format": "hcd_priya.carry_validation/1", "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"), "code": code, "passed": passed,
        "source": {"path": str(args.historical_cache), **{k: source_ident[k] for k in ("sha256", "size")},
                   "cache_version": version, "created_utc": str(T._attr_value(attrs.get("created_utc"))),
                   "git_sha": str(T._attr_value(attrs.get("git_sha")))},
        "roots": {"old": str(args.old_root), "rebuilt": str(args.rebuilt_root), "raw": str(args.raw_root)},
        "rehash_raw": bool(args.rehash_raw), "global_checks": globals_, "row_path": row_path,
        "n_groups": len(records), "n_groups_in_source": len(src["snap_snap"]), "n_failed": len(failed),
        "failed": failed, "group_records_sha256": files,
        "origins": {"spliced": sorted(f"{s}:{n}" for s, n in origins)}, "rule": CARRY_RULE,
    }
    summary_sha = _write_json(vdir / "summary.json", summary)
    if not passed:
        print(f"CARRY REFUSED: {len(failed)} group(s) failed, global checks "
              f"{ {k: v for k, v in globals_.items() if not v} }; records in {vdir}; no cache written")
        for f in failed[:20]:
            print("  " + f)
        return 2

    # every check passed: the cache
    d = {k: np.asarray(src[k]) for k in T.CARRIED_FIELDS}
    for k in ("sim_name", "snap", "alpha_idx", "alpha_slope", "z_meta", "z_grid", "dv_kms", "nbins_native", "params",
              "tier_c_counts"):
        d[k] = np.concatenate([r[0][k] for r in rebuilt], axis=0)
    d["snap_group_idx"] = np.repeat(np.arange(len(rebuilt), dtype=np.int32), n_alpha)
    for k in T.SNAP_KEYS:
        d[k] = np.array([r[1][k] for r in rebuilt], dtype=object if k == "snap_sim_name" else None)
    for k in ("snap_snap",):
        d[k] = d[k].astype(np.int32)
    for k in T.STATIC_KEYS:
        d[k] = np.asarray(src[k])
    entries = [e for r in records for e in r["entries"]] + [T.sim_ics_entry(r[3], r[1]["snap_sim_name"])
                                                             for r in rebuilt]
    entries.append(L.input_entry(SOURCE_ROLE, T.KIND, {"sha256": source_ident["sha256"], "size": source_ident["size"]},
                                 info=summary["source"]))
    n_k = int(T._attr_value(attrs["n_k"]))
    out_attrs = dict(bt0.cache_attrs(n_k, (grid[0], grid[-1])), n_rows=int(d["sim_name"].size),
                     n_snaps=len(rebuilt))
    rec = T.write_cache(
        out_path, d, out_attrs, inputs=T.combine_inputs(entries),
        settings=T.cache_settings(fidelity=args.fidelity, n_k=n_k, alpha_refine=n_alpha // 10,
                                  alpha_range=(grid[0], grid[-1])),
        groups=[{"sim": r["sim"], "snap": r["snap"], "z_grid": r["z_grid"], "rows": T.ROWS_CARRIED} for r in records],
        carried={"source_sha256": source_ident["sha256"], "source_size": source_ident["size"],
                 "fields": list(T.CARRIED_FIELDS), "rule": CARRY_RULE,
                 "row_path": [{k: r[k] for k in ("old_commit", "new_commit", "identical", "diff_sha256",
                                                 "accepted_sha256")} for r in row_path]},
        validation={"validated_inputs": [SOURCE_ROLE], "all_passed": True, "n_groups": len(records),
                    "summary_sha256": summary_sha, "checks": sorted(records[0]["checks"]) + sorted(globals_)})
    if not rec["release_eligible"]:
        out_path.unlink()
        print("CARRY REFUSED: the output would not be release-eligible: " + "; ".join(rec["ineligible_reasons"]))
        return 2
    print(f"wrote {out_path} sha256 {L.sha256_file(out_path)}: {len(records)} groups carried_validated; "
          f"validation summary sha256 {summary_sha}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--historical-cache", type=Path, required=True, help="the 3.3 cache whose rows are carried")
    ap.add_argument("--old-root", type=Path, required=True, help="snapshot root of the old catalogues")
    ap.add_argument("--rebuilt-root", type=Path, required=True, help="snapshot root of the rebuilt catalogues")
    ap.add_argument("--raw-root", type=Path, required=True, help="raw spectra root (<sim>/output/SPECTRA_NNN)")
    ap.add_argument("--row-path-commit", action="append", required=True, metavar="COMMIT",
                    help="a builder commit of record of the historical cache (repeatable)")
    ap.add_argument("--accept-row-path-diff", action="append", metavar="COMMIT=SHA256",
                    help="accept the reviewed non-empty row-path diff with this sha256")
    ap.add_argument("--rehash-raw", action="store_true", help="re-hash every raw spectra file")
    ap.add_argument("--fidelity", choices=("lf", "hr"), default="lf")
    ap.add_argument("--validation-dir", type=Path, required=True, help="new directory for the validation records")
    ap.add_argument("--output", type=Path, required=True, help="the new cache (format 3.4)")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except CarryRefused as e:
        print(f"CARRY REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
