"""Acceptance of rebuilt catalogues against the old ones (the pre-registered catalogue checks of the clean lineage).

For one (simulation, snapshot) group, the rebuilt snapshot directory (catalogue-only mode) against the old one:
  1. the rebuilt snapshot passes the snapshot verifier: a catalogue stamped with an allowed finder id after the finder
     cross-check of every searched skewer, release-eligible, with the release finder settings, from the whole raw
     file it recorded (re-hashed with --rehash-raw), and its CDDF and meta.json matching their sha256;
  2. rebuilt = old minus its box-edge duplicates: the integer columns (skewer_idx, pix_start, pix_end) exactly equal
     and in order, fit_success and fast_mode equal, b_kms NaN in both; N_HI bitwise, or within one ulp with every
     such entry counted (none beyond one ulp);
  3. the old catalogue's duplicates per class equal the trace's row for the group (exactly one row);
  4. no duplicate or over-span entry is left in the rebuilt catalogue;
  5. the per-sightline Tier-C labels and the 4-class highest-class labels are identical;
  6. meta.json counts equal the old meta.json counts minus the duplicates per class; cddf.npz counts equal the
     histogram of the old catalogue minus its duplicates in every bin, with the same bins and absorption path.
One JSON per group (<out>/groups/<sim>__snap_NNN.json), which also records what it was checked against: the old
catalogue's sha256 and size (and whether it carried a record), the old meta.json's sha256, the trace file's sha256 and
the three roots. ``summarize`` writes <out>/summary.json and exits 0 only when every listed group was checked and
accepted, every group's raw file was re-hashed, every group was checked by one clean, committed acceptance code
identity, every group agrees with the one trace file given to ``summarize`` (by its recorded sha256 or, for a record
written before records named it, by its recorded duplicate counts), the rebuilt catalogues have one code identity
(at ``--expect-commit`` when given) and one finder-digest set.

The group list has one "SIM SNAP" per line. ``check --index I`` checks line I (1-based, e.g. $SLURM_ARRAY_TASK_ID);
without --index every line is checked in turn.

Usage:
    python scripts/accept_rebuilt_catalogues.py check --groups LIST --old-root OLD --new-root NEW --raw-root RAW \\
        --trace-rows TRACE.json --out OUT [--index I] [--rehash-raw]
    python scripts/accept_rebuilt_catalogues.py summarize --groups LIST --trace-rows TRACE.json --out OUT \\
        [--expect-commit COMMIT]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from hcd_analysis import catalog as C  # noqa: E402
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.snapshot_record import verify_snapshot_dir  # noqa: E402

CLASSES = ("LLS", "subDLA", "DLA", "forest")


def read_groups(path):
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"{path}:{i}: expected 'SIM SNAP', got {line!r}")
        out.append((parts[0], int(parts[1])))
    if len(set(out)) != len(out):
        raise SystemExit(f"{path}: a group is listed twice")
    return out


def group_file(out, sim, snap):
    return Path(out) / "groups" / f"{sim}__snap_{snap:03d}.json"


def _columns(cat):
    a = cat._arrays()
    return {k: np.asarray(a[k]) for k in ("skewer_idx", "pix_start", "pix_end", "NHI", "b_kms", "fit_success",
                                          "fast_mode")}


def _class_of(logn):
    return np.where(logn >= C.LOG_NHI_DLA_MIN, "DLA", np.where(logn >= C.LOG_NHI_SUBDLA_MIN, "subDLA",
                    np.where(logn >= C.LOG_NHI_LLS_MIN, "LLS", "forest")))


def _class4(cat, n_sk):
    m = np.full(n_sk, -np.inf)
    if cat.absorbers:
        np.maximum.at(m, np.array([a.skewer_idx for a in cat.absorbers], dtype=np.int64),
                      np.array([a.log_NHI for a in cat.absorbers]))
    return np.where(m >= C.LOG_NHI_DLA_MIN, 3, np.where(m >= C.LOG_NHI_SUBDLA_MIN, 2,
                                                        np.where(m >= C.LOG_NHI_LLS_MIN, 1, 0)))


def check_group(sim, snap, *, old_root, new_root, raw_root, trace_rows, rehash_raw=False, tier="LF",
                trace_sha256=None):
    """The acceptance record of one group: ``{"sim", "snap", "inputs", "checks": {name: {"ok", ...}}, "accepted"}``."""
    from hcd_analysis.priya_p1d import bin_sightlines_by_nhi
    rec = {"sim": sim, "snap": snap, "checks": {},
           "inputs": {"trace_sha256": trace_sha256,
                      "roots": {"old": str(old_root), "new": str(new_root), "raw": str(raw_root)}}}
    ck = rec["checks"]

    def check(name, ok, **info):
        ck[name] = dict(ok=bool(ok), **info)
    new_dir = Path(new_root) / sim / f"snap_{snap:03d}"
    old_dir = Path(old_root) / sim / f"snap_{snap:03d}"
    try:
        meta = json.loads((new_dir / "meta.json").read_text())
        raw_rec = L.find_input(C.AbsorberCatalog.load_npz(new_dir / "catalog.npz").lineage, "raw_spectra")
        raw = Path(raw_rec["info"]["path"])
        if Path(os.path.realpath(raw)).parts[:len(Path(os.path.realpath(raw_root)).parts)] != \
                Path(os.path.realpath(raw_root)).parts:
            raise ValueError(f"the recorded raw file {raw} is not under the raw root")
        v = verify_snapshot_dir(new_dir, mode="catalogue", settings=C.RELEASE_FINDER_SETTINGS, raw_path=raw,
                                rehash=rehash_raw, require_release_eligible=True, sim=sim, snap=snap)
    except Exception as e:
        check("release_loader", False, error=f"{type(e).__name__}: {e}")
        rec["accepted"] = False
        return rec
    new, lrec = v.catalog, v.catalog.lineage
    fin = lrec["product"]["finder"]
    check("release_loader", lrec["finder_ids"] == [C.FINDER_ID] and fin["verified"] is True
          and fin["n_skewers_searched"] == fin["n_skewers_verified"], finder_ids=lrec["finder_ids"],
          code=lrec["code"], raw={k: raw_rec.get(k) for k in ("sha256", "size", "mtime_ns")}, rehashed=rehash_raw,
          n_skewers_searched=fin["n_skewers_searched"], n_skewers_verified=fin["n_skewers_verified"],
          estimator_source_sha256=fin.get("estimator_source_sha256"), source_sha256=fin.get("source_sha256"))
    old_ident = L.file_identity(old_dir / "catalog.npz")
    old = C.load_catalog_historical(old_dir / "catalog.npz", reason="catalogue acceptance: the old catalogue")
    if L.file_identity(old_dir / "catalog.npz") != old_ident:
        raise RuntimeError(f"{old_dir / 'catalog.npz'} changed while it was read")
    old_meta_bytes = (old_dir / "meta.json").read_bytes()
    old_meta = json.loads(old_meta_bytes)
    rec["inputs"]["old_catalogue"] = dict(old_ident, has_record=old.original_lineage_json is not None)
    rec["inputs"]["old_meta_sha256"] = hashlib.sha256(old_meta_bytes).hexdigest()
    n, n_sk = int(meta["nbins"]), int(meta["n_skewers"])
    o, w = _columns(old), _columns(new)
    dup = C.find_wrap_duplicates(o["skewer_idx"], o["pix_start"], o["pix_end"], n)
    ok = {k: x[~dup] for k, x in o.items()}
    same = ok["skewer_idx"].size == w["skewer_idx"].size
    check("entries_old_minus_duplicates", same, old=int(dup.size), old_duplicates=int(dup.sum()),
          new=int(w["skewer_idx"].size))
    if same:
        check("integer_columns_exact_in_order", all(np.array_equal(ok[k], w[k]) for k in ("skewer_idx", "pix_start",
                                                                                           "pix_end")))
        check("flags_equal_b_nan", all(np.array_equal(ok[k], w[k]) for k in ("fit_success", "fast_mode"))
              and bool(np.isnan(ok["b_kms"]).all() and np.isnan(w["b_kms"]).all()))
        d = np.abs(ok["NHI"] - w["NHI"])
        ulp = np.spacing(np.maximum(np.abs(ok["NHI"]), np.abs(w["NHI"])))
        lo, ln = np.log10(np.maximum(ok["NHI"], 1e1)), np.log10(np.maximum(w["NHI"], 1e1))
        check("nhi_bitwise_or_within_one_ulp", bool(np.all(d <= ulp)), bitwise=int(np.sum(d == 0)),
              within_1ulp_not_bitwise=int(np.sum((d > 0) & (d <= ulp))), beyond_1ulp=int(np.sum(d > ulp)),
              log_nhi_bitwise_here=int(np.sum(lo == ln)))
    dcls = _class_of(np.log10(np.maximum(o["NHI"][dup], 1e1)))
    dups = {c: int(np.sum(dcls == c)) for c in CLASSES}
    hits = [r for r in trace_rows if r.get("tier") == tier and r.get("sim") == sim and int(r.get("snap")) == snap]
    check("duplicates_per_class_equal_trace", len(hits) == 1 and all(dups[c] == hits[0]["dups"].get(c, 0)
                                                                     for c in CLASSES),
          duplicates=dups, trace=hits[0]["dups"] if len(hits) == 1 else f"{len(hits)} trace rows")
    left = C.find_wrap_duplicates(w["skewer_idx"], w["pix_start"], w["pix_end"], n)
    check("no_duplicate_or_overspan_left", not left.any(), n_left=int(left.sum()))
    lab_o, lab_n = bin_sightlines_by_nhi(old, n_sk), bin_sightlines_by_nhi(new, n_sk)
    c4o, c4n = _class4(old, n_sk), _class4(new, n_sk)
    check("sightline_labels_identical", np.array_equal(lab_o, lab_n) and np.array_equal(c4o, c4n),
          fine_changed=int(np.sum(lab_o != lab_n)), class4_changed=int(np.sum(c4o != c4n)),
          sightlines_per_class4={k: int(np.sum(c4n == i)) for i, k in enumerate(("clean", "LLS", "subDLA", "DLA"))})
    want = {c: int(old_meta["n_absorbers"].get(c, 0)) - dups[c] for c in CLASSES}
    check("meta_counts_equal_old_minus_duplicates", {c: int(meta["n_absorbers"].get(c, 0)) for c in CLASSES} == want,
          new=meta["n_absorbers"], expected=want)
    edges = np.asarray(v.cddf["log_nhi_edges"])
    with np.load(old_dir / "cddf.npz", allow_pickle=False) as oc:
        old_edges, old_counts, old_path = np.asarray(oc["log_nhi_edges"]), np.asarray(oc["n_absorbers"]), \
            float(oc["total_path"])
    exp = np.histogram(np.log10(np.maximum(o["NHI"][~dup], 1e1)), bins=edges)[0]
    check("cddf_counts_equal_old_minus_duplicates_per_bin",
          np.array_equal(np.asarray(v.cddf["n_absorbers"]), exp) and np.array_equal(edges, old_edges)
          and float(v.cddf["total_path"]) == old_path,
          bins_differing_from_old=int(np.sum(np.asarray(v.cddf["n_absorbers"]) != old_counts)))
    rec["info"] = {"old_cddf_equals_old_catalogue_histogram": bool(np.array_equal(
        old_counts, np.histogram(np.log10(np.maximum(o["NHI"], 1e1)), bins=old_edges)[0])),
        "new_products": v.products, "snapshot_record_eligible": v.record["release_eligible"]}
    rec["accepted"] = all(c["ok"] for c in ck.values())
    return rec


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def cmd_check(a):
    groups = read_groups(a.groups)
    todo = [groups[a.index - 1]] if a.index is not None else groups
    trace = json.loads(Path(a.trace_rows).read_text())
    trace_sha = L.sha256_file(a.trace_rows)
    bad = 0
    for sim, snap in todo:
        rec = check_group(sim, snap, old_root=a.old_root, new_root=a.new_root, raw_root=a.raw_root,
                          trace_rows=trace, rehash_raw=a.rehash_raw, trace_sha256=trace_sha)
        rec["checked_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec["code"] = L.code_identity()
        _write(group_file(a.out, sim, snap), rec)
        bad += not rec["accepted"]
        print(f"{'ACCEPTED' if rec['accepted'] else 'REFUSED '} {sim} {snap}"
              + ("" if rec["accepted"] else ": " + ", ".join(k for k, c in rec["checks"].items() if not c["ok"])))
    return 1 if bad else 0


def _trace_row(trace, sim, snap, tier="LF"):
    hits = [r for r in trace if r.get("tier") == tier and r.get("sim") == sim and int(r.get("snap")) == snap]
    return hits[0]["dups"] if len(hits) == 1 else None


def cmd_summarize(a):
    groups = read_groups(a.groups)
    trace = json.loads(Path(a.trace_rows).read_text())
    trace_sha = L.sha256_file(a.trace_rows)
    missing, refused, files, totals = [], [], {}, {c: 0 for c in CLASSES}
    commits, digests, checkers = set(), set(), set()
    not_rehashed, trace_mismatch, off_commit = [], [], []
    for sim, snap in groups:
        p = group_file(a.out, sim, snap)
        if not p.exists():
            missing.append(f"{sim} {snap}")
            continue
        rec = json.loads(p.read_text())
        files[str(p.relative_to(a.out))] = L.sha256_file(p)
        if not rec.get("accepted"):
            refused.append(f"{sim} {snap}: " + ", ".join(k for k, c in rec["checks"].items() if not c["ok"]))
            continue
        for c in CLASSES:
            totals[c] += rec["checks"]["duplicates_per_class_equal_trace"]["duplicates"][c]
        lr = rec["checks"]["release_loader"]
        commits.add(json.dumps(lr["code"], sort_keys=True))
        digests.add(json.dumps([lr["estimator_source_sha256"], lr["source_sha256"]], sort_keys=True))
        checkers.add(json.dumps(rec.get("code"), sort_keys=True))
        if lr.get("rehashed") is not True:
            not_rehashed.append(f"{sim} {snap}")
        recorded = (rec.get("inputs") or {}).get("trace_sha256")
        want = _trace_row(trace, sim, snap)
        if (recorded is not None and recorded != trace_sha) or want is None or \
                rec["checks"]["duplicates_per_class_equal_trace"].get("trace") != want:
            trace_mismatch.append(f"{sim} {snap}")
        if a.expect_commit and not str((lr.get("code") or {}).get("commit") or "").startswith(a.expect_commit):
            off_commit.append(f"{sim} {snap}")
    checker = [json.loads(c) for c in checkers]
    clean_checker = len(checker) == 1 and isinstance(checker[0], dict) and isinstance(checker[0].get("commit"), str) \
        and len(checker[0]["commit"]) == 40 and checker[0].get("dirty") is False
    summary = {"format": "hcd_priya.catalogue_acceptance/1", "n_groups": len(groups),
               "n_accepted": len(groups) - len(missing) - len(refused), "missing": missing, "refused": refused,
               "duplicates_per_class_total": totals, "code_identities": sorted(commits),
               "finder_source_digests": sorted(digests), "group_files_sha256": files,
               "group_list_sha256": L.sha256_file(a.groups), "trace_rows_sha256": trace_sha,
               "acceptance_code_identities": sorted(checkers), "not_rehashed": not_rehashed,
               "trace_mismatch": trace_mismatch, "expect_commit": a.expect_commit, "off_commit": off_commit,
               "all_accepted": not missing and not refused and len(commits) == 1 and len(digests) == 1
               and clean_checker and not not_rehashed and not trace_mismatch and not off_commit}
    _write(Path(a.out) / "summary.json", summary)
    print(f"{summary['n_accepted']}/{len(groups)} accepted; {len(missing)} missing, {len(refused)} refused; "
          f"{len(commits)} code identity, {len(digests)} finder digest set(s); {len(checkers)} acceptance identity "
          f"({'clean' if clean_checker else 'NOT one clean commit'}); {len(not_rehashed)} not re-hashed; "
          f"{len(trace_mismatch)} trace mismatch(es); {len(off_commit)} off the expected commit")
    return 0 if summary["all_accepted"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--groups", type=Path, required=True)
    c.add_argument("--old-root", type=Path, required=True)
    c.add_argument("--new-root", type=Path, required=True)
    c.add_argument("--raw-root", type=Path, required=True)
    c.add_argument("--trace-rows", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--index", type=int, default=None, help="1-based line of the group list (an array task)")
    c.add_argument("--rehash-raw", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--groups", type=Path, required=True)
    s.add_argument("--trace-rows", type=Path, required=True, help="the trace file every group must agree with")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--expect-commit", default=None, help="the rebuilt catalogues' code commit (prefix)")
    a = ap.parse_args(argv)
    return cmd_check(a) if a.cmd == "check" else cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
