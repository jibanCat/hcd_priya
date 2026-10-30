"""The stratified fresh-row sample of the R1 carry: its pre-registration, and the comparison of the freshly built rows
with the carried ones.

``select`` writes the pre-registration (JSON), before any rebuilt row exists. It reads only the group list of the
historical cache and chooses groups by the rule written into the file:
  1. spliced origin: every group the historical cache's repair record lists as added;
  2. z extremes: one base-origin group at the lowest and one at the highest z_grid, each drawn (seeded) among the
     base-origin groups at that z_grid;
  3. a named simulation: if no group chosen so far belongs to a simulation whose name starts with --include-sim-prefix,
     one base-origin group of it, drawn (seeded);
  4. the remainder: base-origin groups drawn (seeded) from simulations not yet chosen, until --n-total groups.
Each draw is ``numpy.random.default_rng(seed).choice`` over the candidates in canonical (sim, snap) order, with one
generator used in the order above. The file records the rule, the parameters, the historical cache's sha256, this
script's sha256 and the code identity, and the acceptance criterion.

``compare`` checks freshly built caches (scripts/build_emulator_cache_tau0.py --select, from the rebuilt catalogues
and the raw spectra: every field and all mean-flux rungs computed, scale and target_F included) against the carried
cache, group by group. Both are read with the release reader; the fresh groups must be exactly the pre-registered
ones and must name the same rebuilt catalogues as the carried cache. Criterion (pre-registered): for every float row
field, per row, max|fresh - carried| / max|carried| <= rtol with the same NaN positions; integer and string fields
exactly equal; the snapshot block exactly equal. Bitwise equality is reported per field. The report gives the largest
deviation per field.

Usage:
    python scripts/fresh_row_sample.py select --cache HISTORICAL.h5 --seed S --n-total N \\
        --include-sim-prefix PREFIX --out PREREGISTRATION.json
    python scripts/fresh_row_sample.py compare --prereg PREREGISTRATION.json --prereg-sha256 SHA \\
        --carried CARRIED.h5 --fresh FRESH.h5 [--fresh ...] --out REPORT.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402

FORMAT = "hcd_priya.fresh_row_preregistration/1"
RTOL = 1e-12
RULE = [
    "1. spliced origin: every group the historical cache's repair record lists as added, in canonical order",
    "2. z extremes: one base-origin group at the lowest z_grid and one at the highest z_grid, each drawn uniformly "
    "among the base-origin groups at that z_grid",
    "3. named simulation: if no group chosen so far belongs to a simulation whose name starts with the include prefix, "
    "one base-origin group of it, drawn uniformly",
    "4. remainder: base-origin groups drawn uniformly without replacement from simulations not chosen yet, until "
    "n_total groups (from any unchosen group once every simulation is chosen)",
    "draws: one numpy.random.default_rng(seed) Generator, Generator.choice over the candidates in canonical "
    "(sim, snap) order, used in the order of the steps",
]
CRITERION = {
    "fields": "every row field of the group's rows (all mean-flux rungs) and every field of its snapshot block",
    "float_row_fields": f"per row: max|fresh - carried| / max|carried| <= {RTOL}, NaN at the same positions",
    "integer_and_string_fields": "exactly equal",
    "snapshot_block": "exactly equal (both are rebuilt from the same rebuilt catalogue and meta.json)",
    "reported": "bitwise equality per field; the largest deviation per field over all groups",
    "rtol": RTOL,
}


# ------------------------------------------------------------------------------------------------------ select
def select_groups(groups, added, *, seed, n_total, include_sim_prefix):
    """The pre-registered sample: ``[(sim, snap, z_grid, origin, stratum)]``. ``groups``: [(sim, snap, z_grid)] of
    the historical cache; ``added``: the (sim, snap) its repair record lists as added."""
    rng = np.random.default_rng(seed)
    added = {(str(s), int(n)) for s, n in added}
    groups = sorted((str(s), int(n), float(z)) for s, n, z in groups)
    chosen = []

    def pick(cands, stratum):
        cands = [g for g in cands if (g[0], g[1]) not in {(c[0], c[1]) for c in chosen}]
        if cands:
            g = cands[int(rng.choice(len(cands)))]
            chosen.append((g[0], g[1], g[2], "spliced" if (g[0], g[1]) in added else "base", stratum))
    for g in groups:
        if (g[0], g[1]) in added:
            chosen.append((g[0], g[1], g[2], "spliced", "spliced origin"))
    base = [g for g in groups if (g[0], g[1]) not in added]
    zs = sorted({round(g[2], 6) for g in base})
    if zs:
        pick([g for g in base if round(g[2], 6) == zs[0]], "lowest z_grid")
        pick([g for g in base if round(g[2], 6) == zs[-1]], "highest z_grid")
    if include_sim_prefix and not any(c[0].startswith(include_sim_prefix) for c in chosen):
        pick([g for g in base if g[0].startswith(include_sim_prefix)], f"simulation {include_sim_prefix}")
    while len(chosen) < n_total:
        sims = {c[0] for c in chosen}
        free = [g for g in base if g[0] not in sims] or base
        before = len(chosen)
        pick(free, "remainder")
        if len(chosen) == before:
            break
    return chosen


def cmd_select(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a pre-registration is never rewritten")
    reason = "fresh-row sample selection: the group list only"
    groups, attrs = T.read_groups_historical(a.cache, reason=reason)
    added = []
    if "repair_record" in attrs:
        added = [(s, n) for s, n, _z in json.loads(T._attr_value(attrs["repair_record"])).get("added", [])]
    chosen = select_groups([(s, n, z) for s, n, z, _r in groups], added, seed=a.seed, n_total=a.n_total,
                           include_sim_prefix=a.include_sim_prefix)
    ident = L.file_identity(a.cache)
    doc = {
        "format": FORMAT,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "groups whose rows are rebuilt from raw spectra and compared with the carried rows",
        "historical_cache": {"path": str(a.cache), "sha256": ident["sha256"], "size": ident["size"],
                             "cache_version": str(T._attr_value(attrs.get("cache_version"))),
                             "n_groups": len(groups), "repair_record_added": [list(x) for x in added]},
        "rule": RULE,
        "parameters": {"seed": int(a.seed), "n_total": int(a.n_total), "include_sim_prefix": a.include_sim_prefix},
        "criterion": CRITERION,
        "groups": [{"sim": s, "snap": n, "z_grid": z, "origin": o, "stratum": st} for s, n, z, o, st in chosen],
        "selector": "SIM:SNAP strings for build_emulator_cache_tau0.py --select (a full simulation name)",
        "generator": {"script": "scripts/fresh_row_sample.py", "script_sha256": L.sha256_file(__file__),
                      "code": L.code_identity()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} sha256 {L.sha256_file(out)}: {len(chosen)} groups")
    for c in chosen:
        print(f"  {c[3]:8s} {c[4]:22s} {c[0]} {c[1]} z={c[2]:.1f}")
    return 0


# ----------------------------------------------------------------------------------------------------- compare
def _rows_of(d, sim, snap):
    rows = np.flatnonzero((np.asarray(d["sim_name"]).astype(str) == sim) & (np.asarray(d["snap"]) == snap))
    order = np.argsort(np.asarray(d["alpha_idx"])[rows], kind="stable")
    return rows[order]


def _block_of(d, sim, snap):
    hit = np.flatnonzero((np.asarray(d["snap_sim_name"]).astype(str) == sim) & (np.asarray(d["snap_snap"]) == snap))
    return int(hit[0]) if hit.size == 1 else None


def compare_field(a, b):
    """``(ok, bitwise, max_rel)`` of a row field (leading axis = rows) under the pre-registered criterion."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, False, float("inf")
    if a.dtype.kind not in "fc":
        eq = bool(np.array_equal(a.astype(str) if a.dtype.kind == "O" else a, b.astype(str) if b.dtype.kind == "O"
                                 else b))
        return eq, eq, 0.0 if eq else float("inf")
    bitwise = bool(np.array_equal(a, b, equal_nan=True))
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False, bitwise, float("inf")
    fa, fb = np.where(np.isnan(a), 0.0, a).reshape(a.shape[0], -1), np.where(np.isnan(b), 0.0, b).reshape(
        b.shape[0], -1)
    scale = np.max(np.abs(fb), axis=1)
    diff = np.max(np.abs(fa - fb), axis=1)
    rel = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), np.where(diff > 0, np.inf, 0.0))
    m = float(np.max(rel)) if rel.size else 0.0
    return m <= RTOL, bitwise, m


def cmd_compare(a):
    pre_path = Path(a.prereg)
    if L.sha256_file(pre_path) != a.prereg_sha256:
        raise SystemExit(f"{pre_path}: sha256 differs from the registered {a.prereg_sha256}")
    pre = json.loads(pre_path.read_text())
    if pre.get("format") != FORMAT or pre["criterion"]["rtol"] != RTOL:
        raise SystemExit(f"{pre_path}: not a {FORMAT} file with this script's criterion")
    want = [(g["sim"], int(g["snap"])) for g in pre["groups"]]
    car, _ca, car_rec = T.read_cache_staged(a.carried)                # the staged carry, before its promotion
    source = (car_rec["product"].get("carried") or {}).get("source_sha256")
    if source is None or (pre.get("historical_cache") or {}).get("sha256") != source:
        raise SystemExit(f"{pre_path} does not name the carried cache's source ({source}); it pre-registers "
                         f"{(pre.get('historical_cache') or {}).get('sha256')}")
    fresh = [T.read_cache(p) for p in a.fresh]
    car_inputs = {i["role"]: i["sha256"] for i in car_rec["inputs"]}
    found, report, worst = {}, {"groups": [], "fields": {}}, {}
    for (d, _attrs, rec), p in zip(fresh, a.fresh):
        for g in rec["product"]["groups"]:
            key = (g["sim"], int(g["snap"]))
            if key in found:
                raise SystemExit(f"group {key} built twice ({found[key][1]} and {p})")
            found[key] = (d, p, {i["role"]: i["sha256"] for i in rec["inputs"]})
    if sorted(found) != sorted(want):
        raise SystemExit(f"the fresh groups {sorted(found)} are not the pre-registered {sorted(want)}")
    ok_all = True
    for sim, snap in want:
        d, p, inputs = found[(sim, snap)]
        role = T.group_role("catalog", sim, snap)
        same_cat = inputs.get(role) is not None and inputs.get(role) == car_inputs.get(role)
        rf, rc = _rows_of(d, sim, snap), _rows_of(car, sim, snap)
        g = {"sim": sim, "snap": snap, "fresh_file": str(p), "same_rebuilt_catalogue": same_cat,
             "n_rows": [int(rf.size), int(rc.size)], "fields": {}}
        ok = same_cat and rf.size == rc.size and rf.size > 0 and \
            np.array_equal(np.asarray(d["alpha_idx"])[rf], np.asarray(car["alpha_idx"])[rc])
        for k in T.ROW_KEYS:
            if k == "snap_group_idx" or not rf.size == rc.size:
                continue
            fok, bit, m = compare_field(np.asarray(d[k])[rf], np.asarray(car[k])[rc])
            g["fields"][k] = {"ok": fok, "bitwise": bit, "max_rel": m}
            ok &= fok
        bf, bc = _block_of(d, sim, snap), _block_of(car, sim, snap)
        for k in T.SNAP_KEYS:
            eq = bf is not None and bc is not None and compare_field(np.asarray(d[k])[bf:bf + 1],
                                                                     np.asarray(car[k])[bc:bc + 1])[1]
            g["fields"][k] = {"ok": eq, "bitwise": eq, "max_rel": 0.0 if eq else float("inf")}
            ok &= eq
        for k, f in g["fields"].items():
            w = worst.setdefault(k, {"max_rel": 0.0, "bitwise_all": True})
            w["max_rel"] = max(w["max_rel"], f["max_rel"])
            w["bitwise_all"] &= f["bitwise"]
        g["ok"] = bool(ok)
        ok_all &= bool(ok)
        report["groups"].append(g)
    report.update(fields=worst, passed=bool(ok_all), criterion=pre["criterion"], source_sha256=source,
                  prereg={"path": str(pre_path), "sha256": a.prereg_sha256},
                  carried={"path": str(a.carried), "sha256": L.sha256_file(a.carried)},
                  fresh=[{"path": str(p), "sha256": L.sha256_file(p)} for p in a.fresh], code=L.code_identity())
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    print(f"{'PASSED' if ok_all else 'FAILED'}: {len(want)} groups; largest deviation per field:")
    for k, w in sorted(worst.items()):
        print(f"  {k:22s} max_rel {w['max_rel']:.3e}  bitwise {w['bitwise_all']}")
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--n-total", type=int, required=True)
    s.add_argument("--include-sim-prefix", default=None)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--prereg", type=Path, required=True)
    c.add_argument("--prereg-sha256", required=True)
    c.add_argument("--carried", type=Path, required=True)
    c.add_argument("--fresh", type=Path, action="append", required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return cmd_select(a) if a.cmd == "select" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
