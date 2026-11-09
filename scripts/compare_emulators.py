#!/usr/bin/env python3
"""Run one level of a frozen cross-emulator comparison spec (hcd_analysis/comparison/<spec>.json) into a NEW directory.

  python scripts/compare_emulators.py --level {L1b,L1c,L2-LF} --out NEW_DIR --cache LF_CACHE.h5 \
      --ks-basedir RO_COPY --lyaemu-parent RO_LYAEMU_PARENT --ks-python REF_VENV_PYTHON [--ks-master MASTER_DIR]

L1b  harness self-consistency: record JSON round trip, k round trip, the reference adapter's per-z assembly against
     the reference's own MultiBinGP.predict, and the reference's own k conversion at hcd_priya-computed nodes.
L1c  data lineage: hcd_priya cache P_tier_p against the reference's LF flux vectors, every matched row.
L2-LF  hcd_priya LF ensemble (C1) against the reference AR1 LF output (R1, res 0) on families D and I.
Every result (records, residuals, summaries) is written under --out only; the terminal gets verdicts. The reference
runs in its own interpreter through hcd_analysis/comparison/adapters/lyaemu_priya.py, which refuses to refit it.

Exit status: 0 only for a PASS with the reference master tree unchanged (or not hashed). When a level returns, the
verdict is written to result.json and FAIL, NOT RUN, BLOCKED and NOT COMPARABLE each exit with their own non-zero code
(EXIT_CODES); a reference master tree that changed during the run turns the verdict into FAIL and exits with
EXIT_MASTER_CHANGED. A refusal ("refusing: ...") or an uncaught error exits 1, Python's default, without writing
result.json, so exit 1 alone does not mean that a comparison ran: read result.json."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hcd_analysis.comparison import cases as CS  # noqa: E402
from hcd_analysis.comparison import conventions as C  # noqa: E402
from hcd_analysis.comparison import metrics as M  # noqa: E402
from hcd_analysis.comparison import schema as S  # noqa: E402
from hcd_analysis.comparison.adapters import lace  # noqa: E402

KS_LF = "mf_emulator_flux_vectors_tau1000000.hdf5"
EXIT_CODES = {"PASS": 0, "FAIL": 1, "NOT RUN": 3, "BLOCKED": 4, "NOT COMPARABLE": 5}   # 2 is argparse's usage error
EXIT_MASTER_CHANGED = 6
EXIT_UNKNOWN_VERDICT = 7
_KS_DESIGN = None                     # the reference's LF design parameters (context strata only)


def _git_head():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def ks_design(basedir):
    import h5py
    with open(os.path.join(basedir, "emulator_params.json")) as f:
        ep = json.load(f)
    with h5py.File(os.path.join(basedir, KS_LF), "r") as h:
        d = {k: h[k][...] for k in ("params", "flux_vectors", "zout", "kfkms", "kfmpc")}
    d["param_index"] = ep["param_names"]
    d["param_limits"] = ep["param_limits"]
    d["rungs"] = np.unique(d["params"][:, 0])
    return d


def build_cases(cfg, ks):
    from hcd_analysis.emulator.data import PARAM_LIMITS
    limits = CS.limits_by_name(PARAM_LIMITS)
    names = sorted(ks["param_index"], key=ks["param_index"].get)
    if names != list(C.PARAM_NAMES) or not np.array_equal(np.asarray(ks["param_limits"], float), PARAM_LIMITS):
        sys.exit("refusing: the reference parameter names or limits differ from hcd_priya's (no guessing)")
    nsim = ks["params"].shape[0] // ks["rungs"].size
    if not np.all(ks["params"][:nsim, 0] == ks["rungs"][0]):
        sys.exit("refusing: reference rows are not ordered [rung][sim]")
    cases = CS.family_D(cfg, ks["params"][:nsim, 1:], ks["param_index"], ks["rungs"]) + CS.family_I(cfg, limits)
    for c in cases:
        c["support"] = CS.domain_support(c, cfg, limits)
    return cases


def run_reference(a, cfg, cases, res, checks, out):
    req = os.path.join(a.out, f"request_R1_res{res}.json")
    with open(req, "x") as f:
        json.dump({"pins": cfg["comparators"]["R1"], "res": res, "config": cfg, "comparator_id": "R1",
                   "cases": cases}, f)
    cmd = [a.ks_python, "-I", "-B", os.path.join(REPO, "hcd_analysis/comparison/adapters/lyaemu_priya.py"), "--repo", REPO,
           "--lyaemu-parent", a.lyaemu_parent, "--basedir", a.ks_basedir, "--request", req, "--out", out]
    env = dict(os.environ, MPLCONFIGDIR=os.path.join(a.out, "mplconfig"))
    r = subprocess.run(cmd + (["--checks"] if checks else []), capture_output=True, text=True, env=env, cwd=a.out)
    with open(os.path.join(a.out, f"adapter_R1_res{res}.log"), "x") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode:
        return None, r.stderr[-1500:]
    with open(out) as f:
        return json.load(f), ""


def tree_hash(root):
    from hcd_analysis.comparison.adapters.lyaemu_priya import tree_sha256
    return tree_sha256(root) if root else None


def level_L1b(a, cfg, cases, res_out):
    from hcd_analysis.comparison.adapters import hcd_priya as H
    th = cfg["thresholds"]
    fam_i = [c for c in cases if c["family"] == "I"]
    from hcd_analysis.emulator.kcoord import k_skm_from_kcom as kcoord_k_skm
    rec = H.predict(fam_i, cache_path=a.cache, manifest_path=a.manifest, pins=cfg["comparators"]["C1"])
    with open(os.path.join(a.out, "record_C1_familyI.json"), "x") as f:
        f.write(S.dumps(rec))
    if {c["case_id"] for c in rec["cases"]} != {c["case_id"] for c in fam_i}:
        res_out.update(verdict="NOT RUN", reason="the hcd_priya record does not hold every declared family-I case")
        return
    checks = {"json_round_trip_exact": S.loads(S.dumps(rec)) == rec}
    kmax, rt = 0.0, 0.0
    kc = C.k_com_modes(np.asarray(rec["modes"]), rec["L_box_hmpc"])
    for c in rec["cases"]:
        if c["support"]["status"] != "COMPARABLE":
            continue
        p = c["state"]["params"]
        for zi, ks in zip(c["z"], c["k_skm"]):
            ref = C.k_skm_from_kcom(kc, zi, p["hub"], p["omegamh2"])
            kmax = max(kmax, float(np.max(np.abs(np.asarray(ks) / ref - 1))))
            fwd = np.asarray(kcoord_k_skm(kc, zi, p["hub"], p["omegamh2"]))          # the deployed forward formula
            rt = max(rt, float(np.max(np.abs(C.kcom_from_kskm(fwd, zi, p["hub"], p["omegamh2"]) / kc - 1))))
    checks["C1_kskm_vs_conventions_max_rel"] = kmax
    checks["k_round_trip_max_rel"] = rt
    ref, err = run_reference(a, cfg, fam_i, 1, True, os.path.join(a.out, "record_R1_res1_familyI.json"))
    if ref is None:
        res_out.update(verdict="BLOCKED", reason=err, checks=checks)
        return
    checks.update({f"R1_{k}": v for k, v in ref["checks"].items()})
    n_comparable = sum(c["support"]["status"] == "COMPARABLE" for c in fam_i)
    if n_comparable == 0:                                       # nothing checked is never a pass
        res_out.update(verdict="NOT RUN", reason="no comparable family-I case", checks=checks)
        return
    passes = {"json_round_trip_exact": checks["json_round_trip_exact"],
              "C1_kskm": kmax <= th["L1b_kround"], "k_round_trip": rt <= th["L1b_kround"],
              "R1_glue_res1": ref["checks"]["glue_res1_max_rel"] <= th["L1b_glue"],
              "R1_kconv": ref["checks"]["kconv_max_rel"] <= th["L1b_kconv"],
              "R1_checked_all_cases": ref["checks"]["n_checked_cases"] == n_comparable}
    res_out.update(verdict="PASS" if all(passes.values()) else "FAIL", checks=checks, passes=passes)


def level_L1c(a, cfg, ks, res_out):
    import h5py
    th = cfg["thresholds"]
    with h5py.File(a.cache, "r") as h:
        cp = h["params"][...]
        names = [s.decode() if isinstance(s, bytes) else s for s in h["param_names"][...]]
        zg, al, P, kk = h["z_grid"][...], h["alpha_slope"][...], h["P_tier_p"][...], h["kfkms"][...]
        sims = np.array([s.decode() if isinstance(s, bytes) else s for s in h["sim_name"][...]])
        snaps = h["snap"][...]
        added = json.loads(h.attrs["repair_record"]).get("added", [])
    if names != list(C.PARAM_NAMES):
        sys.exit(f"refusing: cache param_names {names}")
    added_set = {(s, int(n)) for s, n, _ in added}
    nz, nk = ks["zout"].size, ks["kfmpc"].size
    FV = ks["flux_vectors"].reshape(-1, nz, nk)
    r_all, r_added, unmatched, pairs = [], [], [], []
    for i, row in enumerate(ks["params"]):
        sim = np.all(np.abs(cp - row[1:]) <= 1e-6 * np.abs(row[1:]), axis=1)
        for j, z in enumerate(ks["zout"]):
            m = np.where(sim & (np.abs(zg - z) <= 1e-6) & (np.abs(al - row[0]) <= 1e-6))[0]
            if m.size != 1:
                unmatched.append([i, float(z), int(m.size)])
                continue
            r = M.ratio_residual(P[m[0]], FV[i, j])
            (r_added if (sims[m[0]], int(snaps[m[0]])) in added_set else r_all).append(r)
            with np.errstate(divide="ignore", invalid="ignore"):          # 0/0 or x/0 is a non-finite label: FAIL
                dk = float(np.max(np.abs(kk[m[0]] / ks["kfkms"][i, j] - 1)))
            pairs.append((i, float(z), float(np.max(np.abs(r))), dk, str(sims[m[0]])))
    r_all, r_added = np.array(r_all), np.array(r_added)
    both = np.concatenate([r_all.ravel(), r_added.ravel()])
    np.savez(os.path.join(a.out, "L1c_residuals.npz"), r_unrepaired=r_all, r_s4_added=r_added,
             pair_ks_row=np.array([q[0] for q in pairs]), pair_z=np.array([q[1] for q in pairs]),
             pair_max_abs_r=np.array([q[2] for q in pairs]), pair_kfkms_max_rel=np.array([q[3] for q in pairs]))
    # np.max propagates NaN (Python's max(0.0, nan) is 0.0 and would drop it); "not <=" counts NaN as a failure
    kf_max = float(np.max([q[3] for q in pairs])) if pairs else 0.0
    bad = [q for q in pairs if q[2] > th["L1c_lineage"] or not q[3] <= th["L1c_kfkms"]]
    bad_sz = sorted({(q[4], q[1]) for q in bad})
    rest = [q for q in pairs if (q[4], q[1]) not in set(bad_sz)]
    lineage = "FAIL" if unmatched else M.verdict(both, th["L1c_lineage"])
    kfkms = "NOT RUN" if not pairs else ("PASS" if kf_max <= th["L1c_kfkms"] else "FAIL")
    res_out.update(
        verdict=S.level_verdict([lineage, kfkms]),      # PASS only when the power AND the k labels agree
        lineage_verdict=lineage, kfkms_verdict=kfkms, kfkms_max_rel=kf_max,
        n_rows_matched=int(r_all.shape[0] + r_added.shape[0]), n_unmatched=len(unmatched), unmatched=unmatched[:50],
        summary_all=M.summarize(both, th["L1c_lineage"]),
        summary_unrepaired=M.summarize(r_all, th["L1c_lineage"]),
        summary_s4_added=M.summarize(r_added, th["L1c_lineage"]),
        post_hoc_failing_sim_z=[list(x) for x in bad_sz],
        post_hoc_outside_failing_sim_z={"n_pairs": len(rest), "max_abs_r": max((q[2] for q in rest), default=None),
                                       "kfkms_max_rel": max((q[3] for q in rest), default=None)})


def family_verdicts(cmp, cases):
    """Per-family verdicts over the DECLARED case list (a family the adapters dropped is reported, never lost)."""
    return S.group_verdicts(cmp["cases"], {c["case_id"]: c["family"] for c in cases})


def _strata(rec_test, rec_ref, cmp, cfg, cases):
    th = cfg["thresholds"]["L2_priya"]
    tc = {c["case_id"]: c for c in rec_test["cases"]}
    rc = {c["case_id"]: c for c in rec_ref["cases"]}
    edges = cfg["k_bands_skm"]
    out = {"by_case": {}, "by_z": {}, "by_k_band": {}, "by_family": {}, "data_range": None, "within_2sigma_ref": None,
           "post_hoc_context": {}}
    acc = {"z": {}, "band": {}, "fam": {}, "dr": [], "sig": [], "sigfam": {}, "argmax": {}, "famband": {}}
    for cid, r in cmp["r"].items():
        out["by_case"][cid] = M.summarize(r, th)
        K = np.asarray(tc[cid]["k_skm"])
        b = M.band_index(K, edges)
        sd = np.asarray(rc[cid]["native"]["std"])
        dP = np.abs(np.asarray(tc[cid]["P"]) - np.asarray(rc[cid]["P"]))
        fam = tc[cid]["family"]
        acc["sig"].append((dP <= 2 * sd).ravel())
        acc["sigfam"].setdefault(fam, []).append((dP <= 2 * sd).ravel())
        acc["fam"].setdefault(fam, []).append(r.ravel())
        i, j = np.unravel_index(int(np.argmax(np.abs(r))), r.shape)
        best = acc["argmax"].get(fam)
        if best is None or abs(r[i, j]) > abs(best["r"]):
            acc["argmax"][fam] = {"case": cid, "z": tc[cid]["z"][i], "mode": int(j) + 1, "k_skm": float(K[i, j]),
                                  "r": float(r[i, j])}
        for jb in range(len(edges)):
            acc["famband"].setdefault(f"{fam}|{jb}", []).append(r[b == jb])
        acc["dr"].append(r[K >= cfg["data_range_kmin_skm"]])
        for i, z in enumerate(tc[cid]["z"]):
            acc["z"].setdefault(f"{z:.1f}", []).append(r[i])
        for j in range(len(edges)):
            acc["band"].setdefault(str(j), []).append(r[b == j])
    for z, v in acc["z"].items():
        out["by_z"][z] = M.summarize(np.concatenate(v), th)
    for j, v in acc["band"].items():
        lo = edges[int(j)]
        hi = edges[int(j) + 1] if int(j) + 1 < len(edges) else None
        out["by_k_band"][f"[{lo}, {hi if hi is not None else 'inf'})"] = M.summarize(np.concatenate(v), th)
    groups = family_verdicts(cmp, cases)
    for f, g in groups.items():                                  # every family, with the level precedence
        allr = np.concatenate(acc["fam"][f]) if f in acc["fam"] else np.array([])
        out["by_family"][f] = dict(M.summarize(allr, th), **g)
    if acc["dr"]:
        out["data_range"] = M.summarize(np.concatenate(acc["dr"]), th)
        out["within_2sigma_ref"] = float(np.mean(np.concatenate(acc["sig"])))
    # context only (never verdict-changing): per family, where the largest residual sits, family x band
    pc = out["post_hoc_context"]
    pc["note"] = "post hoc (decided after the phase0_v1 results); never verdict-changing"
    pc["within_2sigma_ref_by_family"] = {f: float(np.mean(np.concatenate(v))) for f, v in acc["sigfam"].items()}
    pc["argmax_by_family"] = acc["argmax"]
    pc["family_by_k_band"] = {k: M.summarize(np.concatenate(v), th) for k, v in acc["famband"].items()}
    pc.update(_weights_and_support_context(rec_test, rec_ref, cmp, cfg))
    return out


def _weights_and_support_context(rec_test, rec_ref, cmp, cfg):
    """Context only: family D recomputed with the emulated Head-A weights instead of the cache weights; family I
    split by the sampling box; each family-I point's unit-cube distance to the nearest LF design point."""
    from hcd_analysis.emulator.data import PARAM_LIMITS, SAMPLING_LIMITS, normalize_params
    th = cfg["thresholds"]["L2_priya"]
    tc = {c["case_id"]: c for c in rec_test["cases"]}
    rc = {c["case_id"]: c for c in rec_ref["cases"]}
    rD, dw = [], []
    for cid, c in tc.items():
        if c["family"] != "D" or cid not in cmp["r"]:
            continue
        nat = c["native"]
        pf, wh, wc = (np.asarray(nat[k]) for k in ("P_filt_by_class", "w_c_head_a", "w_c"))
        rD.append(np.einsum("zc,zck->zk", wh, pf) / np.asarray(rc[cid]["P"]) - 1)
        dw.append(np.abs(wh - wc))
    sl = np.asarray(SAMPLING_LIMITS, float)
    inside, outside = [], []
    for cid, c in tc.items():
        if c["family"] == "I" and cid in cmp["r"]:
            v = np.array([c["state"]["params"][n] for n in C.PARAM_NAMES])
            (inside if np.all((v >= sl[:, 0]) & (v <= sl[:, 1])) else outside).append(cmp["r"][cid].ravel())
    design = np.asarray(normalize_params(np.asarray(_KS_DESIGN, float)), float) if _KS_DESIGN is not None else None
    dist = {}
    for cid, c in tc.items():
        pid = cid.split("-")[0]
        if c["family"] == "I" and pid not in dist and design is not None:
            u = np.asarray(normalize_params(np.array([c["state"]["params"][n] for n in C.PARAM_NAMES])), float)
            dist[pid] = float(np.min(np.linalg.norm(design - u, axis=1)))
    return {"family_D_with_head_a_weights": M.summarize(np.concatenate([x.ravel() for x in rD]), th) if rD else None,
            "family_D_head_a_minus_cache_w_c_max_by_class": (np.max(np.concatenate(dw), axis=0).tolist() if dw else None),
            "family_I_inside_sampling_limits": M.summarize(np.concatenate(inside), th) if inside else {"n": 0},
            "family_I_outside_sampling_limits": M.summarize(np.concatenate(outside), th) if outside else {"n": 0},
            "family_I_nearest_design_unit_distance": dist,
            "param_limits_equal_sampling_box": bool(np.array_equal(np.asarray(PARAM_LIMITS), sl))}


def level_L2LF(a, cfg, cases, res_out):
    from hcd_analysis.comparison.adapters import hcd_priya as H
    th = cfg["thresholds"]["L2_priya"]
    rec = H.predict(cases, cache_path=a.cache, manifest_path=a.manifest, pins=cfg["comparators"]["C1"])
    with open(os.path.join(a.out, "record_C1.json"), "x") as f:
        f.write(S.dumps(rec))
    ref, err = run_reference(a, cfg, cases, 0, False, os.path.join(a.out, "record_R1_res0.json"))
    comps = cfg["comparators"]
    statuses = {"R1-LFshipped": "NOT COMPARABLE" if "lf_shipped_path" in comps["R1"] else "NOT RUN",
                "R2": comps["R2"]["status"], "R3": comps["R3"]["status"], "L2-MF": "NOT RUN"}
    declared = {c["case_id"]: c["support"]["status"] for c in cases}
    lace_cmp = S.compare_records(rec, lace.blocked_record(cases, pins=cfg["comparators"]["R3"],
                                                          n_modes=cfg["n_modes"], L_box_hmpc=cfg["L_box_hmpc"]),
                                 cfg["thresholds"]["L3_lace"], declared=declared)
    statuses["R3_compare_records"] = lace_cmp["verdict"]
    if ref is None:
        res_out.update(verdict="BLOCKED", reason=err, other=statuses)
        return
    cmp = S.compare_records(rec, ref["record"], th, declared=declared)
    strata = _strata(rec, ref["record"], cmp, cfg, cases)
    np.savez(os.path.join(a.out, "L2LF_residuals.npz"), **{k: v for k, v in cmp["r"].items()})
    res_out.update(verdict=cmp["verdict"], complete=cmp["complete"], max_abs=cmp["max_abs"], points=cmp["points"],
                   case_outcomes=cmp["cases"],
                   summary=cmp["summary"], strata=strata, other=statuses)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", required=True, choices=("L1b", "L1c", "L2-LF"))
    ap.add_argument("--spec", default="phase0_v1")
    for k in ("--out", "--cache", "--ks-basedir", "--lyaemu-parent", "--ks-python"):
        ap.add_argument(k, required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ks-master", default=None, help="reference master tree to hash before and after")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"refusing: {a.out} exists")
    cfg = CS.load_config(a.spec)
    master_before = tree_hash(a.ks_master)
    os.makedirs(a.out)
    ks = ks_design(a.ks_basedir)
    cases = build_cases(cfg, ks)
    global _KS_DESIGN
    nsim = ks["params"].shape[0] // ks["rungs"].size
    _KS_DESIGN = ks["params"][:nsim, 1:]
    with open(os.path.join(a.out, "cases.json"), "x") as f:
        json.dump(cases, f, indent=1)
    res = {"spec": a.spec, "level": a.level, "code_commit": _git_head(),
           "n_cases": len(cases), "n_not_comparable": sum(c["support"]["status"] != "COMPARABLE" for c in cases)}
    {"L1b": lambda: level_L1b(a, cfg, cases, res), "L1c": lambda: level_L1c(a, cfg, ks, res),
     "L2-LF": lambda: level_L2LF(a, cfg, cases, res)}[a.level]()
    res["ks_master_unchanged"] = (tree_hash(a.ks_master) == master_before) if a.ks_master else None
    res.setdefault("verdict", "NOT RUN")                       # a level that set no verdict ran nothing checkable
    if res["ks_master_unchanged"] is False:                    # the reference changed under the run: never a pass
        res["level_verdict"] = res["verdict"]
        res["verdict"] = "FAIL"
        res["master_changed_reason"] = "the reference master tree hash changed during the run"
    with open(os.path.join(a.out, "result.json"), "x") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=float)
    print(f"{a.level}: {res['verdict']}" + (f" (reference master unchanged: {res['ks_master_unchanged']})"
                                           if a.ks_master else ""))
    if "level_verdict" in res:
        print(f"  level_verdict={res['level_verdict']} (overridden: {res['master_changed_reason']})")
    sub = {k: res[k] for k in ("lineage_verdict", "kfkms_verdict", "kfkms_max_rel") if k in res}
    if sub:
        print("  " + ", ".join(f"{x}={y}" for x, y in sub.items()))
    for k in ("passes", "other", "case_outcomes"):
        if k in res:
            vals = res[k]
            print(f"  {k}: " + (", ".join(f"{x}={y}" for x, y in vals.items()) if k != "case_outcomes" else
                                ", ".join(f"{v}:{list(vals.values()).count(v)}" for v in sorted(set(vals.values())))))
    sys.exit(exit_status(res))


def exit_status(res):
    """0 only for PASS with the master unchanged or not hashed; a distinct non-zero code otherwise."""
    if res.get("ks_master_unchanged") is False:
        return EXIT_MASTER_CHANGED
    return EXIT_CODES.get(res.get("verdict"), EXIT_UNKNOWN_VERDICT)


if __name__ == "__main__":
    main()
