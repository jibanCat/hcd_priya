#!/usr/bin/env python3
"""Descriptive comparison of two eBOSS real-data products of the same frozen model: the full-range fit (z = 2.2 to 4.6)
and the restricted fit (z >= 2.6) (PI DECISIONS #28, 2026-09-25, item 5).

For every column present in both products (theta, the shared tau0 rungs, the HCD amplitudes, tau0_amp / dtau0 recovered
exactly from each product's own ladder, and the nuisance sites): median, 16/84, sd for both; the shift sub minus full in
units of the full-range posterior sd and of the combined sd; rail fractions of both against the frozen prior box.
The two data vectors are NESTED (the restricted one is a subset), so the shifts are descriptive, not a tension statistic.
Writes <out>.json and <out>.md. Runs once (refuses if the outputs exist).
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def refuse(msg):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(3)


def _cons():
    spec = importlib.util.spec_from_file_location("cons", os.path.join(REPO, "scripts", "eboss_priya_consistency.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_product(cons, chain_dir, root, mode, z_lock, z_pivot):
    health = json.load(open(os.path.join(chain_dir, f"{root}.health.json")))
    if mode == "plain" and health.get("blinded") is not False:
        refuse(f"{root}: plain mode but health.blinded is not false")
    if mode == "unblinded" and health.get("blinded") is not True:
        refuse(f"{root}: unblinded mode but health.blinded is not true")
    z = np.asarray(health["z_kept"], float) if health.get("z_kept") else np.asarray(z_lock, float)
    names, tabs = cons.load_chains(chain_dir, root, mode=mode)
    rows = np.concatenate([t[:, 2:] for t in tabs], axis=0)
    tau_idx = [i for i, n in enumerate(names) if n.startswith("tau0_z")]
    if len(tau_idx) != z.size:
        refuse(f"{root}: {len(tau_idx)} tau0 columns vs {z.size} z values")
    amp, dt = cons.recover_tau0_amp_dtau0(rows[:, tau_idx], z, z_pivot=z_pivot)
    recon = amp[:, None] * ((1.0 + z)[None, :] / (1.0 + z_pivot)) ** dt[:, None] * cons.kim_tau0(z)[None, :]
    if float(np.max(np.abs(recon / rows[:, tau_idx] - 1.0))) > 1e-6:
        refuse(f"{root}: tau0 ladder is not an exact tau0_amp/dtau0 curve")
    cols = {n: rows[:, i] for i, n in enumerate(names) if not n.startswith("tau0_z")}
    for k, i in zip(z, tau_idx):                       # tau0 rungs keyed by REDSHIFT so the products align
        cols[f"tau0(z={k:.1f})"] = rows[:, i]
    cols["tau0_amp"], cols["dtau0"] = amp, dt
    for k, v in cons.load_nuisance(chain_dir, root).items():
        cols[f"nuis:{k}"] = np.asarray(v).reshape(-1)
    nj = os.path.join(chain_dir, f"{root}.nuisance.json")
    nuis_rails = {k: dict(near_lo=v.get("frac_near_lo"), near_hi=v.get("frac_near_hi")) for k, v in json.load(open(nj)).get("sites", {}).items()} if os.path.exists(nj) else {}
    return dict(health=health, z=z.tolist(), cols=cols, n_draws=int(rows.shape[0]), nuis_rails=nuis_rails)


def compare(cons, full, sub):
    out = {}
    for k in full["cols"]:
        if k not in sub["cols"]:
            continue
        a, b = np.asarray(full["cols"][k], float), np.asarray(sub["cols"][k], float)
        sa, sb = cons.summarize(a), cons.summarize(b)
        d = float(sb["median"] - sa["median"])
        d_mean = float(sb["mean"] - sa["mean"])
        rec = dict(full=sa, sub=sb, delta_sub_minus_full=d, delta_basis="median minus median (descriptive); delta_mean_sub_minus_full is mean minus mean",
                   delta_mean_sub_minus_full=d_mean,
                   delta_over_full_sd=(d_mean / sa["sd"] if sa["sd"] > 0 else None),
                   delta_over_combined_sd=(d_mean / float(np.hypot(sa["sd"], sb["sd"])) if (sa["sd"] > 0 or sb["sd"] > 0) else None),
                   delta_over_full_halfwidth1684=(d / sa["sigma"] if sa["sigma"] > 0 else None),
                   # Reviewer L S2: for nested data vectors the expected scatter of the mean difference is sqrt(sd_sub^2 - sd_full^2)
                   # (Gratton and Challinor 2019); n/a when sd_sub <= sd_full or when either product rails (> 0.05) on that column
                   delta_over_nested_scatter=None,
                   sd_ratio_sub_over_full=(sb["sd"] / sa["sd"] if sa["sd"] > 0 else None),
                   note="sd = posterior standard deviation; halfwidth1684 = (q84 - q16) / 2; nested data vectors, descriptive only")
        key = k.split(":", 1)[-1] if k.startswith("nuis:") else k
        if key in cons.BOX:
            rec["rails"] = dict(full=cons.rail_fraction(a, cons.BOX[key]), sub=cons.rail_fraction(b, cons.BOX[key]))
        elif k.startswith("nuis:") and key in full.get("nuis_rails", {}) and key in sub.get("nuis_rails", {}):
            rec["rails"] = dict(full=full["nuis_rails"][key], sub=sub["nuis_rails"][key], note="LogUniform metal-node rails from <root>.nuisance.json (2 percent of the log10 width)")
        railed = False
        if "rails" in rec:
            railed = any((v or 0) > 0.05 for side in ("full", "sub") for v in (rec["rails"][side].get("near_lo"), rec["rails"][side].get("near_hi")))
        if sb["sd"] > sa["sd"] and not railed:
            rec["delta_over_nested_scatter"] = d_mean / float(np.sqrt(sb["sd"] ** 2 - sa["sd"] ** 2))
        out[k] = rec
    return out


def to_md(res):
    L = ["# eBOSS z = 2.2 to 4.6 versus z >= 2.6 (descriptive; nested data vectors, not a tension statistic)", "",
         f"full: {res['full']['root']} ({res['full']['n_draws']} draws, z {res['full']['z'][0]} to {res['full']['z'][-1]}); "
         f"sub: {res['sub']['root']} ({res['sub']['n_draws']} draws, z {res['sub']['z'][0]} to {res['sub']['z'][-1]})", "",
         "| column | full mean, median [16, 84] | sub mean, median [16, 84] | sub minus full (mean) | / full sd | / combined sd | / nested scatter | sd ratio | rails full (lo, hi) | rails sub (lo, hi) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    def f(x): return "" if x is None else f"{x:.4g}"
    def rails(rr):
        if rr is None:
            return ""
        def g(v):
            return "n/a" if v is None else "%.3f" % v
        return "(%s, %s)" % (g(rr.get("near_lo")), g(rr.get("near_hi")))
    for k, r in res["comparison"].items():
        rf = r.get("rails", {}).get("full"); rs = r.get("rails", {}).get("sub")
        fu, su = r["full"], r["sub"]
        L.append("| %s | %.5g, %.5g [%.5g, %.5g] | %.5g, %.5g [%.5g, %.5g] | %+.4g | %s | %s | %s | %s | %s | %s |" % (
            k, fu["mean"], fu["median"], fu["q16"], fu["q84"], su["mean"], su["median"], su["q16"], su["q84"], r["delta_mean_sub_minus_full"],
            f(r["delta_over_full_sd"]), f(r["delta_over_combined_sd"]), f(r["delta_over_nested_scatter"]), f(r["sd_ratio_sub_over_full"]), rails(rf), rails(rs)))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-dir", required=True); ap.add_argument("--full-root", default="real_eboss")
    ap.add_argument("--full-mode", choices=("unblinded", "plain"), default="unblinded")
    ap.add_argument("--sub-dir", required=True); ap.add_argument("--sub-root", default="real_eboss_z26")
    ap.add_argument("--sub-mode", choices=("unblinded", "plain"), default="plain")
    ap.add_argument("--analysis-lock", required=True); ap.add_argument("--leg", default="eBOSS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout-json", default=None, help="the restricted product's consistency readout JSON (Reviewer L S1c): refuse unless its health_gate.label is GREEN or AMBER (order E11 -> E12)")
    ap.add_argument("--execution-record", default=None, help="the restricted product's EXECUTION_RECORD.json: refuse unless sha256(this script) equals records_sha256.compare_script")
    a = ap.parse_args(argv)
    if os.path.exists(a.out + ".json") or os.path.exists(a.out + ".md"):
        refuse(f"output exists: {a.out}.json/.md (runs once)")
    integrity = None
    if a.execution_record:
        import hashlib
        rec = json.load(open(a.execution_record)); me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
        if rec.get("records_sha256", {}).get("compare_script") != me:
            refuse("this compare script's sha256 differs from records_sha256.compare_script in the execution record (post-run edit?)")
        integrity = dict(execution_record=os.path.abspath(a.execution_record), compare_script_sha256=me)
    gate_of_readout = None
    if a.readout_json:
        ro = json.load(open(a.readout_json)); gate_of_readout = (ro.get("health_gate") or {}).get("label")
        if gate_of_readout not in ("GREEN", "AMBER"):
            refuse(f"the restricted product's readout gate is {gate_of_readout!r}: no comparison")
    cons = _cons()
    lock = json.load(open(a.analysis_lock)); leg = lock["legs"][a.leg]
    zp = float(leg.get("prior", {}).get("tau0_pivot_z", cons.TAU0_PIVOT_Z))
    full = load_product(cons, a.full_dir, a.full_root, a.full_mode, leg["z"], zp)
    sub = load_product(cons, a.sub_dir, a.sub_root, a.sub_mode, leg["z"], zp)
    if not set(sub["z"]) <= set(full["z"]):
        refuse("the restricted product's z grid is not a subset of the full product's")
    comp = compare(cons, full, sub)
    res = dict(schema="eboss_zrange_compare.v1", note="descriptive; the restricted data vector is a subset of the full one",
               full=dict(chain_dir=os.path.abspath(a.full_dir), root=a.full_root, z=full["z"], n_draws=full["n_draws"],
                         health={k: full["health"].get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "n_divergent", "seed")}),
               sub=dict(chain_dir=os.path.abspath(a.sub_dir), root=a.sub_root, z=sub["z"], n_draws=sub["n_draws"],
                        health={k: sub["health"].get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "n_divergent", "seed")}),
               dropped_z=[z for z in full["z"] if z not in sub["z"]], comparison=comp, integrity=integrity, readout_gate=gate_of_readout,
               units=dict(sd="posterior standard deviation (ddof 1)", halfwidth1684="(q84 - q16) / 2", nested_scatter="sqrt(sd_sub^2 - sd_full^2), n/a when sd_sub <= sd_full or a rail > 0.05"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1, sort_keys=True); f.write("\n")
    with open(a.out + ".md", "w") as f:
        f.write(to_md(res))
    key = {k: (round(v["delta_over_full_sd"], 2) if v["delta_over_full_sd"] is not None else None) for k, v in comp.items() if k in ("ns", "Ap", "tau0_amp", "dtau0")}
    print(f"wrote {a.out}.json and .md; mean shifts (sub minus full) in full-sd units: {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
