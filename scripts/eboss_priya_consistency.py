#!/usr/bin/env python3
"""eboss_priya_consistency.py -- the PREREGISTERED readout of an UNBLINDED eBOSS real-data chain against the
previous PRIYA eBOSS analysis (Fernandez, Bird & Ho 2024, Table 3, full-range chain = primary reference).

Preregistration: hcd_priya_notes/docs/superpowers/eboss-realdata-2026-09/2026-09-24-EBOSS-REALDATA-PREREGISTRATION-v1.md
(section 5 rules and attribution list). This script implements those rules verbatim and nothing else.

Inputs: ``<chain-dir>/<root>.unblinded.<c>.txt`` (+ ``.unblinded.paramnames``), ``<root>.health.json``,
optional ``<root>.nuisance.npz`` / ``.nuisance.json`` (f_res + metal nodes, unblinded by design), the frozen
reference JSON, and ``analysis.lock`` (for the eBOSS z grid). ``tau0_amp`` / ``dtau0`` are recovered EXACTLY from
the exported per-z tau0 columns because the deployed model is tau0(z) = tau0_amp * ((1+z)/(1+z_p))^dtau0 * Kim(z).

Outputs: ``<out>.json`` (everything) and ``<out>.md`` (tables). The single attribution step runs ONLY when at
least one primary/secondary label is SHIFTED or DISCREPANT (section 4 rule) and is labelled INDICATIVE.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

KIM_AMP, KIM_SLOPE = 2.3e-3, 3.65          # hcd_analysis.emulator.data (checked at runtime when importable)
TAU0_PIVOT_Z = 3.0
PRIMARY = ("ns", "Ap")
SECONDARY = ("tau0_amp", "dtau0")
# Section 3 rules (frozen): scaled thresholds and absolute bands per parameter.
# Preregistration v1.1 section 5 (the rules the PI adopted in the three-lanes response Part 2.3):
#   DISCREPANT (evaluated FIRST) if |delta| > discrepant_abs OR |delta| > k_discrepant * s;
#   else CONSISTENT if |delta| <= max(k_consistent * s, consistent_abs); else SHIFTED.
# s = sqrt(sigma_ours^2 + sigma_ref^2) is a combined-width unit, NOT a tension statistic (both fits share the data).
RULES = {
    "ns":       dict(consistent_abs=0.015, discrepant_abs=0.05, k_consistent=1.0, k_discrepant=3.0),
    "tau0_amp": dict(consistent_abs=0.03,  discrepant_abs=0.10, k_consistent=1.0, k_discrepant=3.0),
    "dtau0":    dict(consistent_abs=0.06,  discrepant_abs=0.20, k_consistent=1.0, k_discrepant=3.0),
}
AP_RULE = dict(upper68=1.33e-9, upper95=1.44e-9, chain1_median=1.69e-9)
RAIL_FRAC = 0.02       # "near a prior bound" = within 2 percent of the prior range of that bound
BOX = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9), "herei": (3.5, 4.1), "heref": (2.6, 3.2), "alphaq": (1.3, 2.5),
       "hub": (0.65, 0.75), "omegamh2": (0.14, 0.146), "hireionz": (6.5, 8.0), "bhfeedback": (0.03, 0.07),
       "tau0_amp": (0.75, 1.25), "dtau0": (-0.4, 0.25)}


def refuse(msg):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(3)


def kim_tau0(z):
    return KIM_AMP * (1.0 + np.asarray(z, float)) ** KIM_SLOPE


def recover_tau0_amp_dtau0(tau0_ladder, z, z_pivot=TAU0_PIVOT_Z):
    """(L, nz) per-z tau0 -> (tau0_amp, dtau0) per draw by the EXACT log-linear inversion."""
    x = np.log((1.0 + np.asarray(z, float)) / (1.0 + z_pivot))
    y = np.log(np.asarray(tau0_ladder, float) / kim_tau0(z)[None, :])
    xc = x - x.mean()
    slope = (y - y.mean(axis=1, keepdims=True)) @ xc / np.sum(xc ** 2)
    intercept = y.mean(axis=1) - slope * x.mean()
    return np.exp(intercept), slope


def summarize(x):
    q = np.quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(median=float(q[2]), q16=float(q[1]), q84=float(q[3]), q025=float(q[0]), q975=float(q[4]),
                mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), sigma=float((q[3] - q[1]) / 2.0), n=int(x.size))


def ref_sigma(entry):
    if "err" in entry:
        return float(entry["err"])
    return float(max(entry.get("err_plus", 0.0), entry.get("err_minus", 0.0)))


def classify_scaled(name, ours, ref_median, ref_sig):
    r = RULES[name]
    delta = ours["median"] - ref_median
    s = float(np.sqrt(ours["sigma"] ** 2 + ref_sig ** 2))
    ad = abs(delta)
    if ad > r["discrepant_abs"] or ad > r["k_discrepant"] * s:
        label = "DISCREPANT"
    elif ad <= max(r["k_consistent"] * s, r["consistent_abs"]):
        label = "CONSISTENT"
    else:
        label = "SHIFTED"
    return dict(parameter=name, ours_median=ours["median"], ours_mean=ours["mean"], ours_sigma=ours["sigma"],
                ref_central=ref_median, ref_central_kind="GetDist posterior mean (Table 3)", ref_sigma=ref_sig,
                delta=float(delta), delta_mean_based=float(ours["mean"] - ref_median), s=s,
                delta_over_s=float(delta / s), delta_over_s_note="combined-width units; not a tension statistic",
                label=label, rule=r)


def classify_Ap(ours, x):
    q16, med, q025 = ours["q16"], ours["median"], ours["q025"]
    if q16 <= AP_RULE["upper68"] or med <= AP_RULE["upper95"]:
        label = "CONSISTENT"
    elif q025 > AP_RULE["chain1_median"]:
        label = "DISCREPANT"
    else:
        label = "SHIFTED"
    return dict(parameter="Ap", ours_median=med, ours_q16=q16, ours_q025=q025, ref_upper68=AP_RULE["upper68"],
                ref_upper95=AP_RULE["upper95"], chain1_median=AP_RULE["chain1_median"], label=label, rule=AP_RULE,
                p_ours_below_upper68=float(np.mean(x <= AP_RULE["upper68"])), p_ours_below_upper95=float(np.mean(x <= AP_RULE["upper95"])),
                consistent_via_q16_only=bool(label == "CONSISTENT" and med > AP_RULE["upper95"]),
                note="one-sided: the reference full-range chain reports A_P as an upper limit at its prior floor")


def rail_fraction(x, bounds):
    lo, hi = bounds
    w = RAIL_FRAC * (hi - lo)
    return dict(near_lo=float(np.mean(x <= lo + w)), near_hi=float(np.mean(x >= hi - w)))


def load_chains(chain_dir, root):
    files = sorted(glob.glob(os.path.join(chain_dir, f"{root}.unblinded.*.txt")))
    if not files:
        refuse(f"no {root}.unblinded.*.txt in {chain_dir}")
    names = None
    tabs = []
    for fp in files:
        with open(fp) as f:
            cols = f.readline()[1:].split()
        if cols[:2] != ["weight", "minusloglike"]:
            refuse(f"bad header in {fp}")
        if names is None:
            names = cols[2:]
        elif cols[2:] != names:
            refuse("chain files disagree on columns")
        tabs.append(np.loadtxt(fp, ndmin=2))
    return names, tabs


def load_nuisance(chain_dir, root):
    p = os.path.join(chain_dir, f"{root}.nuisance.npz")
    if not os.path.exists(p):
        return {}
    d = np.load(p)
    return {k: np.asarray(d[k]) for k in d.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--root", default="real_eboss")
    ap.add_argument("--reference", required=True, help="frozen reference JSON (Fernandez+2024 Table 3)")
    ap.add_argument("--analysis-lock", required=True)
    ap.add_argument("--leg", default="eBOSS")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.json and <out>.md)")
    a = ap.parse_args(argv)

    try:
        sys.path.insert(0, "/home/mfho/hcd_priya")
        from hcd_analysis.emulator.data import KIM_AMP as KA, KIM_SLOPE as KS  # noqa
        if abs(KA - KIM_AMP) > 1e-15 or abs(KS - KIM_SLOPE) > 1e-12:
            refuse("Kim constants drifted from the frozen values used here")
    except ImportError:
        pass

    ref = json.load(open(a.reference))
    prim = ref["chains"][ref["primary_chain"]]
    apr = prim.get("A_P_1e-9", {})
    if apr and (abs(apr.get("upper68", 1.33) * 1e-9 - AP_RULE["upper68"]) > 1e-15 or abs(apr.get("upper95", 1.44) * 1e-9 - AP_RULE["upper95"]) > 1e-15):
        refuse("A_P thresholds in the script disagree with the reference JSON")
    fid = ref["chains"].get("chain1_fiducial_z2.6_4.6", {})
    lock = json.load(open(a.analysis_lock))
    z = np.asarray(lock["legs"][a.leg]["z"], float)

    names, tabs = load_chains(a.chain_dir, a.root)
    per_chain = [t[:, 2:] for t in tabs]
    allrows = np.concatenate(per_chain, axis=0)
    tau_idx = [i for i, n in enumerate(names) if n.startswith("tau0_z")]
    if len(tau_idx) != z.size:
        refuse(f"{len(tau_idx)} tau0 columns vs {z.size} leg z values")
    zp = float(lock["legs"][a.leg].get("prior", {}).get("tau0_pivot_z", TAU0_PIVOT_Z))
    tau_names = [names[i] for i in tau_idx]
    if tau_names != [f"tau0_z{i}" for i in range(len(tau_idx))] or np.any(np.diff(z) <= 0):
        refuse("tau0 columns are not tau0_z0..z{n-1} in ascending z order")
    amp, dt = recover_tau0_amp_dtau0(allrows[:, tau_idx], z, z_pivot=zp)
    recon = amp[:, None] * ((1.0 + z)[None, :] / (1.0 + zp)) ** dt[:, None] * kim_tau0(z)[None, :]
    resid = float(np.max(np.abs(recon / allrows[:, tau_idx] - 1.0)))
    if resid > 1e-6:
        refuse(f"tau0 ladder is not an exact tau0_amp/dtau0 curve (max rel residual {resid:.2e})")
    if amp.min() < BOX["tau0_amp"][0] - 1e-9 or amp.max() > BOX["tau0_amp"][1] + 1e-9 or dt.min() < BOX["dtau0"][0] - 1e-9 or dt.max() > BOX["dtau0"][1] + 1e-9:
        refuse("recovered tau0_amp / dtau0 leave the prior box")
    cols = {n: allrows[:, i] for i, n in enumerate(names)}
    cols["tau0_amp"], cols["dtau0"] = amp, dt
    nuis = load_nuisance(a.chain_dir, a.root)
    for k, v in nuis.items():
        cols[f"nuis:{k}"] = v.reshape(-1)

    summaries = {k: summarize(v) for k, v in cols.items()}
    health = json.load(open(os.path.join(a.chain_dir, f"{a.root}.health.json")))
    gate = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ubo", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eboss_unblind_once.py"))
        ubo = importlib.util.module_from_spec(spec); spec.loader.exec_module(ubo)
        nuis_diag, _ = ubo.nuisance_diagnostics(a.chain_dir, a.root)
        gate = ubo.classify_health(health, nuis_diag)
    except Exception as e:  # noqa: BLE001
        gate = dict(label="UNAVAILABLE", error=f"{type(e).__name__}: {e}")

    tests = {}
    tests["ns"] = classify_scaled("ns", summaries["ns"], prim["n_P"]["median"], ref_sigma(prim["n_P"]))
    tests["Ap"] = classify_Ap(summaries["Ap"], cols["Ap"])
    tests["tau0_amp"] = classify_scaled("tau0_amp", summaries["tau0_amp"], prim["tau0"]["median"], ref_sigma(prim["tau0"]))
    tests["dtau0"] = classify_scaled("dtau0", summaries["dtau0"], prim["dtau0"]["median"], ref_sigma(prim["dtau0"]))
    headline = tests["ns"]["label"]
    descriptive = {}
    if fid:
        descriptive["ns_vs_fiducial_1.009"] = dict(delta=float(summaries["ns"]["median"] - fid["n_P"]["median"]))
        descriptive["Ap_vs_fiducial_1.69e-9"] = {"delta_1e-9": float(summaries["Ap"]["median"] * 1e9 - fid["A_P_1e-9"]["median"])}
        descriptive["tau0_vs_fiducial"] = dict(delta=float(summaries["tau0_amp"]["median"] - fid["tau0"]["median"]))
        descriptive["dtau0_vs_fiducial"] = dict(delta=float(summaries["dtau0"]["median"] - fid["dtau0"]["median"]))
    rails = {k: rail_fraction(cols[k], b) for k, b in BOX.items() if k in cols}
    for k in ("alpha_lls", "alpha_subdla", "alpha_dla"):
        if k in cols:
            x = cols[k]; rails[k] = dict(near_lo=float(np.mean(x <= RAIL_FRAC * float(np.quantile(x, 0.5)))), near_hi=None, note="lower bound 0; near_lo = within 2 percent of the median above zero")
    nuis_json = os.path.join(a.chain_dir, f"{a.root}.nuisance.json")
    nuisance_rails = {k: dict(near_lo=v.get("frac_near_lo"), near_hi=v.get("frac_near_hi")) for k, v in json.load(open(nuis_json)).get("sites", {}).items()} if os.path.exists(nuis_json) else {}
    mc_err = {k: float(summaries[k]["sigma"] * np.sqrt(np.pi / 2.0) / np.sqrt(max(float(health.get("ess_bulk_min", 1)), 1.0))) for k in ("ns", "Ap", "tau0_amp", "dtau0")}
    boundary_flag = {k: bool(rails[k]["near_lo"] > 0.05 or rails[k]["near_hi"] > 0.05) for k in ("ns", "Ap", "tau0_amp", "dtau0")}

    attribution = None
    if any(tests[k]["label"] != "CONSISTENT" for k in ("ns", "Ap", "tau0_amp", "dtau0")):
        keys = [k for k in ("ns", "Ap", "tau0_amp", "dtau0", "alphaq", "hub", "herei", "heref", "omegamh2", "hireionz",
                            "bhfeedback", "alpha_lls", "alpha_subdla", "alpha_dla") if k in cols]
        keys += [k for k in cols if k.startswith("nuis:")]
        # nuisance draws are aligned with the chain rows only if their count matches; otherwise drop them
        keys = [k for k in keys if cols[k].size == allrows.shape[0]]
        M = np.column_stack([cols[k] for k in keys])
        C = np.corrcoef(M, rowvar=False)
        cov = np.cov(M, rowvar=False)
        ins = keys.index("ns")
        refpoints = {"alphaq": prim["alpha_q"]["lower68"], "tau0_amp": prim["tau0"]["median"], "dtau0": prim["dtau0"]["median"],
                     "hub": prim["v_scale_h"]["median"]}
        linresp = {}
        for k, rv in refpoints.items():
            if k in keys:
                j = keys.index(k)
                beta = cov[ins, j] / cov[j, j]
                linresp[k] = dict(beta_ns_per_unit=float(beta), ours_median=float(np.median(cols[k])), ref_point=float(rv),
                                  ns_shift_associated=float(beta * (np.median(cols[k]) - rv)),
                                  note=("INDICATIVE one-at-a-time linear response on our posterior; not a refit; NOT additive across x (correlated regressors). "
                                        "alphaq: the reference point 2.85 is the reference's one-sided 68 percent lower limit at its prior cap 3.0; our prior ends at 2.5, "
                                        "so this entry extrapolates a slope fitted on [1.3, 2.5]: sign informative, magnitude a lower bound. hub: the reference h is sample-variance driven."))
        attribution = dict(note="single preregistered attribution step (section 5): posterior correlations, rail fractions, indicative linear response; entries are not additive",
                           keys=keys, corr_with_ns={k: float(C[ins, j]) for j, k in enumerate(keys)},
                           corr_matrix=C.tolist(), linear_response=linresp, rails=rails)

    out = dict(schema="eboss_priya_consistency.v1", reference=ref["reference"], primary_chain=ref["primary_chain"],
               chain_dir=os.path.abspath(a.chain_dir), root=a.root, n_chains=len(tabs), n_draws_total=int(allrows.shape[0]),
               health={k: health.get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "ebfmi_min", "n_divergent",
                                                   "treedepth_sat_frac", "n_chains", "n_draws", "created_utc", "seed")},
               green=dict(rhat=bool(health.get("rhat_max", 9) < 1.01), ess_bulk=bool(health.get("ess_bulk_min", 0) >= 400),
                          ess_tail=bool(health.get("ess_tail_min", 0) >= 400), divergences=bool(health.get("n_divergent", 1) == 0),
                          ebfmi=bool(health.get("ebfmi_min", 0) >= 0.3), treedepth=bool(health.get("treedepth_sat_frac", 1) < 0.02),
                          nuisance_rails=bool(all((v.get("near_lo") or 0) < 0.05 and (v.get("near_hi") or 0) < 0.05 for v in nuisance_rails.values()))),
               health_gate=gate, mc_error_of_median=mc_err, boundary_proximity_flag=boundary_flag, nuisance_rails=nuisance_rails,
               z_grid=z.tolist(), summaries=summaries, tests=tests, headline_label=headline, descriptive_vs_fiducial=descriptive,
               rails=rails, attribution=attribution)
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True); f.write("\n")
    # ---- markdown ----
    L = [f"# eBOSS real-data readout vs PRIYA (Fernandez+2024 Table 3, {ref['primary_chain']})", "",
         f"chains {len(tabs)} x {int(allrows.shape[0] / len(tabs))} draws; health: R-hat {health.get('rhat_max')}, ESS bulk {health.get('ess_bulk_min')}, ESS tail {health.get('ess_tail_min')}, E-BFMI {health.get('ebfmi_min')}, divergences {health.get('n_divergent')}, tree-depth sat {health.get('treedepth_sat_frac')}; GREEN flags {out['green']}; health gate {gate.get('label') if gate else None}", "",
         "## Preregistered tests (reference central values are GetDist posterior means; delta/s in combined-width units, not a tension statistic)", "",
         "| parameter | ours median [16,84] (MC err) | reference | delta (median) | delta (mean) | s | delta/s | label | boundary flag |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("ns", "tau0_amp", "dtau0"):
        t = tests[k]; s = summaries[k]
        L.append(f"| {k} | {s['median']:.4f} [{s['q16']:.4f}, {s['q84']:.4f}] ({mc_err[k]:.4f}) | {t['ref_central']:.4f} +/- {t['ref_sigma']:.4f} | {t['delta']:+.4f} | {t['delta_mean_based']:+.4f} | {t['s']:.4f} | {t['delta_over_s']:+.2f} | **{t['label']}** | {boundary_flag[k]} |")
    t = tests["Ap"]; s = summaries["Ap"]
    L.append(f"| Ap (1e-9) | {s['median']*1e9:.3f} [{s['q16']*1e9:.3f}, {s['q84']*1e9:.3f}] | < 1.33 (68), < 1.44 (95) | P(<1.33e-9) {t['p_ours_below_upper68']:.3f}; P(<1.44e-9) {t['p_ours_below_upper95']:.3f} | | | one-sided | **{t['label']}**{' (via q16 only)' if t['consistent_via_q16_only'] else ''} | {boundary_flag['Ap']} |")
    L += ["", f"**Headline (n_P): {headline}.**", "", "## Posterior summaries (all columns)", "", "| column | median | 16 | 84 | 2.5 | 97.5 | sd |", "|---|---|---|---|---|---|---|"]
    for k, s in summaries.items():
        L.append(f"| {k} | {s['median']:.6g} | {s['q16']:.6g} | {s['q84']:.6g} | {s['q025']:.6g} | {s['q975']:.6g} | {s['sd']:.3g} |")
    L += ["", "## Rail fractions (within 2 percent of a prior bound)", "", "| parameter | near lo | near hi |", "|---|---|---|"]
    for k, r in list(rails.items()) + list(nuisance_rails.items()):
        L.append(f"| {k} | {r.get('near_lo')} | {r.get('near_hi')} |")
    if descriptive:
        L += ["", "## Descriptive only: versus the authors' fiducial (z >= 2.6) chain", ""] + [f"- {k}: {v}" for k, v in descriptive.items()]
    if attribution:
        L += ["", "## Single preregistered attribution step (INDICATIVE)", "", "| x | corr(ns, x) |", "|---|---|"]
        for k, c in attribution["corr_with_ns"].items():
            L.append(f"| {k} | {c:+.3f} |")
        L += ["", "| x | beta = d ns / d x | ours median | reference point | ns shift associated |", "|---|---|---|---|---|"]
        for k, v in attribution["linear_response"].items():
            L.append(f"| {k} | {v['beta_ns_per_unit']:+.4g} | {v['ours_median']:.4g} | {v['ref_point']:.4g} | {v['ns_shift_associated']:+.4f} |")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}.json and {a.out}.md; headline {headline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
