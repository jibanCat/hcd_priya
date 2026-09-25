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
import hashlib
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
    # PI #28 (z >= 2.6 product vs the canonical chain, whose A_P is two-sided 1.69 +0.14 -0.15 in 1e-9): c = 0.15e-9
    # (about 1 sigma_ref), d = 0.50e-9 (about 3.4 sigma_ref), mirroring the n_P ratios; used ONLY when the primary
    # reference chain carries a two-sided A_P (a 'median' entry); the one-sided rule below stays for upper limits.
    "Ap":       dict(consistent_abs=0.15e-9, discrepant_abs=0.50e-9, k_consistent=1.0, k_discrepant=3.0),
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


def hdi(x, mass=0.6827):
    """Shortest interval containing `mass` of the draws (the sample analogue of GetDist's two-tail density limits)."""
    xs = np.sort(np.asarray(x, float)); n = xs.size; k = max(int(np.ceil(mass * n)), 2)
    if k >= n:
        return float(xs[0]), float(xs[-1])
    widths = xs[k - 1:] - xs[:n - k + 1]; i = int(np.argmin(widths))
    return float(xs[i]), float(xs[i + k - 1])


def summarize(x):
    """Posterior summary. PRIMARY (PI #29, like-for-like with GetDist tables): `mean`, the 68 percent highest-density
    limits about the mean (`hdi68_lo/hi`, `err_minus_68/err_plus_68`) and `sigma_mean` = max(err_plus, err_minus)
    (the same symmetrisation `ref_sigma` applies to the reference). DESCRIPTIVE: median, 16/84, 2.5/97.5, sd, and
    `sigma` = half the 16 to 84 width (the pre PI #29 half-width, kept for the record)."""
    q = np.quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975]); lo, hi = hdi(x); m = float(np.mean(x))
    return dict(median=float(q[2]), q16=float(q[1]), q84=float(q[3]), q025=float(q[0]), q975=float(q[4]),
                mean=m, sd=float(np.std(x, ddof=1)), sigma=float((q[3] - q[1]) / 2.0), n=int(x.size),
                hdi68_lo=lo, hdi68_hi=hi, err_minus_68=float(m - lo), err_plus_68=float(hi - m), sigma_mean=float(max(m - lo, hi - m)))


def ref_sigma(entry):
    if "err" in entry:
        return float(entry["err"])
    return float(max(entry.get("err_plus", 0.0), entry.get("err_minus", 0.0)))


def ref_central(entry):
    """The reference central value: the GetDist posterior MEAN (`mean` if present; older files store it under `median`)."""
    return float(entry["mean"] if "mean" in entry else entry["median"])


def classify_scaled(name, ours, ref_median, ref_sig):
    """PRIMARY shift (PI #29): our posterior MEAN minus the reference GetDist mean, with s = sqrt(sigma_mean^2 + sigma_ref^2)
    where sigma_mean is our symmetrised 68 percent highest-density half-width (like-for-like with the reference's quoted
    interval). The median-based shift is reported alongside as descriptive."""
    r = RULES[name]
    delta = ours["mean"] - ref_median
    s = float(np.sqrt(ours["sigma_mean"] ** 2 + ref_sig ** 2))
    ad = abs(delta)
    if ad > r["discrepant_abs"] or ad > r["k_discrepant"] * s:
        label = "DISCREPANT"
    elif ad <= max(r["k_consistent"] * s, r["consistent_abs"]):
        label = "CONSISTENT"
    else:
        label = "SHIFTED"
    return dict(parameter=name, ours_mean=ours["mean"], ours_err_minus_68=ours["err_minus_68"], ours_err_plus_68=ours["err_plus_68"],
                ours_sigma_mean=ours["sigma_mean"], ours_median=ours["median"], ours_sigma_1684=ours["sigma"],
                ref_central=ref_median, ref_central_kind="GetDist posterior mean", ref_sigma=ref_sig,
                delta=float(delta), delta_basis="mean minus mean (PI #29)", delta_median_based=float(ours["median"] - ref_median), s=s,
                delta_over_s=float(delta / s), delta_over_s_note="combined-width units; not a tension statistic",
                label=label, rule=r)


def classify_Ap(ours, x):
    """One-sided rule (reference reports A_P as upper limits). PI #29: the central value is our MEAN and the 68 percent bound
    is the highest-density lower limit (like-for-like with GetDist limits); the median-based reading is kept descriptively."""
    q16, med, q025 = ours["hdi68_lo"], ours["mean"], ours["q025"]
    if q16 <= AP_RULE["upper68"] or med <= AP_RULE["upper95"]:
        label = "CONSISTENT"
    elif q025 > AP_RULE["chain1_median"]:
        label = "DISCREPANT"
    else:
        label = "SHIFTED"
    return dict(parameter="Ap", ours_mean=med, ours_hdi68_lo=q16, ours_median=ours["median"], ours_q16=ours["q16"], ours_q025=q025, ref_upper68=AP_RULE["upper68"],
                ref_upper95=AP_RULE["upper95"], chain1_median=AP_RULE["chain1_median"], label=label, rule=AP_RULE,
                p_ours_below_upper68=float(np.mean(x <= AP_RULE["upper68"])), p_ours_below_upper95=float(np.mean(x <= AP_RULE["upper95"])),
                consistent_via_q16_only=bool(label == "CONSISTENT" and med > AP_RULE["upper95"]),
                note="one-sided: the reference full-range chain reports A_P as an upper limit at its prior floor")


def rail_fraction(x, bounds):
    lo, hi = bounds
    w = RAIL_FRAC * (hi - lo)
    return dict(near_lo=float(np.mean(x <= lo + w)), near_hi=float(np.mean(x >= hi - w)))


def load_chains(chain_dir, root, mode="unblinded"):
    """mode 'unblinded': the governed unblind's <root>.unblinded.<c>.txt; mode 'plain' (PI #28, a product exported
    unblinded from the start): <root>.<c>.txt, refusing if any <root>.unblinded.* file exists (no mixing)."""
    if mode == "plain":
        if glob.glob(os.path.join(chain_dir, f"{root}.unblinded.*")):
            refuse(f"plain mode but {root}.unblinded.* files exist in {chain_dir}")
        files = sorted(p for p in glob.glob(os.path.join(chain_dir, f"{root}.*.txt"))
                       if os.path.basename(p)[len(root) + 1:-4].isdigit())
        if not files:
            refuse(f"no {root}.<c>.txt in {chain_dir}")
    else:
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
    ap.add_argument("--chain-files", choices=("unblinded", "plain"), default="unblinded",
                    help="'unblinded' = the governed unblind's <root>.unblinded.<c>.txt (default); 'plain' = a product "
                         "exported unblinded from the start (<root>.<c>.txt; health.blinded must be false) (PI #28)")
    ap.add_argument("--execution-record", default=None,
                    help="the wrapper's EXECUTION_RECORD.json of the product (Reviewer L S1): refuse unless sha256(this script) equals "
                         "records_sha256.readout_script and sha256(--reference) equals a records_sha256 reference entry")
    ap.add_argument("--require-gate", action="store_true",
                    help="refuse (exit 3, nothing written) unless the health gate is GREEN or AMBER (PI #28: enforced for "
                         "products that pass through no unblind script)")
    a = ap.parse_args(argv)
    if a.chain_files == "plain":
        a.require_gate = True          # Reviewer K S2: a product that passes through no unblind script is gated HERE, unconditionally
    integrity = None
    if a.execution_record:
        rec = json.load(open(a.execution_record)); rs = rec.get("records_sha256", {})
        me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
        if rs.get("readout_script") != me:
            refuse("this readout script's sha256 differs from records_sha256.readout_script in the execution record (post-run edit?)")
        rsha = hashlib.sha256(open(a.reference, "rb").read()).hexdigest()
        matched = [k for k, v in rs.items() if v == rsha and k.startswith("reference_json")]
        if not matched:
            refuse("the reference JSON's sha256 matches no reference_json* entry of the execution record")
        integrity = dict(execution_record=os.path.abspath(a.execution_record), readout_script_sha256=me, reference_sha256=rsha, reference_record_key=matched[0])
    if os.path.exists(a.out + ".json") or os.path.exists(a.out + ".md"):
        refuse(f"output exists: {a.out}.json/.md (the readout runs once)")

    try:
        sys.path.insert(0, "/home/mfho/hcd_priya")
        from hcd_analysis.emulator.data import KIM_AMP as KA, KIM_SLOPE as KS  # noqa
        if abs(KA - KIM_AMP) > 1e-15 or abs(KS - KIM_SLOPE) > 1e-12:
            refuse("Kim constants drifted from the frozen values used here")
    except ImportError:
        pass

    ref = json.load(open(a.reference))
    prim = ref["chains"][ref["primary_chain"]]
    # Reviewer K M1: every field the readout and the attribution step will touch must exist BEFORE any computation
    for key in ("n_P", "A_P_1e-9", "tau0", "dtau0", "alpha_q", "v_scale_h"):
        if key not in prim:
            refuse(f"primary reference chain {ref['primary_chain']!r} lacks {key!r} (schema check)")
    for key in ("n_P", "tau0", "dtau0", "v_scale_h"):
        if not (("mean" in prim[key]) or ("median" in prim[key])):
            refuse(f"primary reference {key!r} has no central value")
    if "lower68" not in prim["alpha_q"]:
        refuse("primary reference alpha_q lacks lower68")
    apx = prim["A_P_1e-9"]
    if not ((("mean" in apx) or ("median" in apx)) or (("upper68" in apx) and ("upper95" in apx))):
        refuse("primary reference A_P_1e-9 has neither a central value nor upper limits")
    apr = prim.get("A_P_1e-9", {})
    ap_two_sided = bool(apr) and (("mean" in apr) or ("median" in apr)) and not bool(apr.get("onetail_upper", False))
    if apr and not ap_two_sided and ("upper68" in apr) and ("upper95" in apr):
        # the one-sided thresholds come from the reference file itself (released-chain limits or the Table 3 transcription)
        AP_RULE.update(upper68=float(apr["upper68"]) * 1e-9, upper95=float(apr["upper95"]) * 1e-9)
        dk = ref.get("descriptive_chain"); dap = (ref["chains"].get(dk, {}) if dk else {}).get("A_P_1e-9", {})
        if ("mean" in dap) or ("median" in dap):
            AP_RULE.update(chain1_median=ref_central(dap) * 1e-9)
    if apr and not ap_two_sided and not (("upper68" in apr) and ("upper95" in apr)):
        refuse("one-sided A_P reference lacks upper68 / upper95")
    desc_key = ref.get("descriptive_chain", "chain1_fiducial_z2.6_4.6")
    fid = ref["chains"].get(desc_key, {}) if desc_key != ref["primary_chain"] else {}
    lock = json.load(open(a.analysis_lock))
    z_lock = np.asarray(lock["legs"][a.leg]["z"], float)
    health = json.load(open(os.path.join(a.chain_dir, f"{a.root}.health.json")))
    if a.chain_files == "plain":
        if health.get("blinded") is not False:
            refuse("plain chain files but health.json does not say blinded: false")
    elif health.get("blinded") is False:
        refuse("unblinded mode but health.json says blinded: false (use --chain-files plain for a product exported unblinded)")
    if health.get("z_kept"):
        z = np.asarray(health["z_kept"], float); z_source = "health.z_kept"
        if not all(np.any(np.isclose(zz, z_lock, atol=1e-6)) for zz in z):
            refuse("health.z_kept is not a subset of the lock's leg z grid")
    else:
        z = z_lock; z_source = "analysis.lock legs z"

    names, tabs = load_chains(a.chain_dir, a.root, mode=a.chain_files)
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
    gate = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ubo", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eboss_unblind_once.py"))
        ubo = importlib.util.module_from_spec(spec); spec.loader.exec_module(ubo)
        nuis_diag, _ = ubo.nuisance_diagnostics(a.chain_dir, a.root)
        gate = ubo.classify_health(health, nuis_diag)
    except Exception as e:  # noqa: BLE001
        gate = dict(label="UNAVAILABLE", error=f"{type(e).__name__}: {e}")
    if a.require_gate and (gate or {}).get("label") not in ("GREEN", "AMBER"):
        refuse(f"health gate {(gate or {}).get('label')} (--require-gate): no readout")

    tests = {}
    tests["ns"] = classify_scaled("ns", summaries["ns"], ref_central(prim["n_P"]), ref_sigma(prim["n_P"]))
    if ap_two_sided:
        tests["Ap"] = classify_scaled("Ap", summaries["Ap"], ref_central(apr) * 1e-9, ref_sigma(apr) * 1e-9)
        tests["Ap"]["rule_note"] = "two-sided (the primary reference chain reports A_P as median with 68 percent errors); thresholds RULES['Ap']"
    else:
        tests["Ap"] = classify_Ap(summaries["Ap"], cols["Ap"])
    tests["tau0_amp"] = classify_scaled("tau0_amp", summaries["tau0_amp"], ref_central(prim["tau0"]), ref_sigma(prim["tau0"]))
    tests["dtau0"] = classify_scaled("dtau0", summaries["dtau0"], ref_central(prim["dtau0"]), ref_sigma(prim["dtau0"]))
    headline = tests["ns"]["label"]
    descriptive = {}
    if fid:
        tag = "fiducial" if desc_key == "chain1_fiducial_z2.6_4.6" else desc_key
        descriptive[f"ns_vs_{tag}" + ("_1.009" if tag == "fiducial" else "")] = dict(delta=float(summaries["ns"]["mean"] - ref_central(fid["n_P"])), basis="mean minus mean")
        fap = fid.get("A_P_1e-9", {})
        if ("mean" in fap) or ("median" in fap):
            descriptive[f"Ap_vs_{tag}" + ("_1.69e-9" if tag == "fiducial" else "")] = {"delta_1e-9": float(summaries["Ap"]["mean"] * 1e9 - ref_central(fap)), "basis": "mean minus mean"}
        if "upper68" in fap:
            descriptive[f"Ap_vs_{tag}_upper68"] = {"delta_1e-9": float(summaries["Ap"]["mean"] * 1e9 - fap["upper68"]), "note": "mean minus the reference 68 percent upper limit"}
        descriptive[f"tau0_vs_{tag}"] = dict(delta=float(summaries["tau0_amp"]["mean"] - ref_central(fid["tau0"])), basis="mean minus mean")
        descriptive[f"dtau0_vs_{tag}"] = dict(delta=float(summaries["dtau0"]["mean"] - ref_central(fid["dtau0"])), basis="mean minus mean")
    rails = {k: rail_fraction(cols[k], b) for k, b in BOX.items() if k in cols}
    for k in ("alpha_lls", "alpha_subdla", "alpha_dla"):
        if k in cols:
            x = cols[k]; rails[k] = dict(near_lo=float(np.mean(x <= RAIL_FRAC * float(np.quantile(x, 0.5)))), near_hi=None, note="lower bound 0; near_lo = within 2 percent of the median above zero")
    nuis_json = os.path.join(a.chain_dir, f"{a.root}.nuisance.json")
    nuisance_rails = {k: dict(near_lo=v.get("frac_near_lo"), near_hi=v.get("frac_near_hi")) for k, v in json.load(open(nuis_json)).get("sites", {}).items()} if os.path.exists(nuis_json) else {}
    mc_err = {k: float(summaries[k]["sd"] / np.sqrt(max(float(health.get("ess_bulk_min", 1)), 1.0))) for k in ("ns", "Ap", "tau0_amp", "dtau0")}   # MC error of the MEAN (PI #29)
    mc_err_median = {k: float(summaries[k]["sigma"] * np.sqrt(np.pi / 2.0) / np.sqrt(max(float(health.get("ess_bulk_min", 1)), 1.0))) for k in ("ns", "Ap", "tau0_amp", "dtau0")}   # descriptive (pre PI #29)
    boundary_flag = {k: bool(rails[k]["near_lo"] > 0.05 or rails[k]["near_hi"] > 0.05) for k in ("ns", "Ap", "tau0_amp", "dtau0")}

    for k in ("ns", "Ap", "tau0_amp", "dtau0"):
        tests[k]["box_limited"] = bool(boundary_flag[k])
        tests[k]["label_qualified"] = tests[k]["label"] + (" (BOX-LIMITED)" if boundary_flag[k] else "")
        if boundary_flag[k]:
            tests[k]["box_limited_note"] = ("rail fraction > 0.05 at a prior bound: the rule label stands, delta is a lower bound in magnitude and delta/s is not "
                                            "interpretable; the quantities of record are the rail fractions (ours and the reference chain's on the shared bound)")
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
        refpoints = {"alphaq": prim["alpha_q"].get("lower68"), "tau0_amp": ref_central(prim["tau0"]), "dtau0": ref_central(prim["dtau0"]),
                     "hub": (ref_central(prim["v_scale_h"]) if "v_scale_h" in prim else None)}
        skipped_refpoints = [k for k, v in refpoints.items() if v is None]
        refpoints = {k: v for k, v in refpoints.items() if v is not None}
        linresp = {}
        for k, rv in refpoints.items():
            if k in keys:
                j = keys.index(k)
                beta = cov[ins, j] / cov[j, j]
                linresp[k] = dict(beta_ns_per_unit=float(beta), ours_median=float(np.median(cols[k])), ref_point=float(rv),
                                  ns_shift_associated=float(beta * (np.median(cols[k]) - rv)),
                                  note=("INDICATIVE one-at-a-time linear response on our posterior; not a refit; NOT additive across x (correlated regressors). "
                                        f"alphaq: the reference point {refpoints.get('alphaq')} is the reference's one-sided 68 percent lower limit at its prior cap 3.0; our prior ends at 2.5, "
                                        "so this entry extrapolates a slope fitted on [1.3, 2.5]: sign informative, magnitude a lower bound. hub: the reference h is sample-variance driven."))
        attribution = dict(note="single preregistered attribution step (section 5): posterior correlations, rail fractions, indicative linear response; entries are not additive",
                           skipped_reference_points=skipped_refpoints,
                           keys=keys, corr_with_ns={k: float(C[ins, j]) for j, k in enumerate(keys)},
                           corr_matrix=C.tolist(), linear_response=linresp, rails=rails)

    out = dict(schema="eboss_priya_consistency.v2_means", reference=ref["reference"], primary_chain=ref["primary_chain"],
               chain_dir=os.path.abspath(a.chain_dir), root=a.root, n_chains=len(tabs), n_draws_total=int(allrows.shape[0]),
               health={k: health.get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "ebfmi_min", "n_divergent",
                                                   "treedepth_sat_frac", "n_chains", "n_draws", "created_utc", "seed")},
               green=dict(rhat=bool(health.get("rhat_max", 9) < 1.01), ess_bulk=bool(health.get("ess_bulk_min", 0) >= 400),
                          ess_tail=bool(health.get("ess_tail_min", 0) >= 400), divergences=bool(health.get("n_divergent", 1) == 0),
                          ebfmi=bool(health.get("ebfmi_min", 0) >= 0.3), treedepth=bool(health.get("treedepth_sat_frac", 1) < 0.02),
                          nuisance_rails=bool(all((v.get("near_lo") or 0) < 0.05 and (v.get("near_hi") or 0) < 0.05 for v in nuisance_rails.values()))),
               health_gate=gate, require_gate=bool(a.require_gate), integrity=integrity, mc_error_of_mean=mc_err, mc_error_of_median=mc_err_median, boundary_proximity_flag=boundary_flag, nuisance_rails=nuisance_rails,
               z_grid=z.tolist(), z_source=z_source, chain_files_mode=a.chain_files, descriptive_chain=(desc_key if fid else None),
               summaries=summaries, tests=tests, headline_label=headline, descriptive_vs_fiducial=descriptive,
               rails=rails, attribution=attribution)
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True); f.write("\n")
    # ---- markdown ----
    L = [f"# eBOSS real-data readout vs PRIYA (Fernandez+2024 Table 3, {ref['primary_chain']})", "",
         f"chains {len(tabs)} x {int(allrows.shape[0] / len(tabs))} draws; health: R-hat {health.get('rhat_max')}, ESS bulk {health.get('ess_bulk_min')}, ESS tail {health.get('ess_tail_min')}, E-BFMI {health.get('ebfmi_min')}, divergences {health.get('n_divergent')}, tree-depth sat {health.get('treedepth_sat_frac')}; GREEN flags {out['green']}; health gate {gate.get('label') if gate else None}", "",
         "## Preregistered tests (PRIMARY: our posterior MEAN with 68 percent highest-density limits versus the reference GetDist mean with its quoted limits, PI #29; delta/s in combined-width units, not a tension statistic; median [16,84] descriptive)", "",
         "| parameter | ours mean (+err/-err) (MC err of mean) | ours median [16,84] | reference mean (+/-) | delta (mean) | delta (median, descriptive) | s | delta/s | label | boundary flag |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("ns", "tau0_amp", "dtau0"):
        t = tests[k]; s = summaries[k]
        L.append(f"| {k} | {s['mean']:.4f} (+{s['err_plus_68']:.4f}/-{s['err_minus_68']:.4f}) ({mc_err[k]:.4f}) | {s['median']:.4f} [{s['q16']:.4f}, {s['q84']:.4f}] | {t['ref_central']:.4f} +/- {t['ref_sigma']:.4f} | {t['delta']:+.4f} | {t['delta_median_based']:+.4f} | {t['s']:.4f} | {t['delta_over_s']:+.2f} | **{t['label_qualified']}** | {boundary_flag[k]} |")
    t = tests["Ap"]; s = summaries["Ap"]
    if ap_two_sided:
        L.append(f"| Ap (1e-9) | {s['mean']*1e9:.3f} (+{s['err_plus_68']*1e9:.3f}/-{s['err_minus_68']*1e9:.3f}) ({mc_err['Ap']*1e9:.3f}) | {s['median']*1e9:.3f} [{s['q16']*1e9:.3f}, {s['q84']*1e9:.3f}] | {t['ref_central']*1e9:.3f} +/- {t['ref_sigma']*1e9:.3f} | {t['delta']*1e9:+.3f} | {t['delta_median_based']*1e9:+.3f} | {t['s']*1e9:.3f} | {t['delta_over_s']:+.2f} | **{t['label_qualified']}** (two-sided) | {boundary_flag['Ap']} |")
    else:
        L.append(f"| Ap (1e-9) | {s['mean']*1e9:.3f} (+{s['err_plus_68']*1e9:.3f}/-{s['err_minus_68']*1e9:.3f}) | {s['median']*1e9:.3f} [{s['q16']*1e9:.3f}, {s['q84']*1e9:.3f}] | < 1.33 (68), < 1.44 (95) | P(<1.33e-9) {t['p_ours_below_upper68']:.3f}; P(<1.44e-9) {t['p_ours_below_upper95']:.3f} | | | one-sided | **{t['label_qualified']}**{' (via HDI lower bound only)' if t['consistent_via_q16_only'] else ''} | {boundary_flag['Ap']} |")
    L += ["", f"**Headline (n_P): {headline}.**", "", "## Posterior summaries (all columns)", "", "| column | mean | HDI68 lo | HDI68 hi | median | 16 | 84 | 2.5 | 97.5 | sd |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in summaries.items():
        L.append(f"| {k} | {s['mean']:.6g} | {s['hdi68_lo']:.6g} | {s['hdi68_hi']:.6g} | {s['median']:.6g} | {s['q16']:.6g} | {s['q84']:.6g} | {s['q025']:.6g} | {s['q975']:.6g} | {s['sd']:.3g} |")
    EDGE = {"ns": "emulator training box", "Ap": "emulator training box", "herei": "emulator training box (herei <= 4.1 sampling cap)", "heref": "emulator training box (heref >= 2.6 sampling cap)",
            "alphaq": "emulator training box (alpha_q <= 2.5 sampling cap; reference cap 3.0)", "hub": "emulator training box", "omegamh2": "emulator training box", "hireionz": "emulator training box",
            "bhfeedback": "emulator training box", "tau0_amp": "mean-flux prior box (shared with the reference)", "dtau0": "mean-flux prior box (shared with the reference)",
            "alpha_lls": "HCD amplitude, lower bound 0", "alpha_subdla": "HCD amplitude, lower bound 0", "alpha_dla": "HCD amplitude, lower bound 0"}
    ref_rails = prim.get("rails", {})
    L += ["", "## Rail fractions (within 2 percent of a prior bound; edge class; the reference chain's rail on the shared bound where available)", "",
          "| parameter | near lo | near hi | edge | reference near lo | reference near hi |", "|---|---|---|---|---|---|"]
    for k, r in list(rails.items()) + list(nuisance_rails.items()):
        rr = ref_rails.get({"ns": "ns", "Ap": "Ap", "tau0_amp": "tau0", "dtau0": "dtau0", "alphaq": "alphaq", "hub": "hub", "omegamh2": "omegamh2", "herei": "herei", "heref": "heref", "hireionz": "hireionz", "bhfeedback": "bhfeedback"}.get(k, k), {})
        L.append(f"| {k} | {r.get('near_lo')} | {r.get('near_hi')} | {EDGE.get(k, 'LogUniform metal-node bracket' if k.startswith(('f_Si', 'k_Si')) else '')} | {rr.get('near_lo', '')} | {rr.get('near_hi', '')} |")
    if descriptive:
        L += ["", f"## Descriptive only: versus the reference chain {desc_key}", ""] + [f"- {k}: {v}" for k, v in descriptive.items()]
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
