#!/usr/bin/env python3
"""eboss_diag_compare.py -- generic comparison of a DIAGNOSTIC real-data fit against its baseline product (PI #33, preregistration v1.5).

Reads two GetDist-format product directories (the diagnostic run and the baseline of the same redshift range), recovers the
mean-flux parameters from the tau0 ladder (2-parameter log-linear inversion; plus the 3-parameter log-quadratic fit and the
exported ``ctau0`` when present), and writes ``<out>.json`` and ``<out>.md`` with:
  * the health gate of the diagnostic run (GREEN / RED on the frozen sampler thresholds; rails reported separately);
  * mean shifts (diagnostic minus baseline) of every shared column and nuisance site in units of the BASELINE posterior sd;
  * rail fractions of every bounded parameter (each product against its OWN box, taken from its health/yaml when recorded);
  * correlations of ns with tau0_amp, A_P, hub, alpha_subdla, alpha_lls, f_res_amp in both products;
  * the tau_eff(z) posterior-mean curves of both products from their exported ladders on the shared z grid;
  * the fraction of draws whose ladder leaves the emulator tau0 band [0.75, 1.25] at any z (preregistration v1.5 interpretation limit).
When the product exported the third mode ``ctau0``, the sampled amplitude and slope are the 3-parameter fit values (the
2-parameter ones are kept as ``tau0_amp_2par`` / ``dtau0_2par``).
It contains no science values, makes no verdict, and never touches the input directories. The preregistered predictions are read
against these numbers by hand in the private notes.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

KIM_AMP, KIM_SLOPE, TAU0_PIVOT_Z = 2.3e-3, 3.65, 3.0
Z13 = np.array([2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
GATE = dict(rhat=1.01, ess=400.0, ebfmi=0.3, treedepth=0.02)
COSMO_BOX = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9), "herei": (3.5, 4.1), "heref": (2.6, 3.2), "alphaq": (1.3, 2.5), "hub": (0.65, 0.75),
             "omegamh2": (0.14, 0.146), "hireionz": (6.5, 8.0), "bhfeedback": (0.03, 0.07)}
MF_BOX = {"tau0_amp": (0.75, 1.25), "dtau0": (-0.4, 0.25)}
NS_PARTNERS = ("tau0_amp", "Ap", "hub", "alpha_subdla", "alpha_lls", "f_res_amp", "dtau0")


def kim(z):
    return KIM_AMP * (1.0 + np.asarray(z, float)) ** KIM_SLOPE


def find_root(d):
    """The chain root of a product directory; prefers an unblinded export when one exists."""
    pn = sorted(glob.glob(os.path.join(d, "*.paramnames")))
    if not pn:
        raise SystemExit(f"REFUSE: no .paramnames in {d}")
    unb = [p for p in pn if ".unblinded." in os.path.basename(p)]
    p = unb[0] if unb else pn[0]
    return os.path.basename(p)[: -len(".paramnames")]


def load_product(d):
    root = find_root(d)
    names = [l.split()[0] for l in open(os.path.join(d, root + ".paramnames")) if l.strip()]
    files = sorted(glob.glob(os.path.join(d, root + ".[0-9]*.txt")))
    if not files:
        raise SystemExit(f"REFUSE: no chain files for root {root} in {d}")
    rows = np.vstack([np.loadtxt(f) for f in files])
    if rows.shape[1] != len(names) + 2:
        raise SystemExit(f"REFUSE: column count {rows.shape[1]} != 2 + {len(names)} names in {d}")
    cols = {n: rows[:, 2 + i] for i, n in enumerate(names)}
    cols["_minusloglike"] = rows[:, 1]
    base = root.replace(".unblinded", "")
    nz_p = os.path.join(d, base + ".nuisance.npz")
    if os.path.exists(nz_p):
        nz = np.load(nz_p)
        for k in nz.files:
            v = np.asarray(nz[k]).reshape(-1)
            if v.shape[0] == rows.shape[0]:
                cols[k] = v
    health_p = os.path.join(d, base + ".health.json")
    health = json.load(open(health_p)) if os.path.exists(health_p) else {}
    ladder = [n for n in names if n.startswith("tau0_z")]
    z = np.asarray(health.get("z_kept") or Z13[-len(ladder):], float)
    L = np.column_stack([cols[n] for n in ladder])
    x = np.log((1.0 + z) / (1.0 + TAU0_PIVOT_Z)); y = np.log(L / kim(z)[None, :])
    A2 = np.column_stack([np.ones_like(x), x]); c2, *_ = np.linalg.lstsq(A2, y.T, rcond=None)
    cols["tau0_amp"] = np.exp(c2[0]); cols["dtau0"] = c2[1]; resid2 = float(np.abs(y.T - A2 @ c2).max())
    A3 = np.column_stack([np.ones_like(x), x, x ** 2]); c3, *_ = np.linalg.lstsq(A3, y.T, rcond=None)
    cols["tau0_amp_3par"] = np.exp(c3[0]); cols["dtau0_3par"] = c3[1]; cols["ctau0_3par"] = c3[2]; resid3 = float(np.abs(y.T - A3 @ c3).max())
    if "ctau0" in cols:
        # the product sampled the third mode: the SAMPLED amplitude and slope are the 3-parameter ones (the 2-parameter
        # inversion is curvature-biased); keep the 2-parameter values under explicit names for the record
        cols["tau0_amp_2par"] = cols["tau0_amp"]; cols["dtau0_2par"] = cols["dtau0"]
        cols["tau0_amp"] = cols["tau0_amp_3par"]; cols["dtau0"] = cols["dtau0_3par"]
    diag = health.get("diagnostic") or None
    box = dict(COSMO_BOX); box.update(MF_BOX)
    applied = (diag or {}).get("applied") or {}
    if applied.get("tau0_amp_range"):
        box["tau0_amp"] = tuple(float(v) for v in applied["tau0_amp_range"])
    alpha = L / kim(z)[None, :]                                    # the ladder in the emulator's alpha = tau0/Kim coordinate
    band = dict(lo=0.75, hi=1.25, frac_any_z_outside=float(np.mean(np.any((alpha < 0.75) | (alpha > 1.25), axis=1))),
                frac_outside_by_z={float(zz): float(np.mean((alpha[:, i] < 0.75) | (alpha[:, i] > 1.25))) for i, zz in enumerate(z)})
    return dict(dir=d, root=root, names=names, cols=cols, n=rows.shape[0], z=z, ladder=L, health=health, diagnostic=diag, box=box,
                ladder_fit=dict(resid_2par=resid2, resid_3par=resid3, has_ctau0_site=("ctau0" in cols)), emulator_band=band)


def gate(h):
    if not h:
        return dict(label="UNKNOWN", reasons=["no health file"])
    reasons = []
    if h.get("rhat_max", 9) >= GATE["rhat"]: reasons.append("rhat")
    if min(h.get("ess_bulk_min", 0), h.get("ess_tail_min", 0)) < GATE["ess"]: reasons.append("ess")
    if h.get("ebfmi_min", 0) < GATE["ebfmi"]: reasons.append("ebfmi")
    if int(h.get("n_divergent", 1)) != 0: reasons.append("divergences")
    if h.get("treedepth_sat_frac", 1) >= GATE["treedepth"]: reasons.append("treedepth")
    return dict(label=("GREEN" if not reasons else "RED"), reasons=reasons,
                values={k: h.get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "ebfmi_min", "n_divergent", "per_chain_div", "treedepth_sat_frac", "target_accept", "seed")})


def rails(P, frac=0.02):
    out = {}
    for p, (lo, hi) in P["box"].items():
        if p not in P["cols"]: continue
        x = P["cols"][p]; w = hi - lo
        out[p] = dict(box=[lo, hi], near_lo=float(np.mean(x < lo + frac * w)), near_hi=float(np.mean(x > hi - frac * w)))
    for p in ("f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1"):
        if p in P["cols"]:
            lx = np.log10(P["cols"][p]); lo, hi = np.log10(0.003), np.log10(0.03); w = hi - lo
            out[p] = dict(box_log10=[lo, hi], near_lo=float(np.mean(lx < lo + frac * w)), near_hi=float(np.mean(lx > hi - frac * w)))
    for p in ("k_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1"):
        if p in P["cols"]:
            lx = np.log10(P["cols"][p]); lo, hi = np.log10(1e-3), np.log10(0.1); w = hi - lo
            out[p] = dict(box_log10=[lo, hi], near_lo=float(np.mean(lx < lo + frac * w)), near_hi=float(np.mean(lx > hi - frac * w)))
    return out


def corr(P, a, b):
    if a not in P["cols"] or b not in P["cols"]: return None
    return float(np.corrcoef(P["cols"][a], P["cols"][b])[0, 1])


def teff_curve(P, z):
    """Posterior tau_eff(z) from the EXPORTED ladder (exact for any mean-flux model), on the requested z values."""
    idx = [int(np.argmin(np.abs(P["z"] - zz))) for zz in z]
    if any(abs(P["z"][i] - zz) > 1e-6 for i, zz in zip(idx, z)):
        raise SystemExit("REFUSE: requested z not in the product's ladder grid")
    T = P["ladder"][:, idx]
    return dict(mean=T.mean(0).tolist(), p16=np.percentile(T, 16, axis=0).tolist(), p84=np.percentile(T, 84, axis=0).tolist())


def compare(D, B):
    shared = [n for n in D["cols"] if n in B["cols"] and not n.startswith("_") and not n.startswith("tau0_z")]
    shifts = {}
    for n in shared:
        bm, bs = float(B["cols"][n].mean()), float(B["cols"][n].std()); dm, ds = float(D["cols"][n].mean()), float(D["cols"][n].std())
        shifts[n] = dict(baseline_mean=bm, baseline_sd=bs, diag_mean=dm, diag_sd=ds, shift=dm - bm, shift_over_baseline_sd=((dm - bm) / bs if bs > 0 else None), sd_ratio=(ds / bs if bs > 0 else None))
    only_diag = {n: dict(mean=float(D["cols"][n].mean()), sd=float(D["cols"][n].std())) for n in D["cols"] if n not in B["cols"] and not n.startswith("_") and not n.startswith("tau0_z")}
    only_base = [n for n in B["cols"] if n not in D["cols"] and not n.startswith("_") and not n.startswith("tau0_z")]
    z = np.array([zz for zz in D["z"] if np.any(np.abs(B["z"] - zz) < 1e-6)])
    cors = {n: dict(baseline=corr(B, "ns", n), diag=corr(D, "ns", n)) for n in NS_PARTNERS}
    D_on_Bbox = dict(D); D_on_Bbox["box"] = B["box"]                  # the diagnostic draws railed against the BASELINE box (MF1 prediction)
    return dict(schema="eboss_diag_compare.v1", diag_dir=D["dir"], diag_root=D["root"], baseline_dir=B["dir"], baseline_root=B["root"], n_draws=dict(diag=D["n"], baseline=B["n"]),
                diagnostic=D["diagnostic"], gate_diag=gate(D["health"]), gate_baseline=gate(B["health"]), shifts=shifts, only_in_diag=only_diag, only_in_baseline=only_base,
                rails=dict(diag=rails(D), baseline=rails(B), diag_against_baseline_box=rails(D_on_Bbox)), corr_ns=cors, ladder_fit=dict(diag=D["ladder_fit"], baseline=B["ladder_fit"]),
                teff=dict(z=z.tolist(), diag=teff_curve(D, z), baseline=teff_curve(B, z)),
                emulator_band=dict(diag=D["emulator_band"], baseline=B["emulator_band"], note="fraction of draws whose ladder alpha(z) = tau0_z / tau_Kim lies outside the emulator tau0 band [0.75, 1.25] at any z (preregistration v1.5: > 0.10 marks the run EXTRAPOLATED)"))


def markdown(R):
    L = [f"# Diagnostic comparison: {os.path.basename(R['diag_dir'])} vs baseline {os.path.basename(R['baseline_dir'])}", "",
         f"diagnostic overrides (from the health file): {json.dumps(R['diagnostic'])}", "",
         f"health gate of the diagnostic run: **{R['gate_diag']['label']}** {R['gate_diag']['reasons'] or ''} values {json.dumps(R['gate_diag'].get('values'))}", "",
         "## Shifts (diagnostic minus baseline, in baseline sd)", "", "| parameter | baseline mean (sd) | diagnostic mean (sd) | shift | / baseline sd | sd ratio |", "|---|---|---|---|---|---|"]
    for n, s in R["shifts"].items():
        sc = 1e9 if n == "Ap" else 1.0
        f2 = lambda v: (f"{v:+.2f}" if v is not None else "-"); f2u = lambda v: (f"{v:.2f}" if v is not None else "-")
        L.append(f"| {n} | {s['baseline_mean']*sc:.5g} ({s['baseline_sd']*sc:.3g}) | {s['diag_mean']*sc:.5g} ({s['diag_sd']*sc:.3g}) | {s['shift']*sc:+.4g} | {f2(s['shift_over_baseline_sd'])} | {f2u(s['sd_ratio'])} |")
    if R["only_in_diag"]:
        L += ["", "parameters only in the diagnostic run: " + "; ".join(f"{n} mean {v['mean']:.4g} sd {v['sd']:.3g}" for n, v in R["only_in_diag"].items())]
    if R["only_in_baseline"]:
        L += ["", "parameters only in the baseline (not sampled in the diagnostic run): " + ", ".join(R["only_in_baseline"])]
    L += ["", "## Rail fractions (within 2 percent of each product's own box)", "", "| parameter | baseline lo / hi | diagnostic lo / hi | diagnostic box |", "|---|---|---|---|"]
    for n in R["rails"]["baseline"]:
        b = R["rails"]["baseline"][n]; d = R["rails"]["diag"].get(n)
        d_lo = f"{d['near_lo']:.3f}" if d else "-"; d_hi = f"{d['near_hi']:.3f}" if d else "-"; d_box = (d.get("box") or d.get("box_log10")) if d else "-"
        L.append(f"| {n} | {b['near_lo']:.3f} / {b['near_hi']:.3f} | {d_lo} / {d_hi} | {d_box} |")
    L += ["", "diagnostic draws railed against the BASELINE box (for a run whose box changed): " + "; ".join(f"{n} lo {v['near_lo']:.3f} / hi {v['near_hi']:.3f}" for n, v in R["rails"]["diag_against_baseline_box"].items() if n in ("tau0_amp", "dtau0", "Ap", "alphaq", "heref"))]
    L += ["", "## Correlations of ns", "", "| partner | baseline | diagnostic |", "|---|---|---|"]
    for n, c in R["corr_ns"].items():
        L.append(f"| {n} | {('%+.2f' % c['baseline']) if c['baseline'] is not None else '-'} | {('%+.2f' % c['diag']) if c['diag'] is not None else '-'} |")
    L += ["", "## tau_eff(z) posterior means", "", "| z | baseline | diagnostic | ratio - 1 (%) |", "|---|---|---|---|"]
    for i, z in enumerate(R["teff"]["z"]):
        b = R["teff"]["baseline"]["mean"][i]; d = R["teff"]["diag"]["mean"][i]
        L.append(f"| {z:.1f} | {b:.4f} | {d:.4f} | {100*(d/b-1):+.2f} |")
    L += ["", f"emulator tau0 band [0.75, 1.25]: fraction of draws outside at any z, baseline {R['emulator_band']['baseline']['frac_any_z_outside']:.3f}, diagnostic {R['emulator_band']['diag']['frac_any_z_outside']:.3f} (> 0.10 marks the diagnostic EXTRAPOLATED)"]
    L += ["", f"ladder fits: baseline 2-par residual {R['ladder_fit']['baseline']['resid_2par']:.2e}, diagnostic 2-par residual {R['ladder_fit']['diag']['resid_2par']:.2e}, 3-par residual {R['ladder_fit']['diag']['resid_3par']:.2e}, ctau0 site exported: {R['ladder_fit']['diag']['has_ctau0_site']}"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True); ap.add_argument("--baseline-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for ext in (".json", ".md"):
        if os.path.exists(a.out + ext):
            raise SystemExit(f"REFUSE: output exists: {a.out}{ext}")
    D = load_product(a.diag_dir); B = load_product(a.baseline_dir)
    R = compare(D, B)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R, open(a.out + ".json", "w"), indent=1, default=float); open(a.out + ".md", "w").write(markdown(R))
    print(f"wrote {a.out}.json and .md; gate {R['gate_diag']['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
