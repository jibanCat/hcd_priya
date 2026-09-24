#!/usr/bin/env python3
"""Export the certification-campaign material for the emulator paper (PI #24: regenerate F7 / the certification table from
FROZEN campaign outputs; no new sampling): per-mock pulls and SBC ranks for n_P, A_P (unit-cube channels, as gated),
alpha_LLS / alpha_subDLA / alpha_DLA, tau0amp and dtau0 (regressed from the 13-rung ladder exactly as
scripts/analyze_sbc_perleg.py does), the 95 percent Beta(k, N+1-k) rank-ECDF band, the population summaries
cross-checked against the committed gate JSONs, and the F8 layers (pooled within-mock correlation matrices per leg
and one representative mock's draws). Legs: eBOSS A1c (N = 48, PROMOTED) and KS A3c (N = 48 immutable FAIL; N = 96
one-time adjudication). DESI is NOT exported (out of the paper's scope).

Blind status: BLIND-SAFE. Every number is a self-draw SBC product on PRIYA mocks; no observed P1D, no real-data posterior.
Inputs are the frozen pkls (sha256-verified against the committed manifests); nothing inside the code repo is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

REPO = "/home/mfho/hcd_priya"
NOTES = "/home/mfho/hcd_priya_notes"
SCALAR = ["ns", "Ap", "alpha_subdla", "alpha_lls", "alpha_dla"]
Z_TAU0 = np.array([2.2 + 0.2 * i for i in range(13)])
_lx = np.log(1.0 + Z_TAU0); LX_C = _lx - _lx.mean()
F8_COLS = ["ns", "Ap", "tau0amp", "dtau0", "alpha_lls", "alpha_subdla", "alpha_dla"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def tau0_amp_slope(ladder):
    """COPIED semantics of analyze_sbc_perleg.tau0_amp_slope: regress ln tau0(z) on centered ln(1+z)."""
    y = np.log(np.clip(ladder, 1e-8, None))
    slope = np.sum(LX_C * (y - y.mean())) / np.sum(LX_C ** 2)
    return float(np.exp(y.mean())), float(slope)


def load_population(outdir, sha_file, n_total):
    recorded = {}
    with open(sha_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                recorded[os.path.basename(parts[1])] = parts[0]
    mocks = []
    for m in range(n_total):
        p = os.path.join(outdir, f"mock_{m:04d}.pkl")
        base = os.path.basename(p)
        if base not in recorded or sha256(p) != recorded[base]:
            raise RuntimeError(f"sha mismatch or missing manifest entry for {base}")
        with open(p, "rb") as f:
            mocks.append(pickle.load(f))
    return mocks


def per_mock(mocks):
    names = list(mocks[0]["names"])
    idx = {k: names.index(k) for k in SCALAR}
    tau_idx = [names.index(f"tau0_z{i}") for i in range(13)]
    out = {k: dict(pull=[], rank=[], truth=[], post_mean=[], post_sd=[]) for k in SCALAR + ["tau0amp", "dtau0"]}
    L_all, ndiv = [], []
    f8_corr, f8_truth = [], []
    for d in mocks:
        dr = np.asarray(d["draws"], float); t = np.asarray(d["truth_vec"], float); L = dr.shape[0]
        L_all.append(L); ndiv.append(int(d.get("n_div", -1)))
        for k in SCALAR:
            j = idx[k]; col = dr[:, j]; sd = col.std(ddof=1); mu = col.mean()
            out[k]["pull"].append(float((mu - t[j]) / sd) if sd > 0 else np.nan)
            out[k]["rank"].append(float(np.mean(col < t[j]))); out[k]["truth"].append(float(t[j]))
            out[k]["post_mean"].append(float(mu)); out[k]["post_sd"].append(float(sd))
        amps = np.empty(L); slopes = np.empty(L)
        for i in range(L):
            amps[i], slopes[i] = tau0_amp_slope(dr[i, tau_idx])
        amp_t, slope_t = tau0_amp_slope(t[tau_idx])
        for k, arr, tv in (("tau0amp", amps, amp_t), ("dtau0", slopes, slope_t)):
            sd = arr.std(ddof=1); out[k]["pull"].append(float((arr.mean() - tv) / sd)); out[k]["rank"].append(float(np.mean(arr < tv)))
            out[k]["truth"].append(float(tv)); out[k]["post_mean"].append(float(arr.mean())); out[k]["post_sd"].append(float(sd))
        # F8: within-mock correlation among the 7 named channels
        M = np.column_stack([dr[:, idx["ns"]], dr[:, idx["Ap"]], amps, slopes, dr[:, idx["alpha_lls"]], dr[:, idx["alpha_subdla"]], dr[:, idx["alpha_dla"]]])
        f8_corr.append(np.corrcoef(M, rowvar=False)); f8_truth.append([t[idx["ns"]], t[idx["Ap"]], amp_t, slope_t, t[idx["alpha_lls"]], t[idx["alpha_subdla"]], t[idx["alpha_dla"]]])
    return out, np.array(L_all), np.array(ndiv), np.array(f8_corr), np.array(f8_truth), names, idx, tau_idx


def summary(vals):
    a = np.asarray(vals, float); a = a[np.isfinite(a)]
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)), sem=float(a.std(ddof=1) / np.sqrt(a.size)), n=int(a.size))


def beta_band(N, alpha=0.05):
    from scipy.stats import beta
    k = np.arange(1, N + 1)
    return dict(k=k.tolist(), lo=beta.ppf(alpha / 2, k, N + 1 - k).tolist(), hi=beta.ppf(1 - alpha / 2, k, N + 1 - k).tolist(),
                note="pointwise 95 percent Beta(k, N+1-k) band for the k-th order statistic of N uniform ranks")


def representative_mock(pm, N):
    """The mock whose (ns, Ap) truth (unit cube) is closest to the population median truth: a preregistered, value-free rule."""
    T = np.column_stack([pm["ns"]["truth"], pm["Ap"]["truth"]]); med = np.median(T, axis=0)
    return int(np.argmin(np.sum((T - med) ** 2, axis=1)))


def crosscheck(pm, gate_json, leg, keys=("ns", "Ap", "alpha_subdla", "alpha_lls", "alpha_dla", "tau0amp", "dtau0"), rtol=1e-6):
    g = json.load(open(gate_json))["legs"][leg]["pulls"]
    rep = {}
    for k in keys:
        s = summary(pm[k]["pull"]); ok = all(abs(s[q] - g[k][q]) <= rtol * max(1.0, abs(g[k][q])) for q in ("mean", "std", "sem")) and s["n"] == g[k]["n"]
        rep[k] = dict(recomputed=s, gate=g[k], match=bool(ok))
        if not ok:
            raise RuntimeError(f"cross-check failed for {leg} {k}: {s} vs gate {g[k]}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/scratch/cavestru_root/cavestru1/mfho/cert_2026-07")
    ap.add_argument("--out", default=os.path.join(NOTES, "artifacts", "paper_exports", "certification_2026-09-24"))
    a = ap.parse_args()
    art = os.path.join(NOTES, "docs", "superpowers", "a3c-artifacts")
    legs = {
        "eBOSS": dict(dir=os.path.join(a.root, "armp_eBOSS_corrected_v2"), sha=os.path.join(art, "a1c_eboss_n48_sha256.txt"), n=48,
                      gate=os.path.join(NOTES, "figures/analysis/05_likelihood/armp_eboss_corrected_gate.json"), verdict="PROMOTED Row 1 (PI #9 3e.1); PASS on the frozen gate; N = 48 limits binding"),
        "KS_n48": dict(dir=os.path.join(a.root, "armp_KS_corrected_v1"), sha=os.path.join(art, "a3c96_full96_sha256.txt"), n=48,
                       gate=os.path.join(NOTES, "figures/analysis/05_likelihood/armp_ks_corrected_gate.json"), verdict="FAIL (n_s pull dispersion), immutable N = 48 row (PI #10)"),
        "KS_n96": dict(dir=os.path.join(a.root, "armp_KS_corrected_v1"), sha=os.path.join(art, "a3c96_full96_sha256.txt"), n=96,
                       gate=os.path.join(NOTES, "figures/analysis/05_likelihood/armp_ks_corrected_n96_gate.json"), verdict="FAIL, closed-final (PI #12): A_p passes, n_s tail-concentrated dispersion excess, no mechanism"),
    }
    os.makedirs(a.out, exist_ok=True)
    npz = {}; table = {}; prov_inputs = {}
    for name, cfg in legs.items():
        mocks = load_population(cfg["dir"], cfg["sha"], cfg["n"])
        pm, L_all, ndiv, f8c, f8t, names, idx, tau_idx = per_mock(mocks)
        legkey = "eBOSS" if name == "eBOSS" else "KS"
        xc = crosscheck(pm, cfg["gate"], legkey)
        N = cfg["n"]
        rep = representative_mock(pm, N)
        for k, v in pm.items():
            for q, arr in v.items():
                npz[f"{name}/{k}/{q}"] = np.asarray(arr, float)
        band = beta_band(N)
        npz[f"{name}/rank_band_lo"] = np.asarray(band["lo"]); npz[f"{name}/rank_band_hi"] = np.asarray(band["hi"])
        npz[f"{name}/L"] = L_all; npz[f"{name}/n_div"] = ndiv
        npz[f"{name}/f8_within_mock_corr"] = f8c; npz[f"{name}/f8_truths"] = f8t
        npz[f"{name}/f8_pooled_corr"] = np.nanmean(f8c, axis=0)
        dr = np.asarray(mocks[rep]["draws"], float)
        amps = np.array([tau0_amp_slope(dr[i, tau_idx])[0] for i in range(dr.shape[0])]); slopes = np.array([tau0_amp_slope(dr[i, tau_idx])[1] for i in range(dr.shape[0])])
        npz[f"{name}/f8_representative_draws"] = np.column_stack([dr[:, idx["ns"]], dr[:, idx["Ap"]], amps, slopes, dr[:, idx["alpha_lls"]], dr[:, idx["alpha_subdla"]], dr[:, idx["alpha_dla"]]])
        npz[f"{name}/f8_representative_truth"] = f8t[rep]
        table[name] = dict(N=N, verdict=cfg["verdict"], L_median=float(np.median(L_all)), L_range=[int(L_all.min()), int(L_all.max())], n_div_total=int(ndiv[ndiv >= 0].sum()),
                           summaries={k: summary(v["pull"]) for k, v in pm.items()},
                           rank_ks_p={k: float(__import__("scipy.stats", fromlist=["kstest"]).kstest(np.asarray(v["rank"]), "uniform").pvalue) for k, v in pm.items()},
                           gate_crosscheck={k: v["match"] for k, v in xc.items()}, representative_mock=rep, f8_columns=F8_COLS,
                           gate_json=dict(path=cfg["gate"], sha256=sha256(cfg["gate"])))
        prov_inputs[name] = dict(dir=cfg["dir"], resolved=os.path.realpath(cfg["dir"]), manifest=cfg["sha"], manifest_sha256=sha256(cfg["sha"]), n=N)
    np.savez(os.path.join(a.out, "cert_layers.npz"), **npz)
    with open(os.path.join(a.out, "cert_table.json"), "w") as f:
        json.dump(table, f, indent=1, sort_keys=True); f.write("\n")
    md = ["# Certification summary for the paper (self-draw SBC, deployed geometry; BLIND-SAFE)", "",
          "| leg | N | n_P pull mean +/- sd (sem) | A_P pull mean +/- sd (sem) | n_P rank KS p | A_P rank KS p | verdict |", "|---|---|---|---|---|---|---|"]
    for name, t in table.items():
        s1, s2 = t["summaries"]["ns"], t["summaries"]["Ap"]
        md.append(f"| {name} | {t['N']} | {s1['mean']:+.3f} +/- {s1['std']:.3f} ({s1['sem']:.3f}) | {s2['mean']:+.3f} +/- {s2['std']:.3f} ({s2['sem']:.3f}) | {t['rank_ks_p']['ns']:.3f} | {t['rank_ks_p']['Ap']:.3f} | {t['verdict']} |")
    md += ["", "Gate: |pull mean| <= 0.30 and pull sd <= 1.1 on both channels, uniform ranks (KS p > 0.05, ECDF inside the Beta band). Binding limits of the eBOSS certificate: corrected self-consistency certification, residual bias < 0.30 sigma NOT established at N = 48, real-data validation NOT addressed."]
    with open(os.path.join(a.out, "cert_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    prov = dict(artifact="certification material for the emulator paper (F7 replacement layers, certification table, F8 layers)", created_utc=datetime.now(timezone.utc).isoformat(),
                blind_status="BLIND-SAFE (self-draw SBC on PRIYA mocks; no real data)", code_commit=git("rev-parse", "HEAD"), code_dirty=bool(git("status", "--porcelain")),
                script=dict(path="scripts/export_certification_paper.py", sha256=sha256(os.path.abspath(__file__))), inputs=prov_inputs,
                conventions="pull = (posterior mean - truth) / posterior sd (ddof 1); rank = fraction of draws below truth; n_P and A_P in the emulator unit cube (as gated); tau0amp/dtau0 by the analyzer's ln-ladder regression; summaries cross-checked against the committed gate JSONs (match required)",
                outputs={f: sha256(os.path.join(a.out, f)) for f in ("cert_layers.npz", "cert_table.json", "cert_table.md")},
                environment=dict(python=platform.python_version(), numpy=np.__version__, host=platform.node()),
                verdicts_of_record="notes:docs/superpowers/2026-09-24-CERTIFICATION-CAMPAIGN-CLOSURE-RECORD.md",
                figures="F7 replacement (rank ECDFs + Beta band; pull forests), certification table, F8 (pooled within-mock correlations; representative-mock draws chosen by the median-truth rule)")
    with open(os.path.join(a.out, "PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True); f.write("\n")
    print(f"wrote {a.out}: legs {list(table)}; cross-checks {[t['gate_crosscheck'] for t in table.values()]}")


if __name__ == "__main__":
    main()
