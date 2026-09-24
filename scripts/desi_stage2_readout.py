#!/usr/bin/env python3
"""desi_stage2_readout.py -- DESI Stage 2 readout: compare each re-sampled mock (strong run, raw chains) with its stored A2c
draws and apply the preregistered decision rule (preregistration v1.1 sections 4 and 5). Read-only; no sampling.

Per mock, in C4 = (ns, tau0_amp, dtau0, k_SiIII_DESI_z1) (k in log10) and P3 = C4[:3]:
  (a) corr(ns, dtau0) and the full C4 correlation matrix, stored versus strong (pooled 4 chains);
  (b) whitened mean shift delta = Sigma_stored^{-1/2} (mu_strong - mu_stored), and |delta|;
  (c) sd ratios sd_strong / sd_stored;
  (d) the strong run's battery summary; first-chain-only versus pooled summaries;
  (e) the replica comparison (from the runner's JSON).
Finite-L reference: the strong run's draws are subsampled to the STORED L with the stored thinning pattern (one chain, every
`step`-th draw from a random offset; 2000 replicates) and (a) to (c) are recomputed, giving central 95 percent bands.
Decision (section 5): M-A if >= 3 of 4 TAIL mocks have the stored corr(ns, dtau0) outside the strong run's finite-L band, OR
|delta_C4| > 1.0 on >= 3 of 4 tail mocks with <= 1 of 4 controls; M-B-orientation if stored and strong agree within the bands
(corr inside, |delta| <= 1.0) on >= 3 of 4 tail AND >= 3 of 4 control mocks AND the strong runs assert the same negative
coupling (median corr(ns, dtau0) over the 8 mocks < -0.1); AMBIGUOUS otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

C4 = ("ns", "tau0_amp", "dtau0", "k_SiIII_DESI_z1")
LOG10 = ("k_SiIII_DESI_z1",)
TAIL = (45, 9, 11, 28)
CONTROL = (0, 16, 1, 20)
B_FINITE_L = 2000
SEED = 20260926


def _c4_from_stored(z):
    names = list(z["names"]); dr = np.asarray(z["draws"], float)
    X = np.column_stack([dr[:, names.index("ns")], np.asarray(z["sites_extra"]["tau0_amp"]["draws"], float), np.asarray(z["sites_extra"]["dtau0"]["draws"], float),
                         np.log10(np.asarray(z["sites_extra"]["k_SiIII_DESI_z1"]["draws"], float))])
    t = np.array([float(z["truth_vec"][names.index("ns")]), float(z["sites_extra"]["tau0_amp"]["truth"]), float(z["sites_extra"]["dtau0"]["truth"]), np.log10(float(z["sites_extra"]["k_SiIII_DESI_z1"]["truth"]))])
    return X, t


def _c4_from_chain(chain, names):
    dr = np.asarray(chain["draws"], float); s = chain["samples"]
    return np.column_stack([dr[:, list(names).index("ns")], np.asarray(s["tau0_amp"], float).reshape(-1), np.asarray(s["dtau0"], float).reshape(-1),
                            np.log10(np.asarray(s["k_SiIII_DESI_z1"], float).reshape(-1))])


def _sym_inv_sqrt(S):
    w, V = np.linalg.eigh(S)
    return (V / np.sqrt(np.clip(w, 1e-300, None))) @ V.T


def stats(X_stored, X_strong):
    """(a) to (c) for one pair of draw matrices in the same coordinates."""
    mu_s, mu_n = X_stored.mean(axis=0), X_strong.mean(axis=0)
    S_s, S_n = np.cov(X_stored.T, ddof=1), np.cov(X_strong.T, ddof=1)
    C_s, C_n = np.corrcoef(X_stored, rowvar=False), np.corrcoef(X_strong, rowvar=False)
    delta = _sym_inv_sqrt(S_s) @ (mu_n - mu_s)
    d3 = _sym_inv_sqrt(S_s[:3, :3]) @ (mu_n[:3] - mu_s[:3])
    return dict(corr_ns_dtau0_stored=float(C_s[0, 2]), corr_ns_dtau0_strong=float(C_n[0, 2]), corr_ns_k_stored=float(C_s[0, 3]), corr_ns_k_strong=float(C_n[0, 3]),
                corr4_stored=C_s.tolist(), corr4_strong=C_n.tolist(), delta_c4=delta.tolist(), delta_c4_norm=float(np.linalg.norm(delta)),
                delta_p3=d3.tolist(), delta_p3_norm=float(np.linalg.norm(d3)), sd_ratio=(np.sqrt(np.diag(S_n)) / np.sqrt(np.diag(S_s))).tolist(),
                mu_stored=mu_s.tolist(), mu_strong=mu_n.tolist())


def finite_L_bands(X_chains, L, step, rng, B=B_FINITE_L):
    """Subsample the strong chains to the stored L with the stored thinning pattern; return bands for corr(ns,dtau0) and |delta| relative
    to the strong pooled posterior (the reference for 'agreement within finite-L noise')."""
    pooled = np.concatenate(X_chains, axis=0)
    mu_p, S_p = pooled.mean(axis=0), np.cov(pooled.T, ddof=1); W = _sym_inv_sqrt(S_p)
    corr = np.empty(B); dn = np.empty(B); dn3 = np.empty(B)
    for b in range(B):
        X = X_chains[rng.integers(len(X_chains))]
        n = X.shape[0]; off = int(rng.integers(0, max(1, step)))
        idx = (off + step * np.arange(L)) % n
        Y = X[idx]
        corr[b] = np.corrcoef(Y[:, 0], Y[:, 2])[0, 1]
        d = W @ (Y.mean(axis=0) - mu_p); dn[b] = np.linalg.norm(d)
        d3 = _sym_inv_sqrt(S_p[:3, :3]) @ (Y.mean(axis=0)[:3] - mu_p[:3]); dn3[b] = np.linalg.norm(d3)
    return dict(corr_ns_dtau0_band95=[float(np.quantile(corr, 0.025)), float(np.quantile(corr, 0.975))], corr_median=float(np.median(corr)),
                delta_norm_q95=float(np.quantile(dn, 0.95)), delta_p3_norm_q95=float(np.quantile(dn3, 0.95)), B=B, L=int(L), step=int(step))


def analyze_mock(stored_pkl, stage2_pkl, stage2_json, rng):
    z = pickle.load(open(stored_pkl, "rb")); r = pickle.load(open(stage2_pkl, "rb")); j = json.load(open(stage2_json))
    names = list(r["strong"]["names"])
    Xs, t = _c4_from_stored(z)
    Xc = [_c4_from_chain(c, names) for c in r["strong"]["chains"]]
    Xn = np.concatenate(Xc, axis=0)
    st = stats(Xs, Xn)
    first = stats(Xs, Xc[0])
    step = int(j.get("replica", {}).get("step", max(1, int(round(600 / max(1, z["L"]))))))
    bands = finite_L_bands(Xc, int(z["L"]), step, rng)
    corr_inside = bool(bands["corr_ns_dtau0_band95"][0] <= st["corr_ns_dtau0_stored"] <= bands["corr_ns_dtau0_band95"][1])
    delta_ok = bool(st["delta_c4_norm"] <= 1.0)
    per_chain_corr = [float(np.corrcoef(X[:, 0], X[:, 2])[0, 1]) for X in Xc]
    return dict(mock=int(r["mock"]), L_stored=int(z["L"]), step=step, stats=st, first_chain_only=first, finite_L=bands, corr_inside_band=corr_inside,
                delta_within_1=delta_ok, agree=bool(corr_inside and delta_ok), per_chain_corr_ns_dtau0=per_chain_corr,
                battery=j["strong"].get("battery", {}).get("rhat_max") and dict(rhat_max=j["strong"]["battery"].get("rhat_max"), ess_bulk_min=j["strong"]["battery"].get("ess_bulk_min"),
                                                                              ess_tail_min=j["strong"]["battery"].get("ess_tail_min"), ebfmi_min=j["strong"]["battery"].get("ebfmi_min"),
                                                                              treedepth_sat_frac=j["strong"]["battery"].get("treedepth_sat_frac"), n_divergent=j["strong"]["battery"].get("n_divergent")),
                gate=j["strong"].get("gate", {}).get("pilot_gate_passed"), replica=j.get("replica", {}).get("compare"), identity=j.get("identity"),
                truth_c4=t.tolist(), whitened_truth_displacement_stored=(_sym_inv_sqrt(np.cov(Xs.T, ddof=1)) @ (Xs.mean(axis=0) - t)).tolist(),
                whitened_truth_displacement_strong=(_sym_inv_sqrt(np.cov(Xn.T, ddof=1)) @ (Xn.mean(axis=0) - t)).tolist())


def decide(per_mock):
    by = {m["mock"]: m for m in per_mock}
    tail = [by[m] for m in TAIL if m in by]; ctrl = [by[m] for m in CONTROL if m in by]
    n_tail_corr_out = sum(not m["corr_inside_band"] for m in tail); n_tail_delta_out = sum(not m["delta_within_1"] for m in tail); n_ctrl_delta_out = sum(not m["delta_within_1"] for m in ctrl)
    n_tail_agree = sum(m["agree"] for m in tail); n_ctrl_agree = sum(m["agree"] for m in ctrl)
    med_corr_strong = float(np.median([m["stats"]["corr_ns_dtau0_strong"] for m in per_mock])) if per_mock else float("nan")
    complete = (len(tail) == 4 and len(ctrl) == 4)
    if not complete:
        label = "INCOMPLETE"
    elif n_tail_corr_out >= 3 or (n_tail_delta_out >= 3 and n_ctrl_delta_out <= 1):
        label = "M-A (the stored draws misrepresent the target)"
    elif n_tail_agree >= 3 and n_ctrl_agree >= 3 and med_corr_strong < -0.1:
        label = "M-B-orientation (exact-posterior feature)"
    else:
        label = "AMBIGUOUS"
    return dict(label=label, n_tail_corr_outside=n_tail_corr_out, n_tail_delta_over_1=n_tail_delta_out, n_control_delta_over_1=n_ctrl_delta_out,
                n_tail_agree=n_tail_agree, n_control_agree=n_ctrl_agree, median_corr_ns_dtau0_strong=med_corr_strong, complete=complete, tail=list(TAIL), control=list(CONTROL))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stored-dir", required=True); ap.add_argument("--stage2-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mocks", type=int, nargs="*", default=list(TAIL) + list(CONTROL))
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)
    per = []
    for m in a.mocks:
        sp = os.path.join(a.stored_dir, f"mock_{m:04d}.pkl"); p2 = os.path.join(a.stage2_dir, f"stage2_mock_{m:04d}.pkl"); j2 = os.path.join(a.stage2_dir, f"stage2_mock_{m:04d}.json")
        if not (os.path.exists(p2) and os.path.exists(j2)):
            continue
        per.append(analyze_mock(sp, p2, j2, rng))
    dec = decide(per)
    out = dict(schema="desi_stage2_readout.v1", prereg="2026-09-24-DESI-STAGE2-PREREGISTRATION-v1.1", seed=SEED, per_mock=per, decision=dec, coords=list(C4))
    if os.path.exists(a.out + ".json"):
        raise SystemExit(f"REFUSE: output exists: {a.out}.json")
    with open(a.out + ".json.tmp", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, allow_nan=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)); f.write("\n")
    os.replace(a.out + ".json.tmp", a.out + ".json")
    L = [f"# DESI Stage 2 readout ({len(per)} mocks): **{dec['label']}**", "", "| mock | set | L | corr(ns,dtau0) stored | strong | band95 | inside | delta_C4 norm | q95(finite-L) | sd ratios | gate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in per:
        s = m["stats"]; b = m["finite_L"]; tag = "TAIL" if m["mock"] in TAIL else ("CONTROL" if m["mock"] in CONTROL else "-")
        L.append(f"| {m['mock']} | {tag} | {m['L_stored']} | {s['corr_ns_dtau0_stored']:+.3f} | {s['corr_ns_dtau0_strong']:+.3f} | [{b['corr_ns_dtau0_band95'][0]:+.3f}, {b['corr_ns_dtau0_band95'][1]:+.3f}] | {m['corr_inside_band']} | {s['delta_c4_norm']:.2f} | {b['delta_norm_q95']:.2f} | {[round(x, 2) for x in s['sd_ratio']]} | {m['gate']} |")
    L += ["", f"decision counts: {json.dumps({k: v for k, v in dec.items() if k not in ('tail', 'control')})}"]
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}.json/.md: {dec['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
