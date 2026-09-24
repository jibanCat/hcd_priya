#!/usr/bin/env python3
"""desi_stage2_readout.py -- DESI Stage 2 readout: compare each re-sampled mock (strong run, raw chains) with its stored A2c
draws and apply the preregistered decision rule (preregistration v1.1 sections 4 and 5). Read-only; no sampling.

Per mock, in C4 = (ns, tau0_amp, dtau0, k_SiIII_DESI_z1) (k in log10) and P3 = C4[:3]:
  (a) corr(ns, dtau0) and the full C4 correlation matrix, stored versus strong (pooled 4 chains);
  (b) whitened mean shift delta = Sigma_stored^{-1/2} (mu_strong - mu_stored), and |delta|;
  (c) sd ratios sd_strong / sd_stored;
  (d) the strong run's battery summary; first-chain-only versus pooled summaries;
  (e) the replica comparison (from the runner's JSON).
Finite-L reference (prereg v1.2): the strong chains are thinned by the STORED step (derived from the stored L, every step-th draw
at each of the `step` offsets), cut into contiguous blocks of BLOCK thinned draws, and each replicate draws ceil(L / BLOCK) blocks
with replacement across the 4 chains and offsets (a block bootstrap; B = 2000); (a) to (c) are recomputed on each replicate
against the strong pooled posterior, giving central 95 percent bands (corr, |delta| C4 and P3, sd ratios).
Decision (section 5; |delta| = the C4 norm): a mock COUNTS only if its strong run passed the sampler criteria (otherwise INCOMPLETE).
M-A if >= 3 of 4 TAIL mocks have the stored corr(ns, dtau0) outside the strong run's finite-L band, OR |delta_C4| > 1.0 on >= 3 of 4
tail mocks with <= 1 of 4 controls; M-B-orientation if stored and strong agree within the bands
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
BLOCK = 10
N_RAW = 600
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


def stored_step(L, n_raw=N_RAW):
    """The thinning step that produced L stored draws from n_raw: the unique s with ceil(n_raw / s) == L (refuse if ambiguous)."""
    cands = [st for st in range(1, n_raw + 1) if int(np.ceil(n_raw / st)) == int(L)]
    if len(cands) != 1:
        raise SystemExit(f"REFUSE: thinning step ambiguous for L={L}: {cands}")
    return cands[0]


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


def finite_L_bands(X_chains, L, step, rng, B=None, block=BLOCK):
    """Block bootstrap of the stored-L estimator around the strong pooled posterior (prereg v1.2 section 4)."""
    B = B_FINITE_L if B is None else B
    pooled = np.concatenate(X_chains, axis=0)
    blocks = []
    for X in X_chains:
        for off in range(step):
            Y = X[off::step]
            for b0 in range(0, Y.shape[0] - block + 1, block):
                blocks.append(Y[b0:b0 + block])
    nb = int(np.ceil(L / block))
    corr = np.empty(B); dn = np.empty(B); dn3 = np.empty(B); sdr = np.empty((B, pooled.shape[1]))
    for b in range(B):
        Y = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=nb)], axis=0)[:L]
        st = stats(Y, pooled)
        corr[b] = st["corr_ns_dtau0_stored"]          # the L-draw estimate versus the pooled reference
        dn[b] = st["delta_c4_norm"]; dn3[b] = st["delta_p3_norm"]; sdr[b] = np.asarray(st["sd_ratio"])
    return dict(corr_ns_dtau0_band95=[float(np.quantile(corr, 0.025)), float(np.quantile(corr, 0.975))], corr_median=float(np.median(corr)),
                delta_norm_q95=float(np.quantile(dn, 0.95)), delta_p3_norm_q95=float(np.quantile(dn3, 0.95)),
                sd_ratio_band95=[[float(np.quantile(sdr[:, c], 0.025)), float(np.quantile(sdr[:, c], 0.975))] for c in range(sdr.shape[1])],
                B=B, L=int(L), step=int(step), block=int(block), n_blocks_pool=len(blocks))


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def analyze_mock(stored_pkl, stage2_pkl, stage2_json, seed=None):
    z = pickle.load(open(stored_pkl, "rb")); r = pickle.load(open(stage2_pkl, "rb")); j = json.load(open(stage2_json))
    want_s = j.get("stored_pkl", {}).get("sha256"); want_o = j.get("output_pkl_sha256")
    if not want_s or not want_o:
        raise SystemExit(f"REFUSE: runner record lacks the sha256 fields ({stage2_json})")
    if want_s != _sha256(stored_pkl):
        raise SystemExit(f"REFUSE: stored pkl sha256 differs from the runner record ({stored_pkl})")
    if want_o != _sha256(stage2_pkl):
        raise SystemExit(f"REFUSE: stage2 pkl sha256 differs from the runner record ({stage2_pkl})")
    ident = j.get("identity", {})
    flags = [v for k, v in ident.items() if isinstance(v, bool)]
    if not flags or not all(flags):
        raise SystemExit(f"REFUSE: identity checks missing or not all true in {stage2_json}")
    names = list(r["strong"]["names"])
    Xs, t = _c4_from_stored(z)
    Xc = [_c4_from_chain(c, names) for c in r["strong"]["chains"]]
    Xn = np.concatenate(Xc, axis=0)
    if not (np.isfinite(Xs).all() and np.isfinite(Xn).all() and np.isfinite(t).all()):
        raise SystemExit(f"REFUSE: non-finite coordinates for {stage2_pkl}")
    st = stats(Xs, Xn)
    first = stats(Xs, Xc[0])
    step = stored_step(int(z["L"]))
    rng = np.random.default_rng([SEED, int(r["mock"])])
    bands = finite_L_bands(Xc, int(z["L"]), step, rng)
    vals = [st["corr_ns_dtau0_stored"], st["corr_ns_dtau0_strong"], st["delta_c4_norm"]] + bands["corr_ns_dtau0_band95"] + [bands["delta_norm_q95"]]
    if not np.all(np.isfinite(vals)):
        raise SystemExit(f"REFUSE: non-finite decision inputs for mock {r['mock']}")
    corr_inside = bool(bands["corr_ns_dtau0_band95"][0] <= st["corr_ns_dtau0_stored"] <= bands["corr_ns_dtau0_band95"][1])
    delta_ok = bool(st["delta_c4_norm"] <= 1.0)
    per_chain_corr = [float(np.corrcoef(X[:, 0], X[:, 2])[0, 1]) for X in Xc]
    gate = j["strong"].get("gate", {})
    bat = j["strong"].get("battery", {})
    return dict(mock=int(r["mock"]), L_stored=int(z["L"]), step=step, stats=st, first_chain_only=first, finite_L=bands, corr_inside_band=corr_inside,
                delta_within_1=delta_ok, agree=bool(corr_inside and delta_ok), per_chain_corr_ns_dtau0=per_chain_corr,
                sampler_ok=bool(gate.get("passed_sampler_criteria", False)), pilot_gate_passed=gate.get("pilot_gate_passed"),
                battery=dict(rhat_max=bat.get("rhat_max"), ess_bulk_min=bat.get("ess_bulk_min"), ess_tail_min=bat.get("ess_tail_min"), ebfmi_min=bat.get("ebfmi_min"),
                             treedepth_sat_frac=bat.get("treedepth_sat_frac"), n_divergent=bat.get("n_divergent")),
                replica=j.get("replica", {}).get("compare"), identity=ident, truth_c4=t.tolist(),
                inputs_sha256=dict(stored_pkl=want_s, stage2_pkl=want_o, stage2_json=_sha256(stage2_json)),
                whitened_truth_displacement_stored=(_sym_inv_sqrt(np.cov(Xs.T, ddof=1)) @ (Xs.mean(axis=0) - t)).tolist(),
                whitened_truth_displacement_strong=(_sym_inv_sqrt(np.cov(Xn.T, ddof=1)) @ (Xn.mean(axis=0) - t)).tolist())


def decide(per_mock):
    by = {m["mock"]: m for m in per_mock if m.get("sampler_ok")}
    excluded = [m["mock"] for m in per_mock if not m.get("sampler_ok")]
    tail = [by[m] for m in TAIL if m in by]; ctrl = [by[m] for m in CONTROL if m in by]
    n_tail_corr_out = sum(not m["corr_inside_band"] for m in tail); n_ctrl_corr_out = sum(not m["corr_inside_band"] for m in ctrl)
    n_tail_delta_out = sum(not m["delta_within_1"] for m in tail); n_ctrl_delta_out = sum(not m["delta_within_1"] for m in ctrl)
    n_tail_agree = sum(m["agree"] for m in tail); n_ctrl_agree = sum(m["agree"] for m in ctrl)
    both = tail + ctrl
    med_corr_strong = float(np.median([m["stats"]["corr_ns_dtau0_strong"] for m in both])) if both else float("nan")
    complete = (len(tail) == 4 and len(ctrl) == 4)
    if not complete:
        label = "INCOMPLETE"
    elif n_tail_corr_out >= 3 or (n_tail_delta_out >= 3 and n_ctrl_delta_out <= 1):
        label = "M-A (the stored draws misrepresent the target)"
    elif n_tail_agree >= 3 and n_ctrl_agree >= 3 and med_corr_strong < -0.1:
        label = "M-B-orientation (exact-posterior feature)"
    else:
        label = "AMBIGUOUS"
    return dict(label=label, n_tail_corr_outside=n_tail_corr_out, n_control_corr_outside=n_ctrl_corr_out, n_tail_delta_over_1=n_tail_delta_out,
                n_control_delta_over_1=n_ctrl_delta_out, n_tail_agree=n_tail_agree, n_control_agree=n_ctrl_agree,
                median_corr_ns_dtau0_strong=(None if np.isnan(med_corr_strong) else med_corr_strong), complete=complete, excluded_failed_sampler_gate=excluded,
                tail=list(TAIL), control=list(CONTROL), delta_definition="C4 norm")


def _sanitize(o):
    """Non-finite floats -> None so the strict JSON dump cannot fail after the decision is formed."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _git_head():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stored-dir", required=True); ap.add_argument("--stage2-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mocks", type=int, nargs="*", default=list(TAIL) + list(CONTROL))
    a = ap.parse_args(argv)
    for ext in (".json", ".md", ".json.tmp", ".md.tmp"):
        if os.path.exists(a.out + ext):
            raise SystemExit(f"REFUSE: output exists: {a.out}{ext}")
    per = []
    for m in a.mocks:
        sp = os.path.join(a.stored_dir, f"mock_{m:04d}.pkl"); p2 = os.path.join(a.stage2_dir, f"stage2_mock_{m:04d}.pkl"); j2 = os.path.join(a.stage2_dir, f"stage2_mock_{m:04d}.json")
        if not (os.path.exists(p2) and os.path.exists(j2)):
            continue
        per.append(analyze_mock(sp, p2, j2))
    dec = decide(per)
    out = _sanitize(dict(schema="desi_stage2_readout.v2", prereg="2026-09-24-DESI-STAGE2-PREREGISTRATION-v1.2", seed=SEED, block=BLOCK, B=B_FINITE_L, per_mock=per, decision=dec, coords=list(C4),
                         provenance=dict(script_sha256=_sha256(os.path.abspath(__file__)), code_head=_git_head(), stored_dir=os.path.abspath(a.stored_dir), stage2_dir=os.path.abspath(a.stage2_dir),
                                         inputs_sha256={str(m["mock"]): m["inputs_sha256"] for m in per}, mocks_requested=list(a.mocks))))
    js = json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n"        # built fully BEFORE any file is opened
    L = [f"# DESI Stage 2 readout ({len(per)} mocks analysed; {len(dec['excluded_failed_sampler_gate'])} excluded by the sampler gate): **{dec['label']}**", "",
         "| mock | set | L | step | sampler ok | corr(ns,dtau0) stored | strong | band95 | inside | delta_C4 norm | q95(finite-L, descriptive) | sd ratios strong/stored | pilot gate |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in out["per_mock"]:
        s_ = m["stats"]; b = m["finite_L"]; tag = "TAIL" if m["mock"] in TAIL else ("CONTROL" if m["mock"] in CONTROL else "-")
        fmt = lambda x, f: ("nan" if x is None else format(x, f))
        L.append(f"| {m['mock']} | {tag} | {m['L_stored']} | {m['step']} | {m['sampler_ok']} | {fmt(s_['corr_ns_dtau0_stored'], '+.3f')} | {fmt(s_['corr_ns_dtau0_strong'], '+.3f')} | [{fmt(b['corr_ns_dtau0_band95'][0], '+.3f')}, {fmt(b['corr_ns_dtau0_band95'][1], '+.3f')}] | {m['corr_inside_band']} | {fmt(s_['delta_c4_norm'], '.2f')} | {fmt(b['delta_norm_q95'], '.2f')} | {[None if x is None else round(x, 2) for x in s_['sd_ratio']]} | {m['pilot_gate_passed']} |")
    L += ["", f"decision counts: {json.dumps({k: v for k, v in dec.items() if k not in ('tail', 'control')})}"]
    md = "\n".join(L) + "\n"
    with open(a.out + ".json.tmp", "w") as f:
        f.write(js)
    os.replace(a.out + ".json.tmp", a.out + ".json")
    with open(a.out + ".md.tmp", "w") as f:
        f.write(md)
    os.replace(a.out + ".md.tmp", a.out + ".md")
    print(f"wrote {a.out}.json/.md: {dec['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
