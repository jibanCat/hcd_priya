#!/usr/bin/env python3
"""Phase-4b: IGM-stress fiducials -> cosmology bias + HCD-sensitivity.

For each fiducial, pool MCMC samples across its chains (the joint posterior) and report,
matching the run_stepA battery convention bias_z = (truth - pooled_mean)/pooled_sd:

  bias_Ap, bias_ns                          cosmology recovery (sigma)
  corr(Ap, alpha_lls), corr(ns, alpha_subdla), corr(Ap, alpha_subdla)
  alpha_lls/subdla/dla recovery (bias_z)     HCD-nuisance recovery (sigma)

The question: does stressing the IGM (thermal/reion/UVB extremes) make the cosmology
more HCD-prone -- larger |corr(Ap, alpha_HCD)| or larger cosmology bias -- vs the
no-IGM-stress baseline D_f6?
"""
import glob, os, json
import numpy as np

CK = "checkpoints/stepA"
COLS = None  # filled from names

# fiducial -> human label
FIDS = {
    "D_f6":          "baseline (no IGM stress)",
    "IGM_herei_hi":  "HeII-reion start HIGH",
    "IGM_heref_lo":  "HeII-reion end LOW",
    "IGM_heref_hi":  "HeII-reion end HIGH",
    "IGM_alphaq_lo": "QSO spec-slope LOW",
    "IGM_alphaq_hi": "QSO spec-slope HIGH",
    "IGM_bhfb_lo":   "BH-feedback LOW",
}

def chains_for(fid):
    # exact-prefix match: <fid>_c<digit>.npz, but not <fid>_extra_c#
    out = []
    for p in sorted(glob.glob(f"{CK}/{fid}_c*.npz")):
        base = os.path.basename(p)[:-4]
        tail = base[len(fid):]
        if tail.startswith("_c") and tail[2:].isdigit():
            out.append(p)
    return out

def pooled(fid):
    paths = chains_for(fid)
    if not paths:
        return None
    packs, names, truth = [], None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        packs.append(np.asarray(z["packed"]))
        names = list(z["names"]) if names is None else names
        truth = np.asarray(z["truth_vec"]) if truth is None else truth
    nmin = min(p.shape[0] for p in packs)
    P = np.concatenate([p[:nmin] for p in packs], axis=0)  # (C*nmin, 25)
    return P, names, truth, len(paths)

def bias_z(P, truth, names, name):
    j = names.index(name)
    sd = P[:, j].std()
    return float((truth[j] - P[:, j].mean()) / sd) if sd > 0 else float("nan")

def corr(P, names, a, b):
    ia, ib = names.index(a), names.index(b)
    return float(np.corrcoef(P[:, ia], P[:, ib])[0, 1])

rows = []
for fid, label in FIDS.items():
    res = pooled(fid)
    if res is None:
        rows.append(dict(fid=fid, label=label, status="MISSING"))
        continue
    P, names, truth, nc = res
    row = dict(
        fid=fid, label=label, status="ok", n_chains=nc, n_pooled=P.shape[0],
        bias_Ap=round(bias_z(P, truth, names, "Ap"), 3),
        bias_ns=round(bias_z(P, truth, names, "ns"), 3),
        corr_Ap_lls=round(corr(P, names, "Ap", "alpha_lls"), 3),
        corr_ns_subdla=round(corr(P, names, "ns", "alpha_subdla"), 3),
        corr_Ap_subdla=round(corr(P, names, "Ap", "alpha_subdla"), 3),
        rec_lls=round(bias_z(P, truth, names, "alpha_lls"), 3),
        rec_subdla=round(bias_z(P, truth, names, "alpha_subdla"), 3),
        rec_dla=round(bias_z(P, truth, names, "alpha_dla"), 3),
    )
    rows.append(row)

# pretty table
hdr = ["fiducial", "label", "nc", "bias_Ap", "bias_ns",
       "c(Ap,LLS)", "c(ns,subD)", "c(Ap,subD)", "rec_LLS", "rec_subD", "rec_DLA"]
print(" | ".join(f"{h:>11}" if i else f"{h:<14}" for i, h in enumerate(hdr)))
print("-" * 130)
for r in rows:
    if r["status"] != "ok":
        print(f"{r['fid']:<14} | {r['label']:<28} MISSING")
        continue
    print(f"{r['fid']:<14} | {r['label']:<22} | {r['n_chains']:>2} | "
          f"{r['bias_Ap']:>+7.3f} | {r['bias_ns']:>+7.3f} | "
          f"{r['corr_Ap_lls']:>+9.3f} | {r['corr_ns_subdla']:>+10.3f} | {r['corr_Ap_subdla']:>+10.3f} | "
          f"{r['rec_lls']:>+7.3f} | {r['rec_subdla']:>+8.3f} | {r['rec_dla']:>+7.3f}")

with open(f"{CK}/igm_hcd_sensitivity.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nwrote", f"{CK}/igm_hcd_sensitivity.json")
