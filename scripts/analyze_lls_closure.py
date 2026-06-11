"""Per-survey LLS-pin closure: cosmology recovery + alpha_LLS->dN/dX faithfulness.

For each closure fiducial (D_lls, K_lls, D_lls_m, ...) pool its chains and report bias_z =
(truth - pooled_mean)/pooled_sd for Ap/ns/alpha_lls/alpha_subdla (battery convention), plus the
alpha_LLS posterior (mean +/- sd) vs its mock truth -- alpha_LLS IS the effective LLS incidence
(dN/dX-proportional), so "recovered alpha_LLS == truth" == the LLS dN/dX recovered faithfully.
"""
import glob, os, json, numpy as np

CK = "checkpoints/stepA"
FIDS = ["D_lls_m30", "D_lls_m", "D_lls", "K_lls"]
LABEL = {"D_lls_m30": "DESI σ0.30, matched (×1.06)", "D_lls_m": "DESI σ0.15, matched (×1.06)",
         "D_lls": "DESI σ0.15, sim mock (×1.0)", "K_lls": "KS σ0.40, ×2.65 (matched)"}

def pooled(fid):
    ps = sorted(glob.glob(f"{CK}/{fid}_c*.npz"))
    ps = [p for p in ps if os.path.basename(p)[len(fid):].startswith("_c")]
    if not ps: return None
    packs, names, truth = [], None, None
    for p in ps:
        z = np.load(p, allow_pickle=True)
        packs.append(np.asarray(z["packed"])); names = list(z["names"]) if names is None else names
        truth = np.asarray(z["truth_vec"]) if truth is None else truth
    n = min(p.shape[0] for p in packs)
    P = np.concatenate([p[:n] for p in packs], 0)
    return P, names, truth, len(ps)

def bz(P, truth, names, nm):
    j = names.index(nm); sd = P[:, j].std()
    return float((truth[j] - P[:, j].mean()) / sd) if sd > 0 else float("nan")

rows = []
for fid in FIDS:
    r = pooled(fid)
    if r is None:
        rows.append(dict(fid=fid, status="MISSING")); continue
    P, names, truth, nc = r
    il = names.index("alpha_lls")
    rows.append(dict(fid=fid, status="ok", nc=nc,
        bias_Ap=round(bz(P, truth, names, "Ap"), 3), bias_ns=round(bz(P, truth, names, "ns"), 3),
        bias_lls=round(bz(P, truth, names, "alpha_lls"), 3),
        bias_sub=round(bz(P, truth, names, "alpha_subdla"), 3),
        lls_truth=round(float(truth[il]), 4), lls_post=round(float(P[:, il].mean()), 4),
        lls_post_sd=round(float(P[:, il].std()), 4)))
with open(f"{CK}/lls_closure_summary.json", "w") as f: json.dump(rows, f, indent=2)

print(f"{'fiducial':<8} {'nc':>2} | {'biasAp':>7} {'biasns':>7} | {'biasLLS':>7} {'biasSub':>7} | "
      f"{'LLStruth':>8} {'LLSpost':>8} {'±sd':>6}")
print("-"*84)
for r in rows:
    if r["status"] != "ok": print(f"{r['fid']:<8}  MISSING"); continue
    print(f"{r['fid']:<8} {r['nc']:>2} | {r['bias_Ap']:>+7.3f} {r['bias_ns']:>+7.3f} | "
          f"{r['bias_lls']:>+7.3f} {r['bias_sub']:>+7.3f} | {r['lls_truth']:>8.4f} {r['lls_post']:>8.4f} {r['lls_post_sd']:>6.4f}")
print("\nwrote", f"{CK}/lls_closure_summary.json")
