#!/usr/bin/env python3
"""subDLA/LLS dN/dX consistency vs the held-out-sim TRUTH (PI directive 2026-06-19).

Question: the 0.20 arm shows large sigma-pulls on alpha_subdla (-1.7) / alpha_lls (-1.5).
Is that a PHYSICAL inconsistency (recovered dN/dX far from the sim's true incidence) or
just a tight-posterior artifact (small absolute offset blown up by a narrow post_sd)?
If the recovered incidence is ~consistent with the sim truth AND cosmology (ns,Ap) is
unbiased, the pull is not a worry (PI's logic).

Reads both arms; reports per-class truth vs recovered (absolute + fractional + sigma-pull),
pull coherence, the cosmology pulls, and a partial PAIRED ns decomposition on shared mocks.
Usage: PYTHONPATH=/home/mfho/hcd_priya python3 scripts/diag_heldout_dndx_consistency.py
"""
import pickle, glob, os, numpy as np

ARMS = {
    "0.40 pilot": "/scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout",
    "0.20 arm":   "/scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout_amp020",
}
HCD = ["alpha_subdla", "alpha_lls", "alpha_dla"]
COSMO = ["ns", "Ap"]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "mock_*.pkl"))):
        try:
            x = pickle.load(open(f, "rb"))
        except Exception:
            continue
        out[os.path.basename(f)] = x
    return out


def stats(x, j):
    dr = np.asarray(x["draws"])[:, j]
    t = float(x["truth_vec"][j])
    m, s = float(dr.mean()), float(dr.std())
    return t, m, s, (m - t) / s


arms = {k: load(v) for k, v in ARMS.items()}
names = None
for a in arms.values():
    for x in a.values():
        names = list(x["names"]); break
    if names: break
JI = {n: names.index(n) for n in HCD + COSMO}

for arm, mocks in arms.items():
    if not mocks:
        print(f"\n===== {arm}: (nothing landed) ====="); continue
    print(f"\n===== {arm}  (N={len(mocks)} mocks) =====")
    for cls in HCD:
        j = JI[cls]
        rows = [(mk,) + stats(x, j) for mk, x in mocks.items()]
        truth = np.array([r[1] for r in rows])
        rec   = np.array([r[2] for r in rows])
        sd    = np.array([r[3] for r in rows])
        pull  = np.array([r[4] for r in rows])
        ratio = rec / truth
        coh = np.mean(np.sign(pull) == np.sign(pull).mean().__class__(np.sign(np.median(pull))))
        same = int(np.sum(np.sign(pull) == np.sign(np.median(pull))))
        print(f"  {cls:13s} truth_var: {truth.mean():+.4f}±{truth.std():.4f}  "
              f"recovered: {rec.mean():+.4f}  ratio rec/truth: {ratio.mean():.3f}±{ratio.std():.3f}")
        print(f"  {'':13s} sigma-pull: mean={pull.mean():+.3f} std={pull.std():.3f}  "
              f"coherent: {same}/{len(rows)} same sign  | abs dN/dX-offset: {(rec-truth).mean():+.4f} ({100*(ratio.mean()-1):+.1f}%)")
    for cls in COSMO:
        j = JI[cls]
        pull = np.array([stats(x, j)[3] for x in mocks.values()])
        print(f"  [cosmo] {cls:8s} pull mean={pull.mean():+.3f} std={pull.std():.3f}")

# ---- partial PAIRED ns decomposition on SHARED mocks ----
sh = sorted(set(arms["0.40 pilot"]) & set(arms["0.20 arm"]))
print(f"\n===== PAIRED ns decomposition on {len(sh)} shared mocks: {[s.replace('mock_','').replace('.pkl','') for s in sh]} =====")
if sh:
    jns = JI["ns"]
    for arm in ("0.40 pilot", "0.20 arm"):
        post_mean = np.array([np.asarray(arms[arm][mk]["draws"])[:, jns].mean() for mk in sh])
        post_sd   = np.array([np.asarray(arms[arm][mk]["draws"])[:, jns].std()  for mk in sh])
        truth     = np.array([float(arms[arm][mk]["truth_vec"][jns]) for mk in sh])
        bias_scatter = (post_mean - truth).std()
        mean_post_sd = post_sd.mean()
        pull_std = bias_scatter / mean_post_sd
        pull_mean = ((post_mean - truth) / post_sd).mean()
        print(f"  {arm}:  pull_mean={pull_mean:+.3f}  bias_scatter={bias_scatter:.4f}  "
              f"mean_post_sd={mean_post_sd:.4f}  ->  pull_std={pull_std:.3f}")
    print("  (PASS signature: pull_std drops 0.40->0.20, i.e. bias_scatter shrinks >= as fast as post_sd. N small -> NOISY/directional only.)")
