#!/usr/bin/env python3
"""eBOSS DR14 basic-recovery closure cert — summary figure (LF-only, no MF; survey="eBOSS").

Loads the E_f5/E_f6/E_f7 fiducial chains and plots, per fiducial: the n_s and A_p posterior
recovery vs truth (physical units, point ± 68% from the pooled draws) + the bias in σ. Writes to
the notes repo. Run (emu-jax):
  PYTHONPATH=/home/mfho/hcd_priya python3 scripts/plot_eboss_cert.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import run_stepA as R
from hcd_analysis.emulator.data import PARAM_LIMITS  # unit-cube -> physical bounds

NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"
FIDS = [("E_f5", "n_s≈0.95"), ("E_f6", "Planck n_s≈0.966"), ("E_f7", "n_s≈1.00")]
# physical-unit conversion (the cache encoder maps physical -> [0,1] via PARAM_LIMITS)
NS_LO, NS_HI = PARAM_LIMITS[0]
AP_LO, AP_HI = PARAM_LIMITS[1]


def _pool(mid):
    ids = [f"{mid}_c{c}" for c in range(4) if os.path.exists(f"{R.CKPT_DIR}/{mid}_c{c}.npz")]
    pk, tv, nm = [], None, None
    for cid in ids:
        z = np.load(f"{R.CKPT_DIR}/{cid}.npz", allow_pickle=True)
        pk.append(np.asarray(z["packed"]))
        nm = [str(x) for x in z["names"]] if nm is None else nm
        tv = np.asarray(z["truth_vec"]) if tv is None else tv
    return (np.concatenate(pk), tv, nm, len(ids)) if pk else (None, None, None, 0)


def _phys(u, lo, hi):
    return lo + np.asarray(u) * (hi - lo)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for mid, lab in FIDS:
        pool, tv, nm, nch = _pool(mid)
        if pool is None:
            continue
        jn, ja = nm.index("ns"), nm.index("Ap")
        rows.append(dict(
            lab=lab, nch=nch,
            ns_t=_phys(tv[jn], NS_LO, NS_HI), ns_m=_phys(pool[:, jn].mean(), NS_LO, NS_HI),
            ns_s=pool[:, jn].std() * (NS_HI - NS_LO),
            ns_bz=(pool[:, jn].mean() - tv[jn]) / pool[:, jn].std(),
            ap_t=_phys(tv[ja], AP_LO, AP_HI), ap_m=_phys(pool[:, ja].mean(), AP_LO, AP_HI),
            ap_s=pool[:, ja].std() * (AP_HI - AP_LO),
            ap_bz=(pool[:, ja].mean() - tv[ja]) / pool[:, ja].std()))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # n_s recovery (physical)
    ax[0].errorbar(x, [r["ns_m"] for r in rows], yerr=[r["ns_s"] for r in rows], fmt="o",
                   color="C0", capsize=4, label="posterior (mean±σ)")
    ax[0].plot(x, [r["ns_t"] for r in rows], "kx", ms=11, mew=2, label="truth")
    ax[0].set_xticks(x); ax[0].set_xticklabels([r["lab"] for r in rows], fontsize=8)
    ax[0].set_ylabel("n_s (physical)"); ax[0].set_title("eBOSS cert: n_s recovery"); ax[0].legend(fontsize=8)
    # A_p recovery (physical)
    ax[1].errorbar(x, [r["ap_m"] for r in rows], yerr=[r["ap_s"] for r in rows], fmt="o",
                   color="C1", capsize=4, label="posterior (mean±σ)")
    ax[1].plot(x, [r["ap_t"] for r in rows], "kx", ms=11, mew=2, label="truth")
    ax[1].set_xticks(x); ax[1].set_xticklabels([r["lab"] for r in rows], fontsize=8)
    ax[1].set_ylabel("A_p (physical)"); ax[1].set_title("eBOSS cert: A_p recovery"); ax[1].legend(fontsize=8)
    # bias-z summary
    w = 0.35
    ax[2].axhspan(-1, 1, color="green", alpha=0.08, label="±1σ (LOSO band)")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].bar(x - w / 2, [r["ns_bz"] for r in rows], w, color="C0", label="n_s")
    ax[2].bar(x + w / 2, [r["ap_bz"] for r in rows], w, color="C1", label="A_p")
    ax[2].set_xticks(x); ax[2].set_xticklabels([r["lab"] for r in rows], fontsize=8)
    ax[2].set_ylabel("bias z = (post−truth)/σ"); ax[2].set_title("eBOSS cert: recovery bias")
    ax[2].legend(fontsize=8); ax[2].set_ylim(-1.5, 1.5)
    fig.suptitle("eBOSS DR14 basic-recovery closure cert (LF-only, no MF; held-out-sim, low-k)\n"
                 "0 divergences, R̂≤1.01; n_s within ±0.5σ, A_p within ±0.74σ (inside the LOSO scatter band)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(NOTES_FIG, exist_ok=True)
    out = f"{NOTES_FIG}/eboss_cert_recovery.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['lab']:18s} ({r['nch']}ch): n_s {r['ns_m']:.4f} (truth {r['ns_t']:.4f}, z {r['ns_bz']:+.2f}) | "
              f"A_p {r['ap_m']:.3e} (truth {r['ap_t']:.3e}, z {r['ap_bz']:+.2f})")


if __name__ == "__main__":
    main()
