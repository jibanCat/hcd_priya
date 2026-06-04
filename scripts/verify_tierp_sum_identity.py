"""VERIFY (on this 96 GB node, NOT the cluster allocation) that the proposed
memory optimization is bit-identical to the original PRIYA path:

    Tier P via fake_spectra flux_power   (ORIGINAL gold reference, kept)
        vs
    Tier P = sum_c (n_c/N) * filtered-Tier-C piece_c   (proposed cheap path)

Covers the edge cases a reviewer flagged: a different nbins (HR, n_k=525 path),
the idx19 EXTRAPOLATION alpha (above PRIYA's top), and pairs with an EMPTY
Tier-C class (confirm the nspec==0 branch leaks no NaN into the summed total).
If max|ratio-1| ~ 1e-13 across all, the optimization is exact regrouping.
Runs in emu-3.9 env.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import build_emulator_cache_tau0 as bt0
from hcd_analysis.priya_p1d import compute_tier_p_p1d, compute_tier_c_p1d
from hcd_analysis.io import read_header
from hcd_analysis.catalog import AbsorberCatalog

HCD = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
LF = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
HR = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")
IDX19 = 0.6555563811131029 + (1.2956838823026422 - 0.6555563811131029) / 9 / 2 * 19  # 1.3312

lf = bt0.discover_tau0_pairs(HCD, LF, fidelity="lf")
hr = bt0.discover_tau0_pairs(HCD, HR, fidelity="hr")

def nearest(pairs, zt, pfx=None):
    c = [p for p in pairs if (pfx is None or p[0].startswith(pfx))]
    c.sort(key=lambda p: abs(float(json.load(open(p[2] / "meta.json"))["z"]) - zt))
    return c[0]

# (label, pair, alphas) — different nbins (LF z3, LF low-z, HR) + idx19 extrapolation
CASES = [
    ("LF z3",   nearest(lf, 3.0, "ns0.803"), [1.00, IDX19]),
    ("LF low-z", nearest(lf, 2.2),            [0.70, 1.00]),
    ("HR",      nearest(hr, 3.0),             [1.00, IDX19]),
]

worst = 0.0
for label, (sim, snap, sd, raw), alphas in CASES:
    meta = bt0.bec.read_meta(sd)
    z = bt0._snap_z_to_priya_grid(float(meta["z"]))
    vmax = int(read_header(raw).nbins) * float(meta["dv_kms"])
    cat = AbsorberCatalog.load_npz(sd / "catalog.npz")
    tau0 = bt0._read_tau(raw)
    print(f"\n=== {label}: {sim[:20]} snap{snap} z={z} nsk={tau0.shape[0]} nbins={tau0.shape[1]} ===")
    for alpha in alphas:
        tf = tau0.copy()
        kf, P_orig, target_F, scale = compute_tier_p_p1d(tf, vmax, alpha_slope=alpha, z=z)
        _, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
            tf, vmax, alpha_slope=alpha, z=z, catalog=cat,
            external_scale=scale, external_target_F=target_F)
        N = int(n_by_bin.sum())
        P_sum = (n_by_bin[:, None] / N * P_by_bin).sum(axis=0)
        empties = [i for i in range(len(n_by_bin)) if n_by_bin[i] == 0]
        assert np.all(np.isfinite(P_sum[np.isfinite(P_orig)])), \
            f"NaN leaked into summed Tier P ({label} a={alpha})"
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(P_orig / P_sum - 1.0)
        r = r[np.isfinite(r)]
        worst = max(worst, r.max())
        tag = " [idx19 EXTRAP]" if abs(alpha - IDX19) < 1e-6 else ""
        print(f"  a={alpha:.4f}{tag}: max|r-1|={r.max():.2e} median={np.median(r):.2e}"
              f"  empty classes={empties} (sum finite OK)")
        del tf

print(f"\nVERDICT: worst max|r-1| across all cases/alpha = {worst:.2e}")
print("~1e-13 or smaller => Tier P = sum(filtered pieces) is bit-identical to "
      "flux_power across nbins, the extrapolation alpha, and empty-class pairs.")
