"""Hierarchical-HCD (Option B) validation gate — the closure the 4-lens design review demanded.

Reads the HB_* joint-DESI+KS closure chains (HB_f6/HB_f4 × {HB0=legacy OFF, HB1=hierarchical ON,
HBp/HBm = ON with the A_HCD prior center ±1σ}) and certifies (design doc §"Validation gate" +
impl-review must-fix #3):

  (1) REPARAM IS COSMOLOGY-NEUTRAL: n_s/A_p bias_z(ON=HB1) ≈ bias_z(OFF=HB0) within MC noise — the
      reparametrization must not move the headline.
  (2) subDLA MEAN BIAS COLLAPSES: |bias_z(α_subDLA)| on HB1 (ON) < on HB0 (OFF) — Option B removes the
      subDLA↔DLA exchange freedom that drove the all-8 subDLA-down bias.
  (3) A_HCD-CENTER COSMOLOGY SAFETY: under a ±1σ A_HCD prior-center mis-specification (HBp/HBm vs HB1),
      n_s/A_p MEANS stay <0.3σ — the coupling relocated onto A_HCD (Option B's residual risk) is bounded.
  (4) corr(A_HCD, {A_p, n_s, τ₀}) from the HB1 chains — does slaving the HCD sector to one broadband
      amplitude make A_HCD degenerate with cosmology? (the design review's sharpest unanalyzed gap).
  (5) 0 divergences across all HB chains.

ENV: PYTHONNOUSERSITE=1 /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/analyze_hb_hierarchical.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "/home/mfho/hcd_priya/checkpoints/stepA"
OUTDIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"
os.makedirs(OUTDIR, exist_ok=True)
LIMS = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9)}
FIDS = ["HB_f6", "HB_f4"]
PASS = 0.30


def to_phys(x, nm):
    if nm not in LIMS:
        return x
    lo, hi = LIMS[nm]; v = lo + x * (hi - lo)
    return v * 1e9 if nm == "Ap" else v


def load(base):
    ps = sorted(p for p in glob.glob(f"{CKPT}/{base}_c*.npz")
                if os.path.basename(p).split(base)[1].startswith("_c"))
    if not ps:
        return None
    chains, names, truth, ndiv = [], None, None, 0
    for p in ps:
        z = np.load(p, allow_pickle=True)
        if names is None:
            names = [str(x) for x in z["names"]]; truth = np.asarray(z["truth_vec"])
        chains.append(np.asarray(z["packed"]))
        ndiv += int(z["divergences"]) if "divergences" in z else 0
    return dict(P=np.concatenate(chains, 0), names=names, truth=truth, nchain=len(ps), ndiv=ndiv)


def col(arm, nm):
    i = arm["names"].index(nm)
    return to_phys(arm["P"][:, i], nm)


def bias_z(arm, nm):
    i = arm["names"].index(nm)
    x = to_phys(arm["P"][:, i], nm); t = to_phys(arm["truth"][i], nm)
    return (x.mean() - t) / x.std()


def tau0_proxy(arm):
    idx = [i for i, n in enumerate(arm["names"]) if n.startswith("tau0_z")]
    return arm["P"][:, idx].mean(axis=1)   # z-mean τ₀ per draw


print(f"{'fid':7s} {'check':30s} {'value':>12s}  verdict")
rows = []
for fid in FIDS:
    off = load(f"{fid}_HB0"); on = load(f"{fid}_HB1")
    hp = load(f"{fid}_HBp"); hm = load(f"{fid}_HBm")
    if off is None or on is None:
        print(f"{fid}: missing HB0/HB1"); continue
    rec = {"fid": fid, "ndiv": (off["ndiv"], on["ndiv"],
                                hp["ndiv"] if hp else None, hm["ndiv"] if hm else None)}
    # (1) reparam cosmology-neutral
    for nm in ("ns", "Ap"):
        bz_off, bz_on = bias_z(off, nm), bias_z(on, nm)
        d = bz_on - bz_off
        rec[f"neutral_{nm}"] = (bz_off, bz_on, d)
        v = "PASS" if abs(d) < PASS else "WARN"
        print(f"{fid:7s} (1) {nm} bias_z ON−OFF{'':6s} {d:>+12.2f}σ  {v}  (off {bz_off:+.2f} on {bz_on:+.2f})")
    # (2) subDLA bias collapse
    bso, bsn = bias_z(off, "alpha_subdla"), bias_z(on, "alpha_subdla")
    rec["subdla_collapse"] = (bso, bsn)
    v = "PASS" if abs(bsn) < abs(bso) else "WARN"
    print(f"{fid:7s} (2) α_subDLA bias_z OFF→ON{'':2s} {bso:>+6.2f}→{bsn:+.2f}σ  {v}")
    # (3) A_HCD-center cosmology safety
    if hp is not None and hm is not None:
        for nm in ("ns", "Ap"):
            m0 = col(on, nm).mean(); s0 = col(on, nm).std()
            dp = (col(hp, nm).mean() - m0) / s0; dm = (col(hm, nm).mean() - m0) / s0
            rec[f"ahcd_{nm}"] = (dp, dm)
            v = "PASS" if (abs(dp) < PASS and abs(dm) < PASS) else "FAIL"
            print(f"{fid:7s} (3) {nm} Δ/σ @A_HCD±1σ{'':3s} +1σ {dp:>+5.2f} / −1σ {dm:+.2f}  {v}")
        # sanity: the A_HCD knob moved the A_hcd posterior
        am0 = col(on, "A_hcd").mean(); asd = col(on, "A_hcd").std()
        admp = (col(hp, "A_hcd").mean() - am0) / asd
        print(f"{fid:7s}     (A_hcd knob moved {admp:+.2f}σ — {'ACTIVE' if abs(admp) > 0.3 else 'INERT?'})")
    # (4) corr(A_HCD, cosmology)
    A = col(on, "A_hcd"); t0 = tau0_proxy(on)
    cns = np.corrcoef(A, col(on, "ns"))[0, 1]
    cap = np.corrcoef(A, col(on, "Ap"))[0, 1]
    ct0 = np.corrcoef(A, t0)[0, 1]
    rec["corr"] = (cns, cap, ct0)
    print(f"{fid:7s} (4) corr(A_HCD, ns/Ap/τ₀){'':1s} {cns:+.2f} / {cap:+.2f} / {ct0:+.2f}")
    print(f"{fid:7s} (5) divergences (HB0/1/p/m): {rec['ndiv']}")
    rows.append(rec)

# overall verdict
print("\n=== HIERARCHICAL-HCD (OPTION B) VALIDATION VERDICT ===")
if not rows:
    print("  ⏳ INCOMPLETE — no HB chains yet."); raise SystemExit
worst_neutral = max(abs(r[f"neutral_{nm}"][2]) for r in rows for nm in ("ns", "Ap"))
have_ahcd = all(("ahcd_ns" in r) for r in rows)
worst_ahcd = max((abs(r[f"ahcd_{nm}"][s]) for r in rows for nm in ("ns", "Ap") for s in (0, 1)),
                 default=np.nan) if have_ahcd else np.nan
collapses = all(abs(r["subdla_collapse"][1]) < abs(r["subdla_collapse"][0]) for r in rows)
maxdiv = max(max(x for x in r["ndiv"] if x is not None) for r in rows)
print(f"  (1) reparam cosmology-neutral: worst |Δbias_z| = {worst_neutral:.2f}σ "
      f"({'PASS' if worst_neutral < PASS else 'WARN'})")
print(f"  (2) subDLA bias collapses ON<OFF in all mocks: {collapses}")
print(f"  (3) A_HCD-center ±1σ → cosmology: worst |Δ/σ| = {worst_ahcd:.2f}σ "
      f"({'PASS' if (have_ahcd and worst_ahcd < PASS) else ('INCOMPLETE' if not have_ahcd else 'FAIL')})")
print(f"  (5) max divergences across HB chains: {maxdiv}")
net_safe = (have_ahcd and worst_ahcd < PASS)
print("  ✅ Option B CERTIFIED net cosmology-safe (A_HCD-center coupling bounded, 0-div, reparam neutral)"
      if (net_safe and worst_neutral < PASS and maxdiv == 0)
      else "  ⚠️  see per-check flags — the gate is the arbiter (the design argument alone does not certify)")

# figure: A_HCD-center cosmology safety (the headline check)
if have_ahcd:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, nm, lab in zip(axes, ("ns", "Ap"), (r"$n_s$", r"$A_p\,[10^{-9}]$")):
        x = np.arange(len(rows))
        for j, (tag, tl, mk) in enumerate([("HB1", "A_HCD 0σ", "o"), ("HBp", "+1σ", "s"), ("HBm", "−1σ", "D")]):
            means = [col(load(f"{r['fid']}_{tag}"), nm).mean() if load(f"{r['fid']}_{tag}") else np.nan for r in rows]
            sds = [col(load(f"{r['fid']}_{tag}"), nm).std() if load(f"{r['fid']}_{tag}") else np.nan for r in rows]
            ax.errorbar(x + (j - 1) * 0.12, means, yerr=sds, fmt=mk, ms=6, capsize=3, label=tl)
        tv = [to_phys(load(f"{r['fid']}_HB1")["truth"][load(f"{r['fid']}_HB1")["names"].index(nm)], nm) for r in rows]
        ax.scatter(x, tv, marker="_", s=400, color="k", zorder=6, label="truth")
        ax.set_xticks(x); ax.set_xticklabels([r["fid"] for r in rows]); ax.set_ylabel(lab)
        ax.set_title(f"{lab} vs A_HCD prior-center shift"); ax.grid(alpha=0.25)
        if nm == "ns":
            ax.legend(fontsize=8)
    fig.suptitle("Hierarchical-HCD validation: n_s/A_p under ±1σ A_HCD prior-center mis-specification (joint DESI+KS)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{OUTDIR}/hb_hierarchical_safety.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"\nwrote {out}")
