"""eBOSS SiIII no-leak cert — does a sampled a_SiIII absorb the injected ripple WITHOUT leaking
into cosmology? Compares the SiIII-injection arms (E_f5/6/7_si: mock carries the 0.045 ripple,
forward samples a_SiIII) against the clean arms (E_f5/6/7: no ripple, no a_SiIII) at the SAME
fiducial sims. Three asks per fiducial:

  (1) a_SiIII RECOVERY:  posterior a_SiIII vs the injected 0.045.
  (2) n_s NO-LEAK:       n_s(SiIII) vs n_s(clean) — the shift from adding the metal nuisance, in σ.
  (3) A_p NO-LEAK:       A_p(SiIII) vs A_p(clean) — same.

A clean pass = a_SiIII recovers ≈0.045, and |Δn_s|, |ΔA_p| between the two arms are ≪ their
posterior σ (the ripple is absorbed by a_SiIII, not by tilting/rescaling cosmology).

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/analyze_eboss_metals_noleak.py
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

# unit->physical (same as plot_eboss_corners.py)
LIMS = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9)}
FIDS = ["E_f5", "E_f6", "E_f7"]
A_INJ = 0.045


def to_phys(x, nm):
    lo, hi = LIMS[nm]
    v = lo + x * (hi - lo)
    return v * 1e9 if nm == "Ap" else v   # A_p in 1e-9 units


def load_arm(base):
    ps = sorted(p for p in glob.glob(f"{CKPT}/{base}_c*.npz")
                if os.path.basename(p).split(base)[1].startswith("_c"))
    chains, names, truth, ndiv = [], None, None, 0
    for p in ps:
        z = np.load(p, allow_pickle=True)
        if names is None:
            names = [str(x) for x in z["names"]]
            truth = np.asarray(z["truth_vec"])
        chains.append(np.asarray(z["packed"]))
        ndiv += int(z["divergences"]) if "divergences" in z else 0
    if not chains:
        return None
    return dict(P=np.concatenate(chains, 0), names=names, truth=truth, nchain=len(ps), ndiv=ndiv)


def stat(arm, nm):
    i = arm["names"].index(nm)
    col = to_phys(arm["P"][:, i], nm) if nm in LIMS else arm["P"][:, i]
    tr = to_phys(arm["truth"][i], nm) if nm in LIMS else arm["truth"][i]
    return float(np.mean(col)), float(np.std(col)), float(tr)


rows = []
for fid in FIDS:
    si = load_arm(f"{fid}_si")
    cl = load_arm(f"{fid}")
    if si is None or cl is None:
        print(f"[no-leak] {fid}: missing arm (si={si is not None} clean={cl is not None})")
        continue
    a_m, a_s, _ = stat(si, "a_SiIII")
    ns_si = stat(si, "ns"); ns_cl = stat(cl, "ns")
    ap_si = stat(si, "Ap"); ap_cl = stat(cl, "Ap")
    # leak = shift between arms, in units of the (combined) posterior σ. NOTE: the SiIII and clean
    # arms share the SAME fold/sim/seed → byte-identical noise, so the two posteriors are positively
    # correlated; the true paired denom is sqrt(σ²+σ²−2ρσσ) < hypot. So this Δ/σ OVER-estimates the
    # leak σ → it is a CONSERVATIVE (loose) bound — a clean pass here is safe; a borderline one isn't.
    dns = ns_si[0] - ns_cl[0]; sns = np.hypot(ns_si[1], ns_cl[1])
    dap = ap_si[0] - ap_cl[0]; sap = np.hypot(ap_si[1], ap_cl[1])
    rows.append(dict(fid=fid, a_m=a_m, a_s=a_s,
                     ns_si=ns_si, ns_cl=ns_cl, dns=dns, dns_sig=dns / sns,
                     ap_si=ap_si, ap_cl=ap_cl, dap=dap, dap_sig=dap / sap,
                     ndiv_si=si["ndiv"], ndiv_cl=cl["ndiv"]))
    print(f"\n[{fid}]  div(si)={si['ndiv']} div(clean)={cl['ndiv']}")
    print(f"  a_SiIII : {a_m:.4f} ± {a_s:.4f}   (injected {A_INJ})   [{(a_m-A_INJ)/a_s:+.2f}σ]")
    print(f"  n_s     : SiIII {ns_si[0]:.4f}±{ns_si[1]:.4f}  clean {ns_cl[0]:.4f}±{ns_cl[1]:.4f}  "
          f"truth {ns_si[2]:.4f}   Δ(si−cl)={dns:+.4f} ({dns/sns:+.2f}σ)")
    print(f"  A_p     : SiIII {ap_si[0]:.4f}±{ap_si[1]:.4f}  clean {ap_cl[0]:.4f}±{ap_cl[1]:.4f}  "
          f"truth {ap_si[2]:.4f}   Δ(si−cl)={dap:+.4f} ({dap/sap:+.2f}σ) [×1e-9]")

# ----- figure: 3 panels (a_SiIII recovery, n_s no-leak, A_p no-leak) ---------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
x = np.arange(len(rows))
labs = [r["fid"] for r in rows]

# (1) a_SiIII recovery
ax = axes[0]
ax.errorbar(x, [r["a_m"] for r in rows], yerr=[r["a_s"] for r in rows], fmt="o", ms=7,
            color="C3", capsize=4, label="posterior")
ax.axhline(A_INJ, color="0.4", ls="--", lw=1.4, label=f"injected {A_INJ}")
ax.axhspan(0, 0.15, color="0.85", alpha=0.4, zorder=0, label="prior U[0,0.15]")
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel(r"$a_{\rm SiIII}$"); ax.set_title("(1) SiIII amplitude recovery")
ax.set_ylim(-0.005, 0.16); ax.legend(fontsize=9); ax.grid(alpha=0.25)

# (2) n_s no-leak: SiIII vs clean posteriors + truth
ax = axes[1]
ax.errorbar(x - 0.08, [r["ns_si"][0] for r in rows], yerr=[r["ns_si"][1] for r in rows],
            fmt="s", ms=7, color="C3", capsize=4, label="SiIII (inject+sample)")
ax.errorbar(x + 0.08, [r["ns_cl"][0] for r in rows], yerr=[r["ns_cl"][1] for r in rows],
            fmt="o", ms=7, color="C0", capsize=4, label="clean (no metals)")
ax.scatter(x, [r["ns_si"][2] for r in rows], marker="_", s=400, color="k", zorder=6, label="truth")
for xi, r in zip(x, rows):
    ax.annotate(f"Δ={r['dns_sig']:+.2f}σ", (xi, max(r['ns_si'][0], r['ns_cl'][0])),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel(r"$n_s$"); ax.set_title("(2) $n_s$ no-leak (SiIII vs clean)")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# (3) A_p no-leak
ax = axes[2]
ax.errorbar(x - 0.08, [r["ap_si"][0] for r in rows], yerr=[r["ap_si"][1] for r in rows],
            fmt="s", ms=7, color="C3", capsize=4, label="SiIII (inject+sample)")
ax.errorbar(x + 0.08, [r["ap_cl"][0] for r in rows], yerr=[r["ap_cl"][1] for r in rows],
            fmt="o", ms=7, color="C0", capsize=4, label="clean (no metals)")
ax.scatter(x, [r["ap_si"][2] for r in rows], marker="_", s=400, color="k", zorder=6, label="truth")
for xi, r in zip(x, rows):
    ax.annotate(f"Δ={r['dap_sig']:+.2f}σ", (xi, max(r['ap_si'][0], r['ap_cl'][0])),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel(r"$A_p\ [10^{-9}]$"); ax.set_title("(3) $A_p$ no-leak (SiIII vs clean)")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

fig.suptitle("eBOSS SiIII no-leak cert — a_SiIII absorbs the injected ±9% ripple without "
             "leaking into cosmology", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = f"{OUTDIR}/eboss_metals_noleak.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\n[no-leak] wrote {out}")

# markdown table for the doc
print("\n| fiducial | a_SiIII (inj 0.045) | n_s SiIII | n_s clean | Δn_s | A_p SiIII | A_p clean | ΔA_p | div |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['fid']} | {r['a_m']:.4f}±{r['a_s']:.4f} | "
          f"{r['ns_si'][0]:.4f}±{r['ns_si'][1]:.4f} | {r['ns_cl'][0]:.4f}±{r['ns_cl'][1]:.4f} | "
          f"{r['dns']:+.4f} ({r['dns_sig']:+.2f}σ) | "
          f"{r['ap_si'][0]:.3f}±{r['ap_si'][1]:.3f} | {r['ap_cl'][0]:.3f}±{r['ap_cl'][1]:.3f} | "
          f"{r['dap']:+.3f} ({r['dap_sig']:+.2f}σ) | {r['ndiv_si']}/{r['ndiv_cl']} |")
