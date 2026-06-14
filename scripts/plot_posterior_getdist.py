"""GetDist (cobaya-style) triangle plot of a STEP-A closure fiducial's posterior.

Reads the per-chain npz (checkpoints/stepA/<FID>_c*.npz; packed (N,P) in UNIT cube for the 9
theta params, physical for tau0/alpha), converts the shown box params to physical units, and
renders a GetDist triangle with truth markers (dashed). Marginals + truth printed to stdout.

ENV: getdist lives in the `emu-3.9` env (NOT emu-jax). Run with:
  /home/mfho/.conda/envs/emu-3.9/bin/python3 scripts/plot_posterior_getdist.py [FID] [p1,p2,...]
e.g.  ... plot_posterior_getdist.py D_llsmed ns,Ap,hub,bhfeedback,alpha_lls
"""
import sys, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
from getdist import MCSamples, plots

FID = sys.argv[1] if len(sys.argv) > 1 else "D_llsmed"
SHOW = (sys.argv[2].split(",") if len(sys.argv) > 2
        else ["ns", "Ap", "hub", "bhfeedback", "alpha_lls"])

# theta (first 9 packed) is the UNIT cube; convert these to physical via PARAM_LIMITS (Ap -> 1e-9).
LIMS = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9), "herei": (3.5, 4.5), "heref": (2.2, 3.2),
        "alphaq": (1.3, 3.0), "hub": (0.65, 0.75), "omegamh2": (0.14, 0.146),
        "hireionz": (6.5, 8.0), "bhfeedback": (0.03, 0.07)}
LABELS = {"ns": r"n_s", "Ap": r"A_p\,[10^{-9}]", "herei": r"z_{\rm HeII,i}", "heref": r"z_{\rm HeII,f}",
          "alphaq": r"\alpha_q", "hub": "h", "omegamh2": r"\Omega_m h^2", "hireionz": r"z_{\rm HI}",
          "bhfeedback": r"\epsilon_{\rm BH}", "alpha_lls": r"\alpha_{\rm LLS}",
          "alpha_subdla": r"\alpha_{\rm subDLA}", "alpha_dla": r"\alpha_{\rm DLA}",
          "tau0_amp": r"\tau_0", "dtau0": r"d\tau_0"}


def to_phys(x, nm):
    if nm in LIMS:
        lo, hi = LIMS[nm]; v = lo + x * (hi - lo)
        return v * 1e9 if nm == "Ap" else v
    return x


ps = sorted(p for p in glob.glob(f"checkpoints/stepA/{FID}_c*.npz") if p.split(FID)[1].startswith("_c"))
if not ps:
    sys.exit(f"no chains for {FID}")
chains, names, truth = [], None, None
for p in ps:
    z = np.load(p, allow_pickle=True)
    names = list(z["names"]) if names is None else names
    truth = np.asarray(z["truth_vec"]) if truth is None else truth
    chains.append(np.asarray(z["packed"]))
P = np.concatenate(chains, 0)

samp = np.column_stack([to_phys(P[:, names.index(nm)], nm) for nm in SHOW])
tvals = [to_phys(truth[names.index(nm)], nm) for nm in SHOW]
markers = {nm: tvals[i] for i, nm in enumerate(SHOW)}

mcs = MCSamples(samples=samp, names=SHOW, labels=[LABELS.get(n, n) for n in SHOW],
                label=f"{FID} ({len(ps)} chains)")
g = plots.get_subplot_plotter(width_inch=2.0 + 1.5 * len(SHOW))
g.settings.alpha_filled_add = 0.5
g.triangle_plot([mcs], filled=True, markers=markers,
                marker_args={"color": "k", "lw": 1.2}, title_limit=1)
_OUTDIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"  # diagnostic figs -> notes repo
os.makedirs(_OUTDIR, exist_ok=True)
out = f"{_OUTDIR}/posterior_getdist_{FID}.png"
g.export(out)
print("wrote", out)
for i, nm in enumerate(SHOW):
    print(f"  {nm:12} {samp[:, i].mean():>10.4g} +/- {samp[:, i].std():<9.3g}  truth={tvals[i]:.4g}")
