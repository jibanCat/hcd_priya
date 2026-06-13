"""eBOSS DR14 cert — cosmo+HCD and IGM GetDist corner plots (truth markers) for the doc.

Reuses the unit->physical conversion of plot_posterior_getdist.py. Emits TWO corners per fiducial:
  cosmo+HCD: ns, Ap, alpha_lls, alpha_subdla, alpha_dla
  IGM:       herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback
ENV (getdist): /home/mfho/.conda/envs/emu-3.9/bin/python3 scripts/plot_eboss_corners.py [FID]
"""
import sys, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
from getdist import MCSamples, plots

FID = sys.argv[1] if len(sys.argv) > 1 else "E_f6"
GROUPS = {
    "cosmo": ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"],
    "igm": ["herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"],
}
LIMS = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9), "herei": (3.5, 4.5), "heref": (2.2, 3.2),
        "alphaq": (1.3, 3.0), "hub": (0.65, 0.75), "omegamh2": (0.14, 0.146),
        "hireionz": (6.5, 8.0), "bhfeedback": (0.03, 0.07)}
LABELS = {"ns": r"n_s", "Ap": r"A_p\,[10^{-9}]", "herei": r"z_{\rm HeII,i}", "heref": r"z_{\rm HeII,f}",
          "alphaq": r"\alpha_q", "hub": "h", "omegamh2": r"\Omega_m h^2", "hireionz": r"z_{\rm HI}",
          "bhfeedback": r"\epsilon_{\rm BH}", "alpha_lls": r"\alpha_{\rm LLS}",
          "alpha_subdla": r"\alpha_{\rm subDLA}", "alpha_dla": r"\alpha_{\rm DLA}"}
OUTDIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"
os.makedirs(OUTDIR, exist_ok=True)


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
    names = [str(x) for x in z["names"]] if names is None else names
    truth = np.asarray(z["truth_vec"]) if truth is None else truth
    chains.append(np.asarray(z["packed"]))
P = np.concatenate(chains, 0)

for grp, SHOW in GROUPS.items():
    SHOW = [s for s in SHOW if s in names]
    samp = np.column_stack([to_phys(P[:, names.index(nm)], nm) for nm in SHOW])
    tvals = [to_phys(truth[names.index(nm)], nm) for nm in SHOW]
    mcs = MCSamples(samples=samp, names=SHOW, labels=[LABELS.get(n, n) for n in SHOW],
                    label=f"{FID} ({len(ps)} chains), {grp}")
    # set each UNIFORM-prior (box) param's axis to its FULL prior range, so the box BOUNDARIES are
    # the axis edges → a prior-driven param visibly FILLS the box, a constrained one peaks inside it.
    # (the HCD α_* are Gaussian-prior, not box, so they're left to autoscale.)
    plim = {}
    for nm in SHOW:
        if nm in LIMS:
            lo, hi = LIMS[nm]
            plim[nm] = (lo * 1e9, hi * 1e9) if nm == "Ap" else (lo, hi)
    g = plots.get_subplot_plotter(width_inch=2.0 + 1.4 * len(SHOW))
    g.settings.alpha_filled_add = 0.5
    g.triangle_plot([mcs], filled=True, markers={nm: tvals[i] for i, nm in enumerate(SHOW)},
                    marker_args={"color": "k", "lw": 1.2}, title_limit=1, param_limits=plim)
    out = f"{OUTDIR}/eboss_corner_{grp}_{FID}.png"
    g.export(out)
    print("wrote", out)
    for i, nm in enumerate(SHOW):
        print(f"  {nm:12} {samp[:, i].mean():>10.4g} +/- {samp[:, i].std():<9.3g}  truth={tvals[i]:.4g}")
