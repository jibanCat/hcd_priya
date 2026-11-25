"""Figure 2 of the JOSS paper: the class power ratio of the PRIYA training simulations against Rogers et al. (2018).

It shows what the public figure docs/figures/class_ratio_vs_rogers18.png of the code repository shows, redrawn in
the style of the other figures, with little text in the figure and the explanations in the caption. Same content as
the public figure, from the same public inputs, at the tag v1.0.0 of the code repository:
  - simulations: docs/figures/data/class_ratio_ranges_z3p0_lf60.csv, the ratio P_class / P_clean at z = 3.0 of the
    60 low-resolution PRIYA training simulations with the filtered troughs restored. Light band: minimum to maximum
    at all 20 mean-flux rungs. Dark band and line: minimum to maximum and median at the rung nearest the Kim fit;
  - template: Eq. 6 of Rogers et al. (2018), from the public function hcd_analysis.hcd_template.class_ratio. Its
    band: c + alpha T for alpha from 0.5 to 2, where T is the k-dependent term and c the constant (the public
    figure's choice, not a prior of that paper);
  - shaded: k below the lowest mode of the box of Rogers et al., 2 pi / 8000 km/s, where the template is
    extrapolated.
Both inputs are read from the public code (a local clone through HCD_PRIYA_REPO, or GitHub) and checked by sha256.
The template module needs numpy only; it is loaded from its source text, and nothing is written to the code.

Run from the repository root:  python3 paper/figures/src/fig2_ratio.py
"""

import hashlib
import io
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

from _public import fetch  # noqa: E402
from _style import BAND, TEAL, TEAL_LIGHT, WARM, style  # noqa: E402

COMMIT = "82ef21f698fc5cbdb21fcd971a7d2795da1dd984"  # the tag v1.0.0
TABLE = ("docs/figures/data/class_ratio_ranges_z3p0_lf60.csv",
         "0f79f1e5260df432fbe3b69b32dd4d4ebca248e95f789b11c40be549ef926f05")
TEMPLATE = ("hcd_analysis/hcd_template.py", "a0d4ae02b58812aa0f3145e1c53378a7f02687d8ab1776421292893605990b4a")
OUT = Path(__file__).resolve().parent.parent / "fig2_ratio.png"
Z = 3.0
ROGERS_KMIN = 2 * np.pi / 8000.0  # the lowest mode of the box of Rogers et al. (8000 km/s at z = 3.01)
ALPHA = (0.5, 2.0)  # the band of the template, as in the public figure
PANELS = (("LLS", "LLS", "LLS", False), ("subDLA", "subDLA", "Sub-DLA", False),
          ("small DLA", "smallDLA", "Small-DLA", True), ("large DLA", "largeDLA", "Large-DLA", True))


def read(path, sha):
    data = fetch(COMMIT, path)
    if data is None:
        return None
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{path} at {COMMIT[:7]} is not the file this figure was made on")
    return data


def main():
    table, source = read(*TABLE), read(*TEMPLATE)
    if table is None or source is None:
        if OUT.exists():
            print("KEPT", OUT, "(cannot read the public inputs)")
            return 0
        raise SystemExit("cannot read the public inputs, and there is no image to keep")
    text = table.decode()
    header = [ln for ln in text.splitlines() if ln.startswith("#")][-1].lstrip("# ").strip().split(",")
    values = np.loadtxt(io.StringIO(text), delimiter=",", comments="#")
    col = {name: values[:, i] for i, name in enumerate(header)}
    mod = types.ModuleType("hcd_template_public")
    exec(compile(source.decode(), TEMPLATE[0], "exec"), mod.__dict__)  # noqa: S102 - public code, hash-checked

    k = col["k_s_per_km"]
    kk = np.geomspace(k.min(), 0.03, 300)
    ratio, const = mod.class_ratio(kk, Z), mod.class_ratio(np.array([1.0]), Z)

    style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.9), sharex=True)
    for ax, (title, stem, name, logy) in zip(axes.ravel(), PANELS):
        ax.axvspan(k.min() * 0.9, ROGERS_KMIN, color=BAND, lw=0, zorder=0)
        ax.fill_between(k, col[f"R_{stem}_sims60_allrungs_min"], col[f"R_{stem}_sims60_allrungs_max"],
                        color=TEAL_LIGHT, alpha=0.6, lw=0, zorder=1)
        ax.fill_between(k, col[f"R_{stem}_sims60_rung1_min"], col[f"R_{stem}_sims60_rung1_max"],
                        color=TEAL, alpha=0.45, lw=0, zorder=2)
        ax.plot(k, col[f"R_{stem}_sims60_rung1_p50"], color=TEAL, lw=1.3, zorder=3)
        c = float(const[name][0])
        edges = [c + a * (ratio[name] - c) for a in ALPHA]
        ax.fill_between(kk, np.minimum(*edges), np.maximum(*edges), color=WARM, alpha=0.2, lw=0, zorder=4)
        ax.plot(kk, ratio[name], color=WARM, lw=1.3, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(k.min() * 0.9, 0.03)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.text(0.95, 0.9, title, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$k$ [s/km]")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$P_c / P_\mathrm{clean}$")
    fig.legend(handles=[
        (Patch(fc=TEAL_LIGHT, alpha=0.6, lw=0), Line2D([], [], color=TEAL, lw=1.3)),
        (Patch(fc=WARM, alpha=0.2, lw=0), Line2D([], [], color=WARM, lw=1.3)),
    ], labels=["PRIYA simulations", "Rogers et al. (2018)"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0),
        handlelength=1.8, columnspacing=2.0)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.17, hspace=0.06, wspace=0.22)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
