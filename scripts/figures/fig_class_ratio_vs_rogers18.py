#!/usr/bin/env python3
"""Figure: the class power ratio of the PRIYA simulations against the per-class model of Rogers et al. (2018), z = 3.

    python scripts/figures/fig_class_ratio_vs_rogers18.py [--out docs/figures/class_ratio_vs_rogers18.png]

What is drawn, in four small panels (one per class, because the classes span very different ranges):

- points: R = P_class / P_clean of the 60 low-resolution PRIYA training simulations with the filtered troughs
  restored, the mean and the standard deviation over the simulations, from the committed table
  ``docs/figures/data/class_ratio_z3p0_lf60.csv`` (its header says how it was made). The public predictor cannot
  produce these points: it returns the filtered class spectra, and restoring the troughs is one of its two stated
  omissions;
- line: Eq. 6 of Rogers et al. (2018) with its constant c(z), ``hcd_analysis.hcd_template.class_ratio``;
- shaded: wavenumbers below the lowest mode of the box of Rogers et al., where their template is an extrapolation.
  Their box is 8000 km/s long at their snapshot z = 3.01 (their sections 3.1 and 3.2), so its lowest mode is
  2 pi / 8000 = 7.85e-4 s/km. The simulation points are at z = 3.0.

It needs no model: the table and the public function are enough. See docs/rogers_comparison.md for what the
comparison is and is not.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_common as C  # noqa: E402

Z = 3.0
# Rogers et al. (2018), sections 3.1 and 3.2: the box is 8000 km/s long in velocity space at their snapshot z = 3.01.
ROGERS_BOX_KMS = 8000.0
ROGERS_KMIN = (
    2.0 * np.pi / ROGERS_BOX_KMS
)  # 7.85e-4 s/km: below it the template is an extrapolation
# the two classes whose ratio stays within a factor of a few are drawn on a linear axis
LINEAR_Y = ("LLS", "subDLA")
# (panel title, the column stem of the table, the class name of hcd_template.class_ratio)
PANELS = (
    ("LLS", "LLS", "LLS"),
    ("subDLA", "subDLA", "Sub-DLA"),
    ("small DLA", "smallDLA", "Small-DLA"),
    ("large DLA", "largeDLA", "Large-DLA"),
)


def read_table(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        header = [ln for ln in fh if ln.startswith("#")][-1].lstrip("# ").strip().split(",")
    values = np.loadtxt(path, delimiter=",", comments="#")
    return {name: values[:, i] for i, name in enumerate(header)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=os.path.join(C.FIGURES, "class_ratio_vs_rogers18.png"))
    args = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from hcd_analysis.hcd_template import class_ratio

    table = read_table(C.data_file("class_ratio_z3p0_lf60.csv"))
    k = table["k_s_per_km"]
    kk = np.geomspace(k.min(), k.max(), 300)
    template = class_ratio(kk, Z)
    C.style(plt)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.1), sharex=True, constrained_layout=True)
    for ax, (title, stem, name) in zip(axes, PANELS):
        mean, sd = table[f"R_{stem}_addback_mean"], table[f"R_{stem}_addback_sd_over_sims"]
        ax.axvspan(k.min() * 0.9, ROGERS_KMIN, color=C.SHADE, lw=0, zorder=0)
        ax.plot(kk, template[name], color=C.BLUE, zorder=2, label="Rogers et al. (2018), Eq. 6")
        ax.errorbar(
            k, mean, yerr=sd, fmt="o", ms=3.2, color=C.INK, ecolor=C.INK_SECONDARY, elinewidth=0.8,
            mec=C.SURFACE, mew=0.5, zorder=3, label="PRIYA simulations, troughs restored\n(mean and standard deviation over 60)",
        )  # fmt: skip
        ax.set_xscale("log")
        if stem not in LINEAR_Y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlim(k.min() * 0.9, 0.03)
        ax.set_title(title, loc="left")
        ax.set_xlabel("k (s/km)")
    axes[0].set_ylabel("P_class / P_clean")
    axes[0].text(
        k.min() * 0.95, 0.04, "template\nextrapolated", transform=axes[0].get_xaxis_transform(),
        fontsize=7.5, color=C.INK_SECONDARY, va="bottom", ha="left",
    )  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=2)
    fig.suptitle(f"Class power ratio at z = {Z:g}", x=0.01, ha="left", fontsize=10.5)
    C.save(
        fig,
        args.out,
        {
            "figure": "class power ratio against Rogers et al. (2018), Eq. 6",
            "z": Z,
            "points": "docs/figures/data/class_ratio_z3p0_lf60.csv (troughs restored; mean and sd over 60 simulations)",
            "line": "hcd_analysis.hcd_template.class_ratio",
            "template_extrapolated_below_k_s_per_km": ROGERS_KMIN,
            "template_box_km_per_s": ROGERS_BOX_KMS,
            "model": "none: this figure needs no model",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
