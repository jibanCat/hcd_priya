#!/usr/bin/env python3
"""Figure: the class power ratio of the PRIYA simulations against the per-class model of Rogers et al. (2018), z = 3.

    python scripts/figures/fig_class_ratio_vs_rogers18.py [--out docs/figures/class_ratio_vs_rogers18.png]

What is drawn, in four small panels (one per class, because the classes span very different ranges):

- the simulations, in one hue: R = P_class / P_clean of the low-resolution PRIYA training simulations with the
  filtered troughs restored, as ranges from the committed table
  ``docs/figures/data/class_ratio_ranges_z3p0_lf60.csv`` (its header says how it was made). The lighter band is the
  range, minimum to maximum, over the 60 simulations at all 20 mean-flux rungs; the darker band is the range over
  the 60 simulations at the mean-flux rung nearest 1, and the line is its median. The 60 simulations differ in the
  nine cosmological and astrophysical parameters. The public predictor cannot produce these ratios: it returns the
  filtered class spectra, and restoring the troughs is one of its two stated omissions;
- the template, in a second hue: the line is Eq. 6 of Rogers et al. (2018) as published,
  ``hcd_analysis.hcd_template.class_ratio``. The band is c(z) + alpha * T(k, z) for alpha between 0.5 and 2, where
  T is the k-dependent term of Eq. 6 (``class_ratio`` minus its constant c). In a fit Rogers et al. leave the
  amplitudes free and let a free normalisation absorb c(z). The range 0.5 to 2 is this figure's choice, to show
  what a free amplitude does; it is not a prior from the paper;
- shaded: wavenumbers below the lowest mode of the box of Rogers et al., where their template is an extrapolation.
  Their box is 8000 km/s long at their snapshot z = 3.01 (their sections 3.1 and 3.2), so its lowest mode is
  2 pi / 8000 = 7.85e-4 s/km. The simulations are at z = 3.0.

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


# The two sets of a table of ranges, widest first: (the column infix, the opacity of its band, its legend text).
RANGE_SETS = (
    ("sims60_allrungs", 0.16, "PRIYA simulations, troughs restored: 60 simulations at all mean-flux rungs\n(minimum to maximum)"),
    ("sims60_rung1", 0.38, "the 60 simulations at the mean-flux rung nearest 1\n(minimum to maximum, and the median as a line)"),
)  # fmt: skip


def draw_ranges(ax, ranges, stem):
    """The simulations as bands in one hue: for each set the range from its minimum to its maximum, the wider set
    lighter, and the median of the narrower set as a line. Columns: ``R_<stem>_<set>_<min|p16|p50|p84|max>``.
    """
    k = ranges["k_s_per_km"]
    for infix, alpha, label in RANGE_SETS:
        lo, hi = ranges[f"R_{stem}_{infix}_min"], ranges[f"R_{stem}_{infix}_max"]
        ax.fill_between(k, lo, hi, color=C.BLUE, alpha=alpha, lw=0, zorder=1, label=label)
    ax.plot(k, ranges[f"R_{stem}_{RANGE_SETS[-1][0]}_p50"], color=C.BLUE, lw=1.6, zorder=2)


# The amplitude of the k-dependent term of the template is scaled over this range to draw the band: a choice of this
# figure, not a prior of the paper.
ALPHA_RANGE = (0.5, 2.0)
RANGES_FILE = "class_ratio_ranges_z3p0_lf60.csv"


def template_with_band(k, z):
    """For each class of the template: (Eq. 6 as published, the lower and the upper edge of the band
    c + alpha * T for alpha over ``ALPHA_RANGE``), where c is the constant of Eq. 6 and T its k-dependent term.
    """
    from hcd_analysis.hcd_template import class_ratio

    ratio = class_ratio(k, z)
    constant = class_ratio(
        np.array([1.0]), z
    )  # far above the figure's k range the ratio is its constant c(z)
    out = {}
    for name, r in ratio.items():
        c = float(constant[name][0])
        term = r - c
        edges = [c + a * term for a in ALPHA_RANGE]
        out[name] = (r, np.minimum(*edges), np.maximum(*edges))
    return out


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

    ranges = read_table(C.data_file(RANGES_FILE))
    k = ranges["k_s_per_km"]
    kk = np.geomspace(k.min(), k.max(), 300)
    template = template_with_band(kk, Z)
    C.style(plt)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3), sharex=True, constrained_layout=True)
    for ax, (title, stem, name) in zip(axes, PANELS):
        line, lo, hi = template[name]
        ax.axvspan(k.min() * 0.9, ROGERS_KMIN, color=C.SHADE, lw=0, zorder=0)
        draw_ranges(ax, ranges, stem)
        ax.fill_between(
            kk, lo, hi, color=C.ORANGE, alpha=0.22, lw=0, zorder=3,
            label="the same with the amplitude of the k-dependent term between 0.5 and 2 times\n(a free amplitude in a fit)",
        )  # fmt: skip
        ax.plot(kk, line, color=C.ORANGE, lw=1.6, zorder=4, label="Rogers et al. (2018), Eq. 6")
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
    order = [
        0,
        1,
        3,
        2,
    ]  # the two simulation bands, then the template line, then its band ("the same with ...")
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order], loc="outside lower center", ncols=2
    )
    fig.suptitle(f"Class power ratio at z = {Z:g}", x=0.01, ha="left", fontsize=10.5)
    C.save(
        fig,
        args.out,
        {
            "figure": "class power ratio against Rogers et al. (2018), Eq. 6",
            "z": Z,
            "simulations": f"docs/figures/data/{RANGES_FILE}: ranges over the simulations, troughs restored",
            "line": "hcd_analysis.hcd_template.class_ratio",
            "template_band": f"c + alpha * T for alpha from {ALPHA_RANGE[0]:g} to {ALPHA_RANGE[1]:g}: this figure's choice",
            "template_extrapolated_below_k_s_per_km": ROGERS_KMIN,
            "template_box_km_per_s": ROGERS_BOX_KMS,
            "model": "none: this figure needs no model",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
