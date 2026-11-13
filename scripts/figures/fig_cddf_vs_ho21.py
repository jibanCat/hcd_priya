#!/usr/bin/env python3
"""Figure: the column-density distribution f(N_HI) the model predicts, against Ho, Bird and Garnett (2021).

    python scripts/figures/fig_cddf_vs_ho21.py (--published | --release RELEASE_DIR SHA256) \\
        [--design OTHER_DESIGN.csv | --no-band] [--out docs/figures/cddf_vs_ho21.png]

What is drawn, in three panels (one per redshift bin of the measurement):

- line: ``pred.cddf`` of the LF+MF model at a snapshot redshift inside the bin (2.2, 2.8 and 3.4), at the nine
  parameter values written out in ``figures_common.MIDPOINT`` (the midpoint of each parameter's training range: not
  a fiducial cosmology), with ``tau0_amp = 1`` and ``dtau0 = 0``. A bin the model does not support is NaN and is
  simply absent: the line and the band break there. The flags of each drawn prediction go into the side file;
- band: the lowest and the highest ``pred.cddf`` over the 60 design points of the PRIYA suite
  (``docs/figures/data/priya_lf_design_60.csv``; another table with ``--design``, none with ``--no-band``), each
  evaluated in the same way. The script reports every design point that the predictor refuses, and every one for
  which it sets a flag that the drawn line does not have;
- points: f(N_HI) of Ho, Bird and Garnett (2021) with the limits of their 68 per cent interval, from
  ``docs/figures/data/ho21`` (see PROVENANCE.md there). A bin in which the measured f is zero is not drawn, also
  where the table gives an upper limit for it.

The model needs an incidence MF record (a release of format 2): without one it returns no CDDF. The measurement bin
from z = 4 to 5 is left out: it reaches beyond z 4.6, the upper edge of the redshift range the multi-fidelity
correction is validated on.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_common as C  # noqa: E402

# (the measurement's redshift bin, its file, the snapshot redshift the model is drawn at)
PANELS = (
    ("2 < z < 2.5", "cddf_z225.txt", 2.2),
    ("2.5 < z < 3", "cddf_z253.txt", 2.8),
    ("3 < z < 4", "cddf_z34.txt", 3.4),
)


def read_ho21_cddf(path: str):
    """A CDDF table of Ho, Bird and Garnett (2021): six rows. Row 0: log10 N_HI of the bin centre. Row 1: f(N_HI).
    Rows 2 and 3: the lower and upper limit of the 68 per cent interval. Rows 4 and 5: those of the 95 per cent
    interval (not drawn). Bins with f = 0 are left out."""
    a = np.loadtxt(path)
    if a.shape[0] != 6:
        raise SystemExit(f"{path}: expected six rows, found {a.shape[0]}")
    keep = a[1] > 0
    return a[0][keep], a[1][keep], a[2][keep], a[3][keep]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    C.add_model_arguments(ap)
    C.add_design_arguments(ap)
    ap.add_argument("--out", default=os.path.join(C.FIGURES, "cddf_vs_ho21.png"))
    args = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import hcd_analysis.predictor as hp

    emu, what = C.load_model(args)
    if not what["incidence_record"]:
        raise SystemExit(
            "this model has no incidence MF record, so it returns no CDDF: nothing to draw"
        )
    design = C.design_points(args, hp.PARAM_NAMES)
    C.style(plt)
    fig, axes = plt.subplots(
        1, 3, figsize=(10.0, 3.4), sharex=True, sharey=True, constrained_layout=True
    )
    facts = dict(
        what,
        figure="CDDF against Ho, Bird and Garnett (2021)",
        parameters=C.MIDPOINT,
        design_table=C.design_source(args),
        panels=[],
    )
    for ax, (bin_label, name, z) in zip(axes, PANELS):
        pred = emu.predict(C.MIDPOINT, z=z, tau0_amp=1.0, dtau0=0.0)
        x = pred.log_nhi
        panel = {
            "measurement_bin": bin_label,
            "model_z": z,
            "model_bins_absent": int(np.sum(np.isnan(pred.cddf))),
            "model_flags_set": C.flags_set(pred),
        }
        if design is not None:
            lo, hi, flagged, refused = C.envelope(emu, design, z, "cddf", panel["model_flags_set"])
            in_band = len(design) - len(refused)
            C.band(ax, x, lo, hi, color=C.BLUE, alpha=0.22, lw=0, zorder=1, label=f"range over {in_band} design points")  # fmt: skip
            panel.update(
                design_points=len(design),
                design_points_in_band=in_band,
                design_points_flagged=len(flagged),
                design_points_refused=len(refused),
                design_point_flags=C.flag_counts(flagged),
            )
            print(
                f"z = {z}: {len(design)} design points; with a flag the drawn line does not have: {len(flagged)}; "
                f"refused: {len(refused)}"
            )
            for i, hit in flagged:
                print(f"   design point {i}: {', '.join(hit)}")
            for i, why in refused:
                print(f"   design point {i} refused: {why}")
        C.line(ax, x, pred.cddf, color=C.BLUE, zorder=2, label="model (LF+MF), at the parameters of the script")  # fmt: skip
        logn, f, f_lo, f_hi = read_ho21_cddf(C.data_file("ho21", name))
        ax.errorbar(
            logn, f, yerr=[f - f_lo, f_hi - f], fmt="o", ms=3.4, color=C.INK, ecolor=C.INK_SECONDARY, elinewidth=0.8,
            mec=C.SURFACE, mew=0.5, zorder=3, label="Ho, Bird and Garnett (2021), 68 per cent interval",
        )  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(f"measured: {bin_label};  model: z = {z:g}", loc="left")
        ax.set_xlabel("log10 of N_HI (cm^-2)")
        facts["panels"].append(panel)
    axes[0].set_ylabel("f(N_HI) (cm^2)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    fig.suptitle("Column-density distribution of the absorbers", x=0.01, ha="left", fontsize=10.5)
    C.save(fig, args.out, facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
