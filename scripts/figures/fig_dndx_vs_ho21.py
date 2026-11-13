#!/usr/bin/env python3
"""Figure: the incidence dN/dX the model predicts against redshift, with the DLA measurement of Ho, Bird and
Garnett (2021).

    python scripts/figures/fig_dndx_vs_ho21.py (--published | --release RELEASE_DIR SHA256) \\
        [--design OTHER_DESIGN.csv | --no-band] [--ho21-dndx FILE --ho21-dndx-rows-are-centre-and-intervals] \\
        [--out ...]

What is drawn, on one axis:

- lines: ``pred.dndx`` of the LF+MF model for DLAs (the heavier line), subDLAs and LLS, at the snapshot redshifts
  from 2.2 to 4.6, at the nine parameter values written out in ``figures_common.MIDPOINT`` (the midpoint of each
  parameter's training range: not a fiducial cosmology), with ``tau0_amp = 1`` and ``dtau0 = 0``. The curves end at
  z = 4.6: above it the multi-fidelity correction is outside the redshift range it is validated on, and below 2.2
  the model refuses an LF+MF prediction;
- bands: the lowest and the highest value over the 60 design points of the PRIYA suite
  (``docs/figures/data/priya_lf_design_60.csv``; another table with ``--design``, none with ``--no-band``);
- points (only with ``--ho21-dndx``): the DLA dN/dX of Ho, Bird and Garnett (2021).

**The measurement's table has no reader in its source repository.** The file is six rows of eighteen numbers. This
script draws it only when it is told, with ``--ho21-dndx-rows-are-centre-and-intervals``, that row 0 is the redshift
of the bin centre, row 1 the central dN/dX, rows 2 and 3 the limits of the 68 per cent interval and rows 4 and 5
those of the 95 per cent interval. Until that reading is confirmed by the paper's authors the points are not drawn
and the file is not among the committed figure data.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_common as C  # noqa: E402

# the snapshot redshifts inside the range the LF+MF model serves and is validated on
REDSHIFTS = tuple(round(2.2 + 0.2 * i, 1) for i in range(13))  # 2.2, 2.4, ..., 4.6
CLASSES = (("DLA", 2, "BLUE", 2.2), ("subDLA", 1, "ORANGE", 1.3), ("LLS", 0, "AQUA", 1.3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    C.add_model_arguments(ap)
    C.add_design_arguments(ap)
    ap.add_argument("--ho21-dndx", help="the dN/dX table of Ho, Bird and Garnett (2021), six rows")
    ap.add_argument(
        "--ho21-dndx-rows-are-centre-and-intervals",
        action="store_true",
        help="say that the table's rows are: redshift, central value, 68 per cent limits, 95 per cent limits",
    )
    ap.add_argument("--out", default=os.path.join(C.FIGURES, "dndx_vs_ho21.png"))
    args = ap.parse_args(argv)
    if args.ho21_dndx and not args.ho21_dndx_rows_are_centre_and_intervals:
        ap.error(
            "--ho21-dndx needs --ho21-dndx-rows-are-centre-and-intervals: the meaning of its rows is not recorded"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    import hcd_analysis.predictor as hp

    emu, what = C.load_model(args)
    if not what["incidence_record"]:
        raise SystemExit(
            "this model has no incidence MF record: its dN/dX under LF+MF is the LF model's, which is not what "
            "this figure is labelled as"
        )
    design = C.design_points(args, hp.PARAM_NAMES)
    stack = emu.predict_many(C.MIDPOINT, z=list(REDSHIFTS), tau0_amp=1.0, dtau0=0.0)
    facts = dict(
        what, figure="dN/dX against redshift", parameters=C.MIDPOINT, redshifts=list(REDSHIFTS)
    )
    facts["flags_set"] = sorted(n for n, v in stack.flags.items() if v is True)
    bands = None
    if design is not None:
        lo, hi, all_flagged, all_refused = [], [], {}, {}
        for z in REDSHIFTS:
            drawn = C.flags_set(emu.predict(C.MIDPOINT, z=z, tau0_amp=1.0, dtau0=0.0))
            a, b, flagged, refused = C.envelope(emu, design, z, "dndx", drawn)
            lo.append(a), hi.append(b)
            for i, hit in flagged:  # a point counts once, with every flag it had at any redshift
                all_flagged.setdefault(i, set()).update(hit)
            for i, why in refused:
                all_refused.setdefault(i, why)
        bands = (np.array(lo), np.array(hi))
        facts.update(
            design_points=len(design),
            design_points_in_band_at_every_redshift=len(design) - len(all_refused),
            design_points_flagged=len(all_flagged),
            design_points_refused=len(all_refused),
            design_point_flags=C.flag_counts((i, sorted(hit)) for i, hit in all_flagged.items()),
        )
        print(
            f"{len(design)} design points; with a flag the drawn line does not have, at some redshift: "
            f"{len(all_flagged)}; refused at some redshift: {len(all_refused)}"
        )
        for i in sorted(all_flagged):
            print(f"   design point {i}: {', '.join(sorted(all_flagged[i]))}")
        for i in sorted(all_refused):
            print(f"   design point {i} refused: {all_refused[i]}")
    C.style(plt)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    z = np.array(REDSHIFTS)
    for name, col, colour, width in CLASSES:
        colour = getattr(C, colour)
        if bands is not None:
            ax.fill_between(
                z, bands[0][:, col], bands[1][:, col], color=colour, alpha=0.2, lw=0, zorder=1
            )
        ax.plot(
            z, stack.dndx[:, col], color=colour, lw=width, zorder=2, label=f"model (LF+MF): {name}"
        )
        ax.annotate(
            name,
            (z[-1], stack.dndx[-1, col]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )
    if args.ho21_dndx:
        a = np.loadtxt(args.ho21_dndx)
        if a.shape[0] != 6:
            raise SystemExit(f"{args.ho21_dndx}: expected six rows, found {a.shape[0]}")
        ax.errorbar(
            a[0], a[1], yerr=[a[1] - a[2], a[3] - a[1]], fmt="o", ms=3.6, color=C.INK, ecolor=C.INK_SECONDARY,
            elinewidth=0.8, mec=C.SURFACE, mew=0.5, zorder=3, label="DLA, Ho, Bird and Garnett (2021), 68 per cent interval",
        )  # fmt: skip
        facts["measurement"] = (
            "DLA dN/dX of Ho, Bird and Garnett (2021); rows read as centre and intervals"
        )
    else:
        facts["measurement"] = "not drawn: the meaning of the table's rows is not confirmed"
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda v, _: f"{v:g}" if f"{v:e}"[0] in "25" else "")
    )
    ax.set_xlim(1.95, 5.2)
    ax.set_xlabel("redshift z")
    ax.set_ylabel("dN/dX")
    ax.set_title("Incidence of the absorbers per unit absorption distance", loc="left")
    if bands is not None:
        ax.plot(
            [],
            [],
            color=C.INK_SECONDARY,
            alpha=0.3,
            lw=6,
            label=f"range over {len(design) - len(all_refused)} design points",
        )
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
    C.save(fig, args.out, facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
