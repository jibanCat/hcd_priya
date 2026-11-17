#!/usr/bin/env python3
"""Figure: the incidence dN/dX the model predicts against redshift, with the DLA measurement of Ho, Bird and
Garnett (2021) and published measurements of the incidence of Lyman limit systems.

    python scripts/figures/fig_dndx_vs_ho21.py (--published | --release RELEASE_DIR SHA256) \\
        [--design OTHER_DESIGN.csv | --no-band] [--out docs/figures/dndx_vs_ho21.png]

What is drawn, on one axis:

- lines: ``pred.dndx`` of the LF+MF model for DLAs (the heavier line), subDLAs and LLS, at the snapshot redshifts
  from 2.2 to 4.6, at the nine parameter values written out in ``figures_common.MIDPOINT`` (the midpoint of each
  parameter's training range: not a fiducial cosmology), with ``tau0_amp = 1`` and ``dtau0 = 0``. The curves end at
  z = 4.6: above it the multi-fidelity correction of the P1D is outside its grouped-validation domain, and below
  2.2 the model refuses an LF+MF prediction. The correction of dN/dX that is drawn here is not validated at any
  redshift. The classes are bins of column density (``hcd_analysis/catalog.py``): LLS 10^17.2 <= N_HI < 10^19,
  subDLA 10^19 <= N_HI < 10^20.3, DLA N_HI >= 10^20.3, in cm^-2;
- the dashed line: the model's incidence of every absorber with N_HI >= 10^17.5 cm^-2, which is what the LLS
  measurements count (below); from the same predictions, see "The count above 10^17.5";
- bands: the lowest and the highest value over the 60 design points of the PRIYA suite
  (``docs/figures/data/priya_lf_design_60.csv``; another table with ``--design``, none with ``--no-band``);
- bars on the model lines (only when the model's incidence record carries a held-out error): the held-out range
  from ``pred.dndx_lower`` to ``pred.dndx_upper``, the two bounds a prediction carries (``pred.dndx *
  exp(-pred.dndx_err)`` and ``pred.dndx * exp(+pred.dndx_err)``, each computed on its own, so the bar is asymmetric
  about the line); ``pred.dndx_err`` is the held-out rms of ln(prediction / truth) of the corrected dN/dX. The range
  is an indication of size from a handful of held-out simulations, not an interval of a known distribution;
- points: the DLA dN/dX of Ho, Bird and Garnett (2021) in eighteen redshift bins, with the limits of the 68 per cent
  interval, from ``docs/figures/data/ho21/dndx_all.txt``; and the incidence of Lyman limit systems (tau_912 >= 2)
  of five published samples, from ``docs/figures/data/lls_incidence/lls_lz_literature.dat`` (see "The LLS
  measurements").

**The rows of the measurement's table.** Its source repository has no reader for it. The first author of the paper
confirmed the rows on 2026-11-13 (see PROVENANCE.md beside the file): row 0 the redshift of the bin centre, row 1
dN/dX of DLAs, rows 2 and 3 the lower and the upper limit of the 68 per cent interval, rows 4 and 5 those of the 95
per cent interval. The figure shows the 68 per cent interval.

**The points are drawn in the cosmology of the model line.** dN/dX is a count per unit absorption distance X, and
the measurement computed its Delta X in a flat cosmology with Omega_m = 0.279 (the default of the paper's public
code; see PROVENANCE.md beside the file). The line is per unit X of the cosmology of its parameters (flat, Omega_m =
omegamh2 / hub^2 of the midpoint). Each bin's value and limits are multiplied by Delta X(0.279) / Delta X(line),
with Delta X = integral of (1+z)^2 / sqrt(Omega_m (1+z)^3 + 1 - Omega_m) dz over the bin (the table's centres are
evenly spaced; each bin runs half a spacing either side of its centre): ``figures_common.to_line_cosmology``. The
factors go into the side file. The bands are not converted: each design point is per unit X of its own cosmology
(Omega_m from 0.25 to 0.34), so only the lines are in the measurement's cosmology.

**The LLS measurements.** Each count is of every absorber above the Lyman limit threshold, tau_912 >= 2, which is
N_HI >= 10^17.5 cm^-2: subDLAs and DLAs above it are included. The table holds 18 published rows of l(z) = dN/dz
from six papers (PROVENANCE.md beside it). Five samples are drawn, ``LLS_DRAWN``, so that no absorber is counted
twice: the rows left out are contained in a drawn one or count the same absorbers (the comment above
``LLS_DRAWN``). Each row is drawn at the centre of its redshift bin, with a horizontal bar over the bin, as l(X) =
l(z) Delta z / Delta X over the bin, with Delta X in the cosmology of the line (an l(z) needs no cosmology of its
own); its errors take the same factor, which goes into the side file. In the rows of Prochaska, O'Meara and Worseck
(2010) and of O'Meara et al. (2013) the error printed with the plus sign is the smaller one, the reverse of the
Poisson interval of a count: these rows are drawn with the Poisson interval, the larger error upward. The file
keeps the print. Fumagalli, Fotopoulou and Thomson (2020) classify their absorbers as tau >~ 2, and say that their
catalogue may include some just below tau = 2: their point has its own marker.

**The count above 10^17.5.** The dashed line is the sum over the bins of the model's column-density distribution
``pred.cddf`` (f(N_HI), per unit N_HI and per unit X, at the bin centres ``pred.log_nhi``) of f times the width in
N_HI of the bin's part above 10^17.5. The bins are 0.2 wide in log10 N_HI about their centres, so 10^17.5 falls
inside the bin from 10^17.4 to 10^17.6: that bin counts from 10^17.5 to its upper edge, with f taken constant
within the bin. A bin with no value (NaN: no simulation pair has absorbers there at that redshift) is left out of
the sum; the side file lists those above the threshold. As a cross-check the same count is formed from the class
lines (the LLS line less the part of the distribution from 10^17.2 to 10^17.5, plus the subDLA and the DLA line);
the side file gives the ratio of the two. The partial bin enters both in the same way, so the cross-check tests
that the distribution and the class lines agree, not the choice of a constant f within that bin.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_common as C  # noqa: E402

# the snapshot redshifts inside the grouped-validation domain of the LF+MF model's P1D correction (the correction
# of the incidence, which this figure draws, is not validated at any redshift)
REDSHIFTS = tuple(round(2.2 + 0.2 * i, 1) for i in range(13))  # 2.2, 2.4, ..., 4.6
CLASSES = (
    ("DLA", 2, "BLUE", 2.2, r"$N_\mathrm{HI} \geq 10^{20.3}$"),
    ("subDLA", 1, "ORANGE", 1.3, r"$10^{19}$ to $10^{20.3}$"),
    ("LLS", 0, "AQUA", 1.3, r"$10^{17.2}$ to $10^{19}$"),
)

# The Lyman limit threshold of the LLS measurements, tau_912 >= 2: log10 N_HI >= 17.5 (N_HI in cm^-2)
LLS_THRESHOLD = 17.5
CUMULATIVE_LABEL = "all absorbers with N_HI >= 10^17.5"
# in the values of the band: the three classes of pred.dndx, then the count above 10^17.5
CUMULATIVE_COLUMN = 3

LLS_FILE = ("lls_incidence", "lls_lz_literature.dat")
MEASUREMENT_SOURCE_LLS = (
    "Prochaska, O'Meara and Worseck (2010), doi:10.1088/0004-637X/718/1/392; O'Meara et al. (2013), "
    "doi:10.1088/0004-637X/765/2/137; Fumagalli et al. (2013), doi:10.1088/0004-637X/775/1/78; Crighton et al. "
    "(2019), doi:10.1093/mnras/sty2762; Fumagalli, Fotopoulou and Thomson (2020), doi:10.1093/mnras/staa2388; "
    "values copied from the papers' tables and text: docs/figures/data/lls_incidence/PROVENANCE.md"
)
# The samples drawn, by their key in the table: (label, marker, drawn with the Poisson interval). No absorber is
# counted twice. Left out: the WFC3 and the ACS row of O'Meara et al. (2013) and the row of Ribaudo, Lehner and
# Howk (2011) below z = 2.6, which are inside the combined row of O'Meara et al. (2013) that is drawn; and the rows
# of Ribaudo, Lehner and Howk (2011) above z = 3.5, which count the absorbers of the SDSS-DR7 survey of Prochaska,
# O'Meara and Worseck (2010). "Poisson": the printed plus error of these rows is the smaller one, and the figure
# draws the Poisson interval of the count, the larger error upward (the module docstring).
LLS_DRAWN = {
    "prochaska10_t4": ("Prochaska, O'Meara and Worseck (2010)", "^", True),
    "omeara13_t5_comb": ("O'Meara et al. (2013)", "s", True),
    "fumagalli13_nc": ("Fumagalli et al. (2013)", "D", False),
    "crighton19_corr": ("Crighton et al. (2019)", "v", False),
    "fumagalli20_dr16": ("Fumagalli, Fotopoulou and Thomson (2020)", "o", False),
}
# its own marker: classified as tau >~ 2, and the paper says it may include some just below tau = 2
LLS_HOLLOW = {"fumagalli20_dr16"}
# What each sample counts begins its legend label, as it begins the dashed line's, so that the points are not read
# against the LLS class line
LLS_COUNTS = r"$N_\mathrm{HI} \geq 10^{17.5}$ ($\tau_{912} \geq 2$)"
LLS_COUNTS_OF = {"fumagalli20_dr16": r"$\tau \gtrsim 2$"}


def lls_label(source: str) -> str:
    """The legend label of a sample drawn: what it counts, then the paper."""
    return f"{LLS_COUNTS_OF.get(source, LLS_COUNTS)}, {LLS_DRAWN[source][0]}"


def read_ho21_dndx(path: str):
    """The table of Ho, Bird and Garnett (2021): (redshift of the bin centre, dN/dX of DLAs, lower and upper limit
    of the 68 per cent interval). Rows 4 and 5, the 95 per cent interval, are not returned."""
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[0] != 6:
        raise SystemExit(f"{path}: expected six rows, found shape {a.shape}")
    return a[0], a[1], a[2], a[3]


def bin_edges(centres):
    """The redshift range of each bin: the table's centres are evenly spaced (eighteen bins of 1/6 from 2 to 5), and
    each bin runs half a spacing either side of its centre. Refuses centres that are not evenly spaced.
    """
    step = np.diff(centres)
    if not np.allclose(step, step[0], rtol=0.0, atol=1e-9):
        raise SystemExit("the bin centres of the dN/dX table are not evenly spaced")
    half = step[0] / 2.0
    return centres - half, centres + half


def widths_inside(log_nhi, lo, hi=np.inf):
    """The width in N_HI of the part of each column-density bin that lies between 10^lo and 10^hi (zero for a bin
    outside). The bins are of equal width in log10 N_HI, each half a spacing either side of its centre in
    ``log_nhi``. Refuses centres that are not evenly spaced."""
    log_nhi = np.asarray(log_nhi, dtype=float)
    step = np.diff(log_nhi)
    if not np.allclose(step, step[0], rtol=0.0, atol=1e-9):
        raise SystemExit("the bin centres of the CDDF are not evenly spaced")
    a = np.maximum(log_nhi - step[0] / 2.0, lo)
    b = np.minimum(log_nhi + step[0] / 2.0, hi)
    return np.where(b > a, 10.0**b - 10.0**a, 0.0)


def sum_over(cddf, widths):
    """The sum of f times the width over the bins with a width, on the last axis. A bin with no value (NaN) is left
    out; the sum is NaN when no bin with a width has a value."""
    cddf = np.asarray(cddf, dtype=float)
    counted = (widths > 0) & np.isfinite(cddf)
    total = np.where(counted, cddf * widths, 0.0).sum(axis=-1)
    return np.where(counted.any(axis=-1), total, np.nan)


def cumulative_dndx(cddf, log_nhi, threshold=LLS_THRESHOLD):
    """dN/dX of every absorber with N_HI >= 10^threshold, from a column-density distribution f(N_HI) at the bin
    centres ``log_nhi``: the sum of f times the width of each bin's part above the threshold (the bin that holds the
    threshold counts from it to its upper edge, f taken constant within the bin; a NaN bin is left out). On the last
    axis: a distribution of shape (n_z, Nbin) gives one value per redshift."""
    return sum_over(cddf, widths_inside(log_nhi, threshold))


def cross_check(dndx, cddf, log_nhi, lls_min, threshold=LLS_THRESHOLD):
    """The same count from the class lines: the LLS line (which starts at 10^lls_min) less the part of the
    distribution from 10^lls_min to 10^threshold, plus the subDLA and the DLA line. ``dndx`` has the classes LLS,
    subDLA, DLA on its last axis."""
    dndx = np.asarray(dndx, dtype=float)
    below = sum_over(cddf, widths_inside(log_nhi, lls_min, threshold))
    return dndx[..., 0] - below + dndx[..., 1] + dndx[..., 2]


def dndx_and_cumulative(pred):
    """The values the bands are drawn over: ``pred.dndx`` (LLS, subDLA, DLA), then the count above the threshold."""
    return np.append(pred.dndx, cumulative_dndx(pred.cddf, pred.log_nhi))


def read_lls(path: str):
    """The rows of the LLS table as printed: (source, z_lo, z_hi, l(z), lower error, upper error). Refuses a row
    that is not an l(z) (the table has none: a count per unit X would need its own cosmology)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                raise SystemExit(
                    f"{path}: expected eight tab-separated fields, found {len(fields)}"
                )
            z_lo, z_hi, value, err_lo, err_hi, quantity, _, source = fields
            if quantity != "lz":
                raise SystemExit(
                    f"{path}: a row of {source} is {quantity}; this figure reads l(z) rows only"
                )
            rows.append((source, *(float(v) for v in (z_lo, z_hi, value, err_lo, err_hi))))
    return rows


def lls_points(rows):
    """The rows of the samples drawn, per sample: (z_lo, z_hi, l(X), error below, error above, factor), with l(X) =
    factor * l(z) and factor = Delta z / Delta X(line) over the bin. A sample drawn with the Poisson interval takes
    the larger printed error upward; one whose printed plus error is already the larger is refused.
    """
    points = {source: [] for source in LLS_DRAWN}
    for source, z_lo, z_hi, value, err_lo, err_hi in rows:
        if source not in LLS_DRAWN:
            continue
        below, above = err_lo, err_hi
        if LLS_DRAWN[source][2]:
            if err_hi > err_lo:
                raise SystemExit(
                    f"{source}: the printed plus error is the larger; there is nothing to reverse"
                )
            below, above = err_hi, err_lo
        factor = (z_hi - z_lo) / C.absorption_distance(z_lo, z_hi, C.OMEGA_M_LINE)
        points[source].append((z_lo, z_hi, factor * value, factor * below, factor * above, factor))
    missing = [source for source, rows_of in points.items() if not rows_of]
    if missing:
        raise SystemExit(f"no row of {', '.join(missing)} in the LLS table")
    return {source: np.array(rows_of) for source, rows_of in points.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    C.add_model_arguments(ap)
    C.add_design_arguments(ap)
    ap.add_argument("--out", default=os.path.join(C.FIGURES, "dndx_vs_ho21.png"))
    args = ap.parse_args(argv)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    import hcd_analysis.predictor as hp
    from hcd_analysis.catalog import LOG_NHI_LLS_MIN

    emu, what = C.load_model(args)
    if not what["incidence_record"]:
        raise SystemExit(
            "this model has no incidence MF record: its dN/dX under LF+MF is the LF model's, which is not what "
            "this figure is labelled as"
        )
    design = C.design_points(args, hp.PARAM_NAMES)
    stack = emu.predict_many(C.MIDPOINT, z=list(REDSHIFTS), tau0_amp=1.0, dtau0=0.0)
    facts = dict(
        what,
        figure="dN/dX against redshift",
        parameters=C.MIDPOINT,
        redshifts=list(REDSHIFTS),
        design_table=C.design_source(args),
        measurement_source=C.MEASUREMENT_SOURCE,
        measurement_source_lls=MEASUREMENT_SOURCE_LLS,
    )
    facts["flags_set"] = sorted(n for n, v in stack.flags.items() if v is True)
    # the count above the Lyman limit threshold, and its cross-check against the class lines
    cumulative = cumulative_dndx(stack.cddf, stack.log_nhi)
    ratio = cumulative / cross_check(stack.dndx, stack.cddf, stack.log_nhi, LOG_NHI_LLS_MIN)
    above = widths_inside(stack.log_nhi, LLS_THRESHOLD) > 0
    holds = above & (widths_inside(stack.log_nhi, -np.inf, LLS_THRESHOLD) > 0)
    absent = {
        f"{z:g}": [float(v) for v in stack.log_nhi[above & ~np.isfinite(row)]]
        for z, row in zip(REDSHIFTS, stack.cddf)
    }
    facts["cumulative"] = dict(
        label=f"model (LF+MF): {CUMULATIVE_LABEL}",
        threshold_log_nhi=LLS_THRESHOLD,
        method="sum over the bins of pred.cddf of f times the width in N_HI of the bin's part above the threshold; "
        "the bin that holds the threshold counts from it to its upper edge, f taken constant within the bin; a bin "
        "with no value (NaN) is left out",
        bin_holding_the_threshold=[float(v) for v in stack.log_nhi[holds]],
        bins_without_value_above_the_threshold=absent,
        cross_check="the LLS line less the sum of pred.cddf from 10^17.2 to 10^17.5, plus the subDLA and the DLA "
        "line; ratio = the count above 10^17.5 / this sum",
        cross_check_ratio_range=[float(np.nanmin(ratio)), float(np.nanmax(ratio))],
    )
    print(f"the count above 10^{LLS_THRESHOLD:g} against the class lines: ratio {np.nanmin(ratio):.4f} to "
          f"{np.nanmax(ratio):.4f}")  # fmt: skip
    bands = None
    if design is not None:
        lo, hi, all_flagged, all_refused = [], [], {}, {}
        for z in REDSHIFTS:
            drawn = C.flags_set(emu.predict(C.MIDPOINT, z=z, tau0_amp=1.0, dtau0=0.0))
            a, b, flagged, refused = C.envelope(emu, design, z, dndx_and_cumulative, drawn)
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
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    z = np.array(REDSHIFTS)
    drew_error = False
    for name, col, colour, width, span in CLASSES:
        colour = getattr(C, colour)
        if bands is not None:
            ax.fill_between(
                z, bands[0][:, col], bands[1][:, col], color=colour, alpha=0.2, lw=0, zorder=1
            )
        ax.plot(
            z, stack.dndx[:, col], color=colour, lw=width, zorder=2,
            label=f"model (LF+MF): {name}, {span}",
        )  # fmt: skip
        # the held-out range of the corrected dN/dX, as a bar on the model line: from dndx_lower to dndx_upper, the
        # two bounds the prediction carries (each computed on its own from the held-out rms of ln(prediction / truth),
        # so the bar is asymmetric). Nothing is transformed here. Drawn only where the model carries an estimate.
        lower = None if stack.dndx_lower is None else stack.dndx_lower[:, col]
        if lower is not None and np.any(np.isfinite(lower)):
            ok = np.isfinite(lower)
            value = stack.dndx[ok, col]
            lower, upper = lower[ok], stack.dndx_upper[ok, col]
            ax.errorbar(
                z[ok], value, yerr=[value - lower, upper - value], fmt="none", ecolor=colour, elinewidth=1.1,
                capsize=2.2, capthick=1.1, zorder=2, label="held-out range of the model" if not drew_error else None,
            )  # fmt: skip
            drew_error = True
        ax.annotate(
            name,
            (z[-1], stack.dndx[-1, col]),
            xytext=(5, -3 if name == "LLS" else 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )
    facts["heldout_error"] = (
        "drawn on the model lines (LF+MF): the held-out range from pred.dndx_lower to pred.dndx_upper "
        "(pred.dndx * exp(-pred.dndx_err) and pred.dndx * exp(+pred.dndx_err), each bound computed on its own: "
        "asymmetric); pred.dndx_err is the held-out rms of ln(prediction / truth)"
        if drew_error
        else "not drawn: the model carries no held-out error at these redshifts"
    )
    # the count above 10^17.5: dashed as well as yellow, since yellow and orange are the palette's weakest pair
    if bands is not None:
        C.band(
            ax, z, bands[0][:, CUMULATIVE_COLUMN], bands[1][:, CUMULATIVE_COLUMN], color=C.YELLOW, alpha=0.2,
            lw=0, zorder=1,
        )  # fmt: skip
    C.line(
        ax, z, cumulative, color=C.YELLOW, lw=1.8, ls=(0, (5, 2)), zorder=2,
        label=r"model (LF+MF): all absorbers with $N_\mathrm{HI} \geq 10^{17.5}$",
    )  # fmt: skip
    ax.annotate(
        r"$\geq 10^{17.5}$", (z[-1], cumulative[-1]), xytext=(5, 10), textcoords="offset points",
        va="center", fontsize=8.5,
    )  # fmt: skip
    zm, dndx, lo68, hi68 = read_ho21_dndx(C.data_file("ho21", "dndx_all.txt"))
    # each bin's value and limits times its factor to the line's cosmology (module docstring)
    z_lo, z_hi = bin_edges(zm)
    k = np.array([C.to_line_cosmology(a, b, C.OMEGA_M_OF["ho21"]) for a, b in zip(z_lo, z_hi)])
    dndx, lo68, hi68 = k * dndx, k * lo68, k * hi68
    ax.errorbar(
        zm, dndx, yerr=[dndx - lo68, hi68 - dndx], fmt="o", ms=3.6, color=C.INK, ecolor=C.INK_SECONDARY,
        elinewidth=0.8, mec=C.SURFACE, mew=0.5, zorder=3, label="DLA, Ho, Bird and Garnett (2021), 68 per cent interval",
    )  # fmt: skip
    facts["measurement"] = (
        "DLA dN/dX of Ho, Bird and Garnett (2021), 68 per cent interval: docs/figures/data/ho21/dndx_all.txt, "
        f"{zm.size} redshift bins"
    )
    # the LLS samples, as l(X) in the line's cosmology (module docstring)
    points = lls_points(read_lls(C.data_file(*LLS_FILE)))
    for source, (_, marker, _) in LLS_DRAWN.items():
        p = points[source]
        centre = (p[:, 0] + p[:, 1]) / 2.0
        hollow = source in LLS_HOLLOW
        ax.errorbar(
            centre, p[:, 2], xerr=[centre - p[:, 0], p[:, 1] - centre], yerr=[p[:, 3], p[:, 4]], fmt=marker,
            ms=4.2, color=C.INK, ecolor=C.INK_SECONDARY, elinewidth=0.8, mfc=C.SURFACE if hollow else C.INK,
            mec=C.INK if hollow else C.SURFACE, mew=0.9 if hollow else 0.5, zorder=3, label=lls_label(source),
        )  # fmt: skip
    facts["lls_measurements"] = dict(
        file="docs/figures/data/lls_incidence/lls_lz_literature.dat",
        counts="every absorber with tau_912 >= 2 (N_HI >= 10^17.5), per unit redshift as published; drawn per unit "
        "X: l(X) = l(z) Delta z / Delta X(line) over each row's redshift bin, errors times the same factor",
        drawn={source: LLS_DRAWN[source][0] for source in LLS_DRAWN},
        not_drawn="the other rows of the table: contained in a drawn row, or counting the same absorbers",
        poisson_interval=[source for source in LLS_DRAWN if LLS_DRAWN[source][2]],
        poisson_interval_means="drawn with the larger printed error upward; the table keeps the print",
        own_marker={
            "fumagalli20_dr16": "classified as tau >~ 2; may include systems just below tau = 2 (the paper)"
        },
    )
    facts["cosmology"] = dict(
        C.cosmology_facts(),
        factors_ho21=[
            {"z_range": [float(a), float(b)], "factor": float(f)} for a, b, f in zip(z_lo, z_hi, k)
        ],
        factors_lls=[
            {"source": source, "z_range": [float(row[0]), float(row[1])], "factor": float(row[5])}
            for source in LLS_DRAWN
            for row in points[source]
        ],
        lls_conversion="l(z) to l(X): Delta z / Delta X(line Omega_m) over the row's bin",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda v, _: f"{v:g}" if f"{v:e}"[0] in "25" else "")
    )
    ax.set_xlim(1.85, 5.5)
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
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2, fontsize=7.5)
    C.save(fig, args.out, facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
