"""Shared by the figure scripts of this folder: where the committed figure data are, the check of each data file
against the digest recorded beside it, the parameters every model curve is drawn at, how a model is loaded, and the
look of the figures.

Each figure script draws one figure from the installed package, the data files committed under
``docs/figures/data`` and, for the two model figures, a model that holds an incidence record. The bands are drawn
over the 60 design points of the committed table ``priya_lf_design_60.csv``. Two more inputs are optional files
that you supply; they are read without a digest check: another table of design points (``--design``) and, for the
dN/dX figure, the measurement's table (``--ho21-dndx``).
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "docs", "figures", "data")
FIGURES = os.path.join(REPO, "docs", "figures")

# The nine PRIYA parameters every model curve is drawn at, written out. These values are the midpoint of the range
# each parameter was trained on: not a fiducial cosmology, not a best fit and not a simulation.
MIDPOINT = {
    "ns": 0.925,
    "Ap": 1.9e-9,
    "herei": 4.0,
    "heref": 2.7,
    "alphaq": 2.15,
    "hub": 0.70,
    "omegamh2": 0.143,
    "hireionz": 7.25,
    "bhfeedback": 0.05,
}
MIDPOINT_TEXT = ", ".join(f"{name} = {value:g}" for name, value in MIDPOINT.items())

# The look: one hue for the model (a line, and a lighter band of the same hue), dark ink for measured points, a
# recessive grid, text in ink and never in a series colour. The three hues are the first three of a palette that
# was checked for colour-blind separation in every pairing.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SHADE = "#efeeea"


def data_file(*parts: str) -> str:
    """The path of a committed figure-data file, after checking its sha256 against ``SHA256SUMS`` beside the data."""
    rel = "/".join(parts)
    path = os.path.join(DATA, *parts)
    sums = {}
    with open(os.path.join(DATA, "SHA256SUMS"), encoding="utf-8") as fh:
        for line in fh:
            digest, name = line.split()
            sums[name] = digest
    with open(path, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if sums.get(rel) != got:
        raise SystemExit(
            f"{rel}: sha256 {got} is not the one recorded in docs/figures/data/SHA256SUMS"
        )
    return path


def add_model_arguments(parser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--published",
        action="store_true",
        help="draw the model of the installed model package (hp.load_published())",
    )
    group.add_argument(
        "--release",
        nargs=2,
        metavar=("RELEASE_DIR", "SHA256"),
        help="draw a release whose files you hold (hp.load_release)",
    )


def load_model(args):
    """The model the figure is drawn for, and one line that says which (for the figure's side file)."""
    import hcd_analysis.predictor as hp

    emu = hp.load_published() if args.published else hp.load_release(*args.release)
    release = emu.provenance.get("release", {})
    what = {
        "release_format": release.get("format"),
        "synthetic_build": bool(release.get("synthetic_build")),
        "incidence_record": "incidence_mf" in emu.provenance,
    }
    if what["incidence_record"]:
        what["incidence_hr_simulations"] = len(emu.provenance["incidence_mf"]["train_sims"])
    return emu, what


DESIGN_FILE = "priya_lf_design_60.csv"  # the 60 low-resolution design points of the PRIYA suite


def add_design_arguments(parser) -> None:
    parser.add_argument(
        "--design",
        help="another table of design points for the band (nine columns named like the parameters), in place of "
        f"docs/figures/data/{DESIGN_FILE}; it is read without a digest check",
    )
    parser.add_argument(
        "--no-band", action="store_true", help="draw no band over the design points"
    )


def design_points(args, names):
    """The design points of the band: None with ``--no-band``; those of ``--design FILE`` if given; otherwise the 60
    points of the committed table, after its digest check."""
    if args.no_band:
        if args.design:
            raise SystemExit("--no-band and --design contradict each other")
        return None
    return read_design(args.design if args.design else data_file(DESIGN_FILE), names)


def read_design(path: str, names) -> list[dict]:
    """A table of design points: comment lines starting with '#', one header line with the nine parameter names in
    the model's order, then one row of nine numbers per point."""
    import csv

    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(ln for ln in fh if not ln.startswith("#")) if r]
    if not rows or rows[0] != list(names):
        raise SystemExit(f"{path}: the header must be {','.join(names)}")
    return [dict(zip(names, (float(v) for v in r))) for r in rows[1:]]


def flags_set(pred) -> list:
    """The names of the flags that are True in a prediction, sorted."""
    return sorted(name for name, value in pred.flags.items() if value is True)


def envelope(emu, design, z, attribute, baseline=()):
    """The lowest and the highest ``pred.<attribute>`` over the design points at one redshift (LF+MF, tau0_amp = 1,
    dtau0 = 0), ignoring NaN bins, with the points the predictor flagged and those it refused.

    A point is reported as flagged when the predictor sets a flag for it that the drawn line does not have
    (``baseline``: the flags of the prediction at the script's own parameters at this redshift). Every such flag is
    reported, whatever its name. No point is dropped silently: a refused point is reported and is not in the band.
    """
    import warnings

    import numpy as np

    import hcd_analysis.predictor as hp

    values, flagged, refused = [], [], []
    for i, params in enumerate(design):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = emu.predict(params, z=z, tau0_amp=1.0, dtau0=0.0)
        except hp.PredictorInputError as exc:
            refused.append((i, str(exc)[:100]))
            continue
        values.append(np.asarray(getattr(pred, attribute), dtype=float))
        hit = sorted(set(flags_set(pred)) - set(baseline))
        if hit:
            flagged.append((i, hit))
    if not values:
        raise SystemExit(
            f"z = {z:g}: the predictor refused every design point, so there is no band to draw"
        )
    values = np.array(values)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # a bin that is NaN for every point stays NaN
        return np.nanmin(values, axis=0), np.nanmax(values, axis=0), flagged, refused


def flag_counts(flagged) -> dict:
    """{flag name: the number of design points that have it}, from the ``flagged`` list of ``envelope``."""
    counts = {}
    for _, hit in flagged:
        for name in hit:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def line(ax, x, y, **kw):
    """Plot ``y`` against ``x`` with every NaN left in, so that an absent bin breaks the line. Compressing the
    arrays first would join the two neighbours of an absent bin with a straight segment."""
    return ax.plot(x, y, **kw)


def band(ax, x, lo, hi, **kw):
    """Fill between ``lo`` and ``hi`` only where both are finite: an absent bin is a gap in the band."""
    import numpy as np

    ok = np.isfinite(lo) & np.isfinite(hi)
    return ax.fill_between(x, lo, hi, where=ok, **kw)


def style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
        }
    )


def save(fig, out: str, facts: dict) -> None:
    """Write the image and, beside it, a small JSON file that says what it was drawn from."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote the figure:", out)
