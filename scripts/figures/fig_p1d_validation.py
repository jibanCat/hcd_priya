#!/usr/bin/env python3
"""Figures: the held-out prediction residual of the class P1D, drawn from a model release's own P1D held-out record.

    python scripts/figures/fig_p1d_validation.py (--published | --release RELEASE_DIR SHA256)
        [--row mixture|clean|LLS|subDLA|DLA] [--out-dir docs/figures]

Three figures. They show an empirical validation diagnostic, not a calibrated uncertainty: for simulations the model
was made without, how far the prediction of a class spectrum lies from the simulation.

- ``p1d_validation_lf.png`` and ``p1d_validation_lfmf.png``, one per fidelity, drawn from the record alone. Five
  panels: the four classes and the mixture (the class spectra mixed with each held-out simulation's own class
  fractions). In each: e = the rms of ln(prediction / truth) over the held-out simulations and the mean-flux rungs,
  on a logarithmic axis against the wavenumber, one line per snapshot redshift (the colour, light to dark, is the
  redshift). Redshifts above 4.6, outside the validated range, are dashed. The two largest modes of the box, which
  training gave less weight, carry an open marker. Where the record holds a sensitivity table (the same table
  without one named simulation), its line at the highest redshift is dotted, in a second hue.
- ``p1d_validation_by_simulation.png``, drawn from the record's evidence file: the residual itself,
  d = ln(prediction / truth) of the mixture, one line per held-out simulation, at four
  redshifts, for LF (top row) and LF+MF (bottom row), at the mean-flux rung nearest the mean flux of Kim et al.
  (2007). Two simulations are drawn in a hue: the one the record's sensitivity table leaves out, and the one with
  the largest residual of the mixture at redshifts up to 4.6. The others are grey. The vertical axis is linear
  within 0.05 and logarithmic beyond. With ``--row`` the same figure is drawn for one class in place of the
  mixture, into ``p1d_validation_by_simulation_<row>.png``; the two simulations in a hue stay the same.

The wavenumbers: the modes of the box are converted to s/km at each redshift with hub and omegamh2 of the midpoint
of the parameter box (``figures_common.MIDPOINT``), as in the other model figures. Each simulation has its own
cosmology, so its own wavenumbers differ from these by a few per cent.

Each image has a side file (.json) that says what it was drawn from; for the two tables it also holds, per row, the
median and the largest e in the regimes of the record's summary. The figures need a model release that carries a
P1D held-out record; the third also needs the record's evidence file.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_common as C  # noqa: E402

ROW_TITLES = ("clean", "LLS", "subDLA", "DLA", "mixture of the classes")
FIDELITY_LABEL = {"lf": "LF", "lfmf": "LF+MF"}
Z_VALIDATED_MAX = 4.6
Z_BY_SIMULATION = (2.2, 3.0, 4.0, 5.4)
MIXTURE_ROW = 4
# the rows of the figure per simulation: (the value of --row, what the title calls it)
ROW_CHOICES = ("clean", "LLS", "subDLA", "DLA", "mixture")
ROW_NOUN = {
    "clean": "the clean class spectrum",
    "LLS": "the LLS class spectrum",
    "subDLA": "the subDLA class spectrum",
    "DLA": "the DLA class spectrum",
    "mixture": "the mixture",
}
SYMLOG_LINEAR = (
    0.05  # the residual axis of the figure per simulation is linear within this, logarithmic beyond
)
# the redshift as one hue from light to dark (a magnitude, so no second hue)
Z_COLOURS = ("#c5daf6", C.BLUE, "#0d2f61")
GREY_STYLES = ("-", "--", "-.", ":")


def mode_wavenumbers(n_modes: int, z: float) -> np.ndarray:
    """The modes of the box in s/km at ``z``, for hub and omegamh2 of the midpoint of the parameter box."""
    from hcd_analysis.emulator import kcoord
    from hcd_analysis.emulator.schema import L_BOX_HMPC

    k_com = 2.0 * np.pi * np.arange(1, n_modes + 1) / L_BOX_HMPC
    return np.asarray(
        kcoord.k_skm_from_kcom(k_com, float(z), C.MIDPOINT["hub"], C.MIDPOINT["omegamh2"]),
        dtype=np.float64,
    )


def row_summary(e, z) -> dict:
    """Per row: the median and the largest e in each regime of the record's summary (``p1d_heldout.summarise``)."""
    from hcd_analysis.emulator import p1d_heldout as PH

    out = {}
    for i, name in enumerate(PH.ROWS):
        s = PH.summarise(np.asarray(e)[:, i : i + 1, :], z)
        out[name] = {
            regime: {q: round(float(v), 5) for q, v in stats.items()} for regime, stats in s.items()
        }
    return out


def central_rung(target_F, z) -> int:
    """The rung whose mean flux is nearest the fit of Kim et al. (2007) at ``z``: tau = -ln(target_F) nearest
    ``kim_tau0(z)``."""
    from hcd_analysis.emulator.mean_flux_model import kim_tau0

    amplitude = -np.log(np.asarray(target_F, dtype=np.float64)) / float(kim_tau0(float(z)))
    return int(np.nanargmin(np.abs(amplitude - 1.0)))


def _residual_axis(ax, plt):
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def draw_table(plt, held, fid: str, out: str, model: dict) -> None:
    """One fidelity's table: five panels, a line per redshift."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    from hcd_analysis.emulator import p1d_heldout as PH

    rec = held.record
    z, e = np.asarray(held.z[fid]), np.asarray(held.e[fid])
    cmap = LinearSegmentedColormap.from_list("redshift", Z_COLOURS)
    norm = Normalize(vmin=2.0, vmax=5.4)
    sens = rec["sensitivity"] if rec["sensitivity"]["fidelity"] == fid else None
    fig, axes = plt.subplots(
        2, 3, figsize=(11.0, 6.6), sharex=True, sharey=True, constrained_layout=True
    )
    panels = list(axes.ravel())
    note = panels.pop()  # the sixth cell holds the key
    for ax, title, row in zip(panels, ROW_TITLES, range(len(PH.ROWS))):
        for iz, zz in enumerate(z):
            k = mode_wavenumbers(held.n_modes, zz)
            y = e[iz, row]
            colour = cmap(norm(zz))
            ls = "-" if zz <= Z_VALIDATED_MAX + 1e-9 else "--"
            ax.plot(k, y, color=colour, ls=ls, lw=1.1, zorder=2)
            ax.plot(
                k[: PH.N_LARGEST_MODES], y[: PH.N_LARGEST_MODES], ls="", marker="o", ms=3.0,
                mfc=C.SURFACE, mec=colour, mew=0.9, zorder=3,
            )  # fmt: skip
        if sens is not None:
            k = mode_wavenumbers(held.n_modes, z[-1])
            ax.plot(k, held.e_sens[-1, row], color=C.ORANGE, ls=":", lw=1.6, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _residual_axis(ax, plt)
        ax.set_title(title, loc="left")
    for ax in axes[-1, :2]:
        ax.set_xlabel("k (s/km)")
    axes[0, 2].set_xlabel("k (s/km)")
    axes[0, 2].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("e = rms of ln(prediction / truth)")
    note.axis("off")
    handles = [
        Line2D([], [], color=cmap(0.45), lw=1.1, label=f"redshifts up to {Z_VALIDATED_MAX:g}"),
        Line2D(
            [], [], color=cmap(0.95), lw=1.1, ls="--",
            label=f"redshifts above {Z_VALIDATED_MAX:g} (outside the validated range)",
        ),  # fmt: skip
        Line2D(
            [], [], color=cmap(0.6), ls="", marker="o", ms=3.4, mfc=C.SURFACE, mew=0.9,
            label=f"the {PH.N_LARGEST_MODES} largest modes of the box (less weight in training)",
        ),  # fmt: skip
    ]
    if sens is not None:
        handles.append(
            Line2D(
                [], [], color=C.ORANGE, ls=":", lw=1.6,
                label=f"sensitivity test at z = {z[-1]:g}: the same table\nwithout {PH.design_point(sens['without_sim'])}",
            )  # fmt: skip
        )
    note.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=2.6)
    note.text(
        0.0, 0.38,
        "Held-out prediction residual:\nan empirical validation diagnostic,\nnot a calibrated uncertainty.\n\n"
        "e is on the natural logarithm, not a percentage:\non a spectrum the range is the value times\n"
        "exp(-e) to exp(+e). e = 0.01: times 0.990 to 1.010;\ne = 0.7: times 0.50 to 2.01.\n\n"
        f"{int(held.n[fid].max())} held-out simulations. Wavenumbers of the\nmodes at the midpoint of the parameter box.",
        transform=note.transAxes, fontsize=8.5, color=C.INK_SECONDARY, va="top", ha="left",
    )  # fmt: skip
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.82, pad=0.015)
    bar.set_label("redshift")
    bar.set_ticks([2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2])
    fig.suptitle(
        f"Held-out prediction residual of the class P1D, {FIDELITY_LABEL[fid]}: rms of ln(prediction / truth)\n"
        f"{rec[fid]['what'][0].upper()}{rec[fid]['what'][1:]}.",
        x=0.01, ha="left", fontsize=10.0,
    )  # fmt: skip
    facts = {
        "figure": "the held-out prediction residual of the class P1D: e against k, one line per redshift",
        "vertical_axis": "e, the rms of ln(prediction / truth): a number on the natural logarithm, not a percentage; "
        "on a spectrum it is the range from the value times exp(-e) to the value times exp(+e)",
        "what_it_is": "an empirical validation diagnostic, not a calibrated uncertainty",
        "fidelity": FIDELITY_LABEL[fid],
        "held_out": rec[fid]["what"],
        "n_held_out_simulations": int(held.n[fid].max()),
        "redshifts": [float(v) for v in z],
        "record_sha256": held.record_sha256,
        "record_kind": rec["kind"],
        "wavenumbers": "the modes of the box in s/km at each redshift, for hub and omegamh2 of the midpoint",
        "midpoint": {k: C.MIDPOINT[k] for k in ("hub", "omegamh2")},
        "dashed": f"redshifts above {Z_VALIDATED_MAX:g}",
        "open_markers": f"the {PH.N_LARGEST_MODES} largest modes of the box",
        "summary": row_summary(e, z),
        "model": model,
    }
    if sens is not None:
        facts["sensitivity"] = {
            "dotted": f"the table without {PH.design_point(sens['without_sim'])}, at z = {float(z[-1]):g}",
            "statement": sens["statement"],
            "summary": row_summary(held.e_sens, z),
        }
    C.save(fig, out, facts)
    plt.close(fig)


def draw_by_simulation(
    plt, held, record_dir: str, out: str, model: dict, row: str = "mixture"
) -> None:
    """The residual of one row (the mixture, or one class) per held-out simulation, at four redshifts, for both
    fidelities."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    from hcd_analysis.emulator import p1d_heldout as PH
    from hcd_analysis.emulator import safe_io as S

    rec = held.record
    i_row = ROW_CHOICES.index(row)
    arrays, _ = S.read_table_h5(os.path.join(record_dir, PH.TABLES_FILE), PH.table_datasets(rec))
    points = sorted({PH.design_point(s) for s in rec["lf"]["sims"]})
    left_out = PH.design_point(rec["sensitivity"]["without_sim"])
    # the simulation with the largest LF residual of the mixture at redshifts up to 4.6
    z_lf = np.asarray(rec["lf"]["z"])
    sel = z_lf <= Z_VALIDATED_MAX + 1e-9
    d_lf = arrays["lf_d"].astype(np.float64)[:, :, :, MIXTURE_ROW, :][
        :, sel
    ]  # (simulation, redshift, rung, mode)
    with np.errstate(invalid="ignore"):
        rms = np.sqrt(np.nanmean(d_lf**2, axis=(1, 2, 3)))
    largest = PH.design_point(rec["lf"]["sims"][int(np.nanargmax(rms))])
    look = {}
    greys = iter(GREY_STYLES * 3)
    for p in points:
        if p == left_out:
            look[p] = dict(color=C.BLUE, ls="-", lw=1.5, zorder=4)
        elif p == largest:
            look[p] = dict(color=C.ORANGE, ls="-", lw=1.5, zorder=3)
        else:
            look[p] = dict(color=C.INK_SECONDARY, ls=next(greys), lw=0.9, zorder=2, alpha=0.85)
    fig, axes = plt.subplots(
        2, len(Z_BY_SIMULATION), figsize=(11.0, 6.0), sharex=True, sharey=True, constrained_layout=True
    )  # fmt: skip
    drawn, rungs = {}, {}
    for r, fid in enumerate(PH.FIDELITIES):
        z = np.asarray(rec[fid]["z"])
        d = arrays[f"{fid}_d"].astype(np.float64)
        for c, zz in enumerate(Z_BY_SIMULATION):
            ax = axes[r, c]
            hit = np.where(np.abs(z - zz) <= 1e-6)[0]
            ax.axhline(0.0, color=C.GRID, lw=0.8, zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=SYMLOG_LINEAR, linscale=1.0)
            ax.set_title(f"{FIDELITY_LABEL[fid]}, z = {zz:g}", loc="left")
            if hit.size != 1:
                ax.text(
                    0.5,
                    0.5,
                    "no estimate",
                    transform=ax.transAxes,
                    ha="center",
                    color=C.INK_SECONDARY,
                )
                continue
            k = mode_wavenumbers(held.n_modes, zz)
            for i, sim in enumerate(rec[fid]["sims"]):
                tf = arrays[f"{fid}_target_F"][i, hit[0]]
                if not np.isfinite(tf).any():
                    continue  # this simulation has no snapshot at this redshift
                rung = central_rung(tf, zz)
                y = d[i, hit[0], rung, i_row]
                if not np.isfinite(y).any():
                    continue  # no residual of this simulation at this redshift
                rungs.setdefault(FIDELITY_LABEL[fid], {}).setdefault(f"{zz:g}", []).append(rung)
                p = PH.design_point(sim)
                C.line(ax, k, y, **look[p])
                drawn[p] = look[p]
    ticks = [-1, -0.3, -0.1, -0.05, 0, 0.05, 0.1, 0.3]
    for ax in axes.ravel():
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    for ax in axes[-1]:
        ax.set_xlabel("k (s/km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("d = ln(prediction / truth)")

    def text(p):
        if p == left_out:
            return f"{p} (the record's sensitivity table leaves it out)"
        if p == largest:
            return (
                f"{p} (the largest residual of the mixture at redshifts up to {Z_VALIDATED_MAX:g})"
            )
        return p

    handles = [
        Line2D([], [], label=text(p), **{k: v for k, v in drawn[p].items() if k != "zorder"})
        for p in points
        if p in drawn
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, handlelength=2.8)
    fig.suptitle(
        f"Residual of {ROW_NOUN[row]} per held-out simulation: ln(prediction / truth), at the mean flux nearest "
        "Kim et al. (2007).\nAn empirical validation diagnostic, not a calibrated uncertainty.",
        x=0.01, ha="left", fontsize=10.0,
    )  # fmt: skip
    C.save(
        fig,
        out,
        {
            "figure": f"the residual of {ROW_NOUN[row]} per held-out simulation: ln(prediction / truth) against k",
            "row": row,
            "what_it_is": "an empirical validation diagnostic, not a calibrated uncertainty",
            "rows": ["LF", "LF+MF"],
            "redshifts": list(Z_BY_SIMULATION),
            "simulations": points,
            "in_the_first_hue": {
                "design_point": left_out,
                "why": "the record's sensitivity table leaves it out",
            },
            "in_the_second_hue": {
                "design_point": largest,
                "why": f"the largest rms of the LF residual of the mixture at redshifts up to {Z_VALIDATED_MAX:g}",
            },
            "mean_flux": "the rung whose mean flux is nearest the fit of Kim et al. (2007) at the redshift",
            "rung_indices_drawn": rungs,
            "vertical_axis": f"d = ln(prediction / truth), not a percentage; linear within {SYMLOG_LINEAR:g}, "
            "logarithmic beyond",
            "wavenumbers": "the modes of the box in s/km at each redshift, for hub and omegamh2 of the midpoint",
            "record_sha256": held.record_sha256,
            "evidence_file_sha256": rec["tables"]["sha256"],
            "model": model,
        },
    )
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    C.add_model_arguments(ap)
    ap.add_argument(
        "--row",
        choices=ROW_CHOICES,
        default="mixture",
        help="the row of the figure per simulation: the mixture (the default) or one class",
    )
    ap.add_argument("--out-dir", default=C.FIGURES, help="where the three images are written")
    args = ap.parse_args(argv)
    import hcd_analysis.predictor as hp

    emu, model = C.load_model(args)
    held, record_dir = hp.p1d_heldout_record(emu)
    if held is None:
        raise SystemExit(
            "this model carries no P1D held-out record: there is no held-out prediction residual to draw"
        )
    import matplotlib.pyplot as plt

    C.style(plt)
    for fid in ("lf", "lfmf"):
        draw_table(plt, held, fid, os.path.join(args.out_dir, f"p1d_validation_{fid}.png"), model)
    if held.record["tables"] is None:
        print(
            "the record was shipped without its evidence file: the figure per simulation is not drawn"
        )
        return 0
    name = "p1d_validation_by_simulation" + ("" if args.row == "mixture" else f"_{args.row}")
    draw_by_simulation(
        plt, held, record_dir, os.path.join(args.out_dir, name + ".png"), model, row=args.row
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
