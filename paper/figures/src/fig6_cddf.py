"""Figure 4 of the JOSS paper (the CDDF figure): the column-density distribution f(N_HI), with DLA, subDLA and LLS data.

Model curves as in the public figure docs/figures/cddf_vs_ho21.png of the code repository, from the same arrays;
subDLA and LLS points from the public DLA data collection are added:
  - line: f(N_HI) of the model at the snapshots 2.2, 2.8 and 3.4, at the midpoint of the training range; a bin
    without support is left out, not drawn as zero;
  - band: the range over the 60 design points (not an uncertainty). The held-out range of the CDDF is carried by
    each prediction but not drawn here, as on the public page;
Measured points (public collection, read as its own reader reads them; see _dla_data.py):
  - DLAs: Ho, Bird and Garnett (2021) in the redshift bin of each panel, 68 per cent interval; bins with f = 0 left out;
  - subDLAs: Zafar et al. (2013), z 1.5 to 5, drawn in every panel;
  - 19 < log N_HI < 20.3: O'Meara et al. (2007), mean z 3.1, drawn in the panel 3 < z < 4;
  - LLS: Prochaska, O'Meara and Worseck (2010), an allowed region at z 3.7, drawn in the panel 3 < z < 4.
Which panel each sample goes in is this figure's choice, by its redshift.
The dashed lines mark the class edges of the package (paper/facts.yml).
Bottom row: each measurement over the model, with
the model read at the measurement's column density by a straight line in (log N_HI, log f) between the model's
bins; the band is the range over the design points over the model. Measurements where the model has no support
are left out. A lower end of zero (no detection) is drawn to the bottom of the panel.

Run from the repository root:  python3 paper/figures/src/fig6_cddf.py
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import _dla_data  # noqa: E402
import _model_arrays  # noqa: E402
from _style import BAND, INK, TEAL, TEAL_LIGHT, style  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig6_cddf.png"
PANELS = [("z2p2", "2-2.5", r"$2 < z < 2.5$", r"$z = 2.2$"),
          ("z2p8", "2.5-3", r"$2.5 < z < 3$", r"$z = 2.8$"),
          ("z3p4", "3-4", r"$3 < z < 4$", r"$z = 3.4$")]


def model_at(x, f, logn):
    """The model at the column densities logn: a straight line in (log N_HI, log f) between supported bins."""
    ok = np.isfinite(f) & (f > 0)
    out = 10 ** np.interp(logn, x[ok], np.log10(f[ok]), left=np.nan, right=np.nan)
    return out


def main():
    m = _model_arrays.load()
    if m is None:
        if OUT.exists():
            print("KEPT", OUT, f"({_model_arrays.WHY})")
            return 0
        raise SystemExit(f"no model arrays ({_model_arrays.WHY}) and no image to keep")
    facts = yaml.safe_load((HERE.parents[1] / "facts.yml").read_text())["facts"]
    edges = [float(facts[k]["value"]) for k in ("logN_lls_min", "logN_subdla_min", "logN_dla_min")]
    style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, NullFormatter

    zafar, omeara, pro = _dla_data.zafar13(), _dla_data.omeara07(), _dla_data.prochaska10()
    fig, axes = plt.subplots(2, 3, figsize=(5.4, 3.5), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    for (ax, rx), (tag, zbin, measured, modelz) in zip(axes.T, PANELS):
        row = int(np.argmin(abs(m["z"] - float(tag[1:].replace("p", ".")))))  # z2p2 -> the row at z = 2.2
        assert abs(m["z"][row] - float(tag[1:].replace("p", "."))) < 1e-9
        x, f = m["log_nhi"], m["cddf"][row]
        ok = np.isfinite(f)
        bmin, bmax = m["cddf_band_min"][row], m["cddf_band_max"][row]
        ax.fill_between(x, bmin, bmax, where=ok, color=TEAL_LIGHT, lw=0, zorder=1)
        ax.plot(np.where(ok, x, np.nan), f, color=TEAL, lw=1.2, zorder=3)
        rx.fill_between(x, bmin / f, bmax / f, where=ok, color=TEAL_LIGHT, lw=0, zorder=1)
        rx.axhline(1, color=TEAL, lw=1.0, zorder=2)

        def points(axm, axr, logn, v, lo, hi, xerr=None, **kw):
            axm.errorbar(logn, v, xerr=xerr, yerr=[v - lo, hi - v], **kw)
            mod = model_at(x, f, logn)
            keep = np.isfinite(mod)
            xe = None if xerr is None else [xerr[0][keep], xerr[1][keep]]
            r, rlo, rhi = v[keep] / mod[keep], lo[keep] / mod[keep], hi[keep] / mod[keep]
            axr.errorbar(logn[keep], r, xerr=xe, yerr=[r - np.maximum(rlo, 1e-3), rhi - r], **kw)

        n, v, lo, hi = _dla_data.ho21_cddf(zbin)
        points(ax, rx, np.log10(n), v, lo, hi, fmt="o", ms=1.9, color=INK, elinewidth=0.6, capsize=1.0,
               capthick=0.6, zorder=6)
        n, v, lo, hi, nlo, nhi = zafar
        ln = np.log10(n)
        points(ax, rx, ln, v, lo, hi, xerr=[ln - np.log10(nlo), np.log10(nhi) - ln], fmt="s", ms=2.6, mfc="white",
               mec=INK, mew=0.6, color=INK, elinewidth=0.6, capsize=1.0, capthick=0.6, zorder=5)
        if zbin == "3-4":
            n, v, lo, hi, nlo, nhi = omeara
            ln = np.log10(n)
            points(ax, rx, ln, v, lo, hi, xerr=[ln - np.log10(nlo), np.log10(nhi) - ln], fmt="D", ms=2.4,
                   mfc="white", mec="0.35", mew=0.6, color="0.35", elinewidth=0.6, capsize=1.0, capthick=0.6,
                   zorder=5)
            pn, pv = pro
            ax.fill(np.log10(pn), pv, fc=BAND, ec="0.35", lw=0.6, hatch="////", zorder=2)
            # the region over the model: its edges densified in (log N, log f) before dividing
            ln, lf = np.log10(np.append(pn, pn[0])), np.log10(np.append(pv, pv[0]))
            t = np.linspace(0, 1, 40)[:-1]
            dn = np.concatenate([a + (b - a) * t for a, b in zip(ln[:-1], ln[1:])])
            df = np.concatenate([a + (b - a) * t for a, b in zip(lf[:-1], lf[1:])])
            mod = model_at(x, f, dn)
            keep = np.isfinite(mod)
            rx.fill(dn[keep], 10 ** df[keep] / mod[keep], fc=BAND, ec="0.35", lw=0.6, hatch="////", zorder=2)
        for a in (ax, rx):
            for e in edges[1:]:
                a.axvline(e, color="0.55", lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(3e-29, 3e-17)
        ax.set_yticks([1e-27, 1e-24, 1e-21, 1e-18])
        ax.set_title(measured, fontsize=7.5, pad=3)
        rx.set_yscale("log")
        rx.set_ylim(0.15, 6)
        rx.set_yticks([0.3, 1, 3])
        rx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
        rx.yaxis.set_minor_formatter(NullFormatter())
        rx.set_xlim(17.0, 23.2)
        rx.set_xticks([18, 19, 20, 21, 22, 23])
        rx.set_xticklabels(["$18$", "", "$20$", "", "$22$", ""])
        rx.set_xlabel(r"$\log_{10} N_\mathrm{HI}$", fontsize=8)
    axes[0, 0].set_ylabel(r"$f(N_\mathrm{HI})\,[\mathrm{cm}^{2}]$")
    axes[1, 0].set_ylabel("data / model", fontsize=7.5)
    for xm, name in (((edges[0] + edges[1]) / 2, "LLS"), ((edges[1] + edges[2]) / 2, "subDLA"),
                     ((edges[2] + 23.2) / 2, "DLA")):
        axes[0, 0].text(xm, 1.2e-17 / 4, name, ha="center", va="top", fontsize=6.2, color="0.35")
    fig.legend(handles=[
        Line2D([], [], color=TEAL, lw=1.2, label="model"),
        Patch(fc=TEAL_LIGHT, lw=0, label="design points"),
        Line2D([], [], color=INK, lw=0, marker="o", ms=2.2, label=r"Ho et al. (2021)"),
        Line2D([], [], color=INK, lw=0, marker="s", ms=2.6, mfc="white", mew=0.6, label=r"Zafar et al. (2013)"),
        Line2D([], [], color="0.35", lw=0, marker="D", ms=2.4, mfc="white", mew=0.6, label=r"O'Meara et al. (2007)"),
        Patch(fc=BAND, ec="0.35", lw=0.6, hatch="////", label=r"Prochaska et al. (2010)"),
    ], loc="lower center", ncol=3, fontsize=7.0, handlelength=1.5, columnspacing=1.4, labelspacing=0.3,
        bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.94, bottom=0.25, wspace=0.0, hspace=0.0)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
