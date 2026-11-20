"""Build the public tutorial notebook, ``notebooks/tutorial/hcd_priya_tutorial.ipynb``.

Every cell of the notebook is held here as source, in order, in ``CELLS``: a list of ``(kind, cell_id, source)``
with kind ``"markdown"`` or ``"code"``. ``build(path)`` writes them as a notebook without outputs; the committed
notebook is that file, executed. The helper functions the notebook needs are cells of the notebook, so the
notebook imports nothing beyond the installed packages; ``helper_source(cell_id)`` returns the source of one code
cell, which is how the tests (``tests/test_tutorial_notebook.py``) and ``run_mock_chain.py`` run the same code a
reader runs.

Run::

    python notebooks/tutorial/_build_tutorial.py            # writes the notebook beside this file
    jupyter nbconvert --to notebook --execute --inplace notebooks/tutorial/hcd_priya_tutorial.ipynb

The table of stored files in the cell ``helpers-assets`` is written from ``assets/ASSETS.json``.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
NOTEBOOK = HERE / "hcd_priya_tutorial.ipynb"
MANIFEST = HERE / "assets" / "ASSETS.json"

CELLS: list = []


def text(cell_id: str, source: str) -> None:
    """Add a text cell."""
    CELLS.append(("markdown", cell_id, source.strip("\n")))


def code(cell_id: str, source: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", cell_id, source.strip("\n")))


def helper_source(cell_id: str) -> str:
    """The source of the code cell ``cell_id``, as it stands in the notebook.

    Raises
    ------
    KeyError
        If no code cell has that id.
    """
    for kind, name, source in CELLS:
        if name == cell_id and kind == "code":
            return source
    raise KeyError(cell_id)


def assets_literal() -> str:
    """The table ``ASSETS`` of the cell ``helpers-assets``, as Python source, from ``assets/ASSETS.json``."""
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    rows = [
        f'    "{r["path"]}": {{\n        "sha256": "{r["sha256"]}",\n        "url": "{r["url"]}",\n    }},'
        for r in records
    ]
    return "ASSETS = {\n" + "\n".join(rows) + "\n}"


def build(out_path=NOTEBOOK) -> pathlib.Path:
    """Write the notebook, without outputs, to ``out_path``."""
    cells = []
    for kind, cell_id, source in CELLS:
        cell = {
            "cell_type": kind,
            "id": cell_id,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    doc = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = pathlib.Path(out_path)
    out_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


# ============================================================================================ the cells
text(
    "title",
    r"""
# From HCD absorbers to a mock inference: a tutorial for `hcd_priya`

**PROVISIONAL (PI to refine).** This notebook has three parts.

1. **Scientific motivation.** The Lyman-alpha forest and the high-column-density (HCD) absorbers on the same
   sightlines: Lyman limit systems (LLS), sub-damped and damped Lyman-alpha absorbers (subDLAs, DLAs).
2. **The emulator.** What the published model `hcd_priya` 1.0.0 returns: the flux power spectrum (P1D) of four
   classes of sightline, the incidence of the absorbers, and a mixture of the classes.
3. **Mock inference.** A Gaussian likelihood on a mock made by the model itself, with four free parameters. It is
   a self-consistency demonstration: not an independent validation, not a result from data.

**To run it:** `pip install "hcd-priya[tutorials]"` in Python 3.11 or later, then open this file with Jupyter. The
trained model comes with the package. The few stored files the notebook reads are found in a clone of the
repository or downloaded once, each checked against its sha256 (next cell).
""",
)

code(
    "imports",
    r"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

import hcd_analysis
import hcd_analysis.predictor as hp

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 10,
                     "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3})
CLASS_COLOURS = {"clean": "#4d4d4d", "LLS": "#1f77b4", "subDLA": "#ff7f0e", "DLA": "#d62728"}

print("hcd_analysis", hcd_analysis.__version__)
""",
)

code(
    "helpers-assets",
    r"""
# The stored files of this tutorial. Each is named by its path in the repository (a file of the GitHub release by
# its file name) and pinned by its sha256. Every URL is immutable: the tag v1.0.0, a file of the release v1.0.0,
# or one commit.
import hashlib
import os
import pathlib
import tempfile
import urllib.request

@ASSETS@

# Inside a clone of the repository the files are on disk already. A local file is used when, and only when, its
# sha256 is the pinned one, so the notebook reads the same bytes inside a clone and outside one. This notebook's
# folder is two levels below the top of a clone.
LOCAL_ROOTS = (pathlib.Path.cwd(), pathlib.Path.cwd().parent.parent)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tutorial_file(relpath):
    # The path of a stored file of the tutorial, with the pinned sha256: a local copy in a clone, a copy
    # downloaded earlier, or a new download into the cache folder (the environment variable
    # HCD_PRIYA_TUTORIAL_CACHE, default ~/.cache/hcd_priya_tutorial).
    if relpath not in ASSETS:
        raise RuntimeError(f"{relpath} is not a stored file of this tutorial (ASSETS names {len(ASSETS)} files)")
    pinned, url = ASSETS[relpath]["sha256"], ASSETS[relpath]["url"]
    for root in LOCAL_ROOTS:
        local = pathlib.Path(root) / relpath
        if local.is_file() and sha256_of(local) == pinned:
            return local
    cache = pathlib.Path(os.environ.get("HCD_PRIYA_TUTORIAL_CACHE", "~/.cache/hcd_priya_tutorial")).expanduser()
    target = cache / relpath
    if target.is_file() and sha256_of(target) == pinned:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # written under a temporary name and renamed only after the digest is checked: no partial file is left
    handle, partial = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
    digest = hashlib.sha256()
    try:
        try:
            with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
                    digest.update(block)
        except Exception as exc:
            raise RuntimeError(
                f"could not download the tutorial file {relpath} from {url} into the cache folder {cache}: {exc}"
            ) from exc
        if digest.hexdigest() != pinned:
            raise RuntimeError(
                f"the tutorial file {relpath} downloaded from {url} has sha256 {digest.hexdigest()}, "
                f"not the pinned {pinned}: it is refused, and nothing is kept in {cache}"
            )
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return target


print(len(ASSETS), "stored files are pinned by sha256")
""".replace("@ASSETS@", assets_literal()),
)

# ============================================================================================ Part I
text(
    "part-1",
    r"""
## Part I. Scientific motivation

**PROVISIONAL (PI to refine).**
""",
)

text(
    "fig1-placeholder",
    r"""
**Fig. 1.** PLACEHOLDER: the image from the Leinweber talk (the PI supplies the file and its terms).

**PROVISIONAL (PI to refine).** The figure is to show one quasar sightline: the Lyman-alpha forest, and the
absorbers of high column density on the same sightline, a Lyman limit system (LLS), a sub-damped and a damped
Lyman-alpha absorber (subDLA, DLA) with their damping wings.
""",
)

text(
    "classes",
    r"""
### Three classes of absorber

**PROVISIONAL (PI to refine).** The package sorts absorbers by their neutral hydrogen column density $N_\mathrm{HI}$
into three classes, LLS, subDLA and DLA. The next cell prints the edges the package uses and draws the column
density distribution $f(N_\mathrm{HI})$ of published measurements across the three classes. A sightline belongs to
the class of its strongest absorber; a sightline with no such absorber is called clean.
""",
)

code(
    "fig2-cddf",
    r"""
# Fig. 2. The column density distribution f(N_HI) from published measurements, from LLS to DLAs.
import hcd_analysis.catalog as catalog

EDGES = (catalog.LOG_NHI_LLS_MIN, catalog.LOG_NHI_SUBDLA_MIN, catalog.LOG_NHI_DLA_MIN)
print(f"LLS:    10^{EDGES[0]} <= N_HI < 10^{EDGES[1]} cm^-2")
print(f"subDLA: 10^{EDGES[1]} <= N_HI < 10^{EDGES[2]} cm^-2")
print(f"DLA:    N_HI >= 10^{EDGES[2]} cm^-2")

DATA = "docs/figures/data/"
CDDF_PANELS = (
    ("2 < z < 2.5", "ho21/cddf_z225.txt"),
    ("2.5 < z < 3", "ho21/cddf_z253.txt"),
    ("3 < z < 4", "ho21/cddf_z34.txt"),
)


def read_ho21_cddf(path):
    # six rows: log10 N_HI of the bin centre, f, the two limits of the 68 per cent interval, those of the 95 per cent
    # interval (not drawn); bins with f = 0 are left out
    table = np.loadtxt(path)
    keep = table[1] > 0
    return table[0][keep], table[1][keep], table[2][keep], table[3][keep]


# Zafar et al. (2013): log10 N_HI, the half-width of the bin, log10 f, its upper and its lower error in dex
z13_logn, z13_half, z13_logf, z13_up, z13_down = np.loadtxt(tutorial_file(DATA + "lls_subdla/zafar_2013.txt"), ndmin=2).T
# O'Meara et al. (2007), all log10: N_HI, f, the two edges of the bin, the two limits of f
o07_logn, o07_logf, o07_lo_n, o07_hi_n, o07_lo_f, o07_hi_f = np.loadtxt(
    tutorial_file(DATA + "lls_subdla/omeara2007_fn_all19p0.dat"), ndmin=2
).T
# Prochaska, O'Meara and Worseck (2010): the corners of the allowed region, log10 N_HI and log10 f
p10 = np.loadtxt(tutorial_file(DATA + "lls_subdla/prochaska2010_lls_region.dat"), ndmin=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
drawn_logn, drawn_f = [], []
for ax, (title, name) in zip(axes, CDDF_PANELS):
    logn, f, f_lo, f_hi = read_ho21_cddf(tutorial_file(DATA + name))
    ax.errorbar(logn, f, yerr=[f - f_lo, f_hi - f], fmt="o", ms=3.5, color="black", elinewidth=0.8,
                label="DLAs: Ho, Bird and Garnett (2021), 68 per cent interval")
    z13_f = 10.0**z13_logf
    ax.errorbar(z13_logn, z13_f, xerr=z13_half, yerr=[z13_f - 10.0 ** (z13_logf - z13_down), 10.0 ** (z13_logf + z13_up) - z13_f],
                fmt="s", ms=4, color=CLASS_COLOURS["subDLA"], elinewidth=0.8, label="subDLAs: Zafar et al. (2013), z 1.5 to 5")
    drawn_logn += [logn, z13_logn]
    drawn_f += [f, z13_f]
    if title == "3 < z < 4":
        o07_f = 10.0**o07_logf
        ax.errorbar(o07_logn, o07_f, xerr=[o07_logn - o07_lo_n, o07_hi_n - o07_logn],
                    yerr=[o07_f - 10.0**o07_lo_f, 10.0**o07_hi_f - o07_f], fmt="D", ms=4, color=CLASS_COLOURS["DLA"],
                    elinewidth=0.8, label="O'Meara et al. (2007), mean z 3.1")
        ax.fill(p10[:, 0], 10.0 ** p10[:, 1], facecolor=CLASS_COLOURS["LLS"], edgecolor=CLASS_COLOURS["LLS"], alpha=0.45, lw=1.0,
                label="LLS: allowed region of Prochaska, O'Meara and Worseck (2010), z 3.7")
        drawn_logn += [o07_logn, p10[:, 0]]
        drawn_f += [o07_f, 10.0 ** p10[:, 1]]
    for edge in EDGES:
        ax.axvline(edge, color="grey", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(17.0, 23.0)
    ax.set_xlabel(r"$\log_{10} N_\mathrm{HI}$ [cm$^{-2}$]")
    ax.set_title(title)
axes[0].set_ylabel(r"$f(N_\mathrm{HI}, X)$ [cm$^2$]")
fig.legend(*axes[2].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9)
fig.suptitle("Fig. 2. Published column density distributions, as published (each per unit X of its own cosmology);\n"
             "dotted lines: the class edges of the package")
fig.tight_layout(rect=(0, 0.13, 1, 1))
plt.show()

drawn_logn, drawn_f = np.concatenate(drawn_logn), np.concatenate(drawn_f)
print(f"the points drawn run from log10 N_HI = {drawn_logn.min():.2f} to {drawn_logn.max():.2f}")
print(f"and f from {drawn_f.max():.2e} down to {drawn_f.min():.2e} cm^2: {np.log10(drawn_f.max() / drawn_f.min()):.1f} decades of f")
print(f"the forest lies below the lowest class edge, log10 N_HI = {EDGES[0]}, and is not drawn")
""",
)

text(
    "why-apart",
    r"""
### Why the forest and the strong absorbers are usually treated apart

**PROVISIONAL (PI to refine).**

- *In observations.* DLAs are found one by one in the spectra and are masked, or their damping wings are fitted.
  LLS are not identified one by one at the resolution of a survey spectrum.
- *In simulations.* The strong absorbers depend on self-shielding and on resolution, and emulators of the forest
  are trained on spectra with the strong absorbers removed.

The two treatments meet in the one-dimensional flux power spectrum (P1D), which is measured on the spectra as
they are.
""",
)

text(
    "contamination",
    r"""
### What is left in the P1D

**PROVISIONAL (PI to refine). HCD contamination is not negligible for small-scale Lyman-alpha cosmology.**

PLACEHOLDER: the statements of Ho et al. (2026) and of the DESI DR1 analysis (Chaves-Montero et al. 2026) on the
absorbers that remain in a measured P1D and on the scales they affect, quoted with their references. Each
sentence is to be checked against its paper before it is written here.

References to be quoted: Ho et al., JCAP 2026 (07) 094; Chaves-Montero et al., JCAP 2026 (06) 040; the DESI DR1
measurements of the P1D, Karacayli et al. (2025) and Ravoux et al. (2025).
""",
)

# ============================================================================================ Part II
text(
    "part-2",
    r"""
## Part II. The emulator

**PROVISIONAL (PI to refine).** The published model was trained on the PRIYA suite of hydrodynamical simulations.
Given nine cosmological and astrophysical parameters, a redshift and a mean flux, it returns the P1D of four
classes of sightline (clean, LLS, subDLA, DLA), the incidence of the three classes of absorber, and the column
density distribution. The next cell loads it and prints what identifies it.
""",
)

code(
    "load-model",
    r"""
emu = hp.load_published()  # the trained LF+MF model, from the installed package hcd-priya-models

release = emu.provenance["release"]
print("model release:", release["release_version"], "| format:", release["format"])
print("sha256 of its release.json:", release["release_sha256"])

# The point used in this notebook: the midpoint of the range each parameter was trained on. It is a point to try
# the model at: not a fiducial cosmology, not a best fit and not a simulation.
PARAMS = {
    "ns": 0.925,  # slope of the primordial power spectrum at k = 0.78 / Mpc
    "Ap": 1.9e-9,  # its amplitude at the same wavenumber
    "herei": 4.0,  # redshift at which helium reionization starts
    "heref": 2.7,  # redshift at which helium reionization ends
    "alphaq": 2.15,  # quasar spectral index, which sets the heating of helium reionization
    "hub": 0.70,  # dimensionless Hubble parameter h
    "omegamh2": 0.143,  # Omega_m h^2
    "hireionz": 7.25,  # redshift of hydrogen reionization
    "bhfeedback": 0.05,  # black-hole feedback strength
}
print()
print(f"{'parameter':11s} {'trained from':>13s} {'to':>10s} {'used here':>10s}")
for name, (low, high) in zip(hp.PARAM_NAMES, hp.PARAM_LIMITS):
    print(f"{name:11s} {low:13.4g} {high:10.4g} {PARAMS[name]:10.4g}")

# The mean flux is PRIYA's model: tau_eff(z) = tau0_amp ((1 + z) / (1 + z_pivot))**dtau0 times the fit of Kim et
# al. (2007). The model records the range of tau0_amp it was trained on with its training data:
TAU0_AMP_RANGE = tuple(emu.provenance["lineage"]["cache"]["settings"]["alpha_range"])
print(f"{'tau0_amp':11s} {TAU0_AMP_RANGE[0]:13.4g} {TAU0_AMP_RANGE[1]:10.4g} {1.0:10.4g}   (dtau0 = 0 here)")
""",
)

code(
    "helpers-incidence",
    r"""
import numpy as np


def cumulative_incidence(cddf, log_nhi, threshold=17.5):
    # dN/dX of every absorber with N_HI >= 10^threshold, from a column density distribution f(N_HI) at the bin
    # centres log_nhi: the sum of f times the width in N_HI of each bin's part above the threshold. The bin that
    # holds the threshold counts from it to its upper edge, with f taken constant within the bin; a bin without a
    # value (NaN) is left out. On the last axis.
    cddf = np.asarray(cddf, dtype=float)
    log_nhi = np.asarray(log_nhi, dtype=float)
    step = np.diff(log_nhi)
    if not np.allclose(step, step[0], rtol=0.0, atol=1e-9):
        raise ValueError("the bin centres of the CDDF are not evenly spaced")
    low = np.maximum(log_nhi - step[0] / 2.0, threshold)
    high = log_nhi + step[0] / 2.0
    widths = np.where(high > low, 10.0**high - 10.0**low, 0.0)
    counted = (widths > 0) & np.isfinite(cddf)
    total = np.where(counted, cddf * widths, 0.0).sum(axis=-1)
    return np.where(counted.any(axis=-1), total, np.nan)


def absorption_distance(z1, z2, omega_m):
    # Delta X from z1 to z2 in a flat cosmology with matter density omega_m:
    # the integral of (1 + z)^2 / sqrt(omega_m (1 + z)^3 + 1 - omega_m) dz (trapezoid rule on 2001 points)
    z = np.linspace(z1, z2, 2001)
    y = (1.0 + z) ** 2 / np.sqrt(omega_m * (1.0 + z) ** 3 + 1.0 - omega_m)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(z)))
""",
)

code(
    "fig3-incidence-response",
    r"""
# Fig. 3. The predicted incidence of LLS, and of all absorbers above 10^17.5 cm^-2, at z = 3 while one parameter at
# a time crosses the range it was trained on (five values; the other eight at the midpoint, the mean flux fixed).
Z0 = 3.0
I_LLS = hp.HCD_CLASSES.index("LLS")
mid = emu.predict(PARAMS, z=Z0, tau0_amp=1.0, dtau0=0.0)
mid_flags = {name for name, value in mid.flags.items() if value is True}

response, new_flags = {}, {}
for name, (low, high) in zip(hp.PARAM_NAMES, hp.PARAM_LIMITS):
    values = np.linspace(low, high, 5)
    rows = [emu.predict(dict(PARAMS, **{name: value}), z=Z0, tau0_amp=1.0, dtau0=0.0) for value in values]
    lls = np.array([row.dndx[I_LLS] for row in rows]) / mid.dndx[I_LLS]
    above = np.array([cumulative_incidence(row.cddf, row.log_nhi) for row in rows]) / cumulative_incidence(mid.cddf, mid.log_nhi)
    response[name] = (values, lls, above)
    new_flags[name] = sorted({f for row in rows for f, value in row.flags.items() if value is True} - mid_flags)

fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True)
for ax, (name, (values, lls, above)) in zip(axes.ravel(), response.items()):
    ax.plot(values, lls, "o-", color=CLASS_COLOURS["LLS"], label="LLS class, dN/dX")
    ax.plot(values, above, "s--", color="black", label=r"all absorbers with $N_\mathrm{HI} \geq 10^{17.5}$")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_xlabel(name)
fig.supylabel("incidence / its value at the midpoint")
axes[0, 0].legend(fontsize=9)
fig.suptitle("Fig. 3. Predicted absorber incidence at z = 3 across the trained range of each parameter")
fig.tight_layout()
plt.show()

print(f"at the midpoint: dN/dX of the LLS class = {mid.dndx[I_LLS]:.4f}, "
      f"of all absorbers above 10^17.5 = {float(cumulative_incidence(mid.cddf, mid.log_nhi)):.4f}")
print("The change of the predicted LLS incidence across the trained range of each parameter, largest first")
print("(the largest of the five values over the smallest; the value at the low and at the high end over the midpoint):")
RANKING = sorted(response, key=lambda name: -(response[name][1].max() / response[name][1].min()))
for name in RANKING:
    values, lls, above = response[name]
    note = f"   flags set along the way: {', '.join(new_flags[name])}" if new_flags[name] else ""
    print(f"  {name:11s} x {lls.max() / lls.min():.3f}   low end {lls[0]:.3f}, high end {lls[-1]:.3f}{note}")
""",
)

text(
    "reading-fig3",
    r"""
**PROVISIONAL (PI to refine).** Reading Fig. 3: the printed list ranks the nine parameters by how much the
predicted LLS incidence changes across the range each was trained on. The absorber population is itself a
prediction of the model, and it depends on cosmological and on astrophysical parameters; which of the two kinds
matters more is to be read from the list above and written here.
""",
)

code(
    "fig4-architecture",
    r"""
# Fig. 4. How the emulator is built (the schematic of the manuscript; assets/figures/PROVENANCE.md in the repository).
from IPython.display import Image, display

display(Image(filename=str(tutorial_file("notebooks/tutorial/assets/figures/joss_architecture.png")), width=900))
""",
)

text(
    "architecture",
    r"""
**Fig. 4. PROVISIONAL (PI to refine).** One model takes the nine parameters, the redshift and the mean flux, and
returns the four class spectra, with a multi-fidelity correction towards the high-resolution simulations, and the
incidence of the absorbers (dN/dX and the column density distribution).
""",
)

code(
    "one-prediction",
    r"""
pred = emu.predict(PARAMS, z=3.0, tau0_amp=1.0, dtau0=0.0)

print("model:", pred.labels["product"])
print("classes, in the order of P_class:", hp.CLASSES)
print(f"pred.k:       shape {pred.k.shape}, from {pred.k[0]:.2e} to {pred.k[-1]:.2e} s/km")
print(f"pred.P_class: shape {pred.P_class.shape}, in km/s")
print(f"pred.w_c:     the model's sightline fractions of {hp.HCD_CLASSES}: {np.round(pred.w_c, 4)}")
print(f"pred.dndx:    the incidence dN/dX of the same three classes: {np.round(pred.dndx, 4)}")
print(f"mean flux:    tau_eff = {float(pred.tau_eff):.4f}")
print("flags that are set:", sorted(name for name, value in pred.flags.items() if value is True))
""",
)

text(
    "conventions",
    r"""
### Conventions

**PROVISIONAL (PI to refine).**

- The spectrum is that of $\delta_F = F / \langle F \rangle - 1$, with one mean flux $\langle F \rangle$ for the
  whole sample of sightlines, all classes together.
- $k$ is the angular wavenumber, in s/km.
- The class spectra are *PRIYA-filtered*: the cores of the strongest absorbers are masked in the simulated
  spectra, as for the PRIYA forest emulator, and the mixture below does not add back what the filter removed
  (`pred.labels["omissions"]` names it, and a later cell prints it).

A class spectrum is therefore a conditional spectrum under one normalisation. It is not the spectrum of isolated
absorbers. The definitions are in `docs/p1d_definition.md` of the repository.
""",
)

code(
    "fig5-class-spectra",
    r"""
# Fig. 5. The four class spectra, and the ratio of each HCD class to the clean forest, at three redshifts.
Z_SHOWN = (2.2, 3.0, 4.0)
stack = emu.predict_many(PARAMS, z=list(Z_SHOWN), tau0_amp=1.0, dtau0=0.0)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for j, z in enumerate(Z_SHOWN):
    k = stack.k[j]
    for i, name in enumerate(hp.CLASSES):
        axes[0, j].loglog(k, k * stack.P_class[j, i] / np.pi, color=CLASS_COLOURS[name], label=name)
    for i, name in enumerate(hp.CLASSES[1:], start=1):
        axes[1, j].semilogx(k, stack.P_class[j, i] / stack.P_class[j, 0], color=CLASS_COLOURS[name], label=f"{name} / clean")
    axes[1, j].axhline(1.0, color="grey", lw=0.8)
    axes[0, j].set_title(f"z = {z}")
    axes[1, j].set_xlabel("k [s/km]")
axes[0, 0].set_ylabel(r"$k\,P(k)/\pi$")
axes[1, 0].set_ylabel("class spectrum / clean spectrum")
axes[0, 0].legend()
axes[1, 0].legend()
fig.suptitle("Fig. 5. The class spectra of the model (PRIYA-filtered, one mean flux for all classes)")
fig.tight_layout()
plt.show()

def k_ranges(k, mask):
    # the contiguous ranges of k on which mask holds, as text (the first three, then how many more)
    ranges, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(mask) - 1):
            ranges.append(f"{k[start]:.2e} to {k[i if flag else i - 1]:.2e}")
            start = None
    if not ranges:
        return "nowhere"
    return "; ".join(ranges[:3]) + (f"; and {len(ranges) - 3} more ranges" if len(ranges) > 3 else "")


print("Where each class ratio lies above and below one (ranges of k in s/km):")
for j, z in enumerate(Z_SHOWN):
    k = stack.k[j]
    for i, name in enumerate(hp.CLASSES[1:], start=1):
        ratio = stack.P_class[j, i] / stack.P_class[j, 0]
        above = ratio > 1.0
        print(f"  z = {z}  {name:7s} smallest {ratio.min():.3f}, largest {ratio.max():.3f}; above one on {int(above.sum())} of {k.size} modes: "
              f"{k_ranges(k, above)}; below one: {k_ranges(k, ~above)}")
""",
)

text(
    "reading-fig5",
    r"""
**PROVISIONAL (PI to refine).** Reading Fig. 5: the printed lines say where each class has more and where it has
less power than the clean forest, as the model gives it. With the conventions above, a ratio below one is not a
defect: the classes share one mean flux, and the filter removes the core of the strongest absorbers. The physical
reading of each class is to be written here from the printed numbers.
""",
)

text(
    "mixture",
    r"""
### The mixture of the classes

**PROVISIONAL (PI to refine).** `hp.add_hcd(pred, A)` mixes the class spectra linearly:

$$P_\mathrm{mix}(k) = P_\mathrm{clean}(k) + \sum_c A_c\,[P_c(k) - P_\mathrm{clean}(k)], \qquad c \in \{\mathrm{LLS}, \mathrm{subDLA}, \mathrm{DLA}\}.$$

The amplitudes $A_c$ are fractions of sightlines. The model's own fractions at a point are `pred.w_c`.

What the mixture is not: it is not the P1D of an observed sample (the cores removed by the filter are left out,
see `pred.labels["omissions"]`); nothing is fitted; and it carries no law in redshift of its own.
""",
)

code(
    "fig6-mixture",
    r"""
# Fig. 6. The mixture over the clean spectrum at z = 3, with the amplitude of one class at a time at 0, 0.5, 1 and
# 2 times the model's own fraction (the other two classes at the model's own).
FACTORS = (0.0, 0.5, 1.0, 2.0)
clean = pred.P_class[0]
print("the model's own fractions at this point:", ", ".join(f"{name} {value:.4f}" for name, value in zip(hp.HCD_CLASSES, pred.w_c)))
print("what the mixture leaves out:")
for omission in pred.labels["omissions"]:
    print("  -", omission)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for c, name in enumerate(hp.HCD_CLASSES):
    spread = []
    for factor, style in zip(FACTORS, (":", "--", "-", "-.")):
        scale = np.ones(3)
        scale[c] = factor
        mixed = hp.add_hcd(pred, pred.w_c * scale)
        spread.append(mixed / clean)
        axes[c].semilogx(pred.k, mixed / clean, style, color=CLASS_COLOURS[name], label=f"{name} amplitude x {factor:g}")
    axes[c].axhline(1.0, color="grey", lw=0.8)
    axes[c].set_xlabel("k [s/km]")
    axes[c].set_title(f"the {name} amplitude varied alone")
    axes[c].legend(fontsize=9)
    change = np.abs(spread[-1] / spread[0] - 1.0)
    print(f"{name:7s} amplitude from 0 to 2 times the model's own: the mixture changes by at most "
          f"{100 * change.max():.2f} per cent, at k = {pred.k[change.argmax()]:.2e} s/km, "
          f"and by {100 * change[-1]:.2f} per cent at the largest k")
axes[0].set_ylabel("mixture / clean spectrum")
fig.suptitle("Fig. 6. The scale dependence each class leaves on the mixture (amplitudes are sightline fractions)")
fig.tight_layout()
plt.show()
""",
)

code(
    "fig7-incidence",
    r"""
# Fig. 7. The incidence dN/dX the model predicts, against redshift, with published measurements.
Z_VALIDATED = [z for z in hp.Z_GRID if 2.2 - 1e-9 <= z <= 4.6 + 1e-9]
inc = emu.predict_many(PARAMS, z=Z_VALIDATED, tau0_amp=1.0, dtau0=0.0)
inc_above = cumulative_incidence(inc.cddf, inc.log_nhi)
OMEGA_M = PARAMS["omegamh2"] / PARAMS["hub"] ** 2  # the flat cosmology of the model line

# DLAs, Ho, Bird and Garnett (2021): the bin centre in z, dN/dX, the two limits of the 68 per cent interval (and of
# the 95 per cent interval, not drawn), per unit X of their flat cosmology with Omega_m = 0.279. A measurement per
# unit X is drawn in the cosmology of the model line: times Delta X(its own) / Delta X(the line's) over its bin.
ho21 = np.loadtxt(tutorial_file(DATA + "ho21/dndx_all.txt"))
half_bin = (ho21[0][1] - ho21[0][0]) / 2.0
ho21_factor = np.array([
    absorption_distance(z - half_bin, z + half_bin, 0.279) / absorption_distance(z - half_bin, z + half_bin, OMEGA_M)
    for z in ho21[0]
])

# Absorbers with tau_912 >= 2 (N_HI >= 10^17.5, counted cumulatively), published as l(z) = dN/dz. The samples drawn
# are those of the repository's figure docs/figures/dndx_vs_ho21.png: (label, marker, whether the printed plus
# error is the smaller one and the two are swapped, as that figure does).
LLS_DRAWN = {
    "prochaska10_t4": ("Prochaska, O'Meara and Worseck (2010)", "^", True),
    "omeara13_t5_comb": ("O'Meara et al. (2013) with Ribaudo, Lehner and Howk (2011)", "s", False),
    "fumagalli13_nc": ("Fumagalli et al. (2013)", "D", False),
    "crighton19_corr": ("Crighton et al. (2019)", "v", False),
    "fumagalli20_dr16": ("Fumagalli, Fotopoulou and Thomson (2020)", "o", False),
}
lls_points = {source: [] for source in LLS_DRAWN}
with open(tutorial_file(DATA + "lls_incidence/lls_lz_literature.dat"), encoding="utf-8") as handle:
    for line in handle:
        if line.startswith("#") or not line.strip():
            continue
        z_lo, z_hi, value, err_lo, err_hi, quantity, _, source = line.rstrip("\n").split("\t")
        if source not in LLS_DRAWN:
            continue
        if quantity != "lz":
            raise ValueError(f"a row of {source} is not an l(z)")
        z_lo, z_hi, value, below, above = (float(v) for v in (z_lo, z_hi, value, err_lo, err_hi))
        if LLS_DRAWN[source][2]:
            below, above = above, below
        to_x = (z_hi - z_lo) / absorption_distance(z_lo, z_hi, OMEGA_M)  # l(X) = l(z) Delta z / Delta X
        lls_points[source].append((0.5 * (z_lo + z_hi), 0.5 * (z_hi - z_lo), to_x * value, to_x * below, to_x * above))

fig, ax = plt.subplots(figsize=(11, 8.5))
for c, name in enumerate(hp.HCD_CLASSES):
    ax.plot(Z_VALIDATED, inc.dndx[:, c], color=CLASS_COLOURS[name], lw=2, label=f"model, {name} class")
    ax.fill_between(Z_VALIDATED, inc.dndx_lower[:, c], inc.dndx_upper[:, c], color=CLASS_COLOURS[name], alpha=0.25, lw=0)
ax.plot(Z_VALIDATED, inc_above, color="black", lw=2, ls="--", label=r"model, all absorbers with $N_\mathrm{HI} \geq 10^{17.5}$")
ax.errorbar(ho21[0], ho21_factor * ho21[1], yerr=[ho21_factor * (ho21[1] - ho21[2]), ho21_factor * (ho21[3] - ho21[1])],
            fmt="o", ms=4, color=CLASS_COLOURS["DLA"], mec="black", mew=0.5, elinewidth=0.8,
            label="DLAs: Ho, Bird and Garnett (2021)")
for source, (label, marker, _) in LLS_DRAWN.items():
    zc, zh, value, below, above = np.array(lls_points[source]).T
    ax.errorbar(zc, value, xerr=zh, yerr=[below, above], fmt=marker, ms=5, color="black", mfc="white", elinewidth=0.8,
                label=r"$N_\mathrm{HI} \geq 10^{17.5}$: " + label)
ax.set_yscale("log")
ax.set_xlim(1.9, 5.5)
ax.set_xlabel("redshift z")
ax.set_ylabel("dN/dX")
fig.legend(*ax.get_legend_handles_labels(), fontsize=9, ncol=2, loc="lower center")
ax.set_title("Fig. 7. Predicted incidence at the midpoint parameters (bands: the held-out range of each class line);\n"
             "open symbols count every absorber above 10^17.5 and belong with the dashed line, not with the LLS class line")
fig.tight_layout(rect=(0, 0.2, 1, 1))
plt.show()

j3 = Z_VALIDATED.index(3.0)
print("At z = 3, the model's dN/dX with its held-out range (value x exp(-e) to value x exp(+e)):")
for c, name in enumerate(hp.HCD_CLASSES):
    print(f"  {name:7s} {inc.dndx[j3, c]:.4f}   range {inc.dndx_lower[j3, c]:.4f} to {inc.dndx_upper[j3, c]:.4f}")
print(f"  all absorbers above 10^17.5: {inc_above[j3]:.4f}")
print(f"factors applied to the measurements: Ho, Bird and Garnett (2021) {ho21_factor.min():.4f} to {ho21_factor.max():.4f} "
      f"(their Omega_m 0.279 to the line's {OMEGA_M:.4f})")
""",
)

text(
    "two-operations",
    r"""
### Two different operations

**PROVISIONAL (PI to refine).** There are two ways to change how much the LLS contribute to a mixture, and they
are not the same operation.

1. **Changing an amplitude $A_c$** rescales how many sightlines of a class enter the mixture, at fixed class
   spectra. No new prediction is made. The amplitude is a nuisance of the mixture.
2. **Changing the nine parameters** changes what the emulator predicts: the incidence (Fig. 3), the sightline
   fractions and the class spectra together. The incidence is a prediction.

The next cell does each once and prints what changes and what stays the same.
""",
)

code(
    "two-operations-numbers",
    r"""
def largest_change(a, b):
    # the largest relative difference of two spectra over k, in per cent
    return 100.0 * float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1.0)))


mix_0 = hp.add_hcd(pred, pred.w_c)

# A. The LLS amplitude doubled by hand, at fixed parameters. The same prediction is used; nothing is predicted anew.
again = emu.predict(PARAMS, z=3.0, tau0_amp=1.0, dtau0=0.0)
scale = np.ones(3)
scale[I_LLS] = 2.0
mix_a = hp.add_hcd(again, again.w_c * scale)

# B. One parameter moved, at the model's own fractions: the first parameter of the ranking of Fig. 3, moved to the
# end of its trained range where the predicted LLS incidence is the larger.
moved_name = RANKING[0]
moved_value = float(response[moved_name][0][np.argmax(response[moved_name][1])])
pred_b = emu.predict(dict(PARAMS, **{moved_name: moved_value}), z=3.0, tau0_amp=1.0, dtau0=0.0)
mix_b = hp.add_hcd(pred_b, pred_b.w_c)

print(f"A: the LLS amplitude set to 2 times the model's own fraction, parameters fixed")
print(f"B: {moved_name} moved from {PARAMS[moved_name]:g} to {moved_value:g}, amplitudes at the model's own fractions")
print()
print(f"{'':44s} {'A':>22s} {'B':>22s}")
print(f"{'LLS amplitude in the mixture':44s} {'x 2.000 (by hand)':>22s} {'x %.3f (pred.w_c)' % (pred_b.w_c[I_LLS] / pred.w_c[I_LLS]):>22s}")
for c, name in enumerate(hp.HCD_CLASSES):
    same = "the same" if again.dndx[c] == pred.dndx[c] else "x %.3f" % (again.dndx[c] / pred.dndx[c])
    print(f"{'predicted incidence dN/dX, ' + name:44s} {same:>22s} {'x %.3f' % (pred_b.dndx[c] / pred.dndx[c]):>22s}")
for i, name in enumerate(hp.CLASSES):
    same = "the same" if np.array_equal(again.P_class[i], pred.P_class[i]) else "changed"
    print(f"{'class spectrum, ' + name + ', largest change':44s} {same:>22s} {'%.2f per cent' % largest_change(pred_b.P_class[i], pred.P_class[i]):>22s}")
print(f"{'the mixture, largest change':44s} {'%.2f per cent' % largest_change(mix_a, mix_0):>22s} {'%.2f per cent' % largest_change(mix_b, mix_0):>22s}")
print()
print("What stays the same in A: the prediction itself. Its incidence, its sightline fractions and its four class spectra")
print("  are equal to those above:", bool(np.array_equal(again.dndx, pred.dndx) and np.array_equal(again.w_c, pred.w_c)
                                          and np.array_equal(again.P_class, pred.P_class)))
print(f"What stays the same in B: the other eight parameters, the redshift and the mean flux (tau_eff equal: "
      f"{bool(float(pred_b.tau_eff) == float(pred.tau_eff))}), and the rule that the amplitudes are the model's own fractions.")
""",
)

text(
    "validation",
    r"""
### How far to trust it: the held-out prediction residual and the validated range

**PROVISIONAL (PI to refine).** For each class spectrum and for the mixture the model carries a *held-out
prediction residual* $e(k)$: the root mean square of $\ln(\mathrm{prediction} / \mathrm{truth})$ over simulations
the model was made without. As a range on a spectrum it is the value times $\exp(-e)$ to the value times
$\exp(+e)$, each bound on its own. It is an empirical validation diagnostic, not a calibrated predictive
uncertainty.

The LF+MF model is validated from $z = 2.2$ to $4.6$. Above that range a prediction is returned with flags;
below it, and between snapshot redshifts, it is refused.
""",
)

code(
    "validation-numbers",
    r"""
e_mix = inc.P_mix_validation_rms_log  # one row per redshift of Z_VALIDATED, one value per wavenumber
print("The held-out prediction residual e of the mixture, by redshift (the value x exp(-e) to x exp(+e)):")
for j, z in enumerate(Z_VALIDATED):
    e, k = e_mix[j], inc.k[j]
    worst = int(np.nanargmax(e))
    print(f"  z = {z:.1f}  median e = {np.nanmedian(e):.3f} (x {np.exp(-np.nanmedian(e)):.3f} to x {np.exp(np.nanmedian(e)):.3f}); "
          f"largest e = {e[worst]:.3f} (x {np.exp(-e[worst]):.3f} to x {np.exp(e[worst]):.3f}) at k = {k[worst]:.2e} s/km")
print()
print("In the model's own words:", pred.labels["p1d_validation"][:420], "...")
print()

high = emu.predict(PARAMS, z=4.8, tau0_amp=1.0, dtau0=0.0)
set_at_3 = {name for name, value in pred.flags.items() if value is True}
print("flags set at z = 4.8 and not at z = 3.0:", sorted({name for name, value in high.flags.items() if value is True} - set_at_3))
try:
    emu.predict(PARAMS, z=3.1, tau0_amp=1.0, dtau0=0.0)
except hp.PredictorInputError as error:
    print("z = 3.1 is refused:", error)
try:
    emu.predict(PARAMS, z=2.0, tau0_amp=1.0, dtau0=0.0)
except hp.MFRangeError as error:
    print("z = 2.0 is refused:", error)
""",
)

text(
    "rogers",
    r"""
### Beside the template of Rogers et al. (2018)

**PROVISIONAL (PI to refine; each statement about the paper is to be checked against it).** Rogers et al. (2018)
give a template for the effect of HCD absorbers on the P1D: for each of four classes of column density, a fitted
function of $k$ and $z$ for the ratio of the class spectrum to the forest spectrum (their Eq. 6), from one
simulation, used with free amplitudes and with no dependence on cosmology. What differs here: the class spectra
are emulated across nine parameters and the mean flux; they are PRIYA-filtered; and the incidence is predicted.
`docs/rogers_comparison.md` of the repository gives the comparison in full.
""",
)

code(
    "fig8-rogers",
    r"""
# Fig. 8. Left: the PRIYA training simulations with the filtered troughs restored, against Eq. 6 of Rogers et al.
# (2018), at z = 3 (the repository's comparison, redrawn for two classes). Right: the response of the emulator's
# LLS class ratio to n_s and A_p, which the template does not have.
from hcd_analysis.hcd_template import class_ratio


def read_table(path):
    # a table with comment lines; the last comment line holds the column names
    with open(path, encoding="utf-8") as handle:
        names = [line for line in handle if line.startswith("#")][-1].lstrip("# ").strip().split(",")
    values = np.loadtxt(path, delimiter=",", comments="#")
    return {name: values[:, i] for i, name in enumerate(names)}


sims = read_table(tutorial_file(DATA + "class_ratio_ranges_z3p0_lf60.csv"))
k_sims = sims["k_s_per_km"]
eq6 = class_ratio(k_sims, 3.0)  # by class: "LLS", "Sub-DLA", "Small-DLA", "Large-DLA"

fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
ax = axes[0]
for stem, key in (("LLS", "LLS"), ("subDLA", "Sub-DLA")):
    wide = (sims[f"R_{stem}_sims60_allrungs_min"], sims[f"R_{stem}_sims60_allrungs_max"])
    narrow = (sims[f"R_{stem}_sims60_rung1_min"], sims[f"R_{stem}_sims60_rung1_max"])
    ax.fill_between(k_sims, *wide, color=CLASS_COLOURS[stem], alpha=0.18, lw=0)
    ax.fill_between(k_sims, *narrow, color=CLASS_COLOURS[stem], alpha=0.45, lw=0,
                    label=f"{stem}: 60 PRIYA simulations, troughs restored")
    ax.plot(k_sims, eq6[key], color=CLASS_COLOURS[stem], ls="--", lw=2, label=f"{stem}: Rogers et al. (2018), Eq. 6")
    inside_wide = (eq6[key] >= wide[0]) & (eq6[key] <= wide[1])
    inside_narrow = (eq6[key] >= narrow[0]) & (eq6[key] <= narrow[1])
    print(f"{stem:7s} Eq. 6 lies inside the range of the 60 simulations on {int(inside_narrow.sum())} of {k_sims.size} modes "
          f"at the mean-flux rung nearest 1, and on {int(inside_wide.sum())} at all mean-flux rungs")
ax.axvline(2.0 * np.pi / 8000.0, color="grey", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel("k [s/km]")
ax.set_ylabel("class spectrum / clean spectrum")
ax.set_title("troughs restored (lighter band: all mean-flux rungs;\ndotted: below it Eq. 6 is an extrapolation)")
ax.legend(fontsize=8)

ax = axes[1]
shown = pred.k < 0.03
ratio_mid = pred.P_class[1] / pred.P_class[0]
for name, style in (("ns", "-"), ("Ap", "--")):
    low, high = hp.PARAM_LIMITS[hp.PARAM_NAMES.index(name)]
    deviation = 0.0
    for value, shade in zip(np.linspace(low, high, 5), np.linspace(0.15, 0.85, 5)):
        other = emu.predict(dict(PARAMS, **{name: value}), z=3.0, tau0_amp=1.0, dtau0=0.0)
        if not np.allclose(other.k, pred.k, rtol=1e-12, atol=0.0):
            raise ValueError("the wavenumbers moved with the parameter")
        relative = (other.P_class[1] / other.P_class[0]) / ratio_mid
        deviation = max(deviation, float(np.max(np.abs(relative[shown] - 1.0))))
        ax.semilogx(pred.k[shown], relative[shown], style, color=plt.cm.viridis(shade), label=f"{name} = {value:.4g}")
    print(f"the emulator's LLS class ratio changes by up to {100 * deviation:.2f} per cent across the trained range of {name} (k < 0.03 s/km)")
ax.axhline(1.0, color="black", lw=2.5, label="Rogers et al. (2018) template: no dependence")
ax.set_xlabel("k [s/km]")
ax.set_ylabel("LLS class ratio / its value at the midpoint")
ax.set_title("the response to n_s (solid) and A_p (dashed)")
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
fig.suptitle("Fig. 8. The class ratios beside the template of Rogers et al. (2018), z = 3")
fig.tight_layout(rect=(0, 0.1, 1, 1))
plt.show()
""",
)

text(
    "rogers-caveat",
    r"""
**PROVISIONAL (PI to refine).** The two panels of Fig. 8 answer different questions. On the left the filtered
troughs are restored, so levels can be compared with the template; the printed lines say on how many modes the
template lies inside the range of the simulations. On the right the emulator's class spectra are PRIYA-filtered,
so their level is not compared with the template: only the response to the parameters is shown, which a
template with no dependence on cosmology does not carry.
""",
)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = build(argv[0] if argv else NOTEBOOK)
    print(f"{out}: {len(CELLS)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
