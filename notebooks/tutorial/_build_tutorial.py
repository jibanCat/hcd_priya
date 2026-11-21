"""Build the public tutorial notebook, ``notebooks/tutorial/hcd_priya_tutorial.ipynb``.

Every cell of the notebook is held here as source, in order, in ``CELLS``: a list of ``(kind, cell_id, source)``
with kind ``"markdown"`` or ``"code"``. ``build(path)`` writes them as a notebook without outputs; the committed
notebook is that file, executed. The helper functions the notebook needs are cells of the notebook, so the
notebook imports nothing beyond the installed packages; ``helper_source(cell_id)`` returns the source of one code
cell, which is how the tests (``tests/test_tutorial_notebook.py``) run the same code a reader runs.

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
# From HCD absorbers to the public likelihood: a tutorial for `hcd_priya`

**PROVISIONAL (PI to refine).** This notebook has three parts.

1. **Scientific motivation.** The Lyman-alpha forest and the high-column-density (HCD) absorbers on the same
   sightlines: Lyman limit systems (LLS), sub-damped and damped Lyman-alpha absorbers (subDLAs, DLAs).
2. **The emulator.** What the published model `hcd_priya` 1.0.0 returns: the flux power spectrum (P1D) of four
   classes of sightline, the incidence of the absorbers, and a mixture of the classes.
3. **The public likelihood and the mock-recovery example.** The package's Gaussian likelihood on the mock of the
   repository's example, at the mock's truth, and the stored result of that example redrawn. Nothing is fitted.

**To run it:** `pip install "hcd-priya[tutorials]"` in Python 3.11 or later, then open this file with Jupyter. The
trained model comes with the package. The few stored files the notebook reads, and the released scripts it runs,
are found in a clone of the repository or downloaded once, each checked against its sha256 (next cell).
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


def tutorial_cache():
    # the folder downloaded files are kept in
    return pathlib.Path(os.environ.get("HCD_PRIYA_TUTORIAL_CACHE", "~/.cache/hcd_priya_tutorial")).expanduser()


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
    cache = tutorial_cache()
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
    "fig3-incidence-response",
    r"""
# Fig. 3. The predicted incidence of the LLS class at z = 3 while one parameter at a time crosses the range it was
# trained on (five values; the other eight at the midpoint, the mean flux fixed).
Z0 = 3.0
I_LLS = hp.HCD_CLASSES.index("LLS")
mid = emu.predict(PARAMS, z=Z0, tau0_amp=1.0, dtau0=0.0)
mid_flags = {name for name, value in mid.flags.items() if value is True}

response, new_flags = {}, {}
for name, (low, high) in zip(hp.PARAM_NAMES, hp.PARAM_LIMITS):
    values = np.linspace(low, high, 5)
    rows = [emu.predict(dict(PARAMS, **{name: value}), z=Z0, tau0_amp=1.0, dtau0=0.0) for value in values]
    response[name] = (values, np.array([row.dndx[I_LLS] for row in rows]) / mid.dndx[I_LLS])
    new_flags[name] = sorted({f for row in rows for f, value in row.flags.items() if value is True} - mid_flags)

fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True)
for ax, (name, (values, lls)) in zip(axes.ravel(), response.items()):
    ax.plot(values, lls, "o-", color=CLASS_COLOURS["LLS"])
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_xlabel(name)
fig.supylabel("dN/dX of the LLS class / its value at the midpoint")
fig.suptitle("Fig. 3. Predicted incidence of the LLS class at z = 3 across the trained range of each parameter")
fig.tight_layout()
plt.show()

print(f"at the midpoint: dN/dX of the LLS class = {mid.dndx[I_LLS]:.4f}")
print("The change of the predicted LLS incidence across the trained range of each parameter, largest first")
print("(the largest of the five values over the smallest; the value at the low and at the high end over the midpoint):")
RANKING = sorted(response, key=lambda name: -(response[name][1].max() / response[name][1].min()))
for name in RANKING:
    values, lls = response[name]
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
    "helpers-released",
    r"""
# Running a released script of the repository as it is. The scripts and the tables they read are files of the tag
# v1.0.0, each pinned by its sha256 (ASSETS). They are put at their repository paths under one folder, the cache
# folder, because a script finds its data three folders above itself; then the script is run with this Python.
import shutil
import subprocess
import sys
import tarfile

RELEASED_FIGURE_FILES = (
    "scripts/figures/figures_common.py",
    "docs/figures/data/SHA256SUMS",
    "docs/figures/data/priya_lf_design_60.csv",
    "docs/figures/data/class_ratio_ranges_z3p0_lf60.csv",
    "docs/figures/data/ho21/cddf_z225.txt",
    "docs/figures/data/ho21/cddf_z253.txt",
    "docs/figures/data/ho21/cddf_z34.txt",
    "docs/figures/data/ho21/dndx_all.txt",
    "docs/figures/data/lls_incidence/lls_lz_literature.dat",
    "docs/figures/data/lls_subdla/zafar_2013.txt",
    "docs/figures/data/lls_subdla/omeara2007_fn_all19p0.dat",
    "docs/figures/data/lls_subdla/prochaska2010_lls_region.dat",
)
OUT = pathlib.Path("tutorial_output")  # the figures the released code writes, beside this notebook
OUT.mkdir(exist_ok=True)


def released_tree(relpaths):
    # the files named, each with its pinned sha256, at their repository paths under the cache folder
    root = tutorial_cache()
    for relpath in relpaths:
        source, target = tutorial_file(relpath), root / relpath
        if source != target and not (target.is_file() and sha256_of(target) == ASSETS[relpath]["sha256"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, partial = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
            os.close(handle)
            shutil.copyfile(source, partial)
            os.replace(partial, target)
    return root


def run_released_script(script, arguments, needs=RELEASED_FIGURE_FILES):
    # runs a released script, unchanged, and returns what it printed
    root = released_tree([script, *needs])
    done = subprocess.run(
        [sys.executable, str(root / script), *arguments],
        capture_output=True,
        text=True,
        env=dict(os.environ, MPLBACKEND="Agg"),
    )
    if done.returncode != 0:
        raise RuntimeError(f"the released script {script} ended with code {done.returncode}:\n{done.stderr[-2000:]}")
    return done.stdout


def unpack_archive(relpath, top, marker):
    # A pinned archive of the release, unpacked once into the cache folder. Every entry must be a plain file or
    # folder under the one folder `top`; anything else is refused and nothing is unpacked. `marker` is a file of the
    # archive whose presence says that an earlier unpacking was finished.
    root = tutorial_cache() / top
    if not (root / marker).is_file():
        archive_path = tutorial_file(relpath)
        tutorial_cache().mkdir(parents=True, exist_ok=True)
        scratch = pathlib.Path(tempfile.mkdtemp(dir=tutorial_cache(), prefix="unpack."))
        try:
            with tarfile.open(archive_path) as archive:
                members = archive.getmembers()
                for member in members:
                    inside = pathlib.PurePosixPath(member.name)
                    plain = member.isfile() or member.isdir()
                    if inside.is_absolute() or ".." in inside.parts or inside.parts[0] != top or not plain:
                        raise RuntimeError(f"{relpath} holds an entry that is not a plain file under {top}: {member.name}")
                safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                archive.extractall(scratch, members=members, **safe)
            if root.exists():  # an unfinished copy from an earlier run
                shutil.rmtree(root)
            os.replace(scratch / top, root)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return root


def show_script_output(printed):
    # what a released script printed; its indented detail lines are counted, not shown
    lines = printed.splitlines()
    for line in lines:
        if line.strip() and not line.startswith(" "):
            print("the script printed:", line)
    detail = sum(1 for line in lines if line.startswith(" ") and line.strip())
    if detail:
        print(f"({detail} indented detail lines of the script are not shown here)")
""",
)


code(
    "fig7-incidence",
    r"""
# Fig. 7. The incidence dN/dX the model predicts, against redshift, with published measurements: drawn by the
# released script scripts/figures/fig_dndx_vs_ho21.py, unchanged, with the published model.
show_script_output(run_released_script("scripts/figures/fig_dndx_vs_ho21.py", ["--published", "--out", str(OUT / "dndx_vs_ho21.png")]))
display(Image(filename=str(OUT / "dndx_vs_ho21.png"), width=900))

print("At z = 3 at the midpoint, the model's dN/dX with its held-out range (value x exp(-e) to value x exp(+e)):")
for c, name in enumerate(hp.HCD_CLASSES):
    print(f"  {name:7s} {pred.dndx[c]:.4f}   range {pred.dndx_lower[c]:.4f} to {pred.dndx_upper[c]:.4f}")
""",
)


text(
    "fig7-caption",
    r"""
**Fig. 7. PROVISIONAL (PI to refine).** The incidence dN/dX the model predicts for the three classes, and for all
absorbers above $10^{17.5}$ cm$^{-2}$, against redshift, with published measurements. Drawn by the released script
`fig_dndx_vs_ho21.py`, unchanged; the full caption is in `docs/model_figures.md` of the repository.
""",
)


code(
    "fig7b-cddf",
    r"""
# Fig. 7b. The column density distribution the model predicts, with published measurements: drawn by the released
# script scripts/figures/fig_cddf_vs_ho21.py, unchanged, with the published model.
show_script_output(run_released_script("scripts/figures/fig_cddf_vs_ho21.py", ["--published", "--out", str(OUT / "cddf_vs_ho21.png")]))
display(Image(filename=str(OUT / "cddf_vs_ho21.png"), width=900))
""",
)


text(
    "fig7b-caption",
    r"""
**Fig. 7b. PROVISIONAL (PI to refine).** The column density distribution the model predicts, in three redshift
bins, with the published measurements of Fig. 2. Drawn by the released script `fig_cddf_vs_ho21.py`, unchanged;
the full caption is in `docs/model_figures.md` of the repository.
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
Z_VALIDATED = [z for z in hp.Z_GRID if 2.2 - 1e-9 <= z <= 4.6 + 1e-9]
validated = emu.predict_many(PARAMS, z=Z_VALIDATED, tau0_amp=1.0, dtau0=0.0)
e_mix = validated.P_mix_validation_rms_log  # one row per redshift of Z_VALIDATED, one value per wavenumber
print("The held-out prediction residual e of the mixture, by redshift (the value x exp(-e) to x exp(+e)):")
for j, z in enumerate(Z_VALIDATED):
    e, k = e_mix[j], validated.k[j]
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
# Fig. 8, left. The PRIYA training simulations with the filtered troughs restored, against Eq. 6 of Rogers et al.
# (2018), at z = 3: drawn by the released script scripts/figures/fig_class_ratio_vs_rogers18.py, unchanged.
show_script_output(run_released_script("scripts/figures/fig_class_ratio_vs_rogers18.py", ["--out", str(OUT / "class_ratio_vs_rogers18.png")]))
display(Image(filename=str(OUT / "class_ratio_vs_rogers18.png"), width=900))
""",
)


code(
    "fig8-response",
    r"""
# Fig. 8, right. The response of the emulator's LLS class ratio to n_s, A_p and the mean flux, and a ranking over
# all inputs. The template of Rogers et al. (2018) is hcd_template.class_ratio(k, z): it takes a wavenumber and a
# redshift, and nothing else.
from hcd_analysis.hcd_template import class_ratio

shown = pred.k < 0.03  # the box modes below 0.03 s/km at the midpoint
ratio_mid = pred.P_class[1] / pred.P_class[0]
eq6_lls = class_ratio(pred.k[shown], 3.0)["LLS"]
print(f"Eq. 6 of Rogers et al. (2018) for the LLS class at z = 3, on these wavenumbers: {eq6_lls.min():.3f} to {eq6_lls.max():.3f};")
print("it is a function of k and z alone, so across the inputs below it does not change")


def lls_ratio_relative(name, value):
    # the LLS class ratio with one input changed, over its value at the midpoint, box mode by box mode (hub and
    # omegamh2 move the wavenumber in s/km of a mode; the ratio is compared at the same mode)
    params = dict(PARAMS, **({name: value} if name in PARAMS else {}))
    other = emu.predict(params, z=3.0, tau0_amp=value if name == "tau0_amp" else 1.0, dtau0=0.0)
    return (other.P_class[1] / other.P_class[0]) / ratio_mid


INPUTS = list(zip(hp.PARAM_NAMES, hp.PARAM_LIMITS)) + [("tau0_amp", TAU0_AMP_RANGE)]
largest = {}
fig, ax = plt.subplots(figsize=(9, 6.2))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for name, (low, high) in INPUTS:
        largest[name] = max(
            float(np.max(np.abs(lls_ratio_relative(name, value)[shown] - 1.0))) for value in np.linspace(low, high, 5)
        )
    for name, style in (("ns", "-"), ("Ap", "--")):
        low, high = hp.PARAM_LIMITS[hp.PARAM_NAMES.index(name)]
        for value, shade in zip(np.linspace(low, high, 5), np.linspace(0.15, 0.85, 5)):
            ax.semilogx(pred.k[shown], lls_ratio_relative(name, value)[shown], style, color=plt.cm.viridis(shade),
                        label=f"{name} = {value:.4g}")
    for value, colour in ((0.8, "tab:red"), (1.2, "tab:brown")):
        ax.semilogx(pred.k[shown], lls_ratio_relative("tau0_amp", value)[shown], ":", lw=2.2, color=colour,
                    label=f"tau0_amp = {value:g}")
ax.axhline(1.0, color="black", lw=2.5, label="Rogers et al. (2018) template: no dependence")
ax.set_xlabel("k [s/km]")
ax.set_ylabel("LLS class ratio / its value at the midpoint")
ax.set_title("Fig. 8, right. The response of the emulator's LLS class ratio at z = 3\nto n_s (solid), A_p (dashed) and the mean flux (dotted)")
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
plt.show()

print("The largest change of the emulator's LLS class ratio below k = 0.03 s/km at z = 3 across the trained range of")
print("each input (five values; the others at the midpoint, tau0_amp = 1), largest first:")
for name in sorted(largest, key=lambda name: -largest[name]):
    print(f"  {name:11s} {100 * largest[name]:5.2f} per cent")
print("dtau0 is not varied: at one redshift it acts only through tau_eff, which tau0_amp already covers")
if caught:
    print(f"{len(caught)} of these predictions warned, the first: {str(caught[0].message)[:160]} ...")
""",
)

text(
    "rogers-caveat",
    r"""
**Fig. 8. PROVISIONAL (PI to refine).** The two parts of Fig. 8 answer different questions. On the left (drawn by
the released script `fig_class_ratio_vs_rogers18.py`, unchanged; the full caption is in `docs/rogers_comparison.md`
of the repository) the filtered troughs are restored, so levels can be compared with the template. On the right
the emulator's class spectra are PRIYA-filtered, so their level is not compared with the template: only the
response to the inputs is shown, which a template with no dependence on cosmology does not carry. The printed
list ranks the inputs by the size of that response.
""",
)


# ============================================================================================ Part III
text(
    "part-3",
    r"""
## Part III. The public likelihood and the mock-recovery example

**PROVISIONAL (PI to refine).** This part uses the package's own Gaussian likelihood and the mock-recovery example
of the repository, with the files of the release. Nothing is fitted here.

### The Gaussian likelihood

`hcd_analysis.likelihood.GaussianP1DLikelihood` compares a data vector $d$ with covariance $C$ to a model vector
$m(\theta)$:

$$\ln L(\theta) = -\tfrac{1}{2}\,[d - m(\theta)]^{T} C^{-1} [d - m(\theta)] - \tfrac{1}{2} \ln\det C - \tfrac{N}{2}\ln 2\pi .$$

The class builds the model vector in four steps: the prediction at $\theta$ at the redshifts of the data; the
mixture of the class spectra (`hp.add_hcd`); linear interpolation in $k$ to the wavenumbers of the data; and the
data set's own factor on the model, when it has one. `docs/likelihood.md` of the repository describes the class,
its options and its approximations.
""",
)

code(
    "example-files",
    r"""
# The files of the mock-recovery example: the example download of the release, unpacked into the cache folder, and
# the example script of the repository at the tag v1.0.0, loaded from its file. Its function open_download returns
# the model, the mock and the stored reference, each read against the sha256 written in the script.
import importlib.util

from hcd_analysis import likelihood as hl


spec = importlib.util.spec_from_file_location("mock_recovery", tutorial_file("examples/mock_recovery.py"))
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

example_root = unpack_archive("hcd_priya_example_download.tar.gz", "hcd_priya_example_download", "example_download.json")
emu_heldout, data, reference = example.open_download(str(example_root))
print(f"the mock: {data.n} bins at {len(data.z)} redshifts, from z = {min(data.z):.1f} to {max(data.z):.1f}")
print("what the mock is, in its own words:")
print("  ", data.provenance["what"])
print("what the stored reference is, in its own words:")
print("  ", reference.provenance["what"])
print("how its chains were run:", reference.provenance["chains"])
print("its limits:", reference.provenance["limits"])
""",
)

code(
    "which-model",
    r"""
# Which model is which.
production = emu.provenance["release"]
print("Parts I and II: the published production model, from hp.load_published()")
print("  release", production["release_version"], "| sha256 of its release.json:", production["release_sha256"])
print("  kind:", emu.provenance.get("model_kind", "a released model"), "| manifest:", emu.manifest_sha256)
print()
print("Part III: the held-out model of the example download, from the example's open_download()")
print("  kind:", emu_heldout.provenance["model_kind"], "| manifest:", emu_heldout.manifest_sha256)
print("  its stated purpose:", emu_heldout.load_purpose)
print("  simulations held out of its training:", len(emu_heldout.provenance["research"]["split"]["held_out_sims"]),
      "(the simulation the mock was made from)")
print("  (the next cell prints the flags of its prediction at the truth: research_model is among them)")
""",
)

text(
    "which-model-note",
    r"""
**PROVISIONAL (PI to refine).** Parts I and II use the published production model. Part III uses the held-out
model of the example download, because the mock-recovery example of the paper does: the mock comes from a
simulation, and the held-out model was trained without that simulation.
""",
)

code(
    "likelihood-at-truth",
    r"""
# COMPUTED HERE: the likelihood at the point the mock was made at, by the example's own part_one. Warnings are
# caught and printed below, not silenced.
like = hl.GaussianP1DLikelihood(emu_heldout, data, gaussian_terms=example.GAUSSIAN_TERMS)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    truth = example.part_one(like)

print()
print(f"warnings raised while this ran: {len(caught)}")
for warning in caught:
    print(f"  {warning.category.__name__}: {warning.message}")
""",
)

text(
    "heldout-warning",
    r"""
**PROVISIONAL (PI to refine).** The warning printed above comes from the held-out model. It concerns the dN/dX
and the column density distribution of that model, which carry no held-out error. The likelihood uses neither.
""",
)

code(
    "three-chi-squares",
    r"""
# COMPUTED HERE: how the HCD content enters the likelihood. The public option hcd_relative_to_priya sets the
# amplitude of a class in the mixture relative to the model's own sightline fraction.
print("three evaluations of the public likelihood at the mock's truth; nothing is fitted and these are not constraints")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for r_lls in (0.0, 1.0, 2.0):
        scaled = hl.GaussianP1DLikelihood(
            emu_heldout, data, gaussian_terms=example.GAUSSIAN_TERMS,
            hcd_relative_to_priya={"LLS": r_lls, "subDLA": 1.0, "DLA": 1.0},
        )
        print(f"  LLS amplitude at {r_lls:g} times the model's own fraction (subDLA and DLA at 1): chi-square = {scaled.chi2(truth):.4f}")
print(f"warnings raised while this ran: {len(caught)}, of the kind", sorted({w.category.__name__ for w in caught}) or "none")
""",
)

code(
    "fig9-fig10-reference",
    r"""
# REDRAWN HERE from the stored reference of the release (a summary of converged chains made before the release);
# NOT RECOMPUTED. The example's own draw_reference draws both figures from the stored reference and the mock alone.
intervals_png, per_redshift_png = example.draw_reference(reference, data, str(OUT))
plt.close("all")
print("Fig. 9:", intervals_png)
display(Image(filename=intervals_png, width=900))
print("Fig. 10:", per_redshift_png)
display(Image(filename=per_redshift_png, width=900))
""",
)

text(
    "fig9-fig10-caption",
    r"""
**Fig. 9 (the paper's Figure 4). PROVISIONAL (PI to refine).** For each of the eleven parameters, the mean and the
central 68 and 95 per cent intervals of the stored reference, with the truth of the mock. Redrawn here from the
stored reference of the release (a summary of converged chains made before the release); not recomputed.

**Fig. 10. PROVISIONAL (PI to refine).** The mock and the stored bands of the posterior predictive of the
reference, one panel per redshift. Redrawn here from the stored reference of the release (a summary of converged
chains made before the release); not recomputed.
""",
)

text(
    "what-is-reproduced",
    r"""
### What can and cannot be reproduced

**PROVISIONAL (PI to refine).**

| Result | From the published release? |
|---|---|
| Every figure of Parts I and II; the likelihood at the truth of the mock; the three chi-square values | computed when the notebook runs |
| The reference posterior of the mock (the paper's Figure 4; Fig. 9 and 10 here) | read from the stored reference of the release, a summary of converged chains made before the release; not recomputed |
| A chain of one's own on the mock | outside this notebook: `python examples/mock_recovery.py --download DIR --full`, which needs Cobaya; not run here |
| The contours of the paper's Figure 3 and its comparison with the Gaussian-process emulator | cannot be fully reproduced from the public files, because the chains are not public |
""",
)

text(
    "limits",
    r"""
## Limits, and where to go next

**PROVISIONAL (PI to refine).**

- Part III evaluates the public likelihood at the truth of a mock and redraws a stored result. Nothing in this
  notebook is fitted, and nothing here is a constraint from data.
- The likelihood uses the covariance of the data alone. The held-out prediction residual of Part II is an
  empirical validation diagnostic, not a calibrated predictive uncertainty, and it does not enter the likelihood.
- The class spectra are PRIYA-filtered, and the mixture leaves out what the filter removed.

Where to go next, in the repository: `docs/likelihood.md` (the likelihood, its options and what it refuses);
`examples/mock_recovery.py` (the example used above, with its chain modes); `docs/model_figures.md` (the model
against published measurements); `docs/p1d_definition.md` (the conventions).
""",
)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = build(argv[0] if argv else NOTEBOOK)
    print(f"{out}: {len(CELLS)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
