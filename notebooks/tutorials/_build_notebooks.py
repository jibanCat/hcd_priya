"""
Build the four student-facing tutorial notebooks under notebooks/tutorials/.

Run once::

    python3 notebooks/tutorials/_build_notebooks.py

Then execute to verify::

    for nb in notebooks/tutorials/*.ipynb; do
      jupyter nbconvert --to notebook --execute "$nb" --inplace
    done

The output notebooks are committed to the repo; this script is here so the
notebooks can be regenerated cleanly when the underlying code or data
schema changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT_DIR = Path(__file__).parent
KERNEL = {"name": "python3", "display_name": "Python 3"}
LANG = {"name": "python", "version": "3.11"}


def md(text: str):
    # Strip a single leading newline so triple-quoted blocks below read naturally.
    if text.startswith("\n"):
        text = text[1:]
    return new_markdown_cell(text)


def code(text: str):
    if text.startswith("\n"):
        text = text[1:]
    return new_code_cell(text)


def write_notebook(name: str, cells: list) -> Path:
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": KERNEL,
        "language_info": LANG,
    })
    path = OUT_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path}")
    return path


# ============================================================================
# Notebook 00 — dataset layout, parameter encoding, snapshot map
# ============================================================================

def build_00():
    cells = [
        md("""
# 00 — Dataset layout

This is the first of four tutorial notebooks for the HCD (high-column-density
absorber) catalog and emulator-input dataset.  By the end of the four notebooks
you will be able to:

| Notebook | What you'll learn |
|---|---|
| 00 | Where the data lives, how the parameter space is encoded in folder names, how snapshots map to redshift, what files exist for each (sim, snap). |
| 01 | How to read the per-sim **absorber catalog** (`catalog.npz`) and the underlying **raw spectra** HDF5 file; visualise a single DLA. |
| 02 | How to read the cached **CDDF** and **dN/dX** for one (sim, snap) and how to recompute them from the catalog (and reproduce a literature comparison plot). |
| 03 | How to read the cached **per-class P1D** templates (`p1d_per_class.h5`) and how to recompute them from raw spectra + catalog using the project's own helpers. |

Why this matters: the per-sim outputs in this dataset are the inputs to the
HCD emulator we are building.  Before you can train or evaluate that emulator
you need to be confident that you understand — and can reproduce — the four
observables on disk.

**Audience.** A new graduate student or RA picking up this project for the
first time.  We assume comfort with numpy / matplotlib / h5py and basic
familiarity with the Lyman-α forest as a cosmological probe.  We do not
assume any prior exposure to this codebase.
"""),

        md("""
## 1. The two storage locations

The data this project uses lives in two places, and they have very
different sizes:

| Path | Contents | Per-file size |
|---|---|---|
| `/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/output/SPECTRA_NNN/lya_forest_spectra_grid_480.hdf5` | **Raw** Lyman-α optical-depth skewers (`tau`), 691 200 sightlines on a regular 480² grid × the LOS axis | ~3 GB |
| `/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/<sim>/snap_NNN/` | **Processed** per-(sim, snap) outputs — absorber catalog, CDDF, P1D variants, per-class P1D, metadata | ~few MB total |

A few rules to internalise:

* The **raw** files are 3 GB each and there are ~1000 of them — never
  copy them, never load a full file into memory, always stream them in
  batches (we'll show how in notebook 01).
* The **processed** files are tiny — small enough that the whole 1076-row
  dataset fits in memory once aggregated.  These are what the emulator
  trains on.
* Both locations are read-only from your point of view as a student.
  Generate any new outputs into a fresh directory under your scratch.
"""),

        code("""
# Imports used throughout this notebook
from pathlib import Path
import numpy as np

DATA_ROOT = Path('/nfs/turbo/umor-yueyingn/mfho/emu_full')
HCD_OUT_ROOT = Path('/scratch/cavestru_root/cavestru0/mfho/hcd_outputs')

print('raw spectra root :', DATA_ROOT,  '->', DATA_ROOT.exists())
print('processed root   :', HCD_OUT_ROOT, '->', HCD_OUT_ROOT.exists())
"""),

        md("""
## 2. Simulation folder names encode the parameter point

Each simulation directory under `HCD_OUT_ROOT` is named like:

```
ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333
```

The nine parameters that appear in that name are the dimensions of the
emulator parameter space:

| Symbol | Meaning | Approx range |
|---|---|---|
| `ns` | spectral index of primordial power | 0.80 – 1.05 |
| `Ap` | amplitude `A_s` (10⁻⁹) | 1.2e-9 – 2.6e-9 |
| `herei` | start z of HeII reionisation | 3.5 – 4.5 |
| `heref` | end z of HeII reionisation | 2.6 – 3.2 |
| `alphaq` | quasar SED spectral index | 1.3 – 3.0 |
| `hub` | dimensionless Hubble `h` | 0.65 – 0.75 |
| `omegamh2` | matter density `Ω_m h²` | 0.140 – 0.146 |
| `hireionz` | start z of HI reionisation | 6.5 – 8.0 |
| `bhfeedback` | black-hole feedback amplitude | 0.03 – 0.07 |

The package gives you a parser so you don't have to write a regex by hand:
"""),

        code("""
from hcd_analysis.io import parse_sim_params, discover_simulations

example = 'ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333'
params = parse_sim_params(example)
for k, v in params.items():
    print(f'  {k:12s} = {v}')
"""),

        md("""
## 3. The full simulation list

`discover_simulations()` walks a directory and returns every folder whose
name matches the 9-parameter pattern.  Use it on `HCD_OUT_ROOT` to see all
LF (low-fidelity, 60 sims) entries; the 4 HR (high-fidelity) sims live in
the `hires/` subdirectory.
"""),

        code("""
sims_lf = discover_simulations(HCD_OUT_ROOT)
sims_hr = discover_simulations(HCD_OUT_ROOT / 'hires')

print(f'LF sims discovered: {len(sims_lf)}')
print(f'HR sims discovered: {len(sims_hr)}')

# Peek at the first three so you can see the SimInfo structure
for s in sims_lf[:3]:
    print(s.name[:60], ' -> ns=%.3f Ap=%.2e' % (s.params['ns'], s.params['Ap']))
"""),

        md("""
## 4. Snapshot ↔ redshift mapping

Each simulation writes ~20 snapshots between z = 6 and z = 2.  The mapping
from snapshot index to scale factor `a` lives in
`<sim>/output/Snapshots.txt`; redshift is `z = 1/a − 1`.

The standard snapshots and their redshifts (constant across simulations,
because PRIYA picks a fixed list of `a` values):

| Snap | z    |  | Snap | z    |
|------|------|--|------|------|
| 004  | 5.40 |  | 014  | 3.60 |
| 005  | 5.20 |  | 015  | 3.40 |
| 006  | 5.00 |  | 016  | 3.20 |
| 007  | 4.80 |  | 017  | 3.00 |
| 008  | 4.60 |  | 018  | 2.80 |
| 009  | 4.40 |  | 019  | 2.60 |
| 010  | 4.20 |  | 020  | 2.56 |
| 011  | 4.00 |  | 021  | 2.40 |
| 012  | 3.80 |  | 022  | 2.20 |
| 013  | 3.74 |  | 023  | 2.00 |

**Two practical caveats:**

1. Not every sim has every snapshot.  Snap 022 (z = 2.2) and snap 023
   (z = 2.0) are missing in many sims — sims that were not run all the
   way down to z = 2.  Always check before assuming a (sim, snap) is
   present.
2. We ignore snap 016 (z = 3.20) for some analyses where it overlaps
   awkwardly with neighbouring snaps; see the analysis docs for context.
"""),

        code("""
from hcd_analysis.io import read_snapshots_txt

# Note: read_snapshots_txt expects the *raw-data* sim path (with output/Snapshots.txt),
# not the hcd_outputs path. So we point at the Turbo location.
from hcd_analysis.io import SimInfo
example_sim = SimInfo(
    name=example,
    path=DATA_ROOT / example,
    params=params,
)
snap_to_a = read_snapshots_txt(example_sim)

print(f'{len(snap_to_a)} snapshots in Snapshots.txt for this sim:')
print(f'{"snap":>4}  {"a":>8}  {"z":>6}')
for snap in sorted(snap_to_a):
    a = snap_to_a[snap]
    z = 1.0 / a - 1.0
    if 2.0 <= z <= 6.0:
        print(f'{snap:>4d}  {a:>8.4f}  {z:>6.3f}')
"""),

        md("""
## 5. What's inside each `<sim>/snap_NNN/` directory

For every (sim, snap) pair the pipeline writes a small set of files into
`HCD_OUT_ROOT/<sim>/snap_NNN/`.  The ones you'll touch most often:

| File | Contents | Tutorial |
|---|---|---|
| `meta.json` | Sim name, snap, z, dv_kms, n_skewers, parsed `sim_ics`, absorber counts | NB 01 |
| `catalog.npz` | Per-absorber records: `skewer_idx`, `pix_start`, `pix_end`, `NHI`, `b_kms`, `fit_success`, `fast_mode` | NB 01 |
| `cddf_corrected.npz` | Column-density distribution function on a 30-bin log10 NHI grid (the **corrected** version, after the absorption-path bug fix) | NB 02 |
| `p1d_per_class.h5` | Per-class P1D templates (`P_clean`, `P_LLS_only`, `P_subDLA_only`, `P_DLA_only`) on the sim's native rfftfreq k-grid | NB 03 |

A few less-used files — you may see them but should generally ignore unless
you have a specific reason:

* `cddf.npz` — original (buggy) CDDF before the `(1+z)·h` absorption-path
  fix.  **Always prefer `cddf_corrected.npz`.**
* `p1d.npz`, `p1d_excl.npz`, `p1d_ratios.npz` — older / aggregate P1D
  variants superseded by `p1d_per_class.h5` for emulator inputs.
* `done` — empty marker file the pipeline writes when the (sim, snap) is
  fully processed.

Let's verify the files exist for one (sim, snap):
"""),

        code("""
import json
SIM_DIR = HCD_OUT_ROOT / example
SNAP = 22
SNAP_DIR = SIM_DIR / f'snap_{SNAP:03d}'

print('Listing', SNAP_DIR)
for p in sorted(SNAP_DIR.iterdir()):
    print(f'  {p.name:30s}  {p.stat().st_size:>10d} B')

# Quick peek at meta.json so you know it's just a JSON
with open(SNAP_DIR / 'meta.json') as f:
    meta = json.load(f)
print()
print('meta.json keys:', list(meta.keys()))
print('  z          =', meta['z'])
print('  dv_kms     =', meta['dv_kms'])
print('  n_skewers  =', meta['n_skewers'])
print('  n_absorbers=', meta['n_absorbers'])
"""),

        md("""
## 6. One important wrinkle: the k-grid is not shared across sims

The per-class P1D `k` array on disk uses numpy's `rfftfreq` convention,
and its length is `nbins/2 + 1`.  `nbins` is the number of velocity
pixels per skewer, which depends on the box size in km/s — and that
depends on `H(z)` (cosmology-dependent) and the sim's box size.  So:

* **Different sims have different `nbins`** at the same snap (e.g. 1141,
  1228, 1268 are all real values seen in the LF set at snap 022).
* **Different snaps within one sim have different `nbins`** because
  `H(z)` changes with redshift.
* The `k` arrays therefore have different lengths across (sim, snap)
  pairs (e.g. 553–635 entries at snap 022 across 60 sims).

This matters for emulator construction: you can't just stack the on-disk
P1Ds into a `(N_sims, n_k)` matrix without first interpolating onto a
shared k-grid.  We'll come back to this in notebook 03; for now just
remember it.
"""),

        md("""
## Where to next

* **Notebook 01** — open `catalog.npz` and the raw spectra HDF5; visualise
  a single DLA in flux space.
* **Notebook 02** — recompute the CDDF and dN/dX from the catalog and
  reproduce a literature-comparison figure.
* **Notebook 03** — recompute the per-class P1D from raw spectra +
  catalog using the project helpers.

If you ever want a quick reference for any of the file schemas, run::

    python3 -c "import h5py; f = h5py.File('<path>', 'r'); f.visititems(lambda n,o: print(n,o))"

— that prints the full HDF5 tree and is the fastest way to remind yourself
what's in a file.
"""),
    ]
    return write_notebook("00_dataset_layout.ipynb", cells)


# ============================================================================
# Notebook 01 — reading catalogs and raw spectra
# ============================================================================

def build_01():
    cells = [
        md("""
# 01 — Reading the absorber catalog and the raw spectra

In notebook 00 you saw that every (sim, snap) has a small `catalog.npz` of
per-absorber records and a much larger raw-spectra HDF5 file holding the
optical depth on every sightline.  Here you will:

1. Load `catalog.npz` and inspect the per-absorber records.
2. Load `meta.json` and confirm the catalog is internally consistent
   (counts in `meta.json` match what's in `catalog.npz`).
3. Open the raw spectra HDF5 file *without* loading it into memory.
4. Pick one DLA from the catalog and visualise its `tau(v)` and `F(v)`.

The end-state mental model: a DLA is just a contiguous run of high-`tau`
pixels on one specific sightline, identified and characterised by the
catalog builder in `hcd_analysis.catalog`.
"""),

        code("""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py

DATA_ROOT    = Path('/nfs/turbo/umor-yueyingn/mfho/emu_full')
HCD_OUT_ROOT = Path('/scratch/cavestru_root/cavestru0/mfho/hcd_outputs')

# A sim known to have snap_022 fully processed (used in NB00):
SIM = 'ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333'
SNAP = 22

SIM_DIR  = HCD_OUT_ROOT / SIM
SNAP_DIR = SIM_DIR / f'snap_{SNAP:03d}'
RAW_FILE = DATA_ROOT / SIM / 'output' / f'SPECTRA_{SNAP:03d}' / 'lya_forest_spectra_grid_480.hdf5'

print('catalog dir :', SNAP_DIR)
print('raw file    :', RAW_FILE, '->', RAW_FILE.exists())
"""),

        md("""
## 1. `meta.json` — the per-(sim, snap) metadata

This is just a plain JSON file.  Read it first because it tells you what
snapshot you are looking at, what cosmology was used, and how many
absorbers of each class were found.  Anything in `meta.json` is the
authoritative answer for that (sim, snap).
"""),

        code("""
with open(SNAP_DIR / 'meta.json') as f:
    meta = json.load(f)

# Pretty-print the structured fields
import textwrap
print('Top-level meta:')
for k, v in meta.items():
    if isinstance(v, dict):
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'      {kk:20s} = {vv}')
    else:
        print(f'  {k:14s} = {v}')
"""),

        md("""
A few things worth noting in the above:

* `n_skewers = 691200` — 691 200 sightlines per snapshot.  This is `2 ×
  3 × 480²` (the factor of 3 is the LOS axis, factor of 2 because each
  axis position is sampled twice for the regular grid).  Streaming 3 GB
  is fast; loading it whole is a mistake.
* `dv_kms` is the pixel velocity width.  At z = 2.2 this is ~10 km/s.
  It varies smoothly with z and a bit across sims.
* `n_absorbers` is the **total absorber count** (not per sightline).  The
  classes are LLS / subDLA / DLA defined by:

  | Class | log₁₀ NHI |
  |---|---|
  | LLS    | [17.2, 19.0) |
  | subDLA | [19.0, 20.3) |
  | DLA    | ≥ 20.3       |

  These boundaries are fixed across the project — they are the
  conventional Wolfe & Prochaska classifications.
"""),

        md("""
## 2. `catalog.npz` — one row per absorber

`catalog.npz` stores parallel arrays, one entry per absorber found.
Total size = `n_LLS + n_subDLA + n_DLA` (e.g. 81 762 absorbers in the
example sim).
"""),

        code("""
cat = np.load(SNAP_DIR / 'catalog.npz', allow_pickle=True)

print('Fields and their shapes:')
for k in cat.files:
    a = cat[k]
    shape_str = str(getattr(a, 'shape', '()'))
    print(f'  {k:14s} shape={shape_str:>12} dtype={a.dtype}')
print()
print('First 5 rows of the per-absorber arrays:')
n_show = 5
print(f'{"idx":>5} {"skewer":>8} {"pix_lo":>6} {"pix_hi":>6} {"log NHI":>8} {"b kms":>6}  fit  fast')
for i in range(n_show):
    print(f'{i:>5} {int(cat["skewer_idx"][i]):>8} {int(cat["pix_start"][i]):>6} {int(cat["pix_end"][i]):>6} '
          f'{np.log10(cat["NHI"][i]):>8.3f} {cat["b_kms"][i]:>6.1f}'
          f'  {bool(cat["fit_success"][i])!s:>5} {bool(cat["fast_mode"][i])!s:>5}')
"""),

        md("""
Field-by-field:

* **`skewer_idx`** — index into the raw-spectra HDF5 row dimension; pulls
  out which sightline this absorber lives on.
* **`pix_start`, `pix_end`** — first and last pixel of the absorber along
  that sightline (inclusive).  For systems that wrap across the periodic
  box boundary, `pix_end >= nbins`; the wrapped portion is `[pix_start:]`
  and `[:pix_end - nbins + 1]`.
* **`NHI`** — column density in cm⁻² (use `log10(NHI)` for plots).
* **`b_kms`** — Doppler parameter from the Voigt fit (NaN in `fast_mode`).
* **`fit_success`** — True if the Voigt fit converged.  False rows are
  rare and you can usually drop them for plots.
* **`fast_mode`** — True if `NHI` came from the fast tau-integral
  estimator instead of a Voigt fit.  This is the production setting for
  the LF run; the HR run uses Voigt fits.

A consistency check: the per-class counts in `meta.json` should match
what we get from binning `NHI` ourselves.
"""),

        code("""
log_nhi = np.log10(cat['NHI'])

n_lls    = int(((log_nhi >= 17.2) & (log_nhi < 19.0)).sum())
n_subdla = int(((log_nhi >= 19.0) & (log_nhi < 20.3)).sum())
n_dla    = int( (log_nhi >= 20.3).sum())

print(f'computed: LLS={n_lls:,}  subDLA={n_subdla:,}  DLA={n_dla:,}')
print(f'meta:     LLS={meta["n_absorbers"]["LLS"]:,}  '
      f'subDLA={meta["n_absorbers"]["subDLA"]:,}  '
      f'DLA={meta["n_absorbers"]["DLA"]:,}')
"""),

        md("""
## 3. log NHI distribution

A canonical first plot: the per-absorber log NHI distribution, with the
class boundaries marked.  The catalog goes down to log NHI ≈ 17.2 by
construction (any absorber below that is treated as part of the smooth
forest, not an HCD).
"""),

        code("""
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(log_nhi, bins=np.arange(17.0, 22.0, 0.1), histtype='stepfilled', alpha=0.6)
for boundary, label in [(17.2, 'LLS'), (19.0, 'subDLA'), (20.3, 'DLA')]:
    ax.axvline(boundary, color='k', linestyle='--', linewidth=1)
    ax.text(boundary + 0.02, ax.get_ylim()[1] * 0.85, f'≥ {label}', fontsize=9)
ax.set(xlabel='log10 N_HI', ylabel='absorbers per 0.1 dex', title=f'{SIM[:50]}…  snap {SNAP}  z={meta["z"]}')
ax.set_yscale('log')
fig.tight_layout()
plt.show()
"""),

        md("""
## 4. The richer interface: `AbsorberCatalog`

`hcd_analysis.catalog.AbsorberCatalog` is a dataclass that wraps the same
arrays with helpful accessors (`by_class`, `summary`, `to_dataframe`).  Use
it when you want to operate on absorbers as objects rather than arrays.
"""),

        code("""
from hcd_analysis.catalog import AbsorberCatalog

ac = AbsorberCatalog.load_npz(SNAP_DIR / 'catalog.npz')
print('summary:', ac.summary())   # should match meta.json
dlas = ac.by_class('DLA')
print(f'first DLA in catalog: skewer={dlas[0].skewer_idx} '
      f'pix=({dlas[0].pix_start}, {dlas[0].pix_end}) log NHI={dlas[0].log_NHI:.3f}')
"""),

        md("""
## 5. Opening the raw spectra HDF5 — without loading it

The raw file is ~3 GB.  The trick is to keep it on disk and stream only
the rows you need.  h5py datasets behave like numpy arrays for slicing
but the data is only fetched for the slice you ask for.

The full schema is in `docs/data_layout.md`.  The two datasets you almost
always touch:

* `tau/H/1/1215`  — shape `(n_skewers, nbins)`, float32, optical depth.
* `spectra/cofm`  — shape `(n_skewers, 3)`, float64, sightline starting
  position in **kpc/h** (comoving).

Everything else (`colden`, `tau_obs`, `temperature`, `velocity`) exists
but is **empty** — they were not stored.  Don't try to read from them.
"""),

        code("""
with h5py.File(RAW_FILE, 'r') as f:
    print('Header attrs:')
    for k, v in f['Header'].attrs.items():
        print(f'  {k:10s} = {v}')
    print('Datasets:')
    for name in ('tau/H/1/1215', 'spectra/cofm', 'spectra/axis'):
        d = f[name]
        print(f'  {name:20s} shape={d.shape} dtype={d.dtype}')
"""),

        md("""
A quick sanity check on the velocity-pixel width: `pixel_dv_kms()` from
`hcd_analysis.io` recomputes `dv_kms` from cosmology and should match
the value cached in `meta.json`.
"""),

        code("""
from hcd_analysis.io import read_header, pixel_dv_kms
hdr = read_header(RAW_FILE)
dv_recomputed = pixel_dv_kms(hdr)
print(f'meta.dv_kms       = {meta["dv_kms"]:.6f}')
print(f'recomputed dv_kms = {dv_recomputed:.6f}')
"""),

        md("""
## 6. Visualising one DLA

The per-absorber records hand you a `(skewer_idx, pix_start, pix_end)`
triple.  To see what that absorber actually looks like in flux space,
load just that one row of `tau`, convert to `F = exp(-tau)`, and plot.
"""),

        code("""
# Pick the highest-NHI DLA in the catalog so the damping wings are
# obvious.  ac.by_class('DLA') returns a list of Absorber objects.
dla = max(ac.by_class('DLA'), key=lambda a: a.NHI)
print(f'DLA chosen: skewer={dla.skewer_idx}  log NHI={dla.log_NHI:.3f}  '
      f'pix=({dla.pix_start}, {dla.pix_end})')

# Stream just this one row of tau.
with h5py.File(RAW_FILE, 'r') as f:
    tau_row = f['tau/H/1/1215'][dla.skewer_idx, :].astype(np.float64)
F_row = np.exp(-tau_row)

dv = float(meta['dv_kms'])
v  = np.arange(len(tau_row)) * dv  # km/s along the sightline

# Show a window that scales with the absorber's width so the saturated
# core does not fill the visible region.  At minimum ±2500 km/s, or twice
# the core width — whichever is larger.  This DLA's core is ~2760 km/s
# wide, so the window ends up ~±5500 km/s and the damping wings + a
# stretch of continuum on each side are clearly visible.
absorber_width_kms = (dla.pix_end - dla.pix_start + 1) * dv
half_window = max(2.0 * absorber_width_kms, 2500.0)
v_centre = 0.5 * (dla.pix_start + dla.pix_end) * dv
sel = (v > v_centre - half_window) & (v < v_centre + half_window)

fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax_t.plot(v[sel], tau_row[sel], color='C3', lw=1.0)
ax_t.set(ylabel='τ', yscale='log', ylim=(1e-2, max(2.0, tau_row[sel].max())))
ax_t.axvspan(dla.pix_start * dv, dla.pix_end * dv, color='grey', alpha=0.2, label='catalog pix range')
ax_t.legend(loc='upper right', fontsize=9)
ax_t.set_title(f'DLA at log NHI = {dla.log_NHI:.2f}, b = {dla.b_kms:.1f} km/s')

ax_f.plot(v[sel], F_row[sel], color='C0', lw=1.0)
ax_f.set(xlabel='v along sightline (km/s)', ylabel='F = exp(−τ)', ylim=(-0.05, 1.05))
ax_f.axvspan(dla.pix_start * dv, dla.pix_end * dv, color='grey', alpha=0.2)

fig.tight_layout()
plt.show()
"""),

        md("""
**What you're looking at.**  The catalog's `(pix_start, pix_end)` window
(grey band) is the *core* of the absorber — the contiguous run where
τ exceeds the catalog detection threshold (`tau_threshold = 100`).
For a DLA, the *damping wings* extend much further out — you can see
the flux dipping well outside the grey band.  The Voigt fitter widens
its fit window by a few hundred km/s on each side of the core
specifically to capture those wings (where most of the NHI information
lives).  This is why `pix_start..pix_end` is short but `NHI` can still
be 10²⁰·⁵ — the catalog reports the column-density inferred from the
*full* Voigt profile, not the count of pixels above τ_threshold.

## Where to next

* **Notebook 02** — turn the per-absorber catalog into the column-density
  distribution function (CDDF) and dN/dX, and compare to literature.
* **Notebook 03** — turn the raw spectra + catalog into the per-class
  P1D templates that feed the emulator.
"""),
    ]
    return write_notebook("01_reading_catalogs_and_spectra.ipynb", cells)


# ============================================================================
# Notebook 02 — CDDF and dN/dX
# ============================================================================

def build_02():
    cells = [
        md("""
# 02 — CDDF and dN/dX from the catalog

The **column density distribution function** (CDDF) and the **incidence
rate** dN/dX are the two principal *number-statistics* of HCDs.  Both
are derived from the per-absorber catalog you saw in notebook 01.

In this notebook you will:

1. Load the cached `cddf_corrected.npz` for one (sim, snap) and look at
   what it contains.
2. Recompute it from `catalog.npz` using
   `hcd_analysis.cddf.measure_cddf_from_dataframe`, and verify the
   recomputation matches the cache to within rounding.
3. Compute the per-class dN/dX (DLA, subDLA, LLS) and confirm against
   `hcd_summary_lf.h5`.
4. Reproduce a comparison plot against the Ho+21 CDDF measurement (a
   common sanity check; this is the figure-1 of much of the analysis
   work in this repo).
"""),

        code("""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

HCD_OUT_ROOT = Path('/scratch/cavestru_root/cavestru0/mfho/hcd_outputs')
SIM = 'ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333'
SNAP = 22
SNAP_DIR = HCD_OUT_ROOT / SIM / f'snap_{SNAP:03d}'

with open(SNAP_DIR / 'meta.json') as f:
    meta = json.load(f)
print(f'sim={SIM[:50]}…  snap={SNAP}  z={meta["z"]}')
"""),

        md("""
## 1. The cached CDDF

The pipeline writes two CDDF files:

* `cddf.npz` — the **original** (buggy) CDDF.  Do not use.
* `cddf_corrected.npz` — the corrected version after the absorption-path
  fix described in `docs/bugs_found.md` §7.

The bug was a missing factor of `(1+z)·h` in the absorption path
denominator.  The corrected file divides by an extra `(1+z)·h` to fix it
in-place (`patch_factor` is stored as an attribute).  Always use
`_corrected.npz` going forward; only touch `cddf.npz` if you are
auditing the bugfix itself.
"""),

        code("""
cddf = np.load(SNAP_DIR / 'cddf_corrected.npz', allow_pickle=True)
print('Fields:')
for k in cddf.files:
    a = cddf[k]
    print(f'  {k:18s} shape={getattr(a, "shape", "()")} dtype={a.dtype}')
print()
print('z              :', float(cddf['z']))
print('n_sightlines   :', int(cddf['n_sightlines']))
print('dX_per_sightline:', float(cddf['dX_per_sightline']))
print('total_path     :', float(cddf['total_path']))
print('dx_bug_patched :', bool(cddf['dx_bug_patched']))
print('patch_factor   :', float(cddf['patch_factor']))
"""),

        md("""
**Definition reminder.**  The CDDF is

```
f(N_HI) = d²n / (dN_HI dX)
```

where `n` is the count of absorbers and `dX` the absorption distance per
sightline.  In code this is:

```python
counts, edges = np.histogram(log_nhi, bins=log_nhi_edges)
dN     = 10**edges[1:] - 10**edges[:-1]    # cm⁻²
f_nhi  = counts / (dN * total_path)
```

Higher-NHI bins are exponentially harder to fill, so a CDDF plot is
always log-log.
"""),

        code("""
log_nhi_centres = cddf['log_nhi_centres']
f_nhi           = cddf['f_nhi']
n_abs           = cddf['n_absorbers']

mask = f_nhi > 0  # drop empty bins for the log plot

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(10**log_nhi_centres[mask], f_nhi[mask], 'o-', label=f'cached (z={meta["z"]})')
ax.set(xlabel=r'$N_{HI}$ (cm$^{-2}$)', ylabel=r'$f(N_{HI})$ (cm$^2$)',
       title=f'CDDF, snap {SNAP} (z={meta["z"]})')
ax.axvspan(10**17.2, 10**19.0, alpha=0.05, color='C0', label='LLS')
ax.axvspan(10**19.0, 10**20.3, alpha=0.05, color='C1', label='subDLA')
ax.axvspan(10**20.3, 10**22.5, alpha=0.05, color='C2', label='DLA')
ax.legend()
fig.tight_layout()
plt.show()
"""),

        md("""
## 2. Recompute it from the catalog

The cached file is just the result of running `measure_cddf` on the
catalog.  Recomputing should reproduce it bitwise (modulo numpy
histogram tie-breaking on bin edges, which never matters here):
"""),

        code("""
import pandas as pd
from hcd_analysis.cddf import measure_cddf_from_dataframe

# Build a minimal DataFrame the helper expects
arr_cat = np.load(SNAP_DIR / 'catalog.npz', allow_pickle=True)
df = pd.DataFrame({
    'log_nhi':       np.log10(arr_cat['NHI']),
    'absorber_class': np.where(np.log10(arr_cat['NHI']) >= 20.3, 'DLA',
                       np.where(np.log10(arr_cat['NHI']) >= 19.0, 'subDLA',
                       np.where(np.log10(arr_cat['NHI']) >= 17.2, 'LLS', 'forest'))),
})

# Need cosmology and box from meta.json
ics = meta['sim_ics']
hubble = float(ics['ics_hubble'])
omegam = float(ics['omega0'])
omegal = 1.0 - omegam
box_kpc_h = float(meta['box_kpc_h'])
n_sightlines = int(meta['n_skewers'])

cddf_recompute = measure_cddf_from_dataframe(
    df, z=meta['z'], box_kpc_h=box_kpc_h,
    hubble=hubble, omegam=omegam, omegal=omegal,
    n_sightlines=n_sightlines,
    log_nhi_bins=cddf['log_nhi_edges'],
)

# Compare
diff = cddf_recompute['f_nhi'] - cddf['f_nhi']
ok   = np.allclose(cddf_recompute['f_nhi'], cddf['f_nhi'], rtol=1e-6, atol=0)
print(f'identical to cached: {ok}    max |diff| = {np.max(np.abs(diff)):.3e}')
"""),

        md("""
**If your recomputation does not match,** the most likely cause is that
the cached `cddf_corrected.npz` was patched in-place from a prior version
with slightly different bin edges.  Always pass the cached `log_nhi_edges`
to `measure_cddf_from_dataframe` (as we did above) so the two grids are
aligned.

## 3. dN/dX per class

dN/dX is the integral of `f(N_HI) · dN` over the NHI range you care
about.  In histogram form it's just the per-bin count divided by
`total_path`:

```
dN/dX | class = sum_{bins in class} counts_bin / total_path
```

which gives the absorbers-per-unit-absorption-path expectation.  Compute
it for each class and compare against the project-wide aggregate file
`hcd_summary_lf.h5`:
"""),

        code("""
import h5py

centres = cddf['log_nhi_centres']
counts  = cddf['n_absorbers']
total_path = float(cddf['total_path'])

dndx_lls    = counts[(centres >= 17.2) & (centres < 19.0)].sum() / total_path
dndx_subdla = counts[(centres >= 19.0) & (centres < 20.3)].sum() / total_path
dndx_dla    = counts[(centres >= 20.3)].sum() / total_path

print('Computed from cddf_corrected.npz:')
print(f'  dN/dX (LLS)    = {dndx_lls:.4f}')
print(f'  dN/dX (subDLA) = {dndx_subdla:.4f}')
print(f'  dN/dX (DLA)    = {dndx_dla:.4f}')

# Cross-check against the project-wide summary
SUMMARY = Path('/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/_hcd_analysis_data/hcd_summary_lf.h5')
if SUMMARY.exists():
    with h5py.File(SUMMARY, 'r') as f:
        sims  = f['sim'][:]
        snaps = f['snap'][:]
        # Find our (sim, snap) row
        target = SIM.encode()
        row = np.where((sims == target) & (snaps == SNAP))[0]
        if row.size:
            i = int(row[0])
            print()
            print('From hcd_summary_lf.h5 row', i, ':')
            for cls in ('LLS', 'subDLA', 'DLA'):
                print(f'  dN/dX ({cls:6s}) = {f[f"dndx/{cls}"][i]:.4f}')
        else:
            print('(this (sim, snap) is not in hcd_summary_lf.h5)')
else:
    print('(summary file not found; skipped cross-check)')
"""),

        md("""
The two sets should match closely; small discrepancies (~1e-4) are
expected because the summary file uses an integer truncation when
counting per-class absorbers, while we re-derived it from the binned
CDDF.

## 4. Compare against Ho+21

The Ho+21 CDDF tables live at `/home/mfho/DLA_data/ho21/`.  At z = 2.2
the closest table is `cddf_z225.txt`.  We overplot to check the sim
follows the observed CDDF roughly (it should, within a few × 10x at the
DLA range; LLS / subDLA tend to be over-predicted by current sims).

This is exactly the figure that `scripts/plot_cddf_vs_ho21.py` produces
for all four redshift bins; here we do just the one corresponding to our
chosen snap.
"""),

        code("""
HO21 = Path('/home/mfho/DLA_data/ho21/cddf_z225.txt')
fig, ax = plt.subplots(figsize=(7, 5))

# Sim curve from cache
ax.loglog(10**log_nhi_centres[mask], f_nhi[mask], 'o-', label=f'sim (z={meta["z"]})')

# Ho+21
if HO21.exists():
    # File is row-major: row 0 = log NHI centres, row 1 = f central,
    # row 2 = lo68, row 3 = hi68.  Drop bins where central value is 0.
    ho21 = np.loadtxt(HO21)
    logN, fcen, flo, fhi = ho21[0], ho21[1], ho21[2], ho21[3]
    ok = np.isfinite(fcen) & (fcen > 0)
    logN, fcen, flo, fhi = logN[ok], fcen[ok], flo[ok], fhi[ok]
    yerr_lo = np.clip(fcen - flo, 0, fcen * 0.999)
    yerr_hi = np.clip(fhi - fcen, 0, None)
    ax.errorbar(10**logN, fcen, yerr=[yerr_lo, yerr_hi],
                fmt='ks', mfc='white', label='Ho+21 (z≈2.25)')
else:
    print('Ho+21 reference file not found at', HO21, '— skipping overlay.')

ax.set(xlabel=r'$N_{HI}$ (cm$^{-2}$)', ylabel=r'$f(N_{HI})$ (cm$^2$)',
       title=f'CDDF: this sim vs Ho+21 at z≈2.2')
ax.legend()
fig.tight_layout()
plt.show()
"""),

        md("""
## Suggested student exercises

1. **Different (sim, snap).**  Repeat the recomputation for a different
   simulation and a different snapshot.  Pick something at z = 3 (snap
   017) or z = 4 (snap 010) and overplot Ho+21 at the matching redshift
   (`cddf_z34.txt`, `cddf_z45.txt`).
2. **Stack across sims.**  Load `cddf_corrected.npz` for all 60 LF sims
   at one snap and compute the median CDDF and its 16/84-percentile
   spread.  This is the within-LHS scatter at fixed redshift; visualise it.
3. **dN/dX(z).**  For one sim, plot dN/dX (DLA) as a function of redshift
   (snaps 4 → 23).  Compare against PRIYA papers (Bird et al. 2017).

These exercises are good warm-ups for the emulator work, where the
inputs *are* CDDFs and dN/dX values across the whole (sim, snap)
ensemble.
"""),
    ]
    return write_notebook("02_recomputing_cddf_and_dndx.ipynb", cells)


# ============================================================================
# Notebook 03 — per-class P1D
# ============================================================================

def build_03():
    cells = [
        md("""
# 03 — Per-class P1D from raw spectra + catalog

The per-class P1D (`p1d_per_class.h5`) is the principal *clustering*
statistic this dataset provides.  It contains four 1-dimensional power
spectra:

| Key | Meaning |
|---|---|
| `P_clean`        | P1D averaged over sightlines with **no** HCD (clean forest) |
| `P_LLS_only`     | P1D averaged over sightlines whose highest-class absorber is an LLS |
| `P_subDLA_only`  | P1D averaged over sightlines whose highest-class absorber is a subDLA |
| `P_DLA_only`     | P1D averaged over sightlines whose highest-class absorber is a DLA |

These are the **Rogers et al. 2018** per-class templates.  Each P1D uses
its own subset's mean flux for δF normalisation — you'll see why in §3.

By the end of this notebook you will:

1. Load and inspect `p1d_per_class.h5`.
2. Plot the four templates and the canonical Rogers ratio
   `P_DLA_only / P_clean`.
3. Recompute `P_DLA_only` from raw spectra + catalog on a small
   subsample using `hcd_analysis.p1d.compute_p1d_per_class` and confirm
   it matches the cached version.

This is the longest of the four notebooks because it ends at the
boundary of the emulator work.
"""),

        code("""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py

DATA_ROOT    = Path('/nfs/turbo/umor-yueyingn/mfho/emu_full')
HCD_OUT_ROOT = Path('/scratch/cavestru_root/cavestru0/mfho/hcd_outputs')
SIM  = 'ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333'
SNAP = 22
SNAP_DIR = HCD_OUT_ROOT / SIM / f'snap_{SNAP:03d}'
RAW_FILE = DATA_ROOT / SIM / 'output' / f'SPECTRA_{SNAP:03d}' / 'lya_forest_spectra_grid_480.hdf5'

with open(SNAP_DIR / 'meta.json') as f:
    meta = json.load(f)
print(f'sim={SIM[:50]}…  snap={SNAP}  z={meta["z"]}')
"""),

        md("""
## 1. The cached file

`p1d_per_class.h5` is a small HDF5 with one `k` array, four power
arrays, four mean-flux scalars, four sightline counts, and ~10 self-
documenting attributes at the file root.  Run `f.visititems(print)` if
you ever forget the schema.
"""),

        code("""
with h5py.File(SNAP_DIR / 'p1d_per_class.h5', 'r') as f:
    print('--- root attributes ---')
    for k, v in f.attrs.items():
        print(f'  {k}: {v}')
    print()
    print('--- datasets ---')
    f.visititems(lambda n, o: isinstance(o, h5py.Dataset)
                 and print(f'  {n}: shape={o.shape} dtype={o.dtype}'))
"""),

        code("""
with h5py.File(SNAP_DIR / 'p1d_per_class.h5', 'r') as f:
    k       = f['k'][:]
    p_clean = f['P_clean'][:]
    p_lls   = f['P_LLS_only'][:]
    p_sub   = f['P_subDLA_only'][:]
    p_dla   = f['P_DLA_only'][:]
    n_clean = int(f['n_sightlines_clean'][()])
    n_lls   = int(f['n_sightlines_LLS'][()])
    n_sub   = int(f['n_sightlines_subDLA'][()])
    n_dla   = int(f['n_sightlines_DLA'][()])
    n_total = int(f['n_total'][()])
    mF_clean = float(f['mean_F_clean'][()])
    mF_dla   = float(f['mean_F_DLA'][()])

print(f'k grid: len={len(k)}  k_min={k[1]:.4g} (k[0]=0 is DC)  k_max={k[-1]:.4g}  s/km')
print(f'sightlines: total={n_total:,}  clean={n_clean:,}  LLS={n_lls:,}  subDLA={n_sub:,}  DLA={n_dla:,}')
print(f'<F>_clean = {mF_clean:.4f}    <F>_DLA = {mF_dla:.4f}')
"""),

        md("""
## 2. Plot the four templates

The four P1Ds are very close to each other in shape but differ in
amplitude — DLA-bearing sightlines have lower mean flux than clean ones,
which (after δF normalisation by the *subset's own* mean flux) shifts
the P1D up.  This is the Rogers convention:

```
δF_subset(v) = F(v) / <F>_subset − 1
P_subset(k)  = ⟨ |FFT[δF_subset]|² ⟩  / L
```

Importantly, `<F>_subset` is **not** the global mean flux.  Using the
per-subset mean flux is what makes the ratio `P_<class>_only / P_clean`
a clean *template* that depends only on HCD properties, not on the
forest amplitude.
"""),

        code("""
fig, ax = plt.subplots(figsize=(7, 5))

for arr, lbl in [(p_clean, f'clean (n={n_clean:,})'),
                 (p_lls,   f'LLS (n={n_lls:,})'),
                 (p_sub,   f'subDLA (n={n_sub:,})'),
                 (p_dla,   f'DLA (n={n_dla:,})')]:
    sel = (k > 0) & (arr > 0)
    ax.loglog(k[sel], arr[sel], '-', label=lbl)

ax.set(xlabel='k (s/km, cyclic)', ylabel='P1D (km/s)',
       title=f'Per-class P1D, snap {SNAP} (z={meta["z"]})')
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
"""),

        code("""
# The Rogers-style template ratio: P_<class> / P_clean
fig, ax = plt.subplots(figsize=(7, 4))
sel = (k > 0) & (p_clean > 0)

for arr, lbl, c in [(p_lls, 'LLS', 'C1'),
                    (p_sub, 'subDLA', 'C2'),
                    (p_dla, 'DLA', 'C3')]:
    ax.semilogx(k[sel], arr[sel] / p_clean[sel], '-', label=lbl, color=c)

ax.axhline(1.0, color='k', linestyle=':', linewidth=0.8)
ax.set(xlabel='k (s/km, cyclic)', ylabel='P_<class> / P_clean',
       title='Rogers per-class HCD templates')
ax.legend()
fig.tight_layout()
plt.show()
"""),

        md("""
**What you should see.**  The DLA template is well above 1 at all k —
adding even one DLA to a sightline boosts its δF variance enormously.
The subDLA template is a milder boost; the LLS template is closer to
unity (small NHI → small effect).  The downturn at low k is the
finite-box / DC-removal regime; the upturn at high k is where the FFT
starts seeing the absorber's velocity profile directly.

These three ratio curves are exactly what the upstream Lyα emulator
fits with the Rogers α-template, and what our two-headed emulator
will need to learn as a function of the 9-D parameter vector.
"""),

        md("""
## 3. Recomputing P_DLA_only from scratch

Now the round-trip exercise.  We will:

1. Load the catalog (so we know which sightlines are DLA-bearing).
2. Stream the raw `tau` array, classify each sightline, and accumulate a
   P1D per class.
3. Use `hcd_analysis.p1d.compute_p1d_per_class` to do the work.

A full pass over 691 200 sightlines × 1141 pixels takes a couple of
minutes per snap on a Great Lakes login node.  For the tutorial we cap
at `n_skewers = 30000` (≈4 % of the sim) and verify shape + low-k
agreement; the high-k tail will be noisier on the subsample but the
template shape should be correct.
"""),

        code("""
from hcd_analysis.catalog import AbsorberCatalog
from hcd_analysis.p1d     import compute_p1d_per_class

ac = AbsorberCatalog.load_npz(SNAP_DIR / 'catalog.npz')

# Use the same nbins / dv_kms as the cached file (these are sim-snap-specific).
nbins  = int(meta['nbins'])
dv_kms = float(meta['dv_kms'])

result = compute_p1d_per_class(
    RAW_FILE,
    nbins=nbins,
    dv_kms=dv_kms,
    catalog=ac,
    n_skewers=30_000,    # subsample for the tutorial; full = 691200
)

print('Returned keys:', sorted(result.keys()))
print('k shape       :', result['k'].shape)
print('P_DLA_only    :', result['P_DLA_only'].shape)
print('mean_F_DLA    :', result['mean_F_DLA'])
print('n_sightlines_DLA (subsample):', result['n_sightlines_DLA'])
"""),

        md("""
**Caveats of running on a subsample.**  Because the subsample mostly
contains low-NHI sightlines (DLAs are rare — about 1 in 70 sightlines),
`mean_F_DLA` and the DLA P1D have *much* larger sample-noise than the
cached version.  The clean / LLS / subDLA P1Ds will be in much better
agreement.  Plot the cached and recomputed `P_clean` together so you
can see the agreement at low/intermediate k:
"""),

        code("""
k_re = result['k']
p_clean_re = result['P_clean']
p_dla_re   = result['P_DLA_only']

# Same k-grid because nbins and dv_kms match.  Sanity-check that.
assert k_re.shape == k.shape, 'k-grid mismatch — nbins must differ'
assert np.allclose(k_re, k, rtol=1e-12, atol=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax_, cached, sub, lbl in zip(axes,
                                 (p_clean, p_dla),
                                 (p_clean_re, p_dla_re),
                                 ('P_clean', 'P_DLA_only')):
    sel = (k > 0) & (cached > 0)
    ax_.loglog(k[sel], cached[sel], 'k-', label='cached (full 691k sightlines)')
    sel2 = (k > 0) & (sub > 0)
    ax_.loglog(k[sel2], sub[sel2], 'C1--', label='recomputed (30k subsample)')
    ax_.set(xlabel='k (s/km)', ylabel=lbl, title=lbl)
    ax_.legend(fontsize=9)

fig.tight_layout()
plt.show()
"""),

        md("""
**What you should see.**  `P_clean` agrees nearly perfectly across all k
— ~26 000 of the 30 000 subsampled sightlines are clean.  `P_DLA_only`
agrees on the broad shape but the recomputed curve is noisier and the
amplitude can be off by 10–20% at the high-k tail because only a few
hundred DLA sightlines are in the subsample.  This is expected; if you
have the patience for ~3 minutes of CPU you can rerun with
`n_skewers=None` to reproduce the cached values to floating-point
precision.

## 4. The mean-flux dimension (preview of the emulator work)

Each cached `p1d_per_class.h5` is at a single mean flux — the sim's
*natural* `<F>` at that snapshot's redshift.  Real Lyα analyses observe
the forest at many different `τ_eff` values (driven by UVB and IGM
temperature uncertainties), so a P1D emulator needs a **mean-flux
dimension** in addition to the 9 cosmological parameters.

The standard way to add this dimension is to rescale `tau → α · tau`
post-hoc and recompute `F = exp(−α · tau)`, sweeping α to cover the
desired `τ_eff` range.  We will add this rescaling step in phase 1 of
the emulator work — it requires touching the raw spectra again, not
the cached per-class P1D.

For now, just be aware that **`P_DLA_only` on disk is at one specific
`<F>`** and that the emulator work will need to broaden it into a
2-argument function `P_DLA_only(params, τ_eff)`.
"""),

        md("""
## Suggested student exercises

1. **Full-sim reproduction.**  Drop `n_skewers=30000` and rerun on the
   full file.  It will take a couple of minutes; the recomputed
   `P_DLA_only` should match the cached values to floating-point
   precision.
2. **Different NHI threshold.**  In `compute_p1d_per_class`, the
   highest-class label uses the standard 17.2 / 19.0 / 20.3 boundaries.
   Modify the function (or wrap it) to use 17.5 / 19.5 / 20.5 and see
   how the four templates shift.  This is exactly the kind of ablation
   we run when calibrating the emulator threshold choices.
3. **Parseval check.**  Pick one clean sightline, compute its δF and
   FFT, and verify that `(1/L) · ∫ |FFT|² dk = ⟨δF²⟩` to within
   floating-point precision.  This catches normalisation bugs.

## Where to next

You're now equipped to read every observable this dataset produces.  The
next milestone is **phase 1 of the HCD emulator work**: building a
single in-repo HDF5 cache that stacks the per-(sim, snap) observables
across all 1076 outputs onto a shared k- and NHI-grid.  See
`docs/SESSION_HANDOVER_2026_04_28.md` and the next planning thread for
details.
"""),
    ]
    return write_notebook("03_recomputing_per_class_p1d.ipynb", cells)


def build_04():
    cells = [
        md("""
# 04 — Per-spectrum investigation: visual signatures, the finder algorithm, and masking

So far you've worked with **catalogs as tables** — one row per absorber.
This notebook zooms back in to **one skewer at a time** and answers
three pedagogical questions that the table view glosses over:

1. **What does an LLS, a subDLA, and a DLA actually look like in
   `tau(v)` and `F = exp(-tau)`?**  We pick one clean example of each
   class and plot it.
2. **How are absorbers found?**  We walk through
   `hcd_analysis.catalog.find_systems_in_skewer` step-by-step on a real
   multi-absorber skewer, and see how the catalog gets built from raw
   `tau`.  Then we cover the two NHI-measurement modes (the fast COG
   estimator and the full Voigt fit).
3. **How are they masked?**  We compare three masking strategies on the
   same DLA: the catalog `pix_start..pix_end` (core only), the
   τ-space wing-aware mask (`hcd_analysis.masking`), and the PRIYA
   tau-based mask used in the production P1D pipeline.

This is the most code-heavy of the five notebooks because we're
literally retracing what `hcd_analysis` does to build the catalog.  Read
it next to `hcd_analysis/catalog.py` and `hcd_analysis/masking.py`.
"""),

        code("""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py

DATA_ROOT    = Path('/nfs/turbo/umor-yueyingn/mfho/emu_full')
HCD_OUT_ROOT = Path('/scratch/cavestru_root/cavestru0/mfho/hcd_outputs')
SIM  = 'ns0.81Ap1.6e-09herei3.59heref2.79alphaq1.71hub0.668omegamh20.145hireionz7.92bhfeedback0.0333'
SNAP = 22

SNAP_DIR = HCD_OUT_ROOT / SIM / f'snap_{SNAP:03d}'
RAW_FILE = DATA_ROOT / SIM / 'output' / f'SPECTRA_{SNAP:03d}' / 'lya_forest_spectra_grid_480.hdf5'

with open(SNAP_DIR / 'meta.json') as f:
    meta = json.load(f)
cat = np.load(SNAP_DIR / 'catalog.npz', allow_pickle=True)

dv      = float(meta['dv_kms'])
nbins   = int(meta['nbins'])
log_nhi = np.log10(cat['NHI'])
print(f'sim={SIM[:50]}…  snap={SNAP}  z={meta["z"]}  dv={dv:.3f} km/s  nbins={nbins}')
print(f'absorbers in catalog: {len(log_nhi):,}')
"""),

        md("""
## 1. Visual signatures of LLS, subDLA, DLA

We pick one **clean** example of each class — meaning the only absorber
on its sightline — so the visual signature isn't confused by neighbours.
The skewer indices below were found by enumerating the catalog;
recomputing them on the fly is left as an exercise.
"""),

        code("""
EXAMPLES = [
    # (class label, skewer_idx, pix_start, pix_end, log_NHI from catalog)
    ('LLS',    27,    35,   49, 17.42),
    ('subDLA', 184,  474,  510, 19.72),
    ('DLA',    210,  344,  409, 21.27),
]

# Stream just these three rows of tau (one h5py open, three slices).
with h5py.File(RAW_FILE, 'r') as f:
    tau_examples = {
        cls: f['tau/H/1/1215'][sk, :].astype(np.float64)
        for cls, sk, _, _, _ in EXAMPLES
    }

print('loaded tau for skewers:', list(tau_examples.keys()))
"""),

        code("""
fig, axes = plt.subplots(2, 3, figsize=(13, 6.0), sharex='col')
for col, (cls, sk, ps, pe, ln) in enumerate(EXAMPLES):
    tau_row = tau_examples[cls]
    F_row   = np.exp(-tau_row)
    v       = np.arange(len(tau_row)) * dv
    v_centre = 0.5 * (ps + pe) * dv
    # Window scales with absorber width so the DLA core does not fill the
    # whole panel.  At least ±1000 km/s (so the LLS panel still has
    # context); for the DLA panel this gives ~±2000 km/s.
    absorber_width_kms = (pe - ps + 1) * dv
    half = max(2.0 * absorber_width_kms, 1000.0)
    sel = (v > v_centre - half) & (v < v_centre + half)

    ax_t = axes[0, col]
    ax_t.plot(v[sel], tau_row[sel], color='C3', lw=0.8)
    ax_t.set_yscale('log')
    ax_t.axvspan(ps * dv, pe * dv, color='grey', alpha=0.2)
    ax_t.axhline(100, color='k', linestyle=':', linewidth=0.8, label='τ = 100 (detect)')
    ax_t.set_title(f'{cls}: skewer {sk}, log NHI = {ln:.2f}')
    if col == 0:
        ax_t.set_ylabel('τ')
        ax_t.legend(fontsize=8, loc='upper right')

    ax_f = axes[1, col]
    ax_f.plot(v[sel], F_row[sel], color='C0', lw=0.8)
    ax_f.axvspan(ps * dv, pe * dv, color='grey', alpha=0.2)
    ax_f.set_ylim(-0.05, 1.05)
    ax_f.set_xlabel('v (km/s)')
    if col == 0:
        ax_f.set_ylabel('F = exp(−τ)')

fig.suptitle(f'τ(v) (top) and F(v) (bottom): one clean example per HCD class', y=1.02)
fig.tight_layout()
plt.show()
"""),

        md("""
**Visual signatures:**

* **LLS** (left).  Narrow, optically thick core (a few pixels of high
  τ), no resolvable damping wing.  In flux space the absorption is a
  brief dip back up to nearly the continuum.  The catalog window
  (`pix_start..pix_end`, grey band) almost exactly contains the absorber.

* **subDLA** (centre).  Wider absorption, deeper core, a hint of a
  damping wing visible on each side.  The catalog window still mostly
  contains it but the wings spill past the grey band.

* **DLA** (right).  Saturated black core spanning tens of pixels in F,
  surrounded by very prominent damping wings extending hundreds of km/s
  on each side.  **This is the key feature of a DLA**: most of the NHI
  information is in the wings, not the core.  The catalog
  `pix_start..pix_end` only captures the core (where τ > τ_threshold);
  any masking strategy that uses just that range *will* leave damping-
  wing power in the spectrum.  We come back to this in §4.
"""),

        md("""
## 2. The finder algorithm, step by step

The catalog builder runs `find_systems_in_skewer` on every skewer.  It's
a 4-stage pipeline; here we walk through each stage on one real
multi-absorber skewer.

We use **skewer 1172** for this — it has three absorbers in the catalog:

| Catalog entry | pix_start | pix_end | log NHI | class |
|---|---|---|---|---|
| 1 | 374 | 410 | 20.19 | subDLA |
| 2 | 439 | 448 | 19.21 | subDLA |
| 3 | 598 | 613 | 17.45 | LLS |

Notice that entries 1 and 2 are only 28 pixels apart (≈ 280 km/s).
Watch how the merge step in stage 3 *almost* combines them, but doesn't
quite, because the gap exceeds the default `merge_dv_kms = 100 km/s`
(≈ 10 pixels at this dv).
"""),

        code("""
DEMO_SKEWER = 1172
with h5py.File(RAW_FILE, 'r') as f:
    tau_demo = f['tau/H/1/1215'][DEMO_SKEWER, :].astype(np.float64)
v_demo = np.arange(len(tau_demo)) * dv
print(f'skewer {DEMO_SKEWER}: max τ = {tau_demo.max():.2f}, '
      f'pixels above τ=100: {(tau_demo > 100).sum()}')
"""),

        md("""
### Stage 1 — threshold

A pixel is a candidate for being inside an absorber if `τ > tau_threshold`.
The default `tau_threshold = 100` is far above any forest absorption
(typical forest τ < a few) and well below the peak τ of even an LLS
(typical peak τ ~ 10⁴–10⁹ for HCDs).  This is a coarse first pass; the
merging and Voigt-fitting stages refine it.
"""),

        code("""
TAU_THRESHOLD = 100.0
above = tau_demo > TAU_THRESHOLD

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(v_demo, tau_demo, color='C3', lw=0.6)
ax.fill_between(v_demo, 1e-3, tau_demo, where=above, color='red', alpha=0.3,
                label=f'τ > {TAU_THRESHOLD}')
ax.axhline(TAU_THRESHOLD, color='k', linestyle=':', lw=0.8)
ax.set(yscale='log', ylim=(1e-2, max(2.0, tau_demo.max() * 2)),
       xlabel='v (km/s)', ylabel='τ',
       title=f'Stage 1: pixels above the detection threshold (skewer {DEMO_SKEWER})')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
plt.show()
"""),

        md("""
### Stage 2 — connected runs (with periodic-box trick)

The simulation box is periodic, so an absorber straddling pixel
`nbins-1` should connect with one starting at pixel `0`.  The trick used
in `find_systems_in_skewer` is to scan a *doubled* array
`[above | above]` for connected `True` runs — wrap-around runs then
appear as runs that cross the `nbins` boundary.

We reproduce the scan inline so you can see exactly what the catalog
builder does:
"""),

        code("""
def find_runs(boolean_array):
    '''Return list of (start, end_inclusive) for True runs.'''
    n = len(boolean_array)
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(boolean_array):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, n - 1))
    return runs

# Doubled array; keep only runs starting in [0, n)
above2 = np.concatenate([above, above])
all_runs = find_runs(above2)
runs = [(s, e) for (s, e) in all_runs if s < len(above)]

print(f'{len(runs)} connected runs found (before merging or length filter):')
for s, e in runs:
    wraps = e >= len(above)
    print(f'   pix {s:>5d} … {e:>5d}'
          f'  ({(e - s + 1):>3d} pixels)'
          f'{"   [WRAPS]" if wraps else ""}')
"""),

        md("""
### Stage 3 — merge close runs

Two distinct above-threshold runs that are separated by fewer than
`merge_gap_pixels = round(merge_dv_kms / dv)` pixels are considered
parts of the same physical system (e.g. a DLA whose damping wings dip
just below the threshold for a couple of pixels in the middle).
"""),

        code("""
MERGE_DV_KMS = 100.0   # default in catalog.build_catalog
merge_gap_pixels = max(1, int(MERGE_DV_KMS / dv))
print(f'merge_gap_pixels = {merge_gap_pixels} (≈ {merge_gap_pixels * dv:.0f} km/s)')

merged = [runs[0]]
for s, e in runs[1:]:
    gap = s - merged[-1][1] - 1
    if gap <= merge_gap_pixels:
        print(f'   merging ({merged[-1][0]}, {merged[-1][1]}) + ({s}, {e})  [gap = {gap}]')
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))

print(f'{len(runs)} → {len(merged)} runs after merging.')
for s, e in merged:
    print(f'   pix {s:>5d} … {e:>5d}  ({(e - s + 1):>3d} pixels)')
"""),

        md("""
### Stage 4 — drop short runs

Single- or double-pixel above-threshold blips are noise and discarded
(`min_pixels = 2` in production).  Whatever survives is what ends up in
the catalog.
"""),

        code("""
MIN_PIXELS = 2
final = [(s, e) for s, e in merged if (e - s + 1) >= MIN_PIXELS]

print(f'{len(merged)} → {len(final)} runs after length filter (min_pixels={MIN_PIXELS}).')

# Compare to what's actually in the catalog for this skewer.
on_demo = np.where(cat['skewer_idx'] == DEMO_SKEWER)[0]
print()
print(f'Catalog entries for skewer {DEMO_SKEWER}:')
for i in on_demo:
    print(f'   pix=({int(cat["pix_start"][i]):>4d}, {int(cat["pix_end"][i]):>4d})  '
          f'log NHI={np.log10(cat["NHI"][i]):.3f}  class='
          f'{"DLA" if log_nhi[i] >= 20.3 else "subDLA" if log_nhi[i] >= 19.0 else "LLS"}')
print()
print(f'Our reconstructed runs:')
for s, e in final:
    print(f'   pix=({s:>4d}, {e:>4d})')
"""),

        md("""
The reconstructed pixel ranges agree with the catalog (to within 1 pixel
at the edges, where the threshold crossing is ambiguous).  What the
catalog adds beyond our reconstruction is the **NHI per system**, which
needs Voigt fitting and is the subject of the next section.
"""),

        md("""
## 2b. NHI estimation: fast vs Voigt

Once a system's pixel range is known, we have to turn its `tau` profile
into a column density.  Two estimators are available:

* **Fast estimator** (`hcd_analysis.voigt_utils.nhi_from_tau_fast`):
  integrates τ over the core to get the equivalent width, then inverts
  the linear curve-of-growth (optically thin) or peak-area relation
  (thick).  Errors of 0.1–0.3 dex; ~10× faster.  This is what the
  production LF run uses.
* **Voigt fit** (`fit_nhi_from_tau`): expands the window by ±200 pixels
  to capture damping wings, then fits a single Voigt profile in
  log-space.  Slower, more accurate at high NHI where the wings carry
  the column-density information.  Used by the HiRes run.

For DLAs the Voigt fit can be 0.5 dex more accurate than fast mode
because most of the NHI sits in the wings, not the core.  We
demonstrate on the clean DLA from §1.
"""),

        code("""
from hcd_analysis.voigt_utils import nhi_from_tau_fast, fit_nhi_from_tau, tau_voigt

# DLA from §1: skewer 210, pix=(344, 409), catalog log NHI = 21.27
DLA_SK, DLA_PS, DLA_PE = 210, 344, 409

# Pull the row.
with h5py.File(RAW_FILE, 'r') as f:
    tau_dla = f['tau/H/1/1215'][DLA_SK, :].astype(np.float64)

# (a) Fast estimator on the core only
core = tau_dla[DLA_PS:DLA_PE + 1]
NHI_fast = nhi_from_tau_fast(core, dv_kms=dv)

# (b) Voigt fit on the wider window (±200 pixels around the core)
WING_PIX = 200
lo = max(0, DLA_PS - WING_PIX)
hi = min(len(tau_dla) - 1, DLA_PE + WING_PIX)
seg     = tau_dla[lo:hi + 1]
peak_i  = int(np.argmax(seg))
v_seg   = (np.arange(len(seg)) - peak_i) * dv
NHI_fit, b_fit, ok = fit_nhi_from_tau(seg, v_seg)

print(f'Catalog log NHI    = 21.27   (the production fast-mode value)')
print(f'Fast estimator     = {np.log10(NHI_fast):.3f}   (recomputed here on core only)')
print(f'Voigt fit          = {np.log10(NHI_fit):.3f}    (b = {b_fit:.1f} km/s, success={ok})')
"""),

        code("""
# Plot the data and the Voigt-fit model on the same window
v_full = np.arange(len(tau_dla)) * dv
v_lo, v_hi = lo * dv, hi * dv
v_peak = (lo + peak_i) * dv

# Voigt model evaluated on the full velocity grid relative to peak
tau_model = tau_voigt(v_full - v_peak, NHI=NHI_fit, b_kms=b_fit)

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.plot(v_full, tau_dla, color='C3', lw=0.7, label='data (τ from sim)')
ax.plot(v_full, tau_model, color='k', lw=1.0, linestyle='--',
        label=f'Voigt fit  log NHI={np.log10(NHI_fit):.2f}, b={b_fit:.0f} km/s')
ax.axvspan(DLA_PS * dv, DLA_PE * dv, color='grey', alpha=0.2, label='catalog pix range (core only)')
ax.axvspan(v_lo, v_hi, color='C0', alpha=0.07, label=f'Voigt fit window (±{WING_PIX} pix)')
ax.set(yscale='log', ylim=(1e-2, max(2.0, tau_dla.max() * 2)),
       xlim=(v_lo - 200, v_hi + 200),
       xlabel='v (km/s)', ylabel='τ',
       title=f'DLA on skewer {DLA_SK}: data + Voigt fit (note the damping-wing match)')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
plt.show()
"""),

        md("""
**What you should see.**  The Voigt model tracks both the saturated
core (where any model would fit by saturation alone) AND the damping
wings on each side.  The wings extend several hundred km/s past the
catalog's grey band — this is why fitting *only* the core
underestimates NHI for DLAs, and why a serious P1D analysis must mask
the wings as well as the core.
"""),

        md("""
## 3. Three masking strategies, side by side

Now the practical question: how do you remove an HCD's contribution to
the flux power spectrum?  The package provides three strategies, each
embodying a different definition of "where the absorber is":

| Strategy | What's masked | Implementation |
|---|---|---|
| `pixrange` | Catalog `pix_start..pix_end` (the τ > 100 core only) | `hcd_analysis.masking.build_skewer_mask` |
| `tauspace` | Walk outward from each system's τ-peak until τ drops below `wing_threshold[class] + τ_eff`.  Wing thresholds: 0.25 (DLA), 0.50 (subDLA), 1.00 (LLS). | `build_skewer_mask_from_tau` |
| `priya`    | Same wing-walk algorithm but starts from the global τ-peak of the sightline; only triggers if `max(τ) > 10⁶`.  This is the **production** P1D mask. | `priya_dla_mask_row` |

We apply all three to the DLA from §2 and overlay the masks on `F(v)`.
"""),

        code("""
from hcd_analysis.catalog import AbsorberCatalog
from hcd_analysis.masking import (
    build_skewer_mask,
    build_skewer_mask_from_tau,
    priya_dla_mask_row,
    DEFAULT_WING_THRESHOLD,
)

# We need an AbsorberCatalog for the package mask builders.  Slice it down
# to just the absorbers on our DLA skewer.
ac = AbsorberCatalog.load_npz(SNAP_DIR / 'catalog.npz')
abs_on_skewer = [a for a in ac.absorbers if a.skewer_idx == DLA_SK]
print(f'absorbers on skewer {DLA_SK}: {len(abs_on_skewer)}')
for a in abs_on_skewer:
    print(f'   class={a.absorber_class:6s}  pix=({a.pix_start:>4d}, {a.pix_end:>4d})  '
          f'log NHI={a.log_NHI:.2f}')

# Compute τ_eff = -ln(<F>) on this skewer (for a real run we'd use the
# global <F> from the unmasked file; here we use this skewer's <F> as a
# stand-in to keep the demo self-contained).
F_dla = np.exp(-tau_dla)
tau_eff = -np.log(max(F_dla.mean(), 1e-30))
print(f'\\ntau_eff (this-skewer proxy): {tau_eff:.4f}')
"""),

        code("""
# Three masks
mask_pixrange = build_skewer_mask(len(tau_dla), abs_on_skewer)
mask_tauspace = build_skewer_mask_from_tau(
    tau_dla, abs_on_skewer, tau_eff,
    wing_threshold_by_class=DEFAULT_WING_THRESHOLD,
)
mask_priya = priya_dla_mask_row(tau_dla, tau_eff,
                                tau_dla_detect=1e6, tau_mask_scale=0.25)
if mask_priya is None:
    mask_priya = np.zeros_like(tau_dla, dtype=bool)

print(f'pixels masked:  pixrange={mask_pixrange.sum():>4d}   '
      f'tauspace={mask_tauspace.sum():>4d}   priya={mask_priya.sum():>4d}')
"""),

        code("""
v_full = np.arange(len(tau_dla)) * dv

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
labels = [
    ('pixrange  (catalog core only)',         mask_pixrange, 'C0'),
    ('tauspace  (wing-aware, per-class thr.)', mask_tauspace, 'C1'),
    ('priya     (production DLA mask)',        mask_priya,    'C2'),
]
for ax, (lbl, mk, c) in zip(axes, labels):
    ax.plot(v_full, F_dla, color='black', lw=0.5)
    # Show masked region as shaded.  fill_between with where=mask.
    ax.fill_between(v_full, 0, 1.05, where=mk, color=c, alpha=0.35,
                    label=f'{lbl}: {mk.sum()} pix masked')
    ax.set(ylim=(-0.05, 1.05), ylabel='F = exp(−τ)')
    ax.legend(loc='lower right', fontsize=9)

axes[0].set_title(f'DLA on skewer {DLA_SK}: three masking strategies on the same flux')
axes[-1].set_xlabel('v (km/s)')
# Zoom around the absorber for clarity
v_centre = 0.5 * (DLA_PS + DLA_PE) * dv
for ax in axes:
    ax.set_xlim(v_centre - 2500, v_centre + 2500)
fig.tight_layout()
plt.show()
"""),

        md("""
**Reading the figure.**  All three masks cover the same saturated core,
but they differ on what to do with the wings:

* `pixrange` (top) is the narrowest — it only covers the τ > 100 core
  and leaves the entire damping-wing absorption visible.  Using this
  mask in a P1D estimator leaks damping-wing power into the small-k
  tail.  The package keeps `pixrange` only for regression testing and
  diagnostic purposes — **do not use it for science**.
* `tauspace` (middle) walks outward from each system's peak until τ
  drops below `0.25 + τ_eff` (for a DLA), capturing the full damping
  wings.  This is the strategy used for class-selective masking in the
  diagnostic P1D variants.
* `priya` (bottom) does almost the same wing-walk but from the
  *global* τ-peak of the sightline, and only kicks in if the sightline
  has any pixel with τ > 10⁶ (i.e. contains a DLA).  This is the
  production mask used in `compute_p1d_priya_masked` and is what feeds
  the emulator's `no_DLA_priya` variant.
"""),

        md("""
## 3b. Three fill strategies

Once you've decided which pixels to mask, you still have to decide
*what* to put in them.  The package supports three fills, all in
`hcd_analysis.masking`:

| Strategy | What goes in masked pixels | Effect on δF |
|---|---|---|
| `zero_tau`   | τ = 0  → F = 1            | huge positive δF spike (unphysical) |
| `mean_flux`  | τ = τ_eff  → F = ⟨F⟩      | δF = 0 in masked region (PRIYA recipe) |
| `contiguous` | log-τ linear interpolation| smooth bridge across the mask |

The PRIYA P1D pipeline uses `mean_flux` because it neutralises the
masked region without injecting large-scale power.  `zero_tau` is
useful only as a sanity check.  `contiguous` is occasionally useful for
narrow LLS masks where you want continuity but it can underestimate
mean flux when the mask is wide.
"""),

        code("""
from hcd_analysis.masking import _FILL_FUNCTIONS

# Use the tauspace mask we built above and apply each fill strategy
tau_filled = {
    name: fn(tau_dla, mask_tauspace)
    for name, fn in _FILL_FUNCTIONS.items()
}
F_filled = {name: np.exp(-t) for name, t in tau_filled.items()}

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.plot(v_full, F_dla, color='black', lw=0.5, label='unmasked')
for name, F in F_filled.items():
    ax.plot(v_full, F, lw=1.0, label=f'fill = {name}')
ax.axvspan(v_full[mask_tauspace].min(),
           v_full[mask_tauspace].max(),
           color='C1', alpha=0.15, label='mask region')
ax.set(xlim=(v_centre - 2500, v_centre + 2500),
       ylim=(-0.05, 1.10),
       xlabel='v (km/s)', ylabel='F',
       title='Three fill strategies on the same tauspace mask')
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
plt.show()
"""),

        md("""
**What you should see.**

* `zero_tau` (yellow-ish) bumps F up to 1.0 inside the mask — clearly
  wrong, and exactly why this strategy is for diagnostics only.
* `mean_flux` sits at F = ⟨F⟩ ≈ 0.85 across the mask — gentle, no
  spikes.  This is what production P1D pipelines use.
* `contiguous` linearly bridges the boundary values — visually smooth,
  but it can sneak in extra δF if the wing endpoints are very different.
"""),

        md("""
## Suggested student exercises

1. **Reproduce the three-class figure for a different sim, snap, or
   redshift.**  Pick z = 3 (snap 017) and find one clean LLS / subDLA /
   DLA each.  Does the relative damping-wing prominence change with z?
2. **Re-run `find_systems_in_skewer` with different parameters.**  Try
   `tau_threshold = 50` (more sensitive, more spurious LLSs) and
   `tau_threshold = 1000` (less sensitive, miss weak LLSs).  Plot the
   per-class catalog count as a function of `tau_threshold`.
3. **Compare the fast estimator and Voigt fit on a sample of 100
   DLAs.**  Compute the residual `log(NHI_voigt) − log(NHI_fast)` and
   plot its distribution.  This quantifies the systematic bias of
   fast-mode HCD catalogs.
4. **Compute the masked P1D for one sim under each of the three masks
   above.**  Use `compute_p1d_single` with `mask_scheme="pixrange"`,
   then `mask_scheme="tauspace"`, then `compute_p1d_priya_masked`.
   Plot the three P1Ds together — the spread between them is the
   "masking systematic".

## Where to next

* If you want to keep digging into HCD identification: read
  `hcd_analysis/catalog.py` (the catalog builder we just walked
  through) and `hcd_analysis/voigt_utils.py` (the Voigt fitter).
* If you want to dig into mask-induced P1D systematics: read
  `docs/masking_strategy.md` and `scripts/run_test10.py`.
* If you're ready to move past per-spectrum work and start thinking
  about emulator inputs: notebooks 02 and 03 are the population-level
  view, and the next milestone is the phase-1 emulator-data cache.
"""),
    ]
    return write_notebook("04_per_spectrum_inspection_and_masking.ipynb", cells)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_00()
    build_01()
    build_02()
    build_03()
    build_04()


if __name__ == "__main__":
    main()
