"""
Figures and markdown report generation.

All plotting functions accept a results dict or SnapResult objects and
write files to an output directory. Matplotlib is used throughout.

Figure inventory:
  discovery_summary.png   – sim/snap coverage heatmap
  nhi_histograms.png      – log10(NHI) distributions per class
  voigt_fit_examples.png  – example Voigt fits
  cddf.png                – f(NHI, X) measured
  cddf_perturbed.png      – CDDF perturbation examples
  p1d_curves.png          – P1D(k) for all variants
  p1d_ratios.png          – ratio P1D_masked / P1D_all
  timing_summary.png      – timing breakdown

Markdown outputs:
  pipeline_overview.md
  assumptions.md
  data_layout.md
  benchmarking.md
  p1d_definition.md
  cddf_model.md
  fake_spectra_integration.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Defer matplotlib import so the module is importable in headless envs
_MPL_AVAILABLE = None


def _get_plt():
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _MPL_AVAILABLE = plt
        except ImportError:
            _MPL_AVAILABLE = False
    return _MPL_AVAILABLE


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_discovery_summary(snap_map, out_dir: Path) -> None:
    """Heatmap of which (sim, snap) pairs exist."""
    plt = _get_plt()
    if not plt:
        return

    all_snaps = sorted({e.snap for ss in snap_map for e in ss.entries})
    sim_names = [ss.sim.name[:40] for ss in snap_map]
    n_sims = len(snap_map)
    n_snaps = len(all_snaps)

    matrix = np.zeros((n_sims, n_snaps), dtype=int)
    for i, ss in enumerate(snap_map):
        avail = {e.snap for e in ss.entries}
        for j, sn in enumerate(all_snaps):
            matrix[i, j] = 1 if sn in avail else 0

    fig, ax = plt.subplots(figsize=(max(8, n_snaps * 0.5), max(6, n_sims * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_snaps))
    ax.set_xticklabels([f"{sn:03d}" for sn in all_snaps], rotation=90, fontsize=7)
    ax.set_yticks(range(n_sims))
    ax.set_yticklabels(sim_names, fontsize=5)
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Simulation")
    ax.set_title("Data availability: 1=present, 0=missing")
    plt.tight_layout()
    fig.savefig(out_dir / "discovery_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Saved discovery_summary.png")


def plot_nhi_histograms(snap_results: List, out_dir: Path) -> None:
    """Log10(NHI) histograms for all z, coloured by class."""
    plt = _get_plt()
    if not plt:
        return

    fig, axes = plt.subplots(
        len(snap_results), 1,
        figsize=(8, 3 * max(len(snap_results), 1)),
        squeeze=False,
    )

    bins = np.linspace(17, 23, 61)
    class_colors = {"LLS": "steelblue", "subDLA": "darkorange", "DLA": "red"}

    for ax, result in zip(axes[:, 0], snap_results):
        if result is None:
            continue
        cat = result.catalog
        for cls, color in class_colors.items():
            nhi_vals = np.array([a.log_NHI for a in cat.by_class(cls)])
            if len(nhi_vals):
                ax.hist(nhi_vals, bins=bins, alpha=0.7, label=cls, color=color)
        ax.set_xlabel("log10(NHI)")
        ax.set_ylabel("Count")
        ax.set_title(f"z={result.z:.2f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "nhi_histograms.png", dpi=150)
    plt.close(fig)
    logger.info("Saved nhi_histograms.png")


def plot_voigt_fit_examples(snap_result, hdf5_path, out_dir: Path, n_examples: int = 6) -> None:
    """Plot example Voigt fits: observed tau vs fitted model."""
    plt = _get_plt()
    if not plt:
        return

    from .io import read_tau_chunk
    from .voigt_utils import tau_voigt

    cat = snap_result.catalog
    dv = snap_result.dv_kms

    # Pick absorbers with successful Voigt fits
    candidates = [a for a in cat.absorbers
                  if a.fit_success and not a.fast_mode and a.absorber_class in ("LLS", "subDLA", "DLA")]
    if not candidates:
        candidates = [a for a in cat.absorbers if a.absorber_class in ("LLS", "subDLA", "DLA")]

    candidates = candidates[:n_examples]
    if not candidates:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, ab in zip(axes, candidates):
        row_tau = read_tau_chunk(hdf5_path, ab.skewer_idx, ab.skewer_idx + 1)[0]
        seg = row_tau[ab.pix_start:ab.pix_end + 1].astype(np.float64)
        n_pix = len(seg)
        v_arr = np.arange(n_pix) * dv
        v_model = np.linspace(v_arr[0], v_arr[-1], 500)
        v_model_c = v_model - v_arr[np.argmax(seg)]
        v_c = v_arr - v_arr[np.argmax(seg)]

        ax.semilogy(v_c, np.clip(seg, 1e-3, None), "ko", ms=3, label="tau_obs")
        if not ab.fast_mode and np.isfinite(ab.b_kms):
            tau_model = tau_voigt(v_model_c, ab.NHI, ab.b_kms)
            ax.semilogy(v_model_c, np.clip(tau_model, 1e-3, None), "r-", lw=1.5, label="Voigt fit")
        ax.set_xlabel("v (km/s)")
        ax.set_ylabel("tau")
        ax.set_title(f"{ab.absorber_class}: log NHI={ab.log_NHI:.2f}, b={ab.b_kms:.1f} km/s")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "voigt_fit_examples.png", dpi=150)
    plt.close(fig)
    logger.info("Saved voigt_fit_examples.png")


def plot_cddf(cddf_results: List[Dict], out_dir: Path) -> None:
    """Plot f(NHI, X) vs NHI for multiple redshifts."""
    plt = _get_plt()
    if not plt:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    import matplotlib.cm as cm
    cmap = cm.viridis
    zvals = [r.get("z", 0) for r in cddf_results]
    z_min, z_max = min(zvals), max(zvals) + 1e-6

    for r in cddf_results:
        z = r.get("z", 0)
        centres = r.get("log_nhi_centres", np.array([]))
        f_nhi = r.get("f_nhi", np.array([]))
        mask = f_nhi > 0
        if not mask.any():
            continue
        color = cmap((z - z_min) / (z_max - z_min))
        ax.semilogy(centres[mask], f_nhi[mask], "-o", ms=3, color=color, lw=1, label=f"z={z:.1f}")

    ax.axvline(17.2, ls="--", color="gray", alpha=0.5)
    ax.axvline(19.0, ls="--", color="gray", alpha=0.5)
    ax.axvline(20.3, ls="--", color="gray", alpha=0.5)
    ax.text(17.2, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1e-2, "LLS", fontsize=8, va="top")
    ax.text(19.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1e-2, "subDLA", fontsize=8, va="top")
    ax.text(20.3, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1e-2, "DLA", fontsize=8, va="top")
    ax.set_xlabel("log10(NHI) [cm^-2]")
    ax.set_ylabel("f(NHI, X) = d^2n / (dNHI dX)")
    ax.set_title("Column Density Distribution Function")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(out_dir / "cddf.png", dpi=150)
    plt.close(fig)
    logger.info("Saved cddf.png")


def plot_cddf_perturbation(cddf_result: Dict, perturbations: list, out_dir: Path) -> None:
    """Show unperturbed CDDF and several perturbed variants."""
    from .cddf import CDDFPerturbation

    plt = _get_plt()
    if not plt:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    centres = cddf_result.get("log_nhi_centres", np.array([]))
    f_nhi = cddf_result.get("f_nhi", np.array([]))
    mask = f_nhi > 0

    if not mask.any():
        return

    ax.semilogy(centres[mask], f_nhi[mask], "k-o", ms=4, lw=2, label="baseline")

    colors = ["red", "blue", "green", "orange"]
    for i, (A, alpha) in enumerate(perturbations):
        pert = CDDFPerturbation(A=A, alpha=alpha)
        f_pert = pert.perturbed_f_nhi(cddf_result)
        c = colors[i % len(colors)]
        ax.semilogy(centres[mask], f_pert[mask], "--", color=c, lw=1.5,
                    label=f"A={A}, alpha={alpha}")

    ax.set_xlabel("log10(NHI)")
    ax.set_ylabel("f(NHI, X)")
    ax.set_title("CDDF perturbation examples")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "cddf_perturbed.png", dpi=150)
    plt.close(fig)
    logger.info("Saved cddf_perturbed.png")


def plot_p1d_curves(snap_result, out_dir: Path) -> None:
    """P1D(k) for all variants at one (sim, snap)."""
    plt = _get_plt()
    if not plt:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "all": ("k-", "all"),
        "no_DLA": ("r--", "no DLA"),
        "no_subDLA": ("b--", "no subDLA"),
        "no_LLS": ("g--", "no LLS"),
        "no_HCD": ("m-.", "no HCD"),
    }

    for var, (ls, label) in styles.items():
        if var in snap_result.p1d_variants:
            k, p1d, _ = snap_result.p1d_variants[var]
            valid = np.isfinite(p1d) & (p1d > 0) & (k > 0)
            if valid.any():
                ax.loglog(k[valid], k[valid] * p1d[valid] / np.pi, ls, lw=1.5, label=label)

    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("k P1D(k) / pi")
    ax.set_title(f"P1D – z={snap_result.z:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"p1d_z{snap_result.z:.2f}.png", dpi=150)
    plt.close(fig)


def plot_p1d_ratios(snap_result, out_dir: Path) -> None:
    """Ratio plots P1D_masked / P1D_all."""
    plt = _get_plt()
    if not plt:
        return

    ratios = snap_result.p1d_ratios
    if "k" not in ratios:
        return

    k = ratios["k"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, label, color in [
        ("ratio_noDLA_all", "no DLA / all", "red"),
        ("ratio_nosubDLA_all", "no subDLA / all", "blue"),
        ("ratio_noLLS_all", "no LLS / all", "green"),
        ("ratio_noHCD_all", "no HCD / all", "purple"),
    ]:
        if name in ratios:
            r = ratios[name]
            valid = np.isfinite(r) & (k > 0)
            if valid.any():
                ax.semilogx(k[valid], r[valid], label=label, color=color, lw=1.5)

    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("P1D ratio")
    ax.set_title(f"P1D ratios – z={snap_result.z:.2f}")
    ax.set_ylim(0.8, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"p1d_ratios_z{snap_result.z:.2f}.png", dpi=150)
    plt.close(fig)


def plot_timing_summary(benchmark_result: Dict, out_dir: Path) -> None:
    """Bar chart of timing breakdown."""
    plt = _get_plt()
    if not plt:
        return

    records = benchmark_result.get("timing_per_snap", [])
    if not records:
        return

    labels = [f"z={r['z']:.1f}" for r in records]
    t_cat = [r.get("t_catalog_full_est_s", 0) for r in records]
    t_p1d = [r.get("t_p1d_full_est_s", 0) for r in records]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), 5))
    ax.bar(x, t_cat, label="catalog (est)", color="steelblue")
    ax.bar(x, t_p1d, bottom=t_cat, label="P1D (est)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Time (s, estimated full)")
    ax.set_title("Per-snap timing estimate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "timing_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Saved timing_summary.png")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)
    logger.info("Wrote %s", path)


def generate_data_layout_md(out_dir: Path) -> None:
    content = """\
# Data Layout

**Root:** `/nfs/turbo/umor-yueyingn/mfho/emu_full/`

## Simulation Folders (60 total)

Each folder encodes the parameter point as a name string:
```
ns{ns}Ap{Ap}herei{herei}heref{heref}alphaq{alphaq}hub{hub}omegamh2{omegamh2}hireionz{hireionz}bhfeedback{bhfeedback}
```

### Parameter definitions
| Symbol | Description | Range |
|--------|-------------|-------|
| ns | spectral index | 0.80 – 1.05 |
| Ap | amplitude A_s (10^-9) | 1.2e-9 – 2.6e-9 |
| herei | HeII reionisation start z | 3.5 – 4.1 |
| heref | HeII reionisation end z | 2.6 – 3.2 |
| alphaq | quasar spectral index | 1.3 – 3.0 |
| hub | Hubble h | 0.65 – 0.75 |
| omegamh2 | Omega_m h^2 | 0.140 – 0.146 |
| hireionz | HI reionisation z | 6.5 – 8.0 |
| bhfeedback | black hole feedback | 0.03 – 0.07 |

## Directory tree

```
{sim_folder}/
  output/
    Snapshots.txt        ← snap_idx  scale_factor_a
    SPECTRA_004/
      lya_forest_spectra_grid_480.hdf5   ← primary (691200 skewers)
    SPECTRA_005/
      lya_forest_spectra_grid_480.hdf5
    ...
    SPECTRA_023/
      lya_forest_spectra_grid_480.hdf5
```

## HDF5 Schema (`lya_forest_spectra_grid_480.hdf5`)

```
/
  Header/
    attrs:
      redshift   float   e.g. 5.4
      Hz         float   H(z) in km/s/Mpc
      box        float   120000.0 (kpc/h, comoving)
      hubble     float   h
      nbins      int     1556  (pixels per skewer)
      npart      array   particle counts
      omegab     float
      omegal     float
      omegam     float
  spectra/
    axis   (691200,)    int32   LOS axis: 1=x, 2=y, 3=z
    cofm   (691200, 3)  float64 skewer position (kpc/h, comoving)
  tau/
    H/
      1/
        1215   (691200, 1556)  float32   HI Lyman-alpha optical depth
  colden/          ← GROUP EXISTS BUT IS EMPTY
  tau_obs/         ← EMPTY
  temperature/     ← EMPTY
  velocity/        ← EMPTY
  density_weight_density/  ← EMPTY
  num_important/   ← EMPTY
```

**Important:** `colden` is an empty group. Column densities are not pre-computed
and must be inferred from `tau/H/1/1215`.

## Snapshots and Redshifts

Snapshots.txt maps snapshot index → scale factor a. Redshift: z = 1/a - 1.

| Snap | a       | z    | In 2≤z≤6 |
|------|---------|------|-----------|
| 004  | 0.15625 | 5.40 | yes |
| 005  | 0.16129 | 5.20 | yes |
| 006  | 0.16667 | 5.00 | yes |
| 007  | 0.17241 | 4.80 | yes |
| 008  | 0.17857 | 4.60 | yes |
| 009  | 0.18519 | 4.40 | yes |
| 010  | 0.19231 | 4.20 | yes |
| 011  | 0.20000 | 4.00 | yes |
| 012  | 0.20833 | 3.80 | yes |
| 013  | 0.21112 | 3.74 | yes |
| 014  | 0.21739 | 3.60 | yes |
| 015  | 0.22727 | 3.40 | yes |
| 016  | 0.23810 | 3.20 | yes |
| 017  | 0.25000 | 3.00 | yes |
| 018  | 0.26316 | 2.80 | yes |
| 019  | 0.27778 | 2.60 | yes |
| 020  | 0.28077 | 2.56 | yes |
| 021  | 0.29412 | 2.40 | yes |
| 022  | 0.31250 | 2.20 | yes |
| 023  | 0.33333 | 2.00 | yes |

## Pixel velocity width

dv = H(z) × (box_Mpc_physical) / nbins ≈ **10.0 km/s** at z=5.4
(this is a coincidental round number; exact value is stored in each file's Header).

## Snapshot availability

Not all simulations have all snapshots. Summary from the 60-sim dataset:

| Snap | z    | N_sims |
|------|------|--------|
| 004  | 5.40 | 60 / 60 |
| 005  | 5.20 | 60 / 60 |
| ...  |      |         |
| 022  | 2.20 | 48 / 60 |
| 023  | 2.00 | 18 / 60 |

Lower-redshift snapshots (snap 022, 023) are missing in many simulations.
The pipeline handles missing snapshots gracefully.

## Secondary file

Some simulations also contain `lya_forest_spectra.hdf5` (32000 skewers,
1747 pixels, float64). This is a random subsample and is used only when
the grid file is absent.
"""
    _write(out_dir / "data_layout.md", content)


def generate_assumptions_md(out_dir: Path) -> None:
    content = """\
# Explicit Assumptions

## Data

1. **tau is raw simulation optical depth** — not rescaled to any mean flux.
   The UV background is the simulation's native UVB; no post-processing
   mean-flux rescaling is applied before P1D or CDDF computation.

2. **colden is empty** — the `colden` HDF5 group key exists but contains no
   datasets. All column density estimates are derived from `tau/H/1/1215`.

3. **Only Lyman-alpha tau** — only `tau/H/1/1215` (HI 1216 Å) is used.
   No higher Lyman series, no metal lines.

4. **Primary file is grid_480** — 691200 sightlines on a 480×480×3 grid.
   When absent, falls back to `lya_forest_spectra.hdf5` (32000 sightlines).

5. **Snapshots.txt is authoritative** for (snap, a) mapping. If a SPECTRA
   folder exists without a matching Snapshots.txt entry, the redshift is
   read from the HDF5 Header.

## Absorber identification

6. **System = connected tau region** — an absorption system is defined as a
   contiguous set of pixels where `tau > tau_threshold` (default: 1.0).
   This corresponds approximately to log10(NHI) > 16 for a typical b=30 km/s.

7. **Merging criterion** — two adjacent blobs are merged into one system if
   the gap between them is < `merge_dv_kms` (default: 100 km/s) in velocity
   space. This prevents artificial splitting of DLAs with narrow sub-structure.

8. **NHI via Voigt fitting** — NHI is estimated by fitting a single Voigt
   component (2 parameters: NHI, b) to each system's tau profile.
   In fast/benchmark mode, a closed-form approximation is used instead.

9. **Classification thresholds** (standard literature values):
   - LLS    : 10^17.2 ≤ NHI < 10^19.0 cm^-2
   - subDLA : 10^19.0 ≤ NHI < 10^20.3 cm^-2
   - DLA    : NHI ≥ 10^20.3 cm^-2

## P1D

10. **Mean flux normalisation** — delta_F = F/<F> - 1 where <F> is computed
    globally over all skewers in the file (not per-skewer).

11. **No mean flux rescaling** — we do not rescale tau to match observed mean
    flux. If rescaling is needed, apply it externally before running this pipeline.

12. **k units** — k is in s/km (inverse velocity). Conversion to h/Mpc:
    k [h/Mpc] = k [s/km] × H(z)/h × 1000.

13. **Box boundary** — sightlines are periodic (from MP-Gadget). No windowing
    is applied to the FFT.

## CDDF

14. **Absorption path** — uses the narrow-box approximation:
    dX = (1+z) × L_comoving_Mpc × H_0/c.

15. **NHI bins** — log10(NHI) from 17 to 23 in 30 bins by default.

## General

16. **Cosmology is per-sim** — omega_m, omega_l, h are read from the HDF5 Header
    for each file. There is no global assumed cosmology.

17. **Box size is fixed** — all sims use a 120 Mpc/h comoving box.

18. **No peculiar velocity correction** — the tau array from fake_spectra
    already includes peculiar velocity contributions (it is computed in
    redshift space). No correction is applied.
"""
    _write(out_dir / "assumptions.md", content)


def generate_p1d_definition_md(out_dir: Path) -> None:
    content = """\
# P1D Definition

## Flux field

Given the optical depth tau(v) along a sightline (pixel index j, velocity v_j = j × dv):

    F(v) = exp(-tau(v))

The fractional flux fluctuation is:

    delta_F(v) = F(v) / <F> - 1

where <F> is the mean transmission averaged over all N_skewers sightlines:

    <F> = (1 / N_skewers / N_pix) × sum_{all pixels} exp(-tau_ij)

## Fourier transform convention

We use the discrete Fourier transform (DFT) along the velocity axis:

    tilde_F(k_n) = dv × sum_{j=0}^{N-1} delta_F(v_j) × exp(-2 pi i k_n v_j)

where:
    k_n = n / (N × dv)    for n = 0, 1, ..., N/2

Units: tilde_F has units of km/s.

## Power spectrum

    P1D(k_n) = (1 / L) × |tilde_F(k_n)|^2

where L = N × dv is the total velocity extent of the box (km/s).

Substituting:

    P1D(k_n) = dv / N × |DFT(delta_F)[n]|^2

Units: [P1D] = km/s.

In code: `P1D[n] = dv / N * |rfft(delta_F * dv)|^2 / dv^2 / (N * dv)`
       = `dv^2 / (N * dv) * |rfft(delta_F)[n]|^2`
       = `dv / N * |rfft(delta_F)[n]|^2`

(The rfft is normalised by numpy as a sum, not dividing by N, so we divide explicitly.)

## Averaging over sightlines

P1D is averaged over all sightlines (691200 per file):

    <P1D(k)> = (1 / N_skewers) × sum_i P1D_i(k)

## k range and units

- Pixel width: dv ≈ 10 km/s (exact value from Header)
- N_pix = 1556 pixels per sightline
- k_min = 1 / (1556 × 10) ≈ 6.4 × 10^-5 s/km
- k_max = 1 / (2 × 10) = 0.05 s/km  (Nyquist)
- k units: s/km

Default output bins: 35 bins matching the emulator kf grid from 0.00108 to 0.01951 s/km.

## P1D variants

| Variant | Description |
|---------|-------------|
| `all` | All sightlines, no masking |
| `no_DLA` | DLA pixels replaced by mean flux |
| `no_subDLA` | subDLA pixels replaced |
| `no_LLS` | LLS pixels replaced |
| `no_HCD` | All of LLS+subDLA+DLA replaced |

## Ratios

    ratio_noDLA_all    = P1D(no_DLA) / P1D(all)
    ratio_noHCD_all    = P1D(no_HCD) / P1D(all)
    ratio_DLA_contribution = 1 - ratio_noDLA_all

## Parseval check

By Parseval's theorem:

    sum_n P1D(k_n) × dk ≈ <delta_F^2>

where the sum is over positive k modes. This is verified in tests.
"""
    _write(out_dir / "p1d_definition.md", content)


def generate_cddf_model_md(out_dir: Path) -> None:
    content = """\
# CDDF Model and Perturbation

## Definition

The column density distribution function (CDDF) is:

    f(N_HI, X) = d^2 n / (dN_HI dX)

where:
  - N_HI: neutral hydrogen column density (cm^-2)
  - X: dimensionless absorption path length
  - n: number of absorbers per sightline

## Absorption path length

For a flat ΛCDM cosmology:

    dX/dz = (H_0 / H(z)) × (1+z)^2

For a simulation box at redshift z with comoving size L [Mpc]:

    dX = (H_0 / H(z)) × (1+z)^2 × dz_box

where dz_box = H(z)/c × L_phys = H(z)/c × L/(1+z), giving:

    dX = (1+z) × L × H_0/c

Per sightline: dX = (1+z) × (box_Mpc) × (H_0/c)
Total path: dX_total = N_sightlines × dX

## Measurement

From the absorber catalog, we bin absorbers by log10(NHI) and compute:

    f(N_i) = count(N in bin i) / (dN_i × dX_total)

where dN_i = 10^(log_N_max) - 10^(log_N_min) in cm^-2.

## Perturbation model

We define a continuous multiplicative perturbation:

    f'(N) = A × f(N) × (N / N_pivot)^alpha

Parameters:
  A       : amplitude multiplier       (default 1.0, i.e. no change)
  alpha   : power-law tilt             (default 0.0)
  N_pivot : pivot column density (cm^-2, default 10^20)

Special cases:
  - A=1, alpha=0: no perturbation (identity)
  - A>1, alpha=0: uniform increase of all absorbers
  - alpha>0: tilt toward high-NHI systems (more DLAs relative to LLS)
  - alpha<0: tilt toward low-NHI systems

## Propagation to P1D

To compute the P1D effect of the CDDF perturbation, we use Poisson resampling:

1. For each absorber i with column density N_i, compute weight:
       w_i = A × (N_i / N_pivot)^alpha

2. Draw n_copies_i ~ Poisson(w_i) for each absorber.

3. Build a perturbed absorber set from this resampling.

4. Compute masked P1D with the perturbed absorber set as the mask.

5. Repeat for N_realizations to estimate Monte Carlo variance.

The ratio P1D_perturbed / P1D_baseline quantifies the P1D correction
due to the CDDF perturbation.

## Interpretation

This model allows the HCD correction to the P1D to be continuously tunable:
- The amplitude A shifts the overall absorber number density.
- The tilt alpha changes the relative contributions of LLS, subDLA, and DLA
  to the P1D suppression.
- The pivot N_pivot sets where the tilt has no effect (f'(N_pivot) = A × f(N_pivot)).

For emulator purposes, the combination (A, alpha) at fixed N_pivot provides
a 2-parameter family of CDDF corrections that can be marginalised over.
"""
    _write(out_dir / "cddf_model.md", content)


def generate_fake_spectra_integration_md(out_dir: Path) -> None:
    content = """\
# fake_spectra Integration

## Package

- **Repository:** https://github.com/sbird/fake_spectra
- **Author:** Simeon Bird
- **Purpose:** Compute Lyman-alpha forest spectra from cosmological simulations.

## What this repo reuses from fake_spectra

### Directly reused: physics constants and line parameters

`hcd_analysis/voigt_utils.py` uses the same Lyman-alpha line parameters
as fake_spectra:
  - Lyman-alpha wavelength: lambda = 1215.67 Å
  - Oscillator strength: f_lu = 0.4164
  - Einstein A coefficient: Gamma = 6.265e8 s^-1

These values are embedded as constants (the same values fake_spectra uses)
to avoid a hard dependency on fake_spectra's internal module structure,
which has changed across versions.

### Reused: Voigt profile computation approach

`voigt_utils.voigt_profile_phi()` uses `scipy.special.wofz` (Faddeeva function),
which is the same mathematical approach as fake_spectra's internal Voigt
computation. The Voigt-Hjerting function H(a,u) = Re[wofz(u + i*a)] is
standard in all modern Lyman-alpha codes.

### Wrapped: tau_voigt() model function

`voigt_utils.tau_voigt(v, NHI, b, v0)` computes the theoretical optical depth
for a single Voigt component. This is the model function used in absorber
fitting. It replicates what fake_spectra computes internally when generating
synthetic spectra from particle data.

### NOT reused (why)

| fake_spectra component | Status in this repo | Reason |
|------------------------|---------------------|--------|
| `Spectra` class | Not used | Designed for particle data; we have pre-computed tau |
| `Spectra.get_tau()` | Not applicable | tau is already saved |
| `Spectra.get_col_density()` | Not applicable | Requires particle data (GADGET snapshots) |
| `find_absorbers()` | Re-implemented | Our implementation operates on tau directly |
| `rate_interpolate` | Not used | Designed for post-processing Spectra objects |

## What is newly implemented in this repo

| Component | Location | Description |
|-----------|----------|-------------|
| Tau-based system finder | `catalog.find_systems_in_skewer()` | Connected threshold regions in velocity space |
| Voigt fitting from tau | `voigt_utils.fit_nhi_from_tau()` | scipy L-BFGS-B fit in log space |
| AbsorberCatalog | `catalog.AbsorberCatalog` | Serialisable container for absorber sets |
| P1D accumulator | `p1d.P1DAccumulator` | Streaming, memory-efficient P1D computation |
| CDDF measurement | `cddf.measure_cddf()` | From absorption path and simulation box |
| CDDF perturbation | `cddf.CDDFPerturbation` | Continuous (A, alpha, N_pivot) model |
| Pipeline | `pipeline.run_one_snap()` | End-to-end orchestration with resume/restart |
| Batch scripts | `scripts/` | Great Lakes SLURM job templates |

## Installation

See `docs/` section on environment setup or README for installation instructions.
If fake_spectra is not installed, the pipeline falls back to internal Voigt
utilities (the same physics, just without fake_spectra as a dependency).
"""
    _write(out_dir / "fake_spectra_integration.md", content)


def generate_pipeline_overview_md(out_dir: Path) -> None:
    content = """\
# Pipeline Overview

## Goal

Process all `fake_spectra` outputs for the lyman-alpha forest emulator
at Great Lakes. For each (simulation, redshift) pair:

1. Identify HCD absorption systems (LLS, subDLA, DLA) from tau.
2. Compute P1D(k) for the full forest and with each class masked.
3. Measure the CDDF f(NHI, X).
4. Compute the P1D effect of a parametric CDDF perturbation.

## Data flow

```
HDF5 tau array (691200 × 1556)
         │
         ├─── catalog.py ──→ AbsorberCatalog (.npz)
         │        │               │
         │        └─ Voigt fit    └─── cddf.py ──→ f(NHI, X)
         │                               │
         └─── p1d.py ──────────────────→ P1D all variants
                  │                          │
                  └─ masking.py ─────────────┘
                       │
                       └─ cddf.py (perturbation) ──→ P1D perturbed
```

## Entry points

```bash
# One simulation, one redshift
hcd run-one --sim ns0.803... --snap 017 --config config/default.yaml

# One simulation, all redshifts
hcd run-sim --sim ns0.803... --config config/default.yaml

# All simulations, all redshifts
hcd run-all --config config/default.yaml

# Benchmark mode
hcd benchmark --n-sims 2 --n-snaps 3
```

## Outputs

Each (sim, snap) pair writes to:
```
outputs/{sim_name}/snap_{NNN}/
  done              ← sentinel (skip on resume)
  catalog.npz       ← absorber catalog
  p1d.npz           ← P1D variants
  p1d_ratios.npz    ← ratio arrays
  cddf.npz          ← CDDF measurement
  p1d_perturbed.npz ← (optional) perturbed P1D
  meta.json         ← timing + metadata
```

Figures are written to `figures/`.
Markdown docs are in `docs/`.
"""
    _write(out_dir / "pipeline_overview.md", content)


def generate_benchmarking_md(out_dir: Path, benchmark_result: Optional[Dict] = None) -> None:
    if benchmark_result is None:
        content = """\
# Benchmarking

Run `hcd benchmark` to populate this file with actual timing results.

## Extrapolation method

1. Run catalog build and P1D on 10k skewers (≈1.5% of total).
2. Scale timing by `n_total_skewers / 10000`.
3. Multiply by number of (sim, snap) pairs.
4. Divide by n_workers for parallel estimate.

## Expected scale

With 60 sims × ~18 snapshots per sim = ~1080 (sim, snap) pairs,
and ~4–8 SLURM nodes (each 36–48 cores):

- Fast mode (no Voigt fitting): estimated 2–6 hours wall time.
- Full Voigt fitting: estimated 12–48 hours depending on hardware.
"""
    else:
        lines = ["# Benchmarking Results\n"]
        lines.append(f"- Total simulations: {benchmark_result['n_total_sims']}")
        lines.append(f"- Total (sim, snap) pairs: {benchmark_result['n_total_snaps']}")
        lines.append(f"- Average time per snap (full): {benchmark_result['avg_time_per_snap_s']} s")
        lines.append(f"- Campaign serial estimate: {benchmark_result['campaign_serial_hr']} hours")
        lines.append(f"- Campaign parallel estimate ({benchmark_result['n_workers_assumed']} workers): "
                     f"{benchmark_result['campaign_parallel_hr']} hours")
        lines.append("\n## Per-snap timings (sampled)\n")
        lines.append("| sim | snap | z | n_skewers | catalog_est_s | p1d_est_s |")
        lines.append("|-----|------|---|-----------|---------------|-----------|")
        for r in benchmark_result.get("timing_per_snap", []):
            lines.append(
                f"| {r['sim'][:30]} | {r['snap']:3d} | {r['z']:.2f} | "
                f"{r['n_skewers']} | {r['t_catalog_full_est_s']:.1f} | {r['t_p1d_full_est_s']:.1f} |"
            )
        content = "\n".join(lines) + "\n"

    _write(out_dir / "benchmarking.md", content)


def generate_all_markdown(out_dir: Path, benchmark_result: Optional[Dict] = None) -> None:
    """Generate all markdown files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_data_layout_md(out_dir)
    generate_assumptions_md(out_dir)
    generate_p1d_definition_md(out_dir)
    generate_cddf_model_md(out_dir)
    generate_fake_spectra_integration_md(out_dir)
    generate_pipeline_overview_md(out_dir)
    generate_benchmarking_md(out_dir, benchmark_result)
