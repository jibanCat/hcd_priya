"""
Intermediate analysis plots from partially-completed pipeline outputs.

Generates figures from whatever catalog.npz / p1d.npz files exist in the
scratch output directory, without requiring a complete run.

Loads catalog data as raw numpy arrays (no per-absorber object construction),
so 33 × 457K absorbers loads in ~1s instead of 8+ minutes.

Usage:
    python3 scripts/plot_intermediate.py
    python3 scripts/plot_intermediate.py --output-root /scratch/.../hcd_outputs --out-dir figures/intermediate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT_DIR = Path("/home/mfho/hcd_priya/figures/intermediate")

# NHI class boundaries (log10 cm^-2)
LOG_NHI_LLS    = 17.2
LOG_NHI_SUBDLA = 19.0
LOG_NHI_DLA    = 20.3


def truth_cddf_prochaska2014(logN):
    """Prochaska+2014 CDDF spline (PchipInterpolator). Returns log10 f(N)."""
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    spline = PchipInterpolator(_logN, _logf)
    return spline(np.clip(np.asarray(logN, dtype=float), _logN[0], _logN[-1]))


_SIM_PARAM_PATTERNS = {
    "ns":        r"ns([0-9.e+-]+)",
    "Ap":        r"Ap([0-9.e+-]+)",
    "herei":     r"herei([0-9.e+-]+)",
    "heref":     r"heref([0-9.e+-]+)",
    "alphaq":    r"alphaq([0-9.e+-]+)",
    "hub":       r"hub([0-9.e+-]+)",
    "omegamh2":  r"omegamh2([0-9.e+-]+)",
    "hireionz":  r"hireionz([0-9.e+-]+)",
    "bhfeedback":r"bhfeedback([0-9.e+-]+)",
}

def _parse_sim_params(sim_name: str) -> dict | None:
    result = {}
    for key, pat in _SIM_PARAM_PATTERNS.items():
        m = re.search(pat, sim_name)
        if m:
            result[key] = float(m.group(1))
    return result or None


# ---------------------------------------------------------------------------
# Fast data loading — raw numpy arrays, no Python per-object loop
# ---------------------------------------------------------------------------

def load_cat_fast(cat_path: Path) -> dict:
    """Load catalog.npz as a dict of numpy arrays (no Absorber object construction)."""
    d = np.load(str(cat_path), allow_pickle=True)
    NHI = d["NHI"].astype(np.float64)
    log_NHI = np.log10(np.maximum(NHI, 1e1))
    # Read n_skewers and hubble from meta.json if available (correct normalization)
    meta_path = Path(str(cat_path)).parent / "meta.json"
    n_skewers = 691200  # default for LF sims (480^2 * 3)
    hubble = 0.71       # fallback
    box_kpc_h = 120000.0
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            n_skewers = int(meta.get("n_skewers", n_skewers))
            hubble = float(meta.get("hubble", hubble))
            box_kpc_h = float(meta.get("box_kpc_h", box_kpc_h))
        except Exception:
            pass
    return {
        "sim_name": str(d["sim_name"]),
        "snap": int(d["snap"]),
        "z": float(d["z"]),
        "dv_kms": float(d["dv_kms"]),
        "skewer_idx": d["skewer_idx"].astype(np.int64),
        "NHI": NHI,
        "log_NHI": log_NHI,
        "b_kms": d["b_kms"].astype(np.float64),
        "fit_success": d["fit_success"].astype(bool),
        "fast_mode": d["fast_mode"].astype(bool),
        "n_skewers": n_skewers,
        "hubble": hubble,
        "box_kpc_h": box_kpc_h,
    }


def class_mask(cat: dict, cls: str) -> np.ndarray:
    """Boolean mask for absorbers in a given HI class."""
    log = cat["log_NHI"]
    if cls == "LLS":
        return (log >= LOG_NHI_LLS) & (log < LOG_NHI_SUBDLA)
    elif cls == "subDLA":
        return (log >= LOG_NHI_SUBDLA) & (log < LOG_NHI_DLA)
    elif cls == "DLA":
        return log >= LOG_NHI_DLA
    return np.zeros(len(log), dtype=bool)


def n_sightlines(cat: dict) -> int:
    """Total number of sightlines in the sim (from meta.json, not estimated from absorbers)."""
    return cat.get("n_skewers", 691200)


def load_all_catalogs(output_root: Path) -> list:
    """
    Load all catalog.npz files as fast dicts.
    Returns list of (sim_name, snap, z, cat_dict).
    """
    records = []
    cat_paths = sorted(output_root.rglob("catalog.npz"))
    print(f"Found {len(cat_paths)} catalog.npz files — loading ...", flush=True)
    for cat_path in cat_paths:
        try:
            cat = load_cat_fast(cat_path)
            snap_dir = cat_path.parent
            sim_name = snap_dir.parent.name
            snap = int(snap_dir.name.split("_")[1])
            records.append((sim_name, snap, cat["z"], cat))
        except Exception as e:
            print(f"  skip {cat_path}: {e}")
    print(f"Loaded {len(records)} catalogs.", flush=True)
    return records


def load_all_p1d(output_root: Path) -> list:
    """Load all p1d.npz files. Returns list of (sim, snap, z, data_dict)."""
    records = []
    for p1d_path in sorted(output_root.rglob("p1d.npz")):
        try:
            data = dict(np.load(p1d_path))
            snap_dir = p1d_path.parent
            sim_name = snap_dir.parent.name
            snap = int(snap_dir.name.split("_")[1])
            meta_path = snap_dir / "meta.json"
            z = json.load(open(meta_path))["z"] if meta_path.exists() else 0.0
            records.append((sim_name, snap, z, data))
        except Exception as e:
            print(f"  skip {p1d_path}: {e}")
    print(f"Loaded {len(records)} P1D files from {output_root}", flush=True)
    return records


def load_all_p1d_excl(output_root: Path) -> list:
    """
    Load all p1d_excl.npz files (sightline-exclusion sweep).
    Returns list of (sim, snap, z, excl_dict) where excl_dict has:
      'k'             : (nbins,)
      'p1d_excl'      : (n_cuts, nbins)
      'log_nhi_cuts'  : (n_cuts,)
      'frac_remaining': (n_cuts,)
    """
    records = []
    for path in sorted(output_root.rglob("p1d_excl.npz")):
        try:
            d = dict(np.load(path))
            snap_dir = path.parent
            sim_name = snap_dir.parent.name
            snap = int(snap_dir.name.split("_")[1])
            meta_path = snap_dir / "meta.json"
            z = json.load(open(meta_path))["z"] if meta_path.exists() else 0.0
            records.append((sim_name, snap, z, d))
        except Exception as e:
            print(f"  skip {path}: {e}")
    print(f"Loaded {len(records)} p1d_excl files from {output_root}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Figure 1: Pipeline progress dashboard
# ---------------------------------------------------------------------------

def plot_progress_dashboard(output_root: Path, out_dir: Path):
    """Color-coded heatmap of pipeline completion across all sims × snaps."""
    try:
        from hcd_analysis.snapshot_map import build_snapshot_map
        snap_map = build_snapshot_map(
            "/nfs/turbo/umor-yueyingn/mfho/emu_full", z_min=2.0, z_max=6.0
        )
        all_snaps = sorted({e.snap for ss in snap_map for e in ss.entries})
        sim_names = [ss.sim.name for ss in snap_map]
    except Exception as e:
        print(f"  build_snapshot_map failed ({e}), falling back to output_root scan")
        snap_dirs = sorted(output_root.rglob("snap_*"))
        sim_set = sorted({d.parent.name for d in snap_dirs})
        snap_set = sorted({int(d.name.split("_")[1]) for d in snap_dirs})
        sim_names = sim_set
        all_snaps = snap_set
        snap_map = None

    n_sims  = len(sim_names)
    n_snaps = len(all_snaps)
    snap_idx = {s: i for i, s in enumerate(all_snaps)}
    sim_idx  = {s: i for i, s in enumerate(sim_names)}

    # Status: 0=unknown, 1=catalog, 2=p1d, 3=done
    matrix = np.zeros((n_sims, n_snaps), dtype=np.int8)

    for cat_path in output_root.rglob("catalog.npz"):
        snap_dir = cat_path.parent
        sn = int(snap_dir.name.split("_")[1])
        sim = snap_dir.parent.name
        if sim in sim_idx and sn in snap_idx:
            matrix[sim_idx[sim], snap_idx[sn]] = max(1, matrix[sim_idx[sim], snap_idx[sn]])

    for p1d_path in output_root.rglob("p1d.npz"):
        snap_dir = p1d_path.parent
        sn = int(snap_dir.name.split("_")[1])
        sim = snap_dir.parent.name
        if sim in sim_idx and sn in snap_idx:
            matrix[sim_idx[sim], snap_idx[sn]] = max(2, matrix[sim_idx[sim], snap_idx[sn]])

    for done_path in output_root.rglob("done"):
        snap_dir = done_path.parent
        sn = int(snap_dir.name.split("_")[1])
        sim = snap_dir.parent.name
        if sim in sim_idx and sn in snap_idx:
            matrix[sim_idx[sim], snap_idx[sn]] = 3

    n_done = (matrix == 3).sum()
    n_cat  = (matrix >= 1).sum()

    fig, ax = plt.subplots(figsize=(max(8, n_snaps * 0.45), max(6, n_sims * 0.22)))
    cmap = matplotlib.colors.ListedColormap(["white", "#a8d8ea", "#f5a623", "#27ae60"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks(range(n_snaps))
    ax.set_xticklabels([f"{sn:03d}" for sn in all_snaps], rotation=90, fontsize=6)
    ax.set_yticks(range(n_sims))
    ax.set_yticklabels([s[:35] for s in sim_names], fontsize=4)
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Simulation")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="gray", label="not started"),
        Patch(facecolor="#a8d8ea", label="catalog built"),
        Patch(facecolor="#f5a623", label="p1d computed"),
        Patch(facecolor="#27ae60", label="fully done"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)
    ax.set_title(f"Pipeline progress: {n_done}/{n_sims*n_snaps} fully done, "
                 f"{n_cat} catalogs built")

    plt.tight_layout()
    out = out_dir / "pipeline_progress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"  Status: {n_done} fully done, {n_cat} catalogs built, "
          f"{n_sims} sims × {n_snaps} snaps")


# ---------------------------------------------------------------------------
# Figure 2: log10(NHI) distributions
# ---------------------------------------------------------------------------

def plot_nhi_distributions(records: list, out_dir: Path):
    """Stacked NHI histogram from all catalogs, separated by class."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.linspace(17.0, 23.0, 61)
    centres = 0.5 * (bins[:-1] + bins[1:])

    class_cfg = [
        ("LLS",    LOG_NHI_LLS,    LOG_NHI_SUBDLA, "steelblue"),
        ("subDLA", LOG_NHI_SUBDLA, LOG_NHI_DLA,    "darkorange"),
        ("DLA",    LOG_NHI_DLA,    99.0,            "red"),
    ]

    ax = axes[0]
    for cls, lo, hi, color in class_cfg:
        all_lognhi = np.concatenate(
            [cat["log_NHI"][class_mask(cat, cls)] for _, _, _, cat in records]
        ) if records else np.array([])
        if len(all_lognhi):
            h, _ = np.histogram(all_lognhi, bins=bins)
            ax.step(centres, h, where="mid", color=color, lw=1.5, label=cls)

    ax.set_yscale("log")
    ax.set_xlabel("log₁₀(N_HI) [cm⁻²]")
    ax.set_ylabel("Count (all sims stacked)")
    ax.set_title(f"NHI distribution ({len(records)} catalogs)")
    for x, label in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax.axvline(x, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: mean absorber counts per z-bin
    ax = axes[1]
    z_vals = sorted(set(round(z, 2) for _, _, z, _ in records))
    x_pos = np.arange(len(z_vals))
    width = 0.25

    for i, (cls, _, _, color) in enumerate(class_cfg):
        counts_per_z = []
        for z in z_vals:
            cats_at_z = [cat for _, _, rz, cat in records if round(rz, 2) == z]
            counts_per_z.append(
                np.mean([class_mask(cat, cls).sum() for cat in cats_at_z]) if cats_at_z else 0
            )
        ax.bar(x_pos + i * width, counts_per_z, width, color=color, label=cls, alpha=0.8)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f"z={z:.2f}" for z in z_vals], rotation=45, fontsize=8)
    ax.set_ylabel("Mean absorbers per sim")
    ax.set_title("Mean absorber count per z-slice")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = out_dir / "nhi_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Literature dN/dX data (hardcoded from jibancat/dla_data)
# ---------------------------------------------------------------------------

# Prochaska & Wolfe 2009 DLA dN/dX (from dndx.txt)
_DNDX_DLA_LIT = {
    "label": "Prochaska & Wolfe 2009 (DLA)",
    "z_mid": np.array([2.3, 2.55, 2.85, 3.25, 3.75, 4.75]),
    "z_lo":  np.array([2.2, 2.4,  2.7,  3.0,  3.5,  4.0 ]),
    "z_hi":  np.array([2.4, 2.7,  3.0,  3.5,  4.0,  5.5 ]),
    "dndx":  np.array([0.048, 0.055, 0.067, 0.084, 0.075, 0.106]),
    "err":   np.array([0.006, 0.005, 0.006, 0.006, 0.009, 0.018]),
}


# ---------------------------------------------------------------------------
# Figure 3: CDDF from catalogs — one panel per z-bin
# ---------------------------------------------------------------------------

def plot_cddf_from_catalogs(records: list, out_dir: Path):
    """CDDF per z-bin in separate panels + Prochaska+14 spline."""
    try:
        from hcd_analysis.cddf import absorption_path_per_sightline
    except ImportError:
        # Fallback kept in sync with hcd_analysis.cddf (Apr 2026 bug fix —
        # see docs/bugs_found.md §7).
        def absorption_path_per_sightline(box_kpc_h, hubble, omegam, omegal, z):
            L_com_Mpc = box_kpc_h / 1000.0 / hubble
            H0 = hubble * 100.0
            return (1 + z)**2 * L_com_Mpc * H0 / 2.998e5

    log_nhi_bins = np.linspace(17.0, 23.0, 31)
    centres = 0.5 * (log_nhi_bins[:-1] + log_nhi_bins[1:])
    dN = 10.0**log_nhi_bins[1:] - 10.0**log_nhi_bins[:-1]

    z_vals = sorted(set(round(z, 2) for _, _, z, _ in records))
    if not z_vals:
        return

    ncols = min(len(z_vals), 5)
    nrows = (len(z_vals) + ncols - 1) // ncols
    # Do NOT use sharey — each z-bin may have very different f(N) range
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4 * nrows),
                              sharex=True)
    axes_flat = np.array(axes).flatten() if len(z_vals) > 1 else [axes]

    # Prochaska+14 spline (precompute once)
    try:
        logN_ref = np.linspace(17.0, 22.5, 300)
        fN_ref = 10.0 ** truth_cddf_prochaska2014(logN_ref)
    except Exception:
        logN_ref = fN_ref = None

    for ax_i, z in enumerate(z_vals):
        ax = axes_flat[ax_i]
        cats_at_z = [cat for _, _, rz, cat in records if round(rz, 2) == z]
        if not cats_at_z:
            ax.set_visible(False)
            continue

        all_lognhi = np.concatenate([cat["log_NHI"] for cat in cats_at_z])
        total_path = sum(
            n_sightlines(cat) * absorption_path_per_sightline(
                cat.get("box_kpc_h", 120000.0), cat.get("hubble", 0.71),
                0.3, cat.get("hubble", 0.71), z) for cat in cats_at_z)

        sim_f_nhi = None
        if len(all_lognhi) and total_path > 0:
            counts, _ = np.histogram(all_lognhi, bins=log_nhi_bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                f_nhi = np.where(dN * total_path > 0,
                                 counts / (dN * total_path), np.nan)
            sim_f_nhi = f_nhi
            mask = np.isfinite(f_nhi) & (f_nhi > 0)
            if mask.any():
                ax.semilogy(centres[mask], f_nhi[mask], "-o", ms=3, lw=1.5,
                            color="#2980b9", label=f"PRIYA ({len(cats_at_z)} sims)")

        if logN_ref is not None:
            ax.semilogy(logN_ref, fN_ref, "k--", lw=1.5,
                        label="Prochaska+14", zorder=5)

        # Auto-detect y-limits from data + reference
        all_vals = []
        if sim_f_nhi is not None:
            good = sim_f_nhi[np.isfinite(sim_f_nhi) & (sim_f_nhi > 0)]
            all_vals.extend(good.tolist())
        if fN_ref is not None:
            all_vals.extend(fN_ref[(logN_ref >= 17) & (logN_ref <= 22)].tolist())
        if all_vals:
            ylo = max(min(all_vals) / 5, 1e-30)
            yhi = max(all_vals) * 5
        else:
            ylo, yhi = 1e-26, 1e-1
        ax.set_ylim(ylo, yhi)

        for logN, cls in [(17.2, "LLS"), (19.0, "sub"), (20.3, "DLA")]:
            ax.axvline(logN, ls="--", color="gray", alpha=0.4, lw=0.8)
            ax.text(logN + 0.05, ylo * 3, cls, fontsize=6, color="gray", va="bottom")

        ax.set_title(f"z = {z:.2f}", fontsize=10)
        ax.set_xlim(17, 23)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        if ax_i % ncols == 0:
            ax.set_ylabel("f(N_HI, X)")
        if ax_i >= (nrows - 1) * ncols:
            ax.set_xlabel("log₁₀(N_HI) [cm⁻²]")

    for ax_i in range(len(z_vals), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle("CDDF per redshift bin  [PRIYA sims + Prochaska+14]", fontsize=12)
    plt.tight_layout()
    out = out_dir / "cddf_intermediate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_dndx(records: list, out_dir: Path):
    """
    dN/dX(z) for LLS, subDLA, DLA from catalogs, with literature DLA points.
    dN/dX = (total absorbers of class) / (total comoving path)
    """
    try:
        from hcd_analysis.cddf import absorption_path_per_sightline
    except ImportError:
        def absorption_path_per_sightline(box_kpc_h, hubble, omegam, omegal, z):
            box_mpc = box_kpc_h / 1000.0 / hubble
            return (1 + z) * box_mpc * 100.0 / 2.998e5

    class_cfg = [
        ("LLS",    "#27ae60", "o"),
        ("subDLA", "#f39c12", "s"),
        ("DLA",    "#e74c3c", "^"),
    ]
    z_vals = sorted(set(round(z, 2) for _, _, z, _ in records))

    fig, ax = plt.subplots(figsize=(8, 5))

    for cls, color, marker in class_cfg:
        dndx_z, dndx_v = [], []
        for z in z_vals:
            cats_at_z = [cat for _, _, rz, cat in records if round(rz, 2) == z]
            if not cats_at_z:
                continue
            n_abs = sum(class_mask(cat, cls).sum() for cat in cats_at_z)
            tot_path = sum(
                n_sightlines(cat) * absorption_path_per_sightline(
                    cat.get("box_kpc_h", 120000.0), cat.get("hubble", 0.71),
                    0.3, cat.get("hubble", 0.71), z) for cat in cats_at_z)
            if tot_path > 0:
                dndx_z.append(z)
                dndx_v.append(n_abs / tot_path)
        if dndx_z:
            ax.plot(dndx_z, dndx_v, f"{marker}-", color=color,
                    lw=1.5, ms=6, label=f"PRIYA {cls}")

    # Literature: DLA from Prochaska & Wolfe 2009
    lit = _DNDX_DLA_LIT
    ax.errorbar(lit["z_mid"], lit["dndx"], yerr=lit["err"],
                xerr=[lit["z_mid"] - lit["z_lo"], lit["z_hi"] - lit["z_mid"]],
                fmt="k*", ms=8, lw=1.2, capsize=3,
                label=lit["label"], zorder=5)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("dN/dX")
    ax.set_title("Absorber incidence rate dN/dX vs redshift")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out = out_dir / "dndx_vs_z.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: b-parameter distribution
# ---------------------------------------------------------------------------

def plot_b_parameter(records: list, out_dir: Path):
    """Distribution of Doppler b parameters for successful Voigt fits."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    class_cfg = [("LLS", "steelblue"), ("subDLA", "darkorange"), ("DLA", "red")]
    bins_b = np.linspace(0, 300, 61)

    for ax, (cls, color) in zip(axes, class_cfg):
        for _, _, _, cat in records:
            mask = class_mask(cat, cls) & cat["fit_success"] & ~cat["fast_mode"]
            b_vals = cat["b_kms"][mask]
            b_vals = b_vals[np.isfinite(b_vals)]
            if len(b_vals):
                ax.hist(b_vals, bins=bins_b, alpha=0.4, color=color, density=True)

        ax.set_xlabel("b parameter [km/s]")
        ax.set_ylabel("Probability density")
        ax.set_title(f"{cls} b-parameter")
        ax.axvline(30.0, color="k", ls="--", lw=0.8, label="b=30 km/s (init)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Voigt fit b-parameter distributions", fontsize=12)
    plt.tight_layout()
    out = out_dir / "b_parameter_dist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Parameter space coverage
# ---------------------------------------------------------------------------

def plot_param_coverage(records: list, out_dir: Path):
    """
    Multi-panel parameter space coverage.
    Panel 1: ns vs Ap (primary params), colored by N_DLA.
    Panels 2-7: each astrophysical param vs N_HCD (DLA+subDLA),
                colored by hub, to show coverage of the 9D Latin hypercube.
    """
    try:
        from hcd_analysis.io import parse_sim_params
    except ImportError:
        print("  parse_sim_params not available, skipping param coverage plot")
        return

    # Use the highest-z snapshot per sim (usually snap_004, z≈5.4) for counts
    sim_stats = {}
    for sim_name, snap, z, cat in records:
        params = parse_sim_params(sim_name)
        if params is None:
            continue
        if sim_name not in sim_stats or z > sim_stats[sim_name]["z"]:
            sim_stats[sim_name] = {
                "z": z,
                "ns":        params.get("ns", np.nan),
                "Ap":        params.get("Ap", np.nan),
                "hub":       params.get("hub", np.nan),
                "omegamh2":  params.get("omegamh2", np.nan),
                "herei":     params.get("herei", np.nan),
                "heref":     params.get("heref", np.nan),
                "alphaq":    params.get("alphaq", np.nan),
                "hireionz":  params.get("hireionz", np.nan),
                "bhfeedback":params.get("bhfeedback", np.nan),
                "n_LLS":    int(class_mask(cat, "LLS").sum()),
                "n_DLA":    int(class_mask(cat, "DLA").sum()),
                "n_subDLA": int(class_mask(cat, "subDLA").sum()),
            }

    if not sim_stats:
        print("  No parseable sim params found, skipping param coverage plot")
        return

    vals = list(sim_stats.values())
    def arr(k): return np.array([v[k] for v in vals])

    ns_arr      = arr("ns")
    Ap_arr      = arr("Ap")
    hub_arr     = arr("hub")
    omegamh2    = arr("omegamh2")
    herei       = arr("herei")
    heref       = arr("heref")
    alphaq      = arr("alphaq")
    hireionz    = arr("hireionz")
    bhfeedback  = arr("bhfeedback")
    n_DLA       = arr("n_DLA")
    n_subDLA    = arr("n_subDLA")
    n_LLS       = arr("n_LLS")
    n_HCD       = n_DLA + n_subDLA + n_LLS

    # ── Panel layout: 3 rows × 3 cols ─────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()

    panels = [
        # (xarr, xlabel, yarr, ylabel, carr, clabel, title)
        (ns_arr,    "Spectral index $n_s$",
         Ap_arr*1e9,"$A_p$ [×10⁻⁹]",
         n_DLA,     "N_DLA",      "Cosmo: $n_s$ vs $A_p$ | colour=N_DLA"),
        (ns_arr,    "Spectral index $n_s$",
         Ap_arr*1e9,"$A_p$ [×10⁻⁹]",
         n_LLS,     "N_LLS",      "Cosmo: $n_s$ vs $A_p$ | colour=N_LLS"),
        (hub_arr,   "Hubble $h$",
         omegamh2,  r"$\Omega_m h^2$",
         n_HCD,     "N_HCD",      "Cosmo: $h$ vs $\Omega_m h^2$ | colour=N_HCD"),
        (herei,     "HeII reion. start $z_i$",
         heref,     "HeII reion. end $z_f$",
         n_HCD,     "N_HCD",      "HeII reion.: $z_i$ vs $z_f$ | colour=N_HCD"),
        (alphaq,    r"$\alpha_q$ (quasar slope)",
         herei,     "HeII reion. start $z_i$",
         n_HCD,     "N_HCD",      r"HeII: $\alpha_q$ vs $z_i$ | colour=N_HCD"),
        (hireionz,  "HI reion. $z_{re}$",
         n_HCD,     "N_HCD",
         hub_arr,   "$h$",        "HI reion.: $z_{re}$ vs N_HCD | colour=$h$"),
        (bhfeedback,"AGN feedback strength",
         n_HCD,     "N_HCD",
         omegamh2,  r"$\Omega_m h^2$","AGN feedback vs N_HCD | colour=$\\Omega_m h^2$"),
        (hireionz,  "HI reion. $z_{re}$",
         bhfeedback,"AGN feedback",
         n_HCD,     "N_HCD",      "HI reion. vs AGN feedback | colour=N_HCD"),
        (alphaq,    r"$\alpha_q$",
         heref,     "HeII reion. end $z_f$",
         n_HCD,     "N_HCD",      r"HeII: $\alpha_q$ vs $z_f$ | colour=N_HCD"),
    ]

    for ax, (xarr, xlabel, yarr, ylabel, carr, clabel, title) in zip(axes_flat, panels):
        ok = np.isfinite(xarr) & np.isfinite(yarr) & np.isfinite(carr)
        sc = ax.scatter(xarr[ok], yarr[ok], c=carr[ok], cmap="plasma", s=55, alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.85)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Parameter space coverage & HCD statistics  ({len(sim_stats)} sims)",
                 fontsize=13)
    plt.tight_layout()
    out = out_dir / "param_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Absorber incidence vs redshift
# ---------------------------------------------------------------------------

def plot_absorber_counts(records: list, out_dir: Path):
    """Absorber incidence rate per sightline vs redshift."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    class_names = ["LLS", "subDLA", "DLA"]
    colors_list = ["steelblue", "darkorange", "red"]

    for ax, cls, color in zip(axes, class_names, colors_list):
        for _, _, z, cat in records:
            n_abs = class_mask(cat, cls).sum()
            n_sl  = n_sightlines(cat)
            ax.scatter(z, n_abs / max(1, n_sl), color=color, alpha=0.5, s=20, zorder=3)

        ax.set_xlabel("Redshift z")
        ax.set_ylabel(f"N({cls}) per sightline")
        ax.set_title(cls)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Absorber incidence rate ({len(records)} catalogs)", fontsize=12)
    plt.tight_layout()
    out = out_dir / "absorber_counts_vs_z.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 7: P1D curves (only if p1d.npz available)
# ---------------------------------------------------------------------------

def _interp_to_kref(k_src, p_src, k_ref):
    """Interpolate p_src onto k_ref; return NaN where out of range."""
    valid = np.isfinite(p_src) & (k_src > 0) & (p_src > 0)
    if valid.sum() < 2:
        return np.full_like(k_ref, np.nan)
    return np.interp(k_ref, k_src[valid], p_src[valid], left=np.nan, right=np.nan)


def _median_ratio_per_z(p1d_records, k_label, p_label, k_ref_label="k_all"):
    """
    For each z-bin, compute the median ratio p_label/p1d_all across sims.
    Returns {z: (k_ref, median_ratio, p16, p84)}.
    """
    from collections import defaultdict
    by_z = defaultdict(list)
    for _, _, z, d in p1d_records:
        if not (k_ref_label in d and "p1d_all" in d and k_label in d and p_label in d):
            continue
        k_ref = d[k_ref_label]
        p_excl = _interp_to_kref(d[k_label], d[p_label], k_ref)
        p_all  = d["p1d_all"]
        ratio  = np.where((p_all > 0) & (k_ref > 0), p_excl / p_all, np.nan)
        by_z[round(z, 2)].append((k_ref, ratio))

    result = {}
    for z, pairs in by_z.items():
        k_ref = pairs[0][0]
        ratios = np.array([np.interp(k_ref, p[0], p[1],
                                     left=np.nan, right=np.nan) for p in pairs])
        result[z] = (k_ref,
                     np.nanmedian(ratios, axis=0),
                     np.nanpercentile(ratios, 16, axis=0),
                     np.nanpercentile(ratios, 84, axis=0))
    return result


# --- Set 1: contamination ratios  numerator/p1d_no_HCD ---
# Each entry: (label, k_num_key, p_num_key, color)
# Numerator = P1D with those classes PRESENT; denominator = p1d_no_HCD (clean forest)
_CONTAM_CLASSES = [
    ("DLA+subDLA+LLS / forest", "k_all",       "p1d_all",       "#e74c3c"),
    ("DLA+subDLA / forest",     "k_no_LLS",    "p1d_no_LLS",    "#f39c12"),
    ("DLA+LLS / forest",        "k_no_subDLA", "p1d_no_subDLA", "#27ae60"),
    ("subDLA+LLS / forest",     "k_no_DLA",    "p1d_no_DLA",    "#9b59b6"),
]

# --- Set 2: masking-effect ratios  p1d_no_X/p1d_all ---
# Each entry: (label, k_key, p_key, color)
_MASK_CLASSES = [
    ("excl DLA / all",         "k_no_DLA",    "p1d_no_DLA",    "#e74c3c"),
    ("excl subDLA / all",      "k_no_subDLA", "p1d_no_subDLA", "#f39c12"),
    ("excl LLS / all",         "k_no_LLS",    "p1d_no_LLS",    "#27ae60"),
    ("excl all HCD / all",     "k_no_HCD",    "p1d_no_HCD",    "#2980b9"),
]


def _ratio_panels(p1d_records, class_list, denom_k_key, denom_p_key,
                  out_path, title, ylabel, ylim=(0.95, 1.05)):
    """
    Generic helper: one panel per class in class_list showing
    median(numerator / denominator) vs k, coloured by z.

    denom_k_key / denom_p_key: keys for the shared denominator P1D.
    Each class entry: (label, k_num_key, p_num_key, color)  [color unused here;
    curves are z-coloured instead, classes go on separate panels].
    """
    z_vals = sorted(set(round(z, 2) for _, _, z, _ in p1d_records))
    if not z_vals:
        return
    z_cmap = cm.plasma
    z_norm = matplotlib.colors.Normalize(vmin=min(z_vals), vmax=max(z_vals))

    n = len(class_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, k_num, p_num, _color) in zip(axes, class_list):
        # build per-z medians of  num/denom
        by_z = {}
        for z in z_vals:
            stack = []
            for _, _, rz, d in p1d_records:
                if round(rz, 2) != z:
                    continue
                if not all(kk in d for kk in (k_num, p_num, denom_k_key, denom_p_key)):
                    continue
                k_ref = d["k_all"] if "k_all" in d else d[k_num]
                num   = _interp_to_kref(d[k_num],       d[p_num],       k_ref)
                den   = _interp_to_kref(d[denom_k_key], d[denom_p_key], k_ref)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(den > 0, num / den, np.nan)
                stack.append((k_ref, ratio))
            if len(stack) < 1:
                continue
            k_ref = stack[0][0]
            ratios = np.array([np.interp(k_ref, s[0], s[1],
                                         left=np.nan, right=np.nan) for s in stack])
            by_z[z] = (k_ref,
                       np.nanmedian(ratios, axis=0),
                       np.nanpercentile(ratios, 16, axis=0),
                       np.nanpercentile(ratios, 84, axis=0))

        for z, (k_ref, med, p16, p84) in sorted(by_z.items()):
            zcolor = z_cmap(z_norm(z))
            valid = np.isfinite(med) & (k_ref > 0)
            ax.semilogx(k_ref[valid], med[valid], color=zcolor, lw=1.5,
                        label=f"z={z:.2f}")
            ax.fill_between(k_ref[valid], p16[valid], p84[valid],
                            color=zcolor, alpha=0.15)

        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("k [s/km]")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Auto-center y=1: find max deviation across all axes, then set symmetric limits
    all_lines_data = []
    for ax in axes:
        for line in ax.get_lines():
            yd = np.asarray(line.get_ydata(), dtype=float)
            yd = yd[np.isfinite(yd)]
            if len(yd) > 2:   # skip short axhline/axvline stubs
                all_lines_data.extend(yd.tolist())
    if all_lines_data:
        max_dev = max(abs(np.array(all_lines_data) - 1.0).max(), 0.02)
        margin = max_dev * 1.25
        for ax in axes:
            ax.set_ylim(1.0 - margin, 1.0 + margin)
    else:
        for ax in axes:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel)
    sm = plt.cm.ScalarMappable(cmap=z_cmap, norm=z_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Redshift z", shrink=0.85)
    fig.suptitle(title + f"  [median ± 1σ across sims]", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_p1d_from_files(p1d_records: list, out_dir: Path):
    """
    Three output figures:
    1. p1d_by_class.png       — k·P1D/π per z-bin with all exclusion curves overlaid
    2. p1d_contamination.png  — P1D(X present)/P1D(forest)  [contamination view]
    3. p1d_masking.png        — P1D(excl X)/P1D(all)        [masking-gain view]
    4. p1d_intermediate.png   — quick overview (all-sim scatter, kept for compat)
    """
    z_vals = sorted(set(round(z, 2) for _, _, z, _ in p1d_records))
    if not z_vals:
        return
    z_cmap = cm.plasma
    z_norm = matplotlib.colors.Normalize(vmin=min(z_vals), vmax=max(z_vals))

    # ---- Figure 1: absolute k·P1D/π, one panel per z, lines per class ----
    all_curves = [
        ("all",           "k_all",            "p1d_all",            "k",       "-",  2.0),
        ("no DLA (catalog)", "k_no_DLA",      "p1d_no_DLA",         "#e74c3c", "--", 1.0),
        ("no DLA (PRIYA)", "k_no_DLA_priya",  "p1d_no_DLA_priya",   "#c0392b", "-",  1.5),
        ("no subDLA",     "k_no_subDLA",       "p1d_no_subDLA",      "#f39c12", "--", 1.3),
        ("no LLS",        "k_no_LLS",          "p1d_no_LLS",         "#27ae60", "--", 1.3),
        ("no HCD (pure forest)", "k_no_HCD",   "p1d_no_HCD",         "#2980b9", ":", 1.8),
    ]
    n_z = len(z_vals)
    fig_a, axes_a = plt.subplots(1, n_z, figsize=(4 * n_z, 5), sharey=True)
    if n_z == 1:
        axes_a = [axes_a]

    for ax, z in zip(axes_a, z_vals):
        recs_z = [d for _, _, rz, d in p1d_records if round(rz, 2) == z]
        if not recs_z:
            continue
        k_ref = recs_z[0]["k_all"]
        for label, k_key, p_key, color, ls, lw in all_curves:
            stack = [_interp_to_kref(d[k_key], d[p_key], k_ref)
                     for d in recs_z if k_key in d and p_key in d]
            if not stack:
                continue
            med = np.nanmedian(stack, axis=0)
            valid = (k_ref > 0) & np.isfinite(med) & (med > 0)
            ax.loglog(k_ref[valid], k_ref[valid] * med[valid] / np.pi,
                      color=color, ls=ls, lw=lw, label=label)

        ax.set_xlabel("k [s/km]")
        if ax is axes_a[0]:
            ax.set_ylabel("k · P₁D(k) / π")
        ax.set_title(f"z = {z:.2f}  ({len(recs_z)} sims)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig_a.suptitle("P1D by HCD exclusion class (medians across sims)", fontsize=12)
    plt.tight_layout()
    fig_a.savefig(out_dir / "p1d_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved {out_dir / 'p1d_by_class.png'}")

    # ---- Figure 2: contamination ratios  P1D(X present) / P1D(forest) ----
    # Distinguish: PRIYA masking (tau>10^6 detect + tau>0.25+tau_eff mask) vs
    # catalog masking (Voigt-fitted pix_start/pix_end, too narrow for DLA wings).
    # If no_DLA_priya is available, use it as the forest denominator (correct).
    # Otherwise fall back to no_HCD from catalog masking (wings not removed).
    has_priya = any("p1d_no_DLA_priya" in d for _, _, _, d in p1d_records)
    forest_k   = "k_no_DLA_priya"   if has_priya else "k_no_HCD"
    forest_p   = "p1d_no_DLA_priya" if has_priya else "p1d_no_HCD"
    forest_lbl = "PRIYA DLA masked" if has_priya else "catalog masked (no wings!)"

    contam_classes = list(_CONTAM_CLASSES)
    if has_priya:
        # Add PRIYA ratio: P1D_all / P1D_no_DLA_priya  (shows large-scale DLA boost)
        contam_classes = [
            ("all / PRIYA-masked DLA [correct]",
             "k_all", "p1d_all", "#e74c3c"),
        ] + contam_classes

    _ratio_panels(
        p1d_records, contam_classes,
        denom_k_key=forest_k, denom_p_key=forest_p,
        out_path=out_dir / "p1d_contamination.png",
        title=(f"HCD contamination: P1D(X present) / P1D({forest_lbl})\n"
               + ("" if has_priya else
                  "⚠ PRIYA variant not yet computed — re-run pipeline for correct damping-wing masking")),
        ylabel="P₁D(X present) / P₁D(forest)",
        ylim=(0.8, 1.25),
    )

    # ---- Figure 3: masking-gain ratios  P1D(excl X) / P1D(all) ----
    mask_classes = list(_MASK_CLASSES)
    if has_priya:
        mask_classes = [
            ("excl DLA (PRIYA) / all", "k_no_DLA_priya", "p1d_no_DLA_priya", "#c0392b"),
        ] + mask_classes

    _ratio_panels(
        p1d_records, mask_classes,
        denom_k_key="k_all", denom_p_key="p1d_all",
        out_path=out_dir / "p1d_masking.png",
        title="Masking gain: P1D(excl X) / P1D(all)  [PRIYA method when available]",
        ylabel="P₁D(excl X) / P₁D(all)",
        ylim=(0.8, 1.25),
    )

    # ---- Figure 4: quick scatter overview (all sims, colour = z) ----
    fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 5))
    for sim, snap, z, data in p1d_records:
        zcolor = z_cmap(z_norm(round(z, 2)))
        if "k_all" in data and "p1d_all" in data:
            k = data["k_all"]; p = data["p1d_all"]
            valid = np.isfinite(p) & (p > 0) & (k > 0)
            if valid.any():
                axes_d[0].loglog(k[valid], k[valid] * p[valid] / np.pi,
                                 color=zcolor, alpha=0.3, lw=0.7)
        if "k_no_HCD" in data and "p1d_no_HCD" in data and "p1d_all" in data:
            k_all = data["k_all"]; p_all = data["p1d_all"]
            p_no = _interp_to_kref(data["k_no_HCD"], data["p1d_no_HCD"], k_all)
            valid = np.isfinite(p_no) & (p_no > 0) & (p_all > 0) & (k_all > 0)
            if valid.any():
                axes_d[1].semilogx(k_all[valid], p_all[valid] / p_no[valid],
                                   color=zcolor, alpha=0.3, lw=0.7)

    axes_d[0].set_xlabel("k [s/km]"); axes_d[0].set_ylabel("k · P₁D(k) / π")
    axes_d[0].set_title(f"All sims ({len(p1d_records)} snaps)")
    axes_d[0].grid(True, alpha=0.3)
    axes_d[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes_d[1].set_xlabel("k [s/km]")
    axes_d[1].set_ylabel("P₁D(all) / P₁D(no HCD)")
    axes_d[1].set_title("Total HCD contamination (all sims)")
    axes_d[1].grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=z_cmap, norm=z_norm)
    sm.set_array([])
    fig_d.colorbar(sm, ax=axes_d, label="Redshift z", shrink=0.85)
    plt.tight_layout()
    fig_d.savefig(out_dir / "p1d_intermediate.png", dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"Saved {out_dir / 'p1d_intermediate.png'}")


# ---------------------------------------------------------------------------
# Figure 7b: Rogers+2018-style sightline-exclusion P1D ratios
# ---------------------------------------------------------------------------

def plot_p1d_sightline_excl(p1d_records: list, excl_records: list,
                             out_dir: Path):
    """
    Rogers+2018-style contamination plots using sightline-level NHI cuts
    (p1d_excl.npz).  Denominator = p1d_excl at logN_cut=17.2 (pure forest).

    Figure A — absolute k·P1D/π: one panel per z-bin, lines:
      pure forest (logN<17.2), forest+LLS (logN<19.0),
      forest+LLS+subDLA (logN<20.3), all sightlines (p1d_all).

    Figure B — ratio / pure forest (Rogers Fig 4 style):
      y-axis spans boost (>1 at large scales) and suppression (<1),
      one panel per class, curves coloured by z.
    """
    if not excl_records:
        print("  No p1d_excl files yet — skipping sightline-exclusion plots")
        return

    # Standard cut indices in log_nhi_cuts = [17.2, 17.5, 18., 18.5, 19., 19.5, 20., 20.3, 20.5, 21.]
    # Index 0 → 17.2 (pure forest denominator)
    # Index 4 → 19.0 (forest + LLS)
    # Index 7 → 20.3 (forest + LLS + subDLA)
    IDX_FOREST = 0    # logN > 17.2 excluded  → pure forest
    IDX_NO_SUBDLA_DLA = 4   # logN > 19.0 excluded → forest + LLS
    IDX_NO_DLA = 7    # logN > 20.3 excluded → forest + LLS + subDLA

    # Verify indices against actual cuts
    sample_cuts = excl_records[0][3]["log_nhi_cuts"]
    def _find_cut(target):
        idx = np.argmin(np.abs(sample_cuts - target))
        if abs(sample_cuts[idx] - target) > 0.05:
            print(f"  Warning: requested logN cut {target} not found; "
                  f"using {sample_cuts[idx]:.2f} (idx {idx})")
        return idx
    IDX_FOREST       = _find_cut(17.2)
    IDX_NO_SUBDLA_DLA = _find_cut(19.0)
    IDX_NO_DLA       = _find_cut(20.3)

    # Build lookup: (sim, snap) → excl_dict
    excl_lookup = {(sim, snap): d for sim, snap, z, d in excl_records}
    # z lookup from excl records
    z_of = {(sim, snap): z for sim, snap, z, d in excl_records}

    z_vals = sorted(set(round(z, 2) for _, _, z, _ in excl_records))
    if not z_vals:
        return
    z_cmap = cm.plasma
    z_norm = matplotlib.colors.Normalize(vmin=min(z_vals), vmax=max(z_vals))

    # Curves: (label, cut_index_or_'all', ls, color)
    # 'all' means use p1d_all from p1d.npz
    curves = [
        ("pure forest (excl logN>17.2)",        IDX_FOREST,        "k",       "-",  2.2),
        ("forest + LLS (excl logN>19.0)",        IDX_NO_SUBDLA_DLA, "#27ae60", "--", 1.5),
        ("forest + LLS + subDLA (excl logN>20.3)",IDX_NO_DLA,       "#f39c12", "--", 1.5),
        ("all sightlines (p1d_all)",             "all",             "#e74c3c", "-.", 1.5),
    ]

    # ---- Figure A: absolute k·P1D/π per z ----
    n_z = len(z_vals)
    fig_a, axes_a = plt.subplots(1, n_z, figsize=(4.5 * n_z, 5), sharey=True)
    if n_z == 1:
        axes_a = [axes_a]

    for ax, z in zip(axes_a, z_vals):
        recs_z = [(sim, snap, d) for sim, snap, rz, d in excl_records
                  if round(rz, 2) == z]
        p1d_all_z = {sim: d for sim, snap, rz, d in p1d_records
                     if round(rz, 2) == z}

        for label, idx, color, ls, lw in curves:
            stack = []
            if idx == "all":
                for sim, snap, _ in recs_z:
                    pd = p1d_all_z.get(sim)
                    if pd is not None and "k_all" in pd and "p1d_all" in pd:
                        stack.append((pd["k_all"],
                                      _interp_to_kref(pd["k_all"], pd["p1d_all"],
                                                      pd["k_all"])))
            else:
                for sim, snap, d in recs_z:
                    k = d["k"]; p = d["p1d_excl"][idx]
                    valid = (k > 0) & np.isfinite(p) & (p > 0)
                    if valid.sum() > 2:
                        stack.append((k, p))
            if not stack:
                continue
            k_ref = stack[0][0]
            med = np.nanmedian(
                [np.interp(k_ref, s[0], s[1], left=np.nan, right=np.nan)
                 for s in stack], axis=0)
            valid = (k_ref > 0) & np.isfinite(med) & (med > 0)
            ax.loglog(k_ref[valid], k_ref[valid] * med[valid] / np.pi,
                      color=color, ls=ls, lw=lw, label=label)

        ax.set_xlabel("k [s/km]")
        if ax is axes_a[0]:
            ax.set_ylabel("k · P₁D(k) / π")
        ax.set_title(f"z = {z:.2f}  ({len(recs_z)} sims)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig_a.suptitle("P1D by sightline class (Rogers+2018 style, medians across sims)",
                   fontsize=12)
    plt.tight_layout()
    fig_a.savefig(out_dir / "p1d_sightline_absolute.png", dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved {out_dir / 'p1d_sightline_absolute.png'}")

    # ---- Figure B: ratio / pure forest — Rogers Fig 4 style ----
    ratio_curves = [
        ("LLS effect\n(forest+LLS) / forest",
         IDX_NO_SUBDLA_DLA, "#27ae60"),
        ("LLS + subDLA effect\n(forest+LLS+subDLA) / forest",
         IDX_NO_DLA,        "#f39c12"),
        ("All HCDs effect\np1d_all / forest",
         "all",             "#e74c3c"),
    ]

    fig_b, axes_b = plt.subplots(1, len(ratio_curves),
                                  figsize=(5.5 * len(ratio_curves), 5), sharey=True)
    if len(ratio_curves) == 1:
        axes_b = [axes_b]

    for ax, (label, idx, _color) in zip(axes_b, ratio_curves):
        by_z = {}
        for z in z_vals:
            recs_z = [(sim, snap, d) for sim, snap, rz, d in excl_records
                      if round(rz, 2) == z]
            p1d_all_z = {sim: d for sim, snap, rz, d in p1d_records
                         if round(rz, 2) == z}
            stack = []
            for sim, snap, d in recs_z:
                k_ref = d["k"]
                denom = d["p1d_excl"][IDX_FOREST]
                valid_den = (k_ref > 0) & np.isfinite(denom) & (denom > 0)
                if valid_den.sum() < 2:
                    continue
                if idx == "all":
                    pd = p1d_all_z.get(sim)
                    if pd is None or "p1d_all" not in pd:
                        continue
                    num = _interp_to_kref(pd["k_all"], pd["p1d_all"], k_ref)
                else:
                    num = d["p1d_excl"][idx]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(denom > 0, num / denom, np.nan)
                stack.append((k_ref, ratio))
            if len(stack) < 1:
                continue
            k_ref = stack[0][0]
            ratios = np.array([np.interp(k_ref, s[0], s[1],
                                         left=np.nan, right=np.nan) for s in stack])
            by_z[z] = (k_ref,
                       np.nanmedian(ratios, axis=0),
                       np.nanpercentile(ratios, 16, axis=0),
                       np.nanpercentile(ratios, 84, axis=0))

        for z, (k_ref, med, p16, p84) in sorted(by_z.items()):
            zcolor = z_cmap(z_norm(z))
            valid = np.isfinite(med) & (k_ref > 0)
            ax.semilogx(k_ref[valid], med[valid], color=zcolor, lw=1.5,
                        label=f"z={z:.2f}")
            ax.fill_between(k_ref[valid], p16[valid], p84[valid],
                            color=zcolor, alpha=0.15)

        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("k [s/km]")
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    # Auto-center y=1
    all_yd = [y for ax in axes_b for line in ax.get_lines()
              if len(line.get_ydata()) > 2
              for y in np.asarray(line.get_ydata(), dtype=float)
              if np.isfinite(y)]
    if all_yd:
        max_dev = max(abs(np.array(all_yd) - 1.0).max(), 0.02)
        margin = max_dev * 1.25
        for ax in axes_b:
            ax.set_ylim(1.0 - margin, 1.0 + margin)

    axes_b[0].set_ylabel("P₁D(sightlines with HCDs) / P₁D(pure forest)")
    sm = plt.cm.ScalarMappable(cmap=z_cmap, norm=z_norm)
    sm.set_array([])
    fig_b.colorbar(sm, ax=axes_b, label="Redshift z", shrink=0.85)
    fig_b.suptitle(
        "HCD contamination — sightline exclusion (Rogers+2018 style)\n"
        "ratio > 1: large-scale power boost;  ratio < 1: small-scale suppression",
        fontsize=11)
    plt.tight_layout()
    fig_b.savefig(out_dir / "p1d_sightline_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved {out_dir / 'p1d_sightline_ratio.png'}")


# ---------------------------------------------------------------------------
# Figure 8: P1D ratio vs simulation parameters
# ---------------------------------------------------------------------------

def plot_p1d_param_sensitivity(p1d_records: list, out_dir: Path,
                                ref_z: float | None = None, dz: float = 0.3):
    """
    For each key cosmological/astrophysical parameter, plot the HCD suppression
    ratio P1D(withHCD)/P1D(withoutHCD) coloured by that parameter's value.
    Auto-selects the most-populated z bin if ref_z is None.
    """
    params_to_plot = [
        ("ns",         "Spectral index $n_s$"),
        ("Ap",         "Scalar amplitude $A_p$"),
        ("hub",        "Hubble $h$"),
        ("omegamh2",   r"$\Omega_m h^2$"),
        ("hireionz",   "H\,I reion. redshift"),
        ("bhfeedback", "AGN feedback"),
        ("alphaq",     r"$\alpha_q$ (HeII reion.)"),
        ("herei",      "HeII reion. start $z$"),
        ("heref",      "HeII reion. end $z$"),
    ]

    # Auto-pick most-populated z bin
    if ref_z is None:
        from collections import Counter
        z_counts = Counter(round(z, 2) for _, _, z, _ in p1d_records)
        if not z_counts:
            print("  No P1D records — skipping sensitivity plot")
            return
        ref_z = z_counts.most_common(1)[0][0]
        print(f"  Sensitivity plot: auto-selected z={ref_z:.2f} ({z_counts[ref_z]} records)")

    # Collect ratio curves near ref_z
    recs = []
    for sim, snap, z, data in p1d_records:
        if abs(z - ref_z) > dz:
            continue
        if not ("k_all" in data and "p1d_all" in data and "k_no_HCD" in data and "p1d_no_HCD" in data):
            continue
        k_no = data["k_no_HCD"]; p_no = data["p1d_no_HCD"]
        valid_no = np.isfinite(p_no) & (k_no > 0) & (p_no > 0)
        if valid_no.sum() < 2:
            continue
        params = _parse_sim_params(sim)
        if params is None:
            continue
        k_all = data["k_all"]
        p_no_i = np.interp(k_all, k_no[valid_no], p_no[valid_no], left=np.nan, right=np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(p_no_i > 0, data["p1d_all"] / p_no_i, np.nan)
        valid = np.isfinite(ratio) & (k_all > 0)
        recs.append({"params": params, "k": k_all[valid], "ratio": ratio[valid], "z": z})

    if not recs:
        print(f"  No P1D records near z={ref_z:.1f}±{dz} with parsed params — skipping sensitivity plot")
        return

    nparams = len(params_to_plot)
    ncols = 3
    nrows = (nparams + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax_i, (pkey, plabel) in enumerate(params_to_plot):
        ax = axes_flat[ax_i]
        vals = [r["params"][pkey] for r in recs if pkey in r["params"]]
        if not vals:
            ax.set_visible(False)
            continue

        vmin, vmax = min(vals), max(vals)
        cmap = cm.plasma
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for r in recs:
            if pkey not in r["params"]:
                continue
            pval = r["params"][pkey]
            color = cmap(norm(pval))
            ax.semilogx(r["k"], r["ratio"], color=color, alpha=0.6, lw=0.9)

        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("k [s/km]")
        ax.set_ylabel("P₁D(HCD) / P₁D(no HCD)")
        ax.set_title(f"{plabel}\n(z≈{ref_z:.1f}, {len(vals)} sims)")
        # auto-center y=1
        yd = [y for line in ax.get_lines() if len(line.get_ydata()) > 2
              for y in np.asarray(line.get_ydata(), dtype=float) if np.isfinite(y)]
        if yd:
            max_dev = max(abs(np.array(yd) - 1.0).max(), 0.02)
            ax.set_ylim(1.0 - max_dev * 1.25, 1.0 + max_dev * 1.25)
        ax.grid(True, alpha=0.3)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                     ax=ax, label=pkey, shrink=0.85)

    for ax_i in range(nparams, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(f"HCD suppression ratio vs simulation parameters  (z≈{ref_z:.1f}±{dz})",
                 fontsize=13)
    plt.tight_layout()
    out = out_dir / "p1d_param_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  ({len(recs)} records near z={ref_z:.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Intermediate analysis plots")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--out-dir",     default=str(OUT_DIR))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output root : {output_root}")
    print(f"Saving to   : {out_dir}", flush=True)

    # --- Figure 1: progress dashboard (no catalog loading required) ---
    print("\n--- Pipeline progress dashboard ---", flush=True)
    plot_progress_dashboard(output_root, out_dir)

    # --- Load catalogs (fast raw-array path) ---
    print("\n--- Loading catalogs ---", flush=True)
    records = load_all_catalogs(output_root)

    if not records:
        print("No catalogs found yet. Check back later.")
        return

    print(f"\n--- Plotting {len(records)} catalogs ---", flush=True)
    plot_nhi_distributions(records, out_dir)
    plot_cddf_from_catalogs(records, out_dir)
    plot_dndx(records, out_dir)
    plot_b_parameter(records, out_dir)
    plot_param_coverage(records, out_dir)
    plot_absorber_counts(records, out_dir)

    # --- P1D (only if available) ---
    print("\n--- Loading P1D ---", flush=True)
    p1d_records = load_all_p1d(output_root)
    if p1d_records:
        plot_p1d_from_files(p1d_records, out_dir)
        plot_p1d_param_sensitivity(p1d_records, out_dir)
    else:
        print("No P1D files yet (jobs still processing).")

    # --- Sightline-exclusion (Rogers+2018-style) plots ---
    print("\n--- Loading p1d_excl (sightline exclusion) ---", flush=True)
    excl_records = load_all_p1d_excl(output_root)
    if excl_records:
        plot_p1d_sightline_excl(p1d_records, excl_records, out_dir)
    else:
        print("No p1d_excl files yet.")

    print(f"\nAll figures saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
