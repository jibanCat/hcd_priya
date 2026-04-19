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


# ---------------------------------------------------------------------------
# Fast data loading — raw numpy arrays, no Python per-object loop
# ---------------------------------------------------------------------------

def load_cat_fast(cat_path: Path) -> dict:
    """Load catalog.npz as a dict of numpy arrays (no Absorber object construction)."""
    d = np.load(str(cat_path), allow_pickle=True)
    NHI = d["NHI"].astype(np.float64)
    log_NHI = np.log10(np.maximum(NHI, 1e1))
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
    """Estimate number of sightlines from max skewer index."""
    if len(cat["skewer_idx"]) == 0:
        return 691200
    return int(cat["skewer_idx"].max()) + 1


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
# Figure 3: CDDF from catalogs
# ---------------------------------------------------------------------------

def plot_cddf_from_catalogs(records: list, out_dir: Path):
    """Quick CDDF grouped by z from all available catalogs."""
    try:
        from hcd_analysis.cddf import absorption_path_per_sightline
    except ImportError:
        def absorption_path_per_sightline(box_kpc_h, hubble, omegam, omegal, z):
            """Fallback comoving absorption path element per sightline."""
            box_mpc = box_kpc_h / 1000.0 / hubble
            H_z = 100.0 * hubble * np.sqrt(omegam * (1 + z)**3 + omegal)
            dX = (1 + z) * box_mpc * 100.0 / 2.998e5
            return dX

    fig, ax = plt.subplots(figsize=(9, 6))

    log_nhi_bins = np.linspace(17.0, 23.0, 31)
    centres = 0.5 * (log_nhi_bins[:-1] + log_nhi_bins[1:])
    dN = 10.0**log_nhi_bins[1:] - 10.0**log_nhi_bins[:-1]

    z_vals = sorted(set(round(z, 2) for _, _, z, _ in records))
    cmap = cm.viridis
    colors = {z: cmap(i / max(len(z_vals) - 1, 1)) for i, z in enumerate(z_vals)}

    for z in z_vals:
        cats_at_z = [cat for _, _, rz, cat in records if round(rz, 2) == z]
        if not cats_at_z:
            continue

        all_lognhi = np.concatenate([cat["log_NHI"] for cat in cats_at_z])
        total_path = sum(
            n_sightlines(cat) * absorption_path_per_sightline(
                120000.0, 0.71, 0.29, 0.71, z
            ) for cat in cats_at_z
        )

        if not len(all_lognhi) or total_path == 0:
            continue

        counts, _ = np.histogram(all_lognhi, bins=log_nhi_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_nhi = np.where(dN * total_path > 0, counts / (dN * total_path), np.nan)

        mask = np.isfinite(f_nhi) & (f_nhi > 0)
        if mask.any():
            ax.semilogy(centres[mask], f_nhi[mask], "-o", ms=3, lw=1.2,
                        color=colors[z], label=f"z={z:.2f} ({len(cats_at_z)} sims)")

    for logN, cls in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax.axvline(logN, ls="--", color="gray", alpha=0.4, lw=0.8)
        ax.text(logN + 0.05, 1e-5, cls, fontsize=7, color="gray", va="bottom")

    ax.set_xlabel("log₁₀(N_HI) [cm⁻²]")
    ax.set_ylabel("f(N_HI, X)")
    ax.set_title("CDDF from available catalogs")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(17, 23)

    plt.tight_layout()
    out = out_dir / "cddf_intermediate.png"
    fig.savefig(out, dpi=150)
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
    """Scatter of sim parameter space (ns vs Ap) coloured by absorber count."""
    try:
        from hcd_analysis.io import parse_sim_params
    except ImportError:
        print("  parse_sim_params not available, skipping param coverage plot")
        return

    sim_stats = {}
    for sim_name, snap, z, cat in records:
        params = parse_sim_params(sim_name)
        if params is None:
            continue
        if sim_name not in sim_stats or z > sim_stats[sim_name]["z"]:
            sim_stats[sim_name] = {
                "z": z,
                "ns": params["ns"],
                "Ap": params["Ap"],
                "n_LLS":    class_mask(cat, "LLS").sum(),
                "n_DLA":    class_mask(cat, "DLA").sum(),
                "n_subDLA": class_mask(cat, "subDLA").sum(),
            }

    if not sim_stats:
        print("  No parseable sim params found, skipping param coverage plot")
        return

    ns_arr = np.array([v["ns"] for v in sim_stats.values()])
    Ap_arr = np.array([v["Ap"] for v in sim_stats.values()])
    n_DLA  = np.array([v["n_DLA"] for v in sim_stats.values()])
    n_LLS  = np.array([v["n_LLS"] for v in sim_stats.values()])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc1 = axes[0].scatter(ns_arr, Ap_arr * 1e9, c=n_DLA, cmap="plasma", s=60, alpha=0.8)
    axes[0].set_xlabel("Spectral index ns")
    axes[0].set_ylabel("Scalar amplitude Ap [×10⁻⁹]")
    axes[0].set_title("DLA count at highest available z")
    plt.colorbar(sc1, ax=axes[0], label="N_DLA per sim")
    axes[0].grid(True, alpha=0.3)

    sc2 = axes[1].scatter(ns_arr, Ap_arr * 1e9, c=n_LLS, cmap="viridis", s=60, alpha=0.8)
    axes[1].set_xlabel("Spectral index ns")
    axes[1].set_ylabel("Scalar amplitude Ap [×10⁻⁹]")
    axes[1].set_title("LLS count at highest available z")
    plt.colorbar(sc2, ax=axes[1], label="N_LLS per sim")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Parameter space coverage ({len(sim_stats)} sims)", fontsize=12)
    plt.tight_layout()
    out = out_dir / "param_coverage.png"
    fig.savefig(out, dpi=150)
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

def plot_p1d_from_files(p1d_records: list, out_dir: Path):
    """P1D curves from available p1d.npz files + HCD suppression ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cmap = cm.viridis
    z_vals = sorted(set(round(z, 2) for _, _, z, _ in p1d_records))
    colors = {z: cmap(i / max(len(z_vals) - 1, 1)) for i, z in enumerate(z_vals)}

    for sim, snap, z, data in p1d_records:
        color = colors.get(round(z, 2), "gray")
        if "k_all" in data and "p1d_all" in data:
            k = data["k_all"]
            p = data["p1d_all"]
            valid = np.isfinite(p) & (p > 0) & (k > 0)
            if valid.any():
                axes[0].loglog(k[valid], k[valid] * p[valid] / np.pi,
                               color=color, alpha=0.4, lw=0.8)

        if "k_all" in data and "p1d_no_HCD" in data and "p1d_all" in data:
            k    = data["k_all"]
            p_no = data["p1d_no_HCD"]
            p_all = data["p1d_all"]
            valid = np.isfinite(p_no) & np.isfinite(p_all) & (p_all > 0) & (k > 0)
            if valid.any():
                ratio = p_no[valid] / p_all[valid]
                axes[1].semilogx(k[valid], ratio, color=color, alpha=0.4, lw=0.8)

    axes[0].set_xlabel("k [s/km]")
    axes[0].set_ylabel("k P₁D(k) / π")
    axes[0].set_title(f"P1D ({len(p1d_records)} snapshots)")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("k [s/km]")
    axes[1].set_ylabel("P₁D(no HCD) / P₁D(all)")
    axes[1].set_title("HCD suppression ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.05)

    # Colorbar by z
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(min(z_vals), max(z_vals)))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label="Redshift z", shrink=0.8)

    plt.suptitle(f"P1D intermediate ({len(p1d_records)} files)", fontsize=12)
    plt.tight_layout()
    out = out_dir / "p1d_intermediate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


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
    plot_b_parameter(records, out_dir)
    plot_param_coverage(records, out_dir)
    plot_absorber_counts(records, out_dir)

    # --- P1D (only if available) ---
    print("\n--- Loading P1D ---", flush=True)
    p1d_records = load_all_p1d(output_root)
    if p1d_records:
        plot_p1d_from_files(p1d_records, out_dir)
    else:
        print("No P1D files yet (jobs still processing).")

    print(f"\nAll figures saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
