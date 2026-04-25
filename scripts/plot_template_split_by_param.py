"""
Median-split parameter sensitivity for the per-class HCD templates
P_dirty / P_clean = P_class_only(k, z) / P_clean(k, z).

For each of the 9 PRIYA input parameters, split the 60 LF sims at the
median of that parameter into two halves; compute the mean and 1σ
spread of the template across each half.  One figure per species
(LLS / subDLA / DLA), rows = z bins, columns = parameters.

Output:
  figures/analysis/02_param_sensitivity/template_split_by_param_{LLS,subDLA,DLA}.png

Run:
    python3 scripts/plot_template_split_by_param.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import data_dir

SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT = ROOT / "figures" / "analysis" / "02_param_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["LLS", "subDLA", "DLA"]
CLASS_COLORS = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}

# PRIYA angular k range
K_ANG_MIN = 0.0009
K_ANG_MAX = 0.20

# Common log-k grid that every sim is resampled onto.  Different PRIYA
# sims have slightly different k_cyc arrays (dv_kms varies with H(z)),
# so a common grid is required before median-split averaging.
N_KCOMMON = 60
K_COMMON = np.logspace(np.log10(K_ANG_MIN), np.log10(K_ANG_MAX), N_KCOMMON)

# z-bin centres to show (plot rows)
Z_BINS = [2.4, 3.4, 4.4]
Z_TOL = 0.12

PARAM_KEYS = ["ns", "Ap", "herei", "heref", "alphaq",
              "hub", "omegamh2", "hireionz", "bhfeedback"]
PARAM_LABELS = {
    "ns": r"$n_s$",
    "Ap": r"$A_p$",
    "herei": r"HeII $z_i$",
    "heref": r"HeII $z_f$",
    "alphaq": r"$\alpha_q$",
    "hub": r"$h$",
    "omegamh2": r"$\Omega_m h^2$",
    "hireionz": r"HI reion $z$",
    "bhfeedback": r"BH feedback",
}

_SIM_PARAM = {
    "ns":         r"ns([0-9.]+)",
    "Ap":         r"Ap([0-9.e+-]+)",
    "herei":      r"herei([0-9.]+)",
    "heref":      r"heref([0-9.]+)",
    "alphaq":     r"alphaq([0-9.]+)",
    "hub":        r"hub([0-9.]+)",
    "omegamh2":   r"omegamh2([0-9.]+)",
    "hireionz":   r"hireionz([0-9.]+)",
    "bhfeedback": r"bhfeedback([0-9.]+)",
}


def parse_params(sim: str) -> dict:
    out = {}
    for k, pat in _SIM_PARAM.items():
        m = re.search(pat, sim)
        if m:
            out[k] = float(m.group(1))
    return out


def _interp_to_common(k_sim: np.ndarray, y_sim: np.ndarray) -> np.ndarray:
    """Linear interpolation of y(log k) onto K_COMMON.  Bins outside the
    sim's k range are NaN so they don't bias later averages."""
    out = np.full(N_KCOMMON, np.nan, dtype=np.float64)
    if k_sim.size < 2:
        return out
    lo, hi = k_sim[0], k_sim[-1]
    in_range = (K_COMMON >= lo) & (K_COMMON <= hi)
    if in_range.any():
        out[in_range] = np.interp(
            np.log10(K_COMMON[in_range]),
            np.log10(k_sim),
            y_sim,
        )
    return out


def _load_templates() -> dict:
    """
    Return nested dict:  templates[sim][z_rounded][cls] = ratio_on_K_COMMON
    (shape (N_KCOMMON,), NaN outside the sim's k range).  Key "_dirty"
    is also stored with the sightline-weighted total P_dirty / P_clean
    so the same dict can drive both per-class and total P_dirty figures.
    Scans all LF sims × all z-snaps that have a p1d_per_class.h5.
    """
    out = defaultdict(lambda: defaultdict(dict))
    n_loaded = 0
    for sim_dir in sorted(SCRATCH.iterdir()):
        if not sim_dir.is_dir() or not sim_dir.name.startswith("ns"):
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            if not snap_dir.name.startswith("snap_"):
                continue
            pc = snap_dir / "p1d_per_class.h5"
            meta = snap_dir / "meta.json"
            if not (pc.exists() and meta.exists() and (snap_dir / "done").exists()):
                continue
            try:
                z = float(json.load(open(meta))["z"])
            except Exception:
                continue
            try:
                with h5py.File(pc, "r") as f:
                    k_cyc = f["k"][:]
                    P_clean = f["P_clean"][:]
                    n_clean = int(f["n_sightlines_clean"][()])
                    per_cls = {}
                    for cls in CLASSES:
                        per_cls[cls] = (f[f"P_{cls}_only"][:],
                                         int(f[f"n_sightlines_{cls}"][()]))
            except Exception:
                continue

            k_ang = 2.0 * np.pi * k_cyc
            sel_win = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX)
            if not sel_win.any():
                continue

            zkey = round(z, 2)
            # Per-class ratios
            for cls in CLASSES:
                P_cls, n = per_cls[cls]
                if n == 0:
                    continue
                ok = sel_win & (P_clean > 0) & (P_cls > 0)
                if ok.sum() < 10:
                    continue
                r = P_cls[ok] / P_clean[ok]
                out[sim_dir.name][zkey][cls] = _interp_to_common(k_ang[ok], r)

            # Total P_dirty / P_clean (sightline-weighted reconstruction)
            n_tot = n_clean + sum(n for _, n in per_cls.values())
            if n_tot > 0:
                P_dirty = (n_clean / n_tot) * P_clean
                for _, (p, n) in per_cls.items():
                    P_dirty = P_dirty + (n / n_tot) * p
                ok = sel_win & (P_clean > 0) & (P_dirty > 0)
                if ok.sum() >= 10:
                    r_dirty = P_dirty[ok] / P_clean[ok]
                    out[sim_dir.name][zkey]["_dirty"] = _interp_to_common(
                        k_ang[ok], r_dirty
                    )
            n_loaded += 1
    print(f"  loaded {n_loaded} per-class files across "
          f"{len(out)} sims")
    return dict(out)


def _find_z_sims(templates: dict, z_target: float, tol: float, cls: str):
    """Return (sims_list, k_ref, ratios_mat) for the sims that have data
    at this z and class; ratios are already on K_COMMON so we just
    stack.  ratios_mat shape = (n_sim, N_KCOMMON)."""
    chosen = []
    for sim, per_z in templates.items():
        best_z = None
        for zkey in per_z:
            if abs(zkey - z_target) <= tol:
                if best_z is None or abs(zkey - z_target) < abs(best_z - z_target):
                    best_z = zkey
        if best_z is None:
            continue
        if cls not in per_z[best_z]:
            continue
        chosen.append((sim, per_z[best_z][cls]))
    if not chosen:
        return [], None, None
    sims = [c[0] for c in chosen]
    ratios = np.stack([c[1] for c in chosen])
    return sims, K_COMMON, ratios


def _param_medians(sims: list[str]) -> dict[str, float]:
    """Median of each param across the sim list (sim-level, not record-level)."""
    vals = defaultdict(list)
    for sim in sims:
        p = parse_params(sim)
        for k in PARAM_KEYS:
            if k in p:
                vals[k].append(p[k])
    return {k: float(np.median(vs)) for k, vs in vals.items()}


def _halves_mask(sims: list[str], pk: str, median: float):
    """Boolean masks (high, low) over the sims list for parameter pk."""
    vals = np.array([parse_params(s).get(pk, np.nan) for s in sims])
    return vals >= median, vals < median


def _plot_split_figure(templates: dict, cls: str, color: str,
                        ylabel: str, title_prefix: str, outpath: Path):
    """Generic half-split figure.  `cls` is the key inside templates
    (CLASSES entries or '_dirty' for total P_dirty/P_clean)."""
    n_z = len(Z_BINS)
    n_p = len(PARAM_KEYS)
    fig, axes = plt.subplots(n_z, n_p, figsize=(3.1 * n_p, 3.0 * n_z),
                             sharex=True, sharey="row")

    for i, z_target in enumerate(Z_BINS):
        sims, k_ref, ratios = _find_z_sims(templates, z_target, Z_TOL, cls)
        if not sims:
            for j in range(n_p):
                axes[i, j].set_title(
                    f"{PARAM_LABELS[PARAM_KEYS[j]]}  z≈{z_target:.1f}: no data"
                )
            continue

        medians = _param_medians(sims)
        for j, pk in enumerate(PARAM_KEYS):
            ax = axes[i, j]
            hi_mask, lo_mask = _halves_mask(sims, pk, medians[pk])
            if hi_mask.sum() == 0 or lo_mask.sum() == 0:
                continue
            m_hi = np.nanmean(ratios[hi_mask], axis=0)
            s_hi = np.nanstd(ratios[hi_mask], axis=0)
            m_lo = np.nanmean(ratios[lo_mask], axis=0)
            s_lo = np.nanstd(ratios[lo_mask], axis=0)

            ax.plot(k_ref, m_hi, "-",  color=color, lw=1.6,
                    label=f"high (n={hi_mask.sum()})")
            ax.fill_between(k_ref, m_hi - s_hi, m_hi + s_hi,
                            color=color, alpha=0.18)
            ax.plot(k_ref, m_lo, "--", color=color, lw=1.6,
                    label=f"low  (n={lo_mask.sum()})")
            ax.fill_between(k_ref, m_lo - s_lo, m_lo + s_lo,
                            color=color, alpha=0.08, hatch="//",
                            edgecolor=color, linewidth=0)
            ax.axhline(1.0, color="gray", lw=0.4, ls=":")
            ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(
                    f"{PARAM_LABELS[pk]}\nmed={medians[pk]:.3g}",
                    fontsize=9,
                )
            if j == 0:
                ax.set_ylabel(f"{ylabel}\nz ≈ {z_target:.1f}", fontsize=9)
            if i == n_z - 1:
                ax.set_xlabel(r"$k$ [rad·s/km]", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle(
        f"{title_prefix}\n"
        f"solid = high-θ half, dashed = low-θ half;  shaded = 1σ across sims",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    print("Loading per-class templates across LF sims…")
    templates = _load_templates()

    # Per-class P_class_only / P_clean
    for cls in CLASSES:
        outpath = OUT / f"template_split_by_param_{cls}.png"
        print(f"Plotting median-split template for {cls}…")
        _plot_split_figure(
            templates, cls=cls, color=CLASS_COLORS[cls],
            ylabel=f"P_{cls} / P_clean",
            title_prefix=(
                f"Median-split parameter sensitivity of the {cls}-class "
                f"HCD template (P_{cls} / P_clean)"
            ),
            outpath=outpath,
        )
        print(f"  wrote {outpath}")

    # Total P_dirty / P_clean (sightline-weighted reconstruction)
    outpath = OUT / "template_split_by_param_dirty.png"
    print("Plotting median-split P_dirty / P_clean…")
    _plot_split_figure(
        templates, cls="_dirty", color="C0",
        ylabel="P_dirty / P_clean",
        title_prefix=(
            "Median-split parameter sensitivity of the total HCD-dirty "
            "template (P_dirty / P_clean), sightline-weighted across classes"
        ),
        outpath=outpath,
    )
    print(f"  wrote {outpath}")


if __name__ == "__main__":
    main()
