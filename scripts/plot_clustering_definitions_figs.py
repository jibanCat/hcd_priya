"""
Pedagogical figures for ``docs/clustering_definitions.md``.

Generates two diagrams in ``figures/diagnostics/clustering/``:

  fig_geometry.png
      Sightline + pair-decomposition diagram.  Shows how a (DLA,
      Lyα-pixel) pair is decomposed into (r_∥, r_⊥) under the
      "LOS = pixel's sightline axis" convention, plus the periodic
      minimum-image wrap.

  fig_delta_F_example.png
      Concrete example of the all-HCD-masked δ_F field on one
      sightline: τ → F → δ_F panels with the masked region
      highlighted.

Run::

    python3 scripts/plot_clustering_definitions_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "figures" / "diagnostics" / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 — geometry: sightlines, pair, r_∥ / r_⊥ decomposition
# ---------------------------------------------------------------------------

def fig_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Left panel: 2-D slice through the box, showing one sightline,
    #       a DLA on it, a Lyα pixel on a *different* sightline, and the
    #       (r_∥, r_⊥) decomposition.
    ax = axes[0]
    BOX = 120.0
    ax.add_patch(Rectangle((0, 0), BOX, BOX, fill=False, edgecolor="k", lw=1.5))

    # Several sightlines along the x-axis (LOS = x), drawn as horizontal lines
    sl_y = [25, 60, 95]
    for y in sl_y:
        ax.plot([0, BOX], [y, y], "-", color="C0", lw=0.8, alpha=0.8)

    # A DLA on the middle sightline
    dla_x, dla_y = 40, 60
    ax.scatter([dla_x], [dla_y], s=160, color="C3", marker="o",
               zorder=5, edgecolor="k", linewidth=1.0,
               label="DLA (catalog point)")

    # A Lyα pixel on the top sightline
    pix_x, pix_y = 80, 95
    ax.scatter([pix_x], [pix_y], s=120, color="C2", marker="s",
               zorder=5, edgecolor="k", linewidth=1.0,
               label=r"Ly$\alpha$ pixel ($\delta_F$ value)")

    # The pair-separation vector Δ = pix − DLA (drawn as an arrow)
    arrow = FancyArrowPatch(
        (dla_x, dla_y), (pix_x, pix_y),
        arrowstyle="->", mutation_scale=20,
        color="k", lw=2, zorder=4,
    )
    ax.add_patch(arrow)
    ax.text(0.5 * (dla_x + pix_x) + 2, 0.5 * (dla_y + pix_y) + 4,
            r"$\Delta = x_{\rm pix} - x_{\rm DLA}$", fontsize=11)

    # Decompose along the *pixel's* sightline axis (= x for this example).
    # r_∥ along x, r_⊥ orthogonal (y here, since the slice is x-y).
    ax.plot([dla_x, pix_x], [dla_y, dla_y], "--", color="C1", lw=1.5,
            label=r"$r_\parallel$ (along pixel's sightline)")
    ax.plot([pix_x, pix_x], [dla_y, pix_y], "--", color="C4", lw=1.5,
            label=r"$r_\perp$ (transverse)")

    ax.text(0.5 * (dla_x + pix_x) - 3, dla_y - 5,
            r"$r_\parallel$", fontsize=12, color="C1")
    ax.text(pix_x + 1.5, 0.5 * (dla_y + pix_y) - 1,
            r"$r_\perp$", fontsize=12, color="C4")

    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")
    ax.set_xlim(-5, BOX + 5)
    ax.set_ylim(-5, BOX + 5)
    ax.set_aspect("equal")
    ax.set_title(
        "Pair geometry: r_∥ along the *pixel's* sightline axis\n"
        "(blue lines = sightlines, all along x in this slice)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # ----- Right panel: periodic minimum-image wrap
    ax = axes[1]
    ax.add_patch(Rectangle((0, 0), BOX, BOX, fill=False, edgecolor="k", lw=1.5))
    # Replicate boxes (image cells) around the central one
    for dx in [-BOX, 0, BOX]:
        for dy in [-BOX, 0, BOX]:
            if dx == 0 and dy == 0:
                continue
            ax.add_patch(Rectangle(
                (dx, dy), BOX, BOX, fill=False,
                edgecolor="grey", lw=0.5, ls=":",
            ))

    # Two points such that the naive Δ wraps around the box
    p1 = (15, 60)
    p2 = (110, 60)
    ax.scatter(*p1, s=120, color="C0", zorder=5, edgecolor="k")
    ax.scatter(*p2, s=120, color="C0", zorder=5, edgecolor="k")

    # Naive Δ (long, the wrong answer)
    arrow_naive = FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=18,
        color="C3", lw=1.5, alpha=0.6, zorder=3,
    )
    ax.add_patch(arrow_naive)
    ax.text(50, 65, "naive Δ = 95 Mpc/h\n(wrong on a periodic box)",
            fontsize=9, color="C3", alpha=0.8)

    # Minimum-image Δ — points to the image of p2 in the cell to the left
    p2_image = (p2[0] - BOX, p2[1])
    ax.scatter(*p2_image, s=120, color="C0", alpha=0.4, zorder=5,
               edgecolor="k", marker="o")
    arrow_min = FancyArrowPatch(
        p1, p2_image, arrowstyle="->", mutation_scale=18,
        color="C2", lw=2, zorder=4,
    )
    ax.add_patch(arrow_min)
    ax.text(-35, 70, "min-image Δ = -25 Mpc/h\n(|Δ| ≤ box/2)", fontsize=9,
            color="C2")

    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")
    ax.set_xlim(-BOX - 5, BOX + 5)
    ax.set_ylim(-5, BOX + 5)
    ax.set_aspect("equal")
    ax.set_title(
        "Periodic minimum-image: Δ_min = Δ − box·round(Δ/box)\n"
        "(grey-dotted = image cells)", fontsize=11,
    )
    ax.grid(alpha=0.3)

    fig.suptitle("Clustering geometry — sightlines, pair separation, periodicity",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "fig_geometry.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — δ_F masking example on one synthetic sightline
# ---------------------------------------------------------------------------

def fig_delta_F_example():
    """Plant a τ profile with a few absorbers; show τ → F → δ_F with
    the all-HCD mask applied."""
    n_pix = 1250
    dx = 120.0 / n_pix                 # Mpc/h
    rng = np.random.default_rng(2026)

    # Background τ: smoothly varying, ~ Gaussian-shaped IGM forest
    pos = np.arange(n_pix) * dx
    bg = 0.4 + 0.15 * rng.standard_normal(n_pix)
    bg = np.maximum(bg, 0.0)

    # Plant three absorbers at known positions
    absorbers = [
        # (centre [Mpc/h], NHI [cm^-2], width [Mpc/h])
        (25.0, 1e18, 1.5),    # LLS
        (60.0, 5e19, 2.0),    # subDLA
        (95.0, 5e20, 3.0),    # DLA
    ]

    tau = bg.copy()
    pix_starts = []
    pix_ends = []
    classes = []
    for x_c, NHI, w in absorbers:
        # Give it a Gaussian τ profile, scaled so log(NHI) ~ corresponding
        sig = w
        amp_tau = max(np.log10(NHI) - 16, 0.0) * 5    # rough
        gauss = amp_tau * np.exp(-((pos - x_c) / sig) ** 2)
        tau = tau + gauss
        # Absorber pixel range = where tau exceeds a threshold
        in_abs = gauss > 0.5
        if in_abs.any():
            pix_start = int(np.argmax(in_abs))
            pix_end = int(n_pix - 1 - np.argmax(in_abs[::-1]))
            pix_starts.append(pix_start)
            pix_ends.append(pix_end)
            log_nhi = np.log10(NHI)
            if log_nhi < 19.0:
                cls = "LLS"
            elif log_nhi < 20.3:
                cls = "subDLA"
            else:
                cls = "DLA"
            classes.append(cls)

    F = np.exp(-tau)
    mask = np.zeros(n_pix, dtype=bool)
    for s, e in zip(pix_starts, pix_ends):
        mask[s:e + 1] = True
    mean_F = float(F[~mask].mean())
    delta_F = np.zeros_like(F)
    delta_F[~mask] = F[~mask] / mean_F - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    # τ
    ax = axes[0]
    ax.plot(pos, tau, "-", color="C3", lw=0.8)
    ax.set_ylabel(r"$\tau$ (per pixel)")
    ax.set_title(
        "δ_F field construction — one example sightline (synthetic)",
        fontsize=12,
    )
    for s, e, cls in zip(pix_starts, pix_ends, classes):
        ax.axvspan(s * dx, e * dx, color="grey", alpha=0.18)
        ax.text(0.5 * (s + e) * dx, ax.get_ylim()[1] * 0.92,
                cls, ha="center", fontsize=10, weight="bold")
    ax.grid(alpha=0.3)

    # F = exp(-τ)
    ax = axes[1]
    ax.plot(pos, F, "-", color="C0", lw=0.8, label=r"$F = e^{-\tau}$ (raw)")
    ax.plot(pos[~mask], F[~mask], ".", color="C0", ms=2,
            label="unmasked pixels (used in ⟨F⟩)")
    ax.axhline(mean_F, ls="--", color="k", lw=1,
               label=fr"$\langle F\rangle = {mean_F:.3f}$")
    for s, e, cls in zip(pix_starts, pix_ends, classes):
        ax.axvspan(s * dx, e * dx, color="grey", alpha=0.18)
    ax.set_ylabel(r"$F$ (transmitted flux)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    # δ_F = F / ⟨F⟩ - 1; masked pixels = 0
    ax = axes[2]
    ax.plot(pos, delta_F, "-", color="C2", lw=0.8,
            label=r"$\delta_F = F/\langle F\rangle - 1$ (unmasked)")
    ax.axhline(0, color="k", lw=0.5)
    for s, e, cls in zip(pix_starts, pix_ends, classes):
        ax.axvspan(s * dx, e * dx, color="grey", alpha=0.18,
                   label=("masked → δ_F = 0" if s == pix_starts[0] else None))
    ax.set_xlabel("LOS position [Mpc/h]")
    ax.set_ylabel(r"$\delta_F$")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "fig_delta_F_example.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Generating clustering-definitions pedagogical figures →", OUT_DIR)
    fig_geometry()
    fig_delta_F_example()
    print("Done.")


if __name__ == "__main__":
    main()
