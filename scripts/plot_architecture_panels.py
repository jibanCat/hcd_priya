"""Phase-2b emulator architecture as 4 ZOOMED sub-panels (replaces the dense single diagram).

(1) Encoder  (2) Head A (τ₀-invariant abundances)  (3) Head B (θ-blind baseline + cosmology
residual → P_filt)  (4) Structural + the CORRECTED HCD likelihood. Pure matplotlib (no deps).

Env: /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_architecture_panels.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/analysis/04_emulator/emulator_architecture_panels.png"


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="0.3"))


def setup(ax, title):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")


BL, GR, OR, RD, PU = "#cfe2f3", "#d9ead3", "#fce5cd", "#f4cccc", "#e6d0f0"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # (1) Encoder
    a = ax[0, 0]; setup(a, "(1) Encoder — shared, encodes θ")
    box(a, 0.5, 7.5, 9, 1.4, "INPUT  x = [ θ_unit (9) , z_unit (1) ]   (10)", "#eeeeee")
    for i, (lab, yy) in enumerate([("Linear 10→256 + GELU", 5.6), ("Linear 256→128 + GELU", 3.9),
                                   ("Linear 128→64 + GELU", 2.2)]):
        box(a, 1.5, yy, 7, 1.1, lab, BL); arrow(a, 5, yy + 1.1 + (0.3 if i else 0.0), 5, yy + 1.1) if False else None
    arrow(a, 5, 7.5, 5, 6.7); arrow(a, 5, 5.6, 5, 5.0); arrow(a, 5, 3.9, 5, 3.3)
    box(a, 2.5, 0.4, 5, 1.1, "latent (64)", "#bbbbbb"); arrow(a, 5, 2.2, 5, 1.5)

    # (2) Head A
    a = ax[0, 1]; setup(a, "(2) Head A — τ₀-INVARIANT abundances")
    box(a, 3, 8.4, 4, 1.0, "latent (64)", "#bbbbbb")
    box(a, 3, 6.4, 4, 1.1, "trunk 64→64 (GELU)", GR); arrow(a, 5, 8.4, 5, 7.5)
    box(a, 0.6, 4.0, 4, 1.3, "CDDF f_NHI (30)\n[log target]", GR)
    box(a, 5.4, 4.0, 4, 1.3, "dN/dX (3)\nLLS/subDLA/DLA", GR)
    arrow(a, 4.3, 6.4, 2.6, 5.3); arrow(a, 5.7, 6.4, 7.4, 5.3)
    a.text(5, 2.6, "never sees τ₀ → abundances\nτ₀-invariant by construction\n→ sets the structural w_c",
           ha="center", va="center", fontsize=8, style="italic", color="0.3")

    # (3) Head B
    a = ax[1, 0]; setup(a, "(3) Head B — P_filt: θ-blind baseline + cosmology residual")
    box(a, 0.3, 8.4, 4.2, 1.0, "z, τ₀", "#bbbbbb")
    box(a, 5.5, 8.4, 4.2, 1.0, "latent (64), τ₀", "#bbbbbb")
    box(a, 0.3, 6.2, 4.2, 1.4, "BASELINE head m̂(z,τ₀)\nθ-BLIND, deep 3×256", OR)
    box(a, 5.5, 6.2, 4.2, 1.4, "RESIDUAL head r̂(θ,z,τ₀)\nSVD low-rank (n_basis=24)", OR)
    arrow(a, 2.4, 8.4, 2.4, 7.6); arrow(a, 7.6, 8.4, 7.6, 7.6)
    box(a, 1.5, 3.6, 7, 1.6,
        "logP̂ = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂\n→ exp → P_filt (4 classes)", "#ffffff")
    arrow(a, 2.4, 6.2, 4, 5.2); arrow(a, 7.6, 6.2, 6, 5.2)
    box(a, 2.5, 1.2, 5, 1.2, "P_filt (clean, LLS, subDLA, DLA) (4,K)", "#bbbbbb")
    arrow(a, 5, 3.6, 5, 2.4)
    a.text(5, 0.5, "∂logP̂/∂θ = σ_cosmo·∂r̂/∂θ  (baseline θ-blind ⇒ 0)", ha="center",
           fontsize=7.5, style="italic", color="0.3")

    # (4) Structural + corrected HCD likelihood
    a = ax[1, 1]; setup(a, "(4) Structural + HCD likelihood (CORRECTED)")
    box(a, 3, 8.6, 4, 0.9, "P_filt (4,K)", "#bbbbbb")
    box(a, 0.4, 6.6, 4.3, 1.2, "P_clean  (baseline)", RD)
    box(a, 5.0, 6.6, 4.6, 1.2, "excess  P_c − P_clean\n(filt LLS/subDLA, unfilt DLA)", RD)
    arrow(a, 4, 8.6, 2.5, 7.8); arrow(a, 6, 8.6, 7.3, 7.8)
    box(a, 0.8, 3.9, 8.4, 1.5,
        "P_obs = P_clean + Σ_c α_c · (P_c − P_clean)\n≡ P_clean·[1 + Σ α_c(P_c/P_clean − 1)]", "#ffffff")
    arrow(a, 2.5, 6.6, 4, 5.4); arrow(a, 7.3, 6.6, 6, 5.4)
    box(a, 0.6, 1.8, 4.0, 1.3, "α_c incidence prior\n(obs-centered, z-slope)", PU)
    box(a, 5.2, 1.8, 4.2, 1.3, "logL = −½rᵀC⁻¹r − ½logdet C\nr = P_data − P_obs", PU)
    arrow(a, 4.0, 3.9, 2.6, 3.1); arrow(a, 6, 3.9, 7.0, 3.1)
    a.text(2.6, 0.7, "← Head A dN/dX", ha="center", fontsize=7.5, color="0.3")
    arrow(a, 2.6, 1.8, 2.6, 1.1)

    fig.suptitle("Phase-2b emulator architecture (zoomed) — encoder → Head A / Head B → "
                 "structural + corrected HCD likelihood", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
