#!/usr/bin/env python3
"""Draw an as-built architecture diagram of the Phase-2b HCD P1D + CDDF emulator.

Dims/activations/heads transcribed directly from hcd_analysis/emulator/model.py:
  Encoder : Linear(10->256) gelu -> Linear(256->128) gelu -> Linear(128->64) [latent]
  Head A  : Linear(64->64) gelu -> {Linear(64->30)=f_nhi, Linear(64->3)=dN/dX}  (log space)
            consumes ONLY the latent -> tau0-invariant by construction
  Head B  : Linear(65->256) gelu -> Linear(256->7*n_k) -> reshape(7,n_k)
            = 4 P_filt (clean,LLS,subDLA,DLA; log) + 3 delta_c (LLS,subDLA,DLA; arcsinh)
            consumes concat(latent, tau0)
  Downstream (structural, not a network layer): P_tier_p = Sum_c w_c * P_c^filt,
            w_c derived from Head A dN/dX via telescoping-Poisson M0 + delta_c(z).
  n_k = 172 (LF).

No model import needed; this is pure matplotlib drawing.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

N_K = 172

# ---- palette ------------------------------------------------------------
C_INPUT = "#cfd8dc"
C_ENC   = "#90caf9"
C_HEADA = "#a5d6a7"
C_HEADB = "#ffcc80"
C_OUT_A = "#66bb6a"
C_OUT_B = "#fb8c00"
C_STRUCT = "#ce93d8"
C_BLOCK = {"enc": "#e3f2fd", "a": "#e8f5e9", "b": "#fff3e0"}
EDGE = "#37474f"


def box(ax, x, y, w, h, label, fc, fontsize=10, ec=EDGE, lw=1.4, fontweight="normal",
        style="round,pad=0.02,rounding_size=0.06"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, zorder=4, fontweight=fontweight)
    return (x, y, w, h)


def block_bg(ax, x, y, w, h, title, fc, ec):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                       fc=fc, ec=ec, lw=2.0, ls="--", zorder=1, alpha=0.55)
    ax.add_patch(p)
    ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top",
            fontsize=12.5, fontweight="bold", color=ec, zorder=2)


def arrow(ax, p0, p1, label=None, color=EDGE, lw=1.6, label_dy=0.12,
          label_fs=8.5, rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        color=color, lw=lw, zorder=5,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom",
                fontsize=label_fs, color=color, style="italic", zorder=6)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


fig, ax = plt.subplots(figsize=(18, 10.5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10.5)
ax.axis("off")

# ======================= INPUTS =========================================
in_params = box(ax, 0.3, 6.4, 1.9, 1.0,
                "INPUT\nsim params (9) ⊕ z\n= 10", C_INPUT, fontsize=10,
                fontweight="bold")
in_tau0 = box(ax, 0.3, 1.4, 1.9, 0.9,
              "INPUT\n$\\tau_0$ (1)", C_INPUT, fontsize=10.5, fontweight="bold")

# ======================= ENCODER block ==================================
ex, ey, ew, eh = 2.9, 5.4, 4.3, 4.2
block_bg(ax, ex, ey, ew, eh, "Encoder  (shared)", C_BLOCK["enc"], "#1565c0")
e1 = box(ax, ex + 0.4, ey + 2.85, 3.5, 0.85, "Linear  10 → 256", C_ENC)
e2 = box(ax, ex + 0.4, ey + 1.55, 3.5, 0.85, "Linear  256 → 128", C_ENC)
e3 = box(ax, ex + 0.4, ey + 0.25, 3.5, 0.85, "Linear  128 → 64", C_ENC)
ax.text(ex + ew / 2, ey + 0.05, "latent (64)   — no final activation",
        ha="center", va="bottom", fontsize=9, color="#1565c0",
        fontweight="bold")

arrow(ax, right(in_params), left(e1))
arrow(ax, bottom(e1), top(e2), label="gelu", label_dy=-0.02, label_fs=9)
arrow(ax, bottom(e2), top(e3), label="gelu", label_dy=-0.02, label_fs=9)

latent_pt = bottom(e3)

# ======================= HEAD A block (tau0-FREE) =======================
ax_, ay_, aw_, ah_ = 8.4, 5.9, 4.5, 3.9
block_bg(ax, ax_, ay_, aw_, ah_, "Head A  (latent only)", C_BLOCK["a"], "#2e7d32")
a_trunk = box(ax, ax_ + 0.4, ay_ + 2.35, 3.7, 0.8, "Linear  64 → 64", C_HEADA)
a_cddf = box(ax, ax_ + 0.25, ay_ + 0.95, 1.85, 0.8, "Linear\n64 → 30", C_HEADA, fontsize=9)
a_dndx = box(ax, ax_ + 2.4, ay_ + 0.95, 1.85, 0.8, "Linear\n64 → 3", C_HEADA, fontsize=9)
ax.text(ax_ + aw_ / 2, ay_ + 0.35,
        "$\\tau_0$-invariant by construction",
        ha="center", va="center", fontsize=10, color="#2e7d32",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2e7d32", lw=1.2))

# ======================= HEAD B block (tau0-dependent) ==================
bx_, by_, bw_, bh_ = 8.4, 1.0, 4.5, 3.4
block_bg(ax, bx_, by_, bw_, bh_, "Head B  ($\\tau_0$-dependent)", C_BLOCK["b"], "#e65100")
b_concat = box(ax, bx_ + 0.4, by_ + 2.35, 3.7, 0.7,
               "concat(latent, $\\tau_0$) = 65", "#ffe0b2", fontsize=9.5)
b_trunk = box(ax, bx_ + 0.4, by_ + 1.35, 3.7, 0.75, "Linear  65 → 256", C_HEADB)
b_out = box(ax, bx_ + 0.4, by_ + 0.35, 3.7, 0.75,
            f"Linear  256 → 7·n_k = {7*N_K}", C_HEADB, fontsize=9)

# latent fan-out to both heads
arrow(ax, latent_pt, left(a_trunk), label="latent (64)", rad=-0.15, label_fs=9)
arrow(ax, latent_pt, left(b_concat), label="latent (64)", rad=0.18, label_fs=9)
# tau0 into Head B only
arrow(ax, right(in_tau0), left(b_concat), color="#c62828", lw=2.0,
      label="$\\tau_0$", rad=-0.05, label_fs=11)

arrow(ax, bottom(a_trunk), top(a_cddf), label="gelu", rad=0.25, label_fs=8.5)
arrow(ax, bottom(a_trunk), top(a_dndx), label="gelu", rad=-0.25, label_fs=8.5)
arrow(ax, top(b_concat), bottom(b_trunk), label=None)
arrow(ax, bottom(b_trunk), top(b_out), label="gelu", label_dy=-0.02, label_fs=8.5)

# ======================= OUTPUTS ========================================
o_fnhi = box(ax, 13.3, 7.65, 2.5, 0.95,
             "f_nhi  [30]\nCDDF shape (log)", C_OUT_A, fontsize=9.5,
             fontweight="bold")
o_dndx = box(ax, 13.3, 6.35, 2.5, 0.95,
             "dN/dX  [3]\nLLS,subDLA,DLA (log)", C_OUT_A, fontsize=9,
             fontweight="bold")

# Head B reshape -> P_filt + delta
b_reshape = box(ax, 13.3, 2.55, 2.5, 0.7,
                f"reshape → (7, {N_K})", "#ffe0b2", fontsize=9)
o_pfilt = box(ax, 13.3, 1.45, 2.5, 0.95,
              f"P_filt  [4, {N_K}]\nclean,LLS,subDLA,DLA (log)", C_OUT_B,
              fontsize=8.5, fontweight="bold")
o_delta = box(ax, 13.3, 0.30, 2.5, 0.95,
              f"$\\Delta_c$  [3, {N_K}]\nLLS,subDLA,DLA (arcsinh)", C_OUT_B,
              fontsize=8.5, fontweight="bold")

arrow(ax, right(a_cddf), left(o_fnhi), rad=0.05)
arrow(ax, right(a_dndx), left(o_dndx), rad=-0.05)
arrow(ax, right(b_out), left(b_reshape), rad=-0.05)
arrow(ax, bottom(b_reshape), top(o_pfilt), rad=0.0)
arrow(ax, (b_reshape[0] + 0.4, b_reshape[1]), top(o_delta), rad=-0.2)

# ======================= STRUCTURAL SUM (downstream) ====================
sx, sy, sw, sh = 12.9, 3.95, 4.7, 1.85
block_bg(ax, sx, sy, sw, sh, "Downstream (structural)", "#f3e5f5", "#6a1b9a")
s_box = box(ax, sx + 0.25, sy + 0.18, sw - 0.5, 0.78,
            "$P_{\\mathrm{tier}\\,p} = \\sum_c w_c\\, P_c^{\\mathrm{filt}}$",
            C_STRUCT, fontsize=11)
ax.text(sx + sw / 2, sy + sh - 0.42,
        "$w_c$ from dN/dX  (telescoping-Poisson $M_0$ + $\\delta_c(z)$)",
        ha="center", va="center", fontsize=8, color="#6a1b9a", style="italic")

o_tierp = box(ax, 13.3, 9.05, 2.5, 0.8,
              f"$P_{{\\mathrm{{tier}}\\,p}}$  [{N_K}]", C_STRUCT, fontsize=11,
              fontweight="bold")

# dN/dX -> w_c -> structural sum ; P_filt -> structural sum
arrow(ax, right(o_dndx), (sx + sw / 2, sy), color="#6a1b9a", lw=1.6,
      label="dN/dX → $w_c$", rad=0.5, label_fs=8)
arrow(ax, right(o_pfilt), (sx + 0.3, sy + 0.3), color="#6a1b9a", lw=1.6,
      label="$P_c^{filt}$", rad=0.45, label_fs=8.5)
arrow(ax, top(s_box), bottom(o_tierp), color="#6a1b9a", lw=1.8, rad=0.2)

# ======================= legend / notes =================================
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_ENC, markersize=13, label="Encoder layer"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_HEADA, markersize=13, label="Head A layer ($\\tau_0$-free)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_HEADB, markersize=13, label="Head B layer ($\\tau_0$-dep.)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_STRUCT, markersize=13, label="Structural (non-NN) op"),
    Line2D([0], [0], color="#c62828", lw=2.0, label="$\\tau_0$ data path"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9.5,
          framealpha=0.95, ncol=1, bbox_to_anchor=(0.005, 0.005))

ax.text(0.3, 4.6,
        "Activations: gelu between hidden Linear layers;\n"
        "no activation on the latent or on any output head.\n"
        f"n_k = {N_K} (Lyman-$\\alpha$ forest, LF grid).",
        ha="left", va="top", fontsize=9, color="#37474f",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#b0bec5", lw=1.0))

fig.suptitle("Phase-2b HCD P1D + CDDF emulator (as-built)",
             fontsize=18, fontweight="bold", y=0.985)

out_dir = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "emulator_architecture.png")
fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", out_path)
print("size_bytes", os.path.getsize(out_path))
