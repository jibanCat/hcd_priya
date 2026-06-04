#!/usr/bin/env python3
"""Draw an as-built architecture diagram of the Phase-2b HCD P1D + CDDF emulator.

REVISED 2026-06-02 — the NORMALIZATION REDESIGN (baseline + residual P_filt).

Dims/activations/heads transcribed directly from hcd_analysis/emulator/model.py
(Encoder, HeadA, BaselineHead, HeadB) and the combine in
hcd_analysis/emulator/data.reconstruct_P_filt:

  Encoder : Linear(10->256) gelu -> Linear(256->128) gelu -> Linear(128->64) [latent]
            encodes theta (the 9 params + z).
  Head A  : Linear(64->64) gelu -> {Linear(64->30)=f_nhi, Linear(64->3)=dN/dX}  (log)
            consumes ONLY the latent -> tau0-invariant by construction.   [unchanged]
  Head B for P1D is now TWO paths:
    BaselineHead  (theta-BLIND): input (z, tau0) ONLY -> m_hat(z,tau0), the
            (z,tau0)-conditional MEAN log-P1D (the dominant ~99.5%).
            Linear(2->64) gelu -> Linear(64->4*n_k)  [dense default].
    Residual head (the existing Head B): concat(latent, tau0) -> r_hat(theta,z,tau0),
            the COSMOLOGY residual (in sigma_cosmo units) + 3 HCD delta_c (arcsinh).
            Linear(65->256) gelu -> Linear(256->7*n_k) -> (4 P_filt resid + 3 delta).
    Combine : logP_filt = (m_hat*sig_marg + mu_marg) + sig_cosmo*r_hat  -> exp
            -> linear P_filt[4, n_k].   theta-response = sig_cosmo * d r_hat / d theta
            (the baseline is theta-blind: d m_hat / d theta = 0).
  Downstream (structural, not a network layer): P_tier_p = Sum_c w_c * P_c^filt,
            w_c derived from Head A dN/dX via telescoping-Poisson M0 + delta_c(z).
  Likelihood: P_obs = P_tier_p + Sum_c alpha_c * delta_c  (single per-class alpha_c).
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
C_HEADB = "#ffcc80"   # residual (theta-dependent) head — orange family
C_BASE  = "#ce93d8"   # theta-BLIND baseline head — purple family (distinct)
C_BASE_EC = "#6a1b9a"
C_OUT_A = "#66bb6a"
C_OUT_B = "#fb8c00"
C_COMB  = "#f06292"   # combine node — pink, the redesign's join
C_COMB_EC = "#ad1457"
C_STRUCT = "#b39ddb"
C_STRUCT_EC = "#4527a0"
C_LIK    = "#80cbc4"   # teal — likelihood / inference layer
C_ALPHA  = "#26a69a"   # teal accent — free per-class nuisance alpha_c
C_LIK_EC = "#00695c"   # dark teal edge for the likelihood block
C_BLOCK = {"enc": "#e3f2fd", "a": "#e8f5e9", "b": "#fff3e0", "base": "#f3e5f5"}
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


fig, ax = plt.subplots(figsize=(25, 11.5))
ax.set_xlim(0, 25)
ax.set_ylim(0, 11.5)
ax.axis("off")

# ======================= INPUTS =========================================
in_params = box(ax, 0.3, 7.5, 1.9, 1.0,
                "INPUT\nsim params (9) ⊕ z\n= 10", C_INPUT, fontsize=10,
                fontweight="bold")
in_ztau = box(ax, 0.3, 1.2, 1.9, 1.05,
              "INPUT\n$(z,\\ \\tau_0)$\n$\\theta$-blind path", C_INPUT,
              fontsize=9.5, fontweight="bold")

# ======================= ENCODER block ==================================
ex, ey, ew, eh = 2.9, 6.4, 4.3, 4.2
block_bg(ax, ex, ey, ew, eh, "Encoder  (shared, encodes $\\theta$)",
         C_BLOCK["enc"], "#1565c0")
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
ax_, ay_, aw_, ah_ = 8.4, 6.9, 4.5, 3.9
block_bg(ax, ax_, ay_, aw_, ah_, "Head A  (latent only)", C_BLOCK["a"], "#2e7d32")
a_trunk = box(ax, ax_ + 0.4, ay_ + 2.35, 3.7, 0.8, "Linear  64 → 64", C_HEADA)
a_cddf = box(ax, ax_ + 0.25, ay_ + 0.95, 1.85, 0.8, "Linear\n64 → 30", C_HEADA, fontsize=9)
a_dndx = box(ax, ax_ + 2.4, ay_ + 0.95, 1.85, 0.8, "Linear\n64 → 3", C_HEADA, fontsize=9)
ax.text(ax_ + aw_ / 2, ay_ + 0.35,
        "$\\tau_0$-invariant by construction",
        ha="center", va="center", fontsize=10, color="#2e7d32",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2e7d32", lw=1.2))

# ======================= BASELINE HEAD block (theta-BLIND) ==============
# input is ONLY (z, tau0); never sees the latent -> d m_hat / d theta = 0.
gx_, gy_, gw_, gh_ = 8.4, 3.95, 4.5, 2.5
block_bg(ax, gx_, gy_, gw_, gh_,
         "Baseline head  ($\\theta$-BLIND)", C_BLOCK["base"], C_BASE_EC)
g_in = box(ax, gx_ + 0.4, gy_ + 1.55, 3.7, 0.6,
           "input $(z, \\tau_0)$ = 2   — $\\theta$-blind", "#e1bee7", fontsize=9)
g_trunk = box(ax, gx_ + 0.4, gy_ + 0.78, 3.7, 0.62, "Linear  2 → 64", C_BASE,
              fontsize=9.5)
g_out = box(ax, gx_ + 0.4, gy_ + 0.05, 3.7, 0.62,
            f"Linear  64 → 4·n_k = {4*N_K}", C_BASE, fontsize=8.5)

# ======================= RESIDUAL HEAD block (tau0-dependent) ===========
bx_, by_, bw_, bh_ = 8.4, 0.7, 4.5, 3.05
block_bg(ax, bx_, by_, bw_, bh_,
         "Residual head  ($\\tau_0$-dep., cosmology)", C_BLOCK["b"], "#e65100")
b_concat = box(ax, bx_ + 0.4, by_ + 1.6, 3.7, 0.6,
               "concat(latent, $\\tau_0$) = 65", "#ffe0b2", fontsize=9)
b_trunk = box(ax, bx_ + 0.4, by_ + 0.83, 3.7, 0.62, "Linear  65 → 256", C_HEADB,
              fontsize=9.5)
b_out = box(ax, bx_ + 0.4, by_ + 0.08, 3.7, 0.62,
            f"Linear  256 → 7·n_k = {7*N_K}", C_HEADB, fontsize=8.5)

# latent fan-out: Head A + Residual head (NOT the baseline head — theta-blind)
arrow(ax, latent_pt, left(a_trunk), label="latent (64)", rad=-0.15, label_fs=9)
arrow(ax, latent_pt, left(b_concat), label="latent (64)", rad=0.20, label_fs=9)
# (z, tau0) into the theta-BLIND baseline head only
arrow(ax, right(in_ztau), left(g_in), color=C_BASE_EC, lw=2.0,
      label="$(z,\\tau_0)$", rad=0.10, label_fs=10)
# tau0 also into the residual head (concat with latent)
arrow(ax, (in_ztau[0] + in_ztau[2], in_ztau[1] + 0.18), left(b_concat),
      color="#c62828", lw=2.0, label="$\\tau_0$", rad=-0.18, label_fs=10)

arrow(ax, bottom(a_trunk), top(a_cddf), label="gelu", rad=0.25, label_fs=8.5)
arrow(ax, bottom(a_trunk), top(a_dndx), label="gelu", rad=-0.25, label_fs=8.5)
arrow(ax, bottom(g_in), top(g_trunk))
arrow(ax, bottom(g_trunk), top(g_out), label="gelu", label_dy=-0.02, label_fs=8.5)
arrow(ax, top(b_concat), bottom(b_trunk))
arrow(ax, bottom(b_trunk), top(b_out), label="gelu", label_dy=-0.02, label_fs=8.5)

# ======================= HEAD-A OUTPUTS =================================
o_fnhi = box(ax, 13.5, 8.65, 2.5, 0.95,
             "f_nhi  [30]\nCDDF shape (log)", C_OUT_A, fontsize=9.5,
             fontweight="bold")
o_dndx = box(ax, 13.5, 7.35, 2.5, 0.95,
             "dN/dX  [3]\nLLS,subDLA,DLA (log)", C_OUT_A, fontsize=9,
             fontweight="bold")
arrow(ax, right(a_cddf), left(o_fnhi), rad=0.05)
arrow(ax, right(a_dndx), left(o_dndx), rad=-0.05)

# ======================= BASELINE / RESIDUAL OUTPUTS ===================
o_mhat = box(ax, 13.5, 4.55, 2.55, 0.95,
             f"$\\hat m(z,\\tau_0)$  [4, {N_K}]\n$(z,\\tau_0)$-cond. mean\n"
             "($\\theta$-BLIND, $\\sim$99.5%)",
             C_BASE, fontsize=8.0, fontweight="bold", ec=C_BASE_EC)
b_reshape = box(ax, 13.5, 2.55, 2.55, 0.55,
                f"reshape → (7, {N_K})", "#ffe0b2", fontsize=8.5)
o_rhat = box(ax, 13.5, 1.55, 2.55, 0.85,
             f"$\\hat r(\\theta,z,\\tau_0)$  [4, {N_K}]\ncosmology residual\n"
             "($\\sigma_{\\mathrm{cosmo}}$ units)",
             C_OUT_B, fontsize=8.0, fontweight="bold")
o_delta = box(ax, 13.5, 0.30, 2.55, 0.95,
              f"$\\Delta_c$  [3, {N_K}]\nLLS,subDLA,DLA (arcsinh)", C_OUT_B,
              fontsize=8.2, fontweight="bold")

arrow(ax, right(g_out), left(o_mhat), rad=-0.05)
arrow(ax, right(b_out), left(b_reshape), rad=-0.05)
arrow(ax, (b_reshape[0] + 1.4, b_reshape[1]), top(o_rhat), rad=0.0,
      label="rows 0–3", label_fs=7.5, label_dy=-0.18)
arrow(ax, (b_reshape[0] + 0.55, b_reshape[1]), top(o_delta), rad=-0.25,
      label="rows 4–6", label_fs=7.5, label_dy=-0.30)

# ======================= COMBINE node (the redesign's join) =============
cx_, cy_, cw_, ch_ = 16.55, 2.7, 3.85, 2.2
block_bg(ax, cx_, cy_, cw_, ch_,
         "Combine  (normalization redesign)", "#fce4ec", C_COMB_EC)
c_box = box(ax, cx_ + 0.25, cy_ + 0.95, cw_ - 0.5, 0.85,
            "$\\log P_{\\mathrm{filt}} = (\\hat m\\cdot\\sigma_{\\mathrm{marg}}"
            "+\\mu_{\\mathrm{marg}}) + \\sigma_{\\mathrm{cosmo}}\\cdot\\hat r$",
            C_COMB, fontsize=9.0, ec=C_COMB_EC, fontweight="bold")
ax.text(cx_ + cw_ / 2, cy_ + 0.62,
        "$\\Rightarrow\\ \\exp\\ \\Rightarrow\\ P_{\\mathrm{filt}}$  "
        f"[4, {N_K}]  (linear)",
        ha="center", va="center", fontsize=9.5, color=C_COMB_EC,
        fontweight="bold")
ax.text(cx_ + cw_ / 2, cy_ + 0.27,
        "$\\theta$-response $= \\sigma_{\\mathrm{cosmo}}\\,"
        "\\partial\\hat r/\\partial\\theta$  (baseline $\\theta$-blind)",
        ha="center", va="center", fontsize=7.8, color=C_COMB_EC, style="italic")

# m_hat (baseline) and r_hat (residual) -> combine
arrow(ax, right(o_mhat), (cx_, cy_ + ch_ - 0.55), color=C_BASE_EC, lw=1.9,
      label="$\\hat m$", rad=-0.22, label_fs=9)
arrow(ax, right(o_rhat), (cx_, cy_ + 0.55), color=C_OUT_B, lw=1.9,
      label="$\\hat r$", rad=0.18, label_fs=9)

# P_filt output node (linear) from the combine
o_pfilt = box(ax, 16.85, 5.25, 3.25, 0.85,
              f"$P_{{\\mathrm{{filt}}}}$  [4, {N_K}]  (linear)\n"
              "clean,LLS,subDLA,DLA", C_COMB, fontsize=8.5, fontweight="bold",
              ec=C_COMB_EC)
arrow(ax, top(c_box), bottom(o_pfilt), color=C_COMB_EC, lw=1.9, rad=0.0)

# ======================= STRUCTURAL SUM (downstream) ====================
sx, sy, sw, sh = 16.85, 6.45, 3.55, 1.75
block_bg(ax, sx, sy, sw, sh, "Downstream (structural)", "#ede7f6", C_STRUCT_EC)
s_box = box(ax, sx + 0.25, sy + 0.18, sw - 0.5, 0.72,
            "$P_{\\mathrm{tier}\\,p} = \\sum_c w_c\\, P_c^{\\mathrm{filt}}$",
            C_STRUCT, fontsize=10.5)
ax.text(sx + sw / 2, sy + sh - 0.40,
        "$w_c$ from dN/dX ($M_0$ + $\\delta_c(z)$)",
        ha="center", va="center", fontsize=7.6, color=C_STRUCT_EC, style="italic")

o_tierp = box(ax, 17.15, 8.65, 3.0, 0.8,
              f"$P_{{\\mathrm{{tier}}\\,p}}$  [{N_K}]", C_STRUCT, fontsize=11,
              fontweight="bold", ec=C_STRUCT_EC)

# P_filt -> structural sum ; dN/dX -> w_c -> structural sum
arrow(ax, top(o_pfilt), bottom(s_box), color=C_STRUCT_EC, lw=1.7,
      label="$P_c^{filt}$", rad=0.0, label_fs=8.5)
arrow(ax, bottom(o_dndx), (sx + 0.4, sy + sh), color=C_STRUCT_EC, lw=1.6,
      label="dN/dX → $w_c$", rad=-0.42, label_fs=8)
arrow(ax, top(s_box), bottom(o_tierp), color=C_STRUCT_EC, lw=1.8, rad=0.18)

# ======================= LIKELIHOOD / INFERENCE block ===================
# Data-space model:  P_obs(k) = P_tier_p + Sum_{c in HCD} alpha_c * Delta_c
lx, ly, lw, lh = 20.55, 0.55, 4.2, 10.0
block_bg(ax, lx, ly, lw, lh, "Likelihood / inference", C_LIK, C_LIK_EC)

# --- the data-space model equation (banner at the top of the block) ---
l_eq = box(ax, lx + 0.25, ly + lh - 1.1, lw - 0.5, 0.78,
           "$P_{\\mathrm{obs}}(k) = P_{\\mathrm{tier}\\,p} + "
           "\\sum_{c\\in\\mathrm{HCD}} \\alpha_c\\,\\Delta_c$",
           "#b2dfdb", fontsize=11.5, fontweight="bold", ec=C_LIK_EC)

# --- forest baseline node (P_tier_p flows IN) ---
l_base = box(ax, lx + 0.3, ly + 5.95, 1.9, 0.95,
             "$P_{\\mathrm{tier}\\,p}$\nforest baseline", "#b2dfdb",
             fontsize=8.5, ec=C_LIK_EC)

# --- per-class HCD templates (Delta_c flows IN) ---
l_delta = box(ax, lx + 0.3, ly + 2.55, 1.9, 0.95,
              f"$\\Delta_c$  [3, {N_K}]\nper-class HCD", "#b2dfdb",
              fontsize=8.5, ec=C_LIK_EC)

# --- free per-class nuisance node alpha_c ---
l_alpha = box(ax, lx + 0.3, ly + 4.2, 1.9, 0.95,
              "$\\alpha_c$  [3]\nfree nuisance\nLLS, subDLA, DLA",
              C_ALPHA, fontsize=8.0, fontweight="bold", ec=C_LIK_EC)
ax.text(lx + lw / 2, ly + 3.95,
        "$\\alpha_c$ = effective residual\n(post-masking) incidence",
        ha="center", va="top", fontsize=7.2, color=C_LIK_EC, style="italic")

# --- the multiply / sum node ---
l_mul = box(ax, lx + 2.55, ly + 4.3, 0.95, 0.95,
            "$\\alpha_c\\,\\Delta_c$\n$\\oplus$",
            "#4db6ac", fontsize=9.5, fontweight="bold", ec=C_LIK_EC)

# --- final data-space total P1D node ---
l_pobs = box(ax, lx + 2.25, ly + 6.05, 1.7, 1.0,
             f"$P_{{\\mathrm{{obs}}}}(k)$  [{N_K}]\nTOTAL P1D\n→ data",
             "#26a69a", fontsize=8.5, fontweight="bold", ec=C_LIK_EC)

# inflows: P_tier_p baseline + Delta_c templates -> P_obs / multiply node
arrow(ax, right(o_tierp), left(l_base), color=C_LIK_EC, lw=1.7,
      label="baseline", rad=-0.12, label_fs=8)
arrow(ax, right(o_delta), (lx + 0.3, ly + 2.75), color=C_LIK_EC, lw=1.7,
      label="templates", rad=0.22, label_fs=8)
# Delta_c and alpha_c into the multiply node
arrow(ax, right(l_delta), bottom(l_mul), color=C_LIK_EC, lw=1.6, rad=-0.25)
arrow(ax, right(l_alpha), left(l_mul), color=C_ALPHA, lw=2.0, rad=0.0)
# baseline + (alpha_c Delta_c) -> P_obs total
arrow(ax, right(l_base), (l_pobs[0], l_pobs[1] + 0.25),
      color=C_LIK_EC, lw=1.7, rad=-0.18)
arrow(ax, top(l_mul), bottom(l_pobs), color=C_LIK_EC, lw=1.8, rad=0.16)

# --- PRIOR arrow: Head A dN/dX -> w_c -> alpha_c prior (dotted/one-sided) ---
arrow(ax, right(o_dndx), top(l_alpha), color="#00897b", lw=1.8, ls=":",
      rad=-0.30)
ax.text(lx + 0.45, ly + 5.45,
        "prior on $\\alpha_c$: center = sim $w_c$(dN/dX),\n"
        "wide / one-sided (PRIYA-style)",
        ha="left", va="center", fontsize=7.2, color="#00695c",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#00897b",
                  ls=":", lw=1.2))

# --- deliverable annotation ---
ax.text(lx + lw / 2, ly + 1.0,
        "$\\alpha_c$ posterior $\\Rightarrow$ rough\nper-class effective dN/dX\n"
        "(HCD deliverable)",
        ha="center", va="center", fontsize=8.2, color="#004d40",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e0f2f1", ec=C_LIK_EC,
                  lw=1.4))

# ======================= caption box: the redesign =====================
ax.text(2.95, 3.35,
        "Normalization redesign — the $\\theta$-blind baseline carries the "
        "$(z,\\tau_0)$ mean ($\\sim$99.5%);\n"
        "the residual head ($\\sigma_{\\mathrm{cosmo}}$ units) carries cosmology. "
        "Validated $\\theta$-tracking 0.80 → 0.96.",
        ha="left", va="top", fontsize=10.5, color="#ad1457", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff0f5", ec=C_COMB_EC, lw=1.6))

# ======================= legend / notes =================================
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_ENC, markersize=13, label="Encoder layer"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_HEADA, markersize=13, label="Head A layer ($\\tau_0$-free)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BASE, markersize=13, label="Baseline head ($\\theta$-BLIND)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_HEADB, markersize=13, label="Residual head ($\\tau_0$-dep., cosmology)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_COMB, markersize=13, label="Combine ($P_{\\mathrm{filt}}$ join)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_STRUCT, markersize=13, label="Structural (non-NN) op"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_ALPHA, markersize=13, label="Free nuisance $\\alpha_c$ (likelihood)"),
    Line2D([0], [0], color=C_BASE_EC, lw=2.0, label="$(z,\\tau_0)$ $\\theta$-blind path"),
    Line2D([0], [0], color="#c62828", lw=2.0, label="$\\tau_0$ data path (residual)"),
    Line2D([0], [0], color="#00897b", lw=1.8, ls=":", label="$\\alpha_c$ prior (Head A dN/dX $\\to w_c$)"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9.0,
          framealpha=0.95, ncol=1, bbox_to_anchor=(0.005, 0.005))

ax.text(0.3, 6.4,
        "Activations: gelu between hidden Linear layers;\n"
        "no activation on the latent or on any output head.\n"
        f"n_k = {N_K} (Lyman-$\\alpha$ forest, LF grid).",
        ha="left", va="top", fontsize=9, color="#37474f",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#b0bec5", lw=1.0))

fig.suptitle("Phase-2b emulator "
             "(as-built, 2026-06-02; baseline+residual $P_{\\mathrm{filt}}$)",
             fontsize=18, fontweight="bold", y=0.985)

out_dir = "/home/mfho/hcd_priya/figures/analysis/04_emulator"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "emulator_architecture.png")
fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", out_path)
print("size_bytes", os.path.getsize(out_path))
