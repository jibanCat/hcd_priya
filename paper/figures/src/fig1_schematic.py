"""Figure 1 of the JOSS paper: schematic of hcd_priya, from simulations to the public predictor.

Design choices: head tags in plain words, the incidence output in words, no ensemble shadow, and one arrow
label for add_hcd.

Every label comes from the public documentation of hcd_priya (README.md, hcd_analysis/emulator/README.md). The figure holds no result and no number.

Run from the repository root:

    python3 paper/figures/src/fig1_schematic.py

It writes paper/figures/fig1_schematic.png (300 dpi) and paper/figures/fig1_schematic.pdf (vector).
The width is the JOSS text width (about 13.7 cm), so the figure is not rescaled in the paper.

Things the figure leaves to the caption and the text, on purpose:
  - the released emulator is an ensemble of such networks;
  - the per-class power is a baseline plus a residual in log power, on a low-rank basis;
  - the multi-fidelity correction is a separate record that the predictor applies;
  - "HCD" (high-column-density absorber) is defined in the caption;
  - the mixing equation behind add_hcd.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

W = 5.4  # inches: the JOSS text width. Data units are inches too.
XC = W / 2
MARGIN = 0.10
GAP = 0.25  # between two groups

INK = "#1a1a1a"
MUTED = "#55534e"
GREY_FILL = "#ecebe7"  # one neutral grey: the emulator panel
GREY_EDGE = "#8a8984"
DASH = (0, (4, 2))  # dashed outline: external code

FS_TITLE = 8.8
FS_BODY = 8.0
FS_SYM = 9.5

# heights of the four groups, top to bottom
H_EXT, H_DATA, H_EMU, H_OUT = 0.54, 0.36, 1.82, 1.10
H = 2 * MARGIN + H_EXT + H_DATA + H_EMU + H_OUT + 3 * GAP

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, title, body=None, lw=0.7, ls="solid", fc="white", ec=INK, fs_title=FS_TITLE):
    """A rounded box with a bold title and an optional second line. Returns its geometry."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=2))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs_title, fontweight="bold",
                color=INK, zorder=3, linespacing=1.2)
    else:
        lt, lb = fs_title * 1.25 / 72, FS_BODY * 1.3 / 72
        n_t, n_b = title.count("\n") + 1, body.count("\n") + 1
        top = y + h / 2 + (n_t * lt + 0.03 + n_b * lb) / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=fs_title, fontweight="bold", color=INK,
                zorder=3, linespacing=1.2)
        ax.text(x + w / 2, top - n_t * lt - 0.03, body, ha="center", va="top", fontsize=FS_BODY, color=INK,
                zorder=3, linespacing=1.3)
    return {"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2, "r": x + w, "t": y + h}


def arrow(p0, p1, lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, color=INK, zorder=4,
                                 shrinkA=0, shrinkB=0))


def route(points, lw=0.9):
    """A right-angle path through the points, with an arrow head on the last segment."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=INK, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=4)
    arrow(points[-2], points[-1], lw=lw)


def tag(x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FS_BODY, style="italic", color=MUTED, zorder=3)


def sym(x, y, text, fs=FS_SYM):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3)


# ---- 1. external codes (dashed outline) ---------------------------------------------------------------------
ytop = H - MARGIN
y1 = ytop - H_EXT
box(XC - 1.65, y1, 3.30, H_EXT, "PRIYA simulations and mock spectra", "MP-Gadget, fake_spectra", lw=0.9, ls=DASH)

# ---- 2. training data ---------------------------------------------------------------------------------------
y2 = y1 - GAP - H_DATA
arrow((XC, y1), (XC, y2 + H_DATA))
box(XC - 1.25, y2, 2.50, H_DATA, "Per-class training data")

# ---- 3. emulator: encoder and three heads -------------------------------------------------------------------
y3 = y2 - GAP - H_EMU
arrow((XC, y2), (XC, y3 + H_EMU))
tag(XC + 0.06, y2 - GAP / 2, "train")
ax.add_patch(FancyBboxPatch((MARGIN, y3), W - 2 * MARGIN, H_EMU, boxstyle="round,pad=0,rounding_size=0.07",
                            fc=GREY_FILL, ec=GREY_EDGE, lw=0.8, zorder=1))
ax.text(MARGIN + 0.09, y3 + H_EMU - 0.085, "Emulator", ha="left", va="top", fontsize=FS_TITLE,
        fontweight="bold", color=INK, zorder=3)

cy_i, cy_r, cy_b = y3 + 1.32, y3 + 0.81, y3 + 0.31  # centre lines of the three heads
xe, we = 1.02, 0.86
xh, wh, hh = 2.30, 1.56, 0.45
xo, wo = 4.10, 1.08
enc = box(xe, cy_r - 0.12, we, (cy_i + 0.14) - (cy_r - 0.12), "Encoder")
hI = box(xh, cy_i - hh / 2, wh, hh, "Incidence head", "no mean flux")
hR = box(xh, cy_r - hh / 2, wh, hh, "Residual head", "cosmology")
hB = box(xh, cy_b - hh / 2, wh, hh, "Baseline head", "no cosmology")

# inputs, as plain symbols
sym(0.50, enc["cy"], r"$\theta,\ z$")
arrow((0.74, enc["cy"]), (xe, enc["cy"]))
sym(0.50, cy_b + 0.09, r"$\tau_{\rm eff}$")  # the mean-flux input; tau_0 is PRIYA's amplitude
sym(0.50, cy_b - 0.09, r"$z$")
arrow((0.66, cy_b + 0.09), (xh, cy_b + 0.09))
arrow((0.66, cy_b - 0.09), (xh, cy_b - 0.09))
xl = 2.10  # the mean flux also feeds the residual head
route([(xl, cy_b + 0.09), (xl, cy_r - 0.10), (xh, cy_r - 0.10)])
ax.add_patch(Circle((xl, cy_b + 0.09), 0.022, fc=INK, ec=INK, zorder=5))
arrow((enc["r"], cy_i), (xh, cy_i))
sym((enc["r"] + xh) / 2, cy_i + 0.10, r"$\Theta$")  # the shared representation that the encoder hands to the heads
arrow((enc["r"], cy_r + 0.08), (xh, cy_r + 0.08))

# outputs
box(xo, cy_i - hh / 2, wo, hh, r"$dN/dX$, CDDF", fs_title=8.6)  # both outputs of the incidence head
arrow((hI["r"], cy_i), (xo, cy_i))
box(xo, hB["y"], wo, hR["t"] - hB["y"], r"$P_c(k)$", "clean, LLS,\nsubDLA, DLA", fs_title=FS_SYM)
arrow((hR["r"], cy_r), (xo, cy_r))
arrow((hB["r"], cy_b), (xo, cy_b))

# ---- 4. the public predictor and what it returns ------------------------------------------------------------
y4top = y3 - GAP
arrow((XC, y3), (XC, y4top))
tag(XC + 0.06, y3 - GAP / 2, "export, verified on load")
p = box(XC - 0.80, y4top - 0.38, 1.60, 0.38, "Predictor", lw=2.2)
tag(p["x"] - 0.10, p["cy"], "public interface", ha="right")

t = box(0.20, y4top - H_OUT, 2.55, 0.44, r"HCD template  $P_c\,/\,P_{\rm clean}$", fs_title=9.6, lw=1.1)
yg = p["y"] - 0.12
route([(p["cx"] - 0.35, p["y"]), (p["cx"] - 0.35, yg), (t["cx"], yg), (t["cx"], t["t"])])
xr = 4.05
route([(p["cx"] + 0.35, p["y"]), (p["cx"] + 0.35, yg), (xr, yg), (xr, t["t"])])
sym(xr, t["cy"] + 0.10, r"mixed $P_{\rm 1D}$", fs=9.2)
# the default class weights come from the predicted incidence; user-set amplitudes are optional
ax.text(xr, t["cy"] - 0.12, "weights $w_c$ from $dN/dX$ (default)\nor user-set amplitudes", ha="center", va="center", fontsize=FS_BODY,
        style="italic", color=MUTED, zorder=3)

assert abs((y4top - H_OUT) - MARGIN) < 1e-9, (y4top - H_OUT, MARGIN)
out_dir = Path(__file__).resolve().parent.parent
fig.savefig(out_dir / "fig1_schematic.png", dpi=300, facecolor="white")
# no creation date in the PDF, so that the file is the same on every run
fig.savefig(out_dir / "fig1_schematic.pdf", facecolor="white", metadata={"CreationDate": None})
print("wrote", out_dir / "fig1_schematic.png", f"({W * 2.54:.1f} cm x {H * 2.54:.1f} cm)")
