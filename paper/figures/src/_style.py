"""The figure style of the paper, shared by its data figures.

The style of the figures of Ho et al. 2026:
text and mathematics set by LaTeX (Computer Modern), ticks inside on four sides, no grid, capped error bars,
frameless legends, a viridis-derived palette with one warm accent. Needs a LaTeX installation.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAL, TEAL_LIGHT = "#2a788e", "#a9c9d2"  # hcd_priya: 68% and 95%
WARM = "#e16462"  # the one warm accent (a reference model)
PURPLE, GREEN, BLUE = "#440154", "#35b779", "#31688e"
INK = "black"
BAND = "0.88"
FS_TICK, FS_LABEL, FS_LEGEND = 7.0, 9.0, 7.5

# Notation of the eleven parameters (Ho et al. 2026)
SYMBOL = {"ns": r"$n_\mathrm{P}$", "Ap": r"$A_\mathrm{P}/10^{-9}$", "herei": r"$z^\mathrm{HeII}_i$",
          "heref": r"$z^\mathrm{HeII}_f$", "alphaq": r"$\alpha_q$", "hub": r"$h$", "omegamh2": r"$\Omega_M h^2$",
          "hireionz": r"$z_\mathrm{HI}$", "bhfeedback": r"$\epsilon_\mathrm{AGN}$", "tau0_amp": r"$\tau_0$",
          "dtau0": r"$d\tau_0$"}


def style():
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 9, "axes.labelsize": FS_LABEL, "legend.fontsize": FS_LEGEND,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2, "errorbar.capsize": 1.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "axes.grid": False, "figure.facecolor": "white", "legend.frameon": False,
    })
