#!/usr/bin/env python3
"""Phase-5a Test B figure: genuine HF-LOSO closure (real measured HR truth + MF correction
fit EXCLUDING the held-out HR sim) vs the gate-invariant Test A control (M_hr) and the
forward-only MF-LOSO coherent residual.

Reads the committed checkpoints/stepA/*.npz (HFLOSO* + M_hr*) and the forward residual summary
figures/analysis/04_emulator/mf_rescorr_loso.txt. Writes to the NOTES repo figure tree.

Sign convention here: bias = (posterior_mean - truth)/sigma  [positive => posterior OVERestimates].
(The run health log prints (truth - post)/sigma; this script recomputes from the draws.)
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/mfho/hcd_priya"
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"
CKPT = f"{REPO}/checkpoints/stepA"
os.makedirs(NOTES_FIG, exist_ok=True)

# HR-sim n_s labels (unit-cube truth -> physical handled below via the saved truth_vec is unit).
SIMS = {  # mock_id -> physical n_s (for the x-axis), from the HR cache
    "HFLOSO859": 0.859, "HFLOSO885": 0.885, "HFLOSO909": 0.909,
    "HFLOSO972": 0.972, "HFLOSO979": 0.979,
}
# Test-A (gate-invariant control) battery biases, in (post-truth)/sigma (flip of the health-log sign).
TESTA = {0.909: {"ns": -2.325, "Ap": -1.976},
         0.972: {"ns": +0.422, "Ap": +0.076},
         0.979: {"ns": -0.873, "Ap": +0.068}}
# Forward-only per-sim coherent low-z high-k MF-LOSO residual (figures/.../mf_rescorr_loso.txt).
FWD_COH = {0.859: +0.05, 0.885: -0.34, 0.909: +0.26, 0.972: -0.70, 0.979: -0.12}  # percent in P
FWD_RMS = {0.859: 1.18, 0.885: 1.49, 0.909: 1.39, 0.972: 1.54, 0.979: 1.52}       # percent RMS


def load_bias(mock_id):
    fs = sorted(glob.glob(f"{CKPT}/{mock_id}_c*.npz"))
    Z = [np.load(f, allow_pickle=True) for f in fs]
    names = [str(x) for x in Z[0]["names"]]
    truth = Z[0]["truth_vec"]
    draws = np.concatenate([z["packed"] for z in Z], axis=0)
    out = {}
    for p in ("ns", "Ap"):
        i = names.index(p)
        m = draws[:, i].mean(); s = draws[:, i].std()
        out[p] = (m - truth[i]) / s
    return out


def main():
    ns_x = np.array(sorted(SIMS.values()))
    rev = {v: k for k, v in SIMS.items()}
    B = {ns: load_bias(rev[ns]) for ns in ns_x}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- panel 1: n_s closure bias ---
    ax = axes[0]
    ax.axhspan(-1, 1, color="0.85", zorder=0, label="|z|<1 gate")
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(ns_x, [B[n]["ns"] for n in ns_x], "o-", color="C0", ms=8, lw=2,
            label="Test B (genuine HF-LOSO)")
    ta_x = sorted(TESTA); ax.plot(ta_x, [TESTA[n]["ns"] for n in ta_x], "s--", color="C3",
                                  ms=8, lw=1.5, label="Test A (gate-invariant control)")
    for n in ns_x:
        ax.annotate(f"{B[n]['ns']:+.1f}σ", (n, B[n]["ns"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("HR-sim $n_s$"); ax.set_ylabel(r"$(\hat n_s-n_s^{\rm true})/\sigma$")
    ax.set_title("(a) $n_s$ closure — HF-LOSO vs control"); ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-3.4, 3.4)

    # --- panel 2: A_p closure bias ---
    ax = axes[1]
    ax.axhspan(-1, 1, color="0.85", zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(ns_x, [B[n]["Ap"] for n in ns_x], "o-", color="C0", ms=8, lw=2, label="Test B")
    ax.plot(ta_x, [TESTA[n]["Ap"] for n in ta_x], "s--", color="C3", ms=8, lw=1.5, label="Test A")
    for n in ns_x:
        ax.annotate(f"{B[n]['Ap']:+.1f}σ", (n, B[n]["Ap"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("HR-sim $n_s$"); ax.set_ylabel(r"$(\hat A_p-A_p^{\rm true})/\sigma$")
    ax.set_title("(b) $A_p$ closure — robust (ex. LF-LOSO 909)"); ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-3.4, 3.4)

    # --- panel 3: inference n_s bias vs forward coherent eps (sign-consistency) ---
    ax = axes[2]
    coh = np.array([FWD_COH[n] for n in ns_x])
    rms = np.array([FWD_RMS[n] for n in ns_x])
    nsb = np.array([B[n]["ns"] for n in ns_x])
    ax.errorbar(coh, nsb, xerr=rms, fmt="o", color="C2", ms=8, capsize=3,
                label="per-sim (xerr = fwd RMS)")
    for n in ns_x:
        ax.annotate(f"{n:.3f}", (FWD_COH[n], B[n]["ns"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="0.3")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("forward MF-LOSO coherent eps  [% in P, low-z high-k]")
    ax.set_ylabel(r"inference $(\hat n_s-n_s^{\rm true})/\sigma$")
    ax.set_title("(c) sign-consistent: eps$-$ ⇒ $n_s$ high")
    ax.legend(fontsize=8)

    fig.suptitle("Phase-5a Test B — genuine HF-LOSO closure (real measured HR truth, MF correction "
                 "fit excluding it; DESI leg, NO MF floor on DESI)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{NOTES_FIG}/hfloso_closure_bias.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
