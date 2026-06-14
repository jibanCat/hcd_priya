"""Pedagogical ECDF-SBC illustration through our REAL closure_diagnostics.ecdf_pit_bands.

A genuine conjugate-Gaussian SBC simulation (prior θ~N(0,τ²); data y~N(θ,σ²); posterior
θ|y ~ N(μ_post,σ_post²)). For each mock we draw L posterior samples and rank the truth.
With the CORRECT posterior the ranks are Uniform → ECDF hugs the diagonal → PASS. We then
deliberately miscalibrate the posterior used for sampling to show the three canonical
failure shapes the ECDF band catches:
  - bias  (sampler centred high)      -> ranks pile LOW  -> ECDF bows ABOVE the diagonal
  - over-confident (σ too small)      -> ranks pile at ENDS (∪) -> ECDF S-curve
  - under-confident (σ too large)     -> ranks pile in MIDDLE (∩) -> ECDF inverse-S

Top row  = the rank histogram (the intuitive view).
Bottom row = ECDF − diagonal with the SIMULTANEOUS 95% band (the actual pass/fail test).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_ecdf_sbc_illustration.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator import closure_diagnostics as D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/analysis/05_likelihood/ecdf_sbc_illustration.png"
TAU, SIG, L, N = 1.0, 1.0, 99, 600          # prior sd, lik sd, posterior draws, mocks


def sbc_ranks(bias_sd=0.0, disp=1.0, seed=0):
    """Genuine SBC ranks; `bias_sd` shifts the sampling posterior mean (in σ_post units),
    `disp` scales its sd (1=calibrated, <1 over-confident, >1 under-confident)."""
    r = np.random.default_rng(seed)
    sd_post = np.sqrt(TAU**2 * SIG**2 / (TAU**2 + SIG**2))
    ranks = np.empty(N, dtype=int)
    for m in range(N):
        th = r.normal(0.0, TAU)                      # truth from the prior
        y = r.normal(th, SIG)                        # data
        mu_post = y * TAU**2 / (TAU**2 + SIG**2)     # correct posterior mean
        samples = r.normal(mu_post + bias_sd * sd_post, sd_post * disp, L)
        ranks[m] = int(np.sum(samples < th))         # rank of truth among draws
    return ranks


def main():
    rng = np.random.default_rng(7)
    cases = [
        ("Calibrated", sbc_ranks(0.0, 1.0, 1), "ranks ~ Uniform"),
        ("Biased posterior", sbc_ranks(0.55, 1.0, 2), "sampler centred high → ranks pile LOW"),
        ("Over-confident", sbc_ranks(0.0, 0.72, 3), "σ too small → ranks pile at ENDS (∪)"),
        ("Under-confident", sbc_ranks(0.0, 1.45, 4), "σ too large → ranks pile in MIDDLE (∩)"),
    ]

    fig, ax = plt.subplots(2, 4, figsize=(18, 8.2))
    nbin = 20
    for j, (name, ranks, why) in enumerate(cases):
        lower, upper, ecdf, passed, grid = D.ecdf_pit_bands(
            ranks, n_draws=L, prob=0.95, n_grid=100, n_sim=4000, rng=rng)
        tag = "PASS ✓" if passed else "FAIL ✗"
        col = "tab:green" if passed else "tab:red"

        # --- top: rank histogram (intuitive view) ---
        a = ax[0, j]
        a.hist(ranks, bins=nbin, range=(0, L), color=col, alpha=0.55, edgecolor="k", lw=0.4)
        a.axhline(N / nbin, color="k", ls="--", lw=1, label="uniform expectation")
        a.set_title(f"{name}\n{why}", fontsize=10)
        a.set_xlabel("SBC rank of truth (0..L)")
        if j == 0:
            a.set_ylabel("# mocks per bin"); a.legend(fontsize=8)

        # --- bottom: ECDF − diagonal with the simultaneous band (the test) ---
        b = ax[1, j]
        b.fill_between(grid, lower - grid, upper - grid, color="0.8",
                       label="95% simultaneous band")
        b.plot(grid, ecdf - grid, color=col, lw=2.2, label="ECDF − uniform")
        b.axhline(0, color="k", lw=0.6)
        b.set_ylim(-0.13, 0.13)
        b.set_title(f"{tag}", color=col, fontsize=13, fontweight="bold")
        b.set_xlabel("PIT quantile  z")
        if j == 0:
            b.set_ylabel("ECDF(z) − z"); b.legend(fontsize=8, loc="lower center")
        b.grid(alpha=0.3)

    fig.suptitle(
        "ECDF simultaneous-band SBC test (Säilynoja+2022) — our closure_diagnostics.ecdf_pit_bands, "
        f"N={N} mocks, L={L} draws, 95% band\n"
        "PASS = the whole ECDF stays inside the band; the deviation SHAPE diagnoses the failure mode "
        "(above=bias, S=over-confident, inverse-S=under-confident).",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)
    for name, ranks, _ in cases:
        out = D.ecdf_pit_bands(ranks, n_draws=L, prob=0.95, n_grid=100, n_sim=4000,
                               rng=np.random.default_rng(7))
        print(f"  {name:18s} pass={out[3]}  mean-rank={ranks.mean():.1f} (uniform≈{L/2:.0f})")


if __name__ == "__main__":
    main()
