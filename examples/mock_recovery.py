"""Mock recovery with the Gaussian P1D likelihood: an example in three parts.

1. **The likelihood at the truth.** The data set is a mock: a data vector made from a simulation at a known
   parameter point, with a covariance. Part one loads a model and the mock, and prints the chi-square and the log
   likelihood at the point the mock was made at, with the flags of that point.
2. **A chain** (only with ``--demo N`` or ``--full``; needs Cobaya, which is not a requirement of the package:
   ``pip install cobaya``). ``--demo N`` runs one short Metropolis chain of N accepted steps and labels it as a
   demonstration: it is NOT converged, and its numbers mean nothing beyond "the pieces work together". ``--full``
   runs one chain with the settings written out in ``FULL`` below until Cobaya's own convergence test stops it.
   Either chain starts at the truth of the mock. This script is a single process: it does not run under MPI.
3. **The posterior predictive** (only after a chain): model vectors at points of the chain, with one noisy replicate
   of the data for each, drawn as a band over the data vector.

What this example fixes, all written out below: flat boxes on the eleven parameters (``BOXES``), two Gaussian terms
on ``hub`` and ``bhfeedback`` (``GAUSSIAN_TERMS``), and the sampler settings (``FULL``). The class spectra are mixed
with the model's own class fractions, so the HCD content follows the nine parameters and is not a free parameter.
The covariance is the data set's: no emulator error is added, so a posterior from this likelihood is narrower than
one that accounts for the error of the emulator.

Nothing is fitted to a measurement here: the data vector is a simulation's.

    python examples/mock_recovery.py --release RELEASE_DIR SHA256 --data MOCK_FILE SHA256
    python examples/mock_recovery.py --research BUNDLE_DIR SHA256 --purpose "a held-out mock recovery" \\
        --mf MF_DIR SHA256 --incidence INCIDENCE_DIR SHA256 --data MOCK_FILE SHA256 --demo 200 --out out/
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import logging
import os
import sys

import numpy as np

import hcd_analysis.predictor as hp
from hcd_analysis import likelihood as L

# The flat boxes of the eleven parameters, written out. A point outside a box has zero prior. They are the parameter
# limits of Table 2 of Fernandez, Bird and Ho (2024, arXiv:2309.03943; read in the arXiv version), which that paper
# uses as uniform priors:
# - herei and heref: narrower than the box the model was trained on (3.5 to 4.5 and 2.2 to 3.2), and equal to the
#   package's own sampling box (emulator.data.SAMPLING_LIMITS); the model is sparsely trained in the widened corners.
# - alphaq: up to 3.0, the edge of the trained box. That is beyond the package's own sampling box, which stops at 2.5
#   because the training simulations are sparse above it.
# - ns: up to 1.05. Above 0.995 the training simulations are sparse too; the predictor flags such a point (ns_ridge).
# - tau0_amp and dtau0: PRIYA's two mean-flux parameters (hp.tau_eff_priya); the paper's tau_0 and d tau_0.
BOXES = {
    "ns": (0.8, 1.05),
    "Ap": (1.2e-9, 2.6e-9),
    "herei": (3.5, 4.1),
    "heref": (2.6, 3.2),
    "alphaq": (1.3, 3.0),
    "hub": (0.65, 0.75),
    "omegamh2": (0.14, 0.146),
    "hireionz": (6.5, 8.0),
    "bhfeedback": (0.03, 0.07),
    "tau0_amp": (0.75, 1.25),
    "dtau0": (-0.4, 0.25),
}
# Gaussian terms, (mean, standard deviation): this P1D alone does not constrain these two parameters. Fernandez, Bird
# and Ho (2024) quote Gaussian priors of width 0.015 on h and 0.005 on the AGN feedback parameter. This example uses
# standard deviations of 0.015 / sqrt(2) and 0.005 / sqrt(2), narrower by sqrt(2) than the quoted widths. They are the
# widths of this project's own reference chains for such a mock (not shipped here), written as ordinary Gaussians with
# their true standard deviations.
GAUSSIAN_TERMS = {"hub": (0.70, 0.015 / np.sqrt(2.0)), "bhfeedback": (0.05, 0.005 / np.sqrt(2.0))}
# The sampler of --full: Cobaya's Metropolis sampler, one chain, stopped by Cobaya's convergence test on that chain
# (R-1 of the means below 0.01, of the bounds below 0.05). The proposal starts as a fixed fraction of each box and is
# then learned from the chain. The project's reference chains ran four chains; this script runs one.
FULL = {"Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.05, "learn_proposal": True}
PROPOSAL_FRACTION = 0.02  # the starting proposal width, as a fraction of each box: efficiency only
BURN_IN_FRACTION = 0.3  # rows dropped from the start of a chain before anything is read from it


def line(label: str, *values) -> None:
    """Print one labelled line: every number this script prints says what it is."""
    print(label + ":", *values)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--release", nargs=2, metavar=("DIR", "SHA256"), help="a release directory and its sha256"
    )
    ap.add_argument(
        "--research",
        nargs=2,
        metavar=("DIR", "SHA256"),
        help="a research bundle and its manifest sha256",
    )
    ap.add_argument(
        "--purpose", help="with --research: what the research model is used for (recorded on it)"
    )
    ap.add_argument(
        "--mf", nargs=2, metavar=("DIR", "SHA256"), help="with --research: its fold MF record"
    )
    ap.add_argument(
        "--incidence",
        nargs=2,
        metavar=("DIR", "SHA256"),
        help="with --research: its incidence record",
    )
    ap.add_argument(
        "--data", nargs=2, metavar=("FILE", "SHA256"), required=True, help="the mock and its sha256"
    )
    ap.add_argument(
        "--demo", type=int, default=0, metavar="N", help="a demonstration chain of N accepted steps"
    )
    ap.add_argument(
        "--full", action="store_true", help="the chain of the settings FULL, until converged"
    )
    ap.add_argument(
        "--seed", type=int, default=1, help="the seed of the chain and of the predictive draws"
    )
    ap.add_argument(
        "--draws", type=int, default=200, help="points of the chain used for the predictive"
    )
    ap.add_argument(
        "--out", default="mock_recovery_out", help="the folder for the chain and the figure"
    )
    a = ap.parse_args(argv)
    if (a.release is None) == (a.research is None):
        ap.error("give the model once: --release DIR SHA256, or --research DIR SHA256")
    if a.research is None and (a.mf or a.incidence or a.purpose):
        ap.error("--purpose, --mf and --incidence belong to --research")
    if a.research is not None and not a.purpose:
        ap.error("--research needs --purpose")
    if a.demo < 0 or (a.demo and a.full):
        ap.error("give --demo N (N > 0) or --full, not both")
    if a.draws < 1:
        ap.error("--draws must be at least 1")
    return a


def load_model(a: argparse.Namespace):
    """The model the arguments name: a release, or a research bundle with its records."""
    if a.release is not None:
        return hp.load_release(*a.release)
    kw = {}
    if a.mf:
        kw.update(mf_dir=a.mf[0], mf_record_sha256=a.mf[1])
    if a.incidence:
        kw.update(incidence_dir=a.incidence[0], incidence_record_sha256=a.incidence[1])
    return hp.load_research_bundle(a.research[0], a.research[1], purpose=a.purpose, **kw)


def part_one(like: L.GaussianP1DLikelihood) -> dict:
    """The likelihood at the point the mock was made at. Returns that point."""
    data = like.data
    if data.truth is None:
        sys.exit(
            "refusing: this data set states no truth; the example needs a mock made at a known point"
        )
    missing = [name for name in like.param_names if name not in data.truth]
    if missing:
        sys.exit(
            f"refusing: the truth of this data set lacks {missing}; the example needs all eleven parameters"
        )
    theta = {name: float(data.truth[name]) for name in like.param_names}
    outside = [n for n in like.param_names if not BOXES[n][0] <= theta[n] <= BOXES[n][1]]
    print("=== Part 1: the likelihood at the point the mock was made at")
    line("bins in the data vector", data.n)
    line("redshifts", list(data.z))
    line("the truth, by name", theta)
    line("parameters of the truth outside this example's flat boxes", outside)
    pred = like.prediction(theta)
    line(
        "flags set at the truth (read them before you use the numbers)",
        sorted(k for k, v in pred.flags.items() if np.any(v)),
    )
    line(
        "the model's class fractions (LLS, subDLA, DLA) at the first redshift, used for the mixture",
        pred.w_c[0].tolist(),
    )
    chi2 = like.chi2(theta)
    line("chi-square at the truth (0 would mean the model reproduces the mock exactly)", chi2)
    line("chi-square per bin at the truth", chi2 / data.n)
    line("log likelihood at the truth, with its normalisation", like.loglike(theta))
    line("log density of each Gaussian term at the truth", like.log_gaussian_terms(theta))
    return theta


def cobaya_info(
    like: L.GaussianP1DLikelihood, theta: dict, *, demo: int, seed: int, out: str
) -> dict:
    """The input of ``cobaya.run`` for this likelihood: flat boxes, the truth as the starting point, and either the
    demonstration settings (``demo`` accepted steps, no convergence test that it could pass) or ``FULL``.
    """

    def logp(ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback, tau0_amp, dtau0):
        point = dict(
            zip(
                like.param_names,
                (
                    ns,
                    Ap,
                    herei,
                    heref,
                    alphaq,
                    hub,
                    omegamh2,
                    hireionz,
                    bhfeedback,
                    tau0_amp,
                    dtau0,
                ),
            )
        )
        return like.sampler_logp(point)

    params = {
        name: {
            "prior": {"min": lo, "max": hi},
            "ref": theta[name],
            "proposal": PROPOSAL_FRACTION * (hi - lo),
        }
        for name, (lo, hi) in BOXES.items()
    }
    mcmc = dict(FULL, seed=seed)
    if demo:
        mcmc.update(
            max_samples=int(demo), Rminus1_stop=0.0, Rminus1_cl_stop=0.0
        )  # stops at max_samples only
    return {
        "likelihood": {"p1d": {"external": logp}},
        "params": params,
        "sampler": {"mcmc": mcmc},
        "output": os.path.join(out, "chain"),
        "force": True,
    }


@contextlib.contextmanager
def quiet_predictor_log():
    """Switch off the INFO record that the predictor writes for each prediction that applies a correction, and put
    the logger's level back afterwards. A sampler lets INFO records pass and calls the likelihood at every step, so
    the record would be written once per step. The flags of a prediction say the same without any logging.
    """
    logger = logging.getLogger("hcd_analysis.predictor")
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


def part_two(like, theta, a) -> np.ndarray | None:
    """The chain. Returns its points after burn-in, shape (n, 11), or None when no chain was asked for."""
    if not (a.demo or a.full):
        print("=== Part 2: no chain was asked for (--demo N, or --full)")
        return None
    if importlib.util.find_spec("cobaya") is None:
        print(
            "=== Part 2: skipped. Cobaya is not installed (pip install cobaya); it is not a requirement of the package"
        )
        return None
    outside = [n for n in like.param_names if not BOXES[n][0] <= theta[n] <= BOXES[n][1]]
    if outside:
        sys.exit(
            f"refusing: the chain starts at the truth, and the truth lies outside the flat box of {outside}"
        )
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))) > 1:
        sys.exit("refusing: this example is a single process; run it without mpirun")
    from cobaya.run import run

    os.makedirs(a.out, exist_ok=True)
    print(
        "=== Part 2: "
        + (
            "A DEMONSTRATION CHAIN, NOT CONVERGED: its numbers are not results"
            if a.demo
            else "the full chain"
        )
    )
    line("the chain starts at", "the truth of the mock")
    line("an earlier chain with the same prefix is overwritten", os.path.join(a.out, "chain"))
    line(
        "the library's log record for each corrected prediction",
        "switched off while the sampler and the predictive draws run (it would be one line per likelihood call)",
    )
    with quiet_predictor_log():
        _, sampler = run(cobaya_info(like, theta, demo=a.demo, seed=a.seed, out=a.out))
    sample = sampler.products()["sample"]
    rows = np.column_stack([np.asarray(sample[name]) for name in like.param_names])
    weights = np.asarray(sample["weight"])
    keep = np.arange(len(rows)) >= int(BURN_IN_FRACTION * len(rows))
    line("accepted steps in the chain", len(rows))
    line(f"rows kept after dropping the first {BURN_IN_FRACTION:.0%} as burn-in", int(keep.sum()))
    line("points the model refused, by kind (count and first message)", like.refusals)
    line("the chain is written in Cobaya's format with the prefix", os.path.join(a.out, "chain"))
    rows, weights = rows[keep], weights[keep]
    for i, name in enumerate(like.param_names):
        order = np.argsort(rows[:, i])
        cdf = np.cumsum(weights[order]) / weights.sum()
        lo, mid, hi = (rows[order, i][np.searchsorted(cdf, q)] for q in (0.16, 0.5, 0.84))
        line(
            f"{name}: 16th, 50th, 84th percentile of the chain | truth",
            lo,
            mid,
            hi,
            "|",
            theta[name],
        )
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(
        len(rows), size=min(a.draws, len(rows)), replace=True, p=weights / weights.sum()
    )
    return rows[pick]


def part_three(like, points, a) -> None:
    """The posterior predictive of the data vector, as a figure."""
    if points is None:
        print("=== Part 3: no chain, so no posterior predictive")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== Part 3: the posterior predictive of the data vector")
    thetas = [dict(zip(like.param_names, row)) for row in points]
    with quiet_predictor_log():
        models, replicates = like.predictive(thetas, np.random.default_rng(a.seed + 1))
    data = like.data
    sigma = np.sqrt(np.diag(data.cov))
    lo, hi = np.percentile(replicates, [16, 84], axis=0)
    inside = float(np.mean((data.p1d >= lo) & (data.p1d <= hi)))
    line("points of the chain used", len(thetas))
    line(
        "fraction of the bins whose data value lies in the central 68 per cent of the replicates",
        inside,
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    x = np.arange(data.n)
    ax.fill_between(
        x,
        (lo - data.p1d) / sigma,
        (hi - data.p1d) / sigma,
        alpha=0.3,
        lw=0,
        label="replicates, central 68 per cent",
    )
    ax.plot(x, (np.median(models, axis=0) - data.p1d) / sigma, lw=1.2, label="median model")
    ax.axhline(0.0, color="k", lw=0.6)
    for sl in data.slices()[1:]:
        ax.axvline(sl.start - 0.5, color="k", lw=0.3, alpha=0.4)
    ax.set_xlabel("bin of the data vector (redshift by redshift)")
    ax.set_ylabel("(value - data) / sigma of the data")
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(a.out, "posterior_predictive.png")
    fig.savefig(path, dpi=150)
    line("figure written", path)


def main(argv=None) -> int:
    a = parse(argv)
    emu = load_model(a)
    data = L.read_p1d_data(a.data[0], expected_sha256=a.data[1])
    like = L.GaussianP1DLikelihood(emu, data, gaussian_terms=GAUSSIAN_TERMS)
    theta = part_one(like)
    points = part_two(like, theta, a)
    part_three(like, points, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
