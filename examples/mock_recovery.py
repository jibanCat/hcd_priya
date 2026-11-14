"""Mock recovery with the Gaussian P1D likelihood: an example in three parts.

1. **The likelihood at the truth.** The data set is a mock: a data vector made from a simulation at a known
   parameter point, with a covariance. Part one loads a model and the mock, and prints the chi-square and the log
   likelihood at the point the mock was made at, with the flags of that point.
2. **A chain** (only with ``--demo N`` or ``--full``; needs Cobaya, which is not a requirement of the package:
   ``pip install cobaya``). ``--demo N`` runs one short Metropolis chain of N accepted steps and labels it as a
   demonstration: it is NOT converged, and its numbers mean nothing beyond "the pieces work together". ``--full``
   runs one chain with the settings written out in ``FULL`` below until Cobaya's own convergence test stops it.
   Either chain starts at the truth of the mock. This script is a single process: it does not run under MPI.
3. **The figures.** After a chain: the posterior predictive (model vectors at points of the chain, with one noisy
   replicate of the data for each, drawn as a band over the data vector). With the example download: the stored
   reference, a converged posterior for the same mock, drawn underneath, and the intervals of the eleven parameters
   with the truth marked.

**The example download** (``--download DIR``, or the environment variable ``HCD_PRIYA_EXAMPLE_DOWNLOAD``) is one
directory with the mock, the model that goes with it and the stored reference; ``DOWNLOAD`` below names its items and
the sha256 each must have. IT IS NOT PUBLISHED YET. The model in it is a fold model: it was trained without the
simulation the mock was made from, so the recovery is a held-out one. It is not the release model. Without the
download the example runs on a model and a mock of your own (the second and third command lines below), and then
has no stored reference.

What this example fixes, all written out below: flat boxes on the eleven parameters (``BOXES``), two Gaussian terms
on ``hub`` and ``bhfeedback`` (``GAUSSIAN_TERMS``), and the sampler settings (``FULL``). The class spectra are mixed
with the model's own class fractions, so the HCD content follows the nine parameters and is not a free parameter.
The covariance is the data set's: no emulator error is added, so a posterior from this likelihood is narrower than
one that accounts for the error of the emulator.

Nothing is fitted to a measurement here: the data vector is a simulation's.

    python examples/mock_recovery.py --download DOWNLOAD_DIR --demo 200 --out out/
    python examples/mock_recovery.py --release RELEASE_DIR SHA256 --data MOCK_FILE SHA256
    python examples/mock_recovery.py --research BUNDLE_DIR SHA256 --purpose "a held-out mock recovery" \\
        --mf MF_DIR SHA256 --incidence INCIDENCE_DIR SHA256 --data MOCK_FILE SHA256 --demo 200 --out out/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
# widths of this project's own reference chains for such a mock (the chains are not shipped; the stored reference of
# the example download is a summary of one of them), written as ordinary Gaussians with their true standard deviations.
GAUSSIAN_TERMS = {"hub": (0.70, 0.015 / np.sqrt(2.0)), "bhfeedback": (0.05, 0.005 / np.sqrt(2.0))}
# The sampler of --full: Cobaya's Metropolis sampler, one chain, stopped by Cobaya's convergence test on that chain
# (R-1 of the means below 0.01, of the bounds below 0.05). The proposal starts as a fixed fraction of each box and is
# then learned from the chain. The project's reference chains ran four chains; this script runs one.
FULL = {"Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.05, "learn_proposal": True}
PROPOSAL_FRACTION = 0.02  # the starting proposal width, as a fraction of each box: efficiency only
BURN_IN_FRACTION = 0.3  # rows dropped from the start of a chain before anything is read from it

# The example download: one directory, NOT PUBLISHED YET. Each item with its place in the directory and the sha256 it
# must have; an item with another sha256 is refused by the loader that reads it.
# - mock: the data set (hcd_analysis.likelihood.read_p1d_data). A simulation's P1D with a covariance; no measurement.
# - model: a research bundle, a fold model trained WITHOUT the simulation the mock was made from. The sha256 is that
#   of its manifest.json, which names every other file of the bundle by its own sha256.
# - mf_record and incidence_record: the fold's MF record and its incidence record, built without that simulation too.
#   The sha256 is that of the record.
# - reference: the stored reference (hcd_analysis.likelihood.read_reference): a summary of a converged posterior for
#   this mock and its posterior predictive band. Its provenance says how it was made.
# The directory also holds a manifest (DOWNLOAD_MANIFEST) that states the format and the same five sha256; a
# directory whose manifest states anything else was made for another version of this script, and is refused.
DOWNLOAD_ENV = "HCD_PRIYA_EXAMPLE_DOWNLOAD"
DOWNLOAD_FORMAT = "hcd_priya.example_download/1"
DOWNLOAD_MANIFEST = "example_download.json"
DOWNLOAD = {
    "mock": ("mock.p1d.json", "6b5d17d091ad86337d00e863aee30297fb3cdcb692ad2540cbb32a4f0afe4a41"),
    "model": ("model", "07ae506b4105350081692d460c4b623e9083045a6ebe8130257ba31db5a7d758"),
    "mf_record": ("mf_record", "667548cfa5a745d5dfb8d7b669d6486860565efa5d78306c21231529ffbeb4c0"),
    "incidence_record": (
        "incidence_record",
        "33b34ff8df3dcefb25756642dee1cc0c98db944909c79492ec9b24140adbdfe5",
    ),
    "reference": (
        "reference.json",
        "a3c74b2cc09c524ebe760560aaac2394238ac747b2e0ab9361a1c9b2524c4792",
    ),
}
DOWNLOAD_PURPOSE = (
    "the mock-recovery example: a held-out recovery of the mock of the example download"
)


def line(label: str, *values) -> None:
    """Print one labelled line: every number this script prints says what it is."""
    print(label + ":", *values)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--download",
        metavar="DIR",
        help="the example download: the mock, its model and the stored reference (not published yet).\n"
        f"Without any model argument, the environment variable {DOWNLOAD_ENV} is read instead",
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
        "--data",
        nargs=2,
        metavar=("FILE", "SHA256"),
        help="with --release or --research: the mock and its sha256",
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
    if a.download is None and a.release is None and a.research is None:
        a.download = (
            os.environ.get(DOWNLOAD_ENV) or None
        )  # only when no model is named on the command line
    if sum(x is not None for x in (a.download, a.release, a.research)) != 1:
        ap.error(
            "give the model once: --download DIR, or --release DIR SHA256, or --research DIR SHA256"
        )
    if a.download is not None and (a.data or a.mf or a.incidence or a.purpose):
        ap.error(
            "--download brings its own mock and records: give no --data, --mf, --incidence, --purpose"
        )
    if a.download is None and a.data is None:
        ap.error("--release and --research need --data FILE SHA256")
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


def open_download(root: str):
    """The model, the mock and the stored reference of the example download at ``root``, each read against the
    sha256 written in ``DOWNLOAD``. The manifest is a first check only: what protects the read is that every loader
    below refuses an item whose sha256 is not the one written in this script.
    """
    want = {"format": DOWNLOAD_FORMAT, "items": {name: sha for name, (_, sha) in DOWNLOAD.items()}}
    try:
        with open(os.path.join(root, DOWNLOAD_MANIFEST), "rb") as f:
            stated = json.loads(f.read(1 << 16))
    except (OSError, ValueError) as exc:
        sys.exit(
            f"refusing: {root} is not the example download: its manifest {DOWNLOAD_MANIFEST} cannot be read ({exc})"
        )
    if stated != want:
        sys.exit(
            f"refusing: the manifest of {root} does not state the format {DOWNLOAD_FORMAT} with the five sha256 "
            "written in this script (DOWNLOAD); the download and the script are of different versions"
        )
    place = {name: os.path.join(root, rel) for name, (rel, _) in DOWNLOAD.items()}
    sha = {name: digest for name, (_, digest) in DOWNLOAD.items()}
    emu = hp.load_research_bundle(
        place["model"],
        sha["model"],
        purpose=DOWNLOAD_PURPOSE,
        mf_dir=place["mf_record"],
        mf_record_sha256=sha["mf_record"],
        incidence_dir=place["incidence_record"],
        incidence_record_sha256=sha["incidence_record"],
    )
    data = L.read_p1d_data(place["mock"], expected_sha256=sha["mock"])
    reference = L.read_reference(place["reference"], expected_sha256=sha["reference"])
    reference.check_bins(data)
    names = (*hp.PARAM_NAMES, *L.MEAN_FLUX_NAMES)
    truth = [float((data.truth or {}).get(name, np.nan)) for name in reference.param_names]
    if reference.param_names != names or not np.array_equal(reference.truth, truth):
        sys.exit(
            "refusing: the stored reference does not state the eleven parameters of the likelihood with the truth "
            "of the mock"
        )
    return emu, data, reference


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


def part_two(like, theta, a) -> dict | None:
    """The chain. Returns None when no chain was asked for, else {"points": points of the chain after burn-in drawn
    with its weights, shape (n, 11); "summary": the 16th, 50th and 84th percentile of each parameter, shape (11, 3)}.
    """
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
    summary = np.empty((len(like.param_names), 3))
    for i, name in enumerate(like.param_names):
        order = np.argsort(rows[:, i])
        cdf = np.cumsum(weights[order]) / weights.sum()
        lo, mid, hi = (rows[order, i][np.searchsorted(cdf, q)] for q in (0.16, 0.5, 0.84))
        summary[i] = lo, mid, hi
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
    return {"points": rows[pick], "summary": summary}


def part_three(like, chain, reference, a) -> None:
    """The figures: the posterior predictive of the data vector and, with a stored reference, the intervals of the
    parameters. The stored reference is drawn underneath in grey, the chain of this run over it in colour.
    """
    if chain is None and reference is None:
        print("=== Part 3: no chain, so no posterior predictive")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    data = like.data
    sigma = np.sqrt(np.diag(data.cov))
    x = np.arange(data.n)
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    print("=== Part 3: the posterior predictive of the data vector")
    if reference is not None:
        line(
            "the stored reference holds",
            f"the truth, the posterior mean and the central 68 and 95 per cent intervals of {len(reference.param_names)} "
            f"parameters, and the percentiles {list(reference.percentiles)} of the posterior predictive on "
            f"{reference.n} bins",
        )
        line("how the stored reference was made", reference.provenance.get("what", "not stated"))
        for name, mean, (lo68, hi68), truth in zip(
            reference.param_names, reference.mean, reference.interval68, reference.truth
        ):
            line(
                f"stored reference, {name}: mean, lower and upper end of the central 68 per cent interval | truth",
                mean,
                lo68,
                hi68,
                "|",
                truth,
            )
        for (lower, upper), shade in (((2.5, 97.5), 0.15), ((16.0, 84.0), 0.3)):
            lo, hi = reference.band(lower, upper)
            ax.fill_between(
                x,
                (lo - data.p1d) / sigma,
                (hi - data.p1d) / sigma,
                color="0.3",
                alpha=shade,
                lw=0,
                label=f"stored reference, {lower:g} to {upper:g} per cent",
            )
    if chain is None:
        line(
            "the chain of this run",
            "none (--demo N, or --full): the figures show the stored reference alone",
        )
    else:
        thetas = [dict(zip(like.param_names, row)) for row in chain["points"]]
        models, replicates = like.predictive(thetas, np.random.default_rng(a.seed + 1))
        lo, hi = np.percentile(replicates, [16, 84], axis=0)
        inside = float(np.mean((data.p1d >= lo) & (data.p1d <= hi)))
        line("points of the chain used", len(thetas))
        line(
            "fraction of the bins whose data value lies in the central 68 per cent of the replicates",
            inside,
        )
        ax.fill_between(
            x,
            (lo - data.p1d) / sigma,
            (hi - data.p1d) / sigma,
            color="C0",
            alpha=0.3,
            lw=0,
            label="this run: replicates, central 68 per cent"
            + (" (NOT CONVERGED)" if a.demo else ""),
        )
        ax.plot(
            x,
            (np.median(models, axis=0) - data.p1d) / sigma,
            color="C0",
            lw=1.2,
            label="this run: median model",
        )
    ax.axhline(0.0, color="k", lw=0.6)
    for sl in data.slices()[1:]:
        ax.axvline(sl.start - 0.5, color="k", lw=0.3, alpha=0.4)
    ax.set_xlabel("bin of the data vector (redshift by redshift)")
    ax.set_ylabel("(value - data) / sigma of the data")
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(a.out, "posterior_predictive.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    line("figure written", path)
    if reference is None:
        return
    # The intervals of the parameters, each on the scale of its flat box (0 and 1 are the edges of the box).
    names = reference.param_names
    lo_box = np.array([BOXES[n][0] for n in names])
    width = np.array([BOXES[n][1] - BOXES[n][0] for n in names])

    def unit(values):
        return (np.asarray(values, dtype=float).T - lo_box).T / width[:, None]

    y = np.arange(len(names))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.2), constrained_layout=True)
    r95, r68 = unit(reference.interval95), unit(reference.interval68)
    ax.hlines(y, r95[:, 0], r95[:, 1], color="0.3", lw=1.2, label="stored reference, 95 per cent")
    ax.hlines(y, r68[:, 0], r68[:, 1], color="0.3", lw=4.0, label="stored reference, 68 per cent")
    ax.plot(
        unit(reference.mean[:, None])[:, 0], y, "o", color="k", ms=4, label="stored reference, mean"
    )
    ax.plot(unit(reference.truth[:, None])[:, 0], y, "|", color="C3", ms=14, mew=1.6, label="truth")
    if chain is not None:
        c = unit(chain["summary"])
        ax.hlines(
            y - 0.3,
            c[:, 0],
            c[:, 2],
            color="C0",
            lw=2.5,
            label="this run, 16th to 84th percentile" + (" (NOT CONVERGED)" if a.demo else ""),
        )
        ax.plot(c[:, 1], y - 0.3, "o", color="C0", ms=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("position in the flat box of the parameter (0 and 1 are its edges)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    path = os.path.join(a.out, "posterior_intervals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    line("figure written", path)


def main(argv=None) -> int:
    a = parse(argv)
    if a.download is not None:
        emu, data, reference = open_download(a.download)
        line(
            "the model, the mock and the stored reference",
            f"the example download at {a.download}, each item read against the sha256 written in this script",
        )
        line(
            "the model of the example download",
            "a fold model, trained without the simulation of the mock; NOT the release model",
        )
    else:
        emu, reference = load_model(a), None
        data = L.read_p1d_data(a.data[0], expected_sha256=a.data[1])
    like = L.GaussianP1DLikelihood(emu, data, gaussian_terms=GAUSSIAN_TERMS)
    theta = part_one(like)
    chain = part_two(like, theta, a)
    part_three(like, chain, reference, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
