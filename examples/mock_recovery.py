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
   reference, a converged posterior for the same mock, drawn underneath, and two figures drawn from the stored
   reference and the mock alone (``draw_reference``): the intervals of the eleven parameters with the truth marked,
   and the mock per redshift with the stored bands of the posterior predictive.

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

The figures are saved to files and never shown, under whatever matplotlib backend your session has; in a session
without a display set MPLBACKEND=Agg in the environment.

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
    "mock": ("mock.p1d.json", "b7db66e81a20c25de571f39fad1a362d7c844a855487a46b31208354ce5efc34"),
    "model": ("model", "f8a830af3faaffc4f47d90f2fc596d4d8cc9760fc3057127da22cea0771638e2"),
    "mf_record": ("mf_record", "e639c4b187eface66e5e99bad1efe5645e409fe916f353cd0ef2f88d97bc2fa5"),
    "incidence_record": (
        "incidence_record",
        "094522ca1a44f153f6661df5e7eed8211ce836c6864f779021271c109e2eb02f",
    ),
    "reference": (
        "reference.json",
        "35a0ad02dfa502234c79775ba4b37f82a8ec2efd2c162b4fe7b35b0e968cc8ac",
    ),
}
DOWNLOAD_PURPOSE = (
    "the mock-recovery example: a held-out recovery of the mock of the example download"
)
REFERENCE_BANDS = (
    (2.5, 97.5),
    (16.0, 84.0),
)  # the percentile bands of the stored reference that part three draws


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
        "--out",
        default="mock_recovery_out",
        help="the folder for the chain and the figures (files only; without a display, set MPLBACKEND=Agg)",
    )
    a = ap.parse_args(argv)
    if a.download is not None and not a.download:
        ap.error("--download needs a directory")
    named_by = "--download"
    if a.download is None and a.release is None and a.research is None:
        # only when no model is named on the command line; an empty variable is no variable
        a.download = os.environ.get(DOWNLOAD_ENV) or None
        named_by = f"the environment variable {DOWNLOAD_ENV}"
    if sum(x is not None for x in (a.download, a.release, a.research)) != 1:
        ap.error(
            "give the model once: --download DIR, or --release DIR SHA256, or --research DIR SHA256"
        )
    if a.download is not None and (a.data or a.mf or a.incidence or a.purpose):
        ap.error(
            f"{named_by} names the example download, which brings its own mock and records: give no --data, "
            "--mf, --incidence, --purpose"
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
    except (OSError, ValueError, RecursionError) as exc:
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
    try:  # the mock and the reference first: they are quick to check, the model is not
        data = L.read_p1d_data(place["mock"], expected_sha256=sha["mock"])
        reference = L.read_reference(place["reference"], expected_sha256=sha["reference"])
        reference.check_bins(data)
        for lower, upper in REFERENCE_BANDS:  # the bands part three draws must be stored levels
            reference.band(lower, upper)
        names = (*hp.PARAM_NAMES, *L.MEAN_FLUX_NAMES)
        truth = [float((data.truth or {}).get(name, np.nan)) for name in reference.param_names]
        if reference.param_names != names or not np.array_equal(reference.truth, truth):
            raise L.LikelihoodInputError(
                "the stored reference does not state the eleven parameters of the likelihood, in its order, with "
                "the truth of the mock"
            )
        emu = hp.load_research_bundle(
            place["model"],
            sha["model"],
            purpose=DOWNLOAD_PURPOSE,
            mf_dir=place["mf_record"],
            mf_record_sha256=sha["mf_record"],
            incidence_dir=place["incidence_record"],
            incidence_record_sha256=sha["incidence_record"],
        )
    except (
        ValueError,
        OSError,
    ) as exc:  # every refusal of the loaders is a ValueError; a missing file an OSError
        sys.exit(
            f"refusing: the example download at {root} is not the one this script names: {exc}"
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
    try:  # a report only: it never stops the example (a redshift with a single bin has no bin spacing)
        coarseness = f"{min(float(row.min()) for row in like.modes_per_bin(theta)):.2f}"
    except L.LikelihoodInputError as exc:
        coarseness = f"not defined for this data set ({exc})"
    line(
        "model mode spacings per bin spacing, the smallest over all bins and redshifts "
        "(likelihood.modes_per_bin; a report of the bin layout, not a test of the interpolation)",
        coarseness,
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
    """The figures: the posterior predictive of the data vector and, with a stored reference, the two figures of
    ``draw_reference`` (the intervals of the parameters; the mock and the stored bands per redshift). The stored
    reference is drawn in grey, the chain of this run over it in colour.
    """
    if chain is None and reference is None:
        print("=== Part 3: no chain, so no posterior predictive")
        return
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
        for (lower, upper), shade in zip(REFERENCE_BANDS, (0.15, 0.3)):
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
            "none (no chain was run): the figures show the stored reference alone",
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
    if reference is not None:
        draw_reference(
            reference,
            data,
            a.out,
            chain_summary=None if chain is None else chain["summary"],
            demo=bool(a.demo),
        )


def reference_panels(reference, data) -> list:
    """What each panel of the figure per redshift draws, as numbers: for each redshift of the mock its wavenumbers,
    the mock there, and the stored bands of the posterior predictive there (one pair of rows per entry of
    ``REFERENCE_BANDS``), each multiplied by k / pi. Nothing is interpolated: the bins are the mock's and the
    reference's, which ``reference.check_bins`` requires to be the same."""
    reference.check_bins(data)
    bands = {pair: reference.band(*pair) for pair in REFERENCE_BANDS}
    panels = []
    for z, k, sl in zip(data.z, data.k, data.slices()):
        panel = {"z": z, "k": k, "mock": k * data.p1d[sl] / np.pi}
        for pair, (lo, hi) in bands.items():
            panel[pair] = (k * lo[sl] / np.pi, k * hi[sl] / np.pi)
        panels.append(panel)
    return panels


def _download_item(name: str) -> str:
    place, sha = DOWNLOAD[name]
    return f"{place} of the example download, sha256 {sha}"


def _write_figure(fig, path: str, facts: dict) -> None:
    """Write the image and, beside it, a side file (.json) that says in words what it shows and from what."""
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150)
    plt.close(fig)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1, sort_keys=True)
        fh.write("\n")
    line("figure written", path)


def draw_reference(reference, data, out: str, *, chain_summary=None, demo: bool = False) -> list:
    """The two figures of the stored reference. They are drawn from the stored reference and the mock alone: no
    model, no likelihood. Anyone with the example download draws the same two figures.

    1. ``posterior_intervals.png``: for each of the eleven parameters, on the scale of its flat box (``BOXES``), the
       reference's mean and its central 68 and 95 per cent intervals, with the truth. A chain of this run, if given
       (``chain_summary``: the 16th, 50th and 84th percentile of each parameter), is drawn over it in colour.
    2. ``posterior_predictive_per_redshift.png``: one panel per redshift, with the mock and the stored bands of the
       posterior predictive (``REFERENCE_BANDS``), as k P(k) / pi against k.

    Each image has a side file with the same name and the ending .json. It says in words what the image shows and
    which items of the download it was drawn from; it holds no value of the reference or of the mock. Returns the
    paths of the two images.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    panels = reference_panels(
        reference, data
    )  # checks first that the reference and the mock share their bins
    said = {
        "reference": _download_item("reference"),
        "made_by": "examples/mock_recovery.py, part three (draw_reference); it needs the example download",
        "covariance": "the covariance of the mock alone: no emulator error is added",
    }
    not_converged = " (NOT CONVERGED)" if demo else ""

    # 1. The intervals of the parameters, each on the scale of its flat box (0 and 1 are the edges of the box).
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
    if chain_summary is not None:
        c = unit(chain_summary)
        ax.hlines(
            y - 0.3,
            c[:, 0],
            c[:, 2],
            color="C0",
            lw=2.5,
            label="this run, 16th to 84th percentile" + not_converged,
        )
        ax.plot(c[:, 1], y - 0.3, "o", color="C0", ms=3, label="this run, median" + not_converged)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("position in the flat box of the parameter (0 and 1 are its edges)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    if chain_summary is None:
        chain_said = "none: the stored reference alone"
    else:
        chain_said = "a chain of this run, drawn over the stored reference in colour" + (
            ": a demonstration chain, NOT CONVERGED" if demo else ""
        )
    intervals = os.path.join(out, "posterior_intervals.png")
    _write_figure(
        fig,
        intervals,
        dict(
            said,
            figure="the mean and the central 68 and 95 per cent intervals of the stored reference for each of the "
            "eleven parameters, with the truth of the mock",
            scale="each parameter on its flat box of the example (BOXES): 0 and 1 are the edges of the box",
            chain=chain_said,
        ),
    )

    # 2. The mock and the stored bands of the posterior predictive, one panel per redshift.
    ncols = min(3, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 2.2 * nrows + 0.8),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, panel in zip(axes.flat, panels):
        for (lower, upper), shade in zip(REFERENCE_BANDS, (0.15, 0.3)):
            ax.fill_between(
                panel["k"],
                *panel[(lower, upper)],
                color="0.3",
                alpha=shade,
                lw=0,
                label=f"{lower:g} to {upper:g} per cent of the replicates",
            )
        ax.plot(panel["k"], panel["mock"], "o", color="C3", ms=3, label="the mock")
        ax.set_xscale("log")
        ax.set_title(f"z = {panel['z']:.1f}")
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)  # a grid has more places than there are redshifts
    # the x axis is labelled on the lowest panel of each column, also where that panel is not in the last row
    for column in axes.T:
        lowest = [ax for ax in column if ax.get_visible()][-1]
        lowest.set_xlabel("k (s/km)")
        lowest.tick_params(axis="x", which="both", labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("k P(k) / pi")
    fig.suptitle("The mock and the posterior predictive of the stored reference (in grey)")
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    per_redshift = os.path.join(out, "posterior_predictive_per_redshift.png")
    _write_figure(
        fig,
        per_redshift,
        dict(
            said,
            figure="the mock at each of its redshifts, with the stored bands of the posterior predictive of the "
            "stored reference (replicates of the data: the model at a posterior draw plus noise from the covariance)",
            mock=_download_item("mock"),
            bands="the stored percentiles 2.5 to 97.5 and 16 to 84 of the posterior predictive (REFERENCE_BANDS)",
            y="k P(k) / pi, with k angular in s/km",
            chain="none: this figure shows the stored reference and the mock alone",
        ),
    )
    return [intervals, per_redshift]


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
