"""The stored mock chain of the tutorial notebook: its preflight, and the one registered run.

The chain is made by the notebook's own code. This script takes the helper cells from the builder beside it
(``_build_tutorial.helper_source``) and runs them in one namespace, as the notebook does: the stored-file helper,
the four-parameter posterior, the mock, the prior, the sampler, the diagnostics and the chain files. It adds
nothing to the posterior; around it, it only counts.

    python run_mock_chain.py --preflight OUT_DIR
        No chain. Checks the posterior against the public likelihood on the published model, times it, takes the
        Hessian at the truth, measures how far the two edges of the prior that are not plain ranges lie from the
        posterior (the class fractions summing to one; the mean-flux edge of the multi-fidelity tables), runs four
        pilot chains of 300 steps, and writes ``preflight.json``.

    python run_mock_chain.py --run OUT_DIR --from-preflight PREFLIGHT_DIR
        The registered run, once: four chains from the registered starts with the registered seeds, the number of
        steps and the proposal of the preflight. Writes the chain files and ``mock_chain_4p.provenance.json`` with
        the verdict PASSED or FAILED. There is no option that changes a setting, lengthens a run or loosens a
        threshold: on FAILED the diagnostics are written and the script stops.

OUT_DIR must not exist. The settings below are the registered ones.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import resource
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
PREFIX = "mock_chain_4p"
# the cells run, in the notebook's order
CELLS = (
    "helpers-assets",
    "helpers-posterior",
    "mock",
    "posterior",
    "helpers-sampler",
    "helpers-diagnostics",
    "helpers-chain-files",
)

# ---- the registered settings (not options)
N_CHAINS = 4
START_SEED = 20261120  # chain i starts from np.random.default_rng(START_SEED + i)
SAMPLER_SEED = 20261130  # and is sampled with np.random.default_rng(SAMPLER_SEED + i)
START_SCALE = 9.0  # the starts are drawn around the truth with covariance START_SCALE * SIGMA
BURN_IN = 1000  # the first steps of each chain, dropped before the diagnostics and the summaries
MAX_STEPS = 10000
MIN_STEPS = 5000  # the run is not started below this
BUDGET_SECONDS = (
    1800.0  # n_steps = min(MAX_STEPS, floor(BUDGET_SECONDS / (N_CHAINS * seconds per evaluation)))
)
PILOT_STEPS = 300
# the pass criteria, all of them, per parameter and per chain
RHAT_MAX = 1.01
ESS_BULK_MIN = 1000.0
ESS_TAIL_MIN = 1000.0
ACCEPTANCE_RANGE = (0.15, 0.50)
PRIOR_WORDS = "uniform within the stated ranges, restricted to physically allowed class fractions"
PRIOR_RESTRICTION = (
    "the class fractions (r_LLS times the model's own LLS fraction, plus its subDLA and DLA fractions) sum to at "
    "most one at every redshift of the data"
)
# where the multi-fidelity correction's tables end inside the trained range of tau0_amp, at the redshifts of the
# data (the narrowest interval over them, from a scan at the midpoint parameters): outside it the correction is held
# at its nearest table value and the predictor says so with a MeanFluxRangeWarning
MF_TABLE_EDGES = (0.6572, 1.3296)


def builder():
    spec = importlib.util.spec_from_file_location("_build_tutorial", HERE / "_build_tutorial.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_namespace(say=print):
    """The notebook's cells of ``CELLS`` run in one namespace with the published model: (namespace, cell digests)."""
    import hcd_analysis.predictor as hp

    cells = builder()
    ns = {"__name__": "tutorial_cells", "print": say}
    digests = {}
    for cell_id in CELLS:
        source = cells.helper_source(cell_id)
        digests[cell_id] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if cell_id == "mock":  # what the notebook has done before this cell
            ns["emu"] = hp.load_published()
        exec(compile(source, f"<cell {cell_id}>", "exec"), ns)
        if (
            cell_id == "helpers-assets"
        ):  # this script may be started from any folder: the clone is here
            ns["LOCAL_ROOTS"] = (REPO,)
    return ns, digests


class CountedPosterior:
    """``log_posterior`` of the notebook, unchanged, with counts of what happens at the points it is asked for."""

    def __init__(self, ns):
        import hcd_analysis.predictor as hp

        self.ns, self.hp = ns, hp
        self.counts = {
            "evaluations": 0,
            "outside_the_stated_ranges": 0,
            "refused_class_fractions_above_one": 0,
            "refused_otherwise": 0,
            "mean_flux_edge": 0,
        }
        self.other_warnings = {}
        self.first_messages = {}
        self.edge_states = set()

    def __call__(self, x):
        ns, c = self.ns, self.counts
        x = np.asarray(x, dtype=float)
        c["evaluations"] += 1
        if any(
            not (ns["PRIOR"][name][0] <= value <= ns["PRIOR"][name][1])
            for value, name in zip(x, ns["FREE"])
        ):
            c["outside_the_stated_ranges"] += 1
        before = {kind: entry[0] for kind, entry in ns["REFUSALS"].items()}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = ns["log_posterior"](x)
        for kind, entry in ns["REFUSALS"].items():
            if entry[0] > before.get(kind, 0):
                key = (
                    "refused_class_fractions_above_one"
                    if kind == "PredictorInputError"
                    else "refused_otherwise"
                )
                c[key] += entry[0] - before.get(kind, 0)
                self.first_messages.setdefault(kind, entry[1])
        edge = False
        for w in caught:
            if issubclass(w.category, self.hp.MeanFluxRangeWarning):
                edge = True
                self.first_messages.setdefault("MeanFluxRangeWarning", str(w.message))
            else:
                name = w.category.__name__
                self.other_warnings[name] = self.other_warnings.get(name, 0) + 1
        if edge:
            c["mean_flux_edge"] += 1
            self.edge_states.add(x.tobytes())
        return value

    def edge_steps(self, chain):
        """Which steps of a chain sit on a state whose prediction met the mean-flux edge."""
        return np.array([state.tobytes() in self.edge_states for state in chain])


def hessian(f, x0, h):
    """The Hessian of ``f`` at ``x0`` by central differences with the steps ``h``."""
    d = x0.size
    out = np.empty((d, d))
    f0 = f(x0)
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h[i]
        out[i, i] = (f(x0 + ei) - 2.0 * f0 + f(x0 - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(d)
            ej[j] = h[j]
            out[i, j] = out[j, i] = (
                f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return out


def laplace_covariance(ns):
    """SIGMA: the inverse Hessian of minus the log posterior at the truth. The steps are a fifth of the standard
    deviation that a first pass (steps of a thousandth of the prior's width) gives for each parameter alone.
    """
    x0 = np.array(ns["X_TRUTH"], dtype=float)
    width = np.array([ns["PRIOR"][name][1] - ns["PRIOR"][name][0] for name in ns["FREE"]])

    def minus(x):
        return -ns["log_posterior"](x)

    first = np.array([hessian(minus, x0, 1e-3 * width)[i, i] for i in range(x0.size)])
    if not np.all(first > 0):
        raise SystemExit(
            f"the log posterior does not curve downwards at the truth along each parameter: {first}"
        )
    steps = 0.2 / np.sqrt(first)
    h = hessian(minus, x0, steps)
    eigenvalues = np.linalg.eigvalsh(h)
    if not np.all(eigenvalues > 0):
        raise SystemExit(
            f"the Hessian at the truth is not positive definite: eigenvalues {eigenvalues}"
        )
    return np.linalg.inv(h), h, steps


def registered_starts(ns, sigma, logp):
    """The start of each chain: drawn around the truth with covariance START_SCALE * SIGMA and its own generator,
    redrawn until it lies inside the prior (the stated ranges and physically allowed class fractions).
    """
    lower = np.linalg.cholesky(START_SCALE * sigma)
    starts, redraws = [], []
    for i in range(N_CHAINS):
        rng = np.random.default_rng(START_SEED + i)
        n = 0
        while True:
            x = np.array(ns["X_TRUTH"], dtype=float) + lower @ rng.standard_normal(len(ns["FREE"]))
            if np.isfinite(logp(x)):
                break
            n += 1
        starts.append(x)
        redraws.append(n)
    return np.array(starts), redraws


def sample(ns, sigma, n_steps, say=print):
    """Four chains of ``n_steps`` from the registered starts with the registered seeds and the proposal
    (2.38**2 / d) * SIGMA: the chains, their log posteriors, one record per chain, the proposal covariance, and
    for each chain which of its steps sat on a state whose prediction met the mean-flux edge."""
    d = len(ns["FREE"])
    proposal = (2.38**2 / d) * sigma
    setup = CountedPosterior(ns)
    starts, redraws = registered_starts(ns, sigma, setup)
    chains, logps, records, edges = [], [], [], []
    for i in range(N_CHAINS):
        counted = CountedPosterior(ns)
        t0 = time.perf_counter()
        chain, logp, acceptance = ns["metropolis"](
            counted, starts[i], proposal, n_steps, np.random.default_rng(SAMPLER_SEED + i)
        )
        seconds = time.perf_counter() - t0
        edge = counted.edge_steps(chain)
        counts = dict(counted.counts)
        proposals = counts["evaluations"] - 1  # the first evaluation is that of the start
        records.append(
            {
                "chain": i + 1,
                "start": [float(v) for v in starts[i]],
                "start_redraws": redraws[i],
                "start_seed": START_SEED + i,
                "sampler_seed": SAMPLER_SEED + i,
                "steps": int(n_steps),
                "acceptance": float(acceptance),
                "seconds": seconds,
                "proposals": proposals,
                "proposals_outside_the_stated_ranges": counts["outside_the_stated_ranges"],
                "proposals_refused_class_fractions_above_one": counts[
                    "refused_class_fractions_above_one"
                ],
                "proposals_refused_otherwise": counts["refused_otherwise"],
                "proposals_at_the_mean_flux_edge": counts["mean_flux_edge"],
                "steps_at_the_mean_flux_edge": int(edge.sum()),
                "largest_r_LLS": float(chain[:, ns["FREE"].index("r_LLS")].max()),
                "other_warnings": dict(counted.other_warnings),
                "first_messages": dict(counted.first_messages),
            }
        )
        chains.append(chain)
        logps.append(logp)
        edges.append(edge)
        say(
            f"chain {i + 1}: acceptance {acceptance:.3f}, {seconds:.1f} s; of {proposals} proposals "
            f"{counts['outside_the_stated_ranges']} outside the stated ranges, "
            f"{counts['refused_class_fractions_above_one']} refused (class fractions above one), "
            f"{counts['refused_otherwise']} refused otherwise, {counts['mean_flux_edge']} at the mean-flux edge; "
            f"{int(edge.sum())} steps at the mean-flux edge; largest r_LLS {records[-1]['largest_r_LLS']:.3f}"
        )
    return np.array(chains), np.array(logps), records, proposal, np.array(edges)


def diagnostics(ns, chains):
    out = ns["split_rhat_and_ess"](chains)
    return {
        key: {name: float(v) for name, v in zip(ns["FREE"], values)} for key, values in out.items()
    }


def summaries(ns, draws):
    """Mean, standard deviation and three quantiles of each parameter over ``draws``, shape (n, d)."""
    if len(draws) == 0:
        return None
    q = np.quantile(draws, [0.16, 0.5, 0.84], axis=0)
    return {
        name: {
            "mean": float(draws[:, j].mean()),
            "sd": float(draws[:, j].std(ddof=1)) if len(draws) > 1 else float("nan"),
            "q16": float(q[0, j]),
            "median": float(q[1, j]),
            "q84": float(q[2, j]),
        }
        for j, name in enumerate(ns["FREE"])
    }


def boundary_of_class_fractions(ns):
    """Where the class fractions sum to one, as a value of r_LLS: r_max(z) = (1 - w_subDLA - w_DLA) / w_LLS at each
    redshift of the data, at the truth and at corners of the other three free parameters."""
    import hcd_analysis.predictor as hp

    like, truth = ns["LIKE"], ns["TRUTH"]
    i_lls, i_sub, i_dla = (hp.HCD_CLASSES.index(name) for name in ("LLS", "subDLA", "DLA"))

    def r_max(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = np.asarray(like.prediction(theta).w_c)
        return (1.0 - w[:, i_sub] - w[:, i_dla]) / w[:, i_lls]

    at_truth = r_max(truth)
    z = [float(v) for v in like.data.z]
    worst = int(np.argmin(at_truth))
    corners = []
    for ns_value in ns["PRIOR"]["ns"]:
        for ap_value in ns["PRIOR"]["Ap"]:
            for tau in (0.8, 1.2):
                r = r_max(dict(truth, ns=ns_value, Ap=ap_value, tau0_amp=tau))
                corners.append(
                    {
                        "ns": float(ns_value),
                        "Ap": float(ap_value),
                        "tau0_amp": tau,
                        "r_max_at_that_redshift": float(r[worst]),
                        "smallest_r_max": float(r.min()),
                        "at_z": z[int(np.argmin(r))],
                    }
                )
    return {
        "r_max_at_the_truth_by_redshift": dict(
            zip([f"{v:.1f}" for v in z], [float(v) for v in at_truth])
        ),
        "smallest_r_max_at_the_truth": float(at_truth.min()),
        "at_z": z[worst],
        "corners": corners,
    }


def job_resources():
    out = {
        "cpus_this_process_may_use": len(os.sched_getaffinity(0)),
        "slurm_job_submitted_by_this_script": False,
    }
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        r = subprocess.run(["scontrol", "show", "job", job], capture_output=True, text=True)
        tres = [
            word
            for word in r.stdout.split()
            if word.startswith(("TRES=", "ReqTRES=", "AllocTRES="))
        ]
        out["session_job_tres"] = tres
    return out


def code_identity(digests):
    def git(*args):
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--", "notebooks/tutorial")
    return {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes_under_notebooks_tutorial": None if status is None else bool(status),
        "cells_sha256": digests,
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }


def identities(ns):
    return {
        "model_release": dict(ns["emu"].provenance["release"]),
        "example_download_sha256": ns["ASSETS"]["hcd_priya_example_download.tar.gz"]["sha256"],
        "example_mock_p1d_json_sha256": ns["EXAMPLE_MOCK_SHA256"],
        "mock": dict(ns["MOCK"].provenance),
        "truth": dict(ns["TRUTH"]),
        "relative_amplitudes_at_the_truth": dict(ns["REL_TRUTH"]),
        "free": list(ns["FREE"]),
        "prior": {
            "in_words": PRIOR_WORDS,
            "ranges": {k: list(v) for k, v in ns["PRIOR"].items()},
            "restriction": PRIOR_RESTRICTION,
        },
    }


def write_json(path, doc):
    with open(path, "x", encoding="utf-8") as out:
        out.write(json.dumps(doc, indent=2) + "\n")


def new_folder(path):
    path = pathlib.Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists already: an output folder is never reused")
    path.mkdir(parents=True)
    return path


# ============================================================================================ the preflight
def preflight(out_dir):
    out_dir = new_folder(out_dir)
    ns, digests = notebook_namespace()
    like, truth, free = ns["LIKE"], ns["TRUTH"], ns["FREE"]
    doc = {
        "what": "preflight of the tutorial's mock chain: no chain was run",
        "code": code_identity(digests),
    }
    doc.update(identities(ns))

    # 1. the assembly against the public likelihood on the published model (the notebook's cell asserts the same)
    from hcd_analysis import likelihood as hl

    worst_vector = worst_loglike = 0.0
    for r_lls in (0.0, 0.5, 1.0, 2.0):
        rel = dict(ns["REL_TRUTH"], LLS=r_lls)
        public = hl.GaussianP1DLikelihood(ns["emu"], ns["MOCK"], hcd_relative_to_priya=rel)
        for theta in ns["POINTS"]:
            ours, theirs = ns["model_vector_free_hcd"](like, theta, rel), public.model_vector(theta)
            worst_vector = max(worst_vector, float(np.max(np.abs(ours / theirs - 1.0))))
            worst_loglike = max(
                worst_loglike,
                abs(ns["log_likelihood_free_hcd"](like, theta, rel) / public.loglike(theta) - 1.0),
            )
    chi2 = like.chi2(truth)
    doc["equality_with_the_public_likelihood"] = {
        "points": 12,
        "largest_relative_difference_of_the_model_vector": worst_vector,
        "largest_relative_difference_of_the_log_likelihood": worst_loglike,
        "bounds": {"model_vector": 1e-12, "log_likelihood": 1e-10},
    }
    doc["chi_square_at_the_truth"] = chi2
    print(
        f"equality with the public likelihood at 12 points: model vector {worst_vector:.1e}, log likelihood {worst_loglike:.1e}"
    )
    print(f"chi-square at the truth: {chi2}")
    if not (worst_vector < 1e-12 and worst_loglike < 1e-10 and chi2 == 0.0):
        write_json(
            out_dir / "preflight.json",
            dict(doc, verdict="PREFLIGHT FAILED: the equality or the mock"),
        )
        raise SystemExit(
            "PREFLIGHT FAILED: the assembly is not the public likelihood, or the mock is not the model"
        )

    # 2. the Hessian at the truth, SIGMA and the proposal
    sigma, h, steps = laplace_covariance(ns)
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    d = len(free)
    doc["laplace"] = {
        "hessian_steps": dict(zip(free, [float(v) for v in steps])),
        "SIGMA": sigma.tolist(),
        "standard_deviations": dict(zip(free, [float(v) for v in sd])),
        "correlations": corr.tolist(),
        "proposal_covariance": ((2.38**2 / d) * sigma).tolist(),
    }
    print("Laplace standard deviations:", ", ".join(f"{name} {v:.4g}" for name, v in zip(free, sd)))
    print("correlations of SIGMA (order " + ", ".join(free) + "):")
    for row in corr:
        print("   " + " ".join(f"{v:+.3f}" for v in row))

    # 3. the time of one evaluation
    rng = np.random.default_rng(0)
    lower = np.linalg.cholesky(0.25 * sigma)
    ns["log_posterior"](ns["X_TRUTH"])
    times = []
    for _ in range(50):
        x = np.array(ns["X_TRUTH"]) + lower @ rng.standard_normal(d)
        t0 = time.perf_counter()
        ns["log_posterior"](x)
        times.append(time.perf_counter() - t0)
    median, p95 = float(np.median(times)), float(np.quantile(times, 0.95))
    n_steps = int(min(MAX_STEPS, np.floor(BUDGET_SECONDS / (N_CHAINS * median))))
    doc["timing"] = {
        "evaluations_timed": 50,
        "seconds_per_evaluation_median": median,
        "seconds_per_evaluation_95th_percentile": p95,
        "n_steps": n_steps,
        "n_steps_rule": "min(10000, floor(1800 / (4 * seconds per evaluation, median)))",
        "the_run_is_not_started_below": MIN_STEPS,
        "projected_seconds_four_chains_one_after_the_other": [
            N_CHAINS * n_steps * median,
            N_CHAINS * n_steps * p95,
        ],
    }
    print(
        f"seconds per evaluation: median {median:.4f}, 95th percentile {p95:.4f}; n_steps = {n_steps}"
    )
    print(
        f"projected wall time of {N_CHAINS} chains of {n_steps} steps, one after the other: "
        f"{N_CHAINS * n_steps * median / 60:.1f} to {N_CHAINS * n_steps * p95 / 60:.1f} minutes"
    )

    # 4. the class-fraction boundary and the mean-flux edge, against the Laplace widths
    boundary = boundary_of_class_fractions(ns)
    sd_r, sd_tau = sd[free.index("r_LLS")], sd[free.index("tau0_amp")]
    boundary["distance_from_the_truth_in_laplace_sd_of_r_LLS"] = (
        boundary["smallest_r_max_at_the_truth"] - 1.0
    ) / sd_r
    doc["class_fraction_boundary"] = boundary
    doc["mean_flux_edge"] = {
        "table_edges_in_tau0_amp": list(MF_TABLE_EDGES),
        "distance_of_each_edge_from_the_truth_in_laplace_sd_of_tau0_amp": [
            (MF_TABLE_EDGES[0] - truth["tau0_amp"]) / sd_tau,
            (MF_TABLE_EDGES[1] - truth["tau0_amp"]) / sd_tau,
        ],
    }
    print(
        "the value of r_LLS at which the class fractions sum to one, at the truth, by redshift of the data:"
    )
    print(
        "   "
        + ", ".join(
            f"z {z}: {v:.3f}" for z, v in boundary["r_max_at_the_truth_by_redshift"].items()
        )
    )
    print(
        f"   the smallest: {boundary['smallest_r_max_at_the_truth']:.3f} at z = {boundary['at_z']}, which is "
        f"{boundary['distance_from_the_truth_in_laplace_sd_of_r_LLS']:.2f} Laplace standard deviations of r_LLS "
        f"({sd_r:.3f}) above the truth"
    )
    print("   at the corners of (ns, Ap), tau0_amp 0.8 and 1.2, at that redshift:")
    for corner in boundary["corners"]:
        print(
            f"      ns {corner['ns']:.2f}, Ap {corner['Ap']:.1e}, tau0_amp {corner['tau0_amp']:.1f}: "
            f"{corner['r_max_at_that_redshift']:.3f} (smallest over the redshifts {corner['smallest_r_max']:.3f} "
            f"at z = {corner['at_z']})"
        )
    edges = doc["mean_flux_edge"]["distance_of_each_edge_from_the_truth_in_laplace_sd_of_tau0_amp"]
    print(
        f"the mean-flux edges {MF_TABLE_EDGES[0]} and {MF_TABLE_EDGES[1]} lie {edges[0]:+.1f} and {edges[1]:+.1f} "
        f"Laplace standard deviations of tau0_amp ({sd_tau:.4f}) from the truth"
    )

    # 5. four pilot chains: instructional, not evidence of convergence
    print(f"pilot chains of {PILOT_STEPS} steps (instructional, not evidence of convergence):")
    chains, _, records, _, _ = sample(ns, sigma, PILOT_STEPS)
    doc["pilot"] = {
        "label": f"pilot of {PILOT_STEPS} steps: instructional, not evidence of convergence",
        "chains": records,
        "diagnostics": diagnostics(ns, chains),
    }
    for key, values in doc["pilot"]["diagnostics"].items():
        print(
            f"   pilot {key}: "
            + ", ".join(
                f"{name} {v:.3f}" if key == "rhat" else f"{name} {v:.0f}"
                for name, v in values.items()
            )
        )

    doc["resources"] = dict(
        job_resources(),
        peak_rss_megabytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    )
    doc["registered"] = registered_settings()
    doc["verdict"] = (
        "PREFLIGHT DONE: no chain was run"
        if n_steps >= MIN_STEPS
        else "PREFLIGHT DONE: n_steps is below the minimum, the run would not start"
    )
    write_json(out_dir / "preflight.json", doc)
    print(
        f"peak memory of this process: {doc['resources']['peak_rss_megabytes']:.0f} MB; "
        f"CPUs this process may use: {doc['resources']['cpus_this_process_may_use']}; "
        f"the session job: {' '.join(doc['resources'].get('session_job_tres', ['not a SLURM job']))}; "
        "no SLURM job is submitted by this script"
    )
    print(doc["verdict"])
    return 0


def registered_settings():
    return {
        "chains": N_CHAINS,
        "start_seeds": [START_SEED + i for i in range(N_CHAINS)],
        "sampler_seeds": [SAMPLER_SEED + i for i in range(N_CHAINS)],
        "start_covariance": f"{START_SCALE:g} * SIGMA around the truth, redrawn until inside the prior",
        "proposal_covariance": "(2.38**2 / 4) * SIGMA, fixed",
        "burn_in_steps": BURN_IN,
        "pass_criteria": {
            "rank_normalised_split_rhat_at_most": RHAT_MAX,
            "bulk_ess_at_least": ESS_BULK_MIN,
            "tail_ess_at_least": ESS_TAIL_MIN,
            "acceptance_of_each_chain_between": list(ACCEPTANCE_RANGE),
        },
    }


def verdict_of(diag, acceptances):
    """PASSED when every criterion holds; otherwise FAILED with the criteria that do not."""
    failed = []
    for name, value in diag["rhat"].items():
        if not value <= RHAT_MAX:
            failed.append(f"split R-hat of {name} = {value:.4f} is above {RHAT_MAX}")
    for key, least in (("ess_bulk", ESS_BULK_MIN), ("ess_tail", ESS_TAIL_MIN)):
        for name, value in diag[key].items():
            if not value >= least:
                failed.append(f"{key} of {name} = {value:.0f} is below {least:.0f}")
    for i, value in enumerate(acceptances):
        if not ACCEPTANCE_RANGE[0] <= value <= ACCEPTANCE_RANGE[1]:
            failed.append(
                f"acceptance of chain {i + 1} = {value:.3f} is outside {ACCEPTANCE_RANGE}"
            )
    return ("PASSED" if not failed else "FAILED"), failed


# ============================================================================================ the registered run
def finish_run(ns, out_dir, chains, logps, records, proposal, sigma, n_steps, extra):
    """Write the chain files, read them back, and write the provenance file with the diagnostics of what was read.
    ``extra``: ``edge_steps`` (chains, steps) of booleans, ``smallest_r_max_at_the_truth``, and ``identities``, a
    dict that is written into the provenance file as it is."""
    free = ns["FREE"]
    prefix = pathlib.Path(out_dir) / PREFIX
    ns["write_chains"](prefix, chains, logps, free, ns["PRIOR"])
    stored = ns["expand_chains"](*ns["read_chains"](prefix))
    kept = stored[:, BURN_IN:]
    diag = diagnostics(ns, kept)
    verdict, failed = verdict_of(diag, [r["acceptance"] for r in records])

    # the two edges of the prior that are not plain ranges, as blocks of their own
    edge = np.asarray(extra["edge_steps"], dtype=bool)[:, BURN_IN:]
    pooled, pooled_edge = kept.reshape(-1, kept.shape[2]), edge.reshape(-1)
    j_r = free.index("r_LLS")
    r_edge = extra["smallest_r_max_at_the_truth"]
    proposals = sum(r["proposals"] for r in records)
    boundary = {
        "what": PRIOR_RESTRICTION,
        "proposals": proposals,
        "proposals_refused": sum(r["proposals_refused_class_fractions_above_one"] for r in records),
        "share_of_proposals_refused": sum(
            r["proposals_refused_class_fractions_above_one"] for r in records
        )
        / proposals,
        "smallest_r_max_at_the_truth": r_edge,
        "largest_r_LLS_kept": float(pooled[:, j_r].max()),
        "r_LLS_99.9th_percentile_kept": float(np.quantile(pooled[:, j_r], 0.999)),
        "share_of_kept_steps_with_r_LLS_above_0.9_of_that_r_max": float(
            np.mean(pooled[:, j_r] > 0.9 * r_edge)
        ),
    }
    with_all, without = summaries(ns, pooled), summaries(ns, pooled[~pooled_edge])
    mean_flux = {
        "what": "steps and proposals whose prediction met the edge of the multi-fidelity tables in the mean flux",
        "table_edges_in_tau0_amp": list(MF_TABLE_EDGES),
        "proposals_at_the_edge": sum(r["proposals_at_the_mean_flux_edge"] for r in records),
        "share_of_proposals_at_the_edge": sum(r["proposals_at_the_mean_flux_edge"] for r in records)
        / proposals,
        "kept_steps_at_the_edge": int(pooled_edge.sum()),
        "share_of_kept_steps_at_the_edge": float(pooled_edge.mean()),
        "summaries_with_all_kept_steps": with_all,
        "summaries_without_the_steps_at_the_edge": without,
        "largest_shift_of_a_mean_in_units_of_its_sd": (
            None
            if without is None
            else max(
                abs(with_all[n]["mean"] - without[n]["mean"]) / with_all[n]["sd"] for n in free
            )
        ),
    }
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(pathlib.Path(out_dir).glob(PREFIX + ".*"))
    }
    doc = {
        "what": "the stored mock chain of the tutorial notebook: a self-consistency demonstration on a mock made by "
        "the model itself; not an independent validation, not a result from data",
        "verdict": verdict,
        "criteria_not_met": failed,
        "diagnostics_after_burn_in": diag,
        "acceptance": [r["acceptance"] for r in records],
        "steps_per_chain": int(n_steps),
        "kept_steps_per_chain": int(n_steps - BURN_IN),
        "chains": records,
        "class_fraction_boundary": boundary,
        "mean_flux_edge": mean_flux,
        "SIGMA": np.asarray(sigma).tolist(),
        "proposal_covariance": np.asarray(proposal).tolist(),
        "registered": registered_settings(),
        "files_sha256": files,
    }
    doc.update(extra["identities"])
    write_json(pathlib.Path(out_dir) / (PREFIX + ".provenance.json"), doc)
    return doc


def run(out_dir, preflight_dir):
    pre = json.loads((pathlib.Path(preflight_dir) / "preflight.json").read_text(encoding="utf-8"))
    if not pre["verdict"].startswith("PREFLIGHT DONE: no chain was run"):
        raise SystemExit(f"the preflight in {preflight_dir} did not pass: {pre['verdict']}")
    n_steps = int(pre["timing"]["n_steps"])
    if n_steps < MIN_STEPS:
        raise SystemExit(f"n_steps = {n_steps} is below {MIN_STEPS}: the run is not started")
    out_dir = new_folder(out_dir)
    ns, digests = notebook_namespace()
    code = code_identity(digests)
    if code["cells_sha256"] != pre["code"]["cells_sha256"]:
        raise SystemExit(
            "the notebook's cells are not those of the preflight: make the preflight again"
        )
    sigma = np.array(pre["laplace"]["SIGMA"])
    print(f"the registered run: {N_CHAINS} chains of {n_steps} steps")
    t0 = time.perf_counter()
    chains, logps, records, proposal, edge_steps = sample(ns, sigma, n_steps)
    extra = {
        "edge_steps": edge_steps,
        "smallest_r_max_at_the_truth": pre["class_fraction_boundary"][
            "smallest_r_max_at_the_truth"
        ],
        "identities": dict(
            identities(ns),
            code=code,
            preflight=pre,
            wall_seconds=time.perf_counter() - t0,
            resources=dict(
                job_resources(),
                peak_rss_megabytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            ),
        ),
    }
    doc = finish_run(ns, out_dir, chains, logps, records, proposal, sigma, n_steps, extra)
    for key, values in doc["diagnostics_after_burn_in"].items():
        print(
            f"{key}: "
            + ", ".join(
                f"{name} {v:.4f}" if key == "rhat" else f"{name} {v:.0f}"
                for name, v in values.items()
            )
        )
    print("acceptance:", ", ".join(f"{v:.3f}" for v in doc["acceptance"]))
    print(
        f"class-fraction boundary: {doc['class_fraction_boundary']['proposals_refused']} of {doc['class_fraction_boundary']['proposals']} proposals refused"
    )
    print(
        f"mean-flux edge: {doc['mean_flux_edge']['kept_steps_at_the_edge']} kept steps, {doc['mean_flux_edge']['proposals_at_the_edge']} proposals"
    )
    print(
        "VERDICT:",
        doc["verdict"],
        *doc["criteria_not_met"],
        sep="\n   " if doc["criteria_not_met"] else " ",
    )
    return 0 if doc["verdict"] == "PASSED" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        metavar="OUT_DIR",
        help="no chain: the checks, the timing and four pilot chains",
    )
    mode.add_argument("--run", metavar="OUT_DIR", help="the registered run, once")
    parser.add_argument(
        "--from-preflight", metavar="PREFLIGHT_DIR", help="with --run: the folder of the preflight"
    )
    args = parser.parse_args(argv)
    if args.preflight:
        return preflight(args.preflight)
    if not args.from_preflight:
        parser.error("--run needs --from-preflight PREFLIGHT_DIR")
    return run(args.run, args.from_preflight)


if __name__ == "__main__":
    sys.exit(main())
