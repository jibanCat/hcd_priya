"""Science checks of a trained hcd_priya model. Release level: they run in the assets job, never in CI (no runner has a
trained model), and every test skips when no model is configured.

- Golden values: the predictions at 13 fixed cases (12 points, 3 redshifts, and the midpoint at another mean flux)
  against a golden file made from the same model.
- Every class spectrum is positive at every mode.
- The total P1D, mixed with the model's own fractions, rises with A_p at every mode below k = 0.015 s/km, at the 12
  points and 3 redshifts. The response is asserted below that k only.
- The fractions w_c lie in [0, 1] with a sum of at most 1; dN/dX is positive and inside its held-out range; the CDDF
  is inside its held-out range.
- The held-out prediction residual of the class spectra: every value finite and not negative; the range is the
  spectrum times exp(-rms_log) and times exp(+rms_log); the values are the rows of the release's own record for the
  fidelity of the prediction; and every table of that record is recomputed from the residual vectors it ships.
- The CDDF summed over each HCD class's column-density range agrees with that class's dN/dX within a tolerance taken
  from the prediction's own held-out errors (no observed ratio enters it). dN/dX and f(N_HI) are separate outputs of
  the networks, and under LF+MF each has its own correction table, so they agree only within their errors. For a
  class c the tolerance is
      tau_c = dndx_err_c + (the mean of cddf_err over the class's bins, weighted by f dN),
  both the held-out rms of ln(prediction / truth), added; it is applied on the quantity: the CDDF sum must lie between
  dN/dX exp(-tau_c) and dN/dX exp(+tau_c) for LLS and for subDLA + DLA. For subDLA and for DLA alone the bin across
  20.3 cannot be split, so the interval [whole-bin sum, whole-bin sum + that bin] must meet those bounds. NaN bins
  count as zero; a class with a NaN bin is checked on the side the NaN cannot spoil only.

Configuration, by environment variable (nothing in this file is a model value):
  HCD_SCIENCE_MODEL_CONFIG  a JSON file: {"release_dir": ..., "release_sha256": ...} for a release directory, or
                            {"published": NAME} for a model of the installed model package. Unset: every test skips.
  HCD_SCIENCE_GOLDEN        the golden file of the golden-value test (tools/ci/make_science_golden.py writes it);
                            it must have been made from the configured model. Unset: that test skips.
  HCD_PINNED_STACK=1        the golden values are compared to a relative 1e-12 (on the supported pinned stack);
                            otherwise to 1e-6. The True flags are compared exactly.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import catalog as CAT
from hcd_analysis.emulator.data import PARAM_LIMITS
from tests import science_cddf_sums as GC

CONFIG_ENV, GOLDEN_ENV, PINNED_ENV = (
    "HCD_SCIENCE_MODEL_CONFIG",
    "HCD_SCIENCE_GOLDEN",
    "HCD_PINNED_STACK",
)
GOLDEN_FORMAT = "hcd_priya.science_golden/1"
RTOL_PINNED, RTOL_OTHER = 1e-12, 1e-6
Z = (2.2, 3.4, 4.6)
Z_CDDF = (2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6)
K_INDICES = (0, 5, 20, 60, 120, 171)
CDDF_LOG_NHI = (17.5, 18.5, 19.5, 20.5, 21.5)  # five fixed bins (centres, log10 N_HI)
K_MAX_AP_RESPONSE = 0.015  # s/km: the A_p response is asserted below this k only
SEED = 20261117
MIDPOINT = {
    "ns": 0.925,
    "Ap": 1.9e-9,
    "herei": 4.0,
    "heref": 2.7,
    "alphaq": 2.15,
    "hub": 0.70,
    "omegamh2": 0.143,
    "hireionz": 7.25,
    "bhfeedback": 0.05,
}
DEFAULT_MEAN_FLUX = (1.0, 0.0)  # tau0_amp, dtau0
EXTRA_MEAN_FLUX = ("midpoint", 1.1, 0.2)  # one more case: the midpoint at another mean flux
QUANTITIES = (
    "P_class",
    "w_c",
    "dndx",
    "dndx_lower",
    "dndx_upper",
    "cddf",
    "tau_eff",
    # the held-out prediction residual of the class spectra (a release with a P1D held-out record)
    "P_class_validation_rms_log",
    "P_mix_validation_rms_log",
)


# ------------------------------------------------------------------------------------------- points and cases
def _lhs(rng, n, d):
    """A Latin hypercube of n points in [0, 1)^d: one point in each of the n strata of every axis."""
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def science_points():
    """The 12 parameter points: the midpoint of the training box; 8 corners of the inner box (each parameter at 25 or
    75 per cent of its training range, the pattern from a seeded Latin hypercube of 8 points, so that each parameter
    is at 25 per cent in four points and at 75 per cent in the other four); 3 seeded points inside the inner box (the
    package ships no design table to take them from). Built from the package's PARAM_LIMITS only."""
    lim = np.asarray(PARAM_LIMITS, dtype=np.float64)
    lo, span = lim[:, 0], lim[:, 1] - lim[:, 0]
    rng = np.random.default_rng(SEED)
    corners = np.where(_lhs(rng, 8, 9) < 0.5, 0.25, 0.75)
    inner = 0.25 + 0.5 * _lhs(rng, 3, 9)
    points = [("midpoint", dict(MIDPOINT))]
    for tag, rows in (("inner_corner", corners), ("seeded_inner", inner)):
        for i, u in enumerate(rows):
            points.append(
                (
                    f"{tag}_{i}",
                    {n: float(lo[j] + u[j] * span[j]) for j, n in enumerate(hp.PARAM_NAMES)},
                )
            )
    return points


def science_cases():
    """(point name, params, tau0_amp, dtau0) for every golden case: each point at the default mean flux, and the
    midpoint once more at tau0_amp 1.1, dtau0 0.2."""
    pts = science_points()
    cases = [(name, p) + DEFAULT_MEAN_FLUX for name, p in pts]
    cases.append((EXTRA_MEAN_FLUX[0], dict(pts[0][1])) + EXTRA_MEAN_FLUX[1:])
    return cases


def predict_case(emu, params, tau0_amp, dtau0):
    """The one call the golden values, positivity and fraction checks make: the three redshifts as one stack, at the
    model's default fidelity."""
    return emu.predict_many(params, z=list(Z), tau0_amp=tau0_amp, dtau0=dtau0)


def cddf_bin_indices(pred):
    """The indices of CDDF_LOG_NHI among the prediction's bin centres (matched to 1e-9)."""
    centres = np.asarray(pred.log_nhi, dtype=np.float64)
    out = []
    for c in CDDF_LOG_NHI:
        hit = np.where(np.abs(centres - c) <= 1e-9)[0]
        assert hit.size == 1, f"no CDDF bin centred on {c}"
        out.append(int(hit[0]))
    return out


def true_flags(pred):
    return sorted(k for k, v in pred.flags.items() if isinstance(v, (bool, np.bool_)) and bool(v))


def _validation(pred, name):
    """A validation field of a stacked prediction; a model without a P1D held-out record has none, and then the
    golden checks of a release cannot be made."""
    a = getattr(pred, name)
    assert a is not None, f"the configured model carries no P1D held-out record: no {name}"
    return np.asarray(a)


def case_values(pred):
    """The golden quantities of one stacked prediction as float64 arrays, and the sorted names of its True flags."""
    vals = {
        "P_class": np.asarray(pred.P_class)[:, :, list(K_INDICES)],
        "w_c": pred.w_c,
        "dndx": pred.dndx,
        "dndx_lower": pred.dndx_lower,
        "dndx_upper": pred.dndx_upper,
        "cddf": np.asarray(pred.cddf)[:, cddf_bin_indices(pred)],
        "tau_eff": pred.tau_eff,
        "P_class_validation_rms_log": _validation(pred, "P_class_validation_rms_log")[
            :, :, list(K_INDICES)
        ],
        "P_mix_validation_rms_log": _validation(pred, "P_mix_validation_rms_log")[
            :, list(K_INDICES)
        ],
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in vals.items()}, true_flags(pred)


def encode(a):
    """An array as nested lists of repr strings (exact; NaN as 'nan')."""
    return np.vectorize(lambda x: repr(float(x)), otypes=[object])(
        np.asarray(a, dtype=np.float64)
    ).tolist()


def decode(nested):
    return np.vectorize(float, otypes=[np.float64])(np.array(nested, dtype=object))


def load_configured_model(config):
    """The model a configuration names: a release directory checked against its anchor, or a published model."""
    if "published" in config:
        return hp.load_published(config["published"])
    return hp.load_release(config["release_dir"], config["release_sha256"])


def model_identity(config):
    """What a golden file records as its model: the release's anchor, or the published model's name."""
    return config.get("release_sha256") or config.get("published")


# ----------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def config():
    path = os.environ.get(CONFIG_ENV)
    if not path:
        pytest.skip(f"no model configured: set {CONFIG_ENV} to a configuration JSON")
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def emu(config):
    return load_configured_model(config)


@pytest.fixture(scope="module")
def preds(emu):
    """Every prediction of the golden, positivity and fraction checks, made once: {(point, tau0_amp, dtau0): stacked
    Prediction}."""
    return {(name, a, d): predict_case(emu, p, a, d) for name, p, a, d in science_cases()}


@pytest.fixture(scope="module")
def golden(config):
    path = os.environ.get(GOLDEN_ENV)
    if not path:
        pytest.skip(f"no golden file configured: set {GOLDEN_ENV}")
    doc = json.loads(Path(path).read_text())
    assert doc["format"] == GOLDEN_FORMAT
    assert doc["model"] == model_identity(config), "the golden file was made from another model"
    return doc


# --------------------------------------------------------------------------------------------- golden values
def test_the_predictions_equal_the_golden_values(preds, golden):
    """Every golden quantity at every case: to a relative 1e-12 with HCD_PINNED_STACK=1, else 1e-6; NaN where the
    golden file has NaN; the True flags exactly. The golden file records the points, so a change of them fails here.
    """
    rtol = RTOL_PINNED if os.environ.get(PINNED_ENV) == "1" else RTOL_OTHER
    assert golden["settings"]["z"] == list(Z) and golden["settings"]["k_indices"] == list(K_INDICES)
    assert golden["settings"]["cddf_log_nhi"] == list(CDDF_LOG_NHI)
    assert golden["points"] == [
        {"name": n, "params": p} for n, p in science_points()
    ], "the points changed"
    assert len(golden["cases"]) == len(preds) == 13
    problems, n_compared = [], 0
    for case in golden["cases"]:
        key = (case["point"], case["tau0_amp"], case["dtau0"])
        got, flags = case_values(preds[key])
        for q in QUANTITIES:
            want, g = decode(case[q]), got[q]
            assert g.shape == want.shape, (key, q, g.shape, want.shape)
            if not np.array_equal(np.isnan(g), np.isnan(want)):
                problems.append((key, q, "NaN in other places"))
                continue
            fin = np.isfinite(want)
            n_compared += int(fin.sum())
            bad = fin & ~np.isclose(g, want, rtol=rtol, atol=0.0)
            if bad.any():
                rel = np.max(np.abs(g[bad] / want[bad] - 1.0))
                problems.append(
                    (key, q, f"{int(bad.sum())} values differ, largest relative {rel:.3e}")
                )
        if flags != case["flags_true"]:
            problems.append((key, "flags", f"{flags} != {case['flags_true']}"))
    assert n_compared > 1000
    assert not problems, f"rtol {rtol}: {problems[:10]}"


# ------------------------------------------------------------------------------- the P1D validation diagnostics
def test_the_validation_range_of_the_spectra_is_the_record_s_and_the_record_recomputes(emu, preds):
    """The held-out prediction residual on every golden prediction, against the release's own P1D held-out record:
    finite and not negative; the two ends are the spectrum times exp(-rms_log) and exp(+rms_log); each value is the
    record's row for that redshift and the LF+MF fidelity, and an LF prediction carries the LF rows; every table of
    the record equals its recomputation from the residual vectors of the evidence file."""
    from hcd_analysis.emulator import p1d_heldout as PH

    held, held_dir = emu.p1d_heldout_record()
    assert held is not None, "the configured model carries no P1D held-out record"
    n_cells = 0
    for pred in preds.values():
        e, mix = np.asarray(pred.P_class_validation_rms_log), np.asarray(
            pred.P_mix_validation_rms_log
        )
        assert pred.flags["p1d_validation_available"] is True
        assert (
            np.all(np.isfinite(e))
            and np.all(e >= 0)
            and np.all(np.isfinite(mix))
            and np.all(mix >= 0)
        )
        P = np.asarray(pred.P_class)
        np.testing.assert_allclose(
            pred.P_class_validation_lower, P * np.exp(-e), rtol=1e-14, atol=0
        )
        np.testing.assert_allclose(
            pred.P_class_validation_upper, P * np.exp(+e), rtol=1e-14, atol=0
        )
        for i, z in enumerate(np.atleast_1d(pred.z)):
            row = held.err(float(z), True)
            assert np.array_equal(e[i], row[:4]) and np.array_equal(mix[i], row[4])
        n_cells += e.size
    assert n_cells > 1000
    name, params = science_points()[0]
    lf = emu.predict_many(params, z=list(Z), tau0_amp=1.0, dtau0=0.0, fidelity="LF")
    for i, z in enumerate(Z):
        assert np.array_equal(lf.P_class_validation_rms_log[i], held.err(z, False)[:4])
    assert PH.verify_tables(held_dir, held) > 1_000_000
    for text in (pred.labels["p1d_validation"], held.label(False)):
        assert "empirical validation diagnostics, not calibrated predictive uncertainties" in text


# ----------------------------------------------------------------------------------------------- positivity
def test_every_class_spectrum_is_positive(preds):
    bad = []
    for (name, a, d), p in preds.items():
        if (a, d) != DEFAULT_MEAN_FLUX:
            continue
        P = np.asarray(p.P_class)
        assert P.shape[:2] == (len(Z), 4) and P.shape[2] == np.asarray(p.k).shape[1]
        for iz, ic, ik in zip(*np.where(~(P > 0))):
            bad.append((name, Z[iz], hp.CLASSES[ic], int(ik), float(P[iz, ic, ik])))
    assert not bad, bad[:10]


# ------------------------------------------------------------------------------------------ the A_p response
def test_the_total_p1d_rises_with_ap_below_k_0_015(emu):
    """Finite difference at A_p x 1.02 and x 0.98: the total P1D, mixed with the model's own fractions at each A_p,
    rises at every mode below k = 0.015 s/km, at every point and redshift. At least the first ten modes of every
    (point, redshift) are below that k, so the check is never empty."""
    bad, n_cells = [], 0
    for name, params in science_points():
        up = predict_case(emu, dict(params, Ap=params["Ap"] * 1.02), *DEFAULT_MEAN_FLUX)
        dn = predict_case(emu, dict(params, Ap=params["Ap"] * 0.98), *DEFAULT_MEAN_FLUX)
        t_up, t_dn = hp.add_hcd(up, up.w_c), hp.add_hcd(dn, dn.w_c)
        for iz, z in enumerate(Z):
            k = np.asarray(up.k[iz], dtype=np.float64)
            assert np.array_equal(k, np.asarray(dn.k[iz], dtype=np.float64))
            low = k < K_MAX_AP_RESPONSE
            assert low[:10].all(), (name, z)
            n_cells += int(low.sum())
            for ik in np.where(low & ~(t_up[iz] > t_dn[iz]))[0]:
                bad.append((name, z, int(ik), float(k[ik])))
    assert n_cells > 0
    assert (
        not bad
    ), f"{len(bad)} of {n_cells} cells below k = {K_MAX_AP_RESPONSE} s/km do not rise: {bad[:10]}"


# -------------------------------------------------------------------------------- fractions and incidence
def test_fractions_and_incidence_are_physical_and_inside_their_ranges(preds):
    bad = []
    n_dndx_range = n_cddf_range = 0
    for key, p in preds.items():
        w = np.asarray(p.w_c)
        if not (np.all(w >= 0) and np.all(w <= 1) and np.all(w.sum(axis=-1) <= 1)):
            bad.append((key, "w_c", w.tolist()))
        d, lo, hi = np.asarray(p.dndx), np.asarray(p.dndx_lower), np.asarray(p.dndx_upper)
        if not np.all(d > 0):
            bad.append((key, "dndx", d.tolist()))
        fin = np.isfinite(lo) & np.isfinite(hi)
        n_dndx_range += int(fin.sum())
        if not (np.all(lo[fin] < d[fin]) and np.all(d[fin] < hi[fin])):
            bad.append((key, "dndx range"))
        c, cl, cu = np.asarray(p.cddf), np.asarray(p.cddf_lower), np.asarray(p.cddf_upper)
        fin = np.isfinite(c) & np.isfinite(cl) & np.isfinite(cu)
        n_cddf_range += int(fin.sum())
        if not (np.all(cl[fin] < c[fin]) and np.all(c[fin] < cu[fin])):
            bad.append((key, "cddf range"))
    assert n_dndx_range > 0 and n_cddf_range > 0  # the ranges exist: the check is not empty
    assert not bad, bad[:10]


# ------------------------------------------------------------------------------- the CDDF against dN/dX
def _weighted_err(s_bins, err_bins):
    """The mean of the bins' held-out errors weighted by their absorbers per unit X (NaN bins weigh nothing); NaN if a
    bin with weight has no error."""
    w = np.where(np.isfinite(s_bins), s_bins, 0.0)
    if w.sum() <= 0 or not np.all(np.isfinite(err_bins[w > 0])):
        return math.nan
    return float(np.sum(w * np.where(w > 0, err_bins, 0.0)) / w.sum())


def cddf_against_dndx(log_nhi, cddf, cddf_err, dndx, dndx_err):
    """Per class (and subDLA + DLA): the CDDF sum as an interval [lo, hi] (equal ends where no bin straddles; hi
    infinite where a NaN bin could hide absorbers), dN/dX, the tolerance tau, and whether the interval meets
    [dN/dX exp(-tau), dN/dX exp(+tau)]."""
    e = GC.edges_from_centres(log_nhi)
    s_all = GC.per_bin_per_x(cddf, e)
    s = GC.class_sums(cddf, e)
    err = np.asarray(cddf_err, dtype=np.float64)
    d = dict(zip(hp.HCD_CLASSES, np.asarray(dndx, dtype=np.float64)))
    ed = dict(zip(hp.HCD_CLASSES, np.asarray(dndx_err, dtype=np.float64)))
    out = {}
    for c in hp.HCD_CLASSES:
        bins = np.concatenate([s[c]["whole_bins"], s[c]["straddle_bins"]]).astype(int)
        straddle_nan = bool(np.any(~np.isfinite(s_all[s[c]["straddle_bins"]])))
        lo = s[c]["whole"]
        hi = math.inf if (s[c]["nan_bins"].size or straddle_nan) else lo + s[c]["straddle"]
        out[c] = dict(lo=lo, hi=hi, dndx=d[c], e_d=ed[c], e_i=_weighted_err(s_all[bins], err[bins]))
    above = e[:-1] >= CAT.LOG_NHI_SUBDLA_MIN - GC.EDGE_TOL
    tot = float(np.sum(np.where(np.isfinite(s_all) & above, s_all, 0.0)))
    dsum = d["subDLA"] + d["DLA"]
    out["subDLA+DLA"] = dict(
        lo=tot,
        hi=math.inf if np.any(~np.isfinite(s_all[above])) else tot,
        dndx=dsum,
        e_d=(d["subDLA"] * ed["subDLA"] + d["DLA"] * ed["DLA"]) / dsum,
        e_i=_weighted_err(s_all[above], err[above]),
    )
    for v in out.values():
        v["tau"] = v["e_d"] + v["e_i"]
        lower, upper = v["dndx"] * math.exp(-v["tau"]), v["dndx"] * math.exp(v["tau"])
        v["holds"] = v["lo"] <= upper and v["hi"] >= lower
    return out


def test_the_cddf_of_each_class_agrees_with_its_dndx_within_the_held_out_errors(emu):
    """At the 12 points x 7 redshifts (2.2 to 4.6) and both fidelities (the default, and LF): every class whose two
    held-out errors are finite meets the bounds of the module docstring. Cells without a held-out error are counted,
    not checked; at least one cell per class and fidelity must be checked."""
    failures, checked = [], {}
    for fidelity in (None, "LF"):
        for name, params in science_points():
            kw = {} if fidelity is None else {"fidelity": "LF"}
            p = emu.predict_many(params, z=list(Z_CDDF), tau0_amp=1.0, dtau0=0.0, **kw)
            assert p.cddf is not None and p.dndx_err is not None and p.cddf_err is not None
            for iz, z in enumerate(Z_CDDF):
                res = cddf_against_dndx(
                    p.log_nhi, p.cddf[iz], p.cddf_err[iz], p.dndx[iz], p.dndx_err[iz]
                )
                for c, v in res.items():
                    key = (fidelity or "default", c)
                    if not (math.isfinite(v["e_d"]) and math.isfinite(v["e_i"])):
                        continue
                    checked[key] = checked.get(key, 0) + 1
                    if not v["holds"]:
                        failures.append((key, name, z, v))
    for fid in ("default", "LF"):
        for c in (*hp.HCD_CLASSES, "subDLA+DLA"):
            assert checked.get((fid, c), 0) > 0, f"no cell checked for {c} at fidelity {fid}"
    assert not failures, failures[:5]
