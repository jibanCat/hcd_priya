"""The parameter boxes of hcd_priya against the published PRIYA ranges, and their documented relations. Needs the
repository: it reads docs/figures/data/priya_lf_design_60.csv and examples/mock_recovery.py.

Every number here is a published value with its citation, or read from those two files.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import data as D

REPO = Path(__file__).resolve().parents[1]

NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback")
# Bird et al. 2023, "PRIYA: a new suite of Lyman-alpha forest simulations for cosmology", JCAP 2023 (10) 037,
# arXiv:2306.05471 (v2 read): Table 2, "Summary of varied emulator parameters, together with the ranges covered by the
# emulator" (n_P, A_P, z_Hei, z_Hef, alpha_q, h, Omega_M h^2, z_HI, epsilon_AGN). A_P from Section 2.1 ("We vary A_P in
# the range 1.2e-9 and 2.6e-9"): Table 2 of v2 prints 2.2e-9 as the lower end, which the 48 design points contradict
# (they reach 1.22e-9).
BIRD2023 = {
    "ns": (0.8, 0.995),
    "Ap": (1.2e-9, 2.6e-9),
    "herei": (3.5, 4.1),
    "heref": (2.6, 3.2),
    "alphaq": (1.3, 2.5),
    "hub": (0.65, 0.75),
    "omegamh2": (0.14, 0.146),
    "hireionz": (6.5, 8.0),
    "bhfeedback": (0.03, 0.07),
}
# Fernandez, Bird and Ho 2024, "Cosmological constraints from the eBOSS Lyman-alpha forest using the PRIYA
# simulations", JCAP 2024 (07) 029, arXiv:2309.03943 (v2 read): Table 2, "Summary of likelihood function parameters,
# together with the ranges covered by the emulator"; Section 3.4.1: uniform priors within these limits except Gaussian
# priors on h (sigma 0.015) and on epsilon_AGN (sigma 0.005).
FERNANDEZ2024 = dict(
    BIRD2023, ns=(0.8, 1.05), alphaq=(1.3, 3.0), tau0_amp=(0.75, 1.25), dtau0=(-0.4, 0.25)
)
FERNANDEZ2024_GAUSSIAN_SD = {"hub": 0.015, "bhfeedback": 0.005}
# The emulator configuration the package cites as the source of PARAM_LIMITS: the entry param_limits of
# Emulator_Files/emulator_params.json of https://github.com/mafern/InferenceLyaData at commit 2c68e0749aaa (sha256
# 623aa4ca..., the design file of docs/figures/data/priya_lf_design_60.csv). Not a table of a paper: herei up to 4.5 and
# heref from 2.2 appear in no published table.
DESIGN_FILE_LIMITS = {
    "ns": (0.8, 1.05),
    "Ap": (1.2e-9, 2.6e-9),
    "herei": (3.5, 4.5),
    "heref": (2.2, 3.2),
    "alphaq": (1.3, 3.0),
    "hub": (0.65, 0.75),
    "omegamh2": (0.14, 0.146),
    "hireionz": (6.5, 8.0),
    "bhfeedback": (0.03, 0.07),
}


def _box(arr):
    return {
        n: (float(lo), float(hi)) for n, (lo, hi) in zip(NAMES, np.asarray(arr, dtype=np.float64))
    }


def _design_points():
    path = REPO / "docs" / "figures" / "data" / "priya_lf_design_60.csv"
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    assert tuple(lines[0].split(",")) == NAMES
    return np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])


def _example():
    path = REPO / "examples" / "mock_recovery.py"
    spec = importlib.util.spec_from_file_location("hcd_mock_recovery_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_parameter_boxes_are_the_published_priya_ranges_and_consistent():
    """- PARAM_LIMITS (the training box, which normalises the network inputs) is the param_limits of the cited
      emulator configuration; it equals Table 2 of Fernandez et al. (2024) except herei up to 4.5 and heref from 2.2.
    - SAMPLING_LIMITS is the original PRIYA box of Bird et al. (2023) except ns, kept at the training box's 1.05 (as
      data.py documents); NS_RIDGE is Bird et al.'s upper ns, 0.995.
    - Each box has lo < hi, the sampling box lies inside the training box, the predictor exports both unchanged.
    - Z_LIMITS are the ends of the snapshot grid (2.0, 5.4), so every snapshot maps into [0, 1].
    - The 60 design points (docs/figures/data/priya_lf_design_60.csv) lie inside the training box; rows 0 to 47
      inside Bird et al.'s box; rows 48 to 59 each outside the sampling box (the CSV's header and the docs say the
      predictor flags all 12 as outside its sampling box), and they are what reaches beyond Bird et al.'s box.
    - The example (examples/mock_recovery.py) uses Table 2 of Fernandez et al. (2024) as its flat boxes, as it says;
      its herei and heref boxes are the sampling box; its alphaq box reaches the training box's 3.0, beyond the
      sampling box's 2.5; every box lies inside the training box; its Gaussian widths are the paper's over sqrt 2.
    """
    train, samp = _box(D.PARAM_LIMITS), _box(D.SAMPLING_LIMITS)
    assert tuple(hp.PARAM_NAMES) == NAMES
    assert np.array_equal(hp.PARAM_LIMITS, D.PARAM_LIMITS) and np.array_equal(
        hp.SAMPLING_LIMITS, D.SAMPLING_LIMITS
    )
    assert train == DESIGN_FILE_LIMITS
    assert {n: FERNANDEZ2024[n] for n in NAMES if train[n] != FERNANDEZ2024[n]} == {
        "herei": (3.5, 4.1),
        "heref": (2.6, 3.2),
    }
    assert samp == dict(BIRD2023, ns=train["ns"])
    assert hp.NS_RIDGE == BIRD2023["ns"][1]
    for n in NAMES:
        assert train[n][0] < train[n][1] and samp[n][0] < samp[n][1]
        assert train[n][0] <= samp[n][0] and samp[n][1] <= train[n][1], n
        assert train[n][0] <= BIRD2023[n][0] and BIRD2023[n][1] <= train[n][1], n
    assert tuple(D.Z_LIMITS) == (hp.Z_GRID[0], hp.Z_GRID[-1]) == (2.0, 5.4)
    zu = (np.asarray(hp.Z_GRID) - D.Z_LIMITS[0]) / (D.Z_LIMITS[1] - D.Z_LIMITS[0])
    assert zu[0] == 0.0 and zu[-1] == 1.0 and np.all(np.diff(zu) > 0)

    pts = _design_points()
    assert pts.shape == (60, 9)
    lo_t, hi_t = np.asarray(D.PARAM_LIMITS).T
    lo_s, hi_s = np.asarray(D.SAMPLING_LIMITS).T
    lo_b, hi_b = np.array([BIRD2023[n] for n in NAMES]).T
    assert np.all((pts >= lo_t) & (pts <= hi_t))
    assert np.all((D.normalize_params(pts) >= 0) & (D.normalize_params(pts) <= 1))
    assert np.all((pts[:48] >= lo_b) & (pts[:48] <= hi_b))
    assert np.all(np.any((pts[48:] < lo_s) | (pts[48:] > hi_s), axis=1))
    assert np.all(np.any((pts[48:] < lo_b) | (pts[48:] > hi_b), axis=1))

    ex = _example()
    assert ex.BOXES == FERNANDEZ2024
    for n in ("herei", "heref"):
        assert ex.BOXES[n] == samp[n] and ex.BOXES[n] != train[n]
    assert ex.BOXES["alphaq"][1] == train["alphaq"][1] > samp["alphaq"][1]
    assert ex.BOXES["ns"][1] == train["ns"][1] > hp.NS_RIDGE
    for n in NAMES:
        assert train[n][0] <= ex.BOXES[n][0] < ex.BOXES[n][1] <= train[n][1], n
    for n, sd in FERNANDEZ2024_GAUSSIAN_SD.items():
        assert ex.GAUSSIAN_TERMS[n][1] == pytest.approx(sd / math.sqrt(2.0), rel=1e-15)
