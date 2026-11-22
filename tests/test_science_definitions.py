"""The scientific definitions of hcd_priya, checked against the released code. Numerical and behavioural tests.

No page is searched for approved words here (that is tests/test_docs_integrity.py, which is lint). Each test compares
a number or a behaviour of the released code with an independent statement written in the test and, where a retained
page prints that number, with the number read from the page. A page that prints another number fails.

Definitions that existing tests already establish are not repeated:

- the native modes, k_n = 2 pi n / L (1 + z) / (100 E(z)):
  tests/test_science_conventions.py::test_the_native_modes_follow_the_box_convention_in_s_per_km;
- the parameter boxes against the published PRIYA ranges: tests/test_science_parameter_boxes.py;
- the flags and the refusals of a prediction: tests/test_predictor.py (outside_training_box,
  params_outside_training_box, z_downweighted, k_downweighted), tests/test_predictor_mf.py (test_mf_domain_flags,
  test_mf_above_the_validated_range_is_served_unchanged_but_flagged,
  test_mf_below_the_tables_lowest_redshift_is_refused_not_clamped,
  test_mf_below_validated_z_min_is_refused_even_inside_the_tables) and tests/test_likelihood.py (a redshift that is
  not a snapshot is refused).

The number of modes of the published model is not tested here: it lives only in the trained model, which the
pull-request environment does not hold.

Every number is synthetic or a published constant. Needs the repository: it reads four of its pages.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import catalog
from hcd_analysis import p1d as catalogue_p1d
from hcd_analysis.emulator import data as D
from hcd_analysis.emulator import kcoord as K
from hcd_analysis.emulator import mf_release as MR
from tests.tau0_world import use_fake_spectra_stand_in
from tests.test_science_parameter_boxes import DESIGN_FILE_LIMITS

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------- independent constants
L_BOX = 120.0  # Mpc/h, the PRIYA box (Bird et al. 2023)
# log10 of N_HI in cm^-2, the lower edge of each class
CLASS_EDGES = {"LLS": 17.2, "subDLA": 19.0, "DLA": 20.3}
# the redshifts of the grouped validation of the multi-fidelity correction
VALIDATION_Z = (2.2, 4.6)
K_MIN_FULL_WEIGHT = 1e-3  # s/km; training gave less weight to wavenumbers below it
# the first and the last PRIYA snapshot redshift; the snapshots are 0.2 apart
SNAPSHOT_Z = (2.0, 5.4)
Z, HUB, OMEGAMH2 = 3.0, 0.70, 0.143  # one redshift and one cosmology inside the training box


def _page(path: str) -> str:
    """Read a page of the repository with its line breaks and repeated spaces turned into single spaces.

    Parameters
    ----------
    path : str
        The page, relative to the root of the repository.

    Returns
    -------
    str
        The text of the page on one line, so that a sentence broken over two lines can be matched.
    """
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def _between(text: str, start: str, end: str) -> str:
    """The part of a text from one phrase to the next occurrence of another.

    Parameters
    ----------
    text : str
        A page, as ``_page`` returns it.
    start, end : str
        The phrase the part begins with and the phrase it ends with. Both must occur, ``end`` after ``start``.

    Returns
    -------
    str
        The part, both phrases included.
    """
    assert (
        start in text
    ), f"the phrase {start!r} is not in the page: has the sentence been reworded?"
    begin = text.index(start)
    assert end in text[begin:], f"the phrase {end!r} does not follow {start!r} in the page"
    return text[begin : text.index(end, begin) + len(end)]


# ------------------------------------------------------------------------------------------- the two wavenumbers
def test_the_predictor_wavenumber_is_angular() -> None:
    """A sine with wavelength lambda in velocity is the mode with k lambda = 2 pi of the predictor, not with
    k lambda = 1; the predictor refuses a grid of modes written the other way."""
    width = float(K.vbox_kms(Z, HUB, OMEGAMH2))  # the velocity width of the box, km/s
    n_pix, mode = 4096, 7
    wavelength = width / mode
    velocity = np.arange(n_pix) * width / n_pix
    wave = np.sin(2.0 * np.pi * velocity / wavelength)
    assert int(np.argmax(np.abs(np.fft.rfft(wave)))) == mode  # the sine is mode 7 of the box

    k_com = 2.0 * np.pi * np.arange(1, 17) / L_BOX  # the comoving modes of the box, h/Mpc
    k = np.asarray(K.k_skm_from_kcom(k_com, Z, HUB, OMEGAMH2))
    assert k[mode - 1] * wavelength == pytest.approx(2.0 * np.pi, rel=1e-12)
    # the same mode counted in cycles per km/s
    cyclic = np.fft.rfftfreq(n_pix, d=width / n_pix)[mode]
    assert cyclic * wavelength == pytest.approx(1.0, rel=1e-12)
    assert k[mode - 1] / cyclic == pytest.approx(2.0 * np.pi, rel=1e-12)

    K._require_box_modes(k_com)
    with pytest.raises(ValueError, match="2 pi n / L"):
        K._require_box_modes(k_com / (2.0 * np.pi))


def test_the_catalogue_pipeline_wavenumber_is_cyclic_and_differs_by_two_pi() -> None:
    """The P1D files of the catalogue pipeline carry k_n = n / (N dv); times 2 pi it is the predictor's k of the
    same mode of the same box. docs/pipeline.md says so in one sentence."""
    width = float(K.vbox_kms(Z, HUB, OMEGAMH2))
    n_pix = 512
    dv = width / n_pix
    k_catalogue = catalogue_p1d.P1DAccumulator(n_pix, dv)._k_native()
    np.testing.assert_allclose(
        k_catalogue, np.arange(n_pix // 2 + 1) / (n_pix * dv), rtol=1e-13, atol=0
    )

    modes = np.arange(1, 17)
    k_predictor = np.asarray(K.k_skm_from_kcom(2.0 * np.pi * modes / L_BOX, Z, HUB, OMEGAMH2))
    np.testing.assert_allclose(2.0 * np.pi * k_catalogue[modes], k_predictor, rtol=1e-12, atol=0)
    assert "cyclic wavenumber k_n = n / (N dv)" in _page("docs/pipeline.md")


# ------------------------------------------------------------------------------------------- the absorber classes
CLASS_SENTENCES = (
    ("README.md", "LLS at 10^", "cm⁻²."),
    ("README.md", "(10^20.3 cm^-2 and above", ")"),
    ("docs/model_figures.md", "The classes are bins of column density:", "(in cm^-2)"),
    ("docs/pipeline.md", "- LLS: 10^", "- DLA: N_HI ≥ 10^20.3 cm⁻²"),
)


def test_the_class_boundaries_agree_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLS from 10^17.2, subDLA from 10^19, DLA from 10^20.3 cm^-2: in the catalogue code, in the edges of the
    fine column-density bins of the training spectra, in the order of the predictor's classes and in the pages.

    The fine edges are a constant of hcd_analysis.priya_p1d, which imports fake_spectra; where fake_spectra is not
    installed the tests' stand-in for it is used. The constant itself is released code either way.
    """
    edges = (catalog.LOG_NHI_LLS_MIN, catalog.LOG_NHI_SUBDLA_MIN, catalog.LOG_NHI_DLA_MIN)
    assert edges == tuple(CLASS_EDGES.values())
    assert hp.HCD_CLASSES == tuple(CLASS_EDGES)
    for name, edge in CLASS_EDGES.items():
        assert catalog.classify_system(10.0**edge * 1.0001) == name
    assert catalog.classify_system(10.0 ** CLASS_EDGES["LLS"] * 0.9999) == "forest"

    use_fake_spectra_stand_in(monkeypatch)
    import hcd_analysis.priya_p1d as class_spectra

    fine = [float(edge) for edge in class_spectra.FINE_NHI_EDGES]
    assert fine[0] == CLASS_EDGES["LLS"] and set(CLASS_EDGES.values()) <= set(fine)

    for page, start, end in CLASS_SENTENCES:
        sentence = _between(_page(page), start, end)
        printed = sorted(
            {float(exponent) for exponent in re.findall(r"10\^(\d+(?:\.\d+)?)", sentence)}
        )
        assert printed == sorted(CLASS_EDGES.values()), f"{page}: {sentence!r}"


# ------------------------------------------------------------------------------------------- the parameters
NUMBER = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
MIDPOINT_PAGES = (
    ("README.md", r'"{name}":\s*' + NUMBER),
    ("docs/likelihood.md", r'"{name}":\s*' + NUMBER),
    ("docs/model_figures.md", r"\b{name} = " + NUMBER),
)


def test_the_documented_parameter_limits_are_the_predictors() -> None:
    """The predictor's parameter box is the one of the design file, and every value of a parameter that the three
    pages print in an example is the midpoint of that box."""
    assert len(hp.PARAM_NAMES) == len(hp.PARAM_LIMITS) == len(DESIGN_FILE_LIMITS) == 9
    midpoint = {}
    for name, (low, high) in zip(hp.PARAM_NAMES, np.asarray(hp.PARAM_LIMITS, dtype=float)):
        assert (low, high) == DESIGN_FILE_LIMITS[name], name
        midpoint[name] = 0.5 * (low + high)
    for page, pattern in MIDPOINT_PAGES:
        text = _page(page)
        for name in hp.PARAM_NAMES:
            printed = re.findall(pattern.format(name=name), text)
            assert printed, f"{page} prints no value of {name}: has the example been reworded?"
            for value in printed:
                assert float(value) == pytest.approx(
                    midpoint[name], rel=1e-12
                ), f"{page}: {name} = {value}"


# ------------------------------------------------------------------------------------------- the validation domain
VALIDATION_PAGES = (
    "README.md",
    "docs/likelihood.md",
    "docs/model_figures.md",
    "docs/release_format_h5.md",
)


def test_the_validation_domain_and_the_low_k_bound_agree() -> None:
    """The grouped validation covers z 2.2 to 4.6 and training gave less weight below 1e-3 s/km: in the constants
    of the code, and in every sentence of the four pages that prints a number beside those words."""
    assert D.DATA_RANGE == {"z_lo": 2.2, "z_hi": 4.6, "k_min": 1e-3}
    assert (D.DATA_RANGE["z_lo"], D.DATA_RANGE["z_hi"]) == VALIDATION_Z
    assert (MR.VALIDATED_Z_MIN, MR.VALIDATED_Z_MAX) == VALIDATION_Z
    assert D.DATA_RANGE["k_min"] == K_MIN_FULL_WEIGHT
    assert tuple(D.Z_LIMITS) == SNAPSHOT_Z

    ranges, bounds, lists = 0, 0, 0
    for page in VALIDATION_PAGES:
        text = _page(page)
        for sentence in re.split(r"(?<=[.;:])\s", text):
            if re.search(r"grouped[- ]validation", sentence):
                for low, high in re.findall(r"(\d\.\d)(?: to |-)(\d\.\d)", sentence):
                    assert (float(low), float(high)) == VALIDATION_Z, f"{page}: {sentence!r}"
                    ranges += 1
            if "less weight" in sentence:
                for value in re.findall(r"([0-9.]+e-?\d+) s/km", sentence):
                    assert float(value) == K_MIN_FULL_WEIGHT, f"{page}: {sentence!r}"
                    bounds += 1
        for first, second, last in re.findall(r"z = (\d\.\d), (\d\.\d), \.\.\., (\d\.\d)", text):
            assert (float(first), float(last)) == SNAPSHOT_Z and float(second) == pytest.approx(
                2.2
            ), page
            lists += 1
    # the pages do print these numbers: a rewording that hides them all from this test is noticed
    assert ranges >= 4 and bounds >= 2 and lists >= 1, (ranges, bounds, lists)


# ------------------------------------------------------------------------------------------- the mean flux
def test_the_mean_flux_convention_of_the_class_spectra_is_the_whole_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flux contrast of a class is exp(-tau) / F - 1 with F the mean flux of ALL sightlines, not of the class.

    The function under test is the released ``hcd_analysis.priya_p1d._per_class_p1d_at_scale``, the one that the
    constant ``mf_release.MEAN_FLUX_CONVENTION`` names. Limit of this test: the module imports fake_spectra, which
    the pull-request environment does not hold, so the tests' stand-in for fake_spectra is installed there. The
    normalisation (the line ``dflux = np.exp(-tau_eff) / target_F - 1.0``) is then released code, and the Fourier
    transform under it is the stand-in's. Synthetic optical depths.
    """
    assert MR.MEAN_FLUX_CONVENTION["name"] == "global"
    assert MR.MEAN_FLUX_CONVENTION["function"] == "hcd_analysis.priya_p1d._per_class_p1d_at_scale"
    use_fake_spectra_stand_in(monkeypatch)
    import hcd_analysis.priya_p1d as class_spectra

    rng = np.random.default_rng(20261121)
    n_pix, width = 256, 5000.0
    # the sightlines of a class with little absorption, and those of a class with much
    tau_weak = rng.gamma(2.0, 0.15, size=(40, n_pix))
    tau_strong = rng.gamma(2.0, 0.60, size=(10, n_pix))
    flux_all = float(np.mean(np.exp(-np.concatenate([tau_weak, tau_strong]))))
    flux_strong = float(np.mean(np.exp(-tau_strong)))
    assert flux_strong < 0.8 * flux_all  # the two normalisations differ clearly

    def spectrum(contrast: np.ndarray) -> np.ndarray:
        """The mean power of the rows of a flux contrast, written out here; the zero mode is left out.

        Parameters
        ----------
        contrast : np.ndarray
            Flux contrast, one sightline per row.

        Returns
        -------
        np.ndarray
            ``width`` times the mean over the rows of |FFT / n_pix|^2, for the modes 1, 2, ...
        """
        power = (np.abs(np.fft.rfft(contrast, axis=1)) / n_pix) ** 2
        return width * np.mean(power, axis=0)[1:]

    _, computed = class_spectra._per_class_p1d_at_scale(tau_strong, width, 1.0, flux_all)
    whole_sample = spectrum(np.exp(-tau_strong) / flux_all - 1.0)
    own_class = spectrum(np.exp(-tau_strong) / flux_strong - 1.0)
    np.testing.assert_allclose(computed, whole_sample, rtol=1e-12, atol=0)
    assert np.max(np.abs(computed / own_class - 1.0)) > 0.2
