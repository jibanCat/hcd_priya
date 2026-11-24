"""The unfiltered class spectra of a prediction (``P_class_unfiltered``, ``template_unfiltered``) on synthetic bundles.

The networks are untrained and the normalizers made up, so the values mean nothing. The tests check the contract:
the shapes for one redshift and for a stack; the clean row is the filtered clean spectrum; the three HCD rows are the
filtered spectra plus the members' mean ``delta`` output, evaluated here independently of the predictor; the added
term is the low-fidelity one under both fidelities; a requested ``k`` is served by the same operation as
``P_class``; a value that is not positive is flagged and never clipped. No test reads the wording of a label.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator.data import Z_LIMITS, normalize_params
from tests.test_predictor_mf import write_mf

PARAMS = {
    "ns": 0.95,
    "Ap": 2.0e-9,
    "herei": 3.8,
    "heref": 2.9,
    "alphaq": 1.8,
    "hub": 0.7,
    "omegamh2": 0.143,
    "hireionz": 7.0,
    "bhfeedback": 0.05,
}
N_K = RT.SYNTHETIC_ARCH["n_k"]
# one mean-flux input per redshift used here, each inside the range the stand-in model states as trained
TAU = {2.6: 0.25, 3.0: 0.4, 3.4: 0.5}
# The spectra here are of order 0.1 and the added terms of order 1, so a sum or a difference of two of them carries
# an absolute rounding error near 1e-16. This absolute tolerance is far above that and far below any value compared.
ATOL = 1e-13


@pytest.fixture(scope="module")
def info(tmp_path_factory: pytest.TempPathFactory) -> RT.SyntheticBundle:
    """A synthetic bundle of three members with the default made-up normalizer."""
    return RT.write_synthetic_bundle(str(tmp_path_factory.mktemp("u") / "bundle"), n_members=3)


@pytest.fixture(scope="module")
def emu(info: RT.SyntheticBundle) -> hp.Bundle:
    """The predictor of that bundle, without a multi-fidelity record."""
    return hp.load_bundle(info.bundle_dir, info.manifest_sha256)


@pytest.fixture(scope="module")
def emu_mf(info: RT.SyntheticBundle, tmp_path_factory: pytest.TempPathFactory) -> hp.Bundle:
    """The predictor of the same bundle with a synthetic multi-fidelity record of the class spectra."""
    mf_dir = tmp_path_factory.mktemp("um") / "mf"
    digest = write_mf(mf_dir, manifest=info.manifest_sha256, cache=info.lineage["cache"]["sha256"])
    return hp.load_bundle(
        info.bundle_dir, info.manifest_sha256, mf_dir=str(mf_dir), mf_record_sha256=digest
    )


def members_delta(info: RT.SyntheticBundle, z: float, tau: float) -> np.ndarray:
    """The members' mean added term at one input, evaluated without the predictor.

    Parameters
    ----------
    info : RT.SyntheticBundle
        The bundle whose members are loaded with the format's own loader.
    z : float
        A snapshot redshift.
    tau : float
        The mean-flux input.

    Returns
    -------
    numpy.ndarray
        Shape (3, K): the mean over the members of sinh(delta * std + mean), for LLS, subDLA and DLA.
    """
    ens, _, norm = H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    theta = np.asarray(normalize_params(np.array([PARAMS[n] for n in hp.PARAM_NAMES])), float)
    x = jnp.concatenate(
        [jnp.asarray(theta), jnp.asarray([(z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])])]
    )
    mean = np.asarray(norm["delta"]["mean"], dtype=np.float64)
    std = np.asarray(norm["delta"]["std"], dtype=np.float64)
    per_member = [
        np.sinh(np.asarray(m(x, tau)["delta"], np.float64) * std + mean) for m in ens.members
    ]
    return np.mean(per_member, axis=0)


def test_the_shapes_for_one_redshift_and_for_a_stack(emu: hp.Bundle) -> None:
    """One redshift gives (4, K) and (3, K); a stack gains a leading axis and equals the single predictions."""
    pred = emu.predict(PARAMS, 3.0, tau_eff=TAU[3.0])
    assert pred.P_class_unfiltered.shape == (4, N_K) and pred.template_unfiltered.shape == (3, N_K)
    for array in (pred.P_class_unfiltered, pred.template_unfiltered):
        assert array.dtype == np.float64 and not array.flags.writeable
    zs = [2.6, 3.0, 3.4]
    stack = emu.predict_many(PARAMS, zs, tau_eff=[TAU[z] for z in zs])
    assert stack.P_class_unfiltered.shape == (3, 4, N_K)
    assert stack.template_unfiltered.shape == (3, 3, N_K)
    for i, z in enumerate(zs):
        one = emu.predict(PARAMS, z, tau_eff=TAU[z])
        np.testing.assert_allclose(
            stack.P_class_unfiltered[i], one.P_class_unfiltered, rtol=1e-12, atol=ATOL
        )
        np.testing.assert_allclose(
            stack.template_unfiltered[i], one.template_unfiltered, rtol=1e-12, atol=ATOL
        )


def test_the_clean_row_is_the_filtered_clean_spectrum(emu: hp.Bundle) -> None:
    """Clean sightlines hold no filtered pixel: row 0 is ``P_class[0]``, value for value."""
    pred = emu.predict(PARAMS, 3.0, tau_eff=TAU[3.0])
    assert np.array_equal(pred.P_class_unfiltered[0], pred.P_class[0])
    stack = emu.predict_many(PARAMS, [2.6, 3.0], tau_eff=[TAU[2.6], TAU[3.0]])
    assert np.array_equal(stack.P_class_unfiltered[:, 0], stack.P_class[:, 0])


def test_the_hcd_rows_are_the_filtered_spectra_plus_the_members_delta(
    emu: hp.Bundle, info: RT.SyntheticBundle
) -> None:
    """Rows 1 to 3 are ``P_class`` plus the mean over the members of their ``delta`` output in physical units,
    and ``template_unfiltered`` is those rows over the clean spectrum, minus one."""
    for z in (2.6, 3.0):
        pred = emu.predict(PARAMS, z, tau_eff=TAU[z])
        added = members_delta(info, z, TAU[z])
        assert np.max(np.abs(added)) > 1e-3  # the added term is not negligible here
        np.testing.assert_allclose(
            pred.P_class_unfiltered[1:] - pred.P_class[1:], added, rtol=1e-11, atol=ATOL
        )
        np.testing.assert_allclose(
            pred.template_unfiltered,
            pred.P_class_unfiltered[1:] / pred.P_class[0] - 1.0,
            rtol=1e-13,
            atol=ATOL,
        )


def test_the_added_term_is_the_low_fidelity_one_under_both_fidelities(
    emu: hp.Bundle, emu_mf: hp.Bundle, info: RT.SyntheticBundle
) -> None:
    """Under LF+MF the filtered spectra carry the correction and the added term does not: the two fidelities differ
    by exactly what ``P_class`` differs by, and a flag says that the added term is the low-fidelity one.
    """
    z = 3.0
    lf = emu_mf.predict(PARAMS, z, tau_eff=TAU[z], fidelity="LF")
    mf = emu_mf.predict(PARAMS, z, tau_eff=TAU[z], fidelity="LF+MF")
    assert (
        np.max(np.abs(mf.P_class / lf.P_class - 1.0)) > 1e-6
    )  # the correction does act on P_class
    added = members_delta(info, z, TAU[z])
    for pred in (lf, mf):
        np.testing.assert_allclose(
            pred.P_class_unfiltered[1:] - pred.P_class[1:], added, rtol=1e-11, atol=ATOL
        )
    np.testing.assert_allclose(
        mf.P_class_unfiltered - lf.P_class_unfiltered,
        mf.P_class - lf.P_class,
        rtol=1e-11,
        atol=ATOL,
    )
    assert mf.flags["unfiltered_addback_is_lf"] is True
    assert lf.flags["unfiltered_addback_is_lf"] is False
    assert emu.predict(PARAMS, z, tau_eff=TAU[z]).flags["unfiltered_addback_is_lf"] is False


def test_a_requested_k_is_served_as_p_class_is(emu: hp.Bundle) -> None:
    """At requested wavenumbers the field is the linear interpolation in k of its values on the modes, the operation
    that serves ``P_class`` there."""
    z = 3.4
    pred = emu.predict(PARAMS, z, tau_eff=TAU[z])
    k = np.array(
        [pred.k[0], 0.5 * (pred.k[2] + pred.k[3]), 0.25 * pred.k[5] + 0.75 * pred.k[6], pred.k[-1]]
    )
    at_k = emu.predict(PARAMS, z, tau_eff=TAU[z], k=k)
    want = np.stack([np.interp(k, pred.k, pred.P_class_unfiltered[c]) for c in range(4)])
    assert at_k.P_class_unfiltered.shape == (4, k.size)
    np.testing.assert_allclose(at_k.P_class_unfiltered, want, rtol=1e-12, atol=ATOL)
    np.testing.assert_allclose(
        at_k.template_unfiltered,
        at_k.P_class_unfiltered[1:] / at_k.P_class[0] - 1.0,
        rtol=1e-13,
        atol=ATOL,
    )
    # in a stack each redshift has its own span of modes, so each row of k is taken inside its own span
    k_other = emu.predict(PARAMS, 3.0, tau_eff=TAU[3.0]).k[1:5]
    stack = emu.predict_many(PARAMS, [3.0, z], tau_eff=[TAU[3.0], TAU[z]], k=np.stack([k_other, k]))
    np.testing.assert_allclose(stack.P_class_unfiltered[1], want, rtol=1e-12, atol=ATOL)


def test_the_flags_and_the_label_exist(emu: hp.Bundle) -> None:
    """Every prediction carries the three flags as True or False and one label for the field. No held-out range is
    carried for the field, and a flag says so."""
    pred = emu.predict(PARAMS, 3.0, tau_eff=TAU[3.0])
    for name in (
        "unfiltered_addback_is_lf",
        "unfiltered_nonpositive",
        "unfiltered_validation_available",
    ):
        assert type(pred.flags[name]) is bool, name
    assert pred.flags["unfiltered_validation_available"] is False
    label = pred.labels["P_class_unfiltered"]
    assert isinstance(label, str) and label.strip()


def test_a_value_that_is_not_positive_is_flagged_and_never_clipped(
    emu: hp.Bundle, info: RT.SyntheticBundle, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The sum of a filtered spectrum and a negative added term can be zero or negative. Such a value is returned
    as it is, with a flag; where every value is positive the flag is False."""
    z = 3.0
    pred = emu.predict(PARAMS, z, tau_eff=TAU[z])
    added = members_delta(info, z, TAU[z])
    assert np.any(pred.P_class[1:] + added <= 0)  # the made-up normalizer gives negative sums
    assert pred.flags["unfiltered_nonpositive"] is True
    assert np.min(pred.P_class_unfiltered) < 0  # returned, not clipped
    np.testing.assert_allclose(
        pred.P_class_unfiltered[1:], pred.P_class[1:] + added, rtol=1e-11, atol=ATOL
    )
    # a normalizer whose added term is positive at every mode: sinh(0.5) with almost no spread
    norm = RT.synthetic_norm(N_K)
    norm["delta"] = {"mean": np.full((3, N_K), 0.5), "std": np.full((3, N_K), 0.01)}
    positive = RT.write_synthetic_bundle(
        str(tmp_path_factory.mktemp("up") / "bundle"), n_members=2, norm=norm
    )
    pred_positive = hp.load_bundle(positive.bundle_dir, positive.manifest_sha256).predict(
        PARAMS, z, tau_eff=TAU[z]
    )
    assert np.all(pred_positive.P_class_unfiltered > 0)
    assert pred_positive.flags["unfiltered_nonpositive"] is False
