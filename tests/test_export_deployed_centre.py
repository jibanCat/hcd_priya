"""TDD pins for scripts/export_deployed_centre_dndx.py -- the DEPLOYED-vs-CLOSURE key labelling.

DEFECT BEING FIXED (found by the paper agent 2026-07-21, shipped in
notes/artifacts/paper_exports/deployed_centre_2026-07-21/deployed_centre_layers.npz):
the export promised `zslope_closure` + `zslope_deployed`; NEITHER existed. What shipped was a
single UNLABELLED `zslope = [2.465, 2.758, 2.366]` -- the CLOSURE slope -- sitting inside a file
whose entire purpose is the DEPLOYED layer. A reader doing npz['zslope'] to describe the deployed
curve states exactly the error the script's own warning (2) exists to prevent, because the
deployed LLS forward z-slope is HCD_LLS_REALFIT_ZSLOPE = 2.127, not the sim 2.465.

Root cause: a sed edit silently failed to match and nobody verified the resulting key set. So
these tests pin the KEY SET, not just the values, and the script grows a fail-loud postcondition
(`verify_npz_payload`) that re-checks the keys of what it is about to write.

Everything here is built in tmp_path from a tiny synthetic z grid + the (blind-safe) prior-geometry
constants, so the suite passes on a clean checkout with no notes-repo artifact present.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_export_deployed_centre.py -q
"""
import importlib.util
import os

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE
from hcd_analysis.emulator.inference import (
    HCD_LLS_REALFIT_ZSLOPE,
    HCD_LLS_SURVEY_BOOST,
    HCD_LLS_SURVEY_FRAC_SIGMA,
    HCD_Z_PIVOT,
)

REPO = "/home/mfho/hcd_priya"

# --- tiny synthetic inputs: 4 z nodes, real Xbar(z) magnitudes (the z=3 node carries the true
# pivot Xbar so the deployed LLS centre lands on the deployed 0.172112 and the pivot guard fires
# on a genuine value). Prior geometry only -- no data, no posterior, no cosmology.
# The grid stops at z=3.4: under the v3 EXACT inverse (readout defect B, 2026-07-22) the
# current deployed KS prior's hi1 edge leaves the occupancy simplex around z~3.8 (and the KS
# CENTRE at z>=~4.45), at which the exporter now correctly REFUSES instead of silently
# saturating -- that refusal is pinned separately by test_ks_refusal_at_high_z_is_fail_loud.
Z_TINY = np.array([2.2, 2.6, 3.0, 3.4], float)
XB_TINY = np.array([0.4020049401674796, 0.5101543263532812, 0.6316034425658955,
                    0.7663522888053229], float)
MU_CLOSURE = np.array([0.18811862664368107, 0.06218315972222222, 0.004409670138888889], float)
SD_CLOSURE = np.array([0.028217793996552160, 0.02487326388888889, 0.0022048350694444446], float)


def _load_script(name):
    """Load a scripts/<name>.py module by path (scripts/ is not an importable package)."""
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_script("export_deployed_centre_dndx")


@pytest.fixture(scope="module")
def built():
    """(payload, table) from the module's own assembly on the tiny grid."""
    closure_center = MOD.band_curve(MU_CLOSURE, Z_TINY, XB_TINY, float(HCD_Z_PIVOT),
                                    np.asarray(HCD_INCIDENCE_SLOPE, float))
    return MOD.build_npz_payload(Z_TINY, XB_TINY, MU_CLOSURE, SD_CLOSURE,
                                 float(HCD_Z_PIVOT), closure_center)


@pytest.fixture(scope="module")
def npz(built, tmp_path_factory):
    """Round-trip the payload through a real npz in tmp_path -- keys are asserted on the RELOAD,
    which is the artifact surface the paper agent actually sees."""
    payload, _ = built
    p = tmp_path_factory.mktemp("dep") / "deployed_centre_layers.npz"
    np.savez(p, **payload)
    return np.load(p, allow_pickle=True)


# --------------------------------------------------------------------------- the defect itself
def test_no_bare_zslope_key(npz):
    """THE DEFECT. A bare `zslope` inside a deployed-layer file is the ambiguity; removing it is
    the point of the fix. Renaming without deleting would leave the trap in place."""
    assert "zslope" not in npz.files, (
        f"bare 'zslope' still shipped (= {np.asarray(npz['zslope'])!r}); a reader cannot tell "
        f"whether it labels the closure or the deployed curve. keys={sorted(npz.files)}")


def test_both_slope_keys_exist(npz):
    for k in ("zslope_closure", "zslope_deployed"):
        assert k in npz.files, f"missing promised key {k!r}; keys={sorted(npz.files)}"


def test_closure_slope_is_the_sim_incidence_slope(npz):
    np.testing.assert_array_equal(np.asarray(npz["zslope_closure"]),
                                  np.asarray(HCD_INCIDENCE_SLOPE, float))
    assert float(np.asarray(npz["zslope_closure"])[0]) == 2.465


def test_deployed_slope_is_the_realfit_lls_slope(npz):
    dep = np.asarray(npz["zslope_deployed"])
    assert float(dep[0]) == 2.127 == float(HCD_LLS_REALFIT_ZSLOPE)


def test_the_two_slopes_are_not_the_same_array(npz):
    """A test that would pass if someone set both keys to the same array is worthless: the whole
    point is that the deployed curve differs in z-SHAPE, not just normalisation."""
    clo = np.asarray(npz["zslope_closure"])
    dep = np.asarray(npz["zslope_deployed"])
    assert not np.array_equal(clo, dep), "zslope_closure and zslope_deployed are identical"
    assert abs(float(dep[0]) - float(clo[0])) > 0.3, (
        f"LLS slopes should differ by 2.465-2.127=0.338, got {float(clo[0])} vs {float(dep[0])}")


def test_deployed_subdla_dla_slopes_equal_closure(npz):
    """Only the LLS row is re-centred/re-sloped on the real fit; subDLA/DLA stay on the sim slope."""
    clo = np.asarray(npz["zslope_closure"])
    dep = np.asarray(npz["zslope_deployed"])
    np.testing.assert_array_equal(dep[1:], clo[1:])


def test_deployed_lls_curve_was_built_with_the_deployed_slope(npz):
    """The label must describe the ARRAY. Rebuild the deployed LLS curve both ways and assert the
    shipped one matches the DEPLOYED slope and NOT the closure slope -- this is what catches a
    future 'keys renamed, arrays still built from the closure slope' regression."""
    zg = np.asarray(npz["zgrid"]); xb = np.asarray(npz["xbar_grid"])
    zp = float(npz["z_pivot"])
    for sv in [str(s) for s in npz["survey_order"]]:
        mu = np.asarray(npz[f"deployed_alpha_pivot_mu_{sv}"])
        want = MOD.band_curve(mu, zg, xb, zp, np.asarray(npz["zslope_deployed"]))
        wrong = MOD.band_curve(mu, zg, xb, zp, np.asarray(npz["zslope_closure"]))
        got = np.asarray(npz[f"deployed_center_dndx_{sv}"])
        np.testing.assert_allclose(got[:, 0], want[:, 0], rtol=0, atol=1e-14)
        assert not np.allclose(got[:, 0], wrong[:, 0], rtol=0, atol=1e-10), (
            f"{sv}: deployed LLS curve is indistinguishable from the closure-slope build")


# ------------------------------------------------- the paper agent's load-bearing overlay claim
def test_subdla_dla_dndx_renorm_coupled_to_closure(npz):
    """LLS-only overlay, v3 exact-inverse form. Pre-2026-07-22 this pinned BIT-identity of the
    deployed subDLA/DLA dN/dX columns to the closure layer -- true for the old approximate
    inverse, whose telescoping made columns 1,2 independent of the LLS weight. Under
    `alpha_to_dndx_exact` (readout defect B migration) the renorm undo Z couples EVERY class
    to the LLS alpha, which is the only re-centred/re-sloped class, so the honest pin is:
    the sub/DLA columns agree to renorm-coupling level (<1e-2 relative; measured ~4.3e-3
    worst case for the KS boost-2.5 layer at z=3.4, ~3e-4 for eBOSS/DESI -- a genuine
    centre/slope mislabel is a >=10% effect and still fails this) and are NOT
    bit-identical (the coupling is real and nonzero)."""
    clo = np.asarray(npz["closure_center_dndx"])
    for sv in [str(s) for s in npz["survey_order"]]:
        got = np.asarray(npz[f"deployed_center_dndx_{sv}"])
        for j, cname in ((1, "subDLA"), (2, "DLA")):
            rel = np.max(np.abs(got[:, j] / clo[:, j] - 1.0))
            assert rel < 1e-2, (
                f"{sv} {cname} centre differs from closure by {rel:.3e} relative -- far above "
                f"the exact-inverse renorm coupling; centre/slope mislabel suspected")
            assert not np.array_equal(got[:, j], clo[:, j]), (
                f"{sv} {cname} column is bit-identical to closure -- under the exact inverse "
                f"the renorm coupling to the (re-centred) LLS alpha must be nonzero; was this "
                f"array built with the deprecated approximate alpha_to_dndx?")
        assert not np.array_equal(got[:, 0], clo[:, 0]), f"{sv} LLS centre should differ"


# ------------------------------------------------------- other unlabelled / misleading key axes
def test_no_bare_survey_ordered_vectors(npz):
    """`boost` / `lls_frac_sigma` are length-3 per-SURVEY vectors in a file whose class_order is
    also length 3 -- indistinguishable from per-class arrays. They must carry the axis in the name."""
    for bad in ("boost", "lls_frac_sigma"):
        assert bad not in npz.files, f"axis-ambiguous key {bad!r} still shipped"
    sur = [str(s) for s in npz["survey_order"]]
    np.testing.assert_allclose(np.asarray(npz["boost_by_survey"]),
                               [HCD_LLS_SURVEY_BOOST[s] for s in sur])
    np.testing.assert_allclose(np.asarray(npz["lls_frac_sigma_by_survey"]),
                               [HCD_LLS_SURVEY_FRAC_SIGMA[s] for s in sur])


def test_alpha_pivot_keys_say_deployed_or_closure(npz):
    """`alpha_pivot_mu_<sv>` collides in meaning with the frozen f3 layer's `alpha_pivot_mu`
    (which is the CLOSURE pivot). Every pivot array must name its provenance."""
    sur = [str(s) for s in npz["survey_order"]]
    for sv in sur:
        for bad in (f"alpha_pivot_mu_{sv}", f"alpha_pivot_sigma_{sv}"):
            assert bad not in npz.files, f"unprefixed key {bad!r} still shipped"
        assert f"deployed_alpha_pivot_mu_{sv}" in npz.files
        assert f"deployed_alpha_pivot_sigma_{sv}" in npz.files
    for k in ("closure_alpha_pivot_mu", "closure_alpha_pivot_sigma"):
        assert k in npz.files, f"closure pivot geometry {k!r} not shipped for a like-for-like read"
    np.testing.assert_allclose(np.asarray(npz["closure_alpha_pivot_mu"]), MU_CLOSURE)
    # the deployed LLS centre is the corrected lit law x boost, NOT closure x boost
    for sv in sur:
        mu = np.asarray(npz[f"deployed_alpha_pivot_mu_{sv}"])
        b = float(HCD_LLS_SURVEY_BOOST[sv])
        assert abs(mu[0] - 0.17211216067355312 * b) < 1e-12, f"{sv} deployed LLS centre {mu[0]!r}"
        assert abs(mu[0] - MU_CLOSURE[0] * b) > 1e-3, (
            f"{sv} deployed LLS centre equals closure x boost -- that is the construction the "
            f"script's warning (1) exists to prevent")
        np.testing.assert_array_equal(mu[1:], MU_CLOSURE[1:])


# ------------------------------------------------------------------ the sidecar, same class of ambiguity
def test_sidecar_table_has_no_bare_zslope(built):
    _, table = built
    for sv, row in table.items():
        assert "zslope" not in row, (
            f"PROVENANCE prior_geometry_table[{sv}] still has a bare 'zslope' (= the DEPLOYED "
            f"slope there, while the npz's bare 'zslope' was the CLOSURE slope -- the same name "
            f"meant opposite things in the same export dir)")
        assert row["zslope_deployed"][0] == 2.127
        assert row["zslope_closure"][0] == 2.465
        assert row["zslope_deployed"][1:] == row["zslope_closure"][1:]


def test_sidecar_dndx_scalar_names_its_class_and_layer(built):
    _, table = built
    for sv, row in table.items():
        assert "dndx_center_z3" not in row, (
            "'dndx_center_z3' does not say it is the LLS column of the DEPLOYED curve")
        assert "dndx_lls_center_z3_deployed" in row
        assert "alpha_pivot_mu_deployed" in row and "alpha_pivot_sigma_deployed" in row
        assert "alpha_pivot_mu" not in row and "alpha_pivot_sigma" not in row


# --------------------------------------------------------- the postcondition that stops a repeat
def test_verify_npz_payload_rejects_a_bare_zslope():
    """Root-cause guard: the export must re-read its own key set and refuse to write an
    ambiguous one, so a silently-failed edit can never ship again."""
    payload = {"zslope_closure": np.asarray(HCD_INCIDENCE_SLOPE, float),
               "zslope_deployed": np.array([2.127, 2.758, 2.366]),
               "zslope": np.asarray(HCD_INCIDENCE_SLOPE, float)}
    with pytest.raises(AssertionError, match="zslope"):
        MOD.verify_npz_payload(payload)


def test_verify_npz_payload_rejects_identical_slopes():
    payload = {"zslope_closure": np.asarray(HCD_INCIDENCE_SLOPE, float),
               "zslope_deployed": np.asarray(HCD_INCIDENCE_SLOPE, float)}
    with pytest.raises(AssertionError):
        MOD.verify_npz_payload(payload)


def test_verify_npz_payload_accepts_the_real_payload(built, tmp_path):
    payload, _ = built
    MOD.verify_npz_payload(payload)
    p = tmp_path / "x.npz"
    np.savez(p, **payload)
    MOD.verify_npz_payload(dict(np.load(p, allow_pickle=True)))   # survives the round trip


# ----------------------------------------- the v3 exact-inverse fail-loud refusal (KS, high z)
def test_ks_refusal_at_high_z_is_fail_loud():
    """At the CURRENT deployed KS prior the KS centre leaves the occupancy simplex at
    z >= ~4.45 (sum(alpha) ~ 1.047 at z=4.6). The v3 exporter must REFUSE, naming the survey
    and the pending KS prior reparameterization -- NOT silently saturate the LLS class at
    27.631021/Xbar as the pre-v3 approximate inverse did (which is what corrupted shipped
    artifacts). Do NOT 'fix' this by clipping/masking in the exporter: the refusal is the
    contract until the KS prior reparameterization lands."""
    z_hi = np.array([2.2, 3.0, 4.6], float)
    xb_hi = np.array([0.4020049401674796, 0.6316034425658955, 1.253], float)
    closure_center = MOD.band_curve(MU_CLOSURE, z_hi, xb_hi, float(HCD_Z_PIVOT),
                                    np.asarray(HCD_INCIDENCE_SLOPE, float))
    with pytest.raises(ValueError, match="REFUSED for survey=KS") as ei:
        MOD.build_npz_payload(z_hi, xb_hi, MU_CLOSURE, SD_CLOSURE, float(HCD_Z_PIVOT),
                              closure_center)
    assert "reparameterization" in str(ei.value)
    assert "occupancy simplex" in str(ei.value)
