"""TDD pins for scripts/export_deployed_centre_dndx.py -- the DEPLOYED-vs-CLOSURE key labelling
(v1 defect) + the v4 PARAMETERIZATION-axis semantics (native-dN/dX mapped KS).

DEFECT LINEAGE:
  v1 (2026-07-21, shipped deployed_centre_2026-07-21): the export promised `zslope_closure` +
    `zslope_deployed`; NEITHER existed -- a single UNLABELLED `zslope` (= the CLOSURE slope)
    shipped inside a DEPLOYED-layer file. Root cause: a silently-failed sed edit whose key set
    nobody re-read. These tests pin the KEY SET, and the script carries a fail-loud
    postcondition (`verify_npz_payload`).
  v2 (2026-07-22, shipped deployed_centre_2026-07-22): built with the APPROXIMATE
    `alpha_to_dndx`, whose _LOG_FLOOR clip silently SATURATED the out-of-simplex KS LLS column
    at 27.631021/Xbar (up to ~33x the lit law at z>=4.5) -- silent-clamp garbage in a shipped
    paper artifact.
  v3 (never shipped): exact inverse, mode="raise" -- correctly REFUSED for KS (the legacy KS
    alpha centre leaves the occupancy simplex at z>=~4.45), pending the KS reparameterization.
  v4 (this file's pins): the KS prior reparameterization landed (W2 2026-07-22,
    inference.HCD_ALPHA_PARAMETERIZATION['KS'] = 'dndx_mapped_v2'): KS samples IN dN/dX SPACE
    around a deterministic reference, so the deployed KS centre curve IS that reference
    (closure_legb._ks_dndx_reference) -- native dN/dX, NO inverse map, NO saturation possible,
    no refusal. The W2-era "exporter still writes the LEGACY KS surface" tripwire has FLIPPED:
    the KS row is now pinned AS the mapped reference (alpha-space pivot 0.393235480, moved from
    the legacy postmap 0.4302804), and an alpha-space Gaussian width for KS is a REFUSED key.

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
from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE, _ks_dndx_reference
from hcd_analysis.emulator import inference as INF
from hcd_analysis.emulator.inference import (
    HCD_LLS_REALFIT_ZSLOPE,
    HCD_LLS_SURVEY_BOOST,
    HCD_LLS_SURVEY_FRAC_SIGMA,
    HCD_Z_PIVOT,
)

REPO = "/home/mfho/hcd_priya"

# --- tiny synthetic inputs: 6 z nodes, real Xbar(z) magnitudes = EXACT evaluations of the frozen
# deg-2 Xbar polynomial (f3_layers 'xbar_poly_deg2_coeffs' [0.04156166, 0.07087752, 0.04491599])
# so build_npz_payload's polyfit recovery is exact and the z=3 node carries the true pivot Xbar
# (the deployed LLS centre lands on 0.172112 / the mapped KS pivot on 0.393235, and the pivot
# guards fire on genuine values). Prior geometry only -- no data, no posterior, no cosmology.
# The grid RUNS TO z=4.6 on purpose: v3 correctly REFUSED here at the legacy KS prior (centre
# out-of-simplex at z>=~4.45); under the v4 native-dN/dX KS the full grid must BUILD, with the
# KS LLS column equal to the boosted lit law everywhere and no 27.631021/Xbar saturation
# fingerprint anywhere (the v1/v2 shipped-artifact defect).
Z_TINY = np.array([2.2, 2.6, 3.0, 3.4, 4.2, 4.6], float)
XB_TINY = np.array([0.4020049401674796, 0.5101543263532812, 0.6316034425658955,
                    0.7663522888053229, 1.075749171364616, 1.2503972076844816], float)
MU_CLOSURE = np.array([0.18811862664368107, 0.06218315972222222, 0.004409670138888889], float)
SD_CLOSURE = np.array([0.028217793996552160, 0.02487326388888889, 0.0022048350694444446], float)

# saturation fingerprint of the retired approximate inverse: mu = -log(1e-12) = 27.631021
SATURATION_MU = 27.631021
# the W2-pinned mapped KS alpha-space LLS pivot centre (legacy postmap was 0.4302804016838828)
KS_MAPPED_PIVOT = 0.393235480
KS_LEGACY_PIVOT = 0.4302804016838828


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


# --------------------------------------------------------------------------- the v1 defect itself
def test_no_bare_zslope_key(npz):
    """THE v1 DEFECT. A bare `zslope` inside a deployed-layer file is the ambiguity; removing it
    is the point of the fix. Renaming without deleting would leave the trap in place."""
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


def test_deployed_lls_curve_was_built_with_its_own_parameterization(npz):
    """The label must describe the ARRAY, per parameterization. DESI/eBOSS: rebuild the LLS
    curve both ways and assert the shipped one matches the DEPLOYED slope and NOT the closure
    slope. KS (v4 mapped): the LLS column must BE the dndx-premap boosted lit law
    boost * A * (1+z)^gamma -- and must NOT be a band_curve alpha build of any slope."""
    zg = np.asarray(npz["zgrid"]); xb = np.asarray(npz["xbar_grid"])
    zp = float(npz["z_pivot"])
    par = {str(s): str(p) for s, p in zip(npz["survey_order"],
                                          npz["parameterization_by_survey"])}
    boost = {str(s): float(b) for s, b in zip(npz["survey_order"], npz["boost_by_survey"])}
    for sv in [str(s) for s in npz["survey_order"]]:
        got = np.asarray(npz[f"deployed_center_dndx_{sv}"])
        if par[sv] == "alpha_pivot_powerlaw_v1":
            mu = np.asarray(npz[f"deployed_alpha_pivot_mu_{sv}"])
            want = MOD.band_curve(mu, zg, xb, zp, np.asarray(npz["zslope_deployed"]))
            wrong = MOD.band_curve(mu, zg, xb, zp, np.asarray(npz["zslope_closure"]))
            np.testing.assert_allclose(got[:, 0], want[:, 0], rtol=0, atol=1e-14)
            assert not np.allclose(got[:, 0], wrong[:, 0], rtol=0, atol=1e-10), (
                f"{sv}: deployed LLS curve is indistinguishable from the closure-slope build")
        else:
            assert par[sv] == "dndx_mapped_v2" and sv == "KS"
            A, g = [float(v) for v in np.asarray(npz["ks_lit_dndx_law_lls"], float)]
            want = boost[sv] * A * (1.0 + zg) ** g
            np.testing.assert_array_equal(got[:, 0], want)


# ------------------------------------------------- the paper agent's load-bearing overlay claim
def test_subdla_dla_dndx_renorm_coupled_to_closure(npz):
    """LLS-only overlay, exact-inverse form. Pre-2026-07-22 this pinned BIT-identity of the
    deployed subDLA/DLA dN/dX columns to the closure layer -- true for the old approximate
    inverse, whose telescoping made columns 1,2 independent of the LLS weight. Under
    `alpha_to_dndx_exact` the renorm undo Z couples EVERY class to the LLS alpha, which is the
    only re-centred/re-sloped class, so the honest pin is: the sub/DLA columns agree to
    renorm-coupling level (<1e-2 relative; a genuine centre/slope mislabel is a >=10% effect)
    and are NOT bit-identical (the coupling is real and nonzero). Holds for the mapped KS too:
    its reference sub/DLA columns invert the SAME boost-1.0 triple as DESI/eBOSS."""
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


def test_subdla_dla_identical_across_surveys(npz):
    """v4 cross-survey identity: all three deployed sub/DLA columns invert the identical
    boost-1.0 centre triple (the KS mapped reference shares it), so they must agree bit-exactly
    -- a KS reference built on different sub/DLA geometry would silently de-align the overlay."""
    e = np.asarray(npz["deployed_center_dndx_eBOSS"])
    for sv in ("KS", "DESI"):
        got = np.asarray(npz[f"deployed_center_dndx_{sv}"])
        np.testing.assert_allclose(got[:, 1:], e[:, 1:], rtol=0, atol=1e-15)


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
    (which is the CLOSURE pivot). Every pivot array must name its provenance.

    KS SEMANTICS (v4, FLIPPED from the W2-era tripwire): the deployed KS prior IS the
    dN/dX-mapped parameterization and this exporter now ships the MAPPED surface, so the KS
    row is pinned AT the mapped pivot centre 0.393235480 (a derived deterministic through
    w_c_corrected), asserted to DIFFER from the legacy alpha-postmap centre 0.4302804 the
    v1/v2 artifacts shipped -- and the mapped survey may NOT ship an alpha-space sigma."""
    sur = [str(s) for s in npz["survey_order"]]
    par = {str(s): str(p) for s, p in zip(npz["survey_order"],
                                          npz["parameterization_by_survey"])}
    for sv in sur:
        for bad in (f"alpha_pivot_mu_{sv}", f"alpha_pivot_sigma_{sv}"):
            assert bad not in npz.files, f"unprefixed key {bad!r} still shipped"
        assert f"deployed_alpha_pivot_mu_{sv}" in npz.files
    for k in ("closure_alpha_pivot_mu", "closure_alpha_pivot_sigma"):
        assert k in npz.files, f"closure pivot geometry {k!r} not shipped for a like-for-like read"
    np.testing.assert_allclose(np.asarray(npz["closure_alpha_pivot_mu"]), MU_CLOSURE)
    assert INF.HCD_ALPHA_PARAMETERIZATION["KS"] == "dndx_mapped_v2"
    for sv in sur:
        mu = np.asarray(npz[f"deployed_alpha_pivot_mu_{sv}"])
        b = float(HCD_LLS_SURVEY_BOOST[sv])
        if par[sv] == "alpha_pivot_powerlaw_v1":
            assert f"deployed_alpha_pivot_sigma_{sv}" in npz.files
            # the deployed LLS centre is the corrected lit law x boost, NOT closure x boost.
            assert abs(mu[0] - 0.17211216067355312 * b) < 1e-12, f"{sv} deployed LLS centre {mu[0]!r}"
            assert abs(mu[0] - MU_CLOSURE[0] * b) > 1e-3, (
                f"{sv} deployed LLS centre equals closure x boost -- that is the construction the "
                f"script's warning (1) exists to prevent")
            np.testing.assert_array_equal(mu[1:], MU_CLOSURE[1:])
        else:
            assert sv == "KS" and par[sv] == "dndx_mapped_v2"
            # THE v4 PIN: the mapped pivot centre (W2 value), NOT the legacy postmap centre.
            assert abs(mu[0] - KS_MAPPED_PIVOT) < 1e-6, (
                f"KS mapped pivot {mu[0]!r} != W2-pinned {KS_MAPPED_PIVOT}")
            assert abs(mu[0] - KS_LEGACY_PIVOT) > 0.03, (
                "the KS row equals the LEGACY alpha-postmap centre 0.4302804 -- the exporter "
                "regressed to the pre-v4 legacy surface (v1/v2 artifacts)")
            # mapped sub/DLA pivot alphas: renorm-coupled to (not equal to) the closure values
            assert not np.array_equal(mu[1:], MU_CLOSURE[1:]), (
                "KS mapped sub/DLA pivot alphas are bit-identical to the closure centres -- the "
                "forward-map renorm coupling must be nonzero")
            np.testing.assert_allclose(mu[1:], MU_CLOSURE[1:], rtol=1e-2)
            # NO alpha-space Gaussian width may ship for the mapped survey.
            for bad in (f"deployed_alpha_pivot_sigma_{sv}",
                        f"deployed_dla_softplus_quantiles_{sv}"):
                assert bad not in npz.files, (
                    f"{bad!r} shipped for the mapped KS survey -- no such alpha-space object "
                    f"exists in the deployed dndx_mapped_v2 prior")


# --------------------------------------------------------------- v4 native-dN/dX KS semantics
def test_ks_centre_is_the_native_mapped_reference(npz):
    """v4 LOAD-BEARING claim: deployed_center_dndx_KS IS the deterministic reference the
    deployed KS prior samples around -- rebuilt here through the IDENTICAL code path the ctx
    build uses (closure_legb._ks_dndx_reference) and asserted bit-equal. The sampled quantity
    is the paper number; no inverse map is involved in the KS readout."""
    zg = np.asarray(npz["zgrid"], float)
    cf = np.polyfit(zg, np.asarray(npz["xbar_grid"], float), 2)
    ref, xbar_z, ref_piv, xbar_piv = _ks_dndx_reference(MU_CLOSURE, cf, zg)
    got = np.asarray(npz["deployed_center_dndx_KS"], float)
    np.testing.assert_array_equal(got, np.asarray(ref, float))
    np.testing.assert_allclose(np.asarray(npz["ks_dndx_ref_pivot"], float),
                               np.asarray(ref_piv, float), rtol=0, atol=1e-15)
    assert abs(float(np.asarray(npz["ks_xbar_pivot"])) - float(xbar_piv)) < 1e-12


def test_ks_band_is_pivot_amplitude_band(npz):
    """The KS +-1 sigma edges are pivot-AMPLITUDE bands: exp(+-KS_DNDX_SIGMA_EPS) x centre on
    the LLS column (z-INDEPENDENT log-amplitude shift), exp(+-KS_DNDX_SIGMA_MSUB) on subDLA,
    and the softplus amplitude ratio on DLA. The tilt widths (sigma_kappa etc.) add z-dependent
    spread deliberately NOT shown -- mirroring the legacy pivot-amplitude-band-only convention."""
    import jax
    c = np.asarray(npz["deployed_center_dndx_KS"], float)
    hi = np.asarray(npz["deployed_hi1_dndx_KS"], float)
    lo = np.asarray(npz["deployed_lo1_dndx_KS"], float)
    eps = float(np.asarray(npz["ks_sigma_eps"]))
    msub = float(np.asarray(npz["ks_sigma_msub"]))
    raw0 = float(np.asarray(npz["ks_dla_raw_mu0"]))
    assert eps == float(INF.KS_DNDX_SIGMA_EPS) == 0.5310
    assert float(np.asarray(npz["ks_sigma_kappa"])) == float(INF.KS_DNDX_SIGMA_KAPPA) == 0.6681
    assert msub == float(INF.KS_DNDX_SIGMA_MSUB)
    sp0 = float(jax.nn.softplus(raw0))
    np.testing.assert_array_equal(hi[:, 0], np.exp(+eps) * c[:, 0])
    np.testing.assert_array_equal(lo[:, 0], np.exp(-eps) * c[:, 0])
    np.testing.assert_array_equal(hi[:, 1], np.exp(+msub) * c[:, 1])
    np.testing.assert_array_equal(lo[:, 1], np.exp(-msub) * c[:, 1])
    np.testing.assert_array_equal(hi[:, 2], (float(jax.nn.softplus(raw0 + 1.0)) / sp0) * c[:, 2])
    np.testing.assert_array_equal(lo[:, 2], (float(jax.nn.softplus(raw0 - 1.0)) / sp0) * c[:, 2])
    # the band is a z-independent RATIO: constant across the grid (no tilt spread shown)
    np.testing.assert_allclose(hi[:, 0] / c[:, 0], np.exp(eps), rtol=1e-12)


def test_ks_forward_roundtrip_machine_precision(npz):
    """v4 self-check contract: the KS native-dN/dX centre must round-trip through the FORWARD
    occupancy map (w_c_corrected -> alpha -> alpha_to_dndx_exact -> back) at machine precision
    -- the inverse is a CHECK on the shipped curve, not its construction."""
    import jax.numpy as jnp
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact, w_c_corrected
    zg = np.asarray(npz["zgrid"], float)
    xb = np.asarray(npz["xbar_grid"], float)
    c = np.asarray(npz["deployed_center_dndx_KS"], float)
    alpha_z = np.asarray(w_c_corrected(jnp.asarray(c), jnp.asarray(xb), jnp.asarray(zg)))[..., 1:]
    back = np.asarray(alpha_to_dndx_exact(alpha_z, xb, zg))
    assert float(np.max(np.abs(back / c - 1.0))) < 1e-12


def test_no_saturation_fingerprint_anywhere(npz):
    """The v1/v2 shipped-artifact defect: the approximate inverse silently saturated the
    out-of-simplex KS LLS column at mu = 27.631021, i.e. dN/dX = 27.631021/Xbar (up to ~33x the
    lit law at z>=4.5). No deployed or closure array in the v4 export may contain that
    fingerprint at ANY z -- including z=4.6, where v2 shipped garbage and v3 refused."""
    xb = np.asarray(npz["xbar_grid"], float)
    fp = SATURATION_MU / xb                                     # (n_z,) per-z fingerprint
    names = [k for k in npz.files
             if k.startswith(("deployed_center_dndx", "deployed_lo1_dndx",
                              "deployed_hi1_dndx", "closure_center_dndx"))]
    assert len(names) >= 10
    for k in names:
        arr = np.asarray(npz[k], float)
        assert not np.any(np.isclose(arr, fp[:, None], rtol=1e-6)), (
            f"{k} contains the 27.631021/Xbar saturation fingerprint of the retired "
            f"approximate inverse")
        assert np.all(np.isfinite(arr)), f"{k} contains non-finite entries"
    # and the KS centre at the last node is the boosted lit law, NOT the saturated value
    zg = np.asarray(npz["zgrid"], float)
    A, g = [float(v) for v in np.asarray(npz["ks_lit_dndx_law_lls"], float)]
    want_hi_z = 2.5 * A * (1.0 + zg[-1]) ** g
    got_hi_z = float(np.asarray(npz["deployed_center_dndx_KS"])[-1, 0])
    assert abs(got_hi_z / want_hi_z - 1.0) < 1e-12
    assert got_hi_z < 0.2 * (SATURATION_MU / xb[-1])            # nowhere near the clamp value


def test_full_grid_builds_no_ks_refusal():
    """v3 correctly REFUSED on any grid reaching z>=~4.45 (legacy KS alpha centre outside the
    occupancy simplex). Under v4 the KS centre never routes through the inverse, so the FULL
    grid (to z=4.6) must build -- this fixture grid includes z=4.6 and `built` above proves it.
    Here we additionally pin that a refusal in the alpha-parameterized branch is still fail-loud
    (DESI/eBOSS drift guard): an absurd closure mu that leaves the simplex must raise, naming
    the survey."""
    bad_mu = np.array([0.9, 0.6, 0.004], float)                 # sum >> 1 after z-scaling
    closure_center = MOD.band_curve(MU_CLOSURE, Z_TINY, XB_TINY, float(HCD_Z_PIVOT),
                                    np.asarray(HCD_INCIDENCE_SLOPE, float))
    with pytest.raises((ValueError, AssertionError)):
        MOD.build_npz_payload(Z_TINY, XB_TINY, bad_mu, SD_CLOSURE, float(HCD_Z_PIVOT),
                              closure_center)


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
        assert "alpha_pivot_mu" not in row and "alpha_pivot_sigma" not in row
        if row["parameterization"] == "alpha_pivot_powerlaw_v1":
            assert "alpha_pivot_mu_deployed" in row and "alpha_pivot_sigma_deployed" in row
        else:
            assert sv == "KS" and row["parameterization"] == "dndx_mapped_v2"
            # the mapped row names its pivot as MAPPED and carries the dN/dX-space widths;
            # an alpha-space sigma would be a fabrication (none exists in the deployed prior).
            assert "alpha_pivot_mu_deployed_mapped" in row
            assert "alpha_pivot_sigma_deployed" not in row and "alpha_pivot_mu_deployed" not in row
            for k in ("dndx_sigma_eps", "dndx_sigma_kappa", "dndx_sigma_msub", "boost_space"):
                assert k in row, f"KS mapped sidecar row missing {k!r}"
            assert abs(row["alpha_pivot_mu_deployed_mapped"][0] - KS_MAPPED_PIVOT) < 1e-6


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


def test_verify_npz_payload_rejects_alpha_sigma_for_mapped_ks(built):
    """v4 parameterization-axis guard: an alpha-space Gaussian width for the mapped KS survey
    is a key whose object does not exist in the deployed prior -- verify must refuse it."""
    payload, _ = built
    tainted = dict(payload)
    tainted["deployed_alpha_pivot_sigma_KS"] = np.array([0.17, 0.025, 0.0022])
    with pytest.raises(AssertionError, match="dndx_mapped_v2"):
        MOD.verify_npz_payload(tainted)


def test_verify_npz_payload_rejects_legacy_ks_lls_column(built):
    """ARRAY-vs-LABEL, mapped branch: a payload whose KS LLS column is NOT the boosted lit law
    (e.g. a regression to the legacy alpha-postmap build) must be refused."""
    payload, _ = built
    tainted = dict(payload)
    wrong = np.array(payload["deployed_center_dndx_KS"], copy=True)
    wrong[:, 0] *= 1.05
    tainted["deployed_center_dndx_KS"] = wrong
    with pytest.raises(AssertionError, match="lit"):
        MOD.verify_npz_payload(tainted)


def test_verify_npz_payload_rejects_missing_parameterization(built):
    """The v4 schema requires parameterization_by_survey: without it a reader cannot tell the
    mapped KS surface from the legacy one -- the exact ambiguity that shipped v2's garbage."""
    payload, _ = built
    tainted = {k: v for k, v in payload.items() if k != "parameterization_by_survey"}
    with pytest.raises(AssertionError, match="parameterization_by_survey"):
        MOD.verify_npz_payload(tainted)


def test_verify_npz_payload_accepts_the_real_payload(built, tmp_path):
    payload, _ = built
    MOD.verify_npz_payload(payload)
    p = tmp_path / "x.npz"
    np.savez(p, **payload)
    MOD.verify_npz_payload(dict(np.load(p, allow_pickle=True)))   # survives the round trip
