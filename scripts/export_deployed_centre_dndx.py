"""Export the DEPLOYED-SURVEY prior centre in dN/dX for paper figure F3.

Answers ~/Latex/HCDEmulatorPaper/request_to_code_agent_2026-07-21_deployed_centre.md
(PI-authorized 2026-07-21). F3 draws the closure/cosmic-average band (survey=None); the PI
asked for the DEPLOYED SURVEY prior centre alongside it. The frozen export gives those
centres in ALPHA space only, and mapping alpha -> dN/dX needs the exact inverse
(`dndx_wc.alpha_to_dndx_exact` with Xbar(z), the renorm undo, and the telescoping/(1+delta_c)
step), which the paper side is forbidden to reconstruct locally. So the map is done HERE and
shipped as an array.

Delivery option (a) of the request: a small NEW dated export dir. The artifact of record
`dndx_repin_2026-07-20/` is READ ONLY and is never touched -- F3 and the money figure both
assert against its hashes at build time.

CONSTRUCTION for the ALPHA-parameterized surveys (DESI, eBOSS) mirrors the frozen F3 layer
(`scripts/export_dndx_paper_layer.py`), with `survey=None` swapped for the deployed survey key:
    alpha_c(z) = mu_c * ((1+z)/(1+z_pivot))**zslope_c   then alpha_to_dndx_exact(alpha, Xbar, z)
z grid and Xbar(z) are READ FROM the frozen `f3_layers.npz` rather than recomputed, so the
paper can overlay these curves on the existing layer without an interpolation step.

V3 INVERSE-MAP MIGRATION (readout defect B, 2026-07-22): pre-v3 layers -- INCLUDING the
frozen `dndx_repin_2026-07-20/f3_layers.npz` -- were built with the APPROXIMATE
`alpha_to_dndx` (renorm ignored, median rel err ~2e-3, silent saturation at 27.631021/Xbar
outside its domain). v3+ uses `alpha_to_dndx_exact` (mode="raise"), so byte-level reproduction
of the frozen approximate layers is NO LONGER asserted; the renorm-level difference is
computed and printed instead, and the SELF-CHECK below is exact-vs-exact.

V4 KS NATIVE-dN/dX MIGRATION (KS prior reparameterization, W2/W7 2026-07-22): the deployed
KS prior now SAMPLES IN dN/dX SPACE around a deterministic reference
(inference.HCD_ALPHA_PARAMETERIZATION["KS"] = "dndx_mapped_v2"; the 2.5x LLS boost acts on
the LITERATURE dN/dX law PRE-map). The deployed KS centre curve in dN/dX space therefore IS
the reference `ks_dndx_ref` (closure_legb._ks_dndx_reference, the identical code path
build_legb_ctx(survey="KS") uses) -- NO inverse map is involved for KS: the sampled quantity
is the paper number, and the exact inverse is a SELF-CHECK (forward round trip at machine
precision), not the readout. The v3 fail-loud KS refusal (the KS alpha centre left the
occupancy simplex at z >= ~4.45) is RETIRED: the mapped construction has no out-of-domain
region, so no saturation/refusal is possible for the KS centre, on any z. DESI/eBOSS keep the
legacy alpha-space construction (in-domain everywhere on the export grid; any refusal is now
UNEXPECTED and means the prior geometry drifted).

SELF-CHECK (fail-loud): before writing anything, this script (a) re-derives the survey=None
closure layer TWO ways -- through `band_curve` (the code path every alpha-parameterized
deployed curve uses) and directly through `alpha_to_dndx_exact` on the explicitly-constructed
z-resolved alpha -- and asserts they agree to within 1e-12 on the frozen zgrid/Xbar/pivot
inputs; (b) round-trips the KS native-dN/dX centre through the FORWARD occupancy map
(w_c_corrected -> alpha -> alpha_to_dndx_exact -> back) and asserts machine-precision
agreement (<1e-12 relative); (c) pins the MAPPED KS alpha-space pivot centre to the W2 value
0.393235480 +- 1e-6. If any fails, the export refuses to write rather than shipping a curve
the paper would overlay wrongly.

KEY DISCIPLINE (defect fix 2026-07-21): every array is either a CLOSURE or a DEPLOYED quantity and
the two differ in BOTH centre and z-shape, so no key may leave its layer to the reader's guess. The
export ships `zslope_closure` (2.465, the sim slope the closure band uses) AND `zslope_deployed`
(2.127, the litWLS LLS slope the deployed curves use) and NEVER a bare `zslope`. `verify_npz_payload`
re-reads the written file and refuses any ambiguous key -- the original defect was a silently failed
sed edit whose resulting key set nobody checked. v4 extends the discipline to the PARAMETERIZATION
axis: `parameterization_by_survey` names how each survey's prior is parameterized, and the mapped
KS survey may NOT ship an alpha-space Gaussian width (`deployed_alpha_pivot_sigma_KS`) -- no such
object exists in the deployed mapped prior (its widths are the dN/dX-space KS_DNDX_SIGMA_*).

Blind status: BLIND-SAFE. Prior geometry only -- no data, no posterior, no cosmology.

Run (login node):
  cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/export_deployed_centre_dndx.py
"""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE, _ks_dndx_reference
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact, w_c_corrected
from hcd_analysis.emulator.sampler_numpyro import _dla_raw_mu
# KS_DNDX_* floats (and the parameterization table) are read as MODULE ATTRIBUTES of
# `inference` -- NEVER from-imported (the from-import rebinding trap, JAX-specialist review
# 2026-07-22: the sampled sites and the freeze signature read inference's namespace, so a
# from-imported copy here could desync this export from the deployed prior under an override).
from hcd_analysis.emulator import inference as INF
from hcd_analysis.emulator.inference import (
    HCD_LLS_REALFIT_ZSLOPE,
    hcd_lls_realfit_alpha_center,
    HCD_LLS_SURVEY_BOOST,
    HCD_LLS_SURVEY_FRAC_SIGMA,
    HCD_Z_PIVOT,
    assert_hcd_pivot_z3,
    hcd_incidence_prior,
)

CLS = ("LLS", "subDLA", "DLA")
FROZEN = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/dndx_repin_2026-07-20")
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/deployed_centre_2026-07-22_v4")
SURVEYS = ("eBOSS", "KS", "DESI")
RUN_CMD = ("cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu "
           "/home/mfho/.conda/envs/emu-jax/bin/python3 scripts/export_deployed_centre_dndx.py")

# The W2-pinned MAPPED KS alpha-space LLS pivot centre (the z=3 LLS occupancy weight of the
# dN/dX-premap 2.5x lit-law reference through the exact occupancy map; sites at 0). The legacy
# alpha-postmap centre was 0.4302804016838828 -- the mapped migration MOVED it here.
KS_MAPPED_PIVOT_TARGET = 0.393235480
KS_MAPPED_PIVOT_TOL = 1e-6


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def band_curve(alpha_pivot_vec, zgrid, Xb, zp, slope):
    """The F3 construction for ALPHA-parameterized surveys: pivot alpha -> z-resolved alpha ->
    dN/dX, through the EXACT inverse (v3; the frozen F3 layer used the approximate
    `alpha_to_dndx`). mode="raise": an out-of-simplex centre/edge refuses loudly instead of
    silently saturating. NOT used for the mapped KS survey (v4): its centre is native dN/dX."""
    az = (np.asarray(alpha_pivot_vec)[None, :]
          * ((1.0 + np.asarray(zgrid, float))[:, None] / (1.0 + zp)) ** np.asarray(slope)[None, :])
    return np.asarray(alpha_to_dndx_exact(az, np.asarray(Xb, float), np.asarray(zgrid, float)))


# --------------------------------------------------------------------------------------------- #
#  KEY-AMBIGUITY POSTCONDITION.
#  Defect of 2026-07-21 (paper agent): the export promised `zslope_closure` + `zslope_deployed`
#  and shipped NEITHER -- a sed edit silently failed to match, and a single UNLABELLED
#  `zslope` = the CLOSURE slope (2.465,...) went out inside a DEPLOYED-layer file. Anyone reading
#  npz['zslope'] to describe the deployed curve states exactly the error warning (2) below exists
#  to prevent. Root cause was that nobody re-read the resulting key set, so the writer now does it
#  itself and refuses to ship an ambiguous one. Test-pinned by tests/test_export_deployed_centre.py.
# --------------------------------------------------------------------------------------------- #
#  name -> why it is ambiguous in THIS file, and what to use instead.
AMBIGUOUS_KEYS = {
    "zslope": ("closure (2.465) vs deployed (2.127) LLS z-slope -- ship zslope_closure AND "
               "zslope_deployed"),
    "boost": "per-SURVEY length-3 vector, indistinguishable from a per-CLASS one -- boost_by_survey",
    "lls_frac_sigma": "per-SURVEY length-3 vector -- lls_frac_sigma_by_survey",
    "alpha_pivot_mu": ("collides with the frozen f3_layers key of the same name (= CLOSURE) -- "
                       "closure_alpha_pivot_mu / deployed_alpha_pivot_mu_<survey>"),
    "alpha_pivot_sigma": ("collides with the frozen f3_layers key of the same name (= CLOSURE) -- "
                          "closure_alpha_pivot_sigma / deployed_alpha_pivot_sigma_<survey>"),
}


def verify_npz_payload(payload, surveys=SURVEYS):
    """Fail-loud audit of the key SET about to be written (or just read back).

    Every array in this file is either a CLOSURE quantity or a DEPLOYED one, and the two differ in
    BOTH centre and z-shape; a key that does not say which is a mislabelling waiting to happen.
    v4 adds the PARAMETERIZATION axis: each survey's deployed keys are audited against the
    parameterization it deploys (`parameterization_by_survey`), so an alpha-space key cannot ship
    for the mapped KS survey (and vice versa)."""
    keys = set(payload)
    for bad, why in AMBIGUOUS_KEYS.items():
        assert bad not in keys, (
            f"ambiguous key {bad!r} in the deployed-centre export: {why}. Refusing to ship a key "
            f"whose closure-vs-deployed meaning a reader has to guess.")
    for k in ("zslope_closure", "zslope_deployed"):
        assert k in keys, f"missing required key {k!r} (have: {sorted(keys)})"
    clo = np.asarray(payload["zslope_closure"], float)
    dep = np.asarray(payload["zslope_deployed"], float)
    np.testing.assert_array_equal(clo, np.asarray(HCD_INCIDENCE_SLOPE, float))
    assert float(dep[0]) == float(HCD_LLS_REALFIT_ZSLOPE), (
        f"zslope_deployed[LLS] = {dep[0]!r}, expected the deployed real-fit LLS forward z-slope "
        f"HCD_LLS_REALFIT_ZSLOPE = {float(HCD_LLS_REALFIT_ZSLOPE)!r}")
    assert not np.array_equal(clo, dep), (
        "zslope_closure and zslope_deployed are the SAME array -- the deployed LLS curve differs "
        "from the closure one in z-SHAPE (2.127 vs 2.465), not only in normalisation.")
    np.testing.assert_array_equal(dep[1:], clo[1:])          # subDLA/DLA are survey-agnostic
    # ARRAY-vs-LABEL POSTCONDITION (adversarial-panel blocker 4, 2026-07-21; extended to the
    # parameterization axis in v4). Everything above checks NAMES and scalar slope values. A
    # mutant with perfect key names, perfect slope values, and deployed curves rebuilt from the
    # CLOSURE slope passed all of it -- which is the very defect class this file exists to
    # prevent, one layer up. So re-derive every deployed LLS column FROM THE LABELLED PARAMETERS
    # of its OWN parameterization and require it to match the shipped array. All inputs are
    # already in the payload, so this is self-contained.
    if {"zgrid", "xbar_grid", "z_pivot", "survey_order"} <= keys:
        zg = np.asarray(payload["zgrid"], float)
        xb = np.asarray(payload["xbar_grid"], float)
        zpv = float(np.asarray(payload["z_pivot"]))
        svs = [str(s) for s in np.asarray(payload["survey_order"])]
        assert "parameterization_by_survey" in keys, (
            "missing parameterization_by_survey (v4 schema): every deployed survey must name how "
            "its prior is parameterized (alpha_pivot_powerlaw_v1 vs dndx_mapped_v2)")
        par = [str(s) for s in np.asarray(payload["parameterization_by_survey"])]
        assert len(par) == len(svs), "parameterization_by_survey misaligned with survey_order"
        boosts = np.asarray(payload["boost_by_survey"], float)
        for i, sv in enumerate(svs):
            # the artifact's claimed parameterization must match the DEPLOYED one (drift tripwire)
            assert par[i] == str(INF.HCD_ALPHA_PARAMETERIZATION[sv]), (
                f"parameterization_by_survey[{sv}] = {par[i]!r} but the deployed "
                f"inference.HCD_ALPHA_PARAMETERIZATION[{sv!r}] = "
                f"{INF.HCD_ALPHA_PARAMETERIZATION[sv]!r} -- the artifact and the deployed prior "
                f"disagree about what is being shipped")
            got = np.asarray(payload[f"deployed_center_dndx_{sv}"], float)
            if par[i] == "alpha_pivot_powerlaw_v1":
                mu_sv = np.asarray(payload[f"deployed_alpha_pivot_mu_{sv}"], float)
                want = band_curve(mu_sv, zg, xb, zpv, dep)[:, 0]
                assert np.array_equal(got[:, 0], want), (
                    f"deployed_center_dndx_{sv}[:, LLS] does NOT reproduce from the labelled "
                    f"zslope_deployed + deployed_alpha_pivot_mu_{sv}. The arrays and the labels "
                    f"disagree: max|diff| = {np.max(np.abs(got[:, 0] - want)):.6e}. Refusing to "
                    f"ship a file whose keys describe a curve it does not contain.")
            elif par[i] == "dndx_mapped_v2":
                assert sv == "KS", (
                    f"survey {sv!r} claims dndx_mapped_v2 but the ks_* key block is KS-specific; "
                    f"extend the key schema before shipping another mapped survey")
                # (1) NATIVE dN/dX centre: the LLS column IS the boosted lit law (bit-equal).
                for k in ("ks_lit_dndx_law_lls", "ks_dndx_ref_pivot", "ks_xbar_pivot",
                          "ks_sigma_eps", "ks_sigma_kappa", "ks_sigma_msub", "ks_dla_raw_mu0"):
                    assert k in keys, f"mapped KS survey requires key {k!r} (have: {sorted(keys)})"
                A, g = [float(v) for v in np.asarray(payload["ks_lit_dndx_law_lls"], float)]
                want_lls = boosts[i] * A * (1.0 + zg) ** g
                assert np.array_equal(got[:, 0], want_lls), (
                    f"deployed_center_dndx_{sv}[:, LLS] is NOT the boosted literature dN/dX law "
                    f"({boosts[i]:g} x {A!r} x (1+z)^{g!r}) the mapped KS prior is centred on: "
                    f"max|diff| = {np.max(np.abs(got[:, 0] - want_lls)):.6e}")
                # (2) hi1/lo1 LLS band semantics: exp(+-sigma_eps) x centre (z-independent
                #     log-amplitude shift; the kappa tilt width is NOT in this band).
                eps = float(np.asarray(payload["ks_sigma_eps"]))
                hi1 = np.asarray(payload[f"deployed_hi1_dndx_{sv}"], float)
                lo1 = np.asarray(payload[f"deployed_lo1_dndx_{sv}"], float)
                assert np.array_equal(hi1[:, 0], np.exp(+eps) * got[:, 0]), (
                    f"deployed_hi1_dndx_{sv}[:, LLS] != exp(+ks_sigma_eps) x centre")
                assert np.array_equal(lo1[:, 0], np.exp(-eps) * got[:, 0]), (
                    f"deployed_lo1_dndx_{sv}[:, LLS] != exp(-ks_sigma_eps) x centre")
                # (3) the labelled alpha-space MAPPED pivot must reproduce from the shipped
                #     reference through the forward occupancy map, and pin the W2 value.
                ref_piv = np.asarray(payload["ks_dndx_ref_pivot"], float)
                xb_piv = float(np.asarray(payload["ks_xbar_pivot"]))
                a_piv = np.asarray(w_c_corrected(jnp.asarray(ref_piv), jnp.asarray(xb_piv),
                                                 jnp.asarray(zpv)))[..., 1:]
                got_mu = np.asarray(payload[f"deployed_alpha_pivot_mu_{sv}"], float)
                assert np.max(np.abs(a_piv - got_mu)) < 1e-12, (
                    f"deployed_alpha_pivot_mu_{sv} does NOT reproduce from the shipped "
                    f"ks_dndx_ref_pivot through w_c_corrected: max|diff| = "
                    f"{np.max(np.abs(a_piv - got_mu)):.3e}")
                assert abs(float(got_mu[0]) - KS_MAPPED_PIVOT_TARGET) < KS_MAPPED_PIVOT_TOL, (
                    f"mapped KS LLS pivot centre {float(got_mu[0])!r} != the W2-pinned "
                    f"{KS_MAPPED_PIVOT_TARGET} +- {KS_MAPPED_PIVOT_TOL}")
                # (4) NO alpha-space Gaussian width may ship for the mapped survey: the deployed
                #     mapped prior has none (widths are dN/dX-space KS_DNDX_SIGMA_*), and the
                #     legacy alpha-space DLA softplus quantiles are equally a different object.
                for bad in (f"deployed_alpha_pivot_sigma_{sv}",
                            f"deployed_dla_softplus_quantiles_{sv}"):
                    assert bad not in keys, (
                        f"{bad!r} shipped for the dndx_mapped_v2 survey {sv}: the mapped prior "
                        f"has NO alpha-space Gaussian width / softplus alpha quantiles -- its "
                        f"widths live in dN/dX space (ks_sigma_eps/ks_sigma_kappa/ks_sigma_msub/"
                        f"ks_dla_amp_quantiles). Shipping this key re-creates the legacy-surface "
                        f"misread the v4 migration retired.")
            else:
                raise AssertionError(f"unknown parameterization {par[i]!r} for survey {sv}")
    if "survey_order" in keys:
        par = ([str(s) for s in np.asarray(payload["parameterization_by_survey"])]
               if "parameterization_by_survey" in keys else None)
        for i, sv in enumerate([str(s) for s in np.asarray(payload["survey_order"])]):
            for stem in ("alpha_pivot_mu", "alpha_pivot_sigma"):
                assert f"{stem}_{sv}" not in keys, (
                    f"{stem}_{sv!r} does not say whether it is the closure or the deployed pivot; "
                    f"use deployed_{stem}_{sv}")
            assert f"deployed_alpha_pivot_mu_{sv}" in keys, f"missing deployed_alpha_pivot_mu_{sv}"
            if par is not None and par[i] == "alpha_pivot_powerlaw_v1":
                assert f"deployed_alpha_pivot_sigma_{sv}" in keys, \
                    f"missing deployed_alpha_pivot_sigma_{sv}"
            for stem in ("center", "lo1", "hi1"):
                assert f"deployed_{stem}_dndx_{sv}" in keys, f"missing deployed_{stem}_dndx_{sv}"
    return True


NOTE = (
    "DEPLOYED-SURVEY prior centre in dN/dX on the frozen f3_layers zgrid. v4: the KS survey is "
    "NATIVE dN/dX (parameterization dndx_mapped_v2, W2 2026-07-22) -- deployed_center_dndx_KS IS "
    "the deterministic reference the deployed KS prior samples around (LLS column = 2.5 x the "
    "corrected lit law 0.01840092 x (1+z)^2.127; sub/DLA columns = the exact inverse of the "
    "boost-1.0 deployed centre triple), with NO inverse map and NO possible saturation; its "
    "hi1/lo1 edges are pivot-AMPLITUDE bands only (exp(+-0.5310) LLS, exp(+-0.4130) subDLA, "
    "softplus amplitude ratio DLA) -- the exponent widths (kappa 0.6681, t_sub, t_dla) add "
    "z-dependent spread NOT shown, mirroring how the legacy export showed the pivot-amplitude "
    "band only. DESI/eBOSS keep the legacy alpha-space construction through the EXACT "
    "alpha->dN/dX inverse alpha_to_dndx_exact (v3, readout defect B 2026-07-22; pre-v3 layers "
    "including the frozen f3_layers.npz used the APPROXIMATE alpha_to_dndx and differ at the "
    "renorm level). Arrays are (n_z, n_class) with class_order. CLOSURE vs DEPLOYED differ in "
    "BOTH centre and z-shape: the deployed LLS centre is the K1a-corrected literature dN/dX law "
    "(0.172112 at boost 1.0), NOT the closure centre (0.188119); and the deployed LLS z-slope is "
    "2.127 (the litWLS slope), NOT the sim zslope_closure[0]=2.465 the closure band uses. "
    "subDLA/DLA centres are survey-agnostic at the source, and the deployed sub/DLA dN/dX "
    "columns agree ACROSS surveys bit-exactly (KS's reference shares the boost-1.0 triple with "
    "DESI/eBOSS) while differing from closure_center_dndx at the exact-inverse renorm-coupling "
    "level (~3e-4 relative, asserted <1e-2 at export). The survey=None closure layer re-derived "
    "through the SAME exact map is included as closure_center_dndx for a like-for-like overlay. "
    "The export zgrid starts at 2.2 while the KS survey band starts at z=2.4: the z<2.4 rows are "
    "prior geometry shipped for overlay continuity, exactly as previous exports did. Prior "
    "geometry only: no data, no posterior, no cosmology.")


def build_npz_payload(zgrid, xbar_grid, mu_closure, sd_closure, z_pivot, closure_center_dndx,
                      surveys=SURVEYS, xbar_cf=None):
    """Assemble (npz payload dict, PROVENANCE prior_geometry_table) for the deployed layer.

    ALPHA-parameterized surveys (DESI, eBOSS -- inference.HCD_ALPHA_PARAMETERIZATION
    'alpha_pivot_powerlaw_v1'): TWO things differ from the closure band, and BOTH were got
    wrong on the first pass:
      (1) CENTRE. The deployed LLS pin is NOT the closure centre x boost. It is built from the
          CORRECTED literature dN/dX law directly (hcd_lls_realfit_alpha_center: alt-(b), the
          K1a binned [17.2,19.0) law A=0.0184*(1+z)^2.127), because on the real fit PRIYA != data
          so the LLS centre must track the literature, not the sim's z=3 w_c*(lit/sim).
          0.172112 x boost, NOT 0.188119 x boost. See closure_legb._survey_alpha_prior:735.
      (2) Z-SLOPE. The deployed real-fit LLS forward z-slope is HCD_LLS_REALFIT_ZSLOPE = 2.127
          (survey-INDEPENDENT, the litWLS slope), not the sim HCD_INCIDENCE_SLOPE[0] = 2.465 the
          closure band uses. So the deployed curve has a different z-shape, not just a different
          normalization. See closure_legb._survey_alpha_prior (survey_zslope_mu).
    subDLA/DLA centres and slopes are survey-agnostic and stay at the closure values.

    MAPPED survey (KS -- 'dndx_mapped_v2', v4): the deployed centre curve in dN/dX space IS the
    deterministic reference the sampler deviates around, built by the IDENTICAL code path the
    deployed ctx uses (closure_legb._ks_dndx_reference): sub/DLA components = exact inverse of
    the boost-1.0 deployed centre triple; LLS component REPLACED by the dndx-premap boosted lit
    law 2.5 x 0.01840092 x (1+z)^2.127. No inverse map, no saturation, no refusal possible.
    Edges are pivot-amplitude bands (exp(+-sigma_eps) LLS / exp(+-sigma_msub) subDLA / softplus
    amplitude ratio DLA); the exponent widths add z-dependent spread NOT shown. The KS
    alpha-space pivot (deployed_alpha_pivot_mu_KS) is the DERIVED forward-mapped triple
    (LLS 0.393235, moved from the legacy postmap 0.430280), NOT a sampled Gaussian centre, and
    NO alpha-space sigma exists (verify_npz_payload refuses one).

    ``xbar_cf`` = the deg-2 Xbar(z) polynomial coefficients (np.polyval order) the mapped
    reference consumes -- pass the frozen `f3_layers['xbar_poly_deg2_coeffs']` in production;
    None fits them from (zgrid, xbar_grid) and REFUSES if the grid is not that quadratic.

    Pure construction: no file I/O, no frozen-artifact acceptance targets (those live in main so
    this stays testable on a synthetic grid). Runs verify_npz_payload on the way out.
    """
    zgrid = np.asarray(zgrid, float)
    Xb = np.asarray(xbar_grid, float)
    mu0 = np.asarray(mu_closure, float)
    sd0 = np.asarray(sd_closure, float)
    zp = float(z_pivot)
    slope_closure = np.asarray(HCD_INCIDENCE_SLOPE, float)
    slope_deployed = np.array([float(HCD_LLS_REALFIT_ZSLOPE),
                               float(slope_closure[1]), float(slope_closure[2])], float)
    Xb3 = float(np.interp(zp, zgrid, Xb))
    if xbar_cf is None:
        xbar_cf = np.polyfit(zgrid, Xb, 2)
    xbar_cf = np.asarray(xbar_cf, float)
    _rel_cf = float(np.max(np.abs(np.polyval(xbar_cf, zgrid) / Xb - 1.0)))
    assert _rel_cf < 1e-9, (
        f"xbar_grid is not the deg-2 polynomial xbar_cf claims (max rel diff {_rel_cf:.3e}); "
        f"the mapped KS reference consumes the POLYNOMIAL, so a mismatched grid would ship a "
        f"KS layer on different Xbar geometry than the DESI/eBOSS columns")

    rows, table = {}, {}
    par_by_survey = []
    for sv in surveys:
        param = str(INF.HCD_ALPHA_PARAMETERIZATION[sv])
        par_by_survey.append(param)
        boost = float(HCD_LLS_SURVEY_BOOST[sv])
        frac = float(HCD_LLS_SURVEY_FRAC_SIGMA[sv])

        if param == "dndx_mapped_v2":
            # ---- KS NATIVE dN/dX (v4): the centre IS the sampled reference ------------------
            assert zp == float(HCD_Z_PIVOT), (
                f"z_pivot {zp} != HCD_Z_PIVOT {float(HCD_Z_PIVOT)}: _ks_dndx_reference pins its "
                f"pivot internally, so a different export pivot would mislabel the KS layer")
            ref, xbar_chk, ref_piv, xbar_piv = _ks_dndx_reference(mu0, xbar_cf, zgrid)
            centre = np.asarray(ref, float)
            np.testing.assert_allclose(np.asarray(xbar_chk, float), Xb, rtol=1e-9, atol=0,
                                       err_msg="_ks_dndx_reference Xbar(z) != the export "
                                               "xbar_grid -- inconsistent geometry")
            # SELF-CHECK (KS): FORWARD round trip at machine precision. The sampled quantity is
            # the paper number; the exact inverse is the CHECK here, not the readout.
            alpha_z = np.asarray(w_c_corrected(jnp.asarray(centre), jnp.asarray(Xb),
                                               jnp.asarray(zgrid)))[..., 1:]
            back = np.asarray(alpha_to_dndx_exact(alpha_z, Xb, zgrid))
            rt = float(np.max(np.abs(back / centre - 1.0)))
            assert rt < 1e-12, (
                f"KS SELF-CHECK FAILED: native-dN/dX centre does not round-trip through "
                f"w_c_corrected -> alpha_to_dndx_exact at machine precision (max rel {rt:.3e}). "
                f"Refusing to ship a KS layer the forward model would not reproduce.")
            # MAPPED alpha-space pivot (derived): guard band + the W2 value pin.
            a_piv = np.asarray(w_c_corrected(jnp.asarray(ref_piv), jnp.asarray(float(xbar_piv)),
                                             jnp.asarray(zp)))[..., 1:]
            INF.assert_ks_mapped_pivot(float(a_piv[0]), f"deployed-centre export v4 {sv}")
            assert abs(float(a_piv[0]) - KS_MAPPED_PIVOT_TARGET) < KS_MAPPED_PIVOT_TOL, (
                f"mapped KS LLS pivot centre {float(a_piv[0])!r} != the W2-pinned "
                f"{KS_MAPPED_PIVOT_TARGET} +- {KS_MAPPED_PIVOT_TOL} (legacy was 0.4302804)")
            # Pivot-AMPLITUDE +-1 sigma edges (the tilt widths kappa/t_sub/t_dla add z-dependent
            # spread NOT shown -- mirroring the legacy pivot-amplitude-band-only convention).
            eps = float(INF.KS_DNDX_SIGMA_EPS)
            msub = float(INF.KS_DNDX_SIGMA_MSUB)
            raw0 = float(INF.KS_DNDX_DLA_RAW_MU0)
            sp0 = float(jax.nn.softplus(raw0))
            amp_q = {n: float(jax.nn.softplus(raw0 + n)) / sp0 for n in (-2.0, -1.0, 1.0, 2.0)}
            lo1 = centre * np.array([np.exp(-eps), np.exp(-msub), amp_q[-1.0]], float)[None, :]
            hi1 = centre * np.array([np.exp(+eps), np.exp(+msub), amp_q[+1.0]], float)[None, :]
            A_lls, g_lls = (float(v) for v in INF.HCD_LIT_DNDX_LAW["LLS"])
            rows[f"deployed_center_dndx_{sv}"] = centre
            rows[f"deployed_lo1_dndx_{sv}"] = lo1
            rows[f"deployed_hi1_dndx_{sv}"] = hi1
            rows[f"deployed_alpha_pivot_mu_{sv}"] = np.asarray(a_piv, float)  # MAPPED (derived)
            # NO deployed_alpha_pivot_sigma_KS / deployed_dla_softplus_quantiles_KS: the mapped
            # prior has no alpha-space Gaussian width (verify_npz_payload refuses them).
            rows["ks_dndx_ref_pivot"] = np.asarray(ref_piv, float)
            rows["ks_xbar_pivot"] = np.float64(xbar_piv)
            rows["ks_lit_dndx_law_lls"] = np.array([A_lls, g_lls], float)
            rows["ks_sigma_eps"] = np.float64(eps)
            rows["ks_sigma_kappa"] = np.float64(INF.KS_DNDX_SIGMA_KAPPA)
            rows["ks_sigma_msub"] = np.float64(msub)
            rows["ks_dla_raw_mu0"] = np.float64(raw0)
            rows["ks_dla_amp_quantiles"] = np.array(
                [amp_q[-2.0], amp_q[-1.0], amp_q[1.0], amp_q[2.0]], float)
            table[sv] = {"boost": boost, "parameterization": param,
                         "boost_space": str(INF.HCD_LLS_BOOST_SPACE[sv]),
                         # 0.40 is the alpha-space SOURCE of sigma_eps (= 0.40/g_LLS), NOT the
                         # deployed width; kept for the lls_frac_sigma_by_survey cross-read.
                         "lls_frac_sigma_alpha_space_source": frac,
                         "dndx_sigma_eps": eps,
                         "dndx_sigma_kappa": float(INF.KS_DNDX_SIGMA_KAPPA),
                         "dndx_sigma_msub": msub,
                         "alpha_pivot_mu_deployed_mapped": np.asarray(a_piv, float).tolist(),
                         "alpha_pivot_mu_closure": mu0.tolist(),
                         "alpha_pivot_sigma_closure": sd0.tolist(),
                         "dndx_ref_pivot": np.asarray(ref_piv, float).tolist(),
                         "xbar_pivot": float(xbar_piv),
                         "lit_dndx_law_lls": [A_lls, g_lls],
                         # the slopes below describe the REFERENCE construction: sub/DLA
                         # components invert the boost-1.0 alpha triple at these slopes; the
                         # LLS dN/dX slope is the lit gamma 2.127 = zslope_deployed[0].
                         "zslope_deployed": slope_deployed.tolist(),
                         "zslope_closure": slope_closure.tolist(),
                         "dndx_lls_center_z3_deployed": float(
                             np.interp(3.0, zgrid, centre[:, 0]))}
            continue

        # ---- ALPHA-parameterized surveys (DESI, eBOSS): unchanged v3 construction ------------
        mu = mu0.copy()
        mu[0] = float(hcd_lls_realfit_alpha_center(Xb3, z=zp, boost=boost))
        sd = sd0.copy()
        sd[0] = mu[0] * frac                        # width rule: sigma/mu held at the new centre
        assert_hcd_pivot_z3(float(mu[0]), z=zp, where=f"deployed-centre export {sv}", boost=boost)
        try:
            rows[f"deployed_center_dndx_{sv}"] = band_curve(mu, zgrid, Xb, zp, slope_deployed)
            lo1 = band_curve(np.clip(mu - sd, 1e-8, None), zgrid, Xb, zp, slope_deployed)
            hi1 = band_curve(mu + sd, zgrid, Xb, zp, slope_deployed)
            # DLA EDGE CONVENTION (adversarial-panel blocker 3, 2026-07-21). The deployed DLA site
            # is softplus(Normal), NOT Gaussian on alpha (sampler_numpyro; closure_legb.py:2541-
            # 2543), so mu +/- sd is the WRONG edge for column 2: it came out 1.81x too narrow
            # above and 1.36x too wide below, against the frozen dla_eff_sd_over_mean = 1.2785.
            # Overwrite column 2 with the softplus quantiles, exactly as the frozen F3 builder
            # does (export_dndx_paper_layer.py), so every column of these edges is the DEPLOYED
            # prior geometry rather than two conventions silently mixed in one array.
            raw_mu = float(_dla_raw_mu(mu[2]))
            for arr, nsig in ((lo1, -1.0), (hi1, +1.0)):
                v = mu.copy()
                v[2] = float(jax.nn.softplus(raw_mu + nsig))
                arr[:, 2] = band_curve(v, zgrid, Xb, zp, slope_deployed)[:, 2]
        except ValueError as e:
            # UNEXPECTED under v4 (fail-loud, do NOT work around): the alpha-parameterized
            # DESI/eBOSS centres and edges are in-domain everywhere on the export grid (boost
            # 1.0); the KS survey no longer routes through the inverse at all (native dN/dX).
            # A refusal here means the prior geometry drifted -- nothing ships.
            raise ValueError(
                f"deployed-centre export REFUSED for survey={sv}: the prior centre/edge alpha "
                f"leaves the exact-inverse domain on this z grid (z in [{zgrid.min():g}, "
                f"{zgrid.max():g}]). Under v4 this is UNEXPECTED for every shipped survey "
                f"(DESI/eBOSS are in-domain everywhere at boost 1.0; the mapped KS centre does "
                f"not route through the inverse) -- the prior geometry has drifted; refusing to "
                f"ship. Underlying domain violation: {e}") from e
        rows[f"deployed_lo1_dndx_{sv}"] = lo1
        rows[f"deployed_hi1_dndx_{sv}"] = hi1
        rows[f"deployed_dla_softplus_quantiles_{sv}"] = np.array(
            [float(jax.nn.softplus(raw_mu + n)) for n in (-2.0, -1.0, 1.0, 2.0)], float)
        rows[f"deployed_alpha_pivot_mu_{sv}"] = mu
        rows[f"deployed_alpha_pivot_sigma_{sv}"] = sd
        table[sv] = {"boost": boost, "lls_frac_sigma": frac,
                     "parameterization": param,
                     "alpha_pivot_mu_deployed": mu.tolist(),
                     "alpha_pivot_sigma_deployed": sd.tolist(),
                     "alpha_pivot_mu_closure": mu0.tolist(),
                     "alpha_pivot_sigma_closure": sd0.tolist(),
                     "zslope_deployed": slope_deployed.tolist(),
                     "zslope_closure": slope_closure.tolist(),
                     "dndx_lls_center_z3_deployed": float(
                         np.interp(3.0, zgrid, rows[f"deployed_center_dndx_{sv}"][:, 0]))}

    payload = dict(
        class_order=np.array(CLS), survey_order=np.array(tuple(surveys)),
        parameterization_by_survey=np.array(par_by_survey),
        zgrid=zgrid, xbar_grid=Xb, z_pivot=np.float64(zp),
        # BOTH slopes, each naming its layer. NEVER a bare `zslope` here (see AMBIGUOUS_KEYS).
        zslope_closure=slope_closure, zslope_deployed=slope_deployed,
        boost_by_survey=np.array([HCD_LLS_SURVEY_BOOST[s] for s in surveys], float),
        lls_frac_sigma_by_survey=np.array([HCD_LLS_SURVEY_FRAC_SIGMA[s] for s in surveys], float),
        closure_center_dndx=np.asarray(closure_center_dndx),
        closure_alpha_pivot_mu=mu0, closure_alpha_pivot_sigma=sd0,
        note=np.str_(NOTE),
        **rows,
    )
    verify_npz_payload(payload, surveys=surveys)
    return payload, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)

    f3p = FROZEN / "f3_layers.npz"
    assert f3p.exists(), f"missing artifact of record {f3p}"
    f3 = np.load(f3p, allow_pickle=True)
    zgrid = np.asarray(f3["zgrid"], float)
    Xb = np.asarray(f3["xbar_grid"], float)
    assert list(f3["class_order"]) == list(CLS), f"class_order drift: {f3['class_order']}"
    zp = float(HCD_Z_PIVOT)
    slope = np.asarray(HCD_INCIDENCE_SLOPE, float)
    # the frozen deg-2 Xbar(z) coefficients feed the mapped KS reference; the grid must BE that
    # polynomial (bit-level in the frozen artifact) or the KS/DESI/eBOSS geometries would differ.
    xbar_cf = np.asarray(f3["xbar_poly_deg2_coeffs"], float)
    assert np.array_equal(np.polyval(xbar_cf, zgrid), Xb), (
        "frozen xbar_grid != polyval(xbar_poly_deg2_coeffs, zgrid) -- the f3 artifact is "
        "internally inconsistent; refusing to build the KS reference on ambiguous Xbar geometry")

    # --- w_c_z3 fiducial: recovered from the frozen pivot mu so no cache rebuild is needed.
    # hcd_incidence_prior(w_c, survey=S) scales the LLS entry by the survey boost and swaps the
    # LLS width; the subDLA/DLA centres are survey-independent. Rather than reconstruct w_c we
    # take the frozen survey=None mu as ground truth and apply the deployed survey geometry to
    # it EXACTLY as inference.hcd_incidence_prior does, then assert the round trip below.
    mu0 = np.asarray(f3["alpha_pivot_mu"], float)
    sd0 = np.asarray(f3["alpha_pivot_sigma"], float)
    assert_hcd_pivot_z3(float(mu0[0]), z=zp, where="deployed-centre export", boost=1.0)

    # --- SELF-CHECK (v3, exact-vs-exact): the closure layer derived through band_curve (the
    # code path every alpha-parameterized deployed curve uses) must equal a DIRECT
    # alpha_to_dndx_exact evaluation of the explicitly-constructed z-resolved alpha to <1e-12
    # on the frozen zgrid/Xbar/pivot inputs. The pre-v3 check asserted byte-level reproduction
    # of the FROZEN f3 arrays, which were built with the APPROXIMATE alpha_to_dndx; that
    # identity is intentionally broken by the exact-inverse migration (readout defect B), so
    # the frozen-vs-exact renorm-level difference is printed for the record, not asserted.
    chk_c = band_curve(mu0, zgrid, Xb, zp, slope)
    chk_lo = band_curve(np.clip(mu0 - sd0, 1e-8, None), zgrid, Xb, zp, slope)
    chk_hi = band_curve(mu0 + sd0, zgrid, Xb, zp, slope)
    for name, got, piv in (("closure centre", chk_c, mu0),
                           ("closure lo1", chk_lo, np.clip(mu0 - sd0, 1e-8, None)),
                           ("closure hi1", chk_hi, mu0 + sd0)):
        az = piv[None, :] * ((1.0 + zgrid)[:, None] / (1.0 + zp)) ** slope[None, :]
        want = np.asarray(alpha_to_dndx_exact(az, Xb, zgrid))
        d = np.max(np.abs(np.asarray(got) - want))
        assert d < 1e-12, (
            f"SELF-CHECK FAILED: band_curve({name}) differs from the direct exact-inverse "
            f"evaluation by {d:.3e}; the deployed curves below would be built by a different "
            f"map. Refusing to write an overlay the paper would draw on the wrong values.")
    rel_frozen = float(np.max(np.abs(chk_c / np.asarray(f3["prior_center_dndx"]) - 1.0)))
    print("[self-check] band_curve == direct alpha_to_dndx_exact to <1e-12  OK")
    print(f"[self-check] exact vs FROZEN (pre-v3, approximate-inverse) closure centre: "
          f"max rel diff {rel_frozen:.3e} (expected renorm-level; NOT asserted -- the frozen "
          f"f3_layers.npz was built with the approximate alpha_to_dndx)")

    # --- deployed-survey geometry, per survey (construction + the closure-vs-deployed key
    # discipline live in build_npz_payload; see its docstring). The KS survey is NATIVE dN/dX
    # (v4): its centre is the mapped reference, its round trip + pivot pin are asserted inside.
    # closure_center_dndx ships the EXACT-map re-derivation (chk_c) -- the like-for-like
    # overlay layer under v3+ -- NOT the verbatim frozen (approximate) prior_center_dndx.
    payload, table = build_npz_payload(zgrid, Xb, mu0, sd0, zp, chk_c, xbar_cf=xbar_cf)

    # KS self-check readout for the record (the asserts live in build_npz_payload).
    ks_centre = np.asarray(payload["deployed_center_dndx_KS"], float)
    ks_alpha_z = np.asarray(w_c_corrected(jnp.asarray(ks_centre), jnp.asarray(Xb),
                                          jnp.asarray(zgrid)))[..., 1:]
    ks_rt = float(np.max(np.abs(np.asarray(alpha_to_dndx_exact(ks_alpha_z, Xb, zgrid))
                                / ks_centre - 1.0)))
    print(f"[self-check] KS native-dN/dX centre forward round trip "
          f"(w_c_corrected -> alpha_to_dndx_exact): max rel {ks_rt:.3e}  (<1e-12 asserted)")
    print(f"[self-check] KS mapped alpha-space pivot = "
          f"{float(payload['deployed_alpha_pivot_mu_KS'][0]):.9f} "
          f"(W2 pin {KS_MAPPED_PIVOT_TARGET} +- {KS_MAPPED_PIVOT_TOL}; legacy postmap was "
          f"0.430280402)")

    # Acceptance targets: eBOSS/DESI from the frozen prior_geometry_table (unchanged, 1e-12);
    # KS = the W2-pinned MAPPED pivot (v4 -- the legacy 0.4302804 target is retired with the
    # alpha-postmap parameterization).
    TARGET = {"eBOSS": (0.17211216067355312, 1e-12),
              "DESI": (0.17211216067355312, 1e-12),
              "KS": (KS_MAPPED_PIVOT_TARGET, KS_MAPPED_PIVOT_TOL)}
    for sv in SURVEYS:
        got = float(np.asarray(payload[f"deployed_alpha_pivot_mu_{sv}"], float)[0])
        want, tol = TARGET[sv]
        assert abs(got - want) < tol, (
            f"deployed LLS pivot for {sv} is {got!r}, expected {want!r} +- {tol:g}. "
            f"Refusing to ship a curve at the wrong centre.")
        print(f"  {sv:6s} param {table[sv]['parameterization']:24s} boost {table[sv]['boost']:.2f}"
              f"  alpha_LLS_pivot {got:.9f}  dN/dX_LLS(z=3) "
              f"{table[sv]['dndx_lls_center_z3_deployed']:.4f}")
    # subDLA/DLA centres: survey-agnostic at the source. Under the EXACT inverse their dN/dX
    # columns are NOT bit-identical to the closure layer (the renorm undo couples every class
    # to the LLS alpha; ~3e-4 relative, asserted <1e-2 -- a genuine centre/slope mislabel is a
    # >=10% effect). v4 KS: its reference sub/DLA columns invert the SAME boost-1.0 triple as
    # DESI/eBOSS, so all three deployed sub/DLA columns must agree bit-exactly across surveys.
    for sv in SURVEYS:
        for j, cname in ((1, "subDLA"), (2, "DLA")):
            a = payload[f"deployed_center_dndx_{sv}"][:, j]
            b = np.asarray(payload["closure_center_dndx"])[:, j]
            rel = float(np.max(np.abs(a / b - 1.0)))
            assert rel < 1e-2, (
                f"{sv} {cname} centre differs from the closure layer by {rel:.3e} relative -- "
                f"far above the exact-inverse renorm coupling (~3e-4); a centre/slope mislabel, "
                f"not the expected class coupling")
    for sv in ("KS", "DESI"):
        d_x = float(np.max(np.abs(payload[f"deployed_center_dndx_{sv}"][:, 1:]
                                  - payload["deployed_center_dndx_eBOSS"][:, 1:])))
        assert d_x < 1e-12, (
            f"deployed sub/DLA columns differ between {sv} and eBOSS by {d_x:.3e} -- they invert "
            f"the identical boost-1.0 triple and must agree bit-exactly")
    print(f"[check] deployed subDLA/DLA centres within the renorm coupling (<1e-2 rel) of the "
          f"closure layer for {list(SURVEYS)}; cross-survey sub/DLA identity (incl. the KS "
          f"reference) to <1e-12  OK")
    print(f"[check] zslope_closure={payload['zslope_closure'].tolist()}  "
          f"zslope_deployed={payload['zslope_deployed'].tolist()}  (both shipped, no bare 'zslope')")

    # IMMUTABLE-DIR REFUSE GUARD (adversarial-panel blocker 2, 2026-07-21). Every sidecar in this
    # series says "regenerate to a NEW dated directory, never edit in place", while the code would
    # happily have overwritten one: the paper pins deployed_centre_2026-07-21 BY SHA and holds a
    # byte-identical copy. Enforce the sentence instead of documenting it.
    if (out / "PROVENANCE.json").exists():
        raise SystemExit(
            f"REFUSING to write into {out}: it already contains a PROVENANCE.json, i.e. it is a "
            f"delivered immutable artifact that a downstream consumer may pin by sha256. "
            f"Regenerate to a NEW dated directory (--out-dir) instead of editing in place.")
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "deployed_centre_layers.npz"
    np.savez(npz, **payload)
    verify_npz_payload(dict(np.load(npz, allow_pickle=True)))   # re-read what was actually written
    sidecar = {
        "schema": "deployed_centre_v4",
        "purpose": ("deployed-survey prior centre in dN/dX for paper F3 (PI 2026-07-21); "
                    "alpha->dN/dX mapped here because the paper side may not reconstruct it; "
                    "v4 = the ONE-TIME re-issue under the corrected model (exact inverse for "
                    "DESI/eBOSS + the native-dN/dX mapped KS prior)"),
        "supersedes": [
            "deployed_centre_2026-07-21 (v1: approximate inverse; bare 'zslope' defect)",
            "deployed_centre_2026-07-22 (v2: approximate inverse -- the KS LLS column silently "
            "SATURATED at 27.631021/Xbar (up to ~33x the lit law at z>=4.5); superseded, do not "
            "read its KS column)",
        ],
        "inverse_map": ("DESI/eBOSS: alpha_to_dndx_exact (EXACT inverse of the w_c_corrected "
                        "forward incl. the 4-class renormalisation; no clip floor; fail-loud "
                        "outside the occupancy simplex). KS (v4): NO inverse map -- the centre "
                        "IS the native dN/dX reference of the deployed mapped prior "
                        "(dndx_mapped_v2); the exact inverse appears only in the KS SELF-CHECK "
                        "round trip. Pre-v3 layers -- including the frozen "
                        "dndx_repin_2026-07-20/f3_layers.npz this export reads its grid from -- "
                        "were built with the APPROXIMATE alpha_to_dndx (renorm ignored, median "
                        "rel err ~2e-3, silent saturation at 27.631021/Xbar outside its domain) "
                        "and differ from v3+ arrays at the renorm level."),
        "answers_request": "request_to_code_agent_2026-07-21_deployed_centre.md",
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # PROVENANCE MUST BE SELF-VERIFYING (adversarial-panel process fix 3, 2026-07-21).
        # A bare dirty=true boolean is what made the previous export unusable to the paper: it
        # could not tell "the producer was edited" from "unrelated files are untracked". Worse,
        # the previous sidecar recorded commit ee685a9, at which THE PRODUCER DOES NOT EXIST --
        # unreproducible by construction, not merely dirty. So record the literal porcelain, and
        # PROVE the recorded commit actually contains this producer, unmodified.
        "dirty": bool(git("status", "--porcelain")),
        "git_status_porcelain": git("status", "--porcelain"),
        "dirty_tracked_files": bool(git("status", "--porcelain", "--untracked-files=no")),
        "producer": str(Path(__file__).relative_to(ROOT)),
        "producer_present_at_commit": subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e",
             f"HEAD:{Path(__file__).relative_to(ROOT)}"]).returncode == 0,
        "producer_unmodified_vs_commit": subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--",
             str(Path(__file__).relative_to(ROOT))]).returncode == 0,
        "code_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
            **{f"hcd_analysis/emulator/{m}.py": sha256(ROOT / "hcd_analysis" / "emulator" / f"{m}.py")
               for m in ("inference", "dndx_wc", "closure_legb", "sampler_numpyro")},
        },
        "run_command": RUN_CMD,
        "inputs_sha256": {str(f3p): sha256(f3p)},
        "outputs_sha256": {npz.name: sha256(npz)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "host": platform.node()},
        "prior_geometry_table": table,
        "key_semantics": {
            "parameterization_by_survey": ("ordered by survey_order; how each survey's deployed "
                                           "HCD prior is parameterized (inference."
                                           "HCD_ALPHA_PARAMETERIZATION): DESI/eBOSS "
                                           "alpha_pivot_powerlaw_v1, KS dndx_mapped_v2"),
            "zslope_closure": ("sim incidence slope closure_legb.HCD_INCIDENCE_SLOPE = "
                               "(2.465, 2.758, 2.366); the slope the CLOSURE band (survey=None) "
                               "in the frozen f3_layers.npz is drawn with"),
            "zslope_deployed": ("deployed real-fit LLS forward z-slope "
                                "inference.HCD_LLS_REALFIT_ZSLOPE = 2.127 in the LLS slot "
                                "(survey-INDEPENDENT, the litWLS slope); subDLA/DLA keep the sim "
                                "slope. The slope the deployed DESI/eBOSS curves use, AND the "
                                "slope triple the KS mapped reference is built from (sub/DLA "
                                "components invert the boost-1.0 alpha triple at these slopes; "
                                "the KS LLS dN/dX slope is the lit gamma 2.127 directly)"),
            "closure_center_dndx": ("survey=None closure centre RE-DERIVED through the v3 exact "
                                    "inverse from the frozen f3_layers pivot/zgrid/Xbar inputs, "
                                    "for a like-for-like overlay. NOT byte-identical to the "
                                    "frozen prior_center_dndx (pre-v3, approximate inverse; "
                                    "renorm-level difference, printed at export)"),
            "closure_alpha_pivot_mu": "frozen f3_layers alpha_pivot_mu (survey=None), (LLS,subDLA,DLA)",
            "deployed_center_dndx_KS": ("v4 NATIVE dN/dX: the deterministic reference "
                                        "closure_legb._ks_dndx_reference the deployed KS mapped "
                                        "prior samples around -- the sampled quantity IS this "
                                        "curve; no inverse map, no saturation possible, no "
                                        "ks_valid truncation needed (the paper may keep its "
                                        "z<=3.70 display choice as presentation). LLS column = "
                                        "2.5 x lit law A=0.01840092, gamma=2.127 (dndx_premap "
                                        "boost); sub/DLA columns = exact inverse of the "
                                        "boost-1.0 deployed centre triple (bit-identical to the "
                                        "DESI/eBOSS sub/DLA columns)"),
            "deployed_hi1_dndx_KS": ("pivot-AMPLITUDE +1 sigma edge: exp(+KS_DNDX_SIGMA_EPS="
                                     "0.5310) x centre on LLS (z-independent log-amplitude "
                                     "shift), exp(+KS_DNDX_SIGMA_MSUB=0.4130) on subDLA, "
                                     "softplus amplitude ratio on DLA. The tilt widths "
                                     "(sigma_kappa=0.6681 LLS exponent, t_sub, t_dla) add "
                                     "z-DEPENDENT spread NOT shown in this band -- mirroring "
                                     "how the legacy export showed the pivot-amplitude band "
                                     "only. lo1 = the same with exp(-.)"),
            "deployed_alpha_pivot_mu_KS": ("the MAPPED alpha-space pivot triple (z=3 occupancy "
                                           "weights of the reference through w_c_corrected; a "
                                           "DERIVED deterministic, not a sampled Gaussian "
                                           "centre). LLS = 0.393235 -- MOVED from the legacy "
                                           "alpha-postmap 0.430280 (v1/v2 artifacts). There is "
                                           "NO deployed_alpha_pivot_sigma_KS: the mapped prior "
                                           "has no alpha-space Gaussian width (widths are "
                                           "ks_sigma_eps/ks_sigma_kappa/ks_sigma_msub + the "
                                           "DLA softplus latent)"),
            "deployed_alpha_pivot_mu_<survey>": ("DESI/eBOSS: deployed pivot centre; LLS = the "
                                                 "K1a-corrected lit dN/dX law 0.172112 x boost, "
                                                 "NOT the closure 0.188119 x boost. subDLA/DLA "
                                                 "= closure values"),
            "ks_dndx_ref_pivot": ("the reference dN/dX triple at z=HCD_Z_PIVOT=3 (the mapped "
                                  "sites multiply this at the pivot); ks_xbar_pivot = Xbar(3)"),
            "ks_lit_dndx_law_lls": "(A, gamma) of the corrected lit LLS law the KS centre deploys",
            "ks_dla_amp_quantiles": ("KS DLA AMPLITUDE quantiles softplus(raw0+n)/softplus(raw0) "
                                     "for n=(-2,-1,+1,+2) -- multiplicative on the reference DLA "
                                     "column (NOT alpha-space softplus values; the legacy "
                                     "deployed_dla_softplus_quantiles_KS key is retired)"),
            "boost_by_survey": ("ordered by survey_order (NOT class_order). NOTE the KS 2.5x "
                                "acts in dN/dX space PRE-map (HCD_LLS_BOOST_SPACE), multiplying "
                                "the lit law, NOT the alpha centre"),
            "lls_frac_sigma_by_survey": ("ordered by survey_order (NOT class_order). For KS "
                                         "(0.40) this alpha-space fractional width is the "
                                         "SOURCE of the mapped log-amplitude width (sigma_eps "
                                         "= 0.40/g_LLS = 0.5310), NOT the deployed width "
                                         "itself"),
            "subDLA_DLA_identity": ("deployed_center_dndx_<survey>[:, 1:] agrees ACROSS surveys "
                                    "bit-exactly (all three invert the same boost-1.0 triple; "
                                    "asserted <1e-12 at export) and agrees with "
                                    "closure_center_dndx[:, 1:] to <1e-2 relative (measured "
                                    "~3e-4; asserted) but is NOT bit-identical to closure under "
                                    "the v3 exact inverse: the renorm undo couples every class "
                                    "to the LLS alpha, which is the only re-centred/re-sloped "
                                    "class. The DLA lo1/hi1 EDGES are not comparable to the "
                                    "frozen prior_lo1/hi1 DLA column, which the frozen builder "
                                    "makes from softplus quantiles rather than mu-/+sd"),
        },
        "schema_changes_vs_v3": [
            "KS IS NATIVE dN/dX (parameterization dndx_mapped_v2, W2 2026-07-22): "
            "deployed_center_dndx_KS is the deterministic reference ks_dndx_ref the deployed KS "
            "prior samples around -- the sampled quantity IS the paper number; the exact inverse "
            "is a self-check (forward round trip <1e-12), not the readout",
            "NO SATURATION POSSIBLE for KS: every mapped draw is structurally inside the "
            "occupancy simplex; the v3 fail-loud KS refusal (centre out-of-simplex at z>=~4.45) "
            "is RETIRED, and the ks_valid truncation is NO LONGER NEEDED for the KS centre "
            "(the paper may keep its z<=3.70 display choice as a presentation decision)",
            "KS alpha-space pivot centre MOVED 0.4302804 (legacy alpha-postmap, v1/v2) -> "
            "0.3932355 (mapped, derived): deployed_alpha_pivot_mu_KS is now the forward-mapped "
            "pivot triple; deployed_alpha_pivot_sigma_KS and "
            "deployed_dla_softplus_quantiles_KS are REMOVED (no alpha-space width exists for "
            "the mapped prior; see ks_sigma_* / ks_dla_amp_quantiles)",
            "deployed_hi1/lo1_dndx_KS band semantics: pivot-AMPLITUDE bands "
            "(exp(+-0.5310) LLS, exp(+-0.4130) subDLA, softplus amplitude ratio DLA); the "
            "exponent/tilt widths (sigma_kappa=0.6681, t_sub, t_dla) add z-dependent spread "
            "NOT shown -- mirroring the legacy pivot-amplitude-band-only convention",
            "ADDED parameterization_by_survey + the ks_* key block (ks_dndx_ref_pivot, "
            "ks_xbar_pivot, ks_lit_dndx_law_lls, ks_sigma_eps/kappa/msub, ks_dla_raw_mu0, "
            "ks_dla_amp_quantiles)",
            "deployed sub/DLA columns now agree ACROSS surveys bit-exactly (KS's reference "
            "inverts the same boost-1.0 triple as DESI/eBOSS; asserted <1e-12)",
        ],
        "schema_changes_vs_v2": [
            "INVERSE MAP: alpha->dN/dX now alpha_to_dndx_exact (exact incl. 4-class renorm, "
            "fail-loud domain) instead of the approximate/saturating alpha_to_dndx",
            "closure_center_dndx: re-derived through the exact map (was: verbatim frozen "
            "prior_center_dndx copy)",
            "deployed_center_dndx_<survey>[:, 1:] no longer bit-identical to the closure "
            "columns (exact-inverse renorm coupling, <1e-2 rel; asserted)",
            "self-check: exact-vs-exact (band_curve vs direct alpha_to_dndx_exact, <1e-12); "
            "frozen-layer byte reproduction no longer asserted (pre-v3 = approximate inverse)",
        ],
        "schema_changes_vs_v1": [
            "REMOVED bare 'zslope' (it was the CLOSURE slope inside a deployed-layer file)",
            "ADDED 'zslope_closure' (2.465,...) and 'zslope_deployed' (2.127,...)",
            "RENAMED 'boost' -> 'boost_by_survey', 'lls_frac_sigma' -> 'lls_frac_sigma_by_survey'",
            "RENAMED 'alpha_pivot_{mu,sigma}_<survey>' -> 'deployed_alpha_pivot_{mu,sigma}_<survey>'",
            "ADDED 'closure_alpha_pivot_mu' / 'closure_alpha_pivot_sigma'",
            "prior_geometry_table rows: 'zslope' -> 'zslope_deployed' (+ 'zslope_closure'), "
            "'alpha_pivot_{mu,sigma}' -> '..._deployed' (+ '..._closure'), "
            "'dndx_center_z3' -> 'dndx_lls_center_z3_deployed'",
        ],
        "self_check": ("(a) derived the survey=None closure layer through band_curve (the code "
                       "path the alpha-parameterized deployed curves use) AND directly through "
                       "alpha_to_dndx_exact on the explicitly-constructed z-resolved alpha, "
                       "asserted agreement to <1e-12 before writing (exact-vs-exact; the pre-v3 "
                       "byte-reproduction of the frozen approximate-inverse f3 arrays is "
                       "intentionally retired, the renorm-level difference is printed); "
                       "(b) round-tripped the KS native-dN/dX centre through the FORWARD map "
                       "(w_c_corrected -> alpha -> alpha_to_dndx_exact -> back) and asserted "
                       "machine precision (<1e-12 relative); (c) pinned the MAPPED KS "
                       "alpha-space pivot to the W2 value 0.393235480 +- 1e-6 "
                       "(assert_ks_mapped_pivot band + the exact pin); (d) asserted the "
                       "deployed subDLA/DLA dN/dX centres agree with the closure ones to <1e-2 "
                       "relative AND across surveys to <1e-12 (the KS reference inverts the "
                       "identical boost-1.0 triple); (e) re-read the written npz and re-audited "
                       "its key set with verify_npz_payload (no closure-vs-deployed ambiguous "
                       "key, no alpha-space width for the mapped survey, every deployed LLS "
                       "column re-derived from its OWN parameterization's labelled parameters). "
                       "Test-pinned by tests/test_export_deployed_centre.py"),
        "artifact_of_record_untouched": str(FROZEN),
        "blind_status": ("BLIND-SAFE: prior geometry only; no data, no posterior, no real-data "
                         "n_s/A_p in any array or field"),
        "chain_of_record_status": ("presentation-layer derived artifact; not a chain. Immutable: "
                                   "regenerate to a NEW dated directory, never edit in place."),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    print(f"\n[export] wrote {out}")
    print(f"  {npz.name}  sha256={sidecar['outputs_sha256'][npz.name][:16]}...")
    print(f"  commit={sidecar['commit'][:12]} dirty={sidecar['dirty']}")


if __name__ == "__main__":
    main()
