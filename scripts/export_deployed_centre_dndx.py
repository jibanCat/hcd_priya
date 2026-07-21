"""Export the DEPLOYED-SURVEY prior centre in dN/dX for paper figure F3.

Answers ~/Latex/HCDEmulatorPaper/request_to_code_agent_2026-07-21_deployed_centre.md
(PI-authorized 2026-07-21). F3 draws the closure/cosmic-average band (survey=None); the PI
asked for the DEPLOYED SURVEY prior centre alongside it. The frozen export gives those
centres in ALPHA space only, and mapping alpha -> dN/dX needs the exact inverse
(`dndx_wc.alpha_to_dndx` with Xbar(z) and the telescoping/(1+delta_c) step), which the paper
side is forbidden to reconstruct locally. So the map is done HERE and shipped as an array.

Delivery option (a) of the request: a small NEW dated export dir. The artifact of record
`dndx_repin_2026-07-20/` is READ ONLY and is never touched -- F3 and the money figure both
assert against its hashes at build time.

CONSTRUCTION IS IDENTICAL to the frozen F3 layer (`scripts/export_dndx_paper_layer.py`),
with `survey=None` swapped for the deployed survey key:
    alpha_c(z) = mu_c * ((1+z)/(1+z_pivot))**zslope_c        then alpha_to_dndx(alpha, Xbar, z)
z grid and Xbar(z) are READ FROM the frozen `f3_layers.npz` rather than recomputed, so the
paper can overlay these curves on the existing layer without an interpolation step.

SELF-CHECK (fail-loud, the whole point): before writing anything, this script re-derives the
survey=None layer through its OWN code path and asserts it reproduces the frozen
`prior_center_dndx` / `prior_lo1_dndx` / `prior_hi1_dndx` to within 1e-12. If that assert
passes, the deployed-survey curves below were built by the identical map. If it fails, the
export refuses to write rather than shipping a curve the paper would overlay wrongly.

KEY DISCIPLINE (defect fix 2026-07-21): every array is either a CLOSURE or a DEPLOYED quantity and
the two differ in BOTH centre and z-shape, so no key may leave its layer to the reader's guess. The
export ships `zslope_closure` (2.465, the sim slope the closure band uses) AND `zslope_deployed`
(2.127, the litWLS LLS slope the deployed curves use) and NEVER a bare `zslope`. `verify_npz_payload`
re-reads the written file and refuses any ambiguous key -- the original defect was a silently failed
sed edit whose resulting key set nobody checked.

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

from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
from hcd_analysis.emulator.sampler_numpyro import _dla_raw_mu
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
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/deployed_centre_2026-07-22")
SURVEYS = ("eBOSS", "KS", "DESI")
RUN_CMD = ("cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu "
           "/home/mfho/.conda/envs/emu-jax/bin/python3 scripts/export_deployed_centre_dndx.py")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def band_curve(alpha_pivot_vec, zgrid, Xb, zp, slope):
    """The frozen F3 map, verbatim: pivot alpha -> z-resolved alpha -> dN/dX."""
    az = (np.asarray(alpha_pivot_vec)[None, :]
          * ((1.0 + np.asarray(zgrid, float))[:, None] / (1.0 + zp)) ** np.asarray(slope)[None, :])
    return np.array(alpha_to_dndx(jnp.asarray(az), jnp.asarray(Xb), jnp.asarray(zgrid)))


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
    BOTH centre and z-shape; a key that does not say which is a mislabelling waiting to happen."""
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
    # ARRAY-vs-LABEL POSTCONDITION (adversarial-panel blocker 4, 2026-07-21). Everything above
    # checks NAMES and scalar slope values. A mutant with perfect key names, perfect slope values,
    # and deployed curves rebuilt from the CLOSURE slope passed all of it -- which is the very
    # defect class this file exists to prevent, one layer up. So re-derive one representative
    # deployed column FROM THE LABELLED PARAMETERS and require it to match the shipped array. All
    # inputs are already in the payload, so this is self-contained.
    if {"zgrid", "xbar_grid", "z_pivot", "survey_order"} <= keys:
        zg = np.asarray(payload["zgrid"], float)
        xb = np.asarray(payload["xbar_grid"], float)
        zpv = float(np.asarray(payload["z_pivot"]))
        for sv in [str(s) for s in np.asarray(payload["survey_order"])]:
            mu_sv = np.asarray(payload[f"deployed_alpha_pivot_mu_{sv}"], float)
            want = band_curve(mu_sv, zg, xb, zpv, dep)[:, 0]
            got = np.asarray(payload[f"deployed_center_dndx_{sv}"], float)[:, 0]
            assert np.array_equal(got, want), (
                f"deployed_center_dndx_{sv}[:, LLS] does NOT reproduce from the labelled "
                f"zslope_deployed + deployed_alpha_pivot_mu_{sv}. The arrays and the labels "
                f"disagree: max|diff| = {np.max(np.abs(got - want)):.6e}. Refusing to ship a file "
                f"whose keys describe a curve it does not contain.")
    if "survey_order" in keys:
        for sv in [str(s) for s in np.asarray(payload["survey_order"])]:
            for stem in ("alpha_pivot_mu", "alpha_pivot_sigma"):
                assert f"{stem}_{sv}" not in keys, (
                    f"{stem}_{sv!r} does not say whether it is the closure or the deployed pivot; "
                    f"use deployed_{stem}_{sv}")
                assert f"deployed_{stem}_{sv}" in keys, f"missing deployed_{stem}_{sv}"
            for stem in ("center", "lo1", "hi1"):
                assert f"deployed_{stem}_dndx_{sv}" in keys, f"missing deployed_{stem}_dndx_{sv}"
    return True


NOTE = (
    "DEPLOYED-SURVEY prior centre in dN/dX on the frozen f3_layers zgrid. Same alpha->dN/dX map as "
    "the frozen F3 layer (self-checked to <1e-12). Arrays are (n_z, n_class) with class_order. "
    "CLOSURE vs DEPLOYED differ in BOTH centre and z-shape: the deployed LLS centre is the "
    "K1a-corrected literature dN/dX law (0.172112 x boost), NOT the closure centre (0.188119) x "
    "boost; and the deployed LLS forward z-slope is zslope_deployed[0]=2.127 (the litWLS slope), "
    "NOT the sim zslope_closure[0]=2.465 the closure band uses. subDLA/DLA centres and slopes are "
    "survey-agnostic, so their dN/dX columns are BIT-identical to closure_center_dndx. The "
    "survey=None closure layer is included as closure_center_dndx for a like-for-like overlay. "
    "Prior geometry only: no data, no posterior, no cosmology.")


def build_npz_payload(zgrid, xbar_grid, mu_closure, sd_closure, z_pivot, closure_center_dndx,
                      surveys=SURVEYS):
    """Assemble (npz payload dict, PROVENANCE prior_geometry_table) for the deployed layer.

    TWO things differ from the closure band, and BOTH were got wrong on the first pass:
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

    rows, table = {}, {}
    for sv in surveys:
        boost = float(HCD_LLS_SURVEY_BOOST[sv])
        frac = float(HCD_LLS_SURVEY_FRAC_SIGMA[sv])
        mu = mu0.copy()
        mu[0] = float(hcd_lls_realfit_alpha_center(Xb3, z=zp, boost=boost))
        sd = sd0.copy()
        sd[0] = mu[0] * frac                        # width rule: sigma/mu held at the new centre
        assert_hcd_pivot_z3(float(mu[0]), z=zp, where=f"deployed-centre export {sv}", boost=boost)
        rows[f"deployed_center_dndx_{sv}"] = band_curve(mu, zgrid, Xb, zp, slope_deployed)
        lo1 = band_curve(np.clip(mu - sd, 1e-8, None), zgrid, Xb, zp, slope_deployed)
        hi1 = band_curve(mu + sd, zgrid, Xb, zp, slope_deployed)
        # DLA EDGE CONVENTION (adversarial-panel blocker 3, 2026-07-21). The deployed DLA site is
        # softplus(Normal), NOT Gaussian on alpha (sampler_numpyro; closure_legb.py:2541-2543), so
        # mu +/- sd is the WRONG edge for column 2: it came out 1.81x too narrow above and 1.36x
        # too wide below, against the frozen dla_eff_sd_over_mean = 1.2785. Overwrite column 2 with
        # the softplus quantiles, exactly as the frozen F3 builder does
        # (export_dndx_paper_layer.py:198-211), so every column of these edges is the DEPLOYED
        # prior geometry rather than two conventions silently mixed in one array.
        raw_mu = float(_dla_raw_mu(mu[2]))
        for arr, nsig in ((lo1, -1.0), (hi1, +1.0)):
            v = mu.copy()
            v[2] = float(jax.nn.softplus(raw_mu + nsig))
            arr[:, 2] = band_curve(v, zgrid, Xb, zp, slope_deployed)[:, 2]
        rows[f"deployed_lo1_dndx_{sv}"] = lo1
        rows[f"deployed_hi1_dndx_{sv}"] = hi1
        rows[f"deployed_dla_softplus_quantiles_{sv}"] = np.array(
            [float(jax.nn.softplus(raw_mu + n)) for n in (-2.0, -1.0, 1.0, 2.0)], float)
        rows[f"deployed_alpha_pivot_mu_{sv}"] = mu
        rows[f"deployed_alpha_pivot_sigma_{sv}"] = sd
        table[sv] = {"boost": boost, "lls_frac_sigma": frac,
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

    # --- w_c_z3 fiducial: recovered from the frozen pivot mu so no cache rebuild is needed.
    # hcd_incidence_prior(w_c, survey=S) scales the LLS entry by the survey boost and swaps the
    # LLS width; the subDLA/DLA centres are survey-independent. Rather than reconstruct w_c we
    # take the frozen survey=None mu as ground truth and apply the deployed survey geometry to
    # it EXACTLY as inference.hcd_incidence_prior does, then assert the round trip below.
    mu0 = np.asarray(f3["alpha_pivot_mu"], float)
    sd0 = np.asarray(f3["alpha_pivot_sigma"], float)
    assert_hcd_pivot_z3(float(mu0[0]), z=zp, where="deployed-centre export", boost=1.0)

    # --- SELF-CHECK: reproduce the frozen closure layer through THIS code path.
    chk_c = band_curve(mu0, zgrid, Xb, zp, slope)
    chk_lo = band_curve(np.clip(mu0 - sd0, 1e-8, None), zgrid, Xb, zp, slope)
    chk_hi = band_curve(mu0 + sd0, zgrid, Xb, zp, slope)
    for name, got, want in (("prior_center_dndx", chk_c, f3["prior_center_dndx"]),
                            ("prior_lo1_dndx", chk_lo, f3["prior_lo1_dndx"]),
                            ("prior_hi1_dndx", chk_hi, f3["prior_hi1_dndx"])):
        # DLA column uses softplus quantiles in the frozen builder, not mu -/+ sd; compare the
        # two classes whose edges are plain Gaussian, and the CENTRE on all three.
        cols = slice(0, 3) if name == "prior_center_dndx" else slice(0, 2)
        d = np.max(np.abs(np.asarray(got)[:, cols] - np.asarray(want)[:, cols]))
        assert d < 1e-12, (
            f"SELF-CHECK FAILED: re-derived {name} differs from the frozen artifact by {d:.3e}. "
            f"The alpha->dN/dX map here is NOT the one that built F3; refusing to write an "
            f"overlay the paper would draw on the wrong grid.")
    print(f"[self-check] re-derived closure layer matches {f3p.name} to <1e-12  OK")

    # --- deployed-survey geometry, per survey (construction + the closure-vs-deployed key
    # discipline live in build_npz_payload; see its docstring for the two differences).
    payload, table = build_npz_payload(zgrid, Xb, mu0, sd0, zp,
                                       np.asarray(f3["prior_center_dndx"]))

    # Acceptance targets quoted by the paper agent from the frozen prior_geometry_table.
    TARGET = {"eBOSS": 0.17211216067355312, "KS": 0.43028040168388280}
    for sv in SURVEYS:
        got = float(table[sv]["alpha_pivot_mu_deployed"][0])
        if sv in TARGET:
            assert abs(got - TARGET[sv]) < 1e-12, (
                f"deployed LLS centre for {sv} is {got!r}, expected {TARGET[sv]!r} from the "
                f"frozen prior_geometry_table. Refusing to ship a curve at the wrong centre.")
        print(f"  {sv:6s} boost {table[sv]['boost']:.2f}  frac_sigma {table[sv]['lls_frac_sigma']:.3f}"
              f"  alpha_LLS {got:.9f}  "
              f"dN/dX_LLS(z=3) {table[sv]['dndx_lls_center_z3_deployed']:.4f}")
    # subDLA/DLA are survey-agnostic in BOTH centre and slope, so the LLS-only overlay the paper
    # draws requires their dN/dX columns to be BIT-identical to the closure layer. Assert it.
    for sv in SURVEYS:
        for j, cname in ((1, "subDLA"), (2, "DLA")):
            a = payload[f"deployed_center_dndx_{sv}"][:, j]
            b = np.asarray(f3["prior_center_dndx"])[:, j]
            assert np.array_equal(a, b), (
                f"{sv} {cname} centre is not bit-identical to the closure layer "
                f"(max|diff|={np.max(np.abs(a - b)):.3e}); the paper's LLS-only overlay assumes it")
    print(f"[check] deployed subDLA/DLA centres bit-identical to closure for {list(SURVEYS)}  OK")
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
        "schema": "deployed_centre_v2",
        "purpose": ("deployed-survey prior centre in dN/dX for paper F3 (PI 2026-07-21); "
                    "alpha->dN/dX mapped here because the paper side may not reconstruct it"),
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
            "zslope_closure": ("sim incidence slope closure_legb.HCD_INCIDENCE_SLOPE = "
                               "(2.465, 2.758, 2.366); the slope the CLOSURE band (survey=None) "
                               "in the frozen f3_layers.npz is drawn with"),
            "zslope_deployed": ("deployed real-fit LLS forward z-slope "
                                "inference.HCD_LLS_REALFIT_ZSLOPE = 2.127 in the LLS slot "
                                "(survey-INDEPENDENT, the litWLS slope); subDLA/DLA keep the sim "
                                "slope. The slope every deployed_*_dndx_<survey> array uses"),
            "closure_center_dndx": ("verbatim copy of the frozen f3_layers prior_center_dndx "
                                    "(survey=None), for a like-for-like overlay"),
            "closure_alpha_pivot_mu": "frozen f3_layers alpha_pivot_mu (survey=None), (LLS,subDLA,DLA)",
            "deployed_alpha_pivot_mu_<survey>": ("deployed pivot centre; LLS = the K1a-corrected "
                                                 "lit dN/dX law 0.172112 x boost, NOT the closure "
                                                 "0.188119 x boost. subDLA/DLA = closure values"),
            "boost_by_survey": "ordered by survey_order (NOT class_order)",
            "lls_frac_sigma_by_survey": "ordered by survey_order (NOT class_order)",
            "subDLA_DLA_identity": ("deployed_center_dndx_<survey>[:, 1:] is BIT-identical to "
                                    "closure_center_dndx[:, 1:] for every survey (asserted at "
                                    "export): only the LLS class is re-centred/re-sloped. The DLA "
                                    "lo1/hi1 EDGES are not comparable to the frozen prior_lo1/hi1 "
                                    "DLA column, which the frozen builder makes from softplus "
                                    "quantiles rather than mu-/+sd"),
        },
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
        "self_check": ("(a) re-derived the survey=None closure layer through this script's own "
                       "alpha->dN/dX path and asserted equality with the frozen "
                       "dndx_repin_2026-07-20/f3_layers.npz to <1e-12 before writing; "
                       "(b) asserted the deployed subDLA/DLA dN/dX centres are BIT-identical to "
                       "the closure ones for every survey; (c) re-read the written npz and "
                       "re-audited its key set with verify_npz_payload (no closure-vs-deployed "
                       "ambiguous key can ship). Test-pinned by tests/test_export_deployed_centre.py"),
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
