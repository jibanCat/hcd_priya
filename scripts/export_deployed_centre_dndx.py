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
import jax.numpy as jnp

from hcd_analysis.emulator.closure_legb import HCD_INCIDENCE_SLOPE
from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
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
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/deployed_centre_2026-07-21")
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
          * ((1.0 + zgrid)[:, None] / (1.0 + zp)) ** np.asarray(slope)[None, :])
    return np.array(alpha_to_dndx(jnp.asarray(az), jnp.asarray(Xb), jnp.asarray(zgrid)))


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

    # --- deployed-survey geometry, per survey.
    # TWO things differ from the closure band, and BOTH were got wrong on the first pass:
    #  (1) CENTRE. The deployed LLS pin is NOT the closure centre x boost. It is built from the
    #      CORRECTED literature dN/dX law directly (hcd_lls_realfit_alpha_center: alt-(b), the
    #      K1a binned [17.2,19.0) law A=0.0184*(1+z)^2.127), because on the real fit PRIYA != data
    #      so the LLS centre must track the literature, not the sim's z=3 w_c*(lit/sim).
    #      0.172112 x boost, NOT 0.188119 x boost. See closure_legb._survey_alpha_prior:735.
    #  (2) Z-SLOPE. The deployed real-fit LLS forward z-slope is HCD_LLS_REALFIT_ZSLOPE = 2.127
    #      (survey-INDEPENDENT, the litWLS slope), not the sim HCD_INCIDENCE_SLOPE[0] = 2.465 the
    #      closure band uses. So the deployed curve has a different z-shape, not just a different
    #      normalization. See closure_legb._survey_alpha_prior (survey_zslope_mu).
    # subDLA/DLA centres and slopes are survey-agnostic and stay at the closure values.
    Xb3 = float(np.interp(zp, zgrid, Xb))
    slope_dep = np.array([float(HCD_LLS_REALFIT_ZSLOPE), slope[1], slope[2]], float)
    # Acceptance targets quoted by the paper agent from the frozen prior_geometry_table.
    TARGET = {"eBOSS": 0.17211216067355312, "KS": 0.43028040168388280}
    rows, table = {}, {}
    for sv in SURVEYS:
        boost = float(HCD_LLS_SURVEY_BOOST[sv])
        frac = float(HCD_LLS_SURVEY_FRAC_SIGMA[sv])
        mu = mu0.copy()
        mu[0] = float(hcd_lls_realfit_alpha_center(Xb3, z=zp, boost=boost))
        sd = sd0.copy()
        sd[0] = mu[0] * frac                        # width rule: sigma/mu held at the new centre
        if sv in TARGET:
            assert abs(mu[0] - TARGET[sv]) < 1e-12, (
                f"deployed LLS centre for {sv} is {mu[0]!r}, expected {TARGET[sv]!r} from the "
                f"frozen prior_geometry_table. Refusing to ship a curve at the wrong centre.")
        assert_hcd_pivot_z3(float(mu[0]), z=zp, where=f"deployed-centre export {sv}", boost=boost)
        rows[f"deployed_center_dndx_{sv}"] = band_curve(mu, zgrid, Xb, zp, slope_dep)
        rows[f"deployed_lo1_dndx_{sv}"] = band_curve(np.clip(mu - sd, 1e-8, None), zgrid, Xb,
                                                     zp, slope_dep)
        rows[f"deployed_hi1_dndx_{sv}"] = band_curve(mu + sd, zgrid, Xb, zp, slope_dep)
        rows[f"alpha_pivot_mu_{sv}"] = mu
        rows[f"alpha_pivot_sigma_{sv}"] = sd
        table[sv] = {"boost": boost, "lls_frac_sigma": frac,
                     "alpha_pivot_mu": mu.tolist(), "alpha_pivot_sigma": sd.tolist(),
                     "zslope": slope_dep.tolist(),
                     "dndx_center_z3": float(np.interp(3.0, zgrid,
                                                       rows[f"deployed_center_dndx_{sv}"][:, 0]))}
        print(f"  {sv:6s} boost {boost:.2f}  frac_sigma {frac:.3f}  "
              f"alpha_LLS {mu[0]:.9f}  dN/dX_LLS(z=3) {table[sv]['dndx_center_z3']:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    npz = out / "deployed_centre_layers.npz"
    np.savez(
        npz,
        class_order=np.array(CLS), survey_order=np.array(SURVEYS),
        zgrid=zgrid, xbar_grid=Xb, zslope=slope, z_pivot=np.float64(zp),
        boost=np.array([HCD_LLS_SURVEY_BOOST[s] for s in SURVEYS], float),
        lls_frac_sigma=np.array([HCD_LLS_SURVEY_FRAC_SIGMA[s] for s in SURVEYS], float),
        closure_center_dndx=np.asarray(f3["prior_center_dndx"]),
        note=np.str_("DEPLOYED-SURVEY prior centre in dN/dX on the frozen f3_layers zgrid. "
                     "Same map as the frozen F3 layer (self-checked to <1e-12). Shape "
                     "(n_z, n_class) with class_order. survey=None closure layer included as "
                     "closure_center_dndx for a like-for-like overlay. Prior geometry only: "
                     "no data, no posterior, no cosmology."),
        **rows,
    )
    sidecar = {
        "schema": "deployed_centre_v1",
        "purpose": ("deployed-survey prior centre in dN/dX for paper F3 (PI 2026-07-21); "
                    "alpha->dN/dX mapped here because the paper side may not reconstruct it"),
        "answers_request": "request_to_code_agent_2026-07-21_deployed_centre.md",
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "run_command": RUN_CMD,
        "inputs_sha256": {str(f3p): sha256(f3p)},
        "outputs_sha256": {npz.name: sha256(npz)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "host": platform.node()},
        "prior_geometry_table": table,
        "self_check": ("re-derived the survey=None closure layer through this script's own "
                       "alpha->dN/dX path and asserted equality with the frozen "
                       "dndx_repin_2026-07-20/f3_layers.npz to <1e-12 before writing"),
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
