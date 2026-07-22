"""Frozen, self-describing corrected dN/dX export for the HCD-emulator paper
(PI directive 2026-07-20: option A re-pin + artifact-only sourcing).

Produces, in --out-dir (default: the notes-repo artifact path):
  dndx_corrected_export.json   the sec-3.1 export: per-class literature points (value,
                               err, source, cite, ESTIMAND), deployed laws + functional
                               form, PRIYA sim incidence, deployed prior bands labelled
                               by geometry, the K1a kernel record + one-sided bracket,
                               and the sec-3.2 closure-vs-survey prior-geometry table.
  f3_layers.npz                figure-ready numerical layers for F3 fig_dndx_priors
                               (survey=None closure geometry, z in [2.2,4.6] n=49),
                               computed through the same deployed code path as the
                               shipped figure script.
  money_row1_layers.npz        figure-ready layers for money-figure row 1: corrected
                               lit + law curves, plus the propagated prior 68% band in
                               TWO variants: (a) campaign-as-run (the June arms' own
                               pkl-recorded sampler priors, plain-Normal geometry;
                               consistent with the PRESERVED row-2 contours) and
                               (b) corrected-closure (current deployed survey=None
                               prior through the closure_legb TruncatedNormal/softplus
                               geometry). The choice between them is a paper/PI
                               presentation decision, flagged in the handoff.
  PROVENANCE.json              sidecar: commit, branch, dirty flag, run command, env,
                               input sha256s, output sha256s, blind status,
                               chain-of-record status.

Blind status: BLIND-SAFE. Contains NO real-data cosmology values (literature +
prior-constant + closure-mock-truth quantities only).

Run (login node, ~1 min):
  cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/export_dndx_paper_layer.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_CMD = ("cd /home/mfho/hcd_priya && PYTHONNOUSERSITE=1 PYTHONPATH=. JAX_PLATFORMS=cpu "
           "CUDA_VISIBLE_DEVICES=\"\" /home/mfho/.conda/envs/emu-jax/bin/python3 "
           "scripts/export_dndx_paper_layer.py")
DEFAULT_OUT = Path("/home/mfho/hcd_priya_notes/artifacts/paper_exports/dndx_repin_2026-07-20")
ON_HI_PKL = Path("/scratch/cavestru_root/cavestru1/mfho/desi_hcd_prior_on_hi/mock_0000.pkl")
CLS = ("LLS", "subDLA", "DLA")

# paper figure grids (from the paper configs, restated here so the export is
# self-contained; fig_dndx_priors.json z_range [2.2,4.6] n_z 49; money z_range 200)
F3_Z = np.linspace(2.2, 4.6, 49)
ROW1_ZFINE = np.linspace(2.2, 4.6, 200)
ROW1_BAND_N = 20000
ROW1_BAND_SEED = 20260717
ROW1_BAND_PCTS = (16.0, 84.0)


def sha256(path, limit_mb=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True).stdout.strip()


def xbar_fit(d):
    """Cache Xbar(z) deg-2 fit; same construction as hcd_pivot_wc_and_xbar and the
    shipped fig_dndx_priors script."""
    gid = np.asarray(d["snap_group_idx"])
    zrow = np.asarray(d["z_grid"])
    wc = np.asarray(d["w_c_cache"])
    Xtot = np.asarray(d["snap_total_path_dX"])
    dndx = np.asarray(d["snap_dNdX"])
    Ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(Ng)])
    wc_g = np.array([np.nanmedian(wc[gid == g], axis=0) for g in range(Ng)])
    mu_sum = -np.log(np.clip(wc_g[:, 0], 1e-6, None))
    Nsl = np.where(mu_sum > 0, dndx.sum(axis=1) * Xtot / mu_sum, np.nan)
    Xbar = Xtot / Nsl
    ok = np.isfinite(Xbar) & (zg >= 2.1) & (zg <= 4.7)
    cf = np.polyfit(zg[ok], Xbar[ok], 2)
    return (lambda z: np.polyval(cf, np.asarray(z))), cf, zg, dndx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import hcd_analysis.emulator  # noqa: F401 (x64 before jax use)
    import jax
    import jax.numpy as jnp
    from hcd_analysis.emulator.data import load_cache
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx_exact
    from hcd_analysis.emulator import lit_dndx
    from hcd_analysis.emulator.lit_dndx import (
        lit_points_for_display, assert_display_labels, ESTIMAND_ID, ESTIMAND_LABEL)
    from hcd_analysis.emulator.inference import (
        hcd_incidence_prior, hcd_lls_realfit_alpha_center, assert_hcd_pivot_z3,
        assert_dndx_law_estimand, hcd_prior_signature,
        HCD_Z_PIVOT, HCD_LIT_DNDX_LAW, HCD_LIT_DNDX_ESTIMAND, HCD_DLA_RESIDUAL_FRAC,
        HCD_PRIOR_FRAC_SIGMA, HCD_LLS_SURVEY_BOOST, HCD_LLS_SURVEY_FRAC_SIGMA,
        HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X)
    from hcd_analysis.emulator.closure_legb import (
        HCD_INCIDENCE_SLOPE, HCD_LLS_REALFIT_ZSLOPE, ZSLOPE_PRIOR_SIGMA,
        hcd_pivot_wc_and_xbar)
    from hcd_analysis.emulator.sampler_numpyro import _dla_raw_mu

    # ---------------- guards first (the export refuses to build mislabelled) ------
    for c in CLS:
        assert_dndx_law_estimand(c, HCD_LIT_DNDX_ESTIMAND[c], "export law boundary")
        lit_dndx.assert_display_estimand(c, ESTIMAND_ID[c], "export display boundary")
    assert_display_labels(ESTIMAND_LABEL, "export labels")

    # ---------------- inputs ------------------------------------------------------
    cache_path = ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5"
    deriv_json = ROOT / "hcd_analysis/emulator/hcd_lit_dndx_corrected.json"
    j = json.loads(deriv_json.read_text())
    d = load_cache(str(cache_path))
    w_c_z3, Xbar_z3 = hcd_pivot_wc_and_xbar(d)
    xbar_fn, xbar_coef, zg_snap, dndx_snap = xbar_fit(d)
    zp = float(HCD_Z_PIVOT)

    with open(ON_HI_PKL, "rb") as fh:
        ON = pickle.load(fh)
    assert tuple(ON["dndx_class_order"]) == CLS

    # ---------------- literature points + laws (sec 3.1 items 1-2) ---------------
    LIT = lit_points_for_display()
    # per-source-group cites from the module's own ARXIV registry (never re-typed)
    AX = lit_dndx.ARXIV
    cites = {
        "LLS": {"POW10": AX["POW10"], "OMeara13": AX["OMEARA13"],
                "Fumagalli13": AX["FUMAGALLI13"]},
        "subDLA": {"Zafar13": AX["ZAFAR13"]},
        "DLA": {"ProchaskaWolfe09": AX["PW09"]},
    }
    lit_block = {}
    for c in CLS:
        zl, vl, el, src = LIT[c]
        lit_block[c] = {
            "z": np.asarray(zl, float).tolist(),
            "value": np.asarray(vl, float).tolist(),
            "err": np.asarray(el, float).tolist(),
            "source": str(src),
            "cites_arxiv": cites[c],
            "estimand_id": ESTIMAND_ID[c],
            "estimand_label": ESTIMAND_LABEL[c],
        }
    laws_block = {c: {"A": float(HCD_LIT_DNDX_LAW[c][0]),
                      "gamma": float(HCD_LIT_DNDX_LAW[c][1]),
                      "form": "dN/dX = A * (1+z)**gamma",
                      "estimand_id": HCD_LIT_DNDX_ESTIMAND[c]} for c in CLS}

    # ---------------- PRIYA sim incidence (sec 3.1 item 3) -----------------------
    sim_block = {}
    for jcls, c in enumerate(CLS):
        zb, vb = [], []
        for zz in np.unique(np.round(zg_snap, 1)):
            sel = np.isclose(zg_snap, zz, atol=0.05) & (dndx_snap[:, jcls] > 0)
            if sel.sum() >= 2:
                zb.append(float(zz))
                vb.append(float(np.nanmean(dndx_snap[sel, jcls])))
        sim_block[c] = {"z": zb, "dndx_mean": vb,
                        "note": "mean over positive LF-cache snap_dNdX entries per z"}

    # ---------------- F3 layers: survey=None closure geometry ---------------------
    mu, sd = map(np.asarray, hcd_incidence_prior(jnp.asarray(w_c_z3), z=zp, survey=None))
    assert_hcd_pivot_z3(float(mu[0]), z=zp, where="export survey=None", boost=1.0)
    s_c = np.asarray(HCD_INCIDENCE_SLOPE, float)
    Xb_f3 = np.asarray(xbar_fn(F3_Z))

    def band_curve(alpha_pivot_vec, zgrid, Xb):
        # EXACT inverse (readout defect B, 2026-07-22), mode="raise": a prior centre/edge
        # outside the occupancy simplex refuses loudly instead of silently saturating.
        az = (np.asarray(alpha_pivot_vec)[None, :]
              * ((1.0 + zgrid)[:, None] / (1.0 + zp)) ** s_c[None, :])
        return np.asarray(alpha_to_dndx_exact(az, np.asarray(Xb, float),
                                              np.asarray(zgrid, float)))

    raw_mu = float(_dla_raw_mu(mu[2]))
    q = lambda nsig: float(jax.nn.softplus(raw_mu + nsig))
    dla_q = {n: q(n) for n in (-2.0, -1.0, 1.0, 2.0)}
    rng = np.random.default_rng(0)
    samp = np.log1p(np.exp(np.minimum(raw_mu + rng.standard_normal(400_000), 30)))
    dla_eff_width = float(samp.std() / samp.mean())

    center = band_curve(mu, F3_Z, Xb_f3)
    lo1 = band_curve(np.clip(mu - sd, 1e-8, None), F3_Z, Xb_f3)
    hi1 = band_curve(mu + sd, F3_Z, Xb_f3)
    lo2 = band_curve(np.clip(mu - 2 * sd, 1e-8, None), F3_Z, Xb_f3)
    hi2 = band_curve(mu + 2 * sd, F3_Z, Xb_f3)
    for arr, nsig in ((lo1, -1.0), (hi1, 1.0), (lo2, -2.0), (hi2, 2.0)):
        v = mu.copy()
        v[2] = dla_q[nsig]
        arr[:, 2] = band_curve(v, F3_Z, Xb_f3)[:, 2]

    np.savez(out / "f3_layers.npz",
             class_order=np.array(CLS), zgrid=F3_Z, xbar_grid=Xb_f3,
             xbar_poly_deg2_coeffs=np.asarray(xbar_coef),
             prior_center_dndx=center, prior_lo1_dndx=lo1, prior_hi1_dndx=hi1,
             prior_lo2_dndx=lo2, prior_hi2_dndx=hi2,
             alpha_pivot_mu=mu, alpha_pivot_sigma=sd, zslope=s_c,
             dla_raw_mu=np.float64(raw_mu),
             dla_softplus_quantiles=np.array([dla_q[n] for n in (-2., -1., 1., 2.)]),
             dla_eff_sd_over_mean=np.float64(dla_eff_width),
             law_A=np.array([HCD_LIT_DNDX_LAW[c][0] for c in CLS]),
             law_gamma=np.array([HCD_LIT_DNDX_LAW[c][1] for c in CLS]),
             dla_residual_frac=np.float64(HCD_DLA_RESIDUAL_FRAC),
             sim_z_LLS=np.array(sim_block["LLS"]["z"]),
             sim_dndx_LLS=np.array(sim_block["LLS"]["dndx_mean"]),
             sim_z_subDLA=np.array(sim_block["subDLA"]["z"]),
             sim_dndx_subDLA=np.array(sim_block["subDLA"]["dndx_mean"]),
             sim_z_DLA=np.array(sim_block["DLA"]["z"]),
             sim_dndx_DLA=np.array(sim_block["DLA"]["dndx_mean"]),
             lit_z_LLS=np.array(lit_block["LLS"]["z"]),
             lit_v_LLS=np.array(lit_block["LLS"]["value"]),
             lit_e_LLS=np.array(lit_block["LLS"]["err"]),
             lit_z_subDLA=np.array(lit_block["subDLA"]["z"]),
             lit_v_subDLA=np.array(lit_block["subDLA"]["value"]),
             lit_e_subDLA=np.array(lit_block["subDLA"]["err"]),
             lit_z_DLA=np.array(lit_block["DLA"]["z"]),
             lit_v_DLA=np.array(lit_block["DLA"]["value"]),
             lit_e_DLA=np.array(lit_block["DLA"]["err"]),
             geometry=np.array("closure/cosmic-average survey=None: pivot prior "
                               "hcd_incidence_prior(w_c_z3, z=3, survey=None); z-slope "
                               "HCD_INCIDENCE_SLOPE; TruncatedNormal(low=0) LLS/subDLA, "
                               "DLA softplus(Normal(softplus^-1(mu),1.0)); "
                               "alpha_to_dndx_exact EXACT inverse of the w_c_corrected "
                               "forward incl. the 4-class renormalisation (2026-07-22; "
                               "pre-2026-07-22 layers used the APPROXIMATE alpha_to_dndx "
                               "apply_delta=True map, misdescribed here as 'exact inverse')"))

    # ---------------- money row 1 layers ------------------------------------------
    zg_r1 = np.asarray(ON["dndx_z"], float)
    Xb_r1 = np.asarray(ON["dndx_Xbar"], float)
    mu_camp = np.asarray(ON["alpha_hcd_mu"], float)
    sd_camp = np.asarray(ON["alpha_hcd_sigma"], float)

    def sampler_band(mu_v, sd_v, truncated):
        """Propagated band. truncated=False reproduces the June campaign sampler
        (sampler_numpyro: plain Normal LLS/subDLA); truncated=True is the deployed
        closure_legb geometry (TruncatedNormal low=0). DLA always softplus latent 1.0.

        Draws map through alpha_to_dndx_exact in mode="mask": plain-Normal campaign
        draws can be negative and z-scaled draws can leave the occupancy simplex at
        high z -- both legitimate under the pre-2026-07-22 unbounded prior geometry.
        Such draws are EXCLUDED from the percentiles (NaN) and COUNTED (returned +
        recorded in PROVENANCE), instead of the old silent saturation at
        27.631021/Xbar which biased the band edges."""
        rng = np.random.default_rng(ROW1_BAND_SEED)
        smp = rng.normal(mu_v, sd_v, size=(ROW1_BAND_N, 3))
        if truncated:  # resample the <0 mass (softplus DLA slot overwritten below)
            for jj in (0, 1):
                bad = smp[:, jj] < 0
                while bad.any():
                    smp[bad, jj] = rng.normal(mu_v[jj], sd_v[jj], size=int(bad.sum()))
                    bad = smp[:, jj] < 0
        rmu = float(_dla_raw_mu(mu_v[2]))
        smp[:, 2] = np.log1p(np.exp(np.minimum(rmu + rng.standard_normal(ROW1_BAND_N),
                                               30.0)))
        shape = ((1.0 + zg_r1)[:, None] / (1.0 + zp)) ** s_c[None, :]
        lo = np.empty((len(zg_r1), 3))
        hi = np.empty((len(zg_r1), 3))
        n_invalid = 0
        for jj, z in enumerate(zg_r1):
            dd, ok = alpha_to_dndx_exact(smp * shape[jj][None, :], float(Xb_r1[jj]),
                                         float(z), mode="mask")
            n_invalid += int(np.size(ok) - np.count_nonzero(ok))
            lo[jj], hi[jj] = np.nanpercentile(dd, ROW1_BAND_PCTS, axis=0)
        return lo, hi, n_invalid, int(len(zg_r1) * ROW1_BAND_N)

    r1_lo_camp, r1_hi_camp, r1_ninv_camp, r1_ntot = sampler_band(mu_camp, sd_camp,
                                                                 truncated=False)
    r1_lo_corr, r1_hi_corr, r1_ninv_corr, _ = sampler_band(mu, sd, truncated=True)
    print(f"[row1 band] out-of-domain draws excluded (not saturated): "
          f"campaign {r1_ninv_camp}/{r1_ntot}, corrected {r1_ninv_corr}/{r1_ntot}")

    np.savez(out / "money_row1_layers.npz",
             class_order=np.array(CLS), zfine=ROW1_ZFINE, zg=zg_r1, xbar=Xb_r1,
             law_A=np.array([HCD_LIT_DNDX_LAW[c][0] for c in CLS]),
             law_gamma=np.array([HCD_LIT_DNDX_LAW[c][1] for c in CLS]),
             dla_display_residual_frac=np.float64(0.10),
             prior68_campaign_lo=r1_lo_camp, prior68_campaign_hi=r1_hi_camp,
             prior68_campaign_alpha_mu=mu_camp, prior68_campaign_alpha_sigma=sd_camp,
             prior68_corrected_lo=r1_lo_corr, prior68_corrected_hi=r1_hi_corr,
             prior68_corrected_alpha_mu=mu, prior68_corrected_alpha_sigma=sd,
             dndx_truth=np.asarray(ON["dndx_truth"], float),
             lit_z_LLS=np.array(lit_block["LLS"]["z"]),
             lit_v_LLS=np.array(lit_block["LLS"]["value"]),
             lit_e_LLS=np.array(lit_block["LLS"]["err"]),
             lit_z_subDLA=np.array(lit_block["subDLA"]["z"]),
             lit_v_subDLA=np.array(lit_block["subDLA"]["value"]),
             lit_e_subDLA=np.array(lit_block["subDLA"]["err"]),
             lit_z_DLA=np.array(lit_block["DLA"]["z"]),
             lit_v_DLA=np.array(lit_block["DLA"]["value"]),
             lit_e_DLA=np.array(lit_block["DLA"]["err"]),
             band_variant_note=np.array(
                 "prior68_campaign_* = the June arms' pkl-recorded sampler priors, "
                 "plain-Normal geometry (consistent with the preserved row-2 contours, "
                 "which SAMPLED this prior); prior68_corrected_* = current deployed "
                 "closure survey=None prior (post-528ba89 center) through the "
                 "closure_legb TruncatedNormal/softplus geometry. Which to draw is a "
                 "paper/PI presentation decision; mixing corrected band with "
                 "old-campaign posteriors must be disclosed in the caption."))

    # ---------------- sec 3.2 prior-geometry table ---------------------------------
    forms_closure = {
        "LLS": "TruncatedNormal(mu, sigma, low=0) [closure_legb._hcd_sites]",
        "subDLA": "TruncatedNormal(mu, sigma, low=0) [closure_legb._hcd_sites]",
        "DLA": ("alpha_dla = softplus(Normal(softplus^-1(mu), 1.0)); SAMPLED latent "
                "width 1.0 (the nominal fractional 0.50 is NOT the sampled width)"),
    }
    geom = {}
    for skey in (None, "DESI", "eBOSS", "KS", "DESI+KS"):
        mu_s, sd_s = map(np.asarray,
                         hcd_incidence_prior(jnp.asarray(w_c_z3), z=zp, survey=skey))
        row = {}
        if skey is not None:
            c_lls = float(hcd_lls_realfit_alpha_center(
                Xbar_z3, z=zp, boost=HCD_LLS_SURVEY_BOOST[skey]))
            w_lls = float(HCD_LLS_SURVEY_FRAC_SIGMA[skey]) * c_lls
            row["LLS"] = {"center": c_lls, "width": w_lls,
                          "frac_sigma": float(HCD_LLS_SURVEY_FRAC_SIGMA[skey]),
                          "boost": float(HCD_LLS_SURVEY_BOOST[skey]),
                          "sampling_form": forms_closure["LLS"]}
        else:
            row["LLS"] = {"center": float(mu_s[0]), "width": float(sd_s[0]),
                          "frac_sigma": float(HCD_PRIOR_FRAC_SIGMA[0]), "boost": 1.0,
                          "sampling_form": forms_closure["LLS"]}
        row["subDLA"] = {"center": float(mu_s[1]), "width": float(sd_s[1]),
                         "frac_sigma": float(HCD_PRIOR_FRAC_SIGMA[1]),
                         "sampling_form": forms_closure["subDLA"]}
        row["DLA"] = {"center": float(mu_s[2]),
                      "nominal_frac_sigma_UNUSED": float(HCD_PRIOR_FRAC_SIGMA[2]),
                      "sampled_latent_width": 1.0,
                      "effective_sd_over_mean_approx": 1.2,
                      "residual_frac": float(HCD_DLA_RESIDUAL_FRAC),
                      "sampling_form": forms_closure["DLA"]}
        geom["closure_SBC" if skey is None else skey] = row
    geom["_zslope"] = {
        "closure_path_centers": [float(x) for x in HCD_INCIDENCE_SLOPE],
        "survey_path_centers": [float(HCD_LLS_REALFIT_ZSLOPE),
                                float(HCD_INCIDENCE_SLOPE[1]),
                                float(HCD_INCIDENCE_SLOPE[2])],
        "prior_sigma": [float(x) for x in ZSLOPE_PRIOR_SIGMA],
        "form": "alpha_c(z) = alpha_c(3) * ((1+z)/4)**s_c, s_c ~ Normal(center, sigma)",
    }
    geom["_notes"] = {
        "DESI+KS": "legacy shared-alpha joint key; retained for payload stability, "
                   "superseded for joint fits by the per-leg alpha design",
        "hedge2x_lls_frac_sigma": {k: float(v) for k, v
                                   in HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X.items()},
        "june_campaign_sampler": "the desi_hcd_prior_arms (money row 2) used "
                                 "sampler_numpyro: PLAIN Normal LLS/subDLA (no "
                                 "truncation), DLA softplus latent 1.0, closure "
                                 "geometry with the pkl-recorded (mu, sigma)",
    }

    # ---------------- kernel record -------------------------------------------------
    kernel_block = {
        "kernel_chosen": j["kernel_chosen"],
        "alpha_lls_z3_per_kernel": j["alpha_lls_z3"],
        "adopted_band": j["adopted_band"],
        "adopted_width": j["adopted_width"],
        "kernel_table": j["kernel_table"],
        "bracket": j["bracket"],
        "pi_adoption": j.get("pi_adoption"),
        "derivation_json": "hcd_analysis/emulator/hcd_lit_dndx_corrected.json",
        "naming_note": ("in alpha_lls_z3_per_kernel, 'deployed'=0.19393 is the "
                        "PRE-SWAP historical center (RETIRED with the tau_LL>=2 "
                        "object); 'adopted' (K1a, 0.17219, pinned-Xbar tier-b "
                        "evaluation) is the CURRENT deployed center. The live-cache "
                        "evaluation of the same center is "
                        "prior_geometry_table['DESI'].LLS.center = 0.172112; the "
                        "~4e-5 difference is the pinned-vs-live Xbar(z=3) only."),
    }

    export = {
        "schema": "dndx_corrected_export_v1",
        "purpose": "HCD-emulator paper dN/dX re-pin (PI directive 2026-07-20 option A); "
                   "artifact-only sourcing channel",
        "estimand_id": dict(ESTIMAND_ID),
        "estimand_label": dict(ESTIMAND_LABEL),
        "retired": {"lls_display": "the cumulative tau_LL>=2 object is RETIRED and must "
                                   "not be used or named (guard: "
                                   "lit_dndx.assert_display_estimand / __getattr__ "
                                   "tombstones)"},
        "literature_points": lit_block,
        "laws": laws_block,
        "sim_incidence": sim_block,
        "prior_geometry_table": geom,
        "kernel": kernel_block,
        "hcd_prior_signature": hcd_prior_signature(),
        "figure_layers": {"f3": "f3_layers.npz", "money_row1": "money_row1_layers.npz"},
    }
    (out / "dndx_corrected_export.json").write_text(json.dumps(export, indent=1,
                                                               sort_keys=True))

    # ---------------- provenance sidecar --------------------------------------------
    import jax as _jax
    inputs = {}
    for p in (deriv_json, ON_HI_PKL,
              ROOT / "hcd_analysis/emulator/lit_dndx.py",
              ROOT / "hcd_analysis/emulator/inference.py",
              ROOT / "hcd_analysis/emulator/dndx_wc.py"):
        inputs[str(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    inputs[str(cache_path)] = {
        "sha256": sha256(cache_path), "bytes": cache_path.stat().st_size,
        "internal_build_metadata": {"cache_version": "3.3",
                                    "git_sha": "964e935adad9e520f56d12d6ad0043cd5f19864c",
                                    "created_utc": "2026-06-02T16:56:48Z",
                                    "note": "h5 root attrs; readable cheaply without "
                                            "hashing the file"}}
    outputs = {}
    for name in ("dndx_corrected_export.json", "f3_layers.npz", "money_row1_layers.npz"):
        p = out / name
        outputs[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    prov = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain",
                          "hcd_analysis/", "scripts/export_dndx_paper_layer.py")),
        "run_command": RUN_CMD,
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "jax": _jax.__version__, "hostname": platform.node(),
            "conda_env": "emu-jax", "platform": platform.platform(),
        },
        "inputs_sha256": inputs,
        "outputs_sha256": outputs,
        "inverse_map": ("alpha_to_dndx_exact (EXACT inverse of the w_c_corrected forward "
                        "incl. the 4-class renormalisation; fail-loud raise mode for prior "
                        "centres/edges, mask mode for sampled bands; readout defect B, "
                        "2026-07-22). Pre-2026-07-22 runs of this exporter -- including the "
                        "frozen dndx_repin_2026-07-20 artifact of record -- used the "
                        "APPROXIMATE alpha_to_dndx (renorm ignored, silent saturation at "
                        "27.631021/Xbar) and differ at the renorm level."),
        "row1_band_invalid_draws": {
            "campaign": [r1_ninv_camp, r1_ntot], "corrected": [r1_ninv_corr, r1_ntot],
            "policy": ("out-of-domain draws (negative alpha or sum(alpha)>=1 after z-scaling; "
                       "legitimate under the pre-2026-07-22 unbounded prior geometry) are "
                       "EXCLUDED from the band percentiles and counted here, instead of the "
                       "old silent saturation at 27.631021/Xbar")},
        "blind_status": "BLIND-SAFE: no real-data cosmology values (literature, prior "
                        "constants, and closure-mock truth quantities only)",
        "chain_of_record_status": "ARTIFACT OF RECORD for the paper dN/dX re-pin "
                                  "(PI directive 2026-07-20). Not an MCMC chain. "
                                  "Immutable: regenerate to a NEW dated directory, "
                                  "never edit in place.",
        "hcd_prior_signature": hcd_prior_signature(),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, sort_keys=True))

    print(f"[export] wrote {out}")
    for name, meta in outputs.items():
        print(f"  {name}  sha256={meta['sha256'][:16]}...  {meta['bytes']} B")
    print(f"  commit={prov['commit'][:12]} dirty={prov['dirty']} "
          f"prior_sig={prov['hcd_prior_signature'][:12]}...")


if __name__ == "__main__":
    main()
