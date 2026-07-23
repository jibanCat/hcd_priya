"""Corrected literature dN/dX law re-derivation — the FULL decision table + JSON emitter
(spec hcd_priya_notes/docs/superpowers/2026-07-18-corrected-law-spec.md, steps 6-8).

Prints: the 0.88-vs-0.96 adjudication (both prior recipes under the pinned definition +
attribution), every kernel candidate's intermediate b(z), both estimators per count class,
the corrected laws per kernel with uncertainties, alpha_LLS(z=3) per kernel through the
DEPLOYED path (tiered fallback, stated in the table), the LIT_OVER_SIM refit, both width
variants, bracket endpoints, sensitivity arms, and telescoping pulls.

Writes hcd_analysis/emulator/hcd_lit_dndx_corrected.json. PI ADOPTION (2026-07-18,
decision bundle resolved): kernel_chosen = "K1a" (the pinned per-point cache-kernel recipe).
PI reasoning (verbatim): "the simulations evolve the Lya forest, LLSs, subDLAs, and DLAs
jointly under the same structure formation, so the relative decomposition across absorber
classes is physically self-consistent; the observational decomposition may still contain
unresolved Eddington-bias and classification issues and is not trusted at the 30-50% level
over a physically consistent simulation; the absolute incidence stays anchored to
observations." (PI, 2026-07-18)
DEPLOYED LLS LAW (PI decision 1c): the K1a-corrected points refit with gamma CONSTRAINED
to the deployed z-slope 2.127 (amplitude-only GLS, same covariance treatment); the
free-gamma K1a fit (~2.137) is recorded as the consistency evidence justifying the
constraint. The full decision table (all kernels) stays in the JSON for the record.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/derive_hcd_dndx_corrected.py
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys

import numpy as np

REPO = "/home/mfho/hcd_priya"
sys.path.insert(0, REPO)

from hcd_analysis.emulator import lit_dndx as LD            # noqa: E402
from hcd_analysis.emulator import lit_dndx_kernel as LK     # noqa: E402

OUT_JSON = f"{REPO}/hcd_analysis/emulator/hcd_lit_dndx_corrected.json"
# the PRE-2026-07-18 laws (validation baseline / wrong-object references ONLY; the deployed
# laws now live in inference.HCD_LIT_DNDX_LAW — closing-panel fix 4)
PRE_SWAP_LAW = {"LLS": (0.0201, 2.127), "subDLA": (0.0211, 0.937), "DLA": (0.0076, 1.592)}
# PI adoption (2026-07-18): K1a kernel; deployed LLS = constrained-slope refit at 2.127.
ADOPTED_KERNEL = "K1a"
ADOPTED_GAMMA_LLS = 2.127          # keep the deployed z-slope (PI decision 1c/2)
BAND_REL = (0.83, 1.21)            # today's relative band margins (PI decision 5/7)
PRE_SWAP_ALPHA_Z3 = 0.19393       # the PRE-2026-07-18 deployed center (validation baseline ONLY;
                                  # the deployed center is now 0.1722 — closing-panel fix 4)
XBAR_Z3_PINNED = 0.632            # tests pin hcd_pivot_wc_and_xbar Xbar(z=3) ~ 0.632 +/- 0.03
Z_EVAL = (2.4, 3.0, 3.6, 4.2)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not callable(v)}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def law_at(law, z):
    return law["A"] * (1.0 + np.asarray(z, float)) ** law["gamma"]


def law_tuple(law):
    return (float(law["A"]), float(law["gamma"]))


# --------------------------------------------------------------------------- #
#  alpha_LLS(z=3) through the DEPLOYED path (tiered; the tier is REPORTED)     #
# --------------------------------------------------------------------------- #
def deployed_alpha_fn():
    """Tier (a): full deployed path (load_cache -> hcd_pivot_wc_and_xbar Xbar). Tier (b):
    deployed hcd_lls_realfit_alpha_center with the pinned Xbar=0.632. Tier (c): pure
    law-ratio x 0.19393 fallback (no jax). Returns (fn(laws_dict)->alpha, tier_note)."""
    if os.environ.get("DERIVE_ALPHA_FALLBACK", "0") == "1":
        return None, None, "tier-c FORCED by env"
    try:
        from hcd_analysis.emulator import inference as INF
    except Exception as e:                                   # pragma: no cover
        return None, None, f"tier-c (inference import failed: {e})"
    xbar, tier = None, None
    if os.environ.get("DERIVE_ALPHA_FULL", "0") == "1":
        # tier (a) is opt-in: load_cache materializes the ~1 GB cache + P1D collapses,
        # heavy for the 8-CPU/32-GB login node (repo memory: submit heavy work via SLURM)
        try:
            from hcd_analysis.emulator import closure_legb as CL
            from hcd_analysis.emulator.data import load_cache
            d = load_cache(LK.CACHE_PATH)
            _, xbar = CL.hcd_pivot_wc_and_xbar(d)
            tier = f"tier-a (full deployed path, Xbar(z=3)={xbar:.4f} via load_cache)"
            del d
        except Exception as e:
            xbar = None
            print(f"[alpha] tier-a failed ({e}); falling back to tier-b")
    if xbar is None:
        xbar = XBAR_Z3_PINNED
        tier = (f"tier-b (deployed inference.hcd_lls_realfit_alpha_center with the PINNED "
                f"Xbar(z=3)={xbar} — the test-pinned cache value; load_cache skipped on "
                f"the login node; set DERIVE_ALPHA_FULL=1 for tier-a)")

    def alpha_for(laws):
        saved = INF.HCD_LIT_DNDX_LAW
        try:
            INF.HCD_LIT_DNDX_LAW = laws                     # runtime-only override
            return float(INF.hcd_lls_realfit_alpha_center(xbar))
        finally:
            INF.HCD_LIT_DNDX_LAW = saved
    return alpha_for, xbar, tier


def _deployed_laws():
    """The DEPLOYED law tuples for the consistency-vs-refit gate: from inference.py when
    importable, else from the committed derivation JSON's laws_deployed block."""
    try:
        from hcd_analysis.emulator import inference as INF
        return {c: tuple(v) for c, v in INF.HCD_LIT_DNDX_LAW.items()}
    except Exception:                                        # pragma: no cover
        import json as _json
        with open(LD.__file__.replace("lit_dndx.py", "hcd_lit_dndx_corrected.json")) as fh:
            j = _json.load(fh)
        # closing-panel fix 3: laws_deployed lives under adopted_law, not pi_adoption
        return {c: tuple(v) for c, v in j["adopted_law"]["laws_deployed"].items()}


def main():
    print("=" * 100)
    print("CORRECTED LITERATURE dN/dX LAW RE-DERIVATION — decision table "
          "(kernel_chosen = None; PI decision 1 pending)")
    print("=" * 100)

    inputs = LK.load_kernel_inputs(LK.CACHE_PATH)
    k1_tab = LK.k1_table(inputs)
    F_tab = LK.floor_factor_table(inputs)
    sfc = LK.same_file_consistency(inputs)

    def r_k1(z):
        return LK.interp_ln1pz(k1_tab["z_vals"], k1_tab["r"], z)

    k1b = LK.k1_smooth_fit(inputs, _tab=k1_tab)

    def floor_factor(z):
        return LK.interp_ln1pz(F_tab["z_vals"], F_tab["F"], z)

    # ---------------- step 6: the 0.88-vs-0.96 adjudication ----------------- #
    adj = LK.adjudication(inputs)
    print("\n[ADJUDICATION] 0.88-vs-0.96 same-cache kernel ambiguity "
          "(both prior recipes under the pinned per-z aggregation):")
    print(f"  {'recipe':28s} " + "  ".join(f"r({z:.1f})" for z in adj["z_eval"]))
    for name, rec in adj["recipes"].items():
        rr = "  ".join(f"{v:.4f}" for v in rec["r"])
        print(f"  {name:28s} {rr}   | {rec['note']}")
    print("  attribution:")
    for k, v in adj["attribution"].items():
        nums = {kk: vv for kk, vv in v.items() if kk != "note"}
        print(f"    {k}: {nums}  — {v['note']}")

    # ---------------- first pass: laws without the kernel budget ------------ #
    pre = LD.run_fit_ordering(floor_factor, include_sensitivity=False)
    sub_law, dla_law, cum_law = pre["subDLA"], pre["DLA"], pre["cum_uncorrected"]

    def r_k2(z):
        cap = LD.cap_fraction_from_laws(z, None, dla_law, cum_law, slls_own_words=True)
        return cap * np.asarray(floor_factor(z), float)

    budget = LK.kernel_budget(inputs, r_k2, _tab=k1_tab)
    print("\n[KERNEL BUDGET] s_r3 = quadrature(suite 16-84 half-spread, half K1-K2 spread):")
    print(f"  suite_half={budget['suite_half']:.4f}  model_half={budget['model_half']:.4f}"
          f"  -> s_r3={budget['s_r3']:.4f} ;  eta_K1={budget['eta_k1']:+.4f} "
          f"eta_K2={budget['eta_k2']:+.4f} -> sigma_eta={budget['sigma_eta']:.4f}")
    kb = dict(s_r3=budget["s_r3"], sigma_eta=budget["sigma_eta"])

    # ---------------- final ordering with kernel covariance ----------------- #
    # reference_laws = the DEPLOYED module constants, so consistency_vs_refit is the real
    # deployed-vs-refit gate (review meta finding 6: never self-vs-self)
    _dep = _deployed_laws()
    res = LD.run_fit_ordering(floor_factor, r_k1=r_k1, r_k1_smooth=k1b["r_at"],
                              kernel_budget=kb,
                              reference_laws={"subDLA": _dep["subDLA"], "DLA": _dep["DLA"]})

    # ---------------- count-class estimators (both, spec 4a) ---------------- #
    print("\n[COUNT CLASSES] Poisson GLM (ADOPTED) vs log-WLS, same points:")
    s = res["subDLA"]
    print(f"  subDLA GLM : A={s['A']:.5f} gamma={s['gamma']:+.4f} "
          f"(A_p={s['A_pivot']:.4f}+-{s['sigma_lnAp']*s['A_pivot']:.4f}, "
          f"sigma_gamma={s['sigma_gamma']:.3f})  deviance/dof={s['deviance_dof']:.3f}")
    print(f"       vs WLS: delta_A_frac={s['glm_vs_wls']['delta_A_frac']:+.4f} "
          f"delta_gamma={s['glm_vs_wls']['delta_gamma']:+.4f}  "
          f"(sibling transplant (0.0050,2.426) sits within "
          f"{100*abs(LD.law_consistency_vs_refit(0.0050,2.426,s,LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3['z_bar'])['max_frac_dev']):.1f}% of the GLM)")
    d = res["DLA"]
    print(f"  DLA GLM    : A={d['A']:.5f} gamma={d['gamma']:+.4f}  "
          f"deviance/dof={d['deviance_dof']:.3f}")
    print(f"     WLS anchor: A={d['wls_anchor']['A']:.5f} gamma={d['wls_anchor']['gamma']:+.4f}"
          f"  (deployed (0.0076, 1.592) — regression anchor)  GLM-vs-WLS: "
          f"dA={d['glm_vs_wls']['delta_A_frac']:+.4f} dg={d['glm_vs_wls']['delta_gamma']:+.4f}")
    c = res["cum_uncorrected"]
    print(f"  cum tau>=2 : A={c['A']:.5f} gamma={c['gamma']:+.4f} chi2/dof={c['chi2_red']:.3f}"
          f"  (uncorrected compilation GLS; deployed-wrong-object analog (0.0201, 2.127))")

    # ---------------- kernel candidates: b(z) + corrected laws -------------- #
    alpha_for, xbar_used, tier = deployed_alpha_fn()
    if alpha_for is None:
        tier = tier or "tier-c"

        def alpha_for(laws):
            ratio = (laws["LLS"][0] * 4.0 ** laws["LLS"][1]) / \
                    (PRE_SWAP_LAW["LLS"][0] * 4.0 ** PRE_SWAP_LAW["LLS"][1])
            return ratio * PRE_SWAP_ALPHA_Z3
        xbar_used = None
    print(f"\n[ALPHA PATH] {tier}")
    alpha_deployed_check = alpha_for({k: v for k, v in PRE_SWAP_LAW.items()})
    print(f"  validation: alpha at the DEPLOYED laws = {alpha_deployed_check:.5f} "
          f"(expected ~{PRE_SWAP_ALPHA_Z3})")

    # the ACTUAL kernel callables (b(z) columns computed from the definitions, not
    # re-interpolated through the 8 lit z_bars)
    def r_k3(z):
        cap = LD.cap_fraction_from_laws(z, s, d, c, slls_own_words=False)
        return cap * np.asarray(floor_factor(z), float)

    r_fns = {"K1a": r_k1, "K1b": k1b["r_at"], "K2": r_k2, "K3": r_k3}

    kernels = {}
    for name in ("K1a", "K1b", "K2", "K3"):
        if name not in res:
            continue
        k = res[name]
        law = k["law"]["fit"]
        laws_triple = {"LLS": law_tuple(law), "subDLA": law_tuple(s), "DLA": law_tuple(d)}
        alpha = alpha_for(laws_triple)
        kernels[name] = dict(
            law=dict(A=law["A"], gamma=law["gamma"], A_pivot=law["A_pivot"],
                     sigma_lnAp=law["sigma_lnAp"], sigma_gamma=law["sigma_gamma"],
                     chi2_red=law["chi2_red"], cov=law["cov"]),
            r_z_eval={f"{z:.1f}": float(np.asarray(r_fns[name](np.array([z])))[0])
                      for z in Z_EVAL},
            r3=k["r3"], alpha_lls_z3=alpha,
            telescoping=dict(z=k["telescoping"]["z"], pull=k["telescoping"]["pull"],
                             ratio=k["telescoping"]["ratio"]),
            stats=k["law"]["stats"], one_x=one_x_widths(k["law"], kb),
        )

    print("\n[DECISION TABLE] kernel candidates (b(z) intermediates, corrected LLS law, "
          "alpha_LLS(z3), telescoping pulls):")
    hdr = (f"  {'K':4s} {'b(2.4)':>7s} {'b(3.0)':>7s} {'b(3.6)':>7s} {'b(4.2)':>7s} "
           f"{'A':>8s} {'gamma':>7s} {'A_p(z3)':>8s} {'s_lnAp':>7s} {'s_gam':>6s} "
           f"{'chi2/dof':>8s} {'alpha(z3)':>9s} {'d_alpha':>8s} {'tele-pull(2.5..4)':>20s}")
    print(hdr)
    for name, k in kernels.items():
        r = k["r_z_eval"]
        tp = " ".join(f"{p:+.2f}" for p in k["telescoping"]["pull"])
        print(f"  {name:4s} {r['2.4']:7.4f} {r['3.0']:7.4f} {r['3.6']:7.4f} {r['4.2']:7.4f} "
              f"{k['law']['A']:8.5f} {k['law']['gamma']:7.4f} {k['law']['A_pivot']:8.4f} "
              f"{k['law']['sigma_lnAp']:7.4f} {k['law']['sigma_gamma']:6.3f} "
              f"{k['law']['chi2_red']:8.3f} {k['alpha_lls_z3']:9.5f} "
              f"{k['alpha_lls_z3']/PRE_SWAP_ALPHA_Z3-1:+8.1%} {tp:>20s}")
    print(f"  (PRE-SWAP baseline alpha = {PRE_SWAP_ALPHA_Z3}, pre-swap LLS law (0.0201, 2.127); "
          f"K3b per-point diagnostic: {res['K3b_diagnostic']['n_nonpositive']} non-positive "
          f"corrected points of 8)")
    k3b = res["K3b_diagnostic"]
    print(f"  K3b corrected points: " +
          " ".join(f"{z:.2f}:{v:+.3f}" for z, v in zip(k3b["z_bar"], k3b["corrected_points"])))

    # ---------------- HARD GATES (spec sec 4/6: assert in script AND test) --- #
    # weighted-mean bias: every fitted class vs its own points
    for cls in ("subDLA", "DLA", "cum_uncorrected"):
        blk = res[cls]
        assert abs(blk["stats"]["wmean_frac_resid"]) < 0.05, (cls, blk["stats"])
    # DEPLOYED-vs-refit (the transplant-killer; non-vacuous by construction — reference_laws
    # threaded above; kernel-candidate laws have no deployed referent and carry None):
    for cls in ("subDLA", "DLA"):
        assert res[cls]["consistency_vs_refit"]["max_frac_dev"] < 0.05, (
            cls, res[cls]["consistency_vs_refit"])
    assert np.all(np.abs(np.asarray(kernels["K3"]["telescoping"]["pull"])) < 1.0), (
        "K3 must be telescoping-consistent by construction")
    print("\n[GATES] internal-consistency (<5% weighted-mean bias) + DEPLOYED-vs-refit "
          "(<5%, subDLA/DLA; adopted LLS gated in the adoption block) and "
          "K3 telescoping: ALL PASS (weighted-rms values are REPORTED per class in stats; "
          "see the implementation report for why raw wrms cannot be a 5% gate on "
          "Poisson-scale points)")

    # ---------------- widths (PI decision 2) -------------------------------- #
    print("\n[WIDTH VARIANTS] sigma_LLS on the corrected points (per kernel):")
    for name, k in kernels.items():
        w = k["one_x"]
        print(f"  {name:4s} measurement-only one_x={w['meas_only']:.3f} "
              f"(norm_infl={w['norm_infl']:.3f}, scatter={w['scatter']:.3f}) ; "
              f"meas+kernel-common-mode={w['with_kernel']:.3f} (s_r3={kb['s_r3']:.3f})")

    # ---------------- LIT_OVER_SIM refit (spec 5.5) ------------------------- #
    lit_over_sim = lit_over_sim_refit(inputs, kernels, s, d, cum_law)
    print("\n[LIT_OVER_SIM REFIT] (plot_dndx_vs_literature.py:95-108 construction, "
          "corrected laws; PRE-SWAP (1.06, 1.00, 1.34) / slopes (0.95, 0.15, 0.40); deployed now (0.995, 1.00, 1.34)/(0.764, 0.15, 0.40)):")
    for cls, v in lit_over_sim.items():
        if cls == "note":
            continue
        print(f"  {cls}: {v}")
    print(f"  note: {lit_over_sim['note']}")

    # ---------------- sensitivity arms (spec step 7) ------------------------ #
    sens = res["sensitivity"]
    print("\n[SENSITIVITY ARMS]")
    sd = sens["drop_z180"]["subDLA"]
    print(f"  drop z=1.80 (subDLA): A={sd['A']:.5f} gamma={sd['gamma']:+.4f} "
          f"deviance/dof={sd['deviance_dof']:.3f} "
          f"(vs {s['A']:.5f}/{s['gamma']:+.4f})")
    for arm in ("larger_side", "journal_fix"):
        ac = sens[arm]["cum_uncorrected"]
        a3 = sens[arm]["K3"]["law"]["fit"]
        print(f"  {arm:12s}: cum A={ac['A']:.5f} gamma={ac['gamma']:+.4f} | "
              f"K3 law A={a3['A']:.5f} gamma={a3['gamma']:+.4f}")

    # ---------------- bracket + band proposals ------------------------------ #
    bracket = dict(center_kernel=f"{ADOPTED_KERNEL} (PI ADOPTED 2026-07-18; K3/K2 = the "
                                 f"one-sided minus arms of the honest bracket)",
                   minus_arm="K2", mid_arm="K3", plus_arm="K1a",
                   alpha=dict((n, kernels[n]["alpha_lls_z3"]) for n in kernels))
    print("\n[BRACKET] center = PI-ADOPTED K1a; K3/K2 = the minus (mid/far) bracket arms:")
    for n, a in bracket["alpha"].items():
        band = (round(a * 0.83, 2), round(a * 1.21, 2))
        print(f"  {n:4s} alpha_LLS(z3)={a:.5f}  default band proposal (x0.83,1.21) -> {band}"
              f"  {'TRIPS the deployed (0.16,0.24) floor' if a < 0.16 else 'inside the deployed band'}")

    # ---------------- PI ADOPTION (2026-07-18): K1a + constrained slope ------ #
    # Deployed LLS law = K1a-corrected points refit with gamma CONSTRAINED to the deployed
    # z-slope 2.127 (amplitude-only GLS under the SAME covariance: diag + s_r3 common-mode
    # + eta tilt). The free-gamma K1a fit is the consistency evidence for the constraint.
    k1a_pts = res["K1a"]["law"]["corrected_points"]
    free_fit = res["K1a"]["law"]["fit"]
    con_fit = LD.gls_powerlaw_log(k1a_pts["z_bar"], k1a_pts["lx"], k1a_pts["sig_log"],
                                  s_common=kb["s_r3"], sigma_eta=kb["sigma_eta"],
                                  gamma_fixed=ADOPTED_GAMMA_LLS)
    assert abs(free_fit["gamma"] - ADOPTED_GAMMA_LLS) < 0.5 * free_fit["sigma_gamma"], (
        "constraint unjustified: free K1a gamma inconsistent with the deployed 2.127")
    # adopted-LLS deployed-vs-refit gate (completes the non-vacuous gate set)
    _dep_lls = _deployed_laws().get("LLS")
    if _dep_lls is not None:
        _dev = LD.law_consistency_vs_refit(_dep_lls[0], _dep_lls[1], con_fit,
                                           k1a_pts["z_bar"])
        assert _dev["max_frac_dev"] < 0.05, ("deployed LLS law vs constrained refit", _dev)
    laws_adopted = {"LLS": (con_fit["A"], con_fit["gamma"]),
                    "subDLA": law_tuple(s), "DLA": (0.0076, 1.592)}
    alpha_adopted = alpha_for(laws_adopted)
    band_adopted = (round(alpha_adopted * BAND_REL[0], 2),
                    round(alpha_adopted * BAND_REL[1], 2))
    # band requirements (spec 5.4 / PI decision 5): contains the adopted center; contains
    # the sim z=3 w_c center 0.2004 (closure-guard reuse); still EXCLUDES 0.2909.
    assert band_adopted[0] <= alpha_adopted <= band_adopted[1], band_adopted
    assert band_adopted[0] <= 0.2004 <= band_adopted[1], band_adopted
    assert not (band_adopted[0] <= 0.2909 <= band_adopted[1]), band_adopted
    con_tele = LD._telescoping_check(con_fit, floor_factor, s, d, c)
    con_stats = LD.law_vs_points_stats(con_fit["A"], con_fit["gamma"], k1a_pts["z_bar"],
                                       k1a_pts["lx"], k1a_pts["sig_log"] * k1a_pts["lx"])
    # adopted width (PI decision 4): the measurement+kernel-common-mode variant for K1a;
    # deployed knob = that value rounded to 3 decimals.
    w_k1a = kernels["K1a"]["one_x"]
    width_knob = round(w_k1a["with_kernel"], 3)
    # adopted LIT_OVER_SIM (PI decision 6): LLS slots from the CONSTRAINED (deployed) law
    # against the same sim construction; subDLA slot STAYS 1.00/0.15 (PI decision 9, the
    # corrected lit/sim recorded as evidence); DLA slots unchanged (re-verified above).
    g_sim_lls = lit_over_sim["LLS"]["gamma_sim"]
    sim_zp_lls = lit_over_sim["LLS"]["sim_zp"]
    lit_zp_con = con_fit["A"] * 4.0 ** con_fit["gamma"]
    los_adopted = dict(ratio_zp=float(lit_zp_con / sim_zp_lls),
                       ratio_slope=float(con_fit["gamma"] - g_sim_lls),
                       deploy_ratio_zp=round(float(lit_zp_con / sim_zp_lls), 3),
                       deploy_ratio_slope=round(float(con_fit["gamma"] - g_sim_lls), 3))
    assert los_adopted["ratio_slope"] < 1.5, "LLS ratio_slope trips the 1.5 guard floor"
    print(f"\n[ADOPTED — PI 2026-07-18] kernel={ADOPTED_KERNEL}, deployed LLS law = "
          f"constrained refit at gamma={ADOPTED_GAMMA_LLS}:")
    print(f"  LLS law     : A={con_fit['A']:.10g} gamma={con_fit['gamma']} "
          f"(A_p={con_fit['A_pivot']:.6f}+-{con_fit['sigma_lnAp']*con_fit['A_pivot']:.4f}) "
          f"chi2/dof={con_fit['chi2_red']:.3f}")
    print(f"  free-gamma evidence: A={free_fit['A']:.10g} gamma={free_fit['gamma']:.4f} "
          f"+- {free_fit['sigma_gamma']:.3f} (consistent with 2.127 at "
          f"{abs(free_fit['gamma']-ADOPTED_GAMMA_LLS)/free_fit['sigma_gamma']:.2f} sigma)")
    print(f"  subDLA law  : A={s['A']:.10g} gamma={s['gamma']:.10g} (Poisson GLM)")
    print(f"  DLA law     : (0.0076, 1.592) KEPT (WLS anchor {d['wls_anchor']['A']:.5f}/"
          f"{d['wls_anchor']['gamma']:.4f} equals it at rounding)")
    print(f"  alpha_LLS(z3) adopted = {alpha_adopted:.6f}  band = {band_adopted} "
          f"(x{BAND_REL[0]}, x{BAND_REL[1]}; contains 0.2004, excludes 0.2909)")
    print(f"  width       : meas-only={w_k1a['meas_only']:.4f}, with kernel common-mode="
          f"{w_k1a['with_kernel']:.4f} -> deployed knob {width_knob} (PI decision 4)")
    print(f"  LIT_OVER_SIM LLS: ratio_zp={los_adopted['ratio_zp']:.4f} -> deploy "
          f"{los_adopted['deploy_ratio_zp']}; ratio_slope={los_adopted['ratio_slope']:.4f}"
          f" -> deploy {los_adopted['deploy_ratio_slope']} (< 1.5 guard floor)")
    print(f"  telescoping pulls (constrained law, z 2.5..4.0): "
          + " ".join(f"{p:+.2f}" for p in con_tele["pull"])
          + "  [EXPECTED +: PRIYA-shape vs lit-class-decomposition tension, PI-accepted]")

    # ---------------- JSON ------------------------------------------------- #
    git = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    payload = dict(
        schema_version="1.1",
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=git,
        kernel_chosen=ADOPTED_KERNEL,           # PI decision 1 (2026-07-18): K1a
        # the estimand each deployed law is a law OF (mirrored by inference.HCD_LIT_DNDX_ESTIMAND;
        # test-enforced dict equality module-vs-JSON)
        hcd_lit_dndx_estimand=dict(LLS="binned_17.2_19.0", subDLA="binned_19.0_20.3",
                                   DLA="binned_ge20.3"),
        pi_adoption=dict(
            date="2026-07-18",
            kernel="K1a",
            reasoning=("the simulations evolve the Lya forest, LLSs, subDLAs, and DLAs "
                       "jointly under the same structure formation, so the relative "
                       "decomposition across absorber classes is physically "
                       "self-consistent; the observational decomposition may still "
                       "contain unresolved Eddington-bias and classification issues and "
                       "is not trusted at the 30-50% level over a physically consistent "
                       "simulation; the absolute incidence stays anchored to "
                       "observations. (PI, 2026-07-18)"),
            zslope=("KEEP the deployed 2.127: the deployed LLS law is the K1a-corrected "
                    "points refit with gamma CONSTRAINED to 2.127 (amplitude-only GLS, "
                    "same covariance); the free-gamma fit is the recorded consistency "
                    "evidence (PI decision 1c)."),
            telescoping_note=("K1 telescoping pulls +1.7..+2.4 sigma at z>=3 are EXPECTED "
                              "by design (PRIYA-CDDF class shares vs the literature class "
                              "decomposition tension); K3 closes by construction; "
                              "PI-accepted."),
        ),
        adopted_law=dict(
            kernel=ADOPTED_KERNEL,
            A=con_fit["A"], gamma=con_fit["gamma"], A_pivot=con_fit["A_pivot"],
            sigma_lnAp=con_fit["sigma_lnAp"], chi2_red=con_fit["chi2_red"],
            estimator=con_fit["estimator"],
            corrected_points=dict(z_bar=k1a_pts["z_bar"], lx=k1a_pts["lx"],
                                  sig_log=k1a_pts["sig_log"], r=k1a_pts["r"]),
            budget=dict(s_r3=kb["s_r3"], sigma_eta=kb["sigma_eta"]),
            free_fit_evidence=dict(A=free_fit["A"], gamma=free_fit["gamma"],
                                   sigma_gamma=free_fit["sigma_gamma"],
                                   A_pivot=free_fit["A_pivot"],
                                   chi2_red=free_fit["chi2_red"]),
            telescoping=dict(z=con_tele["z"], pull=con_tele["pull"],
                             ratio=con_tele["ratio"]),
            stats=con_stats,
            laws_deployed=dict(LLS=list(laws_adopted["LLS"]),
                               subDLA=list(laws_adopted["subDLA"]),
                               DLA=list(laws_adopted["DLA"])),
        ),
        adopted_band=dict(band=list(band_adopted), rel_margins=list(BAND_REL),
                          contains_center=True, contains_sim_wc_02004=True,
                          excludes_allz_02909=True),
        adopted_width=dict(meas_only=w_k1a["meas_only"],
                           with_kernel=w_k1a["with_kernel"],
                           deployed_knob=width_knob,
                           rule=("PI decision 4 (2026-07-18): sigma_LLS adopts the "
                                 "measurement+kernel-common-mode K1a value, rounded to 3 "
                                 "decimals; 2x hedge = 2*knob; KS stays 0.40 broad")),
        adopted_lit_over_sim=dict(
            LLS=los_adopted,
            subDLA=dict(deploy_ratio_zp=1.00, deploy_ratio_slope=0.15,
                        evidence_ratio_zp=lit_over_sim["subDLA"]["ratio_zp"],
                        evidence_ratio_slope=lit_over_sim["subDLA"]["ratio_slope"],
                        note="STAYS sim-anchored (PI decision 9); corrected lit/sim "
                             "recorded as evidence, inside the 0.40 width"),
            DLA=dict(deploy_ratio_zp=1.34, deploy_ratio_slope=0.40,
                     evidence_ratio_zp=lit_over_sim["DLA"]["ratio_zp"],
                     evidence_ratio_slope=lit_over_sim["DLA"]["ratio_slope"],
                     note="unchanged (re-verified; deliberately weak z-slope kept)")),
        sources=dict(
            arxiv=LD.ARXIV,
            verified=dict(
                POW10_T4="implementer ar5iv 2026-07-18 + both designers (v1 only)",
                ZAFAR13_T3="implementer ar5iv 2026-07-18 + both designers (verbatim)",
                PW09_T1="implementer ar5iv 2026-07-18 + Bayesian designer",
                OMEARA13_wmean="implementer + both designers (verbatim)",
                OMEARA13_T5_row="Bayesian designer ONLY (ar5iv truncates in-house)",
                OMEARA13_T9="UNVERIFIED — EXCLUDED",
                POW10_journal_fixes="UNVERIFIED (IOP paywall) — delta-widened default arm",
                FUMAGALLI13="implementer + both designers (verbatim)")),
        input_arrays=dict(
            ELL_X_CUMULATIVE_GE17P5_TAU2=LD.ELL_X_CUMULATIVE_GE17P5_TAU2,
            ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3=LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3,
            ELL_X_DLA_GE20P3_PW09_T1=LD.ELL_X_DLA_GE20P3_PW09_T1),
        estimator_per_class=dict(LLS="gls_log(corrected points, common-mode kernel cov)",
                                 subDLA="poisson_glm", DLA="poisson_glm (WLS anchor printed)"),
        laws=dict(
            subDLA=dict(A=s["A"], gamma=s["gamma"], A_pivot=s["A_pivot"],
                        cov=s["cov"], deviance_dof=s["deviance_dof"],
                        glm_vs_wls=s["glm_vs_wls"]),
            DLA=dict(A=d["A"], gamma=d["gamma"], A_pivot=d["A_pivot"], cov=d["cov"],
                     deviance_dof=d["deviance_dof"], wls_anchor=d["wls_anchor"],
                     glm_vs_wls=d["glm_vs_wls"]),
            cum_uncorrected=dict(A=c["A"], gamma=c["gamma"], A_pivot=c["A_pivot"],
                                 cov=c["cov"], chi2_red=c["chi2_red"]),
            LLS_per_kernel={n: k["law"] for n, k in kernels.items()}),
        kernel_table=dict(
            z_vals=k1_tab["z_vals"], K1_r=k1_tab["r"], K1_r16=k1_tab["r16"],
            K1_r84=k1_tab["r84"], K1b_smooth=dict(r_pivot=k1b["r_pivot"], eta=k1b["eta"]),
            floor_factor=dict(z_vals=F_tab["z_vals"], F=F_tab["F"]),
            per_kernel_r_z_eval={n: k["r_z_eval"] for n, k in kernels.items()},
            budget=budget),
        adjudication=adj,
        alpha_lls_z3=dict(per_kernel={n: k["alpha_lls_z3"] for n, k in kernels.items()},
                          deployed=PRE_SWAP_ALPHA_Z3, adopted=alpha_adopted,
                          path=tier, xbar_z3=xbar_used,
                          deployed_law_validation=alpha_deployed_check),
        bracket=bracket,
        band_proposals={n: [round(k["alpha_lls_z3"] * 0.83, 2),
                            round(k["alpha_lls_z3"] * 1.21, 2)] for n, k in kernels.items()},
        widths={n: k["one_x"] for n, k in kernels.items()},
        lit_over_sim=lit_over_sim,
        telescoping={n: k["telescoping"] for n, k in kernels.items()},
        K3b_diagnostic=res["K3b_diagnostic"],
        sensitivity=dict(
            drop_z180_subDLA=dict(A=sd["A"], gamma=sd["gamma"],
                                  deviance_dof=sd["deviance_dof"]),
            larger_side_cum=dict(A=sens["larger_side"]["cum_uncorrected"]["A"],
                                 gamma=sens["larger_side"]["cum_uncorrected"]["gamma"]),
            journal_fix_cum=dict(A=sens["journal_fix"]["cum_uncorrected"]["A"],
                                 gamma=sens["journal_fix"]["cum_uncorrected"]["gamma"]),
            journal_fix_K3=dict(A=sens["journal_fix"]["K3"]["law"]["fit"]["A"],
                                gamma=sens["journal_fix"]["K3"]["law"]["fit"]["gamma"])),
        cache_provenance=dict(path=inputs["path"], shapes=inputs["shapes"],
                              sha256_f_nhi=inputs["sha256_f_nhi"],
                              same_file_consistency=sfc),
        xbar_z3=xbar_used,
    )
    if os.environ.get("DERIVE_WRITE_JSON", "0") == "1":
        # closing-panel fix 4: never silently overwrite the committed artifact of record —
        # a casual re-run would re-stamp created/git_commit and drift the sha pin
        with open(OUT_JSON, "w") as fh:
            json.dump(to_jsonable(payload), fh, indent=1, sort_keys=True)
        print(f"\nwrote {OUT_JSON} (kernel_chosen={ADOPTED_KERNEL}; PI-adopted 2026-07-18, "
              f"spec step 10/12)")
    else:
        print(f"\nJSON NOT written (committed artifact preserved); set DERIVE_WRITE_JSON=1 "
              f"to regenerate {OUT_JSON}")


def one_x_widths(law_block, kb):
    """Width variants (delegates to the shared module; PI decision 2)."""
    return LD.corrected_width_variants(law_block, kb["s_r3"])


def lit_over_sim_refit(inputs, kernels, sub_law, dla_law, cum_law):
    """HCD_LIT_OVER_SIM(+SLOPE) refit against the corrected definition-matched laws,
    the plot_dndx_vs_literature.py:95-108 construction executed in-house: sim slope from
    np.polyfit of log(class dN/dX) on log(1+z) over z in [2.2,4.6]; ratio at z_pivot=3;
    ratio_slope = gamma_lit - gamma_sim."""
    z_p = 3.0
    zg = inputs["zg"]
    out = {}
    cols = dict(LLS=inputs["lls"], subDLA=inputs["sub"], DLA=inputs["dla"])
    lit_laws = dict(subDLA=(sub_law["A"], sub_law["gamma"]),
                    DLA=(dla_law["A"], dla_law["gamma"]))
    for cls, col in cols.items():
        inr = (zg >= 2.2) & (zg <= 4.6) & (col > 0)
        gs = np.polyfit(np.log(1 + zg[inr]), np.log(col[inr]), 1)
        sim_zp = float(np.exp(gs[1]) * (1 + z_p) ** gs[0])
        if cls == "LLS":
            per_kernel = {}
            for n, k in kernels.items():
                lit_zp = k["law"]["A"] * (1 + z_p) ** k["law"]["gamma"]
                per_kernel[n] = dict(ratio_zp=float(lit_zp / sim_zp),
                                     ratio_slope=float(k["law"]["gamma"] - gs[0]))
            out[cls] = dict(gamma_sim=float(gs[0]), sim_zp=sim_zp, per_kernel=per_kernel,
                            pre_swap=dict(ratio_zp=1.06, ratio_slope=0.95))
        else:
            A_l, g_l = lit_laws[cls]
            lit_zp = A_l * (1 + z_p) ** g_l
            out[cls] = dict(gamma_sim=float(gs[0]), sim_zp=sim_zp,
                            ratio_zp=float(lit_zp / sim_zp),
                            ratio_slope=float(g_l - gs[0]),
                            deployed=dict(ratio_zp={"subDLA": 1.00, "DLA": 1.34}[cls],
                                          ratio_slope={"subDLA": 0.15, "DLA": 0.40}[cls]))
    out["note"] = ("subDLA prior slot STAYS sim-anchored 1.00 (PI decision 9; the corrected "
                   "lit/sim ratio above is recorded as EVIDENCE, inside the 0.40 width). "
                   "LLS ratio_slope must stay < 1.5 (the forward-zslope guard floor) or the "
                   "guard goes to the PI.")
    return out


if __name__ == "__main__":
    main()
