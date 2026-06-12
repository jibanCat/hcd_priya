"""Per-mock corner-plot GALLERY for the STEP-A closures, split COSMO vs IGM, into a .md.

For every distinct closure fiducial with chains (checkpoints/stepA/<FID>_c*.npz), render two GetDist
triangles with truth markers (dashed):
  COSMO block : ns, Ap, alpha_LLS, alpha_subDLA, alpha_DLA  (cosmology + the HCD nuisances it is
                degenerate with — the science block of this analysis)
  IGM block   : herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback  (the 7 PRIYA astro params)
and write a grouped markdown gallery embedding them + a per-mock summary (n_s, bias_Ap, bias_ns).

bias_z = (truth - pooled_mean)/pooled_sd (the run_stepA battery convention).

ENV: getdist lives in emu-3.9. Run:
  /home/mfho/.conda/envs/emu-3.9/bin/python3 scripts/plot_corner_gallery.py
Optional: pass a space-separated FID list to restrict; default = all distinct completed mocks.
"""
import sys, glob, os, re, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
from getdist import MCSamples, plots

CK = "checkpoints/stepA"                                    # chains + all_mocks_bias.json (code repo, local)
NOTES = "/home/mfho/hcd_priya_notes"                        # outputs go to the PRIVATE notes repo
OUTDIR = f"{NOTES}/figures/analysis/05_likelihood/corner"   # where the corner PNGs are written
LINK = "../../figures/analysis/05_likelihood/corner"        # md is in notes/docs/superpowers/ -> relative
MD = f"{NOTES}/docs/superpowers/2026-06-11-corner-gallery.md"
os.makedirs(OUTDIR, exist_ok=True)

# training-box (PARAM_LIMITS) for unit->physical; Ap shown in 1e-9.
LIMS = {"ns": (0.8, 1.05), "Ap": (1.2e-9, 2.6e-9), "herei": (3.5, 4.5), "heref": (2.2, 3.2),
        "alphaq": (1.3, 3.0), "hub": (0.65, 0.75), "omegamh2": (0.14, 0.146),
        "hireionz": (6.5, 8.0), "bhfeedback": (0.03, 0.07)}
LAB = {"ns": r"n_s", "Ap": r"A_p[10^{-9}]", "alpha_lls": r"\alpha_{\rm LLS}",
       "alpha_subdla": r"\alpha_{\rm subDLA}", "alpha_dla": r"\alpha_{\rm DLA}",
       "herei": r"z_{\rm HeII,i}", "heref": r"z_{\rm HeII,f}", "alphaq": r"\alpha_q", "hub": "h",
       "omegamh2": r"\Omega_m h^2", "hireionz": r"z_{\rm HI}", "bhfeedback": r"\epsilon_{\rm BH}"}
COSMO = ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"]
IGM = ["herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]

# grouped presentation order for the md
GROUPS = [
    ("Phase-4 anchors (separate-inference, sim-mean center)", ["D_f3", "D_f4", "D_f6", "D_f7", "K_f4", "K_f6"]),
    ("HCD width / sensitivity arms (D_f6)", ["D_f6_lit", "D_f6_sig40", "D_f6_tau0x", "D_f6_sigL08",
        "D_f6_sigL25", "D_f6_sigL80", "D_f6_sigS020", "D_f6_sigS080", "D_f6_sigS150"]),
    ("IGM-parameter stress fiducials", ["IGM_herei_hi", "IGM_heref_lo", "IGM_heref_hi",
        "IGM_alphaq_lo", "IGM_alphaq_hi", "IGM_bhfb_lo"]),
    ("Per-survey LLS pin", ["D_lls", "D_lls_m", "D_lls_m30", "D_llsmed", "D_llsmed30", "K_lls"]),
    ("Multi-fold width check (matched center)", ["D_lmed3_15", "D_lmed3_30", "D_lmed5_15",
        "D_lmed5_30", "D_lmed7_15", "D_lmed7_30"]),
]


def to_phys(x, nm):
    if nm in LIMS:
        lo, hi = LIMS[nm]; v = lo + x * (hi - lo)
        return v * 1e9 if nm == "Ap" else v
    return x


def pool(fid):
    ps = sorted(p for p in glob.glob(f"{CK}/{fid}_c*.npz") if re.match(rf"{re.escape(fid)}_c\d+\.npz$", os.path.basename(p)))
    if not ps:
        return None
    packs, names, truth = [], None, None
    for p in ps:
        z = np.load(p, allow_pickle=True)
        names = list(z["names"]) if names is None else names
        truth = np.asarray(z["truth_vec"]) if truth is None else truth
        packs.append(np.asarray(z["packed"]))
    n = min(x.shape[0] for x in packs)
    return np.concatenate([x[:n] for x in packs], 0), names, truth, len(ps), float(np.load(ps[0], allow_pickle=True)["n_s"])


def corner(P, names, truth, block, fid, tag):
    cols = [to_phys(P[:, names.index(b)], b) for b in block]
    tv = [to_phys(truth[names.index(b)], b) for b in block]
    mcs = MCSamples(samples=np.column_stack(cols), names=block, labels=[LAB[b] for b in block])
    g = plots.get_subplot_plotter(width_inch=1.4 * len(block))
    g.settings.alpha_filled_add = 0.6
    g.triangle_plot([mcs], filled=True, markers={b: tv[i] for i, b in enumerate(block)},
                    marker_args={"color": "k", "lw": 1.0})
    g.export(f"{OUTDIR}/{fid}_{tag}.png"); import matplotlib.pyplot as plt; plt.close("all")
    return f"{LINK}/{fid}_{tag}.png"


def bias(P, names, truth, nm):
    j = names.index(nm); sd = P[:, j].std()
    return (truth[j] - P[:, j].mean()) / sd if sd > 0 else float("nan")


PRIOR_TABLE = """## Current priors (the likelihood baseline)

**Cosmology + IGM** — θ ~ Uniform over the sampling box (IGM params restricted to original PRIYA; n_s kept extended + C_emu step-inflation > 0.995):

| param | prior | | param | prior |
|---|---|---|---|---|
| n_s | U[0.8, 1.05] | | hub | U[0.65, 0.75] |
| A_p | U[1.2, 2.6]×10⁻⁹ | | Ω_m h² | U[0.14, 0.146] |
| herei | U[3.5, **4.1**] *(restricted; train box →4.5)* | | hireionz | U[6.5, 8.0] |
| heref | U[**2.6**, 3.2] *(restricted; train box 2.2→)* | | bhfeedback | U[0.03, 0.07] |
| alphaq | U[1.3, **2.5**] *(restricted; train box →3.0)* | | | |

**Mean flux (τ₀, PRIYA 2-param):** `tau0_amp ~ U[0.75, 1.25]`, `dtau0 ~ U[−0.40, 0.25]`; τ₀(z)=amp·((1+z)/4)^dtau0·Kim07.

**HCD incidence α_c** — TruncatedNormal(≥0) for LLS/subDLA, softplus(≥0) for DLA; center = (lit/sim @z=3)·w_c (per-survey on the real-fit path):

| class | lit/sim @z3 | σ/μ (DESI) | σ/μ (KS) | notes |
|---|---|---|---|---|
| LLS | 1.06 × boost (DESI ×1.0 / **KS ×2.5**) | **0.30** | **0.40** | KS = selection excess (arXiv:2509.18271); width set by D_lmed multi-fold check |
| subDLA | 1.00 | 0.40 | 0.40 | sim-centered (PRIYA in-situ); was 0.76 |
| DLA | 1.34 × 0.10 (masked-DLA residual) | 0.50 | 0.50 | widened above z=3.5 |

**HCD z-slope s_c** (marginalized): center (0.95, 0.15, 0.40), width (0.52, 0.53, 0.33), pivot z=3.
**Metals** OFF in the closure (DESI real-fit forward-models SiIII/SiII via `_metal_factor`; KS side-band-subtracts → off).
"""

# --- per-mock "purpose of this test" blurbs (md-expert agents, 2026-06-11) ---------------------
PURPOSE = {
  "D_f3": "DESI separate-inference anchor at n_s=0.901, sim-mean LLS center, default σ_LLS=0.15: checks cosmology recovery and contributes the low-tilt point to the zero-mean cross-fiducial A_p scatter that shows the LLS prior is a ~1σ A_p lever, not a likelihood bug.",
  "D_f4": "DESI anchor at n_s=0.920, sim-mean center, σ_LLS=0.15: a sub-Planck tilt point mapping where DESI A_p lands under the tight default LLS prior, fixing the cross-fiducial A_p scatter as zero-mean.",
  "D_f6": "DESI anchor at the Planck reference n_s=0.966 — the baseline same-mock for every HCD width/sensitivity arm; its +1.02σ A_p is the canonical demonstration that a tight LLS prior offset from the sim's true LLS abundance pushes DESI A_p ~1σ.",
  "D_f7": "DESI anchor at n_s=0.998 (eBOSS/extended ridge): tests recovery at the high-tilt edge near the C_emu inflation ridge, anchoring the high-n_s end of the scatter (A_p +0.01σ).",
  "K_f4": "KODIAQ-SQUAD anchor at n_s=0.920: the survey contrast to DESI — KS is far less exposed to the LLS prior on A_p (−0.43σ) and routes residual bias into n_s, motivating per-survey treatment.",
  "K_f6": "KODIAQ-SQUAD anchor at Planck n_s=0.966: the matched KS counterpart to D_f6 — KS keeps A_p clean (+0.02σ) while any LLS bias appears in n_s, confirming the DESI-A_p-vs-KS-n_s split.",
  "D_f6_lit": "D_f6 mock with the LLS prior CENTER set to the literature dN/dX instead of sim-mean (σ0.15): the clean center-swap arm showing the center barely moves A_p (+0.99 vs +1.02σ) — the demonstrated lever is the prior WIDTH, not which offset center is enforced.",
  "D_f6_sig40": "D_f6 with the LLS prior loosened to σ_LLS=0.40: the clean same-mock width cut — A_p +1.02→+0.29σ but it redistributes into n_s (+0.53σ) and widens A_p, proving flattening reshuffles rather than removes the bias.",
  "D_f6_tau0x": "D_f6 with τ₀ truth at the extreme upper PRIYA corner: the τ₀-funnel / τ₀×cosmology stress — the smooth 2-param τ₀ handles it with 0 divergences (no funnel), while residual A_p (+1.16σ) still comes via the tight-LLS×offset channel, not a sampling pathology.",
  "D_f6_sigL08": "D_f6 σ_LLS scan point 0.08 (very tight, σ_subDLA=0.40): the tight end of the LLS-width sweep, pinning α_LLS hardest so A_p compensates near maximum (+1.13σ).",
  "D_f6_sigL25": "D_f6 σ_LLS scan point 0.25: a moderate-width point that (with 0.40) brackets the joint-bias minimum (A_p +0.55σ, n_s +0.30σ), identifying the recommended LLS width band.",
  "D_f6_sigL80": "D_f6 σ_LLS scan point 0.80 (very flat): A_p even crosses zero (−0.21σ) but the bias migrates wholly into n_s (+0.92σ) — flat is no better than tight; the joint bias rises at both ends.",
  "D_f6_sigS020": "D_f6 σ_subDLA scan point 0.20 (LLS width fixed 0.15): the tight end — subDLA width behaves OPPOSITELY to LLS, tightening gives the smallest joint bias, the basis for keep-subDLA-tight.",
  "D_f6_sigS080": "D_f6 σ_subDLA scan point 0.80: widening the least-constrained HCD class monotonically WORSENS A_p (+1.43σ) by handing the emulator/τ₀ misfit a free low-k knob.",
  "D_f6_sigS150": "D_f6 σ_subDLA scan point 1.50 (very flat): confirms the monotonic worsening (A_p +1.50σ) — the fix is not 'loosen everything'; subDLA must stay tight.",
  "IGM_herei_hi": "Held-out sim at the HIGH edge of HeII-reionization START, default DESI priors: does an IGM extreme bias cosmology via HCD marginalization or emulator-LOSO? Here the closure A_p (+2.0σ) exceeds its standalone emulator-LOSO Fisher bias, so the tight-LLS mechanism adds on top of edge inaccuracy.",
  "IGM_heref_lo": "Held-out sim at the LOW edge of HeII-reionization END: tests the IGM-extreme bias (mainly n_s) and confirms it is emulator-LOSO edge error with a flat HCD↔cosmology coupling, not HCD marginalization.",
  "IGM_heref_hi": "Held-out sim at the HIGH edge of HeII-reionization END: a large n_s pull driven by emulator-LOSO extrapolation at a design-sparse edge, with corr(A_p,α_LLS) flat.",
  "IGM_alphaq_lo": "Held-out sim at the LOW edge of QSO spectral slope alphaq: the well-behaved counterpart — a benign IGM extreme leaves cosmology in-gate with flat HCD coupling.",
  "IGM_alphaq_hi": "Held-out sim at the HIGH edge of alphaq (widened-box extension): the worst-case corner — its large A_p bias is almost entirely standalone emulator-LOSO Fisher error (+3.45σ ≈ closure +3.74σ); the likelihood/HCD adds nothing → motivates the box restriction.",
  "IGM_bhfb_lo": "Held-out sim at the LOW edge of BH-feedback: an IGM extreme where HCD/τ₀ freedom partially ABSORBS emulator-LOSO error (closure A_p −0.65σ vs standalone emulator +1.29σ), with flat HCD coupling.",
  "D_lls": "DESI per-survey LLS pin, tight σ0.15, mock at the cosmic-average LLS (boost ×1.0, ~6% below the pin center): the deliberately-mismatched arm exposing the dominant DESI A_p risk — a tight LLS prior enforcing a center offset pins α_LLS and forces A_p to compensate ~1σ.",
  "D_lls_m": "DESI LLS pin, σ0.15, mock matched to the pin center (×1.06): the matched-center control — the ~1σ A_p persists, first evidence σ0.15 is too tight (later refined: this fold6 sim is itself LLS-poor, so the 'match' was still +0.8σ high).",
  "D_lls_m30": "DESI LLS pin, moderate σ0.30, matched (×1.06): validates widening the DESI width 0.15→0.30 — both A_p (+1.01→+0.65σ) and n_s land in-gate via the predicted A_p↔n_s redistribution.",
  "D_llsmed": "DESI LLS pin σ0.15 on a sim whose w_LLS ≈ the population median, so the lit pin center genuinely equals truth — the UN-CONFOUNDING arm: with a truly-matched center, σ0.15 gives A_p −0.05σ and faithful α_LLS, proving the DESI A_p lever is the prior CENTER, not the width.",
  "D_llsmed30": "Median-w_LLS sim (truly matched center) with moderate σ0.30: the companion to D_llsmed — once the center is correct a looser prior lets the A_p↔α_LLS degeneracy drift (A_p −0.73σ), confirming σ0.30 was compensating for a wrong center, not a fundamental fix.",
  "K_lls": "KODIAQ-SQUAD LLS pin, broad σ0.40, mock injecting the KS selection excess (boost ×2.65 = 2.5× cosmic, the arXiv:2509.18271 α_LLS≈2 level): the cleanest closure — A_p +0.004σ, LLS dN/dX faithful, honest wide α_LLS posterior; the broad prior lets the data set the incidence without baking in the contested number.",
  "D_lmed3_15": "Multi-fold width check at fold3 (n_s 0.907, the −0.92σ DESI A_p LOSO tail), median-w_LLS matched center, tight σ0.15: extends the center-vs-width un-confounding across n_s and tests whether the n_s residual is the LLS width or this fold's LOSO scatter.",
  "D_lmed3_30": "Fold3 (n_s 0.907) matched center, σ0.30: the width-paired counterpart to D_lmed3_15 — does widening on the adversarial low-A_p fold redistribute as predicted, and is the n_s residual width or LOSO scatter?",
  "D_lmed5_15": "Multi-fold width check at fold5 (n_s 0.953, mid), matched center, σ0.15: validates σ0.15 away from the worst fold to establish the zero-mean LOSO A_p scatter rather than a single-fold point.",
  "D_lmed5_30": "Fold5 (n_s 0.953) matched center, σ0.30: the mid-fold width pair checking the σ0.15→0.30 redistribution holds and the n_s residual source.",
  "D_lmed7_15": "Multi-fold width check at fold7 (n_s 0.982, high-n_s ridge), matched center, σ0.15: the upper-n_s point — its large A_p (+2.44σ) flags emulator-LOSO at the sparse high-n_s ridge, not a prior effect.",
  "D_lmed7_30": "Fold7 (n_s 0.982) matched center, σ0.30: completes the across-folds σ0.15-vs-σ0.30 comparison at the high-n_s end; the bias here is the high-ridge emulator-LOSO outlier.",
}

bias_all = json.load(open(f"{CK}/all_mocks_bias.json")) if os.path.exists(f"{CK}/all_mocks_bias.json") else {}
want = sys.argv[1:] if len(sys.argv) > 1 else None
md = ["# STEP-A closure corner gallery (cosmo vs IGM)\n",
      "*Auto-generated by `scripts/plot_corner_gallery.py` (code repo) into this private notes repo. "
      "Per mock: a purpose blurb, a cosmology-bias row, a COSMO corner (n_s, A_p + the HCD α nuisances) "
      "and an IGM corner (the 7 PRIYA astro params). Dashed = truth; bias = (truth−post)/σ.*\n",
      PRIOR_TABLE]
# cosmology-bias summary table (all runs, in group order)
bt = ["## Cosmology-bias summary (all runs)\n",
      "| mock | n_s | chains | bias A_p (σ) | bias n_s (σ) | div | R̂_max |",
      "|---|---|---|---|---|---|---|"]
for _t, fids in GROUPS:
    for fid in fids:
        b = bias_all.get(fid)
        if b:
            bt.append(f"| {fid} | {b['n_s']} | {b['nchains']} | {b['bias_Ap']:+.2f} | "
                      f"{b['bias_ns']:+.2f} | {b['ndiv']} | {b['rhat'] if b['rhat'] else '—'} |")
md.append("\n".join(bt) + "\n")
done = set()
for title, fids in GROUPS:
    rows = []
    for fid in fids:
        if want and fid not in want:
            continue
        r = pool(fid)
        if r is None:
            continue
        P, names, truth, nc, ns = r
        c_png = corner(P, names, truth, COSMO, fid, "cosmo")
        i_png = corner(P, names, truth, IGM, fid, "igm")
        bA, bN = bias(P, names, truth, "Ap"), bias(P, names, truth, "ns")
        rows.append(f"### {fid}  (n_s={ns:.3f}, {nc} chains; bias A_p={bA:+.2f}σ, n_s={bN:+.2f}σ)\n\n"
                    f"*{PURPOSE.get(fid, '')}*\n\n"
                    f"![{fid} cosmo]({c_png})\n\n![{fid} IGM]({i_png})\n")
        done.add(fid)
    if rows:
        md.append(f"\n## {title}\n\n" + "\n".join(rows))
with open(MD, "w") as f:
    f.write("\n".join(md))
print(f"wrote {MD}  ({len(done)} mocks)")
