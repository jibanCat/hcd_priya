"""Walkthrough reference: plot + tabulate the Phase-C Leg-B INFERENCE PRIORS.

Reads the actual prior definitions (no training, no NUTS):
  - θ9 cosmology box       : data.PARAM_LIMITS (Uniform in-box).
  - τ₀ mean-flux ladder    : meanflux_prior.meanflux_tau0_prior(center="becker13").
  - HCD incidence α        : inference.hcd_incidence_prior (per-class μ,σ on the cache w_c).
  - HCD z-slope nuisance   : closure_legb.ZSLOPE_PRIOR_SIGMA (+ z-edge inflation).
  - metals / resolution    : data_likelihood defaults (OFF in the Leg-B closure model).

Emits:
  figures/analysis/05_likelihood/stepA_priors.png
  figures/analysis/05_likelihood/stepA_priors.txt
"""
from __future__ import annotations

import functools
import numpy as np

print = functools.partial(print, flush=True)

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import PARAM_LIMITS, KIM_AMP, KIM_SLOPE, load_cache
from hcd_analysis.emulator.meanflux_prior import (
    meanflux_tau0_prior, becker13_tau0, DEFAULT_FRAC_SIGMA,
    BECKER13_TAU0, BECKER13_BETA, BECKER13_C, BECKER13_ZREF)
from hcd_analysis.emulator.inference import (
    hcd_incidence_prior, lit_over_sim_at_z, HCD_PRIOR_FRAC_SIGMA,
    HCD_DLA_RESIDUAL_FRAC, HCD_DLA_Z_RELIABLE, HCD_Z_PIVOT,
    HCD_LIT_OVER_SIM, HCD_LIT_OVER_SIM_SLOPE)
from hcd_analysis.emulator.closure_legb import ZSLOPE_PRIOR_SIGMA

REPO = "/home/mfho/hcd_priya"
CACHE = f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
OUT_PNG = f"{REPO}/figures/analysis/05_likelihood/stepA_priors.png"
OUT_TXT = f"{REPO}/figures/analysis/05_likelihood/stepA_priors.txt"

PARAM_NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub",
               "omegamh2", "hireionz", "bhfeedback")
# external cosmology reference points for n_s (where Planck/eBOSS sit in the box)
NS_PLANCK = 0.965     # Planck 2018 TT,TE,EE+lowE+lensing
NS_EBOSS = 1.009      # eBOSS Lya-only (du Mas des Bourboux+2020 ~1.01)

# z-edge inflation factor for the slope width (closure_legb dla_inflate / slope-tradeoff
# script edge(z)=1+beta*(clip(2.5-z)+clip(z-3.5)); the per-leg z-mean ≈ 1.25, beta=1).
ZSLOPE_EDGE_INFLATE = 1.25


def main():
    print("[stepA-priors] loading cache for the structural HCD weights w_c ...")
    d = load_cache(CACHE)
    w_c_med = np.median(d["w_c_cache"][:, 1:], axis=0)   # (3,) [LLS,subDLA,DLA]

    # ---- θ9 box ----
    lo = PARAM_LIMITS[:, 0]; hi = PARAM_LIMITS[:, 1]

    # ---- τ₀(z) prior, Becker13 center (production anchor) ----
    z_grid = np.linspace(2.2, 4.6, 50)
    mu_b13, sig_b13 = meanflux_tau0_prior(jnp.asarray(z_grid), center="becker13")
    mu_b13 = np.asarray(mu_b13); sig_b13 = np.asarray(sig_b13)
    kim_z = KIM_AMP * (1.0 + z_grid) ** KIM_SLOPE
    alpha_mu = mu_b13 / kim_z         # ladder coord center the Normal is placed on
    alpha_sig = sig_b13 / kim_z       # ladder coord width

    # ---- HCD incidence α prior at z_pivot=3 (the per-leg anchor the closure uses) ----
    a_mu, a_sig = hcd_incidence_prior(jnp.asarray(w_c_med), z=HCD_Z_PIVOT)
    a_mu = np.asarray(a_mu); a_sig = np.asarray(a_sig)
    r_at_pivot = np.asarray(lit_over_sim_at_z(HCD_Z_PIVOT))

    # ---- HCD z-slope nuisance prior ----
    s_mu = np.asarray(HCD_LIT_OVER_SIM_SLOPE)
    s_sig_lit = np.asarray(ZSLOPE_PRIOR_SIGMA)
    s_sig_edge = s_sig_lit * ZSLOPE_EDGE_INFLATE

    # =================================================================== FIGURE
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24,
                          left=0.10, right=0.97, top=0.92, bottom=0.07)
    fig.suptitle("Phase-C Leg-B inference priors (closure / real fit)  —  "
                 "as the NUTS sampler sees them", fontsize=15, fontweight="bold")

    # (a) θ9 Uniform boxes ------------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    yy = np.arange(len(PARAM_NAMES))[::-1]
    for i, (name, l, h) in enumerate(zip(PARAM_NAMES, lo, hi)):
        y = yy[i]
        axa.plot([l, h], [y, y], lw=7, color="#4477AA", solid_capstyle="butt", alpha=0.85)
        axa.text(l, y + 0.28, f"{l:g}", ha="left", va="bottom", fontsize=7.5, color="#225")
        axa.text(h, y + 0.28, f"{h:g}", ha="right", va="bottom", fontsize=7.5, color="#225")
    # n_s reference points (top bar)
    ns_y = yy[0]
    axa.plot(NS_PLANCK, ns_y, "v", color="crimson", ms=9, zorder=5)
    axa.text(NS_PLANCK, ns_y - 0.42, f"Planck\n{NS_PLANCK}", ha="center", va="top",
             fontsize=7, color="crimson")
    axa.plot(NS_EBOSS, ns_y, "^", color="darkgreen", ms=9, zorder=5)
    axa.text(NS_EBOSS, ns_y - 0.42, f"eBOSS\n{NS_EBOSS}", ha="center", va="top",
             fontsize=7, color="darkgreen")
    axa.set_yticks(yy); axa.set_yticklabels(PARAM_NAMES, fontsize=9)
    axa.set_xlim(-0.05, 1.10)
    axa.set_ylim(-1.5, len(PARAM_NAMES) - 0.2)
    axa.set_xlabel("parameter value (physical units; ranges differ per row)")
    axa.set_title("(a) θ9 cosmology/IGM  —  Uniform(lo, hi)  [flat in-box]", fontsize=11)
    axa.text(0.02, 0.02, "Ap shown ×1e-9; each bar spans its own design box.\n"
             "Sampled as Uniform(0,1)^9 in the unit cube + auto-bijector.",
             transform=axa.transAxes, fontsize=7, va="bottom", color="#555")
    # rescale Ap row for readability (Ap in 1e-9)
    iAp = PARAM_NAMES.index("Ap")
    yAp = yy[iAp]
    axa.plot([lo[iAp] * 1e9, hi[iAp] * 1e9], [yAp, yAp], lw=7, color="#CC6677",
             solid_capstyle="butt", alpha=0.9)
    axa.text(lo[iAp] * 1e9, yAp + 0.28, f"{lo[iAp]*1e9:g}e-9", ha="left", va="bottom",
             fontsize=7.5, color="#722")
    axa.text(hi[iAp] * 1e9, yAp + 0.28, f"{hi[iAp]*1e9:g}e-9", ha="right", va="bottom",
             fontsize=7.5, color="#722")

    # (b) τ₀(z) prior band ------------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    axb.plot(z_grid, mu_b13, color="#117733", lw=2,
             label=r"Becker+2013 center $\mu_z=\tau_{\rm eff}(z)$")
    axb.fill_between(z_grid, mu_b13 - sig_b13, mu_b13 + sig_b13, color="#117733",
                     alpha=0.25, label=r"$\pm\,\sigma_z$  ($\sigma=5\%\,\mu$)")
    axb.plot(z_grid, kim_z, color="grey", ls="--", lw=1.3,
             label=r"Kim2013 $(\alpha{=}1)$ ladder anchor")
    axb.set_xlabel("redshift z"); axb.set_ylabel(r"$\tau_0(z)=-\ln\langle F\rangle$")
    axb.set_title(r"(b) $\tau_0$ mean-flux prior  —  per-z Gaussian on the $\alpha=\tau_0/$Kim ladder",
                  fontsize=10.5)
    axb.legend(fontsize=8, loc="upper left")
    # inset: the ladder-coordinate center+width the Normal is actually placed on
    axins = axb.inset_axes([0.56, 0.10, 0.40, 0.40])
    axins.plot(z_grid, alpha_mu, color="#332288", lw=1.6)
    axins.fill_between(z_grid, alpha_mu - alpha_sig, alpha_mu + alpha_sig,
                       color="#332288", alpha=0.25)
    axins.axhline(1.0, color="grey", ls=":", lw=1)
    axins.set_title(r"ladder coord $\alpha=\tau_0/$Kim", fontsize=7.5)
    axins.set_xlabel("z", fontsize=7); axins.tick_params(labelsize=6.5)

    # (c) HCD incidence α prior -------------------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    cls = ["LLS", "subDLA", "DLA"]
    xc = np.arange(3)
    colors = ["#4477AA", "#DDCC77", "#CC6677"]
    for i in range(3):
        axc.errorbar(xc[i], a_mu[i], yerr=a_sig[i], fmt="o", ms=9, capsize=6,
                     color=colors[i], elinewidth=2.5, mec="k")
        axc.text(xc[i] + 0.12, a_mu[i],
                 f"μ={a_mu[i]:.4f}\nσ={a_sig[i]:.4f}\n(σ/μ={HCD_PRIOR_FRAC_SIGMA[i]:.0%})",
                 fontsize=7.8, va="center")
    axc.axhline(0.0, color="crimson", ls="--", lw=1.2, alpha=0.7)
    axc.text(2.0, a_sig[2] * 0.5, "DLA: softplus(Normal)\none-sided, Gaussian-at-0",
             fontsize=7.5, color="crimson", ha="center")
    axc.set_xticks(xc); axc.set_xticklabels(cls)
    axc.set_ylabel(r"$\alpha_c$ incidence (sightline weight) at $z_{\rm pivot}=3$")
    axc.set_title("(c) HCD incidence α prior  —  Normal, center = (lit/sim)·$w_c$ (DLA: ×0.30 residual)",
                  fontsize=10.5)
    axc.set_xlim(-0.4, 2.7)
    axc.text(0.02, 0.97,
             f"cache $w_c$=[{w_c_med[0]:.3f}, {w_c_med[1]:.4f}, {w_c_med[2]:.4f}]\n"
             f"lit/sim@z3=[{r_at_pivot[0]:.2f}, {r_at_pivot[1]:.2f}, {r_at_pivot[2]:.2f}]\n"
             f"per-leg anchor: KS σ_scale=0.27, DESI σ_scale=0.12",
             transform=axc.transAxes, fontsize=7.2, va="top", color="#444")

    # (d) HCD z-slope nuisance prior -------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    sc_names = ["s_lls", "s_subdla", "s_dla"]
    yd = np.arange(3)[::-1]
    for i in range(3):
        y = yd[i]
        # edge-inflated width (lighter, wider) then literature width (darker)
        axd.plot([s_mu[i] - s_sig_edge[i], s_mu[i] + s_sig_edge[i]], [y, y],
                 lw=9, color="#BBBBBB", solid_capstyle="round", alpha=0.7,
                 zorder=1)
        axd.plot([s_mu[i] - s_sig_lit[i], s_mu[i] + s_sig_lit[i]], [y, y],
                 lw=9, color="#882255", solid_capstyle="round", alpha=0.6, zorder=2)
        axd.plot(s_mu[i], y, "o", ms=8, color="k", zorder=3)
        axd.text(s_mu[i], y + 0.22,
                 f"{sc_names[i]}: μ={s_mu[i]:.2f}, σ_lit={s_sig_lit[i]:.2f} "
                 f"(edge ×1.25 → {s_sig_edge[i]:.2f})",
                 ha="center", va="bottom", fontsize=7.8)
    axd.axvline(0.0, color="grey", ls=":", lw=1)
    axd.set_yticks(yd); axd.set_yticklabels(sc_names, fontsize=9)
    axd.set_ylim(-0.7, 2.7)
    axd.set_xlabel(r"$s_c$ = d ln(dN/dX) / d ln(1+z)  (HCD incidence z-slope)")
    axd.set_title("(d) HCD z-slope nuisance  —  Normal(center=lit slope, σ=lit WLS 1σ)  [M3-marginalized]",
                  fontsize=10.5)
    axd.text(0.02, 0.03,
             "center = HCD_LIT_OVER_SIM_SLOPE (NOT 0 — isolates marginalization cost,\n"
             "not a center shift).  dark = lit WLS 1σ;  grey = z-edge-inflated (×1.25).",
             transform=axd.transAxes, fontsize=7.2, va="bottom", color="#444")

    fig.savefig(OUT_PNG, dpi=150)
    print(f"[stepA-priors] wrote {OUT_PNG}")

    # =================================================================== TXT
    lines = []
    def w(s=""): lines.append(s)
    w("=" * 78)
    w("Phase-C Leg-B inference priors  (closure-under-misspecification / real fit)")
    w("Source: hcd_analysis/emulator/{data,meanflux_prior,inference,sampler_numpyro,")
    w("         closure_legb}.py   --   numbers extracted, NOT hand-typed.")
    w("=" * 78)
    w("")
    w("GROUP 1 -- cosmology / IGM theta9   [Uniform in-box, flat]")
    w("  Sampled as Uniform(0,1)^9 in the unit cube + numpyro auto-bijector (exactly")
    w("  flat in-box, finite grads). PARAM_LIMITS = PRIYA design box (data.py).")
    w(f"  {'param':<12}{'form':<10}{'lo':>12}{'hi':>14}")
    w("  " + "-" * 48)
    for name, l, h in zip(PARAM_NAMES, lo, hi):
        w(f"  {name:<12}{'Uniform':<10}{l:>12.4g}{h:>14.4g}")
    w(f"  (n_s box [0.8,1.05] brackets Planck {NS_PLANCK} and eBOSS {NS_EBOSS}.)")
    w("  dim = 9")
    w("")
    w("GROUP 2 -- tau0 mean-flux ladder  [per-z Normal on alpha = tau0/Kim(z)]")
    w("  meanflux_tau0_prior(center='becker13'): mu_z = Becker+2013 Eq.6 tau_eff(z),")
    w(f"    tau_eff(z) = {BECKER13_TAU0}*((1+z)/(1+{BECKER13_ZREF}))^{BECKER13_BETA} + ({BECKER13_C})")
    w(f"  sigma_z = frac_sigma * mu_z, frac_sigma = {DEFAULT_FRAC_SIGMA} (5% measurement width).")
    w(f"  Kim(z) ladder anchor: KIM_AMP={KIM_AMP}, KIM_SLOPE={KIM_SLOPE}  (alpha=1 at Kim).")
    w("  The Normal is placed on the ladder coord alpha=tau0/Kim(z): mu_a=mu_z/Kim,")
    w("  sigma_a=sigma_z/Kim; tau0=alpha*Kim(z) deterministic. Per-z over the GLOBAL z grid.")
    w(f"  {'z':>6}{'mu tau0':>12}{'sig tau0':>12}{'mu alpha':>12}{'sig alpha':>12}")
    w("  " + "-" * 54)
    for zz in (2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6):
        m = float(becker13_tau0(jnp.asarray(zz)))
        s = DEFAULT_FRAC_SIGMA * m
        kk = KIM_AMP * (1.0 + zz) ** KIM_SLOPE
        w(f"  {zz:>6.1f}{m:>12.4f}{s:>12.4f}{m/kk:>12.4f}{s/kk:>12.4f}")
    w("  dim = n_z = 13 (GLOBAL union: DESI z=2.2..4.2 [11] u KS z=2.4..4.6 [12]; verified)")
    w("")
    w("GROUP 3 -- HCD incidence alpha   [per-class Normal; DLA softplus(Normal) one-sided]")
    w("  hcd_incidence_prior(w_c, z=3): center = (lit/sim)(z)*w_c  [LLS,subDLA];")
    w("    DLA center = HCD_DLA_RESIDUAL_FRAC*(lit/sim)*w_DLA  (residual post ~70% masking).")
    w(f"  cache structural w_c (median, [LLS,subDLA,DLA]) = "
      f"[{w_c_med[0]:.4f}, {w_c_med[1]:.5f}, {w_c_med[2]:.5f}]")
    w(f"  HCD_LIT_OVER_SIM @z3 = {HCD_LIT_OVER_SIM}  (data/sim dN/dX ratio at pivot)")
    w(f"  HCD_DLA_RESIDUAL_FRAC = {HCD_DLA_RESIDUAL_FRAC}; "
      f"HCD_PRIOR_FRAC_SIGMA (sig/mu) = {HCD_PRIOR_FRAC_SIGMA}")
    w(f"  HCD_DLA_Z_RELIABLE = {HCD_DLA_Z_RELIABLE} (widen sig_DLA above this z).")
    w(f"  {'class':<10}{'form':<18}{'mu (alpha)':>14}{'sigma':>14}{'sig/mu':>10}")
    w("  " + "-" * 66)
    forms = ["Normal", "Normal", "softplus(Normal)"]
    for i, c in enumerate(["LLS", "subDLA", "DLA"]):
        w(f"  {c:<10}{forms[i]:<18}{a_mu[i]:>14.5f}{a_sig[i]:>14.5f}"
          f"{HCD_PRIOR_FRAC_SIGMA[i]:>10.2f}")
    w("  Per-leg amplitude anchor (real-fit literature widths): KS sigma=0.27, DESI sigma=0.12.")
    w("  alpha is z-RESOLVED at sample time: alpha_c(z)=alpha_pivot*((1+z)/(1+3))^s_c.")
    w("  dim = 3  (alpha_lls, alpha_subdla, alpha_dla_raw)")
    w("")
    w("GROUP 4 -- HCD z-slope nuisance s_c   [Normal; only when M3-marginalized]")
    w("  closure_legb._zslope_sites: SAMPLED only if ctx.marginalize_zslope (real-fit).")
    w("  center = HCD_LIT_OVER_SIM_SLOPE (the forward's own fixed slope; isolates the")
    w("    marginalization COST, not a center shift).")
    w(f"  HCD_LIT_OVER_SIM_SLOPE (center) = {HCD_LIT_OVER_SIM_SLOPE}")
    w(f"  ZSLOPE_PRIOR_SIGMA (lit WLS 1sigma) = {tuple(ZSLOPE_PRIOR_SIGMA)}")
    w(f"  z-edge inflation ~x{ZSLOPE_EDGE_INFLATE} -> edge-inflated sigma "
      f"= {tuple(float(x) for x in np.round(s_sig_edge,3))}")
    w(f"  {'site':<12}{'form':<10}{'center':>10}{'sig_lit':>10}{'sig_edge':>12}")
    w("  " + "-" * 54)
    for i, nm in enumerate(["s_lls", "s_subdla", "s_dla"]):
        w(f"  {nm:<12}{'Normal':<10}{s_mu[i]:>10.2f}{s_sig_lit[i]:>10.2f}{s_sig_edge[i]:>12.3f}")
    w("  dim = 3  (s_lls, s_subdla, s_dla)  -- the M3 marginalized config")
    w("")
    w("GROUP 5 -- metals (SiIII/SiII) + resolution (b_res)")
    w("  In the Leg-B closure model (_legb_model) these are NOT sampled: a_SiIII=a_SiII=")
    w("  b_res=0.0 (fixed defaults). data_likelihood leg defaults: DESI metals_on=True /")
    w("  resolution_on=False; KS metals_on=False / resolution_on=False (the conservative")
    w("  KS leg already subtracts metals+resolution & inflates its cov).")
    w("  Since the closure sim-truth carries no metals and the sites are not sampled, the")
    w("  metal/resolution model factor is identity in the closure. dim (sampled) = 0.")
    w("")
    w("=" * 78)
    w("TOTAL SAMPLED DIMENSION")
    w("  theta9 (9) + tau0-ladder (n_z=13) + HCD alpha (3)        = 25-dim dense-mass block")
    w("  + HCD z-slope s_c (3)  [when M3-marginalized, real-fit]  = 28-dim")
    w("  metals/resolution: 0 sampled (fixed OFF in the closure).")
    w("=" * 78)

    txt = "\n".join(lines)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print(f"[stepA-priors] wrote {OUT_TXT}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
