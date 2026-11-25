---
title: "hcd_priya: A High-Column-Density Absorber Emulator for Lyman-α Forest Cosmology"
tags: ["Python","JAX","cosmology","Lyman-alpha forest","emulator","absorbers","DLA","LLS"]
authors: [{"name":"Ming-Feng Ho","orcid":"0000-0002-4457-890X","affiliation":"1,2"},{"name":"Shreyanshi Garg","affiliation":1},{"name":"Camille Avestruz","orcid":"0000-0001-8868-0810","affiliation":"1,2"}]
affiliations: [{"name":"Department of Physics, University of Michigan, USA","index":1},{"name":"Leinweber Institute for Theoretical Physics, University of Michigan, USA","index":2}]
date: 25 November 2026
bibliography: paper.bib
---

# Summary

The one-dimensional flux power spectrum (P1D) of the Lyman-$\alpha$ forest probes small-scale structure along quasar lines of sight, with sensitivity to the scale dependence of primordial fluctuations, alternatives to cold dark matter, and neutrino masses.
Nevertheless, the effects of high-column-density absorbers (HCDs), particularly Lyman-limit systems (LLSs), can be strongly degenerate with cosmological signal on the P1D of high-resolution spectra [@Ho2026], and neglecting the effect of LLSs could shift the cosmology constraint [@ChavesMontero2026].
The diffuse intergalactic medium (IGM) and HCDs arise from the same neutral-hydrogen distribution, but are often treated separately.
We introduce hcd_priya, a neural-network emulator built on PRIYA galaxy-formation simulations [@Bird2023PRIYA], that jointly predicts HCD-class-dependent P1D and HCD population statistics.

We identify absorbers in simulated sightlines, classify the spectra into clean, LLS, sub-damped Lyman-$\alpha$ absorber (subDLA), and damped Lyman-$\alpha$ absorber (DLA), and measure the P1D of each HCD class.
Given the cosmological and astrophysical parameters, the hcd_priya emulator outputs the P1D of each absorber class, its ratio to the clean forest, and the absorber incidence and column-density distribution function (CDDF) (\autoref{fig:schematic}).
The emulator can be used to construct HCD templates for other Lyman-$\alpha$ emulators or incorporated into cosmological likelihood analyses using Cobaya [@Torrado2021]; the JAX forward model is differentiable, for gradient-based samplers such as NumPyro [@Phan2019NumPyro].
The code and the trained emulator can be installed via `pip install hcd-priya`.

![**Software architecture of hcd_priya.** PRIYA mock spectra, classified by absorber class, are used to train a shared encoder with three heads. The predictor combines an ensemble of models with multi-fidelity corrections. External software: MP-Gadget [@Feng2018MPGadget] and fake_spectra [@Bird2017FSFE;@Bird2015].\label{fig:schematic}](figures/fig1_schematic.png){width="100%"}

# Statement of need

Cosmological likelihood analyses of the Lyman-$\alpha$ forest usually account for the contamination from HCDs using the templates of @Rogers2018, which are calibrated on a single Illustris simulation at one cosmology.
The @Rogers2018 templates are useful because in practice the simulations designed for the diffuse IGM often do not fully simulate galaxies and their surrounding dense gas associated with HCDs.
Simulating a sufficiently large IGM volume while achieving the resolution required for the small-scale forest and galaxy formation remains computationally challenging.
However, a template calibrated at one cosmology does not directly predict how the HCD contribution changes with cosmological and astrophysical parameters (\autoref{fig:ratio}), which is often required in a cosmological likelihood analysis.

PRIYA addresses this challenge by combining variations in cosmology and astrophysics with a full galaxy-formation model [@Bird2023PRIYA].
PRIYA's high-resolution simulations reach a mass resolution comparable to TNG100 [@Nelson2019], which is unusual for a simulation suite designed for Lyman-$\alpha$ forest cosmology.
By better resolving the underlying neutral-hydrogen distribution beyond the IGM, the simulations naturally capture the relative abundances of different absorber classes and their correlations with the surrounding forest.
hcd_priya measures the class-dependent P1D and absorber incidence across the simulation suite and emulates their dependence on cosmological and astrophysical parameters.
We use the predicted $dN/dX$ incidence to define the default weights of each absorber class, bridging the definition of HCD contamination to the physical definition of the absorber population, rather than treating them entirely as nuisance parameters.

![**HCD contributions to the P1D in PRIYA and @Rogers2018.** The ratio $P_c/P_{\rm{}clean}$ at $z=3.0$, measured with the unfiltered spectra, for the absorber classes defined in @Rogers2018. Teal: median of 60 PRIYA simulations; the dark band spans the simulations, while the light band spans all 20 mean-flux values. Red: predictions from @Rogers2018, with amplitudes ranging from 0.5 to 2. Grey: scales beyond the largest mode of the Rogers simulation.\label{fig:ratio}](figures/fig2_ratio.png){width="100%"}

hcd_priya is designed for P1D analyses, including DESI DR1 [@Karacayli2025;@Ravoux2025] and high-resolution spectra [@Ho2026].
The emulator also provides an absorber-cosmology connection baseline for studies of LLS, subDLA, and DLA.

# State of the field

Several tools on the market already model the Lyman-$\alpha$ forest P1D: the PRIYA emulators [@Bird2023PRIYA;@Fernandez2024;@Ho2026], LaCE [@CabayolGarcia2023;@Pedersen2021], and the Nyx-based Lyssa emulator [@Walther2025].
cup1d provides a cosmological likelihood framework [@ChavesMontero2026]; effective field theory offers a complementary analytic approach on quasi-linear scales [@Ivanov2024].
Nevertheless, existing P1D emulators do not jointly predict the HCD population and its cosmology-dependent contribution to the forest; HCDs are often treated as either contamination or a data-level correction.

We developed hcd_priya as a standalone package. The advantage is that hcd_priya can directly serve as an additional HCD template on top of existing P1D likelihoods.
Also, extending the @Ho2026 Gaussian-process (GP) emulator to multiple HCD classes and population statistics would be computationally expensive.
We instead train a shared multi-head neural network in JAX and Equinox [@Bradbury2018JAX;@Kidger2021Equinox], as suggested by the recent developments in neural network multi-fidelity emulation [@Yang2026].

# Software design

We classify each mock sightline by its highest-column-density absorber, keeping the full flux field and correlations between systems.
With $m=\max_i\log_{10}(N_{{\rm{}HI},i}/{\rm{}cm^{-2}})$, where $N_{{\rm{}HI},i}$ is the column density of neutral hydrogen absorber $i$, the class $c$ is

$$c=\begin{cases}{\rm{}DLA},&m\ge20.3,\\{\rm{}subDLA},&19.0\le{}m<20.3,\\{\rm{}LLS},&17.2\le{}m<19.0,\\{\rm{}clean},&{\rm{}otherwise}.\end{cases}$$

For each mean flux $\bar{F}$, we define a common optical-depth scaling factor $s$ from the PRIYA DLA-filtered spectra [@Bird2023PRIYA;@Fernandez2024], satisfying

$$\left\langle{}e^{-s\tau_{\rm{}filtered}}\right\rangle=\bar{F},$$

where $\tau_{\rm{}filtered}$ is the optical depth after the DLA filter and $\langle\cdot\rangle$ averages over all pixels.

We apply the same mean-flux scaling factor to the filtered spectra of all absorber classes, and measure their P1D relative to the target mean flux using fake_spectra [@Bird2017FSFE;@Bird2015]:

$$\delta_F(v)=\frac{e^{-s\tau_{\rm{}filtered}(v)}}{\bar{F}}-1,$$

where $\delta_F(v)$ is the flux contrast at velocity $v$. The class P1D, $P_c(k)$, is the two-point statistic of $\delta_F$ at wavenumber $k$, as defined in fake_spectra.

The filter cuts on sightlines whose peak optical depth goes beyond $10^{6}$, and replaces the absorbed pixels around that peak with the mean flux.

The low-fidelity emulator maps the nine parameters and redshift to a shared latent representation $\Theta$.
Three heads predict the absorber statistics, the P1D baseline, and its parameter-dependent residual.
For each absorber class $c$, we use a structured mean [@KennedyOHagan2001]:

$$\ln{}P_c(k)=b_c(k;z,\tau_{\rm{}eff})+\sigma_c(k)\,r_c(k;\Theta,\tau_{\rm{}eff}),$$

where $z$ is the redshift, $\tau_{\rm{}eff}=-\ln\bar{F}$ the effective optical depth, $b_c$ the baseline, $r_c$ the residual, and $\sigma_c(k)$ the scatter of $\ln{}P_c$ across the training simulations.
The P1D heads predict coefficients in separate, learned low-rank bases over wavenumbers, initialized using singular value decomposition of the training spectra.
We average predictions from five independently trained networks as an ensemble.
Using five high-resolution runs, we apply a multi-fidelity correction to correct the resolution convergence, similar to @Ho2026:

$$P_c^{\rm{}MF}(k)=P_c^{\rm{}LF}(k)\,e^{g_c},$$

where LF and MF mark the low-fidelity and multi-fidelity outputs, and $g_c$ is a multi-fidelity correction function defined in the code.

The predicted absorber incidence, $dN/dX$, determines the default weights of each sightline class.
For class $c$, the expected number of absorbers is

$$\mu_c=\left(\frac{dN}{dX}\right)_c\Delta{}X,$$

where $(dN/dX)_c$ is the number of absorbers (class $c$) per unit distance $X$, and $\Delta{}X$ the absorption distance of a sightline.

Assuming Poisson counts, the classification rule gives

$$\begin{aligned}w_{\rm{}DLA}&=1-e^{-\mu_{\rm{}DLA}},\\w_{\rm{}subDLA}&=\left(1-e^{-\mu_{\rm{}subDLA}}\right)e^{-\mu_{\rm{}DLA}},\\w_{\rm{}LLS}&=\left(1-e^{-\mu_{\rm{}LLS}}\right)e^{-\mu_{\rm{}subDLA}-\mu_{\rm{}DLA}},\\w_{\rm{}clean}&=e^{-\sum_c\mu_c},\end{aligned}$$

where $w_c$ is the fraction of sightlines in class $c$; the sum in $w_{\rm{}clean}$ is over the three HCD classes.

A separate fitted redshift correction accounts for departures from the Poisson approximation.
The weights reconstruct the total P1D through $P_{\rm{}total}=\sum_cw_cP_c$, and users can vary the HCD amplitudes in cosmological likelihood analyses.
The predictor flags extrapolation beyond the training range.

# Research impact statement

hcd_priya is a continuation of our work in @Ho2026, where we found a strong degeneracy between LLS abundance and cosmological parameters inferred from the small-scale Lyman-$\alpha$ forest.
The HCD catalogues output from hcd_priya are currently being analyzed by S. Garg et al. (in prep.) to investigate how different physical prescriptions and cosmological models influence the distribution of neutral hydrogen, and to assess the impact of HCDs as contaminants in the Lyman-$\alpha$ forest.
We also plan to revisit high-resolution P1D constraints using this simulation-informed HCD prior constructed from hcd_priya.
As a public package, hcd_priya provides an HCD model that can be incorporated into existing cosmological analyses.

The mock-recovery example in the repository uses the noiseless P1D from a held-out PRIYA high-resolution simulation, computed at 121 KODIAQ-SQUAD bins and covariance [@Karacayli2022;@Murphy2019;@OMeara2015;@OMeara2017].
\autoref{fig:mock} compares the hcd_priya reference posterior with that of the GP emulator of @Ho2026 on the same mock.
Both chains are statistically consistent: the posterior means differ by at most 0.27 posterior standard deviations, with posterior width ratios between 1.00 and 1.09.
Both fits use KODIAQ-SQUAD data covariance alone and keep the HCD amplitudes fixed.
We chose to leave HCD marginalization out of the likelihood because the cosmological analysis using hcd_priya belongs to a separate analysis paper.
The mock, the model trained without that simulation, and the stored reference are distributed with the software release.

![**Cosmological parameter recovery from a held-out PRIYA simulation.** (a, b) Posterior constraints from hcd_priya (filled 68% and 95% regions) and the PRIYA GP emulator (red contours; the GP was trained on this simulation). Dashed lines indicate the input parameters. (c) The mock P1D and its posterior median prediction, with residuals in units of the observational uncertainty. The GP chain is not publicly available.\label{fig:mock}](figures/fig3_mock.png){width="100%"}

On the absorber population side, we compare the predicted CDDF with published measurements of LLSs, subDLAs, and DLAs (\autoref{fig:cddf}).
The CDDFs are evaluated at the midpoint of the PRIYA parameter prior ranges, without fitting to observational data points.

![**Predicted HCD column-density distribution compared with observations.** The emulated CDDF, $f(N_{\rm{}HI})$, at three redshifts is compared with observations: DLAs from @HoBirdGarnett2021, subDLAs from @Zafar2013 and @OMeara2007, and LLSs from @Prochaska2010. Bands span the 60 design points of parameter variations. Dashed lines indicate the absorber class boundaries; the lower panels show the ratio of observations to the model.\label{fig:cddf}](figures/fig6_cddf.png){width="100%"}

# AI usage disclosure

The authors thank Claude (Fable 5 and 5.2, Opus 4.7, 4.8, and 5.5, and Sonnet 4.6) through Claude Code, GPT (GPT 5.6 Sol and GPT 6 Astra), and GitHub Copilot for their contributions throughout the development of hcd_priya. AI agents contributed substantially to implementation, testing, documentation, and code review. The authors defined the scientific questions, emulator architecture, validation metrics, and inference workflow.

AI also assisted in preparing the manuscript, including expanding the initial human-written draft, checking references and numerical results, and developing the plotting code.

AI-assisted contributions were validated through automated tests and comparisons with analytical expectations, existing emulators, and existing scientific software. The authors evaluated alternative approaches, made the final scientific decisions, and take responsibility for the software and manuscript.

# Acknowledgements

MFH and CA are supported by the Leinweber Foundation and DOE grant DE-SC0019193.
This research was supported in part through computational resources and services provided by Advanced Research Computing (ARC), a division of Information and Technology Services (ITS) at the University of Michigan, Ann Arbor.

# References
