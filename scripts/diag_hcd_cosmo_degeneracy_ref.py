"""Cosmology/P1D referee: QUANTIFY the A_HCD <-> n_s / A_p Fisher degeneracy on the
joint DESI+KS leg via emulator finite-diff/jax.grad responses.

Builds the production Leg-B ctx (hierarchical_hcd, hcd_2d_tilt), assembles the WHITENED
C_data-weighted Jacobians J_Ap, J_ns, J_AHCD, J_LLS, J_sub, J_B, J_rsub, J_tau0, J_dtau0
on the flat (z,k) data vector, and computes:
  - the C_data-weighted PAIRWISE cosine between the per-CLASS template (pure LLS, pure
    subDLA) and the forest A_p / n_s responses  -> the HEADLINE +0.81 / -0.93 numbers;
  - the fraction of the LLS template explained by the (A_p,n_s) subspace (DESI + KS legs)
    and by the full-nuisance subspace (A_p, n_s, tau0, dtau0);
  - the n_s<->A_HCD Fisher correlation (5x5 Fisher on [A_p, n_s, A_HCD, B_HCD, r_sub]);
  - the prior-center -> n_s GLS response (the 0.5sigma systematic);
  - the data leverage sqrt(F_ii) of A_HCD vs the orthogonal z-tilt B_HCD;
  - the z-discrimination: per-z whitened profiles + z-leverage.

THE HEADLINE / FIGURE NUMBERS (note s4 + panel wf_4e8d25e2):
  cos(LLS, A_p)  = +0.81   <- the PAIRWISE whitened cosine of the PURE-LLS class template
  cos(subDLA, ns)= -0.93   <- the PAIRWISE whitened cosine of the PURE-subDLA class template
  LLS in-plane(A_p,n_s) = 91% on DESI (97% with the tau0 nuisances), 64% on KS.
The pure-CLASS direction is the key: the OLD script differentiated the *mixed* A_HCD
(scaling all three classes by their fixed ratios at once), which gives a flatter 0.62/0.69
cosine.  The headline is the per-class LLS template -- the object the note calls
"the LLS template" -- recovered here.

Also writes 4 inline figures (matching the note):
  degeneracy_lls_inplane_projection.png   (panel 1: cos + in-plane fractions)
  degeneracy_snr_amplitude_vs_ztilt.png   (panel 2: sqrt(F_ii) A_HCD vs B_HCD)
  degeneracy_fisher_corr_matrix.png       (panel 3: posterior corr; corr(A_HCD,n_s))
  degeneracy_prior_center_law.png         (panel 4: prior-center -> n_s linear law)
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import hcd_analysis.emulator  # x64
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.predict import predict_P_obs, predict_excess
from hcd_analysis.emulator.data import PARAM_LIMITS, SAMPLING_LIMITS
from hcd_analysis.emulator.meanflux_prior import (
    tau0_alpha_priya, TAU0_PIVOT_Z, TAU0_AMP_RANGE, DTAU0_RANGE)

np.set_printoptions(precision=4, suppress=True, linewidth=160)

FIGDIR = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator"
NPZ = "/home/mfho/hcd_priya_notes/diag_hcd_cosmo_degeneracy.npz"

# ---------------------------------------------------------------------------
# Build the joint DESI+KS leg context (2D amplitude x tilt config, fold6 anchor).
# ---------------------------------------------------------------------------
ctx, dla_core_global = CL.build_legb_ctx(
    hierarchical_hcd=True, hcd_2d_tilt=True, use_xclass=True)

legs = ctx.legs
print("legs:", [(lg.name, lg.n_z, int(lg.k.shape[0])) for lg in legs])
print("z_global:", np.round(ctx.z_global, 3))
print("alpha_hcd_mu (LLS,sub,DLA):", np.asarray(ctx.alpha_hcd_mu))
print("alpha_hcd_sigma:", np.asarray(ctx.alpha_hcd_sigma))
print("hcd_ratio_mu (r_sub,r_dla):", np.asarray(ctx.hcd_ratio_mu))
print("hcd_ratio_sigma:", np.asarray(ctx.hcd_ratio_sigma))
print("hcd_btilt_mu:", ctx.hcd_btilt_mu, "hcd_btilt_sigma:", ctx.hcd_btilt_sigma)
print("hcd_dslope (delta s_c):", np.asarray(ctx.hcd_dslope))

# ---------------------------------------------------------------------------
# Fiducial parameters.
#   theta_unit fiducial = 0.5 in each (center of the SAMPLING box).
#   tau0 at the Kim center (amp=1, dtau0=0) => alpha=1 => tau0_vec = Kim07.
#   A_HCD = alpha_hcd_mu[0] (LLS prior center); ratios at their centers; B_HCD = btilt_mu.
# ---------------------------------------------------------------------------
lo_u = np.asarray(CL._THETA_UNIT_LO); hi_u = np.asarray(CL._THETA_UNIT_HI)
theta0 = 0.5 * (lo_u + hi_u)        # (9,) unit-cube center of the sampling box
# index mapping: PARAM_NAMES = (ns, Ap, ...). data.PARAM_LIMITS row 0 = ns, row 1 = Ap.
I_NS, I_AP = 0, 1

A_HCD0 = float(ctx.alpha_hcd_mu[0])
r_sub0, r_dla0 = float(ctx.hcd_ratio_mu[0]), float(ctx.hcd_ratio_mu[1])
B_HCD0 = float(ctx.hcd_btilt_mu)
dslope = np.asarray(ctx.hcd_dslope)             # (3,) delta s_c
zp = CL.HCD_Z_PIVOT
TAU0_AMP0, DTAU00 = 1.0, 0.0                     # Kim center for the mean-flux nuisances

# unit->physical for ns, Ap (PARAM_LIMITS box; the emulator normalizes on PARAM_LIMITS)
def unit_to_phys(theta_u):
    return PARAM_LIMITS[:, 0] + theta_u * (PARAM_LIMITS[:, 1] - PARAM_LIMITS[:, 0])

print("\nfiducial theta_unit:", np.round(theta0, 4))
print("fiducial ns,Ap (phys):", unit_to_phys(theta0)[[I_NS, I_AP]])
print("A_HCD0:", A_HCD0, " B_HCD0:", B_HCD0)

# ---------------------------------------------------------------------------
# Forward: flat P_obs over the joint (z,k) data vector, and the per-leg row map.
# We mirror _data_loglik_legcore's per-leg, per-z assembly but WITHOUT the mock
# noise -- we want the model P_obs(theta) and the C_data (cosmic) covariance.
#
# Per-CLASS HCD incidence: instead of the OLD single A_HCD that scales (LLS, sub, DLA)
# together by their fixed ratios, we carry the THREE class pivots (aL, aS, aD)
# INDEPENDENTLY, so we can take the PURE-LLS and PURE-subDLA template derivatives --
# the objects the note calls "the LLS / subDLA template".  The mixed-A_HCD direction
# (A_HCD scales all three) is reconstructed for the Fisher/prior-pin block.
# tau0 enters via the PRIYA (amp, dtau0) mean-flux curve -> two nuisance directions.
# ---------------------------------------------------------------------------
kim = CL._kim(jnp.asarray(ctx.z_global))

def alpha_hcd_z_classes(aL, aS, aD, B_hcd):
    """(n_zglobal, 3) per-z HCD incidence alpha_c(z) from INDEPENDENT class pivots."""
    alpha_pivot = jnp.stack([aL, aS, aD])                              # (3,)
    s_c = B_hcd + jnp.asarray(dslope)                                  # (3,)
    zg = jnp.asarray(ctx.z_global)
    shape_zg = ((1.0 + zg)[:, None] / (1.0 + zp)) ** s_c               # (n_zg,3)
    return alpha_pivot[None, :] * shape_zg                             # (n_zg,3)

# Build the flat row layout once (z-major per leg).
zg_np = np.asarray(ctx.z_global)
row_specs = []        # list of (leg_name, leg_obj, iz, z_unit, gz_idx, rows-in-leg)
for leg in legs:
    z_idx = np.asarray(leg.z_idx)
    sel_g = np.array([int(np.argmin(np.abs(zg_np - zz))) for zz in leg.z])
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        row_specs.append((leg.name, leg, iz, float(leg.z_unit[iz]), int(sel_g[iz]), rows))

# z-mean DLA core per leg (matches _data_loglik_legcore's documented MVP).
core_zmean = {nm: jnp.asarray(np.mean(np.asarray(v), axis=0))
              for nm, v in ctx.dla_core_leg.items()}

def forward_flat(theta9, aL, aS, aD, B_hcd, tau0_amp, dtau0):
    """Flat P_obs over ALL legs with INDEPENDENT per-class HCD pivots + (amp,dtau0)."""
    alpha_z = alpha_hcd_z_classes(aL, aS, aD, B_hcd)        # (n_zg,3)
    a_scale = tau0_alpha_priya(jnp.asarray(ctx.z_global), tau0_amp, dtau0,
                               z_pivot=TAU0_PIVOT_Z)         # (n_zg,)
    tau0_vec = jnp.asarray(kim) * a_scale                    # (n_zg,)
    out_blocks = []
    for (lname, leg, iz, z_unit, gz, rows) in row_specs:
        core = jnp.asarray(core_zmean[lname])               # (K,) z-mean core per leg
        a_z = alpha_z[gz]                                   # (3,)
        # per-leg DLA-forward scaling (KS dla_forward_frac=0)
        dff = float(getattr(leg, "dla_forward_frac", 1.0))
        if dff != 1.0:
            a_z = a_z * jnp.array([1.0, 1.0, dff])
        tau0 = tau0_vec[gz]
        P_cache = predict_P_obs(ctx.model, theta9, z_unit, tau0, a_z,
                                ctx.pf_stats, core)          # (Kc,)
        k_sub = jnp.asarray(leg.k[rows])
        P_z = jnp.interp(k_sub, ctx.cache_k, P_cache)
        out_blocks.append(P_z)
    return jnp.concatenate(out_blocks)

# Build the matching flat C_data (block) by concatenating the per-row C_data in the
# SAME order. DESI and KS are independent surveys -> block-diag across legs.
def build_flat_index():
    out = {}
    for leg in legs:
        z_idx = np.asarray(leg.z_idx)
        order = []
        for iz in range(leg.n_z):
            rows = np.where(z_idx == iz)[0]
            order.extend(list(rows))
        out[leg.name] = np.asarray(order)
    return out

flat_idx = build_flat_index()
blocks = []
leg_order = []
for leg in legs:
    idx = flat_idx[leg.name]
    blocks.append(np.asarray(leg.C_data)[np.ix_(idx, idx)])
    leg_order.append((leg.name, idx.size))
N = sum(b.shape[0] for b in blocks)
C_data = np.zeros((N, N))
o = 0
for b in blocks:
    n = b.shape[0]
    C_data[o:o+n, o:o+n] = b
    o += n
print("\nflat N (joint DESI+KS rows):", N, " leg blocks:", leg_order)

# row-level (z,k,leg) metadata in the flat order
flat_z = []; flat_k = []; flat_leg = []
for leg in legs:
    idx = flat_idx[leg.name]
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)][idx]
    flat_z.extend(list(z_row)); flat_k.extend(list(np.asarray(leg.k)[idx]))
    flat_leg.extend([leg.name]*idx.size)
flat_z = np.asarray(flat_z); flat_k = np.asarray(flat_k); flat_leg = np.asarray(flat_leg)
mD = flat_leg == "DESI"; mK = flat_leg == "KS"

# ---------------------------------------------------------------------------
# Whitening: L = chol(C_data); whitened residual = L^{-1} r. The Fisher metric is
#   <a,b>_F = a^T C_data^{-1} b = (L^{-1}a) . (L^{-1}b).
# ---------------------------------------------------------------------------
jit_C = 1e-12 * np.mean(np.diag(C_data))
L = np.linalg.cholesky(C_data + jit_C * np.eye(N))
def whiten(v):
    return jax.scipy.linalg.solve_triangular(jnp.asarray(L), jnp.asarray(v), lower=True)

# ---------------------------------------------------------------------------
# Jacobians via jax.jacfwd at the fiducial.
# ---------------------------------------------------------------------------
theta0_j = jnp.asarray(theta0)

# forest cosmology Jacobians (full forward, mixed/centred HCD held fixed)
def f_theta(theta9):
    return forward_flat(theta9, A_HCD0, A_HCD0*r_sub0, A_HCD0*r_dla0, B_HCD0,
                        TAU0_AMP0, DTAU00)

J_theta = jax.jacfwd(f_theta)(theta0_j)        # (N, 9) d P_obs / d theta_unit
J_ns_unit = J_theta[:, I_NS]
J_Ap_unit = J_theta[:, I_AP]
dns_param = (PARAM_LIMITS[I_NS, 1] - PARAM_LIMITS[I_NS, 0])
dAp_param = (PARAM_LIMITS[I_AP, 1] - PARAM_LIMITS[I_AP, 0])
J_ns_phys = J_ns_unit / dns_param
J_Ap_phys = J_Ap_unit / dAp_param

# PURE-CLASS templates: derivative wrt one class pivot, the others held fixed.
def f_LLS(a):  # pure LLS class incidence
    return forward_flat(theta0_j, a, A_HCD0*r_sub0, A_HCD0*r_dla0, B_HCD0, TAU0_AMP0, DTAU00)
def f_sub(a):  # pure subDLA class incidence
    return forward_flat(theta0_j, A_HCD0, a, A_HCD0*r_dla0, B_HCD0, TAU0_AMP0, DTAU00)
def f_dla(a):  # pure DLA class incidence
    return forward_flat(theta0_j, A_HCD0, A_HCD0*r_sub0, a, B_HCD0, TAU0_AMP0, DTAU00)
J_LLS = jax.jacfwd(f_LLS)(jnp.asarray(A_HCD0))
J_sub = jax.jacfwd(f_sub)(jnp.asarray(A_HCD0*r_sub0))
J_dla = jax.jacfwd(f_dla)(jnp.asarray(A_HCD0*r_dla0))

# MIXED A_HCD (the prior-pinned amplitude that scales all 3 classes together) + B + r.
def f_Ahcd(A):
    return forward_flat(theta0_j, A, A*r_sub0, A*r_dla0, B_HCD0, TAU0_AMP0, DTAU00)
def f_B(B):
    return forward_flat(theta0_j, A_HCD0, A_HCD0*r_sub0, A_HCD0*r_dla0, B, TAU0_AMP0, DTAU00)
def f_rsub(r):
    return forward_flat(theta0_j, A_HCD0, A_HCD0*r, A_HCD0*r_dla0, B_HCD0, TAU0_AMP0, DTAU00)
J_AHCD = jax.jacfwd(f_Ahcd)(jnp.asarray(A_HCD0))
J_B = jax.jacfwd(f_B)(jnp.asarray(B_HCD0))
J_rsub = jax.jacfwd(f_rsub)(jnp.asarray(r_sub0))

# tau0 nuisance directions (PRIYA amp + slope).
def f_t0(a):
    return forward_flat(theta0_j, A_HCD0, A_HCD0*r_sub0, A_HCD0*r_dla0, B_HCD0, a, DTAU00)
def f_dt0(d):
    return forward_flat(theta0_j, A_HCD0, A_HCD0*r_sub0, A_HCD0*r_dla0, B_HCD0, TAU0_AMP0, d)
J_t0 = jax.jacfwd(f_t0)(jnp.asarray(TAU0_AMP0))
J_dt0 = jax.jacfwd(f_dt0)(jnp.asarray(DTAU00))

# whiten everything
def wf(v):
    return np.asarray(whiten(v))
wJ_ns = wf(J_ns_phys); wJ_Ap = wf(J_Ap_phys)
wJ_LLS = wf(J_LLS); wJ_sub = wf(J_sub); wJ_dla = wf(J_dla)
wJ_AHCD = wf(J_AHCD); wJ_B = wf(J_B); wJ_rsub = wf(J_rsub)
wJ_t0 = wf(J_t0); wJ_dt0 = wf(J_dt0)

def fisher_cos(a, b):
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)

def frac_in_span(target, basis_cols):
    Bm = np.column_stack(basis_cols)
    coef, *_ = np.linalg.lstsq(Bm, target, rcond=None)
    proj = Bm @ coef
    return float(proj @ proj) / float(target @ target), coef

# ===========================================================================
# (A) PAIRWISE DEGENERACY COSINES (the HEADLINE numbers).
#     The PURE-LLS and PURE-subDLA templates vs the forest A_p / n_s response,
#     in the C_data-whitened metric, on the JOINT DESI+KS leg (the panel leg)
#     and on each survey separately.
# ===========================================================================
print("\n========== (A) PAIRWISE TEMPLATE COSINES (C_data-whitened) ==========")
def report_cos(tag, t):
    print(f"  {tag:18s} JOINT cos(.,A_p)={fisher_cos(t,wJ_Ap):+.4f}  cos(.,n_s)={fisher_cos(t,wJ_ns):+.4f}"
          f"   DESI cos(.,A_p)={fisher_cos(t[mD],wJ_Ap[mD]):+.4f} cos(.,n_s)={fisher_cos(t[mD],wJ_ns[mD]):+.4f}")
report_cos("LLS template", wJ_LLS)
report_cos("subDLA template", wJ_sub)
report_cos("mixed A_HCD", wJ_AHCD)

cos_LLS_Ap = fisher_cos(wJ_LLS, wJ_Ap)            # HEADLINE +0.81 (joint leg)
cos_LLS_ns = fisher_cos(wJ_LLS, wJ_ns)
cos_sub_ns = fisher_cos(wJ_sub, wJ_ns)            # HEADLINE -0.93 (joint leg)
cos_sub_Ap = fisher_cos(wJ_sub, wJ_Ap)
cos_LLS_Ap_DESI = fisher_cos(wJ_LLS[mD], wJ_Ap[mD])
cos_LLS_ns_DESI = fisher_cos(wJ_LLS[mD], wJ_ns[mD])
cos_sub_ns_DESI = fisher_cos(wJ_sub[mD], wJ_ns[mD])
print(f"\n  >>> HEADLINE cos(LLS, A_p)  = {cos_LLS_Ap:+.4f}  (joint) / {cos_LLS_Ap_DESI:+.4f} (DESI)")
print(f"  >>> HEADLINE cos(subDLA,n_s)= {cos_sub_ns:+.4f}  (joint) / {cos_sub_ns_DESI:+.4f} (DESI)")

# ===========================================================================
# (B) IN-PLANE FRACTIONS of the LLS template (DESI 91%, full-nuisance 97%, KS 64%).
# ===========================================================================
print("\n========== (B) LLS TEMPLATE IN-PLANE FRACTIONS ==========")
frac_2d_DESI, _ = frac_in_span(wJ_LLS[mD], [wJ_Ap[mD], wJ_ns[mD]])
frac_2d_KS, _ = frac_in_span(wJ_LLS[mK], [wJ_Ap[mK], wJ_ns[mK]])
frac_full_DESI, _ = frac_in_span(
    wJ_LLS[mD], [wJ_Ap[mD], wJ_ns[mD], wJ_t0[mD], wJ_dt0[mD]])
frac_ns_DESI = fisher_cos(wJ_LLS[mD], wJ_ns[mD])**2
frac_Ap_DESI = fisher_cos(wJ_LLS[mD], wJ_Ap[mD])**2
# keep joint-leg frac (the npz historically reported frac_2d ~0.90 on joint)
frac_2d_joint, _ = frac_in_span(wJ_LLS, [wJ_Ap, wJ_ns])
print(f"  DESI in-plane (A_p,n_s)            = {frac_2d_DESI:.4f}  (HEADLINE 91%)")
print(f"  DESI full-nuisance (+tau0,dtau0)   = {frac_full_DESI:.4f}  (HEADLINE 97%)")
print(f"  KS   in-plane (A_p,n_s)            = {frac_2d_KS:.4f}  (HEADLINE 64%)")
print(f"  DESI n_s-alone={frac_ns_DESI:.4f}  A_p-alone={frac_Ap_DESI:.4f}  orthogonal={1-frac_2d_DESI:.4f}")
print(f"  JOINT in-plane (A_p,n_s)           = {frac_2d_joint:.4f}")

# ===========================================================================
# (C) FISHER MATRIX + n_s<->A_HCD CORRELATION (the corr-matrix figure).
#     5x5 on [A_p, n_s, A_HCD, B_HCD, r_sub] + HCD Gaussian priors. A_HCD here is
#     the PRIOR-PINNED amplitude (scales all classes) -- the object the prior acts on.
# ===========================================================================
print("\n========== (C) FISHER CORRELATION ==========")
cols = {"A_p": wJ_Ap, "n_s": wJ_ns, "A_HCD": wJ_AHCD, "B_HCD": wJ_B, "r_sub": wJ_rsub}
names = list(cols.keys())
W = np.column_stack([cols[n] for n in names])
F = W.T @ W
prior_prec = np.zeros(len(names))
sig_AHCD = float(ctx.alpha_hcd_sigma[0]); sig_B = float(ctx.hcd_btilt_sigma)
sig_rsub = float(ctx.hcd_ratio_sigma[0])
prior_prec[names.index("A_HCD")] = 1.0 / sig_AHCD**2
prior_prec[names.index("B_HCD")] = 1.0 / sig_B**2
prior_prec[names.index("r_sub")] = 1.0 / sig_rsub**2
print("HCD prior sigmas: A_HCD=%.4g  B_HCD=%.4g  r_sub=%.4g" % (sig_AHCD, sig_B, sig_rsub))
F_post = F + np.diag(prior_prec)
Cov_post = np.linalg.inv(F_post)
sd = np.sqrt(np.diag(Cov_post))
Corr = Cov_post / np.outer(sd, sd)
i_ns, i_A = names.index("n_s"), names.index("A_HCD")
print("posterior param order:", names)
print("posterior sigmas:", np.round(sd, 5))
for i, n in enumerate(names):
    print(f"  {n:6s}", " ".join(f"{Corr[i,j]:+.3f}" for j in range(len(names))))
corr_AHCD_ns = Corr[i_A, i_ns]
Cov_data = np.linalg.inv(F + 1e-6*np.trace(F)/len(names)*np.eye(len(names)))
sd0 = np.sqrt(np.diag(Cov_data)); Corr0 = Cov_data/np.outer(sd0, sd0)
corr_AHCD_ns_dataonly = Corr0[i_A, i_ns]
print(f"\n>>> corr(A_HCD, n_s) [data Fisher + HCD priors] = {corr_AHCD_ns:+.4f}")
print(f">>> corr(A_HCD, n_s) [DATA ONLY, no priors]     = {corr_AHCD_ns_dataonly:+.4f}")

# data leverage sqrt(F_ii) for the S/N figure
sqrtF = {n: float(np.sqrt(F[i, i])) for i, n in enumerate(names)}
print("\ndata leverage sqrt(F_ii):", {k: round(v, 2) for k, v in sqrtF.items()})

# ===========================================================================
# (D) PRIOR-CENTER -> n_s RESPONSE (the prior-center-law figure).
# ===========================================================================
print("\n========== (D) PRIOR-CENTER -> n_s RESPONSE ==========")
e_A = np.zeros(len(names)); e_A[i_A] = 1.0
dtheta_dmuA = Cov_post @ (prior_prec * e_A)
dns_dmuA = dtheta_dmuA[i_ns]
dns_per_sigA_in_signs = dns_dmuA * sig_AHCD / sd[i_ns]
frac_A_prior = prior_prec[i_A] * Cov_post[i_A, i_A]
print("d n_s_MAP / d mu_A (per unit A)         =", round(float(dns_dmuA), 4))
print("n_s shift for a +1 sigma_A center move  =", round(float(dns_dmuA*sig_AHCD), 6), "(phys n_s)")
print("  ... in units of posterior sigma_ns    =", round(float(dns_per_sigA_in_signs), 4), "sigma")
print("compare: corr(n_s,A_HCD)                =", round(float(corr_AHCD_ns), 4))
print("fraction of A_HCD posterior precision from its PRIOR =", round(float(frac_A_prior), 4))

# ===========================================================================
# (E) Z-DISCRIMINATION (kept for the npz; not plotted here).
# ===========================================================================
uz_flat = np.unique(np.round(flat_z, 3))
def perz_white_norm(wJ):
    return {z: float(np.linalg.norm(wJ[np.abs(np.round(flat_z,3)-z) < 1e-6])) for z in uz_flat}
nA = perz_white_norm(wJ_AHCD); nns = perz_white_norm(wJ_ns)
nAp = perz_white_norm(wJ_Ap); nB = perz_white_norm(wJ_B)
frac_B, _ = frac_in_span(wJ_B, [wJ_Ap, wJ_ns])
print("\nfraction of J_B (z-tilt dir) in span(J_Ap,J_ns) =", round(frac_B, 4))

# ===========================================================================
# FIGURES
# ===========================================================================
plt.rcParams.update({"font.size": 11})

# --- Figure 1: LLS template projected into the forest (A_p, n_s) plane ------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))
# left: a schematic of the whitened LLS vector in the (A_p, n_s) frame, with the
# pairwise cosines annotated.  Build an orthonormal 2D frame from (wJ_Ap, wJ_ns) on DESI
ea = wJ_Ap[mD] / np.linalg.norm(wJ_Ap[mD])
en = wJ_ns[mD] - (wJ_ns[mD] @ ea) * ea
en = en / np.linalg.norm(en)
v = wJ_LLS[mD]
cx = float(v @ ea) / np.linalg.norm(v)
cy = float(v @ en) / np.linalg.norm(v)
# draw forest axes (A_p along x; n_s shown at its true angle to A_p)
ax0.annotate("", xy=(1.0, 0.0), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2.5))
ax0.text(1.03, 0.0, r"$\hat J_{A_p}$", color="tab:orange", fontsize=13, va="center")
# n_s axis at angle = arccos(cos(Ap,ns))
ang_ns = np.arccos(np.clip(fisher_cos(wJ_Ap[mD], wJ_ns[mD]), -1, 1))
ax0.annotate("", xy=(np.cos(ang_ns), np.sin(ang_ns)), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.5))
ax0.text(np.cos(ang_ns)*1.07, np.sin(ang_ns)*1.07, r"$\hat J_{n_s}$",
         color="tab:blue", fontsize=13, va="center")
# LLS vector (unit length, in the orthonormal (ea,en) frame mapped back to drawn axes)
# drawn x = component along A_p direction; drawn position uses cx, cy in the (ea,en) basis
lx, ly = cx, cy
ax0.annotate("", xy=(lx, ly), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color="black", lw=3.0))
ax0.text(lx*0.55 + 0.05, ly*0.55 - 0.10, "LLS\ntemplate", color="black", fontsize=11,
         ha="left", va="top")
ax0.text(0.30, 0.78,
         f"|cos(LLS, $A_p$)| = {abs(cos_LLS_Ap_DESI):.2f}\n|cos(LLS, $n_s$)| = {abs(cos_LLS_ns_DESI):.2f}",
         fontsize=12, color="dimgray",
         bbox=dict(boxstyle="round", fc="white", ec="0.6"))
ax0.text(-0.55, -0.70,
         f"in-plane (forest) fraction\n"
         r"$\|\mathrm{proj}\|^2/\|J_\mathrm{LLS}\|^2$ = "
         f"{frac_2d_DESI*100:.0f}%\northogonal = {(1-frac_2d_DESI)*100:.0f}%",
         fontsize=11,
         bbox=dict(boxstyle="round", fc="#fff8e1", ec="0.6"))
ax0.set_xlim(-0.6, 1.25); ax0.set_ylim(-0.75, 1.05)
ax0.axhline(0, color="0.8", lw=0.8); ax0.axvline(0, color="0.8", lw=0.8)
ax0.set_aspect("equal")
ax0.set_xlabel("whitened forest-amplitude axis"); ax0.set_ylabel("whitened tilt axis")
ax0.set_title("PURE-LLS template inside the\n"
              r"forest $(A_p, n_s)$ plane ($C_\mathrm{data}$-whitened, DESI)")

# right: bar chart of how much of the LLS template the forest absorbs
labels = [r"$n_s$ alone", r"$A_p$ alone", r"span$(A_p,n_s)$",
          r"span$(A_p,n_s,\tau_0)$", "KS\n" + r"span$(A_p,n_s)$"]
vals = [frac_ns_DESI, frac_Ap_DESI, frac_2d_DESI, frac_full_DESI, frac_2d_KS]
colors = ["tab:blue", "tab:orange", "tab:purple", "tab:cyan", "tab:red"]
bars = ax1.bar(range(len(vals)), vals, color=colors, edgecolor="k")
for b, vv in zip(bars, vals):
    ax1.text(b.get_x()+b.get_width()/2, vv+0.015, f"{vv*100:.0f}%",
             ha="center", fontweight="bold")
ax1.axhline(frac_2d_DESI, ls="--", color="0.5", lw=1)
ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=9.5)
ax1.set_ylim(0, 1.08); ax1.set_ylabel(r"fraction of whitened $J_\mathrm{LLS}$ explained")
ax1.set_title("How much of the LLS template the forest can absorb\n"
              f"({frac_2d_DESI*100:.0f}% in the $A_p,n_s$ plane on DESI; "
              f"drops to {frac_2d_KS*100:.0f}% on high-$k$ KS)")
fig.suptitle("LLS amplitude template projected into the forest cosmology plane",
             fontweight="bold", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIGDIR}/degeneracy_lls_inplane_projection.png", dpi=130)
plt.close(fig)
print("\nwrote degeneracy_lls_inplane_projection.png")

# --- Figure 2: data leverage sqrt(F_ii) A_HCD vs B_HCD ---------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
lev_names = ["A_HCD", "r_sub", "B_HCD"]
lev_lab = [r"$A_\mathrm{HCD}$" + "\n(pivot amplitude)",
           r"$r_\mathrm{sub}$" + "\n(subDLA ratio)",
           r"$B_\mathrm{HCD}$" + "\n(z-tilt: the orthogonal lever)"]
lev_vals = [sqrtF["A_HCD"], sqrtF["r_sub"], sqrtF["B_HCD"]]
lev_colors = ["tab:red", "tab:green", "tab:blue"]
bars = ax.bar(range(3), lev_vals, color=lev_colors, edgecolor="k")
for b, vv in zip(bars, lev_vals):
    ax.text(b.get_x()+b.get_width()/2, vv*1.05, f"{vv:.1f}", ha="center", fontweight="bold")
ax.set_yscale("log")
ax.set_xticks(range(3)); ax.set_xticklabels(lev_lab, fontsize=10)
ax.set_ylabel(r"whitened data leverage  $\sqrt{F_{ii}} = \|J_i\|_{C_\mathrm{data}}$")
ratio = sqrtF["A_HCD"]/sqrtF["B_HCD"]
ax.set_title("Why the orthogonal z-lever is data-starved\n"
             rf"$A_\mathrm{{HCD}}$ leverage $\approx${ratio:.0f}$\times$ the z-tilt $B_\mathrm{{HCD}}$")
ax.text(0.97, 0.04,
        "single-fiducial Fisher diagonal.\nThe note quotes S/N($A_\\mathrm{HCD}$)$\\approx$62 vs "
        "S/N($B$)$\\approx$1.7\n(the prior/value-scaled chain version);\nsame conclusion: amplitude "
        "is measured, the z-tilt is not.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
fig.tight_layout()
fig.savefig(f"{FIGDIR}/degeneracy_snr_amplitude_vs_ztilt.png", dpi=130)
plt.close(fig)
print("wrote degeneracy_snr_amplitude_vs_ztilt.png")

# --- Figure 3: posterior correlation matrix --------------------------------
fig, ax = plt.subplots(figsize=(7.4, 6.4))
disp = [r"$A_p$", r"$n_s$", r"$A_\mathrm{HCD}$", r"$B_\mathrm{HCD}$", r"$r_\mathrm{sub}$"]
im = ax.imshow(Corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(names)):
    for j in range(len(names)):
        bold = (i, j) in [(i_A, i_ns), (i_ns, i_A)]
        ax.text(j, i, f"{Corr[i,j]:+.2f}", ha="center", va="center",
                color="white" if abs(Corr[i,j]) > 0.55 else "black",
                fontweight="bold" if bold else "normal",
                fontsize=12 if bold else 11)
for (ii, jj) in [(i_ns, i_A), (i_A, i_ns)]:
    ax.add_patch(Rectangle((jj-0.5, ii-0.5), 1, 1, fill=False, ec="lime", lw=3))
ax.set_xticks(range(len(names))); ax.set_xticklabels(disp)
ax.set_yticks(range(len(names))); ax.set_yticklabels(disp)
fig.colorbar(im, ax=ax, label="correlation", shrink=0.85)
ax.set_title("Posterior correlation (data Fisher + HCD priors)\n"
             rf"corr$(A_\mathrm{{HCD}}, n_s)$ = {corr_AHCD_ns:+.2f}")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/degeneracy_fisher_corr_matrix.png", dpi=130)
plt.close(fig)
print("wrote degeneracy_fisher_corr_matrix.png")

# --- Figure 4: prior-center -> n_s linear law ------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
slope = float(dns_per_sigA_in_signs)
xx = np.linspace(-2, 2, 50)
axL.plot(xx, slope*xx, color="tab:red", lw=2.5)
for s in (-1, 1):
    axL.plot(s, slope*s, marker="*", ms=22, color="gold", mec="k", mew=1)
axL.annotate(rf"$+1\sigma_A$ center $\to$" + "\n" + rf"$+{slope:.2f}\sigma_{{n_s}}$",
             xy=(1, slope), xytext=(0.05, 1.05),
             arrowprops=dict(arrowstyle="->"), fontsize=11,
             bbox=dict(boxstyle="round", fc="#fff8e1", ec="0.6"))
axL.axhline(0, color="0.8", lw=0.8); axL.axvline(0, color="0.8", lw=0.8)
axL.set_xlabel(r"$A_\mathrm{HCD}$ prior-center shift  [$\sigma_A$]")
axL.set_ylabel(r"induced $n_s$ MAP shift  [$\sigma_{n_s}$]")
axL.set_xlim(-2, 2); axL.set_ylim(-1.5, 1.5)
axL.grid(alpha=0.3)
axL.set_title("The prior-center systematic is LINEAR\n"
              rf"slope = corr $\times$ prior-dominance = {slope:.2f}")
# right: factorization bars
prod = float(corr_AHCD_ns * frac_A_prior)
fac_lab = [r"corr$(A_\mathrm{HCD}, n_s)$" + "\n[data+prior]",
           r"prior-dominance" + "\n" + r"of $A_\mathrm{HCD}$",
           r"product" + "\n" + r"($\approx$ slope)", "measured\nslope (npz)"]
fac_vals = [float(corr_AHCD_ns), float(frac_A_prior), prod, slope]
fac_colors = ["tab:blue", "tab:green", "0.6", "tab:red"]
bars = axR.bar(range(4), fac_vals, color=fac_colors, edgecolor="k")
for b, vv in zip(bars, fac_vals):
    axR.text(b.get_x()+b.get_width()/2, vv+0.02, f"{vv:.2f}", ha="center", fontweight="bold")
axR.set_xticks(range(4)); axR.set_xticklabels(fac_lab, fontsize=9.5)
axR.set_ylim(0, 1.0); axR.set_ylabel("value")
axR.set_title(r"Why $\sim$0.5$\sigma$: corr $\times$ prior-dominance $\approx$ slope"
              "\n(a tighter prior shrinks dominance, not corr)")
fig.suptitle(rf"Prior-width $\times$ degeneracy: the measured $+{slope:.2f}\sigma_{{n_s}}$ "
             rf"per $+1\sigma_A$ systematic (DESI+KS operating point)",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIGDIR}/degeneracy_prior_center_law.png", dpi=130)
plt.close(fig)
print("wrote degeneracy_prior_center_law.png")

# ===========================================================================
# SAVE NPZ
# ===========================================================================
np.savez(
    NPZ,
    # headline pairwise cosines (pure-class templates)
    cos_LLS_Ap=cos_LLS_Ap, cos_LLS_ns=cos_LLS_ns,
    cos_LLS_Ap_DESI=cos_LLS_Ap_DESI, cos_LLS_ns_DESI=cos_LLS_ns_DESI,
    cos_sub_ns=cos_sub_ns, cos_sub_ns_DESI=cos_sub_ns_DESI, cos_sub_Ap=cos_sub_Ap,
    # in-plane fractions
    frac_2d=frac_2d_DESI, frac_2d_DESI=frac_2d_DESI, frac_2d_KS=frac_2d_KS,
    frac_full_DESI=frac_full_DESI, frac_2d_joint=frac_2d_joint,
    frac_ns=frac_ns_DESI, frac_Ap=frac_Ap_DESI,
    # Fisher / correlation
    corr_AHCD_ns=corr_AHCD_ns, corr_AHCD_ns_dataonly=corr_AHCD_ns_dataonly,
    Corr=Corr, F=F, sd=sd, names=names,
    sqrtF_AHCD=sqrtF["A_HCD"], sqrtF_B=sqrtF["B_HCD"], sqrtF_rsub=sqrtF["r_sub"],
    # prior-center law
    dns_per_sigA_in_signs=dns_per_sigA_in_signs, frac_A_prior=frac_A_prior,
    frac_B=frac_B,
)
print("\nsaved npz to", NPZ)
print("\n========== HEADLINE CHECK ==========")
print(f"  cos(LLS, A_p)      = {cos_LLS_Ap:+.3f} (joint) / {cos_LLS_Ap_DESI:+.3f} (DESI)   [headline +0.81]")
print(f"  cos(subDLA, n_s)   = {cos_sub_ns:+.3f} (joint) / {cos_sub_ns_DESI:+.3f} (DESI)   [headline -0.93]")
print(f"  LLS in-plane DESI  = {frac_2d_DESI*100:.0f}%   [headline 91%]")
print(f"  LLS full-nuisance  = {frac_full_DESI*100:.0f}%   [headline 97%]")
print(f"  LLS in-plane KS    = {frac_2d_KS*100:.0f}%   [headline 64%]")
print(f"  corr(A_HCD, n_s)   = {corr_AHCD_ns:+.2f}   [figure boxed]")
print(f"  sqrt(F) A_HCD/B    = {sqrtF['A_HCD']:.1f}/{sqrtF['B_HCD']:.1f} (ratio {sqrtF['A_HCD']/sqrtF['B_HCD']:.0f}x)   [note S/N 62/1.7 chain-scaled]")
print(f"  prior-center slope = {float(dns_per_sigA_in_signs):.2f} sigma_ns / sigma_A   [headline +0.47..0.5]")
