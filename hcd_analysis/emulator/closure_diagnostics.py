"""Reference implementations for the Phase-C T4 closure / SBC statistical machinery.

Pure numpy/jax, NUTS-free, unit-testable. Items 1-6 of the T4 statistical design.

Conventions matched to ``hcd_analysis/emulator/likelihood.py``:
  - residual r = P_data - P_obs
  - C SPD, jittered: C += jitter * mean(diag C) * I  BEFORE Cholesky
  - L = cholesky(C) is LOWER triangular
  - ½ logdet C = Σ log diag(L)
  - whitening:  r_white = L⁻¹ r  = solve_triangular(L, r, lower=True)

References
----------
Talts+2018  (arXiv:1804.06788)  Simulation-Based Calibration: the rank statistic
   r = #{ θ_s < θ_true } is Uniform{0,...,L} iff sampler+model are correct.
Säilynoja+Vehtari+Bürkner 2022 (Stat&Comp; arXiv:2103.10522)  Graphical test for
   discrete uniformity: ECDF simultaneous confidence bands (the correct multiple-
   comparison-adjusted envelope for the rank/PIT ECDF; replaces per-bin χ²/KS).
Modrak+2023  (Bayesian Analysis; arXiv:2211.02383)  "Simulation-based calibration
   checking ... the joint log-likelihood rank is a near-sufficient 1-D summary";
   ranking log_lik(θ_true) among log_lik(θ_s) is the single most powerful SBC quantity.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


# ============================================================================
# ITEM 1a — SBC rank statistic + thinning to ESS
# ============================================================================

def autocorr_ess(x):
    """Effective sample size of a 1-D MCMC chain via the initial-monotone-sequence
    (Geyer) estimator of the integrated autocorrelation time.

    ESS = N / (1 + 2 Σ_{t≥1} ρ_t), truncated at the first negative pair sum
    (Geyer 1992). Matches arviz's `ess` closely for a single chain. Pure numpy.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0.0:
        return float(n)  # constant chain: treat as fully informative (degenerate)
    # autocovariance via FFT
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n] / n
    rho = acov / acov[0]
    # Geyer initial monotone sequence: pair sums Γ_k = ρ_{2k} + ρ_{2k+1}, keep while >0
    # and enforce monotone non-increasing.
    gamma = rho[1:2 * (n // 2 - 1):2] + rho[2:2 * (n // 2 - 1) + 1:2]
    # truncate at first non-positive
    neg = np.where(gamma <= 0)[0]
    cut = neg[0] if neg.size else gamma.size
    gamma = gamma[:cut]
    # enforce monotone non-increasing (initial monotone sequence)
    gamma = np.minimum.accumulate(gamma)
    tau = 1.0 + 2.0 * gamma.sum()
    tau = max(tau, 1.0)
    return float(n / tau)


def thin_to_ess(draws, target_per_param=None):
    """Thin a (L, D) chain to near-independence for valid SBC ranking.

    SBC ranks are only Uniform{0..L'} if the L' retained draws are ~independent
    (Talts+2018 §4: autocorrelation inflates the variance of the rank ⇒ artifactual
    ∩-shaped / spiky rank histograms that mimic miscalibration). We thin by the
    WORST (smallest-ESS) parameter so every marginal is valid on the same grid.

    Args:
      draws: (L, D) post-warmup draws for ONE mock, ONE chain.
      target_per_param: if given, also cap retained draws at this count.
    Returns:
      thinned: (L', D) every-`step`-th draw, L' = ceil(L/step) (or fewer if capped).
      step:    the thinning interval used.
      ess_min: the limiting (min over params) ESS.
    """
    draws = np.asarray(draws, dtype=float)
    L, D = draws.shape
    ess_min = min(autocorr_ess(draws[:, d]) for d in range(D))
    ess_min = max(ess_min, 1.0)
    step = max(1, int(np.ceil(L / ess_min)))
    thinned = draws[::step]
    if target_per_param is not None and thinned.shape[0] > target_per_param:
        idx = np.linspace(0, thinned.shape[0] - 1, target_per_param).round().astype(int)
        thinned = thinned[idx]
    return thinned, step, ess_min


def sbc_rank(theta_true, draws):
    """SBC rank of a scalar truth among L posterior draws (Talts+2018 Eq. 1).

    rank = #{ draws < theta_true }  ∈ {0, 1, ..., L}.   (L+1 possible values.)
    Ties broken by randomization (matters only for discrete/identical draws) to
    keep the discrete-uniform null exact.

    Args:
      theta_true: scalar truth.
      draws: (L,) thinned, near-independent posterior draws of that parameter.
    Returns: integer rank in [0, L].
    """
    draws = np.asarray(draws, dtype=float)
    lt = int(np.sum(draws < theta_true))
    eq = int(np.sum(draws == theta_true))
    if eq:  # randomized tie-break preserves exact uniformity
        lt += int(np.random.randint(0, eq + 1))
    return lt


def sbc_ranks_multiparam(theta_true_vec, draws):
    """Vectorized sbc_rank over D params for one mock. Returns (D,) int ranks."""
    theta_true_vec = np.asarray(theta_true_vec, dtype=float)
    draws = np.asarray(draws, dtype=float)
    return np.array([sbc_rank(theta_true_vec[d], draws[:, d])
                     for d in range(draws.shape[1])], dtype=int)


def rank_histogram_bins(L, n_mocks, target_per_bin=20):
    """Choose B so that (L+1) is a MULTIPLE of B (Talts+2018: else the histogram has
    structurally uneven bin occupancy that fakes non-uniformity), and ~target_per_bin
    mocks land per bin. Returns B (number of bins)."""
    B_target = max(1, int(round(n_mocks / target_per_bin)))
    # largest divisor of (L+1) that is <= B_target (fall back to 1)
    divisors = [b for b in range(1, L + 2) if (L + 1) % b == 0]
    feasible = [b for b in divisors if b <= B_target]
    return max(feasible) if feasible else 1


# ============================================================================
# ITEM 1b — ECDF simultaneous confidence bands (Säilynoja+2022), STANDALONE
# ============================================================================

def _ecdf_band_gamma(n_draws, K, n_sim=4000, prob=0.95, rng=None):
    """Find the per-point coverage γ such that the SIMULTANEOUS coverage over the K
    evaluation points of the ECDF equals `prob`, for L=n_draws iid uniform PIT values.

    Säilynoja+2022 §2.2: the ECDF of n iid Uniform(0,1) evaluated on a grid is, at each
    point, Binomial(n, z_k)/n. A simultaneous band that controls family-wise error uses
    the SAME pointwise tail probability γ at every grid point; γ is calibrated by Monte
    Carlo so that P( ∀k: ecdf_k ∈ [B⁻¹(γ/2), B⁻¹(1-γ/2)] ) = prob. This is the
    "optimal simultaneous" band (their `adjust_gamma`); it does NOT assume the pointwise
    bands are independent (they are not), so it is the correct multiple-comparison fix.
    """
    rng = np.random.default_rng(rng)
    n = int(n_draws)
    # simulate the max/min pointwise binomial tail prob attained across the grid
    # For each simulated uniform sample, at grid point z_k the ECDF count ~ Binom(n, z_k).
    # We need the distribution of  min_k min(F_bin(count), 1 - F_bin(count-1))  i.e. the
    # smallest two-sided binomial p-value over the grid; γ = its `1-prob` quantile.
    z = np.linspace(1.0 / (K + 1), K / (K + 1), K)  # interior grid points
    # precompute binomial cdf table is expensive; use the simulation of counts directly.
    min_pval = np.empty(n_sim)
    for s in range(n_sim):
        u = rng.random(n)
        # ecdf counts at each z_k = #{u <= z_k}
        u_sorted = np.sort(u)
        counts = np.searchsorted(u_sorted, z, side="right")
        # two-sided binomial tail p-value at each point
        # P(X <= counts) and P(X >= counts) under Binom(n, z_k)
        lower_p = stats.binom.cdf(counts, n, z)
        upper_p = stats.binom.sf(counts - 1, n, z)
        pval = 2.0 * np.minimum(lower_p, upper_p)
        min_pval[s] = pval.min()
    gamma = np.quantile(min_pval, 1.0 - prob)
    return float(gamma)


def ecdf_pit_bands(ranks, n_draws, prob=0.95, n_grid=100, n_sim=2000, rng=None):
    """ECDF simultaneous confidence bands as the SBC pass test (Säilynoja+2022).

    Converts SBC ranks → PIT-like values, builds the empirical ECDF, and a simultaneous
    `prob`-level envelope under the discrete-uniform null. PASS iff the ECDF stays inside
    the band at every grid point (family-wise; the multiple-comparison-correct test that
    replaces per-bin χ²/KS — the latter either over- or under-rejects on rank histograms).

    Args:
      ranks:   (N,) integer SBC ranks in {0,...,n_draws} for ONE quantity over N mocks.
      n_draws: L, the number of posterior draws ranked against (ranks ∈ {0..L}).
      prob:    simultaneous coverage of the band (0.95 → 95% bands).
      n_grid:  number of evaluation points for the ECDF curve.
      n_sim:   Monte-Carlo draws to calibrate the band width γ.
    Returns:
      (lower, upper, ecdf, pass_bool) each of length n_grid+1 evaluated on the same grid;
      `pass_bool` is True iff ecdf ∈ [lower, upper] everywhere.
      Also returns the grid as the 5th element for plotting.
    """
    ranks = np.asarray(ranks, dtype=float)
    N = ranks.size
    L = int(n_draws)
    # PIT-like transform: map rank in {0..L} to (0,1). r/(L) would land on {0,1};
    # the field-standard fractional-rank PIT is (rank + U)/(L+1) but for the band
    # test we use the discrete ECDF of rank/(L+1)... we instead build the ECDF of the
    # *fractional ranks* and compare to a Uniform null via the simultaneous binomial band.
    pit = (ranks + 0.5) / (L + 1.0)          # continuity-corrected, in (0,1)
    pit_sorted = np.sort(pit)

    grid = np.linspace(0.0, 1.0, n_grid + 1)
    # empirical ECDF of the PIT values on the grid
    ecdf = np.searchsorted(pit_sorted, grid, side="right") / N

    # simultaneous band: at grid point z, count ~ Binom(N, z); band on count/N.
    gamma = _ecdf_band_gamma(N, n_grid, n_sim=n_sim, prob=prob, rng=rng)
    # interior + endpoints; at z=0 lower=0,upper=0-ish; at z=1 both =1.
    lower = stats.binom.ppf(gamma / 2.0, N, grid) / N
    upper = stats.binom.ppf(1.0 - gamma / 2.0, N, grid) / N
    # clean endpoints (binom.ppf gives nan-ish behavior at p=0/1 grid edges)
    lower[grid <= 0] = 0.0
    upper[grid <= 0] = 0.0
    lower[grid >= 1] = 1.0
    upper[grid >= 1] = 1.0
    # tolerance: ECDF is a step function; allow it to touch the band (>=, <=).
    inside = (ecdf >= lower - 1e-12) & (ecdf <= upper + 1e-12)
    pass_bool = bool(np.all(inside))
    return lower, upper, ecdf, pass_bool, grid


# ============================================================================
# ITEM 2 — log-likelihood rank as the PRIMARY joint SBC statistic (Modrak+2023)
# ============================================================================

def loglik_rank(loglik_true, loglik_draws):
    """SBC rank of the joint log-lik at the TRUTH among the posterior-draw log-liks.

    Modrak+2023: ranking ℓ(θ_true, y) within {ℓ(θ_s, y)} is a 1-D summary that is
    near-sufficient for the most damaging SBC failures (a global location/scale error
    of the posterior moves the truth's likelihood relative to the bulk). It collapses
    the D marginal tests into ONE, avoiding the multiple-testing dilution of 20-40
    marginal rank tests, and it is sensitive to JOINT (correlation) miscalibration that
    every marginal can individually pass.

    IMPORTANT: ℓ must be the SAME function for truth and draws (here `gaussian_loglik`,
    evaluated at the same mock data y), so the rank is exactly the SBC rank of a scalar
    derived statistic and inherits the discrete-uniform null. Feed the resulting ranks
    through `ecdf_pit_bands` exactly like a parameter rank.

    Args:
      loglik_true:  scalar ℓ(θ_true, y) for this mock.
      loglik_draws: (L,) ℓ(θ_s, y) over the thinned posterior draws for this mock.
    Returns: integer rank in [0, L].
    """
    loglik_draws = np.asarray(loglik_draws, dtype=float)
    lt = int(np.sum(loglik_draws < loglik_true))
    eq = int(np.sum(loglik_draws == loglik_true))
    if eq:
        lt += int(np.random.randint(0, eq + 1))
    return lt


# ============================================================================
# ITEM 3 — fixed Fisher-eigenvector projection rank (τ₀–θ and cross-z τ₀)
# ============================================================================

def joint_fisher(J, C, jitter=1e-10):
    """Joint Fisher information F = JᵀC⁻¹J at a fiducial point.

    J: (K, P) Jacobian of the forward model P_obs wrt the P joint params [θ, τ₀(per z), α]
       evaluated at the FIDUCIAL truth (NOT a per-mock posterior — see project_eigvec_rank).
    C: (K, K) the assembled covariance (cosmic + C_emu) at the fiducial.
    Returns F: (P, P) SPD.
    """
    J = np.asarray(J, dtype=float)
    C = np.asarray(C, dtype=float)
    K = C.shape[0]
    C = C + jitter * np.mean(np.diag(C)) * np.eye(K)
    L = np.linalg.cholesky(C)
    # C⁻¹ J = L⁻ᵀ (L⁻¹ J)
    Linv_J = np.linalg.solve(L, J)
    Cinv_J = np.linalg.solve(L.T, Linv_J)
    return J.T @ Cinv_J


def correlation_eigvec(F, idx_a, idx_b):
    """The leading eigenvector of the CORRELATION between two parameter blocks A,B from
    the truth Fisher F, returned in the full P-dim coordinate (zeros outside A∪B).

    Construction (de-circularized; the vector is FIXED from truth, not the per-mock post):
      Σ = F⁻¹ (the Cramér-Rao posterior covariance at the fiducial). Restrict to the
      A∪B block, convert to correlation, and take the eigenvector of the OFF-DIAGONAL
      coupling that maximizes the A–B correlation. We use the eigenvector of the block
      correlation matrix with the largest |off-block| weight, i.e. the direction along
      which τ₀ and θ (or τ₀_z1 and τ₀_z2) are most degenerate. Ranking the truth along
      this fixed direction probes the JOINT τ₀–θ geometry that marginals miss.

    Args:
      F: (P,P) joint Fisher at fiducial.
      idx_a, idx_b: index arrays into the P params for the two blocks (e.g. θ and τ₀).
    Returns: v (P,) unit vector, nonzero only on idx_a ∪ idx_b.
    """
    F = np.asarray(F, dtype=float)
    P = F.shape[0]
    Sigma = np.linalg.inv(F)
    idx = np.concatenate([np.asarray(idx_a), np.asarray(idx_b)])
    S = Sigma[np.ix_(idx, idx)]
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)                 # correlation matrix of the A∪B block
    na = len(idx_a)
    # Eigendecompose the block correlation and pick the mode with the largest
    # cross-block L2 weight: the eigenvector whose mass is shared between A and B is
    # the τ₀–θ (or τ₀_z1–τ₀_z2) degeneracy direction. (A within-block-only eigenvector
    # would have zero weight in the other block and would not probe the coupling.)
    w, V = np.linalg.eigh(R)
    cross_weight = np.array([np.linalg.norm(V[:na, j]) * np.linalg.norm(V[na:, j])
                             for j in range(V.shape[1])])
    v_block = V[:, np.argmax(cross_weight)]
    v = np.zeros(P)
    v[idx] = v_block / np.linalg.norm(v_block)
    return v


def project_eigvec_rank(theta_true, draws, v):
    """Project truth and posterior draws onto the FIXED Fisher eigenvector v, then SBC-rank.

    The projection scalar s = vᵀθ collapses the joint degeneracy direction to 1-D; ranking
    s_true within {s_s} tests calibration ALONG the τ₀–θ (or cross-z τ₀) degeneracy — the
    axis a per-param test is blind to. v MUST be the truth-Fisher eigenvector (same for all
    mocks); using each mock's own posterior eigenvector would be circular (the direction
    would adapt to the very miscalibration we test for).

    Args:
      theta_true: (P,) joint truth vector.
      draws: (L, P) thinned posterior draws.
      v: (P,) fixed unit projection vector.
    Returns: integer rank of vᵀθ_true in [0, L].
    """
    theta_true = np.asarray(theta_true, dtype=float)
    draws = np.asarray(draws, dtype=float)
    v = np.asarray(v, dtype=float)
    s_true = float(v @ theta_true)
    s_draws = draws @ v
    return sbc_rank(s_true, s_draws)


# ============================================================================
# ITEM 4 — Leg-B coverage + cemu_inflate calibration
# ============================================================================

def central_interval(draws, q):
    """Central equal-tailed q-credible interval (the [ (1-q)/2, (1+q)/2 ] quantiles)."""
    draws = np.asarray(draws, dtype=float)
    lo = np.quantile(draws, (1.0 - q) / 2.0)
    hi = np.quantile(draws, (1.0 + q) / 2.0)
    return lo, hi


def hpd_interval(draws, q):
    """Highest-posterior-density q-interval (narrowest interval containing q mass).
    Assumes unimodal-ish marginals; for multimodal use the central interval. HUMAN CHOICE:
    HPD (reported here) vs central/quantile — quantile is reparam-invariant and the safer
    default for skewed τ₀; HPD is tighter for Gaussian-ish cosmo params. Flagged in the doc.
    """
    draws = np.sort(np.asarray(draws, dtype=float))
    n = draws.size
    m = int(np.floor(q * n))
    if m < 1:
        return draws[0], draws[-1]
    widths = draws[m:] - draws[:n - m]
    j = int(np.argmin(widths))
    return draws[j], draws[j + m]


def empirical_coverage(truths, intervals):
    """Fraction of mocks whose truth lands in its interval, with a Wilson binomial CI.

    Args:
      truths: (N,) per-mock scalar truths of one parameter.
      intervals: (N, 2) per-mock (lo, hi) credible intervals at level q.
    Returns: dict(coverage, n, ci_low, ci_high) — Wilson 95% interval on the coverage.
    """
    truths = np.asarray(truths, dtype=float)
    lo = np.asarray(intervals)[:, 0]
    hi = np.asarray(intervals)[:, 1]
    hit = (truths >= lo) & (truths <= hi)
    N = truths.size
    k = int(hit.sum())
    cov = k / N
    # Wilson score interval (better than Wald near 0/1)
    zc = 1.959963984540054
    denom = 1.0 + zc**2 / N
    centre = (cov + zc**2 / (2 * N)) / denom
    half = (zc / denom) * np.sqrt(cov * (1 - cov) / N + zc**2 / (4 * N**2))
    return dict(coverage=cov, n=N, k=k, ci_low=centre - half, ci_high=centre + half)


def coverage_at_inflate(truths_per_mock, draws_per_mock, inflate, q, sigma_scale_fn,
                        interval="central"):
    """Empirical coverage of param-block when each mock's posterior σ is rescaled by an
    inflation that comes from scaling C_emu by `inflate`.

    For the CALIBRATION root-find we do NOT re-run NUTS per inflate (too expensive). The
    cheap, defensible surrogate: a Gaussian-posterior re-scaling — widen each mock's draws
    about their mean by the factor the inflated C_emu implies on that param's variance,
    supplied by `sigma_scale_fn(inflate)` (≥1). This is the field surrogate used to set
    `cemu_inflate` before one confirmatory NUTS pass at the chosen value.

    Args:
      truths_per_mock: (N,) truths.
      draws_per_mock:  list of (L,) per-mock draw arrays for ONE param.
      inflate:         scalar C_emu inflation factor.
      q:               coverage level (0.95).
      sigma_scale_fn:  callable inflate -> per-param posterior-σ multiplier (≥1).
      interval:        "central" or "hpd".
    Returns: coverage fraction.
    """
    s = float(sigma_scale_fn(inflate))
    ints = []
    fn = central_interval if interval == "central" else hpd_interval
    for d in draws_per_mock:
        d = np.asarray(d, dtype=float)
        mu = d.mean()
        d_scaled = mu + (d - mu) * s
        ints.append(fn(d_scaled, q))
    return empirical_coverage(truths_per_mock, np.array(ints))["coverage"]


def calibrate_cemu_inflate(truths_per_mock, draws_per_mock, sigma_scale_fn,
                           q=0.95, lo=1.0, hi=10.0, tol=1e-3, max_iter=60):
    """1-D bisection root-find for the C_emu inflation that restores `q` coverage.

    Monotone-increasing coverage(inflate) ⇒ bracket [lo,hi] and bisect on
    coverage(inflate) - q. Returns the smallest inflate with coverage ≥ q (one-sided pass).
    If coverage already ≥ q at lo, returns lo (no inflation needed). If hi still under-covers,
    returns hi with a flag.

    Per-(class,z) calibration: call this once per (HCD class, z) sub-block whose draws/truths
    you pass in; the design only inflates where the under-coverage localizes (plan §2).
    """
    def cov(infl):
        return coverage_at_inflate(truths_per_mock, draws_per_mock, infl, q, sigma_scale_fn)

    c_lo = cov(lo)
    if c_lo >= q:
        return dict(inflate=lo, coverage=c_lo, converged=True, note="no inflation needed")
    c_hi = cov(hi)
    if c_hi < q:
        return dict(inflate=hi, coverage=c_hi, converged=False,
                    note="hi bound still under-covers; widen `hi`")
    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        cm = cov(m)
        if abs(cm - q) < tol:
            return dict(inflate=m, coverage=cm, converged=True, note="")
        if cm < q:
            a = m
        else:
            b = m
    m = 0.5 * (a + b)
    return dict(inflate=m, coverage=cov(m), converged=True, note="max_iter")


# ============================================================================
# ITEM 5 — C_emu whitening test (emulator-agent M3)
# ============================================================================

def whiten_residual(r, C, jitter=1e-10):
    """r_white = L⁻¹ r with L = cholesky(C) lower-triangular (matches gaussian_loglik).

    Under a correct error model, r ~ N(0, C) ⇒ r_white ~ N(0, I): components are iid
    standard normal. This is the single most diagnostic emulator-error check (plan §2).
    """
    r = np.asarray(r, dtype=float)
    C = np.asarray(C, dtype=float)
    K = C.shape[0]
    C = C + jitter * np.mean(np.diag(C)) * np.eye(K)
    L = np.linalg.cholesky(C)
    # forward-substitution: solve L z = r
    z = np.linalg.solve(L, r)   # exact for triangular L; (use solve_triangular for speed)
    return z


def whitening_test(residuals, covs, jitter=1e-10):
    """Whiten a stack of held-out-sim residuals and test r_white ~ N(0, I).

    Args:
      residuals: (M, K) per-held-out-sim P1D residuals r = P_data - P_obs.
      covs:      (M, K, K) the assembled C at each sim's (θ,τ₀)  (OR (K,K) shared).
    Returns dict with:
      mean, var       : mean and variance of the pooled whitened residuals (target 0, 1)
      mean_z, mean_p  : z-stat & p-value for mean≈0  (sem = 1/sqrt(M*K))
      var_chi2_p      : χ² p-value for var≈1
      ks_stat, ks_p   : KS test of whitened residuals vs N(0,1) (uniformity of Φ(z))
      chi2_over_dof   : Σ r_white² / (M*K)  — the amplitude check (target 1)
      whitened        : (M, K) the whitened residuals
    """
    residuals = np.asarray(residuals, dtype=float)
    M, K = residuals.shape
    covs = np.asarray(covs, dtype=float)
    shared = covs.ndim == 2
    W = np.empty((M, K))
    for m in range(M):
        Cm = covs if shared else covs[m]
        W[m] = whiten_residual(residuals[m], Cm, jitter=jitter)
    flat = W.reshape(-1)
    n = flat.size
    mean = float(flat.mean())
    var = float(flat.var(ddof=1))
    # mean test: under H0, mean ~ N(0, 1/n)
    mean_z = mean * np.sqrt(n)
    mean_p = 2.0 * stats.norm.sf(abs(mean_z))
    # variance test: (n-1) s² ~ χ²_{n-1}
    chi2_var = (n - 1) * var
    var_p = 2.0 * min(stats.chi2.cdf(chi2_var, n - 1), stats.chi2.sf(chi2_var, n - 1))
    # KS vs standard normal
    ks_stat, ks_p = stats.kstest(flat, "norm")
    chi2_over_dof = float(np.sum(flat**2) / n)
    return dict(mean=mean, var=var, mean_z=mean_z, mean_p=mean_p,
                var_chi2_p=var_p, ks_stat=float(ks_stat), ks_p=float(ks_p),
                chi2_over_dof=chi2_over_dof, whitened=W, n=n)


# ============================================================================
# ITEM 6 — power calculation for N
# ============================================================================

def sbc_power_N(bias_sigma=0.2, prob=0.95, power=0.80):
    """N needed for the SBC ECDF/rank test to DETECT a posterior-mean bias of `bias_sigma`.

    A uniform bias of b·σ in the posterior mean shifts the SBC PIT distribution. For a
    one-sided shift, the PIT-ECDF is displaced from the diagonal by ≈ Φ(b/√2)−Φ(0) at the
    median (the rank of the truth among draws when both truth and draws carry σ; the √2 is
    truth-noise + posterior-width). Detection requires this displacement to exceed the
    half-width of the simultaneous ECDF band at the median, which scales as ~zc/(2√N).

    Solving  Δ_PIT(b) ≥ z_power · sqrt( Δ(1-Δ)/N ) + half_band(N)  for N gives the formula
    below (a conservative closed form; the human runs the exact MC `sbc_power_mc` to confirm).

    Returns dict(N, displacement, note).
    """
    # PIT displacement at the median from a b·σ posterior-mean bias.
    # truth ~ N(0,σ), draws ~ N(bias, σ): rank statistic ≈ Φ( -b/√2 ); displacement from 0.5:
    disp = abs(stats.norm.cdf(bias_sigma / np.sqrt(2.0)) - 0.5)
    z_pow = stats.norm.ppf(power)
    z_band = stats.norm.ppf(0.5 + prob / 2.0)   # ~1.96 for 95%
    # detection: disp ≥ (z_pow + z_band) * sqrt(0.25 / N)  (Bernoulli sd at p=0.5)
    N = ((z_pow + z_band)**2 * 0.25) / disp**2
    return dict(N=int(np.ceil(N)), displacement=disp,
                note=f"to detect a {bias_sigma}σ mean bias at {int(prob*100)}% bands, "
                     f"{int(power*100)}% power")


def sbc_power_mc(N, bias_sigma=0.2, n_draws=99, prob=0.95, n_rep=400, rng=None):
    """Monte-Carlo confirmation: empirical rejection rate of `ecdf_pit_bands` when the
    posterior mean carries a `bias_sigma` shift, at sample size N. Returns the power."""
    rng = np.random.default_rng(rng)
    rej = 0
    for _ in range(n_rep):
        # truths ~ N(0,1); each mock's posterior draws ~ N(truth + bias, 1) (mean-shifted)
        truths = rng.standard_normal(N)
        ranks = np.empty(N, dtype=int)
        for i in range(N):
            draws = (truths[i] + bias_sigma) + rng.standard_normal(n_draws)
            ranks[i] = int(np.sum(draws < truths[i]))
        _, _, _, passed, _ = ecdf_pit_bands(ranks, n_draws, prob=prob, n_sim=600, rng=rng)
        rej += (not passed)
    return rej / n_rep
