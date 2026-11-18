"""Science checks of hcd_priya that need no trained model: units, conventions and statistics.

- The wavenumber convention of the native modes: k_n = 2 pi n / L (1 + z) / (100 E(z)) in s/km, n = 1, 2, ...,
  checked against the formula written out here at three redshifts and at the corners of the box.
- PRIYA's mean-flux law, tau_eff = tau0_amp ((1 + z) / 4)^dtau0 0.0023 (1 + z)^3.65 (Kim et al. 2007), at every
  snapshot and at its pivot.
- Wavenumbers in h/Mpc above the model's highest mode are refused, never evaluated; wavenumbers in s/km inside the
  modes are accepted. Wavenumbers carry no unit: a data set in h/Mpc whose bins all lie below about 0.1 may pass
  undetected (docs/likelihood.md says so).
- The Gaussian data likelihood, at the true model, has the statistics of a Gaussian (2,000 seeded draws).
- The interpolation of a model spectrum to data bins is exact for a line and has the analytic error for a quadratic.

Every number here is synthetic or a published constant (the PRIYA box length, the fit of Kim et al. 2007). The tests
need only the installed package, and a stand-in model written here for the likelihood.
"""

import math

import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import likelihood as L
from hcd_analysis.emulator import kcoord as K
from hcd_analysis.emulator.data import PARAM_LIMITS
from hcd_analysis.emulator.schema import L_BOX_HMPC

# ------------------------------------------------------------------------------------------- independent constants
L_BOX = 120.0  # Mpc/h, the PRIYA box (Bird et al. 2023)
N_MODES = 172  # the modes of a production model: n = 1 .. 172
KIM_AMP, KIM_SLOPE = 0.0023, 3.65  # tau_Kim(z) = 0.0023 (1 + z)^3.65 (Kim et al. 2007)
Z_PIVOT = 3.0  # PRIYA's (1 + z) / 4
Z_MODES = (2.2, 3.0, 4.6)
MIDPOINT = {
    "ns": 0.925,
    "Ap": 1.9e-9,
    "herei": 4.0,
    "heref": 2.7,
    "alphaq": 2.15,
    "hub": 0.70,
    "omegamh2": 0.143,
    "hireionz": 7.25,
    "bhfeedback": 0.05,
}
THETA = dict(MIDPOINT, tau0_amp=1.0, dtau0=0.0)
I_HUB, I_OMH2 = hp.PARAM_NAMES.index("hub"), hp.PARAM_NAMES.index("omegamh2")


def box_hub_omegamh2():
    """(hub, omegamh2) at the four corners and at the midpoint of the training box, as the package states the box."""
    lim = np.asarray(PARAM_LIMITS, dtype=np.float64)
    (h0, h1), (o0, o1) = lim[I_HUB], lim[I_OMH2]
    return [(h0, o0), (h0, o1), (h1, o0), (h1, o1), (0.5 * (h0 + h1), 0.5 * (o0 + o1))]


def k_skm_independent(z, hub, omegamh2, n_modes=N_MODES):
    """2 pi n / L (1 + z) / (100 E(z)), n = 1 .. n_modes, with E(z) = sqrt(Om (1 + z)^3 + 1 - Om) and
    Om = omegamh2 / hub^2. Written out here, without the package."""
    n = np.arange(1, n_modes + 1, dtype=np.float64)
    om = omegamh2 / hub**2
    e = math.sqrt(om * (1.0 + z) ** 3 + 1.0 - om)
    return 2.0 * np.pi * n / L_BOX * (1.0 + z) / (100.0 * e)


class StubModel:
    """A stand-in for a loaded model, for the likelihood only. ``predict_many`` returns a stacked ``hp.Prediction`` in
    which the four class spectra at a redshift are all the row given for it, so that the mixed spectrum is that row
    whatever the class fractions (P_c - P_clean is exactly zero). Synthetic values only."""

    def __init__(self, rows):
        self.rows = {
            hp.snapshot_z(z): (np.asarray(k, float), np.asarray(p, float))
            for z, (k, p) in rows.items()
        }
        self.calls = 0

    def predict_many(
        self, params, z, *, tau_eff=None, tau0_amp=None, dtau0=None, k=None, fidelity=None
    ):
        self.calls += 1
        zs = [hp.snapshot_z(v) for v in z]
        kk = np.stack([self.rows[v][0] for v in zs])
        pc = np.stack([np.repeat(self.rows[v][1][None, :], 4, axis=0) for v in zs])
        n = len(zs)
        return hp.Prediction(
            z=np.asarray(zs),
            k=kk,
            P_class=pc,
            template=np.zeros((n, 3, kk.shape[1])),
            w_c=np.tile([0.03, 0.01, 0.005], (n, 1)),
            dndx=np.full((n, 3), 0.1),
            member_std=None,
            tau_eff=np.ones(n),
            params=dict(params),
            flags={},
            labels={},
            provenance={},
            stacked=True,
        )


# ---------------------------------------------------------------------------------- the wavenumber convention
def test_the_native_modes_follow_the_box_convention_in_s_per_km():
    """k_skm,n = 2 pi n / L (1 + z) / (100 E(z)) at z 2.2, 3.0, 4.6 and the corners and midpoint of (hub, omegamh2)."""
    assert L_BOX_HMPC == L_BOX
    k_com = 2.0 * np.pi * np.arange(1, N_MODES + 1, dtype=np.float64) / L_BOX
    for z in Z_MODES:
        for hub, omh2 in box_hub_omegamh2():
            om = (
                omh2 / hub**2
            )  # the package's Om: omegamh2 / hub^2, flat, no radiation (checked through E itself)
            assert float(K.E_of_z(z, hub, omh2)) == pytest.approx(
                math.sqrt(om * (1.0 + z) ** 3 + 1.0 - om), rel=1e-14, abs=0
            )
            got = np.asarray(K.k_skm_from_kcom(k_com, z, hub, omh2), dtype=np.float64)
            np.testing.assert_allclose(got, k_skm_independent(z, hub, omh2), rtol=1e-12, atol=0)
            assert got[0] == pytest.approx(
                got[1] - got[0], rel=1e-12, abs=0
            )  # the modes start at n = 1


# ---------------------------------------------------------------------------------------- the mean-flux law
def test_the_priya_mean_flux_law_at_the_default_and_at_the_pivot():
    """tau_eff_priya(z, 1, 0) = 0.0023 (1 + z)^3.65 at every snapshot; at z = 3 it is tau0_amp 0.0023 4^3.65 for any
    dtau0; and everywhere tau0_amp ((1 + z) / 4)^dtau0 0.0023 (1 + z)^3.65. Relative 1e-13."""
    assert hp.Z_GRID == tuple(round(2.0 + 0.2 * i, 10) for i in range(18))  # 2.0, 2.2, ..., 5.4
    for z in hp.Z_GRID:
        want = KIM_AMP * (1.0 + z) ** KIM_SLOPE
        assert abs(hp.tau_eff_priya(z, 1.0, 0.0) - want) <= 1e-13 * want
    at_pivot = KIM_AMP * 4.0**KIM_SLOPE
    for amp in (0.5, 0.9, 1.0, 1.1, 1.7):
        for dtau0 in (-2.0, -0.4, 0.0, 0.2, 0.75, 3.0):
            assert (
                abs(hp.tau_eff_priya(Z_PIVOT, amp, dtau0) - amp * at_pivot)
                <= 1e-13 * amp * at_pivot
            )
            for z in hp.Z_GRID:
                want = (
                    amp * ((1.0 + z) / (1.0 + Z_PIVOT)) ** dtau0 * KIM_AMP * (1.0 + z) ** KIM_SLOPE
                )
                assert abs(hp.tau_eff_priya(z, amp, dtau0) - want) <= 1e-13 * want


# ----------------------------------------------------------------------------------------------- the units
def test_wavenumbers_in_h_per_mpc_above_the_modes_are_refused_and_in_s_per_km_accepted():
    """Data in h/Mpc (0.1 to 5) lie above every mode in s/km at every redshift and box point used here: refused by
    interp_to_bins and by the likelihood (a sampler gets minus infinity, counted), never evaluated. The refusal works
    because the highest mode lies below 0.1 s/km; h/Mpc data wholly below 0.1 may not be caught (module docstring).
    """
    k_hmpc = np.geomspace(0.1, 5.0, 25)
    for z in Z_MODES:
        for hub, omh2 in box_hub_omegamh2():
            modes = k_skm_independent(z, hub, omh2)
            p = 10.0 / (1.0 + (modes / 0.02) ** 2)
            for k_data in (
                k_hmpc,
                k_hmpc[:3],
                np.array([modes[5], 0.1]),
            ):  # all, the lowest few, one in one out
                with pytest.raises(L.OutsideModesError):
                    L.interp_to_bins(modes, p, k_data)
            inside = np.linspace(modes[0], modes[-1], 40)
            assert np.all(np.isfinite(L.interp_to_bins(modes, p, inside)))
    # through the likelihood, with a stand-in model on the s/km modes at the midpoint
    z = 3.0
    modes = k_skm_independent(z, MIDPOINT["hub"], MIDPOINT["omegamh2"])
    stub = StubModel({z: (modes, 10.0 / (1.0 + (modes / 0.02) ** 2))})
    bad = L.P1DData.from_blocks(
        z=[z], k=[k_hmpc], p1d=[np.ones(k_hmpc.size)], cov_blocks=[np.eye(k_hmpc.size)]
    )
    like = L.GaussianP1DLikelihood(stub, bad)
    with pytest.raises(L.OutsideModesError):
        like.loglike(THETA)
    assert like.sampler_logp(THETA) == -math.inf
    assert like.refusals["OutsideModesError"][0] == 1
    k_ok = np.linspace(modes[2], modes[-3], 30)
    good = L.P1DData.from_blocks(
        z=[z], k=[k_ok], p1d=[np.ones(k_ok.size)], cov_blocks=[np.eye(k_ok.size)]
    )
    assert np.isfinite(L.GaussianP1DLikelihood(stub, good).loglike(THETA))


# -------------------------------------------------------------------------- the statistics of the likelihood
Z_STATS = (2.2, 2.6, 3.0, 3.4)
SIZES_STATS = (11, 12, 13, 14)  # N = 50 bins
N_DRAWS = 2000


def _statistics_setup():
    """The fixed model vector m (positive), its stand-in model, and a fixed non-diagonal covariance C (seeded)."""
    rng = np.random.default_rng(20261117)
    rows, k_rows, m_rows = {}, [], []
    for z, size in zip(Z_STATS, SIZES_STATS):
        modes = k_skm_independent(z, MIDPOINT["hub"], MIDPOINT["omegamh2"], n_modes=24)
        p = (8.0 + 2.0 * z) / (1.0 + (modes / 0.015) ** 2)  # positive at every mode
        pick = np.sort(
            rng.choice(np.arange(1, 23), size=size, replace=False)
        )  # data bins on the model's modes
        rows[z] = (modes, p)
        k_rows.append(modes[pick])
        m_rows.append(p[pick])
    m = np.concatenate(m_rows)
    n = m.size
    a = rng.normal(size=(n, n))
    c0 = a @ a.T / n + 0.2 * np.eye(n)  # positive definite, correlated between every pair of bins
    s = np.diag(0.05 * m)
    cov = s @ c0 @ s
    cov = 0.5 * (cov + cov.T)
    return StubModel(rows), k_rows, m, cov


def test_the_gaussian_likelihood_has_gaussian_statistics_at_the_true_model():
    """2000 data sets d = m + noise of covariance C; at the true point: mean chi2 = N, var chi2 = 2N, mean ln L =
    -N/2 - ln det C / 2 - N/2 ln 2 pi, each within 4 standard errors; the model vector is m; ln det C and the chi2 of
    a few draws equal an independent computation."""
    stub, k_rows, m, cov = _statistics_setup()
    n = m.size
    assert n == 50 and not np.allclose(cov, np.diag(np.diag(cov)))
    chol = np.linalg.cholesky(cov)
    noise = (chol @ np.random.default_rng(7).standard_normal((n, N_DRAWS))).T
    logdet = float(np.linalg.slogdet(cov)[1])
    chi2, lnl = np.empty(N_DRAWS), np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        d = m + noise[i]
        like = L.GaussianP1DLikelihood(stub, L.P1DData(z=Z_STATS, k=tuple(k_rows), p1d=d, cov=cov))
        if i == 0:
            np.testing.assert_allclose(like.model_vector(THETA), m, rtol=1e-14, atol=0)
            assert like.logdet == pytest.approx(logdet, rel=1e-12, abs=1e-10)
        chi2[i], lnl[i] = like.chi2(THETA), like.loglike(THETA)
        if i < 5:  # the quadratic form and the normalisation, recomputed without the package
            r = d - m
            assert chi2[i] == pytest.approx(float(r @ np.linalg.solve(cov, r)), rel=1e-10)
            want = -0.5 * chi2[i] - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
            assert lnl[i] == pytest.approx(want, rel=1e-12, abs=1e-9)
    se_mean = math.sqrt(2.0 * n / N_DRAWS)
    assert abs(chi2.mean() - n) <= 4.0 * se_mean, (chi2.mean(), n, se_mean)
    var = chi2.var(ddof=1)
    m4 = np.mean((chi2 - chi2.mean()) ** 4)
    se_var = math.sqrt(
        (m4 - var**2) / N_DRAWS
    )  # the standard error of a sample variance, estimated
    assert abs(var - 2.0 * n) <= 4.0 * se_var, (var, 2 * n, se_var)
    want_lnl = -0.5 * n - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
    assert abs(lnl.mean() - want_lnl) <= 4.0 * 0.5 * se_mean, (lnl.mean(), want_lnl, 0.5 * se_mean)
    assert stub.calls == 2 * N_DRAWS + 1  # every value came through the likelihood's own path


# --------------------------------------------------------------------------------------- the interpolation
def test_interpolation_is_exact_for_a_line_and_has_the_analytic_error_for_a_quadratic():
    """On the s/km modes at z = 3 (midpoint): a + b k is reproduced to 1e-14 relative at seeded bin centres; for
    a + b k^2 the error at a centre is b (k - k_n)(k_(n+1) - k), with its sign, to 1e-12 of the spectrum.
    """
    modes = k_skm_independent(3.0, MIDPOINT["hub"], MIDPOINT["omegamh2"])
    rng = np.random.default_rng(5)
    k_data = np.concatenate([rng.uniform(modes[0], modes[-1], 400), modes[[0, 7, 100, -1]]])
    a, b = 2.5, 40.0
    got = L.interp_to_bins(modes, a + b * modes, k_data)
    want = a + b * k_data
    assert np.max(np.abs(got / want - 1.0)) <= 1e-14
    a, b = 1.0, 2000.0
    p_true = a + b * k_data**2
    err = L.interp_to_bins(modes, a + b * modes**2, k_data) - p_true
    j = np.clip(np.searchsorted(modes, k_data, side="right") - 1, 0, modes.size - 2)
    analytic = b * (k_data - modes[j]) * (modes[j + 1] - k_data)
    assert np.all(np.abs(err - analytic) <= 1e-12 * p_true)
    clear = (
        analytic > 1e-9 * p_true
    )  # well above rounding: the sign is that of b (the line lies above a convex P)
    assert clear.sum() > 350 and np.all(np.sign(err[clear]) == np.sign(b))
