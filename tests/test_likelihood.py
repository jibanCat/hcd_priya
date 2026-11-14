"""The Gaussian P1D likelihood (hcd_analysis/likelihood.py): the data container, the rule that brings the model to
the data's wavenumbers, and the Gaussian. Synthetic values only: a synthetic model with untrained networks, and data
vectors and covariances that are made up here."""

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import hcd_analysis.predictor as hp
from hcd_analysis import likelihood as L
from tests.synthetic_release import make_release
from tests.test_predictor_mf import PARAMS

E = L.LikelihoodInputError
ZS = (2.6, 3.0)  # two snapshot redshifts that the synthetic model serves under LF+MF
THETA = dict(PARAMS, tau0_amp=1.0, dtau0=0.0)


@pytest.fixture(scope="module")
def emu(tmp_path_factory):
    info = make_release(tmp_path_factory.mktemp("likelihood") / "release", incidence=True)
    return hp.load_release(str(info["root"]), info["sha"])


def _bins(emu, n=(4, 6)):
    """Data wavenumbers well inside the model's modes at THETA: n[i] points at redshift i, between modes."""
    pred = emu.predict_many(PARAMS, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    rows = []
    for i, m in enumerate(n):
        k = np.asarray(pred.k[i])
        lo, hi = k[len(k) // 4], k[len(k) // 2]
        rows.append(np.linspace(lo, hi, m + 2)[1:-1])  # strictly inside, not on a mode
    return rows


def _cov(m, scale, rho=0.3):
    """A made-up covariance block: positive definite, with correlations between neighbouring bins."""
    i = np.arange(m)
    return scale * rho ** np.abs(i[:, None] - i[None, :])


def _data(emu, **over):
    rows = _bins(emu)
    kw = dict(
        z=ZS,
        k=rows,
        p1d=[np.linspace(3.0, 2.0, r.size) for r in rows],
        cov_blocks=[_cov(rows[0].size, 0.04), _cov(rows[1].size, 0.09)],
    )
    kw.update(over)
    return L.P1DData.from_blocks(**kw)


# ------------------------------------------------------------------------------------------- the container
def test_a_data_set_from_blocks_is_ordered_redshift_by_redshift_and_read_only():
    d = L.P1DData.from_blocks(
        z=[2.6, 3.0],
        k=[[0.01, 0.02], [0.01, 0.02, 0.03]],
        p1d=[[10.0, 8.0], [12.0, 9.0, 7.0]],
        cov_blocks=[np.eye(2), 4.0 * np.eye(3)],
        model_factor=[[1.0, 1.0], [1.0, 0.5, 0.25]],
        truth={"ns": 0.9},
        provenance={"what": "made up"},
    )
    assert d.sizes == (2, 3) and d.n == 5 and d.z == (2.6, 3.0)
    assert d.slices() == (slice(0, 2), slice(2, 5))
    np.testing.assert_array_equal(d.p1d, [10.0, 8.0, 12.0, 9.0, 7.0])
    np.testing.assert_array_equal(d.model_factor, [1.0, 1.0, 1.0, 0.5, 0.25])
    assert np.all(d.cov[:2, 2:] == 0) and d.cov[4, 4] == 4.0  # independent redshifts
    assert d.truth == {"ns": 0.9} and d.provenance == {"what": "made up"}
    for a in (d.p1d, d.cov, d.model_factor, d.k[0], d.k[1]):
        with pytest.raises(ValueError):
            a[...] = 0.0
    with pytest.raises(Exception):  # frozen: a field cannot be replaced
        d.p1d = np.zeros(5)


@pytest.mark.parametrize(
    "change, why",
    [
        (dict(z=[2.6, 2.6]), "each once"),
        (dict(k=[[0.02, 0.01], [0.01, 0.02, 0.03]]), "strictly ascending"),
        (dict(k=[[0.0, 0.01], [0.01, 0.02, 0.03]]), "positive"),
        (dict(p1d=[[10.0], [12.0, 9.0, 7.0]]), "p1d at z = 2.6 must have shape \\(2,\\)"),
        (dict(p1d=[[10.0, np.nan], [12.0, 9.0, 7.0]]), "p1d must be finite"),
        (dict(cov_blocks=[np.eye(3), np.eye(3)]), "covariance block at z = 2.6 must have shape"),
        (dict(cov_blocks=[np.array([[1.0, 0.5], [0.4, 1.0]]), np.eye(3)]), "must be symmetric"),
        (
            dict(cov_blocks=[np.array([[1.0, 0.0], [0.0, -1.0]]), np.eye(3)]),
            "a variance is not positive",
        ),
        (dict(cov_blocks=[np.array([[1.0, 2.0], [2.0, 1.0]]), np.eye(3)]), "positive definite"),
        (dict(model_factor=[[1.0, 0.0], [1.0, 1.0, 1.0]]), "model_factor must be positive"),
        (dict(cov_blocks=[np.eye(2)]), "one entry per redshift"),
    ],
)
def test_a_malformed_data_set_is_refused(change, why):
    kw = dict(
        z=[2.6, 3.0],
        k=[[0.01, 0.02], [0.01, 0.02, 0.03]],
        p1d=[[10.0, 8.0], [12.0, 9.0, 7.0]],
        cov_blocks=[np.eye(2), np.eye(3)],
    )
    kw.update(change)
    with pytest.raises(E, match=why):
        L.P1DData.from_blocks(**kw)


def test_a_covariance_that_is_symmetric_up_to_rounding_is_kept_as_given(emu):
    """A matrix that was summed or inverted differs from its transpose in the last digits. It is accepted and stored
    as given, the likelihood uses its symmetric part, and a larger difference is refused."""
    rows = _bins(emu)
    exact = [_cov(rows[0].size, 0.04), _cov(rows[1].size, 0.09)]
    rounded = [b.copy() for b in exact]
    rounded[0][0, 1] *= 1.0 + 4e-14  # the transpose keeps the other value
    d = _data(emu, cov_blocks=rounded)
    assert d.cov[0, 1] != d.cov[1, 0] and d.cov[0, 1] == rounded[0][0, 1]  # kept as given
    sym = _data(emu, cov_blocks=[0.5 * (b + b.T) for b in rounded])
    a, b = L.GaussianP1DLikelihood(emu, d), L.GaussianP1DLikelihood(emu, sym)
    assert a.loglike(THETA) == b.loglike(THETA) and a.logdet == b.logdet
    bad = [b.copy() for b in exact]
    bad[0][0, 1] *= 1.0 + 1e-6
    with pytest.raises(E, match="differ by more than rounding"):
        _data(emu, cov_blocks=bad)


def test_a_full_covariance_may_couple_redshifts():
    c = np.eye(3) + 0.2
    d = L.P1DData(z=(2.6, 3.0), k=([0.01], [0.01, 0.02]), p1d=[1.0, 2.0, 3.0], cov=c)
    np.testing.assert_array_equal(d.cov, c)
    with pytest.raises(E, match="shape \\(3, 3\\)"):
        L.P1DData(z=(2.6, 3.0), k=([0.01], [0.01, 0.02]), p1d=[1.0, 2.0, 3.0], cov=np.eye(2))


# ------------------------------------------------------------------------------------------- the binning
def test_interpolation_is_exact_for_a_line_and_never_extrapolates():
    k = np.array([0.001, 0.004, 0.010, 0.030])
    line = 5.0 - 40.0 * k
    at = np.array([0.001, 0.002, 0.02, 0.030])
    np.testing.assert_allclose(L.interp_to_bins(k, line, at), 5.0 - 40.0 * at, rtol=1e-14)
    for bad in ([0.0005], [0.031]):
        with pytest.raises(E, match="nothing is extrapolated"):
            L.interp_to_bins(k, line, bad)
    with pytest.raises(E, match="strictly ascending"):
        L.interp_to_bins(k[::-1], line, at)


# ----------------------------------------------------------------------------------------- the likelihood
def test_the_model_vector_is_the_mixed_spectrum_at_the_data_wavenumbers_times_the_factor(emu):
    rows = _bins(emu)
    factor = [np.linspace(1.0, 0.8, r.size) for r in rows]
    data = _data(emu, model_factor=factor)
    like = L.GaussianP1DLikelihood(emu, data)
    assert like.param_names == hp.PARAM_NAMES + ("tau0_amp", "dtau0")
    pred = emu.predict_many(PARAMS, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    mixed = hp.add_hcd(pred, pred.w_c)
    want = np.concatenate(
        [np.interp(rows[i], pred.k[i], mixed[i]) * factor[i] for i in range(len(ZS))]
    )
    got = like.model_vector(THETA)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(
        got, like.model_vector(dict(THETA))
    )  # the same call, the same numbers
    # without the factor the vector is the plain interpolation
    plain = L.GaussianP1DLikelihood(emu, _data(emu)).model_vector(THETA)
    np.testing.assert_allclose(plain * np.concatenate(factor), got, rtol=1e-13, atol=0)
    assert like.prediction(THETA).flags == pred.flags


def test_fixed_amplitudes_replace_the_models_fractions(emu):
    data = _data(emu)
    rows = _bins(emu)
    pred = emu.predict_many(PARAMS, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    for amplitudes in ([0.1, 0.05, 0.02], [[0.1, 0.05, 0.02], [0.0, 0.0, 0.0]]):
        like = L.GaussianP1DLikelihood(emu, data, hcd=amplitudes)
        mixed = hp.add_hcd(pred, amplitudes)
        want = np.concatenate([np.interp(rows[i], pred.k[i], mixed[i]) for i in range(len(ZS))])
        np.testing.assert_allclose(like.model_vector(THETA), want, rtol=1e-13, atol=0)
    with pytest.raises(E, match='hcd must be "model_fractions"'):
        L.GaussianP1DLikelihood(emu, data, hcd="free")
    with pytest.raises(hp.PredictorInputError):
        L.GaussianP1DLikelihood(emu, data, hcd=[0.9, 0.9, 0.9])  # more than all sightlines


def test_the_log_likelihood_is_the_gaussian_density_with_its_normalisation(emu):
    data = _data(emu)
    like = L.GaussianP1DLikelihood(emu, data)
    m = like.model_vector(THETA)
    want = multivariate_normal(mean=m, cov=data.cov).logpdf(data.p1d)
    assert like.loglike(THETA) == pytest.approx(want, rel=1e-12)
    r = m - data.p1d
    assert like.chi2(THETA) == pytest.approx(float(r @ np.linalg.solve(data.cov, r)), rel=1e-12)
    assert like.logdet == pytest.approx(np.linalg.slogdet(data.cov)[1], rel=1e-12)
    # redshift by redshift, with the blocks, is the same number
    parts = 0.0
    for sl in data.slices():
        parts += multivariate_normal(mean=m[sl], cov=data.cov[sl, sl]).logpdf(data.p1d[sl])
    assert like.loglike(THETA) == pytest.approx(parts, rel=1e-12)


def test_a_mock_made_at_a_point_has_zero_chi2_there_and_a_lower_likelihood_elsewhere(emu):
    like0 = L.GaussianP1DLikelihood(emu, _data(emu))
    rows = _bins(emu)
    truth = like0.model_vector(THETA)
    sl = like0.data.slices()
    mock = _data(emu, p1d=[truth[s] for s in sl], truth=THETA)
    like = L.GaussianP1DLikelihood(emu, mock)
    assert like.chi2(THETA) == 0.0 and mock.truth == THETA and len(rows) == len(sl)
    other = dict(THETA, tau0_amp=1.1)
    assert like.chi2(other) > 0.0 and like.loglike(other) < like.loglike(THETA)


def test_a_point_needs_exactly_the_eleven_names(emu):
    like = L.GaussianP1DLikelihood(emu, _data(emu))
    missing = {k: v for k, v in THETA.items() if k != "dtau0"}
    with pytest.raises(E, match="missing \\['dtau0'\\]"):
        like.loglike(missing)
    with pytest.raises(E, match="not known \\['A_dla'\\]"):
        like.model_vector(dict(THETA, A_dla=0.1))
    with pytest.raises(E, match="theta must be a mapping"):
        like.chi2(list(THETA.values()))


def test_gaussian_terms_are_stated_log_densities_added_by_logp_only(emu):
    data = _data(emu)
    terms = {"hub": (0.70, 0.01), "bhfeedback": (0.05, 0.004)}
    like = L.GaussianP1DLikelihood(emu, data, gaussian_terms=terms)
    got = like.log_gaussian_terms(THETA)
    assert set(got) == set(terms)
    for name, (mean, sd) in terms.items():
        x = THETA[name]
        want = -0.5 * ((x - mean) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))
        assert got[name] == pytest.approx(want, rel=1e-14)
    plain = L.GaussianP1DLikelihood(emu, data)
    assert plain.log_gaussian_terms(THETA) == {} and plain.logp(THETA) == plain.loglike(THETA)
    assert like.loglike(THETA) == plain.loglike(THETA)  # loglike has no term in it
    assert like.logp(THETA) == pytest.approx(like.loglike(THETA) + sum(got.values()), rel=1e-14)
    with pytest.raises(E, match="not one of"):
        L.GaussianP1DLikelihood(emu, data, gaussian_terms={"A_dla": (0.0, 1.0)})
    with pytest.raises(E, match="standard deviation above zero"):
        L.GaussianP1DLikelihood(emu, data, gaussian_terms={"hub": (0.7, 0.0)})


def test_what_the_likelihood_refuses_when_it_is_made_or_called(emu):
    data = _data(emu)
    with pytest.raises(E, match="data must be a P1DData"):
        L.GaussianP1DLikelihood(emu, {"p1d": [1.0]})
    with pytest.raises(E, match="fidelity must be"):
        L.GaussianP1DLikelihood(emu, data, fidelity="HF")
    off_grid = L.P1DData.from_blocks(z=[3.1], k=[[0.01]], p1d=[[1.0]], cov_blocks=[np.eye(1)])
    with pytest.raises(hp.PredictorInputError, match="not a PRIYA snapshot redshift"):
        L.GaussianP1DLikelihood(emu, off_grid)
    # a data wavenumber beyond the model's modes: refused with the redshift named, never extrapolated
    pred = emu.predict_many(PARAMS, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    beyond = L.P1DData.from_blocks(
        z=ZS,
        k=[[float(pred.k[0][3])], [float(pred.k[1][-1]) * 2.0]],
        p1d=[[1.0], [1.0]],
        cov_blocks=[np.eye(1), np.eye(1)],
    )
    with pytest.raises(E, match="at z = 3: a data wavenumber lies outside"):
        L.GaussianP1DLikelihood(emu, beyond).loglike(THETA)
    # the predictor's own refusal of a point is let through, not turned into a number
    with pytest.raises(hp.PredictorInputError):
        L.GaussianP1DLikelihood(emu, data).loglike(dict(THETA, ns=float("nan")))


def test_a_sampler_gets_minus_infinity_where_the_model_refuses_and_a_count_of_why(emu):
    like = L.GaussianP1DLikelihood(emu, _data(emu))
    assert like.sampler_logp(THETA) == like.logp(THETA) and like.refusals == {}
    for _ in range(2):  # the predictor refuses a parameter that is not finite
        assert like.sampler_logp(dict(THETA, ns=float("nan"))) == -math.inf
    (kind,) = like.refusals
    assert like.refusals[kind][0] == 2 and isinstance(like.refusals[kind][1], str)
    assert issubclass(getattr(hp, kind), hp.PredictorInputError)
    # a data wavenumber beyond the modes of the point is a refusal of the point too
    pred = emu.predict_many(PARAMS, z=list(ZS), tau0_amp=1.0, dtau0=0.0)
    beyond = L.P1DData.from_blocks(
        z=ZS,
        k=[[float(pred.k[0][3])], [float(pred.k[1][-1]) * 2.0]],
        p1d=[[1.0], [1.0]],
        cov_blocks=[np.eye(1), np.eye(1)],
    )
    far = L.GaussianP1DLikelihood(emu, beyond)
    assert far.sampler_logp(THETA) == -math.inf and far.refusals["OutsideModesError"][0] == 1
    assert issubclass(L.OutsideModesError, L.LikelihoodInputError)
    with pytest.raises(
        E, match="missing"
    ):  # a malformed point is the caller's error, not a refusal
        like.sampler_logp({"ns": 0.9})


def test_the_predictive_is_the_model_plus_noise_of_the_data_covariance(emu):
    data = _data(emu)
    like = L.GaussianP1DLikelihood(emu, data)
    thetas = [THETA, dict(THETA, tau0_amp=1.05)]
    models, replicates = like.predictive(thetas, np.random.default_rng(7))
    assert models.shape == replicates.shape == (2, data.n)
    np.testing.assert_array_equal(models[0], like.model_vector(THETA))
    # the noise is a standard normal draw coloured by the Cholesky factor of the covariance
    white = np.random.default_rng(7).standard_normal((2, data.n))
    np.testing.assert_allclose(
        replicates - models, white @ np.linalg.cholesky(data.cov).T, rtol=1e-12, atol=1e-15
    )
    with pytest.raises(E, match="numpy.random.Generator"):
        like.predictive(thetas, 7)


def test_the_low_fidelity_likelihood_differs_and_uses_the_low_fidelity_prediction(emu):
    data = _data(emu)
    lf = L.GaussianP1DLikelihood(emu, data, fidelity="LF")
    mf = L.GaussianP1DLikelihood(emu, data)
    assert lf.prediction(THETA).flags["mf_applied"] is False
    assert mf.prediction(THETA).flags["mf_applied"] is True
    assert not np.array_equal(lf.model_vector(THETA), mf.model_vector(THETA))


# ------------------------------------------------------------------------------------- a data set as a file
def _made_up():
    return L.P1DData.from_blocks(
        z=[2.6, 3.0],
        k=[[0.01, 0.02], [0.01, 0.02, 0.03]],
        p1d=[[10.0, 1.0 / 3.0], [12.0, 9.0, 7.0]],
        cov_blocks=[_cov(2, 0.1), _cov(3, 0.7)],
        model_factor=[[1.0, 1.0], [1.0, 0.5, 0.25]],
        truth={"ns": 0.9, "tau0_amp": 1.0},
        provenance={"what": "made up", "inputs": ["a", "b"]},
    )


def test_a_data_set_round_trips_through_its_file_exactly(tmp_path):
    d = _made_up()
    path = tmp_path / "data.json"
    sha = L.write_p1d_data(path, d)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    back = L.read_p1d_data(path, expected_sha256=sha)
    assert back.z == d.z and back.sizes == d.sizes and back.truth == d.truth
    for name in ("p1d", "cov", "model_factor"):
        np.testing.assert_array_equal(
            getattr(back, name), getattr(d, name)
        )  # the same float64 values
    for a, b in zip(back.k, d.k):
        np.testing.assert_array_equal(a, b)
    assert back.provenance == dict(d.provenance, file_sha256=sha)
    with pytest.raises(E, match="was not written"):  # write once
        L.write_p1d_data(path, d)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == "hcd_priya.p1d_data/1" and sorted(doc) == sorted(L.DATA_KEYS)


def test_a_data_set_file_is_read_only_against_its_digest_and_only_if_strict(tmp_path):
    d = _made_up()
    path = tmp_path / "data.json"
    sha = L.write_p1d_data(path, d)
    with pytest.raises(E, match="not the expected"):
        L.read_p1d_data(path, expected_sha256="0" * 64)
    with pytest.raises(E, match="64 lowercase hexadecimal digits"):
        L.read_p1d_data(path, expected_sha256=None)
    with pytest.raises(E, match="not a readable data set file"):
        L.read_p1d_data(tmp_path / "absent.json", expected_sha256=sha)
    good = json.loads(path.read_text(encoding="utf-8"))

    def refused(doc_or_text, why):
        text = doc_or_text if isinstance(doc_or_text, str) else json.dumps(doc_or_text)
        p = tmp_path / f"bad_{len(list(tmp_path.iterdir()))}.json"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(E, match=why):
            L.read_p1d_data(p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    refused(dict(good, extra=1), "unknown \\['extra'\\]")
    refused({k: v for k, v in good.items() if k != "truth"}, "missing \\['truth'\\]")
    refused(dict(good, format="hcd_priya.p1d_data/2"), "this code reads")
    refused(dict(good, p1d=[[10.0], [12.0, 9.0, 7.0]]), "p1d at z = 2.6 must have shape")
    refused(
        dict(good, cov_blocks=[[[1.0, 2.0], [2.0, 1.0]], good["cov_blocks"][1]]),
        "positive definite",
    )
    refused(dict(good, provenance=None), "provenance must be an object")
    refused(json.dumps(good)[:-1] + ', "format": "hcd_priya.p1d_data/1"}', "duplicate")
    refused(json.dumps(good).replace("10.0", "NaN", 1), "not allowed")
    refused("not json", "data")


def test_a_covariance_between_redshifts_does_not_fit_the_file_format(tmp_path):
    coupled = L.P1DData(
        z=(2.6, 3.0), k=([0.01], [0.01, 0.02]), p1d=[1.0, 2.0, 3.0], cov=np.eye(3) + 0.2
    )
    with pytest.raises(E, match="one covariance block per redshift"):
        L.write_p1d_data(tmp_path / "coupled.json", coupled)
    assert not (tmp_path / "coupled.json").exists()
    with pytest.raises(E, match="plain JSON values"):
        L.write_p1d_data(
            tmp_path / "odd.json",
            L.P1DData(z=(2.6,), k=([0.01],), p1d=[1.0], cov=np.eye(1), provenance={"x": object()}),
        )


# ------------------------------------------------------------------------------------- the XQ-100 loader
def _xq100_stand_in(tmp_path, *, drop_row=False, header=None, asym=0.0):
    """Synthetic stand-ins of the two XQ-100 files, in their layout, with made-up numbers. Returns the two paths
    and their digests."""
    zs = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]
    ks = np.geomspace(0.003, 0.06, 13)
    rows = [header or L.XQ100_HEADER]
    for z in zs:
        for j, k in enumerate(ks):
            if drop_row and z == 3.6 and j == 5:
                continue
            k, p = float(k), float(10.0 * (1 + z) / (1 + 100 * k))
            rows.append(f"{k!r},{p!r},1.0,,3.0,,5,6,7,{z!r}")  # unused columns may be empty
    data = tmp_path / L.XQ100_FILES["data"][0]
    data.write_text("\n".join(rows) + "\n", encoding="utf-8")
    i = np.arange(65)
    cov = 0.04 * 0.5 ** np.abs(i[:, None] - i[None, :])
    cov[0, 1] *= 1.0 + asym
    covf = tmp_path / L.XQ100_FILES["covariance"][0]
    covf.write_text(
        "".join(f"{a} {b} {float(cov[a, b])!r}\n" for a in range(65) for b in range(65)),
        encoding="utf-8",
    )
    pins = {
        "data": hashlib.sha256(data.read_bytes()).hexdigest(),
        "covariance": hashlib.sha256(covf.read_bytes()).hexdigest(),
    }
    return data, covf, pins, cov


def test_the_xq100_loader_reads_five_redshifts_of_thirteen_bins_from_the_users_files(tmp_path):
    data, covf, pins, cov = _xq100_stand_in(tmp_path)
    full = L.load_xq100(data, covf, cross_redshift=True, expected_sha256=pins)
    assert full.z == (3.4, 3.6, 3.8, 4.0, 4.2) and full.sizes == (13,) * 5 and full.n == 65
    assert full.truth is None and full.model_factor is None
    np.testing.assert_array_equal(full.cov, cov)  # the matrix of the file, element by element
    np.testing.assert_allclose(full.k[0], np.geomspace(0.003, 0.06, 13), rtol=1e-15)
    assert full.p1d[0] == pytest.approx(
        10.0 * 4.4 / (1 + 100 * 0.003)
    )  # z 3.4 is the first redshift kept
    blocks = L.load_xq100(data, covf, cross_redshift=False, expected_sha256=pins)
    assert np.all(blocks.cov[:13, 13:] == 0) and np.array_equal(
        blocks.cov[13:26, 13:26], cov[13:26, 13:26]
    )
    assert (
        "set to zero" in blocks.provenance["covariance"]
        and "full matrix" in full.provenance["covariance"]
    )
    assert full.provenance["data_file"] == {
        "name": L.XQ100_FILES["data"][0],
        "sha256": pins["data"],
    }


def test_the_xq100_loader_pins_the_two_files_and_states_no_default_choice(tmp_path):
    data, covf, pins, _ = _xq100_stand_in(tmp_path)
    with pytest.raises(E, match="is not that of the XQ-100 data file this loader reads"):
        L.load_xq100(data, covf, cross_redshift=True)  # a stand-in is not the pinned file
    with pytest.raises(TypeError):
        L.load_xq100(data, covf)  # no default for the covariance between redshifts
    with pytest.raises(E, match="state the choice"):
        L.load_xq100(data, covf, cross_redshift=1, expected_sha256=pins)
    with pytest.raises(E, match="expected_sha256 must be"):
        L.load_xq100(data, covf, cross_redshift=True, expected_sha256={"data": pins["data"]})
    with pytest.raises(E, match="not a readable XQ-100 data file"):
        L.load_xq100(tmp_path / "absent.csv", covf, cross_redshift=True, expected_sha256=pins)


@pytest.mark.parametrize(
    "kw, why",
    [
        (dict(header="k,paa,z"), "the header is not"),
        (dict(drop_row=True), "z = 3.6 must have 13 consecutive rows"),
        (dict(asym=1e-6), "do not make a valid data set: cov must be symmetric"),
    ],
)
def test_xq100_files_of_another_layout_are_refused(tmp_path, kw, why):
    data, covf, pins, _ = _xq100_stand_in(tmp_path, **kw)
    with pytest.raises(E, match=why):
        L.load_xq100(data, covf, cross_redshift=True, expected_sha256=pins)


def test_an_xq100_covariance_file_must_give_every_element_once(tmp_path):
    data, covf, pins, _ = _xq100_stand_in(tmp_path)
    lines = covf.read_text(encoding="utf-8").splitlines()
    covf.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    pins["covariance"] = hashlib.sha256(covf.read_bytes()).hexdigest()
    with pytest.raises(E, match="expected 4225 lines"):
        L.load_xq100(data, covf, cross_redshift=True, expected_sha256=pins)
    covf.write_text(
        "\n".join(lines[:-1] + [lines[0]]) + "\n", encoding="utf-8"
    )  # one twice, one missing
    pins["covariance"] = hashlib.sha256(covf.read_bytes()).hexdigest()
    with pytest.raises(E, match="not every element"):
        L.load_xq100(data, covf, cross_redshift=True, expected_sha256=pins)
