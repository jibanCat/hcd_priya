"""A Gaussian likelihood of a P1D data vector under the per-class predictor.

This module joins three things that the package already has into one number, the log likelihood:
a model loaded with ``hcd_analysis.predictor`` (``load_release``, ``load_published``), PRIYA's
two-parameter model of the mean flux (``predictor.tau_eff_priya``), and the mixture of the class
spectra (``predictor.add_hcd``). It adds a container for a data set (``P1DData``), a rule for
bringing the model to the data's wavenumbers, and the Gaussian itself.

**The mathematics.** For a point ``theta`` (the nine PRIYA parameters, ``tau0_amp`` and
``dtau0``) and a data set with redshifts z_i, wavenumbers k_ij, values d and covariance C:

1. The mean flux at every redshift is PRIYA's model:
   ``tau_eff(z) = tau0_amp * ((1 + z) / (1 + z_pivot))**dtau0 * tau_Kim(z)``.
2. The predictor gives the class spectra on its native modes at each z_i. They are mixed into
   one spectrum, ``P = P_clean + sum_c A_c (P_c - P_clean)``, with the model's own class fractions
   (``A = pred.w_c``, the default) or with amplitudes that you fix.
3. The model's wavenumbers at z_i depend on ``hub`` and ``omegamh2`` of the point (the predictor
   returns them in s/km). The mixed spectrum is interpolated linearly in k to each k_ij. This is
   the value of the model AT the wavenumber of the data bin. It is not an average over the window
   of a bin, and no window or resolution function is applied here.
4. The result is multiplied by the data set's ``model_factor`` when it has one (for example a
   resolution table that the data vector also carries). That gives the model vector m(theta).
5. With r = m(theta) - d and N the length of d::

       ln L = -1/2 r^T C^-1 r  -  1/2 ln det C  -  N/2 ln(2 pi)

**What the covariance is.** Whatever the data set states: C is used as given. This module adds no
emulator error to it. A likelihood built this way treats the model as exact, so a posterior from
it is narrower than one that accounts for the error of the emulator.

**What is fixed when the likelihood is made, and what a call takes.** Fixed: the model, the data,
the covariance (factorised once), the fidelity, the treatment of the HCD amplitudes and any
Gaussian terms on parameters. A call takes ``theta`` only, as a mapping with exactly the eleven
names of ``param_names``.

**Outside the model's domain.** The predictor refuses a point that it cannot serve (a parameter
that is not finite, a mean flux below the tables of the multi-fidelity correction, and so on) by
raising ``predictor.PredictorInputError``. The methods here let that error through, so that a
refusal is never mistaken for a small likelihood. A sampler that needs a number for such a point
catches the error and returns minus infinity; the example script shows how. The flags of a
prediction (extrapolation, clamping, a mean flux off the trained range) are NOT turned into
refusals: read them from ``GaussianP1DLikelihood.prediction(theta).flags``.

Wavenumbers are angular, in s/km, and the P1D is in km/s: the predictor's convention
(``predictor.Prediction``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import predictor as _hp
from .emulator import safe_io as _S

__all__ = [
    "GaussianP1DLikelihood",
    "LikelihoodInputError",
    "OutsideModesError",
    "P1DData",
    "interp_to_bins",
    "read_p1d_data",
    "write_p1d_data",
]

MEAN_FLUX_NAMES = ("tau0_amp", "dtau0")
# The file format of a data set (``write_p1d_data``, ``read_p1d_data``): one strict JSON file with a closed set of
# keys, anchored by its sha256. The covariance is stored as one block per redshift.
DATA_FORMAT = "hcd_priya.p1d_data/1"
DATA_KEYS = ("format", "z", "k", "p1d", "cov_blocks", "model_factor", "truth", "provenance")
MAX_DATA_BYTES = 8 << 20
# A covariance is taken as symmetric when |C_ij - C_ji| <= SYMMETRY_RTOL * sqrt(C_ii C_jj) for every pair: the size
# of rounding in a matrix that was summed or inverted, far below anything that matters.
SYMMETRY_RTOL = 1e-10


class LikelihoodInputError(ValueError):
    """A data set, an option or a parameter point that the likelihood cannot use."""


class OutsideModesError(LikelihoodInputError):
    """A data wavenumber lies outside the wavenumbers of the model at this parameter point."""


def _frozen(a):
    out = np.array(a, dtype=np.float64)
    out.setflags(write=False)
    return out


def _vector(name, a, n=None):
    """A finite float64 vector of length ``n`` (any length when ``n`` is None)."""
    try:
        v = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError):
        raise LikelihoodInputError(f"{name} must be numbers") from None
    if v.ndim != 1 or (n is not None and v.shape != (n,)):
        want = "a vector" if n is None else f"a vector of length {n}"
        raise LikelihoodInputError(f"{name} must be {want}, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise LikelihoodInputError(f"{name} must be finite")
    return v


@dataclasses.dataclass(frozen=True, eq=False)
class P1DData:
    """One P1D data set: a data vector with its wavenumbers, redshifts and full covariance.

    The vector is ordered redshift by redshift, and inside a redshift by wavenumber, as ``k`` is
    given. The redshifts may have different numbers of wavenumbers. Make one with
    ``P1DData.from_blocks`` when the covariance comes as one block per redshift.

    Attributes
    ----------
    z : tuple of float
        The redshifts, shape (n_z,), distinct. Each must be a snapshot redshift of the model the
        data are used with.
    k : tuple of numpy.ndarray
        One array per redshift: the wavenumbers of its bins, angular, in s/km, strictly ascending.
    p1d : numpy.ndarray
        Shape (N,), N the total number of bins. The data vector in km/s.
    cov : numpy.ndarray
        Shape (N, N). The covariance of ``p1d``, symmetric and positive definite. It is kept as
        given. A difference between ``cov`` and its transpose at the level of rounding
        (``SYMMETRY_RTOL``) is accepted, and the likelihood then uses the symmetric part,
        ``(cov + cov.T) / 2``; a larger difference is refused.
    model_factor : numpy.ndarray or None
        Shape (N,). A factor by which the model vector is multiplied before it is compared with
        ``p1d`` (for example a resolution table that the data carry). None means no factor.
    truth : dict or None
        For a mock: the parameter point it was made at, by name. None for a measurement.
    provenance : dict
        What the data set is and where it came from. Free form; carried along, never interpreted.

    Raises
    ------
    LikelihoodInputError
        If the shapes do not fit, a number is not finite, the redshifts repeat, a row of
        wavenumbers is not strictly ascending and positive, a factor is not positive, or the
        covariance is not symmetric and positive definite.
    """

    z: tuple
    k: tuple
    p1d: np.ndarray
    cov: np.ndarray
    model_factor: np.ndarray | None = None
    truth: Mapping | None = None
    provenance: Mapping = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        z = _vector("z", self.z)
        if z.size == 0 or len(set(z.tolist())) != z.size:
            raise LikelihoodInputError("z must hold at least one redshift, each once")
        if not isinstance(self.k, (list, tuple)) or len(self.k) != z.size:
            raise LikelihoodInputError(f"k must be a sequence of {z.size} arrays, one per redshift")
        rows = []
        for zi, row in zip(z, self.k):
            r = _vector(f"k at z = {zi:g}", row)
            if r.size == 0 or np.any(r <= 0) or np.any(np.diff(r) <= 0):
                raise LikelihoodInputError(
                    f"k at z = {zi:g} must be positive and strictly ascending, with at least one bin"
                )
            rows.append(_frozen(r))
        n = sum(r.size for r in rows)
        p1d = _vector("p1d", self.p1d, n)
        try:
            cov = np.asarray(self.cov, dtype=np.float64)
        except (TypeError, ValueError):
            raise LikelihoodInputError("cov must be numbers") from None
        if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
            raise LikelihoodInputError(
                f"cov must be a finite matrix of shape ({n}, {n}), got shape {cov.shape}"
            )
        diag = np.diag(cov)
        if np.any(diag <= 0):
            raise LikelihoodInputError("cov must be positive definite (a variance is not positive)")
        scale = np.sqrt(np.outer(diag, diag))
        if np.any(np.abs(cov - cov.T) > SYMMETRY_RTOL * scale):
            raise LikelihoodInputError(
                "cov must be symmetric: cov and its transpose differ by more than rounding "
                f"({SYMMETRY_RTOL:g} of sqrt(C_ii C_jj))"
            )
        try:
            np.linalg.cholesky(0.5 * (cov + cov.T))
        except np.linalg.LinAlgError:
            raise LikelihoodInputError("cov must be positive definite") from None
        factor = None
        if self.model_factor is not None:
            factor = _vector("model_factor", self.model_factor, n)
            if np.any(factor <= 0):
                raise LikelihoodInputError("model_factor must be positive")
            factor = _frozen(factor)
        if self.truth is not None and not isinstance(self.truth, Mapping):
            raise LikelihoodInputError("truth must be a mapping of names to values, or None")
        if not isinstance(self.provenance, Mapping):
            raise LikelihoodInputError("provenance must be a mapping")
        object.__setattr__(self, "z", tuple(float(v) for v in z))
        object.__setattr__(self, "k", tuple(rows))
        object.__setattr__(self, "p1d", _frozen(p1d))
        object.__setattr__(self, "cov", _frozen(cov))
        object.__setattr__(self, "model_factor", factor)
        object.__setattr__(self, "truth", None if self.truth is None else dict(self.truth))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def sizes(self) -> tuple:
        """The number of bins at each redshift, shape (n_z,)."""
        return tuple(r.size for r in self.k)

    @property
    def n(self) -> int:
        """N, the length of the data vector."""
        return int(self.p1d.size)

    def slices(self) -> tuple:
        """One ``slice`` per redshift: where its bins lie in the data vector."""
        out, a = [], 0
        for m in self.sizes:
            out.append(slice(a, a + m))
            a += m
        return tuple(out)

    @classmethod
    def from_blocks(
        cls,
        z: Sequence[float],
        k: Sequence[ArrayLike],
        p1d: Sequence[ArrayLike],
        cov_blocks: Sequence[ArrayLike],
        *,
        model_factor: Sequence[ArrayLike] | None = None,
        truth: Mapping | None = None,
        provenance: Mapping | None = None,
    ) -> "P1DData":
        """Make a data set from one block per redshift.

        Parameters
        ----------
        z : sequence of float
            The redshifts, shape (n_z,).
        k, p1d : sequence of array_like
            One row per redshift: the wavenumbers in s/km and the data in km/s, each of the
            length of that redshift.
        cov_blocks : sequence of array_like
            One square matrix per redshift, the covariance of its bins. The full covariance is
            these blocks on the diagonal and zero between redshifts: the redshifts are taken as
            independent.
        model_factor : sequence of array_like, optional
            One row per redshift, as ``p1d``.
        truth, provenance : dict, optional
            As in the class.

        Returns
        -------
        P1DData

        Examples
        --------
        Two redshifts with two and three bins. Every number is made up:

        >>> import numpy as np
        >>> from hcd_analysis.likelihood import P1DData
        >>> data = P1DData.from_blocks(
        ...     z=[2.6, 3.0],
        ...     k=[[0.01, 0.02], [0.01, 0.02, 0.03]],
        ...     p1d=[[10.0, 8.0], [12.0, 9.0, 7.0]],
        ...     cov_blocks=[np.eye(2), 4.0 * np.eye(3)],
        ... )
        >>> print("bins per redshift:", data.sizes, "| length of the data vector:", data.n)
        bins per redshift: (2, 3) | length of the data vector: 5
        >>> print("the covariance between the two redshifts is zero:", bool(np.all(data.cov[:2, 2:] == 0)))
        the covariance between the two redshifts is zero: True
        """
        n_z = len(z)
        for name, rows in (("k", k), ("p1d", p1d), ("cov_blocks", cov_blocks)):
            if len(rows) != n_z:
                raise LikelihoodInputError(f"{name} must have one entry per redshift ({n_z})")
        if model_factor is not None and len(model_factor) != n_z:
            raise LikelihoodInputError(f"model_factor must have one entry per redshift ({n_z})")
        k_rows = [np.asarray(r, dtype=np.float64) for r in k]
        sizes = [r.size for r in k_rows]
        n = int(sum(sizes))
        cov = np.zeros((n, n))
        a = 0
        for zi, m, block in zip(z, sizes, cov_blocks):
            b = np.asarray(block, dtype=np.float64)
            if b.shape != (m, m):
                raise LikelihoodInputError(
                    f"the covariance block at z = {zi:g} must have shape ({m}, {m}), got {b.shape}"
                )
            cov[a : a + m, a : a + m] = b
            a += m

        def flat(name, rows):
            parts = [np.asarray(r, dtype=np.float64) for r in rows]
            for zi, m, p in zip(z, sizes, parts):
                if p.shape != (m,):
                    raise LikelihoodInputError(
                        f"{name} at z = {zi:g} must have shape ({m},), got {p.shape}"
                    )
            return np.concatenate(parts)

        return cls(
            z=tuple(z),
            k=tuple(k_rows),
            p1d=flat("p1d", p1d),
            cov=cov,
            model_factor=None if model_factor is None else flat("model_factor", model_factor),
            truth=truth,
            provenance={} if provenance is None else provenance,
        )


def interp_to_bins(k_model: ArrayLike, p_model: ArrayLike, k_data: ArrayLike) -> np.ndarray:
    """Bring a model spectrum to the wavenumbers of data bins, by linear interpolation in k.

    The result is the value of the model at each ``k_data``, read off the straight line between
    the two neighbouring model wavenumbers. It is not an average over a bin.

    Parameters
    ----------
    k_model : array_like
        Shape (K,). The model's wavenumbers, strictly ascending.
    p_model : array_like
        Shape (K,). The model's spectrum at ``k_model``.
    k_data : array_like
        Shape (M,). The wavenumbers of the data bins, in the units of ``k_model``.

    Returns
    -------
    numpy.ndarray
        Shape (M,).

    Raises
    ------
    OutsideModesError
        If a data wavenumber lies outside the model's wavenumbers: nothing is extrapolated. It is a
        ``LikelihoodInputError``.

    Examples
    --------
    >>> from hcd_analysis.likelihood import interp_to_bins
    >>> out = interp_to_bins([0.0, 1.0, 2.0], [10.0, 20.0, 40.0], [0.5, 1.5])
    >>> print("halfway between the model's points:", out.tolist())
    halfway between the model's points: [15.0, 30.0]
    """
    km = _vector("k_model", k_model)
    pm = _vector("p_model", p_model, km.size)
    kd = _vector("k_data", k_data)
    if km.size < 2 or np.any(np.diff(km) <= 0):
        raise LikelihoodInputError("k_model must be strictly ascending, with at least two points")
    if kd.min() < km[0] or kd.max() > km[-1]:
        raise OutsideModesError(
            f"a data wavenumber lies outside the model's wavenumbers [{km[0]:.6g}, {km[-1]:.6g}] "
            f"(data from {kd.min():.6g} to {kd.max():.6g}): nothing is extrapolated"
        )
    return np.interp(kd, km, pm)


class GaussianP1DLikelihood:
    """The Gaussian likelihood of a ``P1DData`` under a loaded model.

    The module docstring gives the mathematics. Everything except the parameter point is fixed
    here, when the object is made.

    Parameters
    ----------
    emu : predictor.Bundle
        A model from ``predictor.load_release`` or ``predictor.load_published`` (or a research
        predictor of the same interface).
    data : P1DData
        The data set. Each of its redshifts must be a snapshot redshift (``predictor.Z_GRID``).
    fidelity : {None, "LF", "LF+MF"}, optional
        Passed to the predictor. None (the default) is the predictor's default: "LF+MF" when the
        model has a multi-fidelity record, "LF" otherwise.
    hcd : "model_fractions" or array_like, optional
        How the class spectra are mixed. "model_fractions" (the default) uses the model's own
        class fractions at each point, ``A = pred.w_c``: the HCD content then follows the nine
        parameters and is not a free parameter. An array of shape (3,) or (n_z, 3) fixes the
        amplitudes [A_LLS, A_subDLA, A_DLA] instead (``predictor.add_hcd``).
    gaussian_terms : dict, optional
        Gaussian terms on parameters, ``{name: (mean, standard deviation)}``, each name one of
        ``param_names``. They are added by ``logp``; ``loglike`` does not include them.

    Attributes
    ----------
    param_names : tuple of str
        The eleven names a point must have: the nine PRIYA parameters
        (``predictor.PARAM_NAMES``), then "tau0_amp" and "dtau0".
    data : P1DData
    logdet : float
        ln det C.
    refusals : dict
        Filled by ``sampler_logp``: ``{kind of refusal: [count, the first message of that kind]}``.

    Raises
    ------
    LikelihoodInputError
        If ``data`` is not a ``P1DData``, an option has a value that is not listed above, or a
        Gaussian term names an unknown parameter or has a standard deviation that is not positive.
    predictor.PredictorInputError
        If a redshift of the data is not a snapshot redshift, or fixed amplitudes are not valid.

    Notes
    -----
    No flat prior and no parameter box is applied here: those belong to the sampler. The
    covariance is the data set's, with no emulator error added.

    Examples
    --------
    A mock made from the model itself at one point, with a made-up covariance, and its likelihood
    there. ``emu`` is a loaded model and ``params`` the nine parameters by name:

    >>> import numpy as np
    >>> from hcd_analysis.likelihood import GaussianP1DLikelihood, P1DData
    >>> theta = dict(params, tau0_amp=1.0, dtau0=0.0)        # the eleven names of a point
    >>> pred = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0)
    >>> k = 0.5 * (pred.k[:-1] + pred.k[1:])                 # wavenumbers halfway between the model's modes
    >>> n = k.size
    >>> blank = P1DData.from_blocks(z=[3.0], k=[k], p1d=[np.ones(n)], cov_blocks=[np.eye(n)])
    >>> made_at = GaussianP1DLikelihood(emu, blank).model_vector(theta)
    >>> mock = P1DData.from_blocks(
    ...     z=[3.0], k=[k], p1d=[made_at], cov_blocks=[0.01 * np.eye(n)], truth=theta
    ... )
    >>> like = GaussianP1DLikelihood(emu, mock)
    >>> print("chi2 at the point the mock was made at:", like.chi2(theta))
    chi2 at the point the mock was made at: 0.0
    >>> other = dict(theta, tau0_amp=1.1)                    # a higher mean optical depth
    >>> print("the likelihood is lower away from that point:", like.loglike(other) < like.loglike(theta))
    the likelihood is lower away from that point: True
    >>> print("the last two names of a point:", like.param_names[-2:])
    the last two names of a point: ('tau0_amp', 'dtau0')
    """

    def __init__(
        self,
        emu,
        data: P1DData,
        *,
        fidelity: str | None = None,
        hcd: str | ArrayLike = "model_fractions",
        gaussian_terms: Mapping[str, Sequence[float]] | None = None,
    ):
        if not isinstance(data, P1DData):
            raise LikelihoodInputError("data must be a P1DData")
        if fidelity not in (None, "LF", "LF+MF"):
            raise LikelihoodInputError('fidelity must be None, "LF" or "LF+MF"')
        self.emu = emu
        self.data = data
        self.fidelity = fidelity
        self.param_names = tuple(_hp.PARAM_NAMES) + MEAN_FLUX_NAMES
        self._z = [_hp.snapshot_z(z) for z in data.z]
        if isinstance(hcd, str):
            if hcd != "model_fractions":
                raise LikelihoodInputError(
                    'hcd must be "model_fractions" or an array of amplitudes, shape (3,) or (n_z, 3)'
                )
            self._amplitudes = None
        else:
            self._amplitudes = _hp.check_amplitudes(hcd, self._z)
        terms = {}
        for name, pair in dict(gaussian_terms or {}).items():
            if name not in self.param_names:
                raise LikelihoodInputError(
                    f"a Gaussian term names {name!r}, which is not one of {self.param_names}"
                )
            mean, sd = (float(x) for x in _vector(f"the Gaussian term on {name}", pair, 2))
            if sd <= 0:
                raise LikelihoodInputError(
                    f"the Gaussian term on {name} needs a standard deviation above zero"
                )
            terms[name] = (mean, sd)
        self.gaussian_terms = terms
        # the symmetric part: a covariance may differ from its transpose by rounding (SYMMETRY_RTOL)
        self._chol = np.linalg.cholesky(0.5 * (data.cov + data.cov.T))
        self.logdet = float(2.0 * np.sum(np.log(np.diag(self._chol))))
        self._slices = data.slices()
        self.refusals = {}  # by sampler_logp: {kind of refusal: [count, the first message]}

    # ------------------------------------------------------------------------------- the point
    def _point(self, theta):
        if not isinstance(theta, Mapping):
            raise LikelihoodInputError(f"theta must be a mapping with the names {self.param_names}")
        missing = [n for n in self.param_names if n not in theta]
        extra = [n for n in theta if n not in self.param_names]
        if missing or extra:
            raise LikelihoodInputError(
                f"theta must have exactly the names {self.param_names} "
                f"(missing {missing}, not known {extra})"
            )
        params = {n: theta[n] for n in _hp.PARAM_NAMES}
        return params, theta["tau0_amp"], theta["dtau0"]

    def prediction(self, theta: Mapping[str, float]) -> "_hp.Prediction":
        """The predictor's output at ``theta``, at the redshifts of the data, on its native modes.

        Use it to read the flags of the point (``.flags``), its class fractions (``.w_c``) and
        everything else a ``Prediction`` carries.

        Parameters
        ----------
        theta : dict
            The eleven parameters by name (``param_names``).

        Returns
        -------
        predictor.Prediction
            A stack over the redshifts of the data.
        """
        params, tau0_amp, dtau0 = self._point(theta)
        return self.emu.predict_many(
            params, z=list(self._z), tau0_amp=tau0_amp, dtau0=dtau0, fidelity=self.fidelity
        )

    def model_vector(self, theta: Mapping[str, float]) -> np.ndarray:
        """The model vector m(theta): steps 1 to 4 of the module docstring.

        Parameters
        ----------
        theta : dict
            The eleven parameters by name (``param_names``).

        Returns
        -------
        numpy.ndarray
            Shape (N,), in the order of ``data.p1d``, in km/s.

        Raises
        ------
        LikelihoodInputError
            If ``theta`` does not have exactly the eleven names, or a data wavenumber lies
            outside the model's wavenumbers at this point.
        predictor.PredictorInputError
            If the predictor refuses the point.
        """
        pred = self.prediction(theta)
        amplitudes = pred.w_c if self._amplitudes is None else self._amplitudes
        mixed = _hp.add_hcd(pred, amplitudes)
        out = np.empty(self.data.n)
        for i, (sl, kd) in enumerate(zip(self._slices, self.data.k)):
            try:
                out[sl] = interp_to_bins(pred.k[i], mixed[i], kd)
            except OutsideModesError as exc:
                raise OutsideModesError(f"at z = {self._z[i]:g}: {exc}") from None
        if self.data.model_factor is not None:
            out *= self.data.model_factor
        return out

    def chi2(self, theta: Mapping[str, float]) -> float:
        """r^T C^-1 r, with r = m(theta) - d.

        Parameters
        ----------
        theta : dict
            The eleven parameters by name.

        Returns
        -------
        float
        """
        return self._chi2_of(self.model_vector(theta))

    def _chi2_of(self, model):
        white = np.linalg.solve(self._chol, model - self.data.p1d)
        return float(white @ white)

    def loglike(self, theta: Mapping[str, float]) -> float:
        """ln L = -1/2 r^T C^-1 r - 1/2 ln det C - N/2 ln(2 pi).

        The log of the Gaussian density of the data given the model, with its normalisation. No
        Gaussian term on a parameter is included (see ``logp``).

        Parameters
        ----------
        theta : dict
            The eleven parameters by name.

        Returns
        -------
        float
        """
        n = self.data.n
        return -0.5 * self.chi2(theta) - 0.5 * self.logdet - 0.5 * n * math.log(2.0 * math.pi)

    def log_gaussian_terms(self, theta: Mapping[str, float]) -> dict:
        """The log density of each Gaussian term on a parameter, by name.

        For a term (mean, sd) on a parameter x: ``-1/2 ((x - mean) / sd)**2 - ln(sd sqrt(2 pi))``.

        Parameters
        ----------
        theta : dict
            The eleven parameters by name.

        Returns
        -------
        dict
            ``{name: log density}``; empty when the likelihood was made without Gaussian terms.
        """
        self._point(theta)
        out = {}
        for name, (mean, sd) in self.gaussian_terms.items():
            x = float(theta[name])
            out[name] = -0.5 * ((x - mean) / sd) ** 2 - math.log(sd * math.sqrt(2.0 * math.pi))
        return out

    def logp(self, theta: Mapping[str, float]) -> float:
        """``loglike`` plus the Gaussian terms. No flat prior is included: that is the sampler's.

        Parameters
        ----------
        theta : dict
            The eleven parameters by name.

        Returns
        -------
        float
        """
        return self.loglike(theta) + sum(self.log_gaussian_terms(theta).values())

    def sampler_logp(self, theta: Mapping[str, float]) -> float:
        """``logp`` for a sampler: minus infinity at a point the model cannot serve.

        A sampler needs a number at every point it proposes. This returns ``logp(theta)``, and
        minus infinity where the predictor refuses the point (``predictor.PredictorInputError``) or
        a data wavenumber lies outside the model's wavenumbers there (``OutsideModesError``). Each
        such refusal is counted in ``refusals``, by the kind of the error, with the first message
        of that kind, so that a chain can say how often and why it was turned back. A point with
        missing or unknown names still raises: that is an error of the caller, not of the point.

        Parameters
        ----------
        theta : dict
            The eleven parameters by name.

        Returns
        -------
        float
            ``logp(theta)``, or ``-inf``.
        """
        self._point(theta)
        try:
            return self.logp(theta)
        except (_hp.PredictorInputError, OutsideModesError) as exc:
            kind = type(exc).__name__
            entry = self.refusals.setdefault(kind, [0, str(exc)])
            entry[0] += 1
            return -math.inf

    def predictive(self, thetas: Sequence[Mapping[str, float]], rng: np.random.Generator):
        """Model vectors at a set of points, and one noisy replicate of the data for each.

        With points drawn from a posterior, the rows of the first array are the posterior of the
        model vector, and the rows of the second are draws from the posterior predictive: each is
        a model vector plus noise drawn from the data covariance.

        Parameters
        ----------
        thetas : sequence of dict
            The points, each with the eleven names.
        rng : numpy.random.Generator
            The random generator the noise is drawn with.

        Returns
        -------
        models : numpy.ndarray
            Shape (n, N).
        replicates : numpy.ndarray
            Shape (n, N): ``models`` plus noise of covariance C.
        """
        if not isinstance(rng, np.random.Generator):
            raise LikelihoodInputError("rng must be a numpy.random.Generator")
        models = np.array([self.model_vector(t) for t in thetas]).reshape(-1, self.data.n)
        noise = rng.standard_normal(models.shape) @ self._chol.T
        return models, models + noise


# ------------------------------------------------------------------------------------------ a data set as a file
def _data_bytes(data: P1DData) -> bytes:
    """The bytes of ``data`` in the file format ``DATA_FORMAT``."""
    if not isinstance(data, P1DData):
        raise LikelihoodInputError("data must be a P1DData")
    slices = data.slices()
    off = np.array(data.cov, copy=True)
    for sl in slices:
        off[sl, sl] = 0.0
    if np.any(off != 0.0):
        raise LikelihoodInputError(
            f"the file format {DATA_FORMAT} holds one covariance block per redshift; this data set has a "
            "covariance between redshifts"
        )
    doc = {
        "format": DATA_FORMAT,
        "z": list(data.z),
        "k": [r.tolist() for r in data.k],
        "p1d": [data.p1d[sl].tolist() for sl in slices],
        "cov_blocks": [data.cov[sl, sl].tolist() for sl in slices],
        "model_factor": (
            None if data.model_factor is None else [data.model_factor[sl].tolist() for sl in slices]
        ),
        "truth": data.truth,
        "provenance": data.provenance,
    }
    try:
        return json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LikelihoodInputError(
            f"truth and provenance must hold plain JSON values (text, finite numbers, lists, mappings): {exc}"
        ) from None


def write_p1d_data(path: str | os.PathLike, data: P1DData) -> str:
    """Write a data set as one JSON file and return the file's sha256.

    The file is strict UTF-8 JSON in the format ``DATA_FORMAT``: the redshifts, and per redshift
    the wavenumbers, the data, the covariance block and the model factor; then ``truth`` and
    ``provenance``. Numbers are written so that reading them back gives the same float64 values.
    ``read_p1d_data`` reads such a file only against its sha256.

    Parameters
    ----------
    path : str or os.PathLike
        The file to create. An existing file is not overwritten.
    data : P1DData
        The data set. Its covariance must be zero between redshifts: the format stores one block
        per redshift.

    Returns
    -------
    str
        The sha256 of the file, 64 hexadecimal digits.

    Raises
    ------
    LikelihoodInputError
        If the covariance couples redshifts, ``truth`` or ``provenance`` holds a value that is not
        plain JSON, or the file exists or cannot be written.
    """
    raw = _data_bytes(data)
    try:
        return _S._write_exclusive(os.fspath(path), [raw])
    except (_S.SafeFormatError, OSError) as exc:
        raise LikelihoodInputError(f"{path}: the data set was not written ({exc})") from None


def _parse_p1d_data(raw: bytes, where: str, expected_sha256: str) -> P1DData:
    if type(expected_sha256) is not str or not _S.exact_hex64(expected_sha256):
        raise LikelihoodInputError(
            f"{where}: a data set is read against its sha256, 64 lowercase hexadecimal digits"
        )
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise LikelihoodInputError(
            f"{where}: sha256 {sha[:12]}... is not the expected {expected_sha256[:12]}...; the file was "
            "not parsed"
        )
    try:
        doc = _S._json_loads(raw, where)
    except _S.SafeFormatError as exc:
        raise LikelihoodInputError(str(exc)) from None
    if not isinstance(doc, dict) or set(doc) != set(DATA_KEYS):
        got = set(doc) if isinstance(doc, dict) else set()
        raise LikelihoodInputError(
            f"{where}: keys must be exactly {sorted(DATA_KEYS)} (missing "
            f"{sorted(set(DATA_KEYS) - got)}, unknown {sorted(got - set(DATA_KEYS))})"
        )
    if doc["format"] != DATA_FORMAT:
        raise LikelihoodInputError(
            f"{where}: format {doc['format']!r}; this code reads {DATA_FORMAT!r}"
        )
    for key in ("z", "k", "p1d", "cov_blocks"):
        if not isinstance(doc[key], list):
            raise LikelihoodInputError(f"{where}: {key} must be a list")
    if doc["model_factor"] is not None and not isinstance(doc["model_factor"], list):
        raise LikelihoodInputError(f"{where}: model_factor must be a list or null")
    for key in ("truth", "provenance"):
        if doc[key] is not None and not isinstance(doc[key], dict):
            raise LikelihoodInputError(f"{where}: {key} must be an object")
    if doc["provenance"] is None:
        raise LikelihoodInputError(f"{where}: provenance must be an object")
    try:
        return P1DData.from_blocks(
            z=doc["z"],
            k=doc["k"],
            p1d=doc["p1d"],
            cov_blocks=doc["cov_blocks"],
            model_factor=doc["model_factor"],
            truth=doc["truth"],
            provenance=dict(doc["provenance"], file_sha256=sha),
        )
    except LikelihoodInputError as exc:
        raise LikelihoodInputError(f"{where}: {exc}") from None
    except (TypeError, ValueError) as exc:  # a ragged or non-numeric list
        raise LikelihoodInputError(f"{where}: its arrays are not well formed ({exc})") from None


def read_p1d_data(path: str | os.PathLike, *, expected_sha256: str) -> P1DData:
    """Read a data set written by ``write_p1d_data``, against its sha256.

    The file's bytes are hashed before anything is parsed, and a file with another sha256 is
    refused. Then the file must be strict JSON (no duplicate key, no NaN or Infinity) with exactly
    the keys of the format, and its arrays must make a valid ``P1DData``.

    Parameters
    ----------
    path : str or os.PathLike
        The file. It must be a regular file, not a symbolic link.
    expected_sha256 : str
        The sha256 the file must have: 64 lowercase hexadecimal digits.

    Returns
    -------
    P1DData
        Its ``provenance`` is the file's, with one entry added: ``file_sha256``.

    Raises
    ------
    LikelihoodInputError
        If the file cannot be read, has another sha256, or is not a data set of this format.
    """
    try:
        raw = _S.read_regular(os.fspath(path), MAX_DATA_BYTES)
    except (_S.SafeFormatError, OSError) as exc:
        raise LikelihoodInputError(f"{path}: not a readable data set file ({exc})") from None
    return _parse_p1d_data(raw, os.fspath(path), expected_sha256)
