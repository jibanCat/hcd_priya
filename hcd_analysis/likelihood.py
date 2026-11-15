"""A Gaussian likelihood of a P1D data vector under the per-class predictor.

This module joins three things that the package already has into one number, the log likelihood:
a model loaded with ``hcd_analysis.predictor`` (``load_release``, ``load_published``), PRIYA's
two-parameter model of the mean flux (``predictor.tau_eff_priya``), and the mixture of the class
spectra (``predictor.add_hcd``). It adds a container for a data set (``P1DData``), a rule for
bringing the model to the data's wavenumbers, and the Gaussian itself. Beside these it has a
loader for one public data set (``load_xq100``) and a container for a stored reference of a mock
recovery (``P1DReference``), each with what it does not do stated in its docstring.

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

**What the covariance is.** Whatever the data set states: C is used as given (when it differs
from its transpose by rounding, its symmetric part: see ``P1DData``). This module adds no
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
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_triangular

from . import predictor as _hp
from .emulator import safe_io as _S

__all__ = [
    "DiagnosisOnlyWarning",
    "GaussianP1DLikelihood",
    "LikelihoodInputError",
    "OutsideModesError",
    "P1DData",
    "P1DReference",
    "SAMPLER_REFUSALS",
    "interp_to_bins",
    "load_xq100",
    "read_p1d_data",
    "read_reference",
    "write_p1d_data",
    "write_reference",
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


def _numbers(name, a):
    """``a`` as a float64 array. Text and booleans would convert to float64; they are not numbers here."""
    try:
        given = np.asarray(a)
        if given.dtype.kind not in "iuf":
            raise TypeError
        return given.astype(np.float64)
    except (TypeError, ValueError, OverflowError):
        raise LikelihoodInputError(f"{name} must be numbers") from None


def _vector(name, a, n=None):
    """A finite float64 vector of length ``n`` (any length when ``n`` is None)."""
    v = _numbers(name, a)
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
        cov = _numbers("cov", self.cov)
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
        k_rows = [_numbers("k", r) for r in k]
        sizes = [r.size for r in k_rows]
        n = int(sum(sizes))
        cov = np.zeros((n, n))
        a = 0
        for zi, m, block in zip(z, sizes, cov_blocks):
            b = _numbers(f"the covariance block at z = {zi:g}", block)
            if b.shape != (m, m):
                raise LikelihoodInputError(
                    f"the covariance block at z = {zi:g} must have shape ({m}, {m}), got {b.shape}"
                )
            cov[a : a + m, a : a + m] = b
            a += m

        def flat(name, rows):
            parts = [_numbers(name, r) for r in rows]
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


# ---------------------------------------------------------------- from the model's modes to the data's wavenumbers
def interp_to_bins(k_model: ArrayLike, p_model: ArrayLike, k_data: ArrayLike) -> np.ndarray:
    """Bring a model spectrum to the wavenumbers of data bins: linear interpolation in k.

    **The algorithm.** For each data wavenumber k, take the two neighbouring model wavenumbers
    k_j <= k <= k_(j+1) and read the value off the straight line between them::

        P(k) = P_j + (P_(j+1) - P_j) * (k - k_j) / (k_(j+1) - k_j)

    That is all. It is the value of the model AT k: not an average over the width of a data bin,
    not a re-binning, and nothing in it is fitted or tuned. A data wavenumber outside
    [k_model[0], k_model[-1]] is refused, never extrapolated. ``docs/likelihood.md`` says when this
    rule is accurate (the model's modes are close together compared with the data bins) and when it
    is not.

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
        If a data wavenumber lies outside the model's wavenumbers. It is a ``LikelihoodInputError``.
    LikelihoodInputError
        If an input is not a finite vector, ``k_model`` is not strictly ascending, or the shapes
        do not fit.

    Examples
    --------
    >>> from hcd_analysis.likelihood import interp_to_bins
    >>> out = interp_to_bins([0.0, 1.0, 2.0], [10.0, 20.0, 40.0], [0.5, 1.5])
    >>> print("halfway between the model's points:", out.tolist())
    halfway between the model's points: [15.0, 30.0]
    """
    k_model, p_model, k_data = _interp_inputs(k_model, p_model, k_data)
    _refuse_outside_modes(k_model, k_data)
    # numpy.interp evaluates exactly the straight line of the docstring between each pair of neighbouring modes
    return np.interp(k_data, k_model, p_model)


def _interp_inputs(k_model, p_model, k_data):
    """The three inputs of ``interp_to_bins`` as finite float vectors, after the checks that make the line defined:
    at least two model wavenumbers, strictly ascending (so that every interval has a width), one model value per
    model wavenumber, and at least one data wavenumber."""
    km = _vector("k_model", k_model)
    pm = _vector("p_model", p_model, km.size)
    kd = _vector("k_data", k_data)
    if kd.size == 0:
        raise LikelihoodInputError("k_data must hold at least one wavenumber")
    if km.size < 2 or np.any(np.diff(km) <= 0):
        raise LikelihoodInputError("k_model must be strictly ascending, with at least two points")
    return km, pm, kd


def _refuse_outside_modes(k_model, k_data):
    """Refuse a data wavenumber below the first or above the last model wavenumber: there the model has no value,
    and extrapolating a line would invent one."""
    if k_data.min() < k_model[0] or k_data.max() > k_model[-1]:
        raise OutsideModesError(
            f"a data wavenumber lies outside the model's wavenumbers [{k_model[0]:.6g}, {k_model[-1]:.6g}] "
            f"(data from {k_data.min():.6g} to {k_data.max():.6g}): nothing is extrapolated"
        )


# --------------------------------------------------------------------------- when a sampler gets minus infinity
# Every condition under which ``GaussianP1DLikelihood.sampler_logp`` returns minus infinity at the defaults, by the
# kind of the error that ``refusals`` counts. Each is a refusal for correctness: there is no number to return.
SAMPLER_REFUSALS = {
    "PredictorInputError": (
        "The predictor refuses the point: a parameter is not finite (NaN or infinite); or the mean optical depth "
        "tau_eff(z) = tau0_amp ((1 + z) / (1 + z_pivot))**dtau0 tau_Kim(z) is not positive at a redshift of the "
        "data (tau0_amp <= 0); or, with hcd_relative_to_priya, the scaled class fractions sum to more than 1 at a "
        "redshift. Not relaxable: there is no prediction at such a point."
    ),
    "OutsideModesError": (
        "A wavenumber of the data lies outside the model's modes at the point. The modes are fixed in h/Mpc "
        "(2 pi n / 120 Mpc/h, n = 1, 2, ...); in s/km they move with hub and omegamh2, so a data bin near either "
        "end of the modes can fall outside them at some points of the box. Not relaxable: nothing is extrapolated."
    ),
}


class DiagnosisOnlyWarning(UserWarning):
    """A likelihood was made with a switch that changes the posterior, for diagnosis only (``refuse_flags``)."""


def _hcd_table(name: str, given, zs) -> np.ndarray:
    """The HCD values of ``hcd_fractions`` or ``hcd_relative_to_priya`` as an array of shape (n_z, 3), one row per
    redshift of the data in the class order LLS, subDLA, DLA.

    ``given`` must be a mapping that names exactly the three classes, so that no value can land on the wrong class
    by its position. Each value is one number (used at every redshift) or one number per redshift of the data.
    """
    classes = _hp.HCD_CLASSES
    if not isinstance(given, Mapping):
        raise LikelihoodInputError(
            f"{name} must be a mapping of the three classes, {{'LLS': ..., 'subDLA': ..., 'DLA': ...}}: a list or "
            "an array has no class names, so its order could be anything"
        )
    if set(given) != set(classes):
        raise LikelihoodInputError(
            f"{name} must name exactly the classes {classes} (missing {sorted(set(classes) - set(given))}, "
            f"unknown {sorted(set(given) - set(classes), key=str)})"
        )
    table = np.empty((len(zs), 3))
    for c, cls in enumerate(classes):
        values = _numbers(f"{name}[{cls!r}]", given[cls]).ravel()
        if values.size not in (1, len(zs)):
            raise LikelihoodInputError(
                f"{name}[{cls!r}] must be one number, or one per redshift of the data ({len(zs)}), got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise LikelihoodInputError(f"{name}[{cls!r}] must be finite")
        table[:, c] = values  # one number is used at every redshift
    return table


def _check_hcd_fractions(table, zs):
    """Absolute sightline fractions: each in [0, 1], and at each redshift their sum at most 1 (the rest of the
    sightlines is the clean forest)."""
    for i, z in enumerate(zs):
        for c, cls in enumerate(_hp.HCD_CLASSES):
            a = table[i, c]
            if not 0.0 <= a <= 1.0:
                raise LikelihoodInputError(
                    f"hcd_fractions[{cls!r}] = {a:g} at z = {z:g} is not in [0, 1]: it is the fraction of all "
                    "sightlines that belong to that class"
                )
        total = float(np.sum(table[i]))
        if total > 1.0:
            raise LikelihoodInputError(
                f"hcd_fractions sum to {total:g} at z = {z:g}: the fractions of all sightlines cannot exceed 1 (if "
                "you meant PRIYA's own fractions, give no HCD argument, or hcd_relative_to_priya with 1 for each "
                "class)"
            )


def _check_hcd_relative(table, zs):
    """Factors on PRIYA's own fractions: each at least 0 (1 is PRIYA's own, 0 is no sightline of that class)."""
    for i, z in enumerate(zs):
        for c, cls in enumerate(_hp.HCD_CLASSES):
            r = table[i, c]
            if r < 0.0:
                raise LikelihoodInputError(
                    f"hcd_relative_to_priya[{cls!r}] = {r:g} at z = {z:g}: a factor on PRIYA's own fraction is "
                    ">= 0 (1 = PRIYA's own, 0 = none of that class)"
                )


# ------------------------------------------------------------------------------------------- the likelihood
class GaussianP1DLikelihood:
    """The Gaussian DATA likelihood of a ``P1DData`` under a loaded model: C is the data set's covariance only.

    No emulator error is added to the covariance: this is a likelihood of the data given the model,
    with the model treated as exact, not a likelihood of "data + emulator". A posterior from it is
    narrower than one that accounts for the error of the emulator. ``docs/likelihood.md`` gives the
    mathematics step by step; in short, at a point theta::

        m(theta) = F * interp_k( P_clean + sum_c A_c (P_c - P_clean) )      (the model vector)
        ln L     = -1/2 r^T C^-1 r - 1/2 ln det C - N/2 ln(2 pi),  r = d - m(theta)

    with the mean flux from PRIYA's two-parameter model, A_c the class fractions (see the HCD
    arguments), F the data set's ``model_factor`` and C the symmetric part of the data covariance.
    Everything except the parameter point is fixed when the object is made.

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
    hcd_fractions : dict, optional
        ABSOLUTE class fractions: ``{"LLS": a, "subDLA": b, "DLA": c}``, each the fraction of all
        sightlines in that class, in [0, 1], with a sum of at most 1 at each redshift. 0 means no
        sightline of that class; 1 would mean every sightline. A value is one number or one number
        per redshift of the data. The HCD content is then fixed, the same at every point.
    hcd_relative_to_priya : dict, optional
        Class fractions RELATIVE to PRIYA's own at the point: ``{"LLS": r1, "subDLA": r2, "DLA":
        r3}``, and A_c = r_c times the model's own fraction ``pred.w_c`` of class c. 1 means
        PRIYA's own fraction, 0 means no sightline of that class, 2 twice PRIYA's own; each is at
        least 0. A value is one number or one number per redshift of the data. Give at most one of
        the two HCD arguments. With neither (the default), the model's own fractions are used,
        ``A = pred.w_c``: the HCD content follows the nine parameters and is not a free parameter.
    gaussian_terms : dict, optional
        Gaussian terms on parameters, ``{name: (mean, standard deviation)}``, each name one of
        ``param_names``. They are added by ``logp``; ``loglike`` does not include them.
    refuse_flags : tuple of str, optional
        For diagnosis only: names of prediction flags (``Prediction.flags``, for example
        "outside_sampling_box" or "mf_tau0_clamped") at which ``sampler_logp`` returns minus
        infinity. The default, an empty tuple, refuses nothing beyond ``SAMPLER_REFUSALS``. A
        non-empty tuple changes the posterior and emits ``DiagnosisOnlyWarning``.

    Attributes
    ----------
    param_names : tuple of str
        The eleven names a point must have: the nine PRIYA parameters
        (``predictor.PARAM_NAMES``), then "tau0_amp" and "dtau0".
    data : P1DData
    hcd_mode : str
        "priya" (the model's own fractions, the default), "fractions" or "relative_to_priya".
    logdet : float
        ln det C.
    refuse_flags : tuple of str
    refusals : dict
        Filled by ``sampler_logp``: ``{kind of refusal: [count, the first message of that kind]}``.

    Raises
    ------
    LikelihoodInputError
        If ``data`` is not a ``P1DData``; an option has a value that is not listed above; both HCD
        arguments are given, or one of them is not a mapping of the three classes with values in
        its range; or a Gaussian term names an unknown parameter or has a standard deviation that
        is not positive.
    predictor.PredictorInputError
        If a redshift of the data is not a snapshot redshift.

    Notes
    -----
    No flat prior and no parameter box is applied here: those belong to the sampler.

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
        hcd_fractions: Mapping[str, ArrayLike] | None = None,
        hcd_relative_to_priya: Mapping[str, ArrayLike] | None = None,
        gaussian_terms: Mapping[str, Sequence[float]] | None = None,
        refuse_flags: tuple = (),
    ):
        if not isinstance(data, P1DData):
            raise LikelihoodInputError("data must be a P1DData")
        if fidelity not in (None, "LF", "LF+MF"):
            raise LikelihoodInputError('fidelity must be None, "LF" or "LF+MF"')
        self.emu = emu
        self.data = data
        self.fidelity = fidelity
        self.param_names = tuple(_hp.PARAM_NAMES) + MEAN_FLUX_NAMES
        # each redshift of the data, matched to a snapshot of the model
        self._z = [_hp.snapshot_z(z) for z in data.z]
        if len(set(self._z)) != len(self._z):
            raise LikelihoodInputError(
                f"two redshifts of the data are the same snapshot redshift: {sorted(self._z)}"
            )
        self.hcd_mode, self._hcd = self._hcd_choice(hcd_fractions, hcd_relative_to_priya)
        self.gaussian_terms = self._gaussian_terms(gaussian_terms)
        self.refuse_flags = self._refuse_flags(refuse_flags)
        # the covariance, factorised once: C_sym = L L^T, with C_sym the symmetric part of the data's covariance
        # (it may differ from its transpose by rounding: SYMMETRY_RTOL)
        self._chol = np.linalg.cholesky(0.5 * (data.cov + data.cov.T))
        self.logdet = float(2.0 * np.sum(np.log(np.diag(self._chol))))
        self._slices = data.slices()
        self.refusals = {}  # by sampler_logp: {kind of refusal: [count, the first message]}

    # --------------------------------------------------------------------------- the options, checked once
    def _hcd_choice(self, fractions, relative):
        if fractions is not None and relative is not None:
            raise LikelihoodInputError("give hcd_fractions or hcd_relative_to_priya, not both")
        if fractions is not None:
            table = _hcd_table("hcd_fractions", fractions, self.data.z)
            _check_hcd_fractions(table, self.data.z)
            return "fractions", table
        if relative is not None:
            table = _hcd_table("hcd_relative_to_priya", relative, self.data.z)
            _check_hcd_relative(table, self.data.z)
            return "relative_to_priya", table
        return "priya", None

    def _gaussian_terms(self, given):
        terms = {}
        for name, pair in dict(given or {}).items():
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
        return terms

    def _refuse_flags(self, given):
        if not isinstance(given, tuple) or not all(type(f) is str and f for f in given):
            raise LikelihoodInputError(
                "refuse_flags must be a tuple of flag names (non-empty text)"
            )
        if given:
            warnings.warn(
                DiagnosisOnlyWarning(
                    f"refuse_flags={given}: a sampler gets minus infinity wherever the predictor sets any of these "
                    "flags. That changes the posterior, so it is for diagnosis only: for example a chain with and "
                    "one without, to see what the points at an edge contribute."
                ),
                stacklevel=3,
            )
        return given

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
        """The model vector m(theta): the prediction at theta, mixed, interpolated and scaled.

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
        return self._model_from(self.prediction(theta))

    def _model_from(self, pred):
        # step 1, the class fractions A_c at each redshift of the data
        if self.hcd_mode == "priya":
            amplitudes = pred.w_c
        elif self.hcd_mode == "fractions":
            amplitudes = self._hcd
        else:  # relative to PRIYA's own at this point
            amplitudes = self._hcd * pred.w_c
        # step 2, one spectrum per redshift: P_clean + sum_c A_c (P_c - P_clean), on the model's own modes
        mixed = _hp.add_hcd(pred, amplitudes)
        # step 3, that spectrum at the data's wavenumbers, redshift by redshift (linear in k: interp_to_bins)
        out = np.empty(self.data.n)
        for i, (sl, kd) in enumerate(zip(self._slices, self.data.k)):
            try:
                out[sl] = interp_to_bins(pred.k[i], mixed[i], kd)
            except OutsideModesError as exc:
                raise OutsideModesError(f"at z = {self._z[i]:g}: {exc}") from None
        # step 4, the data set's factor on the model vector (for example a resolution table), when it has one
        if self.data.model_factor is not None:
            out *= self.data.model_factor
        return out

    # --------------------------------------------------------------------------- the Gaussian
    def chi2(self, theta: Mapping[str, float]) -> float:
        """r^T C^-1 r, with r = d - m(theta).

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
        # with C = L L^T: r^T C^-1 r = |L^-1 r|^2, and L^-1 r is one triangular solve (no inverse is formed)
        white = solve_triangular(self._chol, model - self.data.p1d, lower=True)
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
        return self._loglike_of(self.model_vector(theta))

    def _loglike_of(self, model):
        n = self.data.n
        return -0.5 * self._chi2_of(model) - 0.5 * self.logdet - 0.5 * n * math.log(2.0 * math.pi)

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

    # ------------------------------------------------------------------------------- for a sampler
    def sampler_logp(self, theta: Mapping[str, float]) -> float:
        """``logp`` for a sampler: minus infinity at a point the model cannot serve.

        A sampler needs a number at every point it proposes. This returns ``logp(theta)``, or minus
        infinity, and counts every minus infinity in ``refusals`` by its kind, with the first
        message of that kind, so that a chain can say how often and why it was turned back.

        **When it returns minus infinity, at the defaults:** exactly the conditions of the module's
        ``SAMPLER_REFUSALS``, each a refusal for correctness (no number exists there):

        - ``PredictorInputError``: a parameter is not finite; or tau_eff(z) is not positive at a
          redshift of the data (tau0_amp <= 0); or, with ``hcd_relative_to_priya``, the scaled
          fractions sum to more than 1 at a redshift.
        - ``OutsideModesError``: a data wavenumber lies outside the model's modes at this point
          (they move with hub and omegamh2).

        **What is NOT minus infinity:** a point outside the box the model was trained on or the
        package's sampling box, on the n_s ridge, with a mean flux off the trained ladder, or where
        the multi-fidelity tables are clamped. The predictor serves such a point and sets a flag
        (``prediction(theta).flags``). The flat boxes of a chain belong to the sampler.

        **For diagnosis only:** with ``refuse_flags`` (given when the likelihood is made), a point
        at which the predictor sets any of the named flags is minus infinity too, counted as
        "flag:<name>". Comparing a chain with and one without shows what such points contribute.
        To see the effect of the multi-fidelity correction, make a second likelihood with
        ``fidelity="LF"``.

        **What still raises:** a point with missing or unknown names (an error of the caller); a
        fidelity the model cannot serve and a data redshift below the multi-fidelity correction
        (``predictor.MFUnavailableError``, ``predictor.MFRangeError``), which hold for every point,
        so minus infinity everywhere would hide the reason; a name in ``refuse_flags`` that is not
        a flag of a prediction.

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
            if not self.refuse_flags:
                return self.logp(theta)
            pred = self.prediction(theta)
            flagged = self._flags_set(pred)
            if flagged:
                return self._refused(
                    f"flag:{flagged[0]}", f"the predictor set the flag {flagged[0]} here"
                )
            return self._loglike_of(self._model_from(pred)) + sum(
                self.log_gaussian_terms(theta).values()
            )
        except (_hp.MFUnavailableError, _hp.MFRangeError):
            raise  # wrong for every point (no MF record; a data redshift below the correction): not a refusal
        except (_hp.PredictorInputError, OutsideModesError) as exc:
            return self._refused(type(exc).__name__, str(exc))

    def _flags_set(self, pred):
        """The names of ``refuse_flags`` that the prediction sets, in the order given."""
        unknown = [f for f in self.refuse_flags if f not in pred.flags]
        if unknown:
            raise LikelihoodInputError(
                f"refuse_flags names {unknown[0]!r}, which is not a flag of a prediction of this model "
                f"({sorted(pred.flags)})"
            )
        return [f for f in self.refuse_flags if np.any(pred.flags[f])]

    def _refused(self, kind, message):
        entry = self.refusals.setdefault(kind, [0, message])
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
    if len(raw) > MAX_DATA_BYTES:
        raise LikelihoodInputError(
            f"{path}: the data set takes {len(raw)} bytes as a file, above the {MAX_DATA_BYTES} that "
            "read_p1d_data reads; it was not written"
        )
    try:
        return _S._write_exclusive(os.fspath(path), [raw])
    except (_S.SafeFormatError, OSError) as exc:
        raise LikelihoodInputError(f"{path}: the data set was not written ({exc})") from None


def _pinned_doc(raw: bytes, where: str, expected_sha256: str, what: str, keys: tuple, fmt: str):
    """The parsed JSON object of a file of this module, and its sha256. The bytes are hashed before anything is
    parsed; then they must be strict JSON with exactly ``keys`` and the format ``fmt``."""
    if type(expected_sha256) is not str or not _S.exact_hex64(expected_sha256):
        raise LikelihoodInputError(
            f"{where}: {what} is read against its sha256, 64 lowercase hexadecimal digits"
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
    if not isinstance(doc, dict) or set(doc) != set(keys):
        got = set(doc) if isinstance(doc, dict) else set()
        raise LikelihoodInputError(
            f"{where}: keys must be exactly {sorted(keys)} (missing "
            f"{sorted(set(keys) - got)}, unknown {sorted(got - set(keys))})"
        )
    if doc["format"] != fmt:
        raise LikelihoodInputError(f"{where}: format {doc['format']!r}; this code reads {fmt!r}")
    return doc, sha


def _parse_p1d_data(raw: bytes, where: str, expected_sha256: str) -> P1DData:
    doc, sha = _pinned_doc(raw, where, expected_sha256, "a data set", DATA_KEYS, DATA_FORMAT)
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
    except (TypeError, ValueError, OverflowError) as exc:  # a ragged or non-numeric list
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


# ------------------------------------------------------------------------- a stored reference of a mock recovery
REFERENCE_FORMAT = "hcd_priya.p1d_reference/1"
_REFERENCE_FIELDS = (  # the fields of P1DReference that the file holds as lists, by name
    "param_names",
    "truth",
    "mean",
    "interval68",
    "interval95",
    "z",
    "k",
    "percentiles",
    "predictive",
)
REFERENCE_KEYS = ("format", *_REFERENCE_FIELDS, "provenance")


@dataclasses.dataclass(frozen=True, eq=False)
class P1DReference:
    """A stored reference of a mock recovery: a summary of a posterior, and a predictive band.

    It holds what a figure needs to show a chain against a reference, and nothing else: for each
    parameter the truth of the mock, the posterior mean and two central intervals; and for each
    bin of the data vector some percentiles of the posterior predictive. It holds no chain, no
    covariance and no likelihood value. What the numbers were made with is stated in
    ``provenance``, which this class carries and never interprets.

    Attributes
    ----------
    param_names : tuple of str
        The names of the parameters, shape (n_p,), each once.
    truth : numpy.ndarray
        Shape (n_p,). The point the mock was made at.
    mean : numpy.ndarray
        Shape (n_p,). The posterior mean of each parameter.
    interval68 : numpy.ndarray
        Shape (n_p, 2). The lower and the upper end of the central 68 per cent interval.
    interval95 : numpy.ndarray
        Shape (n_p, 2). The same for 95 per cent. It contains ``interval68``.
    z : tuple of float
        The redshifts of the data vector, shape (n_z,), distinct.
    k : tuple of numpy.ndarray
        One array per redshift: the wavenumbers of its bins, angular, in s/km, strictly ascending.
    percentiles : tuple of float
        The percentile levels of ``predictive``, strictly ascending, each between 0 and 100.
    predictive : numpy.ndarray
        Shape (n_percentiles, N), N the total number of bins: row i is the ``percentiles[i]``-th
        percentile of the posterior predictive of the data vector, in km/s, bin by bin, in the
        order of ``z`` and ``k``.
    provenance : dict
        What the reference is and how it was made. Free form.

    Raises
    ------
    LikelihoodInputError
        If the shapes do not fit, a number is not finite, a name or a redshift repeats, an
        interval is not ordered or the 95 per cent one does not contain the 68 per cent one, the
        percentile levels do not ascend, or the percentile rows decrease from one level to the
        next.
    """

    param_names: tuple
    truth: np.ndarray
    mean: np.ndarray
    interval68: np.ndarray
    interval95: np.ndarray
    z: tuple
    k: tuple
    percentiles: tuple
    predictive: np.ndarray
    provenance: Mapping = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        names = self.param_names
        if (
            not isinstance(names, (list, tuple))
            or not names
            or not all(type(n) is str and n for n in names)
        ):
            raise LikelihoodInputError("param_names must be a sequence of names (non-empty text)")
        if len(set(names)) != len(names):
            raise LikelihoodInputError("param_names must hold each name once")
        n_p = len(names)
        truth = _vector("truth", self.truth, n_p)
        mean = _vector("mean", self.mean, n_p)
        ends = {}
        for name in ("interval68", "interval95"):
            a = _numbers(name, getattr(self, name))
            if a.shape != (n_p, 2) or not np.all(np.isfinite(a)):
                raise LikelihoodInputError(
                    f"{name} must be a finite array of shape ({n_p}, 2), got shape {a.shape}"
                )
            if np.any(a[:, 0] > a[:, 1]):
                raise LikelihoodInputError(f"{name}: a lower end lies above its upper end")
            ends[name] = a
        i68, i95 = ends["interval68"], ends["interval95"]
        if np.any(i95[:, 0] > i68[:, 0]) or np.any(i95[:, 1] < i68[:, 1]):
            raise LikelihoodInputError("interval95 must contain interval68 for every parameter")
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
        levels = _vector("percentiles", self.percentiles)
        if levels.size == 0 or np.any(levels <= 0) or np.any(levels >= 100):
            raise LikelihoodInputError("percentiles must be levels between 0 and 100")
        if np.any(np.diff(levels) <= 0):
            raise LikelihoodInputError("percentiles must be strictly ascending")
        band = _numbers("predictive", self.predictive)
        if band.shape != (levels.size, n) or not np.all(np.isfinite(band)):
            raise LikelihoodInputError(
                f"predictive must be a finite array of shape ({levels.size}, {n}), got shape {band.shape}"
            )
        if np.any(np.diff(band, axis=0) < 0):
            raise LikelihoodInputError(
                "predictive must not decrease from one percentile level to the next, in any bin"
            )
        if not isinstance(self.provenance, Mapping):
            raise LikelihoodInputError("provenance must be a mapping")
        object.__setattr__(self, "param_names", tuple(names))
        object.__setattr__(self, "truth", _frozen(truth))
        object.__setattr__(self, "mean", _frozen(mean))
        object.__setattr__(self, "interval68", _frozen(i68))
        object.__setattr__(self, "interval95", _frozen(i95))
        object.__setattr__(self, "z", tuple(float(v) for v in z))
        object.__setattr__(self, "k", tuple(rows))
        object.__setattr__(self, "percentiles", tuple(float(v) for v in levels))
        object.__setattr__(self, "predictive", _frozen(band))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def n(self) -> int:
        """N, the number of bins of the data vector the reference is on."""
        return int(self.predictive.shape[1])

    def band(self, lower: float, upper: float) -> tuple:
        """The rows of ``predictive`` at two stored percentile levels.

        Parameters
        ----------
        lower, upper : float
            Two levels of ``percentiles``, for example 16 and 84. Nothing is interpolated between
            stored levels.

        Returns
        -------
        tuple of numpy.ndarray
            Two arrays of shape (N,), in km/s.

        Raises
        ------
        LikelihoodInputError
            If a level is not one of ``percentiles``, or ``lower`` is not below ``upper``.
        """
        if not lower < upper:
            raise LikelihoodInputError(f"a band needs lower < upper, got {lower!r} and {upper!r}")
        out = []
        for level in (lower, upper):
            if level not in self.percentiles:
                raise LikelihoodInputError(
                    f"this reference holds the percentiles {list(self.percentiles)}, not {level!r}"
                )
            out.append(self.predictive[self.percentiles.index(level)])
        return tuple(out)

    def check_bins(self, data: P1DData) -> None:
        """Refuse a data set whose redshifts or wavenumbers are not exactly this reference's.

        Parameters
        ----------
        data : P1DData
            The data set the reference is to be drawn with.

        Raises
        ------
        LikelihoodInputError
            If ``data`` is not a ``P1DData``, or its ``z`` and ``k`` are not equal to the
            reference's, value by value.
        """
        if not isinstance(data, P1DData):
            raise LikelihoodInputError("data must be a P1DData")
        same = (
            data.z == self.z
            and len(data.k) == len(self.k)
            and all(np.array_equal(a, b) for a, b in zip(data.k, self.k))
        )
        if not same:
            raise LikelihoodInputError(
                "the reference is not on the bins of this data set: its redshifts and wavenumbers "
                "must equal the data set's exactly"
            )


def write_reference(path: str | os.PathLike, reference: P1DReference) -> str:
    """Write a stored reference as one JSON file and return the file's sha256.

    The file is strict UTF-8 JSON in the format ``REFERENCE_FORMAT`` with exactly the keys
    ``REFERENCE_KEYS``. Numbers are written so that reading them back gives the same float64
    values. ``read_reference`` reads such a file only against its sha256.

    Parameters
    ----------
    path : str or os.PathLike
        The file to create. An existing file is not overwritten.
    reference : P1DReference
        The reference.

    Returns
    -------
    str
        The sha256 of the file, 64 hexadecimal digits.

    Raises
    ------
    LikelihoodInputError
        If ``provenance`` holds a value that is not plain JSON, or the file exists or cannot be
        written.
    """
    if not isinstance(reference, P1DReference):
        raise LikelihoodInputError("reference must be a P1DReference")
    doc = {
        "format": REFERENCE_FORMAT,
        "param_names": list(reference.param_names),
        "truth": reference.truth.tolist(),
        "mean": reference.mean.tolist(),
        "interval68": reference.interval68.tolist(),
        "interval95": reference.interval95.tolist(),
        "z": list(reference.z),
        "k": [r.tolist() for r in reference.k],
        "percentiles": list(reference.percentiles),
        "predictive": reference.predictive.tolist(),
        "provenance": reference.provenance,
    }
    try:
        raw = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LikelihoodInputError(
            f"provenance must hold plain JSON values (text, finite numbers, lists, mappings): {exc}"
        ) from None
    if len(raw) > MAX_DATA_BYTES:
        raise LikelihoodInputError(
            f"{path}: the reference takes {len(raw)} bytes as a file, above the {MAX_DATA_BYTES} that "
            "read_reference reads; it was not written"
        )
    try:
        return _S._write_exclusive(os.fspath(path), [raw])
    except (_S.SafeFormatError, OSError) as exc:
        raise LikelihoodInputError(f"{path}: the reference was not written ({exc})") from None


def read_reference(path: str | os.PathLike, *, expected_sha256: str) -> P1DReference:
    """Read a stored reference written by ``write_reference``, against its sha256.

    The file's bytes are hashed before anything is parsed, and a file with another sha256 is
    refused. Then the file must be strict JSON (no duplicate key, no NaN or Infinity) with exactly
    the keys of the format, and its arrays must make a valid ``P1DReference``.

    Parameters
    ----------
    path : str or os.PathLike
        The file. It must be a regular file, not a symbolic link.
    expected_sha256 : str
        The sha256 the file must have: 64 lowercase hexadecimal digits.

    Returns
    -------
    P1DReference
        Its ``provenance`` is the file's, with one entry added: ``file_sha256``.

    Raises
    ------
    LikelihoodInputError
        If the file cannot be read, has another sha256, or is not a reference of this format.
    """
    where = os.fspath(path)
    try:
        raw = _S.read_regular(where, MAX_DATA_BYTES)
    except (_S.SafeFormatError, OSError) as exc:
        raise LikelihoodInputError(f"{path}: not a readable reference file ({exc})") from None
    doc, sha = _pinned_doc(
        raw, where, expected_sha256, "a reference", REFERENCE_KEYS, REFERENCE_FORMAT
    )
    for key in _REFERENCE_FIELDS:
        if not isinstance(doc[key], list):
            raise LikelihoodInputError(f"{where}: {key} must be a list")
    if not isinstance(doc["provenance"], dict):
        raise LikelihoodInputError(f"{where}: provenance must be an object")
    try:
        return P1DReference(
            **{key: doc[key] for key in _REFERENCE_FIELDS},
            provenance=dict(doc["provenance"], file_sha256=sha),
        )
    except LikelihoodInputError as exc:
        raise LikelihoodInputError(f"{where}: {exc}") from None
    except (TypeError, ValueError, OverflowError) as exc:  # a ragged or non-numeric list
        raise LikelihoodInputError(f"{where}: its arrays are not well formed ({exc})") from None


# ------------------------------------------------------------------------------- XQ-100: a loader, and nothing more
# The two files of the XQ-100 P1D that this loader reads, with the sha256 each must have. Neither is shipped: the
# user supplies them. Public copies (2026-11): the data file in github.com/sbird/lya_emulator and in
# github.com/jibanCat/InferenceLyaData under lyaemu/data/xq100/; the covariance file in the second only.
XQ100_FILES = {
    "data": (
        "pk_obs_corrNR_offset_DLATrue_metalTrue_res0.csv",
        "615048b825d3ba2f43f62c1e04969cfba050d761d1b07bde794810cbabf95b58",
    ),
    "covariance": (
        "cov_xq100_lyb_corrected_onlyA_fromMCMC.txt",
        "9e626de99692d483c6f641a72eac9f4a8b01714e921220e787d8d2a42dfb176b",
    ),
}
XQ100_HEADER = "k,paa,ptt,pab,qab,pbb,npix_aa,npix_tt,npix_ab,z"
XQ100_Z = (3.4, 3.6, 3.8, 4.0, 4.2)  # the redshifts the covariance file covers
XQ100_NK = 13


def _read_pinned(path, expected_sha256, what):
    try:
        raw = _S.read_regular(os.fspath(path), MAX_DATA_BYTES)
    except (_S.SafeFormatError, OSError) as exc:
        raise LikelihoodInputError(f"{path}: not a readable {what} ({exc})") from None
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise LikelihoodInputError(
            f"{path}: sha256 {sha[:12]}... is not that of the {what} this loader reads "
            f"({expected_sha256[:12]}...); the file was not parsed"
        )
    try:
        return raw.decode("utf-8"), sha
    except UnicodeDecodeError:
        raise LikelihoodInputError(f"{path}: the {what} is not UTF-8 text") from None


def load_xq100(
    data_file: str | os.PathLike,
    covariance_file: str | os.PathLike,
    *,
    cross_redshift: bool,
    expected_sha256: Mapping[str, str] | None = None,
) -> P1DData:
    """Load the XQ-100 Lyman-alpha P1D from two files that you supply. A loader only.

    Nothing is shipped and nothing is fitted: this reads the measurement and its covariance into
    a ``P1DData`` and stops. No result of this package rests on these data, and a likelihood made
    from the returned object carries none of the terms a real analysis needs (no emulator error,
    no resolution or metal model, no inflation of the low wavenumbers).

    The two files are named in ``XQ100_FILES`` with the sha256 each must have. The data file is a
    table with the header ``k,paa,ptt,pab,qab,pbb,npix_aa,npix_tt,npix_ab,z``: 13 wavenumbers at
    each of 7 redshifts. The loader takes the columns ``z``, ``k`` and ``paa`` (the Lyman-alpha
    auto power) at the five redshifts 3.4 to 4.2, the ones the covariance covers: 65 bins; the
    other columns are not read and may be empty. The
    covariance file lists a 65 x 65 matrix as lines ``i j value`` in the same order (redshift
    ascending, then wavenumber ascending). Wavenumbers are angular, in s/km; the power is in
    km/s.

    Parameters
    ----------
    data_file, covariance_file : str or os.PathLike
        Your copies of the two files.
    cross_redshift : bool
        True keeps the full 65 x 65 matrix of the file. False keeps the five 13 x 13 blocks on
        the diagonal and sets the covariance between redshifts to zero. There is no default: the
        choice changes the likelihood, so it is yours to state.
    expected_sha256 : dict, optional
        ``{"data": sha256, "covariance": sha256}`` to read other files of the same layout. The
        default is the two digests of ``XQ100_FILES``.

    Returns
    -------
    P1DData
        Five redshifts with 13 bins each. ``truth`` is None. ``provenance`` names the files, their
        digests and the choice of ``cross_redshift``.

    Raises
    ------
    LikelihoodInputError
        If a file cannot be read, has another sha256, or does not have the layout described above.

    Notes
    -----
    Where the files come from: the code they are distributed with attributes them to the XQ-100
    analysis of Wilson, Irsic and McQuinn (2021, arXiv:2106.04837, "A measurement of the
    Lyman-beta forest power spectrum and its cross with the Lyman-alpha forest in X-Shooter
    XQ-100"). This project has NOT verified that attribution against the paper, nor the
    definition of the column ``paa`` against it. The covariance file holds indices and values
    only: that index 0 to 64 means redshift 3.4 to 4.2 ascending and then wavenumber ascending
    is taken from the same code, which itself marks it as unchecked. It is NOT verified here
    either, and a wrong order would give a wrong likelihood without any error. The
    repositories that hold the files state no licence.
    """
    if type(cross_redshift) is not bool:
        raise LikelihoodInputError("cross_redshift must be True or False: state the choice")
    pins = {k: v[1] for k, v in XQ100_FILES.items()}
    if expected_sha256 is not None:
        if set(expected_sha256) != set(pins) or not all(
            _S.exact_hex64(v) for v in expected_sha256.values()
        ):
            raise LikelihoodInputError(
                'expected_sha256 must be {"data": sha256, "covariance": sha256}, each 64 lowercase '
                "hexadecimal digits"
            )
        pins = dict(expected_sha256)
    text, data_sha = _read_pinned(data_file, pins["data"], "XQ-100 data file")
    lines = text.splitlines()
    if not lines or lines[0].strip() != XQ100_HEADER:
        raise LikelihoodInputError(f"{data_file}: the header is not {XQ100_HEADER!r}")
    fields = [ln.split(",") for ln in lines[1:] if ln.strip()]
    if not fields or any(len(f) != 10 for f in fields):
        raise LikelihoodInputError(
            f"{data_file}: every row must have the ten columns of the header"
        )
    try:  # only k, paa and z are read; the other columns may be empty
        table = np.array([[float(f[0]), float(f[1]), float(f[9])] for f in fields])
    except ValueError:
        raise LikelihoodInputError(f"{data_file}: k, paa or z of a row is not a number") from None
    if not np.all(np.isfinite(table)):
        raise LikelihoodInputError(f"{data_file}: k, paa and z must be finite")
    k_col, p_col, z_col = table[:, 0], table[:, 1], table[:, 2]
    z_rows, k_rows, p_rows = [], [], []
    for z in XQ100_Z:
        at = np.where(np.abs(z_col - z) < 1e-6)[0]
        if at.size != XQ100_NK or np.any(np.diff(at) != 1) or np.any(np.diff(k_col[at]) <= 0):
            raise LikelihoodInputError(
                f"{data_file}: z = {z:g} must have {XQ100_NK} consecutive rows with ascending k"
            )
        z_rows.append(float(z))
        k_rows.append(k_col[at])
        p_rows.append(p_col[at])
    firsts = [int(np.where(np.abs(z_col - z) < 1e-6)[0][0]) for z in XQ100_Z]
    if firsts != sorted(firsts):
        raise LikelihoodInputError(f"{data_file}: the redshifts are not in ascending order")
    n = XQ100_NK * len(XQ100_Z)
    text, cov_sha = _read_pinned(covariance_file, pins["covariance"], "XQ-100 covariance file")
    cov = np.full((n, n), np.nan)
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    if len(rows) != n * n or any(len(r) != 3 for r in rows):
        raise LikelihoodInputError(
            f"{covariance_file}: expected {n * n} lines of the form 'i j value', got {len(rows)}"
        )
    for i, j, v in rows:
        if not (i.isdigit() and j.isdigit() and int(i) < n and int(j) < n):
            raise LikelihoodInputError(
                f"{covariance_file}: a line does not start with two indices from 0 to {n - 1}"
            )
        if not np.isnan(cov[int(i), int(j)]):
            raise LikelihoodInputError(f"{covariance_file}: the element ({i}, {j}) is given twice")
        try:
            cov[int(i), int(j)] = float(v)
        except ValueError:
            raise LikelihoodInputError(
                f"{covariance_file}: the value of ({i}, {j}) is not a number"
            ) from None
    if not np.all(np.isfinite(cov)):
        raise LikelihoodInputError(
            f"{covariance_file}: not every element of the {n} x {n} matrix is given"
        )
    if not cross_redshift:
        blocks = np.zeros_like(cov)
        for a in range(0, n, XQ100_NK):
            blocks[a : a + XQ100_NK, a : a + XQ100_NK] = cov[a : a + XQ100_NK, a : a + XQ100_NK]
        cov = blocks
    provenance = {
        "what": "the XQ-100 Lyman-alpha P1D (column paa) at five redshifts, read from files supplied by the user",
        "data_file": {"name": os.path.basename(os.fspath(data_file)), "sha256": data_sha},
        "covariance_file": {
            "name": os.path.basename(os.fspath(covariance_file)),
            "sha256": cov_sha,
        },
        "covariance": (
            "the full matrix of the file"
            if cross_redshift
            else "the blocks of the file on the diagonal; the covariance between redshifts is set to zero"
        ),
        "note": "a measurement, loaded as given: no emulator error, no resolution or metal model, no inflation",
        "not_verified": "that the covariance file's index order is redshift 3.4 to 4.2 ascending, then wavenumber "
        "ascending; the attribution of the files; the definition of the column paa",
    }
    try:
        return P1DData(
            z=tuple(z_rows),
            k=tuple(k_rows),
            p1d=np.concatenate(p_rows),
            cov=cov,
            provenance=provenance,
        )
    except LikelihoodInputError as exc:
        raise LikelihoodInputError(
            f"the XQ-100 files do not make a valid data set: {exc}"
        ) from None
