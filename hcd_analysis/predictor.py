"""The public predictor: PRIYA-filtered per-class P1D from an approved, clean-lineage model bundle, with free HCD
amplitudes and no prior.

    import hcd_analysis.predictor as hp
    emu = hp.load_bundle(bundle_dir, "<the published manifest sha256>")     # None is refused
    pred = emu.predict(params, z=3.0, tau_eff=0.36)                          # or tau0_amp=..., dtau0=...
    pred.k, pred.P_class, pred.template, pred.w_c                            # (K,), (4, K), (3, K), (3,)
    P_tot = hp.add_hcd(pred, pred.w_c)                                       # the map's estimate of its HCD content
    stack = emu.predict_many(params, z=[2.6, 2.8, 3.0], tau0_amp=1.0, dtau0=0.0)
    P_tot = hp.add_hcd(stack, A)                                             # A: (3,) for every z, or (n_z, 3)

**Loading.** ``load_bundle`` loads a PRODUCTION bundle of format ``hcd_priya.h5/2`` through
``safe_h5.load_release``, anchored on the manifest sha256 published with the release: the bundle's lineage block must
establish a clean lineage (the corrected finder, a release-eligible training cache, the members trained on it at one
clean commit, the w_c map fitted to it) and every member must be a production member (trained on every simulation of
the cache: ``eligibility.production_recipe_reasons``), or it is refused; ``Bundle`` recomputes that eligibility
itself. ``None`` is refused here; an unanchored development load is
``safe_h5.load_ensemble(..., expected_manifest_sha256=None)``. A RESEARCH bundle (cross-validation folds,
full-simulation research models: ``provenance.research``) is refused by ``load_bundle`` and loads with
``load_research_bundle(bundle_dir, sha256, purpose=...)``, which returns a ``ResearchPredictor`` (the same forward,
every prediction flagged ``research_model``); it is never a ``Bundle``.

**Scope: one bundle is single fidelity.** Format ``hcd_priya.h5/2`` holds the members of ONE training cache and the
w_c map fitted to it, and no multi-fidelity (LF to HR) correction: a bundle loaded alone, trained on the low-fidelity
cache, gives low-fidelity predictions (``labels["product"]`` names the training cache's fidelity;
``labels["mf_correction"]`` is ``none``). The multi-fidelity correction is not part of the bundle format: it is a
separate MF record (``hcd_priya.mf_record/1``) bound to one bundle by its manifest sha256, and an LF+MF release
publishes the two under one release manifest (below). An ``hcd_priya.h5/2`` bundle alone is therefore not the complete
release model.

**A release (LF+MF).** ``load_release(release_dir, "<the published release sha256>")`` loads a whole release from its
release manifest (``release.json``, ``emulator.release_manifest``): the backbone bundle and the MF record it names,
each checked against the digest the manifest gives, through ``load_bundle`` below.

**The MF correction (LF+MF).** ``load_bundle(bundle_dir, sha256, mf_dir=..., mf_record_sha256=...)`` also loads the
release MF record (format ``hcd_priya.mf_record/1``, ``emulator.mf_release``), anchored on its published sha256 and
bound to this bundle: its backbone manifest sha256, training cache, backbone kind and box modes must be this
bundle's, and a historical or fold record is refused. Each member's per-class spectrum is then multiplied by
exp(g(z, tau0, k)) on the native modes, before the mean, the spread and any k placement: P_MF = P_LF exp(g). g does not
depend on the cosmological parameters. Below max(the lowest redshift of the record's tables, the record's
``validated_z_min``, 2.2 by default) the MF is REFUSED (``MFRangeError``; ask for ``fidelity="LF"``): it is never
clamped there. Above the highest redshift and outside the
tau0 tables it is held constant and flagged (``mf_z_clamped``, ``mf_tau0_clamped``). Above z 4.6
(``mf_release.VALIDATED_Z_MAX``, the upper edge of the frozen grouped-validation domain (z 2.2-4.6; the held-out validation is reported separately)) an LF+MF prediction is
served with the same values but flagged ``mf_z_above_validated``, inside the tables too: it lies outside that domain. ``w_c`` and ``dndx`` stay the LF
backbone's. ``fidelity`` ("LF" or "LF+MF"; None: the bundle's own) selects the output; "LF+MF" without a record raises
``MFUnavailableError``. The record's provenance (version, composition and ruling ids, cache digests and their
provenance statements, fit commit, the settled mean-flux convention of its targets, the HR inputs used) is in
``provenance["mf"]``. Without an MF record nothing here changes. A FOLD record (a held-out fit for grouped cross-validation) applies only on its research backbone
(``load_research_bundle(..., mf_dir=..., mf_record_sha256=...)``); ``heldout_P_filt_at(pred, x, tau0)`` gives that
fold's held-out LF and MF predictions at cache inputs from the same object. ``backbone_identity(pred)`` is what an MF
record must declare about a backbone, derived from its member metas; ``lf_P_filt_at`` and ``native_modes`` serve the
MF builder. These are module functions, so the public surface of ``Bundle`` is unchanged.

**Inputs.** ``params``: a dict of the nine PRIYA parameters (``PARAM_NAMES``): ``ns`` and ``Ap`` at the PRIYA pivot
k_p = 0.78 Mpc^-1, ``herei``, ``heref``, ``alphaq``, ``hub``, ``omegamh2``, ``hireionz``, ``bhfeedback``. ``z``: a PRIYA
snapshot redshift (2.0, 2.2, ..., 5.4, matched to 1e-9). The mean flux: either ``tau_eff``, -ln<F> of the PRIYA-filtered
simulated forest (the model's tau0 input), or PRIYA's (``tau0_amp``, ``dtau0``), converted by ``tau_eff_priya``. ``k``:
None for the native modes k_skm(z, hub, omegamh2), or wavenumbers in s/km inside that span (linear interpolation along
the mode axis, kcoord.bind / at_data). Values outside the training box are evaluated and flagged, not refused; n_s
above the original PRIYA design box (``NS_RIDGE``, where the design is sparse) is flagged ``ns_ridge``.

**Outputs** (``Prediction``; host numpy, float64):
  * ``k`` (K,) s/km; ``P_class`` (4, K): clean, LLS, subDLA, DLA, PRIYA-filtered (tau > 1e6 troughs filled), the mean
    over the ensemble members of their reconstructed spectra;
  * ``template`` (3, K) = P_c / P_clean - 1 for LLS, subDLA, DLA;
  * ``w_c`` (3,): an estimate of the simulation's own sightline fractions (LLS, subDLA, DLA) at (params, z), from the
    members' mean Head-A dN/dX (``dndx``, (3,)) through the w_c map shipped in the bundle (``dndx_wc.w_c_corrected``
    with the map's delta_c(z) and Xbar(z)); never the historical constants;
  * ``member_std`` (4, K): the standard deviation over the members (ddof 1), a SEED-SPREAD DIAGNOSTIC (the bundle's
    ``n_members`` networks, differing only in their initialisation), not a calibrated emulator error and not a
    covariance;
  * ``flags``, ``labels`` and ``provenance`` (the manifest sha256, the lineage block, the w_c map's sha256 and the
    manifest's own provenance: the export's identity, software versions, creation time, the source digests).

**HCD.** ``add_hcd(pred, A)`` returns P_clean + sum_c A_c (P_c - P_clean), with one amplitude per class (A of shape (3,))
or per class per redshift ((n_z, 3) on a stack). A_c is the effective sightline fraction of class c. The class
spectra are normalised as in the training cache, by the snapshot's global mean flux (``labels["normalisation"]``):
the settled production convention; normalising each class by its own mean flux is a diagnostic only. Under that
normalisation and with the counted class fractions, sum over the four classes of w_c P_c is the full-sample filtered
P1D; the shipped w_c is the map's fitted estimate of those fractions (the fit residuals are recorded in the map), so
A = w_c gives that total only to the accuracy of the map, and only under this normalisation. Only physical validity is imposed: every A_c >= 0 and the
clean fraction 1 - sum_c A_c >= 0; anything else is a clear error. This module has no prior, no survey key and no DLA
forward-fraction default, and it never imports the historical likelihood or priors (``inference``, ``closure_legb``,
``lit_dndx*``, ``meanflux_prior``). Two contributions are not in the total (documented omissions): the subDLA filtered remnant (the power of
subDLA cores the PRIYA filter removes) and the DLA damping-wing core (the DLA spectrum here is the filtered one).

The checked entry points take host values and refuse a JAX tracer; the forward itself runs as one jitted function per
bundle and input shape."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

import hcd_analysis.emulator  # noqa: F401  (float64 before any array)
import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import kcoord as _K
from .emulator import mf_release as _MR
from .emulator import safe_h5 as _H
from .emulator.data import DATA_RANGE, PARAM_LIMITS, SAMPLING_LIMITS, Z_LIMITS, normalize_params
from .emulator.mean_flux_model import TAU0_PIVOT_Z, kim_tau0, tau0_alpha_priya
from .emulator.predict import reconstruct_P_filt_jax
from .emulator.schema import CHECKPOINT_SCHEMA_VERSION, L_BOX_HMPC

PARAM_NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback")
CLASSES = ("clean", "LLS", "subDLA", "DLA")
HCD_CLASSES = ("LLS", "subDLA", "DLA")
Z_GRID = tuple(
    round(2.0 + 0.2 * i, 10) for i in range(18)
)  # the PRIYA snapshot redshifts 2.0 ... 5.4
Z_TOL = 1e-9
OMISSIONS = (
    "the subDLA filtered remnant: the power of subDLA cores the PRIYA filter removes is not added back",
    "the DLA damping-wing core: the DLA class is the PRIYA-filtered spectrum, without its core",
)
_I_HUB, _I_OMH2 = PARAM_NAMES.index("hub"), PARAM_NAMES.index("omegamh2")
# the upper n_s edge of the original PRIYA design box: the design is sparse above it (flag ns_ridge; a diagnostic)
NS_RIDGE = 0.995


class PredictorInputError(ValueError):
    """An input that the predictor refuses.

    Raised for an input of the wrong type or shape, a value that is not finite, a redshift that is
    not a PRIYA snapshot, a ``k`` outside the native mode span, or an HCD amplitude that is not a
    physical fraction. The message names the input and the reason. It is a ``ValueError``.
    """


class MFUnavailableError(PredictorInputError):
    """An LF+MF prediction was asked for, but no MF record is loaded.

    The MF record holds the multi-fidelity (MF) correction from the low-fidelity (LF) simulations to
    high resolution. ``load_release`` loads it, and so does ``load_bundle`` with ``mf_dir`` and
    ``mf_record_sha256``. The message says how to load it.
    """


class MFRangeError(PredictorInputError):
    """An LF+MF prediction was asked for below the lowest redshift the MF record covers.

    The lower edge is the larger of two values: the lowest redshift in the record's tables, and the
    record's ``validated_z_min`` (2.2 by default). Below it the correction is refused. It is never
    extrapolated or held constant there. Ask for ``fidelity="LF"`` instead. The message gives the
    edge and the redshifts you asked for.
    """


FIDELITIES = ("LF", "LF+MF")


# -------------------------------------------------------------------------------------------------- host checks
def _host(name, v):
    if isinstance(v, jax.core.Tracer):
        raise TypeError(
            f"{name}: the checked predictor takes host values, not a JAX tracer (it is not traced; "
            "differentiate the forward through hcd_analysis.emulator.predict instead)"
        )
    return v


def _scalar(name, v):
    _host(name, v)
    if (
        isinstance(v, (bool, np.bool_, str, bytes))
        or np.ndim(v) != 0
        or np.asarray(v).dtype.kind not in "iuf"
    ):
        raise PredictorInputError(f"{name} must be a real scalar, got {v!r}")
    x = float(v)
    if not np.isfinite(x):
        raise PredictorInputError(f"{name} = {x} is not finite")
    return x


def _array(name, v, ndim=None):
    _host(name, v)
    a = np.asarray(v)
    if a.dtype == bool or not (
        np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)
    ):
        raise PredictorInputError(f"{name} must be real numbers, got dtype {a.dtype}")
    a = a.astype(np.float64)
    if ndim is not None and a.ndim not in ((ndim,) if isinstance(ndim, int) else ndim):
        raise PredictorInputError(f"{name} must have {ndim} dimension(s), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise PredictorInputError(f"{name} is not finite")
    return a


def params_vector(params):
    """Return the nine PRIYA parameters as an array in ``PARAM_NAMES`` order.

    Parameters
    ----------
    params : dict
        Exactly the nine names in ``PARAM_NAMES``, each with a finite real value.

    Returns
    -------
    numpy.ndarray
        Shape (9,), float64, in ``PARAM_NAMES`` order.

    Raises
    ------
    PredictorInputError
        If ``params`` is not a dict, lacks a name or has an extra one (the message lists both), or
        holds a value that is not a finite real number.

    Examples
    --------
    The centre of the parameter box (synthetic values):

    >>> import hcd_analysis.predictor as hp
    >>> params = {n: 0.5 * (lo + hi) for n, (lo, hi) in zip(hp.PARAM_NAMES, hp.PARAM_LIMITS)}
    >>> hp.params_vector(params).shape
    (9,)
    """
    if not isinstance(params, dict):
        raise PredictorInputError(f"params must be a dict of {PARAM_NAMES}")
    keys = set(params)
    if keys != set(PARAM_NAMES):
        raise PredictorInputError(
            f"params: missing {sorted(set(PARAM_NAMES) - keys)}, unknown {sorted(keys - set(PARAM_NAMES))}"
        )
    return np.array([_scalar(f"params[{n!r}]", params[n]) for n in PARAM_NAMES], dtype=np.float64)


def snapshot_z(z):
    """Match a redshift to the PRIYA snapshot grid.

    The predictor works only at the snapshot redshifts 2.0, 2.2, ..., 5.4 (``Z_GRID``). A value
    within ``Z_TOL`` (1e-9) of a grid point is accepted.

    Parameters
    ----------
    z : float
        The redshift.

    Returns
    -------
    float
        The grid value.

    Raises
    ------
    PredictorInputError
        If ``z`` is not a finite real number, or is not within ``Z_TOL`` of a snapshot redshift.

    Examples
    --------
    >>> import hcd_analysis.predictor as hp
    >>> hp.snapshot_z(3.0000000001)
    3.0
    >>> hp.snapshot_z(3.1)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    hcd_analysis.predictor.PredictorInputError: z = 3.1 is not a PRIYA snapshot redshift ...
    """
    x = _scalar("z", z)
    hit = [g for g in Z_GRID if abs(x - g) <= Z_TOL]
    if not hit:
        raise PredictorInputError(
            f"z = {x} is not a PRIYA snapshot redshift {Z_GRID} (tolerance {Z_TOL})"
        )
    return hit[0]


def tau_eff_priya(z, tau0_amp, dtau0, *, z_pivot=TAU0_PIVOT_Z):
    """Convert PRIYA's mean-flux parameters to the predictor's input ``tau_eff``.

    ``tau_eff`` is the effective optical depth of the PRIYA-filtered forest: minus the natural log
    of its mean transmitted flux. PRIYA's model of it is::

        tau_eff(z) = tau0_amp * ((1 + z) / (1 + z_pivot))**dtau0 * tau_Kim(z)

    where ``tau_Kim`` is the mean-flux fit of Kim et al. (2007), ``mean_flux_model.kim_tau0``.

    Parameters
    ----------
    z : float
        The redshift.
    tau0_amp : float
        The amplitude: a multiplier of ``tau_Kim``.
    dtau0 : float
        The slope of the redshift power law.
    z_pivot : float, optional
        The pivot redshift. The default is ``mean_flux_model.TAU0_PIVOT_Z`` (3.0).

    Returns
    -------
    float
        ``tau_eff`` at ``z`` (dimensionless).

    Raises
    ------
    PredictorInputError
        If an input is not a finite real number.

    Examples
    --------
    Doubling ``tau0_amp`` doubles ``tau_eff``:

    >>> import hcd_analysis.predictor as hp
    >>> round(hp.tau_eff_priya(3.0, 2.0, 0.0) / hp.tau_eff_priya(3.0, 1.0, 0.0), 12)
    2.0
    """
    zz = _scalar("z", z)
    return float(
        tau0_alpha_priya(
            zz, _scalar("tau0_amp", tau0_amp), _scalar("dtau0", dtau0), z_pivot=z_pivot
        )
        * kim_tau0(zz)
    )


def _taus(zs, tau_eff, tau0_amp, dtau0):
    given = tau_eff is not None
    if given == (tau0_amp is not None or dtau0 is not None):
        raise PredictorInputError("give either tau_eff, or both tau0_amp and dtau0")
    if not given:
        if tau0_amp is None or dtau0 is None:
            raise PredictorInputError("give both tau0_amp and dtau0")
        out = np.array([tau_eff_priya(z, tau0_amp, dtau0) for z in zs])
    else:
        out = _array("tau_eff", tau_eff, ndim=(0, 1))
        out = np.broadcast_to(out, (len(zs),)) if out.ndim == 0 else out
        if out.shape != (len(zs),):
            raise PredictorInputError(f"tau_eff has {out.size} values for {len(zs)} redshifts")
    if np.any(out <= 0):
        raise PredictorInputError(f"tau_eff must be positive, got {out}")
    return np.array(out, dtype=np.float64)


def check_amplitudes(A, zs=None):
    """Check that HCD amplitudes are physical sightline fractions.

    An HCD amplitude A_c is the effective fraction of sightlines in class c, where c is LLS, subDLA
    or DLA (high-column-density absorbers, from the lowest column density to the highest). Each
    A_c must be zero or more. Together they must leave a clean fraction, 1 - sum_c A_c, of zero or
    more. No other condition is imposed: there is no prior.

    Parameters
    ----------
    A : array_like
        Shape (..., 3). The last axis is [A_LLS, A_subDLA, A_DLA].
    zs : sequence of float, optional
        The redshift of each row, used only to name the redshift in an error message.

    Returns
    -------
    numpy.ndarray
        ``A`` as float64.

    Raises
    ------
    PredictorInputError
        If ``A`` is not finite, its last axis does not have length 3, an A_c is negative, or the
        clean fraction is negative. The message names the class and, with ``zs``, the redshift.

    Examples
    --------
    Synthetic amplitudes:

    >>> import hcd_analysis.predictor as hp
    >>> hp.check_amplitudes([0.01, 0.002, 0.001]).shape
    (3,)
    >>> hp.check_amplitudes([0.6, 0.3, 0.2])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    hcd_analysis.predictor.PredictorInputError: the clean-class fraction 1 - sum_c A_c = ...
    """
    A = _array("A", A)
    if A.shape[-1:] != (3,):
        raise PredictorInputError(
            f"A must end in the three HCD classes {HCD_CLASSES}, got shape {A.shape}"
        )
    flat = A.reshape(-1, 3)
    where = (lambda i: "") if zs is None or flat.shape[0] == 1 else (lambda i: f" at z = {zs[i]}")
    for i, row in enumerate(flat):
        for c, a in zip(HCD_CLASSES, row):
            if a < 0:
                raise PredictorInputError(
                    f"A_{c} = {a!r}{where(i)} is negative: an HCD amplitude is a sightline "
                    "fraction (A_c >= 0)"
                )
        clean = 1.0 - float(np.sum(row))
        if clean < 0:
            raise PredictorInputError(
                f"the clean-class fraction 1 - sum_c A_c = {clean!r}{where(i)} is negative "
                f"(A = {row.tolist()}): the class fractions cannot exceed 1"
            )
    return A


# ------------------------------------------------------------------------------------------------- the forward
def _members_forward(ens, pf, dn_mean, dn_std, x, tau):
    """Per member: the reconstructed per-class P_filt (M, 4, K) and the physical Head-A dN/dX (M, 3)."""
    outs = [m(x, tau) for m in ens.members]
    P = jnp.stack([reconstruct_P_filt_jax(o["P_filt_base"], o["P_filt_resid"], pf) for o in outs])
    dn = jnp.stack([jnp.exp(o["dndx"] * dn_std + dn_mean) for o in outs])
    return P, dn


_forward_one = eqx.filter_jit(_members_forward)
_mf_g = eqx.filter_jit(lambda mf, x, tau: mf(x, tau))  # the MF correction g (4, K) at one input
_forward_many = eqx.filter_jit(
    eqx.filter_vmap(_members_forward, in_axes=(None, None, None, None, 0, 0))
)


def _frozen(a):
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclasses.dataclass(frozen=True)
class Prediction:
    """One prediction, returned by ``predict`` (one redshift) or ``predict_many`` (several).

    Every array is a read-only float64 numpy array. K is the number of wavenumbers. On a stack
    (``stacked`` True), every array gains a leading redshift axis of length n_z.

    Attributes
    ----------
    z : float or numpy.ndarray
        The snapshot redshift, or the redshifts of a stack, shape (n_z,).
    k : numpy.ndarray
        The wavenumbers in s/km, shape (K,): the simulation's own modes, or the ``k`` asked for.
    P_class : numpy.ndarray
        Shape (4, K). The one-dimensional flux power spectrum (P1D, in km/s) of each class, in the
        order clean, LLS, subDLA, DLA. It is the PRIYA-filtered spectrum (pixels with optical depth
        above 1e6 are filled), averaged over the ensemble members.
    template : numpy.ndarray
        Shape (3, K): P_c / P_clean - 1 for LLS, subDLA and DLA.
    w_c : numpy.ndarray
        Shape (3,). The model's estimate of each HCD class's sightline fraction at these
        parameters and redshift, from ``dndx`` through the w_c map of the bundle.
    dndx : numpy.ndarray
        Shape (3,). The predicted incidence dN/dX of each HCD class (absorbers per unit absorption
        distance X), averaged over the members.
    member_std : numpy.ndarray or None
        Shape (4, K). The standard deviation of ``P_class`` over the members. It shows how much the
        members disagree. It is not a calibrated emulator error and not a covariance. None for a
        bundle with one member.
    tau_eff : float or numpy.ndarray
        The mean-flux input used, one value per redshift.
    params : dict
        The nine parameters used.
    flags : dict
        Warnings about this prediction, each True when it applies: for example
        ``outside_training_box``, ``ns_ridge`` or ``mf_z_above_validated``. ``predict`` lists them.
    labels : dict
        Plain-text statements of what the numbers are and are not.
    provenance : dict
        Which model made the prediction: the bundle's manifest sha256, its lineage record (the
        record of how the bundle was made) and, with an MF record, the record's identity.
    stacked : bool
        False for one redshift, True for a stack.

    Notes
    -----
    ``add_hcd`` returns plain arrays without these flags. Keep the ``Prediction``, or its
    ``flags`` and ``provenance``, with any number you derive from it.
    """

    z: object
    k: np.ndarray
    P_class: np.ndarray
    template: np.ndarray
    w_c: np.ndarray
    dndx: np.ndarray
    member_std: object
    tau_eff: object
    params: dict
    flags: dict
    labels: dict
    provenance: dict
    stacked: bool


class _PredictorBase:
    """The forward shared by the production ``Bundle`` and the ``ResearchPredictor`` (moved unchanged from
    ``Bundle``): the members' checks, the jitted forward, the attributes, the labels, the flags."""

    def _setup(self, rb):
        for i, m in enumerate(rb.metas):
            if m.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise _H.SafeFormatError(
                    f"member {i}: checkpoint schema {m.get('schema_version')!r}"
                )
            if not np.array_equal(
                np.asarray(m["param_limits"], float), np.asarray(PARAM_LIMITS, float)
            ):
                raise _H.SafeFormatError(
                    f"member {i}: its parameter box is not data.PARAM_LIMITS (theta would be "
                    "normalised with another box)"
                )
        k_com = np.asarray(rb.metas[0]["k_com_hmpc"], dtype=np.float64)
        if not np.allclose(
            k_com, 2.0 * np.pi * np.arange(1, k_com.size + 1) / L_BOX_HMPC, rtol=1e-12, atol=0
        ):
            raise _H.SafeFormatError("the bundle's k_com_hmpc are not the box modes 2 pi n / L_box")
        self._rb = rb
        self._k_com = k_com
        n = rb.norm
        self._pf = {
            k: jnp.asarray(np.asarray(n["P_filt"][k], dtype=np.float64))
            for k in ("mu_marg", "sig_marg", "sig_cosmo")
        }
        self._dn = (
            jnp.asarray(np.asarray(n["dndx"]["mean"], dtype=np.float64)),
            jnp.asarray(np.asarray(n["dndx"]["std"], dtype=np.float64)),
        )
        settings = (rb.lineage.get("cache") or {}).get("settings") or {}
        ar = settings.get("alpha_range")
        self._alpha_range = None if ar is None else (float(ar[0]), float(ar[1]))
        self._fidelity = str(settings.get("fidelity", "")).upper() or None

    # ------------------------------------------------------------------ what was loaded
    @property
    def manifest_sha256(self):
        return self._rb.manifest_sha256

    @property
    def lineage(self):
        return copy.deepcopy(self._rb.lineage)

    @property
    def n_members(self):
        return len(self._rb.ensemble.members)

    @property
    def wc_map(self):
        return self._rb.wc_map

    @property
    def provenance(self):
        out = {
            "format": _H.FORMAT,
            "manifest_sha256": self._rb.manifest_sha256,
            "lineage": self.lineage,
            "wc_map_sha256": self._rb.wc_map.sha256,
            "n_members": self.n_members,
            "bundle_provenance": copy.deepcopy(self._rb.manifest.get("provenance")),
        }
        if self._mf is not None:
            out["mf"] = self._mf.provenance()
        if self._release is not None:
            out["release"] = copy.deepcopy(self._release)
        return out

    def _lf_P_filt_at(self, x, tau0, *, chunk=256):
        """The LF backbone's per-class P_filt (n, 4, K) on the native modes at encoder inputs ``x`` (n, 10) =
        [theta_unit (9), z_unit] and mean-flux inputs ``tau0`` (n,): the mean over the members, exactly as
        ``predict`` forms ``P_class`` without the MF. Used to form MF targets on cache rows
        (scripts/build_mf_release.py); never applies an MF."""
        x = _array("x", x, ndim=2)
        t = _array("tau0", tau0, ndim=1)
        if x.shape[1] != 10 or t.shape != (x.shape[0],):
            raise PredictorInputError(
                f"x must be (n, 10) and tau0 (n,), got {x.shape} and {t.shape}"
            )
        out = []
        for i in range(0, x.shape[0], chunk):
            P, _ = _forward_many(
                self._rb.ensemble,
                self._pf,
                *self._dn,
                jnp.asarray(x[i : i + chunk]),
                jnp.asarray(t[i : i + chunk]),
            )
            out.append(np.asarray(P).mean(axis=1))
        return np.concatenate(out, axis=0)

    def _backbone_identity(self):
        """What an MF record must declare about this backbone, DERIVED FROM THE MEMBER METAS (never from a caller
        flag): the manifest sha256; the training cache, which every member's meta must name (``cache_sha256``) and
        which must be the lineage block's; the kind; and the simulations every member was trained without. For a
        production bundle that set is empty: ``Bundle`` recomputed the production rule on the metas, which refuses any
        meta that names a proper subset of the simulations. For a research bundle it is each member's excluded set
        (``eligibility._excluded_set``: every simulation set its meta names, under any key), which must be the same
        for every member. Returns {manifest_sha256, kind, cache_sha256, held_out_sims}."""
        from .emulator import eligibility as _E

        metas = list(self._rb.metas)
        block_cache = ((self._rb.lineage or {}).get("cache") or {}).get("sha256")
        caches = sorted({str(m.get("cache_sha256")) for m in metas})
        if not metas or caches != [str(block_cache)]:
            raise _MR.MFRecordError(
                f"the members were not all trained on the bundle's training cache (metas name "
                f"{[c[:12] for c in caches]})"
            )
        research = isinstance(self._rb, _H.ResearchBundle)
        if not research:
            return {
                "manifest_sha256": self._rb.manifest_sha256,
                "kind": "release",
                "cache_sha256": block_cache,
                "held_out_sims": [],
            }
        rs = self._rb.research or {}
        known = list(
            (rs.get("split") or {}).get("held_out_sims") or []
        )  # names to recognise, not the answer
        sets = []
        for i, m in enumerate(metas):
            ex, why, _ = _E._excluded_set(i, m, known, rs.get("wc_map") or {}, None)
            if why:
                raise _MR.MFRecordError(
                    "the members' excluded simulations cannot be derived from their metas: "
                    + "; ".join(why)
                )
            sets.append(sorted(ex))
        if any(x != sets[0] for x in sets):
            raise _MR.MFRecordError(
                f"the members exclude different simulations ({sets}): not one fold backbone"
            )
        return {
            "manifest_sha256": self._rb.manifest_sha256,
            "kind": "research",
            "cache_sha256": block_cache,
            "held_out_sims": sets[0],
        }

    def _heldout_P_filt_at(self, x, tau0, *, chunk=256):
        """The held-out prediction of a FOLD at cache inputs, from this ONE backbone object: {"lf": the members' mean
        LF P_filt, "mf": the members' mean of P_LF exp(g)} (n, 4, K), both on the native modes. The fold record must be
        attached (``load_research_bundle(..., mf_dir=..., mf_record_sha256=...)``), so the backbone that formed the
        record's training targets (its manifest digest, re-derived from the member metas at attach), the one making
        the held-out MF prediction and the one making the LF-alone reference are the same object; and the backbone
        must exclude the held-out group (``backbone_excludes_holdout``, re-derived from the metas), or it is
        refused."""
        if self._mf is None or self._mf.kind != "fold":
            raise _MR.MFRecordError(
                "heldout_P_filt_at needs a fold MF record attached to its research backbone"
            )
        x = _array("x", x, ndim=2)
        t = _array("tau0", tau0, ndim=1)
        if x.shape[1] != 10 or t.shape != (x.shape[0],):
            raise PredictorInputError(
                f"x must be (n, 10) and tau0 (n,), got {x.shape} and {t.shape}"
            )
        self._mf_refuse_low_z(Z_LIMITS[0] + x[:, 9] * (Z_LIMITS[1] - Z_LIMITS[0]))
        lf, mf = [], []
        for i in range(0, x.shape[0], chunk):
            P, _ = _forward_many(
                self._rb.ensemble,
                self._pf,
                *self._dn,
                jnp.asarray(x[i : i + chunk]),
                jnp.asarray(t[i : i + chunk]),
            )
            P = np.asarray(P)  # (n, M, 4, K)
            g = np.stack(
                [
                    np.asarray(_mf_g(self._mf_module, jnp.asarray(x[j]), jnp.asarray(t[j])))
                    for j in range(i, min(i + chunk, x.shape[0]))
                ]
            )
            lf.append(P.mean(axis=1))
            mf.append((P * np.exp(g)[:, None, :, :]).mean(axis=1))
        return {"lf": np.concatenate(lf, axis=0), "mf": np.concatenate(mf, axis=0)}

    # ------------------------------------------------------------------ the MF correction
    _mf = None
    _release = None  # set only by load_release: {format, release_version, sha256}

    def _attach_mf(self, mf, *, kind):
        """Bind a loaded MF record to this backbone: it must come from ``mf_release.load_mf_record`` (anchored), and
        what is applied is re-derived from its sealed bytes (``mf_release.reverify``: an object built or changed in
        memory is refused), so the provenance describes the tables applied. It must be of ``kind`` and declare exactly
        this backbone's identity as RE-DERIVED from the member metas (``_backbone_identity``: manifest, training
        cache, kind, held-out simulations); its modes must be the bundle's. A fold record must hold out a group its
        backbone was trained without (``backbone_excludes_holdout``)."""
        if not isinstance(mf, _MR.MFRecord) or mf.anchored is not True:
            raise _MR.MFRecordError(
                "the predictor takes an MF record loaded by mf_release.load_mf_record"
            )
        mf = _MR.reverify(mf)
        rec = copy.deepcopy(mf.record)
        if rec["kind"] != kind or mf.kind != kind:
            why = {
                "historical": " (historical only: never applied as the release MF)",
                "fold": " (a held-out fold fit: load it on its research backbone, load_research_bundle)",
            }
            raise _MR.MFRecordError(
                f"a {rec['kind']} MF record cannot be applied here (this predictor applies a "
                f"{kind} record){why.get(rec['kind'], '')}"
            )
        b = rec["backbone"]
        if b["manifest_sha256"] != self._rb.manifest_sha256:
            raise _MR.MFRecordError(
                f"the MF record is bound to the backbone {b['manifest_sha256'][:12]}, not this "
                f"bundle {self._rb.manifest_sha256[:12]}"
            )
        ident = self._backbone_identity()
        if (
            b["cache_sha256"] != ident["cache_sha256"]
            or rec["caches"]["lf"]["sha256"] != ident["cache_sha256"]
        ):
            raise _MR.MFRecordError("the MF record's LF cache is not this bundle's training cache")
        if b["kind"] != ident["kind"] or b["held_out_sims"] != ident["held_out_sims"]:
            raise _MR.MFRecordError(
                f"the MF record declares a {b['kind']} backbone holding out {b['held_out_sims']}; "
                f"the member metas say {ident['kind']} holding out {ident['held_out_sims']}"
            )
        if kind == "fold":
            ho = rec["fold"]["holdout_sims"]
            if (
                not set(ho) <= set(ident["held_out_sims"])
                or rec["fold"]["backbone_excludes_holdout"] is not True
            ):
                raise _MR.MFRecordError(
                    f"refusing the fold {rec['fold']['fold_id']!r}: its backbone was trained on "
                    f"held-out simulations {sorted(set(ho) - set(ident['held_out_sims']))} (the "
                    "member metas exclude only "
                    f"{ident['held_out_sims']}): not a held-out "
                    "prediction"
                )
        k = np.asarray(mf.k_com_hmpc, dtype=np.float64)
        if k.shape != self._k_com.shape or not np.allclose(k, self._k_com, rtol=1e-12, atol=0):
            raise _MR.MFRecordError("the MF record's box modes differ from the bundle's modes")
        self._mf = mf
        self._mf_module = mf.module()
        t = mf.tables
        self._mf_dom = (
            np.asarray(t["z_tab"], dtype=np.float64),
            np.asarray(t["tau_by_z"], dtype=np.float64),
        )
        # validated_z_min: the lower edge of the redshift range over which the MF correction is validated (the frozen
        # grouped-validation domain); at build time this is the domain edge, not a passed validation.
        self._mf_vzmin = float(rec["validated_z_min"])

    def _use_mf(self, fidelity):
        if fidelity is None:
            return self._mf is not None
        if fidelity not in FIDELITIES:
            raise PredictorInputError(f"fidelity must be None, 'LF' or 'LF+MF', got {fidelity!r}")
        if fidelity == "LF+MF" and self._mf is None:
            raise MFUnavailableError(
                "fidelity 'LF+MF': this bundle carries no MF record (load it with mf_dir and "
                "mf_record_sha256)"
            )
        return fidelity == "LF+MF"

    def _mf_refuse_low_z(self, zs):
        z0, vz = float(self._mf_dom[0][0]), self._mf_vzmin
        lo = max(z0, vz)
        low = sorted({round(float(z), 9) for z in zs if z < lo - Z_TOL})
        if low:
            raise MFRangeError(
                f"the MF is refused below z = {lo:g} (its tables' lowest redshift {z0:g} and the "
                f"record's validated_z_min {vz:g}; asked at z = {low}); it is not extrapolated or "
                "clamped there: ask for fidelity='LF'"
            )

    def _mf_flags(self, zs, taus, use_mf):
        z_tab, tau_by_z = self._mf_dom
        zc = [
            z > z_tab[-1] + Z_TOL for z in zs
        ]  # below the tables the MF is refused, never clamped
        tc = []
        for z, t in zip(zs, taus):
            ladder = np.array(
                [np.interp(z, z_tab, tau_by_z[:, r]) for r in range(tau_by_z.shape[1])]
            )
            tc.append(bool(t < ladder.min() or t > ladder.max()))
        # above the grouped-validation domain the applied MF is served unchanged but flagged (an LF output is not)
        above = bool(use_mf) and any(z > _MR.VALIDATED_Z_MAX + Z_TOL for z in zs)
        return {
            "mf_applied": bool(use_mf),
            "mf_z_clamped": bool(any(zc)),
            "mf_tau0_clamped": bool(any(tc)),
            "mf_z_above_validated": bool(above),
        }

    def _mf_labels(self, labels, use_mf):
        out = {
            "mf": "the MF correction P_MF = P_LF exp(g(z, tau0, k)) of the loaded record (theta-independent, "
            "fitted on matched HR/LF pairs; refused below max(its lowest z, its validated_z_min); above its "
            "highest z and outside its tau0 tables it is held constant and flagged; above z "
            f"{_MR.VALIDATED_Z_MAX:g}, the upper edge of the frozen grouped-validation domain (z 2.2-4.6; the held-out "
            "validation is reported separately), it is applied unchanged but "
            "flagged mf_z_above_validated); w_c stays the LF backbone's"
        }
        if use_mf:
            out["product"] = f"{labels.get('product') or 'LF'}+MF"
            out["P_class"] = labels["P_class"] + ", each multiplied by exp(g) before the mean"
            out["mf_correction"] = (
                "applied: the multi-fidelity (LF to HR) correction of the loaded MF record, "
                "P_MF = P_LF exp(g), on a single-fidelity LF backbone"
            )
        return out

    def labels(self):
        return {
            "filter": "PRIYA tau > 1e6 trough-filtered",
            "product": self._fidelity,
            "mf_correction": "none: single-fidelity members of one training cache; no multi-fidelity (LF to HR) "
            "correction is applied",
            "resolution": "not resolution-corrected",
            "dla": "the filtered DLA spectrum; the damping-wing core is not included",
            "P_class": "the mean over the ensemble members of their reconstructed per-class spectra",
            "member_std": "seed-spread diagnostic: the standard deviation over the members (ddof 1); not a "
            "calibrated emulator error and not a covariance",
            "w_c": "an estimate of the simulation's own sightline fractions (LLS, subDLA, DLA) from the members' "
            "mean Head-A dN/dX through the w_c map shipped in this bundle",
            "normalisation": "each class spectrum is normalised by the snapshot's global mean flux (the whole "
            "sample's, shared by the four classes), as in the training cache: the settled "
            "production convention; normalising each class by its own mean flux is a "
            "diagnostic only, not used here",
            "amplitudes": "add_hcd: A_c is an effective sightline fraction; with counted fractions and under the "
            "class normalisation above, sum_c w_c P_c is the full-sample filtered P1D, and w_c here "
            "is the map's estimate of those fractions; only A_c >= 0 and 1 - sum_c A_c >= 0 are "
            "imposed; no prior",
            "omissions": list(OMISSIONS),
        }

    # ------------------------------------------------------------------ predictions
    def predict(self, params, z, *, tau_eff=None, tau0_amp=None, dtau0=None, k=None, fidelity=None):
        """Predict the per-class P1D at one snapshot redshift.

        Parameters
        ----------
        params : dict
            The nine PRIYA parameters by name (``PARAM_NAMES``): ``ns`` and ``Ap``, the slope and
            amplitude of the primordial power spectrum at the pivot 0.78 Mpc^-1; ``herei`` and
            ``heref``, the redshifts at which helium reionization starts and ends; ``alphaq``, the
            quasar spectral index that sets the heating of helium reionization; ``hub``, the
            dimensionless Hubble parameter h; ``omegamh2``, Omega_m h^2; ``hireionz``, the
            redshift of hydrogen reionization; ``bhfeedback``, the black-hole feedback strength.
        z : float
            A PRIYA snapshot redshift: 2.0, 2.2, ..., 5.4.
        tau_eff : float, optional
            The effective optical depth of the PRIYA-filtered forest (see ``tau_eff_priya``). Give
            either ``tau_eff``, or both ``tau0_amp`` and ``dtau0``.
        tau0_amp, dtau0 : float, optional
            PRIYA's mean-flux parameters, converted with ``tau_eff_priya``.
        k : array_like, optional
            Wavenumbers in s/km, shape (K',), inside the native mode span at this redshift. The
            prediction is interpolated linearly between modes. None (the default) returns the native
            modes, which depend on ``hub`` and ``omegamh2``.
        fidelity : {None, "LF", "LF+MF"}, optional
            "LF" gives the low-fidelity prediction; "LF+MF" applies the MF correction. None (the
            default) gives "LF+MF" when an MF record is loaded, and "LF" otherwise.

        Returns
        -------
        Prediction
            With ``stacked`` False; ``P_class`` has shape (4, K).

        Raises
        ------
        PredictorInputError
            If an input has the wrong type, shape or name, or is not finite; if ``z`` is not a
            snapshot redshift; if both or neither mean-flux forms are given, or ``tau_eff`` is not
            positive; or if ``k`` lies outside the native mode span (the message gives the span in
            s/km).
        MFUnavailableError
            If ``fidelity`` is "LF+MF" and no MF record is loaded.
        MFRangeError
            If an LF+MF prediction is asked for below the lowest redshift of the MF record. Ask for
            ``fidelity="LF"`` there.
        TypeError
            If an input is a JAX tracer. This method takes ordinary values; to differentiate, use
            ``hcd_analysis.emulator.predict``.

        Notes
        -----
        Values outside the training box are predicted, not refused, so read ``flags`` before using
        the numbers:

        - ``outside_training_box``: a parameter lies outside the box the model was trained on;
          ``params_outside_training_box`` names them. ``outside_sampling_box``: inside the training
          box but outside the original PRIYA design box.
        - ``ns_ridge``: ``ns`` is above 0.995, where the PRIYA design has few simulations.
        - ``mean_flux_off_ladder``: ``tau_eff`` is outside the mean-flux range of the training data.
        - ``z_downweighted`` and ``k_downweighted``: ``z`` is outside 2.2-4.6, or some ``k`` is
          below 1e-3 s/km, where training gave less weight.
        - ``w_c_xbar_extrapolated``: the w_c map is used outside the redshifts it was fitted on.
        - With an MF record: ``mf_applied``; ``mf_z_clamped`` and ``mf_tau0_clamped``, the
          correction is held constant beyond its tables; ``mf_z_above_validated``, an LF+MF
          prediction above z 4.6, outside the frozen grouped-validation domain (z 2.2-4.6). The
          values there are served unchanged.

        The first call for an input shape compiles the model with JAX; later calls reuse it.
        """
        zg = snapshot_z(z)
        p = self._many(
            params,
            [zg],
            tau_eff,
            tau0_amp,
            dtau0,
            None if k is None else _array("k", k, ndim=1),
            single=True,
            fidelity=fidelity,
        )
        return p

    def predict_many(
        self, params, z, *, tau_eff=None, tau0_amp=None, dtau0=None, k=None, fidelity=None
    ):
        """Predict the per-class P1D at several snapshot redshifts, for one parameter set.

        Parameters
        ----------
        params : dict
            The nine PRIYA parameters by name, as in ``predict``.
        z : sequence of float
            Distinct PRIYA snapshot redshifts, shape (n_z,).
        tau_eff : float or array_like, optional
            One value for every redshift, or one per redshift, shape (n_z,). Give either
            ``tau_eff``, or both ``tau0_amp`` and ``dtau0``.
        tau0_amp, dtau0 : float, optional
            PRIYA's mean-flux parameters, the same at every redshift (``tau_eff_priya``).
        k : array_like, optional
            Wavenumbers in s/km: one array (K',) for every redshift, or one row per redshift,
            shape (n_z, K'). None (the default) returns the native modes.
        fidelity : {None, "LF", "LF+MF"}, optional
            As in ``predict``.

        Returns
        -------
        Prediction
            With ``stacked`` True; ``P_class`` has shape (n_z, 4, K).

        Raises
        ------
        PredictorInputError
            As in ``predict``; also if ``z`` repeats a redshift, or if ``tau_eff`` or ``k`` has a
            different number of rows than ``z``.
        MFUnavailableError, MFRangeError
            As in ``predict``. ``MFRangeError`` is raised if any redshift is below the MF record's
            lowest redshift.
        """
        zs = [snapshot_z(v) for v in np.atleast_1d(_array("z", z, ndim=(0, 1)))]
        if len(set(zs)) != len(zs):
            raise PredictorInputError(f"z repeats a redshift: {zs}")
        kk = None if k is None else _array("k", k, ndim=(1, 2))
        if kk is not None and kk.ndim == 2 and kk.shape[0] != len(zs):
            raise PredictorInputError(f"k has {kk.shape[0]} rows for {len(zs)} redshifts")
        return self._many(params, zs, tau_eff, tau0_amp, dtau0, kk, single=False, fidelity=fidelity)

    def _many(self, params, zs, tau_eff, tau0_amp, dtau0, k, *, single, fidelity=None):
        use_mf = self._use_mf(fidelity)
        if use_mf:
            self._mf_refuse_low_z(zs)
        pv = params_vector(params)
        taus = _taus(zs, tau_eff, tau0_amp, dtau0)
        th = np.asarray(normalize_params(pv), dtype=np.float64)
        zu = (np.asarray(zs, dtype=np.float64) - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
        x = np.concatenate([np.tile(th, (len(zs), 1)), zu[:, None]], axis=1)
        ens = self._rb.ensemble
        if single:
            P, dn = _forward_one(ens, self._pf, *self._dn, jnp.asarray(x[0]), jnp.asarray(taus[0]))
            P, dn = np.asarray(P)[None], np.asarray(dn)[None]
        else:
            P, dn = _forward_many(ens, self._pf, *self._dn, jnp.asarray(x), jnp.asarray(taus))
            P, dn = np.asarray(P), np.asarray(dn)  # (n_z, M, 4, K), (n_z, M, 3)
        if use_mf:  # P_MF = P_LF exp(g), per member, on the modes
            g = np.stack(
                [
                    np.asarray(_mf_g(self._mf_module, jnp.asarray(x[i]), jnp.asarray(taus[i])))
                    for i in range(len(zs))
                ]
            )
            P = P * np.exp(g)[:, None, :, :]
        hub, omh2 = pv[_I_HUB], pv[_I_OMH2]
        modes = np.stack(
            [
                np.asarray(_K.k_skm_from_kcom(self._k_com, z, hub, omh2), dtype=np.float64)
                for z in zs
            ]
        )
        if k is not None:
            kreq = np.broadcast_to(k, (len(zs), k.shape[-1])) if k.ndim == 1 else k
            out = []
            for i, z in enumerate(zs):
                lo, hi = modes[i, 0], modes[i, -1]
                if np.any(kreq[i] < lo) or np.any(kreq[i] > hi):
                    raise PredictorInputError(
                        f"k outside the native mode span [{lo:.6g}, {hi:.6g}] s/km at z = {z} "
                        "for these hub and omegamh2: no prediction exists there"
                    )
                kg = _K.KGrid(
                    jnp.asarray(modes[i]),
                    float(z),
                    jnp.asarray(hub),
                    jnp.asarray(omh2),
                    jnp.asarray(self._k_com),
                    CHECKPOINT_SCHEMA_VERSION,
                )
                b = _K.bind(kg, jnp.asarray(kreq[i]))
                out.append(np.asarray(_K.at_data(b, jnp.asarray(P[i]))))
            P = np.stack(out)
            kout = np.array(kreq, dtype=np.float64)
        else:
            kout = modes
        Pm = P.mean(axis=1)  # (n_z, 4, K)
        std = P.std(axis=1, ddof=1) if P.shape[1] > 1 else None
        dmean = dn.mean(axis=1)  # (n_z, 3)
        w4 = np.stack([self._rb.wc_map.w_c(dmean[i], z) for i, z in enumerate(zs)])
        template = Pm[:, 1:, :] / Pm[:, :1, :] - 1.0
        flags, labels, prov = self._flags(pv, th, zs, taus, kout), self.labels(), self.provenance
        if self._mf is not None:
            flags.update(self._mf_flags(zs, taus, use_mf))
            labels.update(self._mf_labels(labels, use_mf))
        sel = (lambda a: a[0]) if single else (lambda a: a)
        return Prediction(
            z=zs[0] if single else _frozen(zs),
            k=_frozen(sel(kout)),
            P_class=_frozen(sel(Pm)),
            template=_frozen(sel(template)),
            w_c=_frozen(sel(w4[:, 1:])),
            dndx=_frozen(sel(dmean)),
            member_std=None if std is None else _frozen(sel(std)),
            tau_eff=float(taus[0]) if single else _frozen(taus),
            params={n: float(v) for n, v in zip(PARAM_NAMES, pv)},
            flags=flags,
            labels=labels,
            provenance=prov,
            stacked=not single,
        )

    def _flags(self, pv, th, zs, taus, k):
        lim, samp = np.asarray(PARAM_LIMITS, float), np.asarray(SAMPLING_LIMITS, float)
        out_train = bool(np.any(th < 0) or np.any(th > 1))
        out_samp = bool(np.any(pv < samp[:, 0]) or np.any(pv > samp[:, 1]))
        alpha = np.asarray(taus) / np.asarray(kim_tau0(np.asarray(zs)), dtype=np.float64)
        ladder = (
            None
            if self._alpha_range is None
            else bool(np.any(alpha < self._alpha_range[0]) or np.any(alpha > self._alpha_range[1]))
        )
        xb_lo, xb_hi = self._rb.wc_map.xbar_z_fit
        return {
            "ns_ridge": bool(pv[PARAM_NAMES.index("ns")] > NS_RIDGE),
            "outside_training_box": out_train,
            "outside_sampling_box": out_samp and not out_train,
            "params_outside_training_box": [
                n for n, v, (lo, hi) in zip(PARAM_NAMES, pv, lim) if v < lo or v > hi
            ],
            "mean_flux_off_ladder": ladder,
            "z_downweighted": bool(
                any(z < DATA_RANGE["z_lo"] or z > DATA_RANGE["z_hi"] for z in zs)
            ),
            "k_downweighted": bool(np.any(np.asarray(k) < DATA_RANGE["k_min"])),
            "w_c_xbar_extrapolated": bool(any(z < xb_lo or z > xb_hi for z in zs)),
        }


class Bundle(_PredictorBase):
    """A production model, ready to predict. ``load_bundle`` and ``load_release`` return it.

    A bundle holds an ensemble of neural networks (the members), trained on the PRIYA
    low-fidelity simulations, and a map from their predicted HCD incidence to sightline fractions
    (the w_c map). With an MF record, it also applies the multi-fidelity correction to high
    resolution.

    Use ``predict`` for one redshift and ``predict_many`` for several; both return a
    ``Prediction``. ``provenance`` and ``labels()`` describe the loaded model; ``manifest_sha256``
    and ``n_members`` identify it.

    Do not build a ``Bundle`` yourself: load it with ``load_bundle`` or ``load_release``. The
    constructor takes only a bundle loaded by ``safe_h5.load_release`` and checked against its
    published digest. It recomputes, from the bundle's own records, whether the bundle is a
    production bundle, so an object whose stored flags were set by hand does not pass. It does not
    check the network arrays of a hand-built object.

    Raises
    ------
    RuntimeError
        If JAX float64 is off. Importing ``hcd_analysis.predictor`` turns it on.
    SafeFormatError
        If the object is not such a bundle, carries a research marking, or fails the production
        checks. The message lists the reasons.
    """

    def __init__(self, release_bundle, mf=None):
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "the predictor needs JAX float64 (import hcd_analysis.emulator first)"
            )
        rb = release_bundle
        if (
            not isinstance(rb, _H.ReleaseBundle)
            or rb.kind != "release"
            or rb.release_verified is not True
            or rb.anchored is not True
        ):
            raise _H.SafeFormatError(
                "the predictor takes only an anchored hcd_priya.h5/2 release bundle "
                "(safe_h5.load_release: kind release, release_verified); a research bundle loads "
                "with predictor.load_research_bundle"
            )
        prov = rb.manifest.get("provenance") if isinstance(rb.manifest, dict) else None
        if not isinstance(prov, dict) or "research" in prov:
            raise _H.SafeFormatError(
                "the predictor refuses a research bundle (provenance.research) or a provenance "
                "that is not a JSON object: load_research_bundle"
            )
        m = rb.metas
        if not (
            isinstance(m, _H.MemberMetas)
            and m.format == _H.FORMAT
            and m.kind == "release"
            and m.release_verified is True
            and m.manifest_sha256 == rb.manifest_sha256
        ):
            raise _H.SafeFormatError(
                "the bundle's metas are not the release-verified metas of this manifest "
                "(safe_h5.load_release)"
            )
        block = rb.lineage if isinstance(rb.lineage, dict) else {}
        why = (
            _H.lineage_block_reasons(block, _H._release_finder_ids())
            + _H.member_reasons(block, list(rb.metas), n_sims=_H.wc_map_n_sims(rb.wc_map))
            + _H.wc_map_production_reasons(block, rb.wc_map)
        )
        if why or block.get("release_eligible") is not True or block.get("ineligible_reasons"):
            raise _H.SafeFormatError(
                "the bundle is not a production release bundle (eligibility recomputed): "
                + "; ".join(why or ["the block is stated ineligible"])
            )
        from . import lineage as _L

        w, wm = block.get("wc_map") or {}, rb.wc_map
        if (
            getattr(wm, "sha256", None) != w.get("sha256")
            or _L.record_sha256(wm.record) != w.get("lineage_record_sha256")
            or wm.cache_sha256 != (block.get("cache") or {}).get("sha256")
        ):
            raise _H.SafeFormatError("the w_c map is not the one the lineage block names")
        self._setup(rb)
        if mf is not None:
            self._attach_mf(mf, kind="release")


class ResearchPredictor(_PredictorBase):
    """A research model, ready to predict. ``load_research_bundle`` returns it.

    A research model is one made for a study: for example one fold of a cross-validation, trained
    with some simulations held out. It is never a production model and never a ``Bundle``. It
    predicts as a ``Bundle`` does, but every ``Prediction`` it returns has
    ``flags["research_model"]`` True, the declared split in ``flags["research_split"]``, and a
    ``labels["model"]`` line saying it is a research model.

    Attributes
    ----------
    research : dict
        The bundle's research record.
    purpose : str
        The purpose the bundle declares.
    load_purpose : str
        The purpose given when it was loaded.
    split : dict
        A summary of the declared split, including the held-out simulations.

    Raises
    ------
    RuntimeError
        If JAX float64 is off. Importing ``hcd_analysis.predictor`` turns it on.
    SafeFormatError
        If the object is not a research bundle loaded by ``safe_h5.load_research``.
    """

    def __init__(self, research_bundle, mf=None):
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "the predictor needs JAX float64 (import hcd_analysis.emulator first)"
            )
        rb = research_bundle
        if (
            not isinstance(rb, _H.ResearchBundle)
            or rb.kind != "research"
            or rb.anchored is not True
            or getattr(rb.metas, "format", None) != _H.FORMAT
        ):
            raise _H.SafeFormatError(
                "ResearchPredictor takes a research bundle from safe_h5.load_research"
            )
        self._setup(rb)
        if mf is not None:
            self._attach_mf(mf, kind="fold")

    @property
    def research(self):
        return copy.deepcopy(self._rb.research)

    @property
    def purpose(self):
        return self._rb.purpose

    @property
    def load_purpose(self):
        return self._rb.load_purpose

    @property
    def split(self):
        return copy.deepcopy(self._rb.split)

    @property
    def provenance(self):
        return dict(
            super().provenance,
            model_kind="research",
            research=self.research,
            load_purpose=self._rb.load_purpose,
        )

    def labels(self):
        sp = self._rb.split
        return dict(
            super().labels(),
            model=(
                f"a RESEARCH model ({self._rb.purpose}; split {sp['rule']}, held out "
                f"{sp['held_out_sims']}): never a production model"
            ),
        )

    def _flags(self, pv, th, zs, taus, k):
        out = super()._flags(pv, th, zs, taus, k)
        out.update(
            research_model=True,
            research_split=copy.deepcopy(self._rb.split),
            research_content_checks={
                c.get("component"): c.get("outcome")
                for c in self._rb.research.get("content_checks") or []
            },
        )
        return out


def _load_mf(rb, mf_dir, mf_record_sha256, kind):
    if (mf_dir is None) != (mf_record_sha256 is None):
        raise _MR.MFRecordError(
            "give both mf_dir and mf_record_sha256 (the record's published sha256), or neither"
        )
    if mf_dir is None:
        return None
    return _MR.load_mf_record(
        mf_dir,
        expected_record_sha256=mf_record_sha256,
        backbone_manifest_sha256=rb.manifest_sha256,
        kind=kind,
    )


def load_bundle(bundle_dir, expected_manifest_sha256, *, mf_dir=None, mf_record_sha256=None):
    """Load a production model bundle, checked against its published digest.

    A bundle is a directory in the release format ``hcd_priya.h5/2``: the member networks, the
    w_c map and ``manifest.json``. The sha256 of ``manifest.json`` is published with the release.
    The load stops unless the directory matches it.

    Parameters
    ----------
    bundle_dir : str or os.PathLike
        The bundle directory.
    expected_manifest_sha256 : str
        The published sha256 of ``manifest.json``: 64 lowercase hex digits. None is refused.
    mf_dir : str or os.PathLike, optional
        The directory of an MF record to apply. Give it together with ``mf_record_sha256``.
    mf_record_sha256 : str, optional
        The published sha256 of that MF record.

    Returns
    -------
    Bundle

    Raises
    ------
    SafeFormatError
        If the digest is None or malformed; if the directory does not match the manifest, or a file
        fails a format check; if the bundle's records do not show a clean lineage (the absorber
        catalogues, the training cache, and members trained on that cache at one committed code
        version); if a member is not a production member (trained on every simulation of the
        cache); or if it is a research bundle (use ``load_research_bundle``). The message lists the
        reasons.
    MFRecordError
        If only one of ``mf_dir`` and ``mf_record_sha256`` is given; if the record does not match
        its digest; if it belongs to another bundle, training cache or k grid; or if it is a
        historical or a fold record.

    Notes
    -----
    A typical call, with placeholder path and digest::

        import hcd_analysis.predictor as hp
        emu = hp.load_bundle("path/to/bundle", "<published manifest sha256>")
        pred = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0)

    For a whole release (the bundle and its MF record), use ``load_release``. A bundle loaded
    alone is the low-fidelity model: ``labels["mf_correction"]`` says so.
    """
    if expected_manifest_sha256 is None:
        raise _H.SafeFormatError(
            "load_bundle needs the published manifest sha256; None is refused (an unanchored "
            "development load is safe_h5.load_ensemble(..., expected_manifest_sha256=None))"
        )
    rb = _H.load_release(bundle_dir, expected_manifest_sha256=expected_manifest_sha256)
    return Bundle(rb, mf=_load_mf(rb, mf_dir, mf_record_sha256, "release"))


def load_release(release_dir, expected_release_sha256):
    """Load a whole release, the backbone bundle and its MF record, from one published digest.

    A release directory holds ``release.json`` (format ``hcd_priya.release/1``) and the two parts
    it names with their sha256: the backbone bundle and the MF record. The one published sha256,
    that of ``release.json``, anchors the whole release.

    Parameters
    ----------
    release_dir : str or os.PathLike
        The release directory, the one that holds ``release.json``.
    expected_release_sha256 : str
        The published sha256 of ``release.json``: 64 lowercase hex digits.

    Returns
    -------
    Bundle
        With the MF record applied. ``provenance["release"]`` names the release.

    Raises
    ------
    ReleaseManifestError
        If the digest is malformed or differs from that of ``release.json``; if the directory is
        not a release directory or is a symbolic link; or if the manifest is malformed (another
        format, an unknown key, or no MF part).
    SafeFormatError, MFRecordError
        As in ``load_bundle``, if the backbone or the MF record fails its checks.

    Notes
    -----
    A typical call, with placeholder path and digest::

        import hcd_analysis.predictor as hp
        emu = hp.load_release("path/to/release", "<published release sha256>")
        pred = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0)   # LF+MF by default
        lf = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0, fidelity="LF")

    A bundle without an MF record loads with ``load_bundle``.
    """
    from .emulator import release_manifest as _RM

    m, parts = _RM.read_release_manifest(release_dir, expected_release_sha256)
    b = load_bundle(
        parts["backbone"],
        m["backbone"]["manifest_sha256"],
        mf_dir=parts["mf"],
        mf_record_sha256=m["mf"]["record_sha256"],
    )
    b._release = {
        "format": m["format"],
        "release_version": m["release_version"],
        "release_sha256": expected_release_sha256,
    }
    return b


def load_research_bundle(
    bundle_dir, expected_manifest_sha256, *, purpose, mf_dir=None, mf_record_sha256=None
):
    """Load a research model bundle, checked against its manifest digest.

    A research bundle holds models made for a study, such as cross-validation folds. It is never
    loaded as a production model. You must say why you load it: ``purpose`` is logged and
    recorded in the provenance of every prediction.

    Parameters
    ----------
    bundle_dir : str or os.PathLike
        The bundle directory.
    expected_manifest_sha256 : str
        The sha256 of its ``manifest.json``: 64 lowercase hex digits.
    purpose : str
        Why this research model is loaded; not empty.
    mf_dir : str or os.PathLike, optional
        The directory of a fold MF record fitted on this bundle, for held-out predictions in a
        grouped cross-validation. Give it together with ``mf_record_sha256``.
    mf_record_sha256 : str, optional
        The sha256 of that MF record.

    Returns
    -------
    ResearchPredictor

    Raises
    ------
    ValueError
        If ``purpose`` is empty.
    SafeFormatError
        If the digest is malformed; if the directory or its records fail a check; or if the bundle
        is a production bundle (use ``load_bundle``).
    MFRecordError
        If only one of ``mf_dir`` and ``mf_record_sha256`` is given; if the record is not a fold
        record of this bundle; or if its held-out simulations were used to train this bundle.
    """
    rb = _H.load_research(
        bundle_dir, expected_manifest_sha256=expected_manifest_sha256, purpose=purpose
    )
    return ResearchPredictor(rb, mf=_load_mf(rb, mf_dir, mf_record_sha256, "fold"))


def _predictor(pred, what):
    if not isinstance(pred, _PredictorBase):
        raise TypeError(f"{what} takes a loaded predictor (load_bundle or load_research_bundle)")
    return pred


def native_modes(pred):
    """The predictor's comoving box modes 2 pi n / L (h/Mpc): the axis of ``P_class`` before any k placement."""
    return _predictor(pred, "native_modes")._k_com.copy()


def backbone_identity(pred):
    """What an MF record must declare about this backbone, derived from its member metas
    (``_PredictorBase._backbone_identity``): {manifest_sha256, kind, cache_sha256, held_out_sims}.
    """
    return _predictor(pred, "backbone_identity")._backbone_identity()


def lf_P_filt_at(pred, x, tau0):
    """The backbone's LF per-class P_filt (n, 4, K) on the native modes at encoder inputs ``x`` (n, 10) and mean-flux
    inputs ``tau0`` (n,), as ``predict`` forms ``P_class`` without the MF (the LF side of the MF targets).
    """
    return _predictor(pred, "lf_P_filt_at")._lf_P_filt_at(x, tau0)


def heldout_P_filt_at(pred, x, tau0):
    """A fold's held-out prediction at cache inputs from ONE backbone object, {"lf", "mf"} (n, 4, K); needs a fold
    record attached to a backbone that excludes the held-out group (``_PredictorBase._heldout_P_filt_at``).
    """
    return _predictor(pred, "heldout_P_filt_at")._heldout_P_filt_at(x, tau0)


def add_hcd(pred, A):
    """Add the HCD classes to the clean P1D, with the given amplitudes.

    Returns P_clean + sum_c A_c (P_c - P_clean), where c runs over LLS, subDLA and DLA, and A_c is
    the effective sightline fraction of class c. With ``A = pred.w_c`` this is the model's
    estimate of the full-sample P1D, to the accuracy of the w_c map.

    Parameters
    ----------
    pred : Prediction
        From ``predict`` or ``predict_many``.
    A : array_like
        Shape (3,): [A_LLS, A_subDLA, A_DLA]. On a stack, shape (3,) applies the same amplitudes at
        every redshift, and shape (n_z, 3) gives one row per redshift.

    Returns
    -------
    numpy.ndarray
        The P1D in km/s, shape (K,), or (n_z, K) on a stack. It carries no flags: keep
        ``pred.flags`` with it.

    Raises
    ------
    TypeError
        If ``pred`` is not a ``Prediction``.
    PredictorInputError
        If ``A`` has the wrong shape or is not finite, an A_c is negative, or 1 - sum_c A_c is
        negative. The message names the class and, on a stack, the redshift.

    Notes
    -----
    No prior is applied, only A_c >= 0 and 1 - sum_c A_c >= 0. Two contributions are not
    included: the power of the subDLA cores that the PRIYA filter removes, and the DLA
    damping-wing core.

    Examples
    --------
    A hand-made ``Prediction`` with synthetic spectra, to show the arithmetic:

    >>> import numpy as np
    >>> import hcd_analysis.predictor as hp
    >>> P = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [5.0, 5.0]])  # clean, LLS, subDLA, DLA
    >>> pred = hp.Prediction(z=3.0, k=np.array([0.01, 0.02]), P_class=P, template=P[1:] / P[0] - 1,
    ...                      w_c=np.zeros(3), dndx=np.zeros(3), member_std=None, tau_eff=0.5,
    ...                      params={}, flags={}, labels={}, provenance={}, stacked=False)
    >>> hp.add_hcd(pred, [0.1, 0.0, 0.0])
    array([1.1, 1.1])
    """
    if not isinstance(pred, Prediction):
        raise TypeError("add_hcd takes a Prediction from Bundle.predict or Bundle.predict_many")
    A = _array("A", A)
    P = np.asarray(pred.P_class)
    if not pred.stacked:
        if A.shape != (3,):
            raise PredictorInputError(
                f"one amplitude per HCD class {HCD_CLASSES}: A must have shape (3,), got "
                f"{A.shape}"
            )
        zs = [pred.z]
    else:
        n_z = P.shape[0]
        if A.shape == (3,):
            A = np.broadcast_to(A, (n_z, 3))
        elif A.shape != (n_z, 3):
            raise PredictorInputError(
                f"A must have shape (3,) or (n_z, 3) = ({n_z}, 3), got {A.shape}"
            )
        zs = list(np.asarray(pred.z))
    check_amplitudes(A, zs)
    P_clean = P[..., 0, :]
    return P_clean + np.einsum("...c,...ck->...k", A, P[..., 1:, :] - P_clean[..., None, :])
