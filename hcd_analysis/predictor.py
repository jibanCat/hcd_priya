"""The public predictor: PRIYA-filtered per-class P1D from an approved, clean-lineage model bundle, with free HCD
amplitudes and no prior.

    import hcd_analysis.predictor as hp
    emu = hp.load_bundle(bundle_dir, "<the published manifest sha256>")     # None is refused
    pred = emu.predict(params, z=3.0, tau_eff=0.36)                          # or tau0_amp=..., dtau0=...
    pred.k, pred.P_class, pred.template, pred.w_c                            # (K,), (4, K), (3, K), (3,)
    P_tot = hp.add_hcd(pred, pred.w_c)                                       # the simulation's own HCD content
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

**Inputs.** ``params``: a dict of the nine PRIYA parameters (``PARAM_NAMES``): ``ns`` and ``Ap`` at the PRIYA pivot
k_p = 0.78 h/Mpc, ``herei``, ``heref``, ``alphaq``, ``hub``, ``omegamh2``, ``hireionz``, ``bhfeedback``. ``z``: a PRIYA
snapshot redshift (2.0, 2.2, ..., 5.4, matched to 1e-9). The mean flux: either ``tau_eff``, -ln<F> of the PRIYA-filtered
simulated forest (the model's tau0 input), or PRIYA's (``tau0_amp``, ``dtau0``), converted by ``tau_eff_priya``. ``k``:
None for the native modes k_skm(z, hub, omegamh2), or wavenumbers in s/km inside that span (linear interpolation along
the mode axis, kcoord.bind / at_data). Values outside the training box are evaluated and flagged, not refused; n_s
above the original PRIYA design box (``NS_RIDGE``, where the design is sparse) is flagged ``ns_ridge``.

**Outputs** (``Prediction``; host numpy, float64):
  * ``k`` (K,) s/km; ``P_class`` (4, K): clean, LLS, subDLA, DLA, PRIYA-filtered (tau > 1e6 troughs filled), the mean
    over the ensemble members of their reconstructed spectra;
  * ``template`` (3, K) = P_c / P_clean - 1 for LLS, subDLA, DLA;
  * ``w_c`` (3,): the simulation's own sightline fractions (LLS, subDLA, DLA) at (params, z), from the members' mean
    Head-A dN/dX (``dndx``, (3,)) through the w_c map shipped in the bundle (``dndx_wc.w_c_corrected`` with the map's
    delta_c(z) and Xbar(z)); never the historical constants;
  * ``member_std`` (4, K): the standard deviation over the members (ddof 1), a SEED-SPREAD DIAGNOSTIC (five networks
    differing only in their initialisation), not a calibrated emulator error and not a covariance;
  * ``flags``, ``labels`` and ``provenance`` (the manifest sha256, the lineage block, the w_c map's sha256 and the
    manifest's own provenance: the export's identity, software versions, creation time, the source digests).

**HCD.** ``add_hcd(pred, A)`` returns P_clean + sum_c A_c (P_c - P_clean), with one amplitude per class (A of shape (3,))
or per class per redshift ((n_z, 3) on a stack). A_c is the effective sightline fraction of class c: A = w_c
reproduces the simulation's own filtered total sum_c w_c P_c. Only physical validity is imposed: every A_c >= 0 and the
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
from .emulator import safe_h5 as _H
from .emulator.data import DATA_RANGE, PARAM_LIMITS, SAMPLING_LIMITS, Z_LIMITS, normalize_params
from .emulator.mean_flux_model import TAU0_PIVOT_Z, kim_tau0, tau0_alpha_priya
from .emulator.predict import reconstruct_P_filt_jax
from .emulator.schema import CHECKPOINT_SCHEMA_VERSION, L_BOX_HMPC

PARAM_NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback")
CLASSES = ("clean", "LLS", "subDLA", "DLA")
HCD_CLASSES = ("LLS", "subDLA", "DLA")
Z_GRID = tuple(round(2.0 + 0.2 * i, 10) for i in range(18))      # the PRIYA snapshot redshifts 2.0 ... 5.4
Z_TOL = 1e-9
OMISSIONS = ("the subDLA filtered remnant: the power of subDLA cores the PRIYA filter removes is not added back",
             "the DLA damping-wing core: the DLA class is the PRIYA-filtered spectrum, without its core")
_I_HUB, _I_OMH2 = PARAM_NAMES.index("hub"), PARAM_NAMES.index("omegamh2")
# the upper n_s edge of the original PRIYA design box: the design is sparse above it (flag ns_ridge; a diagnostic)
NS_RIDGE = 0.995


class PredictorInputError(ValueError):
    """An input the checked predictor refuses (shape, type, non-finite, off the grid, outside the native k span,
    or an HCD amplitude that is not a physical class fraction)."""


# -------------------------------------------------------------------------------------------------- host checks
def _host(name, v):
    if isinstance(v, jax.core.Tracer):
        raise TypeError(f"{name}: the checked predictor takes host values, not a JAX tracer (it is not traced; "
                        "differentiate the forward through hcd_analysis.emulator.predict instead)")
    return v


def _scalar(name, v):
    _host(name, v)
    if isinstance(v, (bool, np.bool_, str, bytes)) or np.ndim(v) != 0 or np.asarray(v).dtype.kind not in "iuf":
        raise PredictorInputError(f"{name} must be a real scalar, got {v!r}")
    x = float(v)
    if not np.isfinite(x):
        raise PredictorInputError(f"{name} = {x} is not finite")
    return x


def _array(name, v, ndim=None):
    _host(name, v)
    a = np.asarray(v)
    if a.dtype == bool or not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        raise PredictorInputError(f"{name} must be real numbers, got dtype {a.dtype}")
    a = a.astype(np.float64)
    if ndim is not None and a.ndim not in ((ndim,) if isinstance(ndim, int) else ndim):
        raise PredictorInputError(f"{name} must have {ndim} dimension(s), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise PredictorInputError(f"{name} is not finite")
    return a


def params_vector(params):
    """The nine physical parameters, in ``PARAM_NAMES`` order, from a dict holding exactly those names."""
    if not isinstance(params, dict):
        raise PredictorInputError(f"params must be a dict of {PARAM_NAMES}")
    keys = set(params)
    if keys != set(PARAM_NAMES):
        raise PredictorInputError(f"params: missing {sorted(set(PARAM_NAMES) - keys)}, unknown {sorted(keys - set(PARAM_NAMES))}")
    return np.array([_scalar(f"params[{n!r}]", params[n]) for n in PARAM_NAMES], dtype=np.float64)


def snapshot_z(z):
    """``z`` matched to the PRIYA snapshot grid (to ``Z_TOL``); returns the grid value."""
    x = _scalar("z", z)
    hit = [g for g in Z_GRID if abs(x - g) <= Z_TOL]
    if not hit:
        raise PredictorInputError(f"z = {x} is not a PRIYA snapshot redshift {Z_GRID} (tolerance {Z_TOL})")
    return hit[0]


def tau_eff_priya(z, tau0_amp, dtau0, *, z_pivot=TAU0_PIVOT_Z):
    """PRIYA's mean-flux model as the predictor's input: tau_eff(z) = tau0_amp ((1 + z) / (1 + z_pivot))^dtau0
    tau_Kim07(z), the optical depth of the PRIYA-filtered forest (mean_flux_model.tau0_alpha_priya times kim_tau0)."""
    zz = _scalar("z", z)
    return float(tau0_alpha_priya(zz, _scalar("tau0_amp", tau0_amp), _scalar("dtau0", dtau0), z_pivot=z_pivot)
                 * kim_tau0(zz))


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
    """Physical validity of HCD amplitudes A (..., 3) [LLS, subDLA, DLA]: finite, every A_c >= 0, and the clean-class
    fraction 1 - sum_c A_c >= 0. Raises PredictorInputError naming the class (and the redshift); returns A."""
    A = _array("A", A)
    if A.shape[-1:] != (3,):
        raise PredictorInputError(f"A must end in the three HCD classes {HCD_CLASSES}, got shape {A.shape}")
    flat = A.reshape(-1, 3)
    where = (lambda i: "") if zs is None or flat.shape[0] == 1 else (lambda i: f" at z = {zs[i]}")
    for i, row in enumerate(flat):
        for c, a in zip(HCD_CLASSES, row):
            if a < 0:
                raise PredictorInputError(f"A_{c} = {a!r}{where(i)} is negative: an HCD amplitude is a sightline "
                                          "fraction (A_c >= 0)")
        clean = 1.0 - float(np.sum(row))
        if clean < 0:
            raise PredictorInputError(f"the clean-class fraction 1 - sum_c A_c = {clean!r}{where(i)} is negative "
                                      f"(A = {row.tolist()}): the class fractions cannot exceed 1")
    return A


# ------------------------------------------------------------------------------------------------- the forward
def _members_forward(ens, pf, dn_mean, dn_std, x, tau):
    """Per member: the reconstructed per-class P_filt (M, 4, K) and the physical Head-A dN/dX (M, 3)."""
    outs = [m(x, tau) for m in ens.members]
    P = jnp.stack([reconstruct_P_filt_jax(o["P_filt_base"], o["P_filt_resid"], pf) for o in outs])
    dn = jnp.stack([jnp.exp(o["dndx"] * dn_std + dn_mean) for o in outs])
    return P, dn


_forward_one = eqx.filter_jit(_members_forward)
_forward_many = eqx.filter_jit(eqx.filter_vmap(_members_forward, in_axes=(None, None, None, None, 0, 0)))


def _frozen(a):
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclasses.dataclass(frozen=True)
class Prediction:
    """A prediction at one redshift (``stacked`` False: arrays as in the module docstring) or at several (``stacked``
    True: every array gains a leading redshift axis, ``z`` and ``tau_eff`` are (n_z,))."""
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
                raise _H.SafeFormatError(f"member {i}: checkpoint schema {m.get('schema_version')!r}")
            if not np.array_equal(np.asarray(m["param_limits"], float), np.asarray(PARAM_LIMITS, float)):
                raise _H.SafeFormatError(f"member {i}: its parameter box is not data.PARAM_LIMITS (theta would be "
                                         "normalised with another box)")
        k_com = np.asarray(rb.metas[0]["k_com_hmpc"], dtype=np.float64)
        if not np.allclose(k_com, 2.0 * np.pi * np.arange(1, k_com.size + 1) / L_BOX_HMPC, rtol=1e-12, atol=0):
            raise _H.SafeFormatError("the bundle's k_com_hmpc are not the box modes 2 pi n / L_box")
        self._rb = rb
        self._k_com = k_com
        n = rb.norm
        self._pf = {k: jnp.asarray(np.asarray(n["P_filt"][k], dtype=np.float64)) for k in ("mu_marg", "sig_marg",
                                                                                           "sig_cosmo")}
        self._dn = (jnp.asarray(np.asarray(n["dndx"]["mean"], dtype=np.float64)),
                    jnp.asarray(np.asarray(n["dndx"]["std"], dtype=np.float64)))
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
        return {"format": _H.FORMAT, "manifest_sha256": self._rb.manifest_sha256, "lineage": self.lineage,
                "wc_map_sha256": self._rb.wc_map.sha256, "n_members": self.n_members,
                "bundle_provenance": copy.deepcopy(self._rb.manifest.get("provenance"))}

    def labels(self):
        return {"filter": "PRIYA tau > 1e6 trough-filtered", "product": self._fidelity,
                "resolution": "not resolution-corrected",
                "dla": "the filtered DLA spectrum; the damping-wing core is not included",
                "P_class": "the mean over the ensemble members of their reconstructed per-class spectra",
                "member_std": "seed-spread diagnostic: the standard deviation over the members (ddof 1); not a "
                              "calibrated emulator error and not a covariance",
                "w_c": "the simulation's own sightline fractions (LLS, subDLA, DLA) from the members' mean Head-A "
                       "dN/dX through the w_c map shipped in this bundle",
                "amplitudes": "add_hcd: A_c is an effective sightline fraction; A = w_c reproduces the simulation's "
                              "filtered total; only A_c >= 0 and 1 - sum_c A_c >= 0 are imposed; no prior",
                "omissions": list(OMISSIONS)}

    # ------------------------------------------------------------------ predictions
    def predict(self, params, z, *, tau_eff=None, tau0_amp=None, dtau0=None, k=None):
        """The prediction at one snapshot redshift (module docstring)."""
        zg = snapshot_z(z)
        p = self._many(params, [zg], tau_eff, tau0_amp, dtau0, None if k is None else _array("k", k, ndim=1),
                       single=True)
        return p

    def predict_many(self, params, z, *, tau_eff=None, tau0_amp=None, dtau0=None, k=None):
        """The stacked prediction at several snapshot redshifts (one parameter set; ``tau_eff`` a scalar or one value
        per z, or PRIYA's global ``tau0_amp``, ``dtau0``; ``k`` None, one array for every z, or (n_z, K'))."""
        zs = [snapshot_z(v) for v in np.atleast_1d(_array("z", z, ndim=(0, 1)))]
        if len(set(zs)) != len(zs):
            raise PredictorInputError(f"z repeats a redshift: {zs}")
        kk = None if k is None else _array("k", k, ndim=(1, 2))
        if kk is not None and kk.ndim == 2 and kk.shape[0] != len(zs):
            raise PredictorInputError(f"k has {kk.shape[0]} rows for {len(zs)} redshifts")
        return self._many(params, zs, tau_eff, tau0_amp, dtau0, kk, single=False)

    def _many(self, params, zs, tau_eff, tau0_amp, dtau0, k, *, single):
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
            P, dn = np.asarray(P), np.asarray(dn)                            # (n_z, M, 4, K), (n_z, M, 3)
        hub, omh2 = pv[_I_HUB], pv[_I_OMH2]
        modes = np.stack([np.asarray(_K.k_skm_from_kcom(self._k_com, z, hub, omh2), dtype=np.float64) for z in zs])
        if k is not None:
            kreq = np.broadcast_to(k, (len(zs), k.shape[-1])) if k.ndim == 1 else k
            out = []
            for i, z in enumerate(zs):
                lo, hi = modes[i, 0], modes[i, -1]
                if np.any(kreq[i] < lo) or np.any(kreq[i] > hi):
                    raise PredictorInputError(f"k outside the native mode span [{lo:.6g}, {hi:.6g}] s/km at z = {z} "
                                              "for these hub and omegamh2: no prediction exists there")
                kg = _K.KGrid(jnp.asarray(modes[i]), float(z), jnp.asarray(hub), jnp.asarray(omh2),
                              jnp.asarray(self._k_com), CHECKPOINT_SCHEMA_VERSION)
                b = _K.bind(kg, jnp.asarray(kreq[i]))
                out.append(np.asarray(_K.at_data(b, jnp.asarray(P[i]))))
            P = np.stack(out)
            kout = np.array(kreq, dtype=np.float64)
        else:
            kout = modes
        Pm = P.mean(axis=1)                                                  # (n_z, 4, K)
        std = P.std(axis=1, ddof=1) if P.shape[1] > 1 else None
        dmean = dn.mean(axis=1)                                              # (n_z, 3)
        w4 = np.stack([self._rb.wc_map.w_c(dmean[i], z) for i, z in enumerate(zs)])
        template = Pm[:, 1:, :] / Pm[:, :1, :] - 1.0
        flags = self._flags(pv, th, zs, taus, kout)
        sel = (lambda a: a[0]) if single else (lambda a: a)
        return Prediction(
            z=zs[0] if single else _frozen(zs), k=_frozen(sel(kout)), P_class=_frozen(sel(Pm)),
            template=_frozen(sel(template)), w_c=_frozen(sel(w4[:, 1:])), dndx=_frozen(sel(dmean)),
            member_std=None if std is None else _frozen(sel(std)), tau_eff=float(taus[0]) if single else _frozen(taus),
            params={n: float(v) for n, v in zip(PARAM_NAMES, pv)}, flags=flags, labels=self.labels(),
            provenance=self.provenance, stacked=not single)

    def _flags(self, pv, th, zs, taus, k):
        lim, samp = np.asarray(PARAM_LIMITS, float), np.asarray(SAMPLING_LIMITS, float)
        out_train = bool(np.any(th < 0) or np.any(th > 1))
        out_samp = bool(np.any(pv < samp[:, 0]) or np.any(pv > samp[:, 1]))
        alpha = np.asarray(taus) / np.asarray(kim_tau0(np.asarray(zs)), dtype=np.float64)
        ladder = None if self._alpha_range is None else bool(np.any(alpha < self._alpha_range[0])
                                                             or np.any(alpha > self._alpha_range[1]))
        xb_lo, xb_hi = self._rb.wc_map.xbar_z_fit
        return {"ns_ridge": bool(pv[PARAM_NAMES.index("ns")] > NS_RIDGE),
                "outside_training_box": out_train,
                "outside_sampling_box": out_samp and not out_train,
                "params_outside_training_box": [n for n, v, (lo, hi) in zip(PARAM_NAMES, pv, lim) if v < lo or v > hi],
                "mean_flux_off_ladder": ladder,
                "z_downweighted": bool(any(z < DATA_RANGE["z_lo"] or z > DATA_RANGE["z_hi"] for z in zs)),
                "k_downweighted": bool(np.any(np.asarray(k) < DATA_RANGE["k_min"])),
                "w_c_xbar_extrapolated": bool(any(z < xb_lo or z > xb_hi for z in zs))}


class Bundle(_PredictorBase):
    """A loaded production release bundle (``load_bundle``). It takes only a ``safe_h5.ReleaseBundle`` of kind
    "release" that is release-verified and anchored, and it RECOMPUTES the eligibility from the bundle's own block and
    member metas (``safe_h5.lineage_block_reasons`` and ``member_reasons``: the ties and the production rule) instead
    of trusting stored flags; a research marking is refused. A hand-built object therefore cannot pass as production
    by its flags alone (it is not a defence against code that forges the metadata itself)."""

    def __init__(self, release_bundle):
        if not jax.config.jax_enable_x64:
            raise RuntimeError("the predictor needs JAX float64 (import hcd_analysis.emulator first)")
        rb = release_bundle
        if not isinstance(rb, _H.ReleaseBundle) or rb.kind != "release" or rb.release_verified is not True \
                or rb.anchored is not True or getattr(rb.metas, "format", None) != _H.FORMAT:
            raise _H.SafeFormatError("the predictor takes only an anchored hcd_priya.h5/2 release bundle "
                                     "(safe_h5.load_release: kind release, release_verified); a research bundle loads "
                                     "with predictor.load_research_bundle")
        prov = rb.manifest.get("provenance") if isinstance(rb.manifest, dict) else None
        if not isinstance(prov, dict) or "research" in prov:
            raise _H.SafeFormatError("the predictor refuses a research bundle (provenance.research) or a provenance "
                                     "that is not a JSON object: load_research_bundle")
        block = rb.lineage if isinstance(rb.lineage, dict) else {}
        why = _H.lineage_block_reasons(block, _H._release_finder_ids()) + _H.member_reasons(block, list(rb.metas))
        if why or block.get("release_eligible") is not True or block.get("ineligible_reasons"):
            raise _H.SafeFormatError("the bundle is not a production release bundle (eligibility recomputed): "
                                     + "; ".join(why or ["the block is stated ineligible"]))
        self._setup(rb)


class ResearchPredictor(_PredictorBase):
    """A loaded RESEARCH bundle (``load_research_bundle``): the same forward as ``Bundle``, on a
    ``safe_h5.ResearchBundle`` only. It is not a ``Bundle``. Every prediction is flagged ``research_model`` True and
    carries the declared split (``research_split``) and the recorded content-check outcomes
    (``research_content_checks``); the labels say it is a research model. ``research``, ``purpose``,
    ``load_purpose`` and ``split`` describe it."""

    def __init__(self, research_bundle):
        if not jax.config.jax_enable_x64:
            raise RuntimeError("the predictor needs JAX float64 (import hcd_analysis.emulator first)")
        rb = research_bundle
        if not isinstance(rb, _H.ResearchBundle) or rb.kind != "research" or rb.anchored is not True \
                or getattr(rb.metas, "format", None) != _H.FORMAT:
            raise _H.SafeFormatError("ResearchPredictor takes a research bundle from safe_h5.load_research")
        self._setup(rb)

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
        return dict(super().provenance, model_kind="research", research=self.research)

    def labels(self):
        sp = self._rb.split
        return dict(super().labels(), model=(f"a RESEARCH model ({self._rb.purpose}; split {sp['rule']}, held out "
                                             f"{sp['held_out_sims']}): never a production model"))

    def _flags(self, pv, th, zs, taus, k):
        out = super()._flags(pv, th, zs, taus, k)
        out.update(research_model=True, research_split=copy.deepcopy(self._rb.split),
                   research_content_checks={c.get("component"): c.get("outcome")
                                            for c in self._rb.research.get("content_checks") or []})
        return out


def load_bundle(bundle_dir, expected_manifest_sha256):
    """Load an approved release bundle (``hcd_priya.h5/2``) anchored on its published manifest sha256; ``None``, any
    bundle whose clean lineage cannot be established, a research bundle and any member that is not a production
    member (trained on every simulation of the cache) are refused (safe_h5.load_release). Returns a ``Bundle``."""
    if expected_manifest_sha256 is None:
        raise _H.SafeFormatError("load_bundle needs the published manifest sha256; None is refused (an unanchored "
                                 "development load is safe_h5.load_ensemble(..., expected_manifest_sha256=None))")
    return Bundle(_H.load_release(bundle_dir, expected_manifest_sha256=expected_manifest_sha256))


def load_research_bundle(bundle_dir, expected_manifest_sha256, *, purpose):
    """Load a RESEARCH bundle (cross-validation folds, full-simulation research models) anchored on its manifest
    sha256, with a non-empty ``purpose`` (safe_h5.load_research). Returns a ``ResearchPredictor``; a production
    bundle is refused here."""
    return ResearchPredictor(_H.load_research(bundle_dir, expected_manifest_sha256=expected_manifest_sha256,
                                              purpose=purpose))


def add_hcd(pred, A):
    """P_clean + sum_c A_c (P_c - P_clean) for a ``Prediction``: ``A`` of shape (3,) [A_LLS, A_subDLA, A_DLA] (on a
    stack: the same amplitudes at every z) or (n_z, 3) on a stack (one per class per redshift). Only physical validity
    is imposed (``check_amplitudes``: every A_c >= 0, 1 - sum_c A_c >= 0). Returns (K,) or (n_z, K). The subDLA
    filtered remnant and the DLA damping-wing core are not included (module docstring)."""
    if not isinstance(pred, Prediction):
        raise TypeError("add_hcd takes a Prediction from Bundle.predict or Bundle.predict_many")
    A = _array("A", A)
    P = np.asarray(pred.P_class)
    if not pred.stacked:
        if A.shape != (3,):
            raise PredictorInputError(f"one amplitude per HCD class {HCD_CLASSES}: A must have shape (3,), got "
                                      f"{A.shape}")
        zs = [pred.z]
    else:
        n_z = P.shape[0]
        if A.shape == (3,):
            A = np.broadcast_to(A, (n_z, 3))
        elif A.shape != (n_z, 3):
            raise PredictorInputError(f"A must have shape (3,) or (n_z, 3) = ({n_z}, 3), got {A.shape}")
        zs = list(np.asarray(pred.z))
    check_amplitudes(A, zs)
    P_clean = P[..., 0, :]
    return P_clean + np.einsum("...c,...ck->...k", A, P[..., 1:, :] - P_clean[..., None, :])
