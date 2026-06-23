"""Phase-C — production-ensemble emulator wrapper tests.

The production emulator is an N=5 seed ensemble; its prediction is the MEAN of the
members' ``predict_P_filt`` (NOT a weight-average — the members are independently
initialised nets). The SBC/likelihood forward funnels through ``predict.predict_P_filt``,
so making THAT ensemble-aware threads the ensemble through the whole likelihood. This
pins the contract:
  (a) ``predict_P_filt(ensemble, …)`` == mean over members of single-member predict_P_filt
      (the mean is taken on the RECONSTRUCTED linear P_filt, post-exp — what validate_
      production_ensemble.py:59 means over);
  (b) an ensemble of ONE member reduces exactly to that member;
  (c) the ensemble forward (predict_P_obs) is differentiable in θ (NUTS needs ∂/∂θ);
  (d) load_ensemble loads the N=5 production checkpoints, asserts a SHARED norm across
      members, and returns a (model, meta, norm) triple like load_checkpoint.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_ensemble_emulator.py -q
"""
import glob
import os

import hcd_analysis.emulator  # noqa: F401  enables x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.predict import predict_P_filt, predict_P_obs
from hcd_analysis.emulator.ensemble import EnsembleEmulator, load_ensemble

REPO = "/home/mfho/hcd_priya"
PROD_PREFIX = f"{REPO}/checkpoints/final_prod_seed"


def _members(n=3, n_k=10, n_basis=6):
    """n independently-initialised synthetic Emulators (distinct keys = distinct weights)."""
    return [Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(s))
            for s in range(n)]


def _pf(n_k=10, seed=0):
    rng = np.random.default_rng(seed)
    return {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
            "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
            "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}


def test_ensemble_predict_P_filt_is_member_mean():
    n_k = 10
    members, pf = _members(3, n_k), _pf(n_k)
    theta9, z_unit, tau0 = jnp.full(9, 0.5), jnp.asarray(0.5), 0.40
    per = jnp.stack([predict_P_filt(m, theta9, z_unit, tau0, pf) for m in members])
    expected = jnp.mean(per, axis=0)                                   # (4,K)
    got = predict_P_filt(EnsembleEmulator(members), theta9, z_unit, tau0, pf)
    assert got.shape == expected.shape == (4, n_k)
    assert jnp.allclose(got, expected, rtol=0, atol=1e-12), \
        f"ensemble predict != member mean (max |Δ|={float(jnp.max(jnp.abs(got-expected))):.2e})"
    # the mean must be NON-trivial (members actually differ) so the test has teeth
    assert float(jnp.max(jnp.abs(per[0] - per[1]))) > 1e-6


def test_ensemble_of_one_reduces_to_member():
    n_k = 10
    members, pf = _members(1, n_k), _pf(n_k)
    theta9, z_unit, tau0 = jnp.full(9, 0.5), jnp.asarray(0.5), 0.40
    got = predict_P_filt(EnsembleEmulator(members), theta9, z_unit, tau0, pf)
    ref = predict_P_filt(members[0], theta9, z_unit, tau0, pf)
    assert jnp.allclose(got, ref, rtol=0, atol=1e-12)


def test_ensemble_predict_P_obs_differentiable():
    n_k = 10
    members, pf = _members(3, n_k), _pf(n_k)
    z_unit, tau0 = jnp.asarray(0.5), 0.40
    alpha = jnp.asarray([0.1, 0.05, 0.02])
    dla_core = jnp.zeros(n_k)
    ens = EnsembleEmulator(members)

    def scalar(theta9):
        return jnp.sum(predict_P_obs(ens, theta9, z_unit, tau0, alpha, pf, dla_core))

    g = jax.grad(scalar)(jnp.full(9, 0.5))
    assert g.shape == (9,)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0, "ensemble forward has zero θ-gradient"


_have_prod = len(glob.glob(PROD_PREFIX + "*.eqx")) >= 2
prod_gate = pytest.mark.skipif(not _have_prod, reason="production ensemble checkpoints absent")


@prod_gate
def test_load_ensemble_shared_norm_and_members():
    paths = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))
    ens, meta, norm = load_ensemble(paths)
    assert isinstance(ens, EnsembleEmulator)
    assert len(ens.members) == len(paths) >= 2
    # the returned norm is usable as pf_stats (the SBC reads these three keys)
    for k in ("mu_marg", "sig_marg", "sig_cosmo"):
        assert k in norm["P_filt"]
    # ensemble predict == numpy mean of per-member predict (matches validate_production_ensemble)
    pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
    theta9, z_unit, tau0 = jnp.full(9, 0.5), jnp.asarray(0.5), 0.40
    per = np.stack([np.asarray(predict_P_filt(m, theta9, z_unit, tau0, pf))
                    for m in ens.members])
    got = np.asarray(predict_P_filt(ens, theta9, z_unit, tau0, pf))
    assert np.allclose(got, per.mean(axis=0), rtol=0, atol=1e-10)


@prod_gate
def test_load_ensemble_rejects_mismatched_norm(tmp_path):
    """load_ensemble must assert the members share a norm (same training data) — a member
    with a different norm is a wiring error that would silently corrupt the mean."""
    import pickle
    paths = sorted(p[:-4] for p in glob.glob(PROD_PREFIX + "*.eqx"))[:2]
    # copy the 2-member set, then perturb the SECOND member's norm and point at the copy
    import shutil
    local = []
    for i, p in enumerate(paths):
        for ext in (".eqx", ".meta.json", ".norm.pkl"):
            shutil.copy(p + ext, tmp_path / f"m{i}{ext}")
        local.append(str(tmp_path / f"m{i}"))
    with open(local[1] + ".norm.pkl", "rb") as fh:
        nrm = pickle.load(fh)
    nrm["P_filt"]["mu_marg"] = np.asarray(nrm["P_filt"]["mu_marg"]) + 1.0  # perturb
    with open(local[1] + ".norm.pkl", "wb") as fh:
        pickle.dump(nrm, fh)
    with pytest.raises(AssertionError):
        load_ensemble(local)
