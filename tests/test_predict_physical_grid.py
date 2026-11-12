"""Gate A, task 7: one prediction object on the canonical coordinate.
Incident: docs/kgrid_incident_2026-10.md"""

import jax
import numpy as np
import pytest

from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator import predict as PR
from hcd_analysis.emulator import schema as S
from hcd_analysis.emulator.model import Emulator


def _tiny():
    K = 12
    model = Emulator(in_dim=10, n_k=K, n_basis=None, key=jax.random.PRNGKey(0))
    pf = {
        k: np.ones((4, K)) * v for k, v in (("mu_marg", 0.0), ("sig_marg", 1.0), ("sig_cosmo", 1.0))
    }
    from hcd_analysis.emulator.data import PARAM_LIMITS

    meta = {
        "schema_version": "2.0",
        "k_com_hmpc": (2 * np.pi * np.arange(1, K + 1) / S.L_BOX_HMPC).tolist(),
        "param_limits": np.asarray(PARAM_LIMITS).tolist(),
    }
    return model, pf, meta, K


def test_prediction_object_is_coherent_and_bin_indexed():
    model, pf, meta, K = _tiny()
    th = np.full(9, 0.5)
    z = 2.6
    alpha = np.array([0.1, 0.05, 0.01])
    pred = PR.predict_on_physical_grid(model, meta, th, z, 0.3, alpha, pf, np.zeros(K))
    assert pred.k_skm.shape == (K,) and pred.P_obs.shape == (K,) and pred.P_filt.shape == (4, K)
    assert np.allclose(
        np.asarray(pred.k_skm), np.asarray(KC.k_skm_from_theta9(meta["k_com_hmpc"], z, th))
    )
    z_unit = (z - 2.0) / 3.4
    assert np.allclose(
        np.asarray(pred.P_obs),
        np.asarray(PR.predict_P_obs(model, th, z_unit, 0.3, alpha, pf, np.zeros(K))),
    )
    assert pred.schema_version == "2.0" and abs(pred.hub - 0.70) < 1e-12 and pred.z == z


def test_grid_moves_with_cosmology_but_bins_do_not():
    model, pf, meta, K = _tiny()
    z = 2.6
    a = PR.predict_on_physical_grid(
        model, meta, np.full(9, 0.5), z, 0.3, np.zeros(3), pf, np.zeros(K)
    )
    th2 = np.full(9, 0.5)
    th2[6] = 1.0  # omegamh2 to its upper edge, same bins
    b = PR.predict_on_physical_grid(model, meta, th2, z, 0.3, np.zeros(3), pf, np.zeros(K))
    assert not np.allclose(np.asarray(a.k_skm), np.asarray(b.k_skm))
    ratio = float(KC.E_of_z(z, b.hub, b.omegamh2) / KC.E_of_z(z, a.hub, a.omegamh2))
    assert np.allclose(np.asarray(a.k_skm) / np.asarray(b.k_skm), ratio, rtol=1e-12)


def test_v1_meta_is_refused():
    model, pf, meta, K = _tiny()
    old = {"kfkms": meta["k_com_hmpc"]}
    with pytest.raises(S.SchemaCollapseError):
        PR.predict_on_physical_grid(
            model, old, np.full(9, 0.5), 2.6, 0.3, np.zeros(3), pf, np.zeros(K)
        )


def test_differentiable_in_theta_tau0_alpha():
    import jax.numpy as jnp

    model, pf, meta, K = _tiny()

    def f(th, t0, al):
        p = PR.predict_on_physical_grid(model, meta, th, 2.6, t0, al, pf, jnp.zeros(K))
        return jnp.sum(p.P_obs) + jnp.sum(p.k_skm)

    g = jax.grad(f, argnums=(0, 1, 2))(jnp.full(9, 0.5), 0.3, jnp.array([0.1, 0.05, 0.01]))
    assert all(np.all(np.isfinite(np.asarray(x))) for x in g)


# --- B-D9 #18 R1-4: the retired mode-indexed DLA core is refused here, as forward.predict_leg refuses it -----------


@pytest.mark.parametrize("core", ["ones", "one_bin", "nan"])
def test_a_mode_indexed_dla_core_is_refused(core):
    """A core added on the mode axis would sit at k_skm(z, theta) and move in physical k with theta; the DLA core is a
    fixed function of physical k (forward.predict_leg takes it at the data bins and refuses a mode-indexed one).
    """
    model, pf, meta, K = _tiny()
    c = np.zeros(K)
    if core == "ones":
        c[:] = 1.0
    elif core == "one_bin":
        c[K // 2] = 1e-9  # synthetic: any non-zero entry
    else:
        c[0] = np.nan
    with pytest.raises(ValueError, match="forward.predict_leg"):
        PR.predict_on_physical_grid(
            model, meta, np.full(9, 0.5), 2.6, 0.3, np.array([0.1, 0.05, 0.01]), pf, c
        )


@pytest.mark.parametrize("core", [None, "zeros", "scalar_zero"])
def test_a_zero_or_absent_dla_core_is_accepted_and_adds_nothing(core):
    model, pf, meta, K = _tiny()
    th, alpha = np.full(9, 0.5), np.array([0.1, 0.05, 0.01])
    c = {None: None, "zeros": np.zeros(K), "scalar_zero": 0.0}[core]
    pred = PR.predict_on_physical_grid(model, meta, th, 2.6, 0.3, alpha, pf, c)
    ref = PR.predict_P_obs(model, th, (2.6 - 2.0) / 3.4, 0.3, alpha, pf, np.zeros(K))
    assert np.array_equal(np.asarray(pred.P_obs), np.asarray(ref))


def test_a_traced_dla_core_is_refused_not_silently_used():
    """The refusal needs concrete values; a core passed through a jax transformation cannot be checked and is refused."""
    import jax.numpy as jnp

    model, pf, meta, K = _tiny()

    def f(core):
        return jnp.sum(
            PR.predict_on_physical_grid(
                model, meta, jnp.full(9, 0.5), 2.6, 0.3, jnp.zeros(3), pf, core
            ).P_obs
        )

    with pytest.raises(ValueError, match="concrete"):
        jax.grad(f)(jnp.zeros(K))
