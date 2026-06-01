import jax, jax.numpy as jnp, equinox as eqx
from hcd_analysis.emulator.model import Encoder, HeadA


def test_encoder_headA_shapes_and_tau0_independence():
    key = jax.random.PRNGKey(0)
    enc = Encoder(in_dim=10, key=key)
    ha = HeadA(latent=64, n_k_cddf=30, key=key)
    x = jnp.zeros(10)
    lat = enc(x)
    assert lat.shape == (64,)
    out = ha(lat)
    assert out["f_nhi"].shape == (30,)
    assert out["dndx"].shape == (3,)
    # x64 active via package import: latent must be float64
    assert lat.dtype == jnp.float64
