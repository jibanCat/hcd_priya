"""Production N-seed ensemble wrapper for the emulator forward.

The production emulator is an ensemble of N independently-initialised members trained on
all sims; its prediction is the MEAN of the members' linear ``predict_P_filt`` (taken on
the reconstructed, post-exp P_filt — what ``validate_production_ensemble.py`` means over),
NOT a weight-average (averaging independently-initialised network weights is meaningless).

Because ``P_obs`` is LINEAR in the per-class P_filt (``P_clean + Σ_c α_c·(P_c − P_clean)``),
mean-over-members of P_obs equals P_obs of the mean P_filt — so making
``predict.predict_P_filt`` ensemble-aware (it duck-types on ``.members``) threads the
ensemble through the ENTIRE likelihood (predict_P_obs / predict_excess / the closure
forward) with a single change.
"""
from __future__ import annotations

import equinox as eqx
import numpy as np

from hcd_analysis.emulator import train as T

_NORM_KEYS = ("mu_marg", "sig_marg", "sig_cosmo")


class EnsembleEmulator(eqx.Module):
    """Holds N trained ``Emulator`` members. ``predict.predict_P_filt`` detects this by its
    ``.members`` attribute and returns the mean over members of the reconstructed P_filt.
    It is a pytree (each member is an ``eqx.Module``), so it is jit/grad/vmap-safe."""

    members: tuple

    def __init__(self, members):
        self.members = tuple(members)


def load_ensemble(paths):
    """Load N checkpoints into an ``EnsembleEmulator``.

    Returns ``(ensemble, meta0, norm0)`` mirroring ``train.load_checkpoint`` so callers can
    swap it in for the single-model triple directly.

    Asserts every member shares the SAME P_filt norm (mu_marg/sig_marg/sig_cosmo): the
    members are trained on the same data with the same recipe (only the init seed differs),
    so a differing norm signals a wiring error that would silently corrupt the mean.
    """
    paths = list(paths)
    assert len(paths) >= 1, "load_ensemble needs >=1 checkpoint path"
    members, metas, norms = [], [], []
    for p in paths:
        model, meta, norm = T.load_checkpoint(p)
        members.append(model)
        metas.append(meta)
        norms.append(norm)
    ref = norms[0]["P_filt"]
    for j in range(1, len(norms)):
        for k in _NORM_KEYS:
            assert np.allclose(np.asarray(norms[j]["P_filt"][k]), np.asarray(ref[k]),
                               rtol=0, atol=1e-12), (
                f"member {paths[j]} P_filt['{k}'] differs from member 0 — ensemble "
                "members must share a norm (same training data / recipe)")
    return EnsembleEmulator(members), metas[0], norms[0]
