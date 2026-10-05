"""Gate A, task 5: the training edge-emphasis weight is a function of the comoving modes only.
Incident: hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md"""
import numpy as np
import pytest

from hcd_analysis.emulator.data import edge_emphasis_k_weight


def test_edge_weight_is_identical_for_every_row_grid():
    n = np.arange(1, 50)
    kcom = 2 * np.pi * n / 120.0
    w_ref = edge_emphasis_k_weight(kcom, edge_gain=3.0, lowk_extra=1.0)
    for vbox in (11316.4, 15565.6, 17578.1):               # three rows' velocity widths
        w_row = edge_emphasis_k_weight(2 * np.pi * n / vbox, edge_gain=3.0, lowk_extra=1.0)
        assert np.allclose(w_row, w_ref, rtol=1e-12)        # the profile depends only on log-k position


def test_edge_weight_refuses_a_per_row_grid():
    kcom = 2 * np.pi * np.arange(1, 50) / 120.0
    with pytest.raises(ValueError, match="per-row"):
        edge_emphasis_k_weight(np.stack([kcom, kcom * 1.3]), edge_gain=3.0)


def test_edge_weight_back_compat_uniform_case():
    kcom = 2 * np.pi * np.arange(1, 10) / 120.0
    assert np.all(edge_emphasis_k_weight(kcom, edge_gain=0.0, lowk_extra=0.0) == 1.0)
