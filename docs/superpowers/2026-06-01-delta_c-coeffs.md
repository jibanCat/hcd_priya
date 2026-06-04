# delta_c(z) calibration — frozen coefficients

Measured from 89 LF shards (1060 snap-blocks, 60 sims, z=2.0-5.4) by `scripts/calibrate_delta_c.py`. NOT invented.

`delta_c(z)` = deg-2 polynomial; w_c_corrected = w_M0*(1+delta_c) renormalised.
The fit removes the mean z-trend; the residual **std** is the irreducible
cosmology dependence at fixed z, carried as the per-class w_c **prior width**.

Coeffs are np.polyval order (highest power first), class order ['clean', 'LLS', 'subDLA', 'DLA']:

```python
_DELTA_C_COEFFS = [  # (clean, LLS, subDLA, DLA) x (deg+1); np.polyval order
    [-0.0044601756018632, 0.02941370276821661, -0.04147804040644924],   # clean
    [-7.391610723256435e-06, 0.008526578464137916, -0.03592195195833433],  # LLS
    [-0.0006304124952415818, 0.009001921148434061, -0.03672596701410354],  # subDLA
    [-0.0002204519439413697, 0.0033098470327118175, -0.0245078526302848],  # DLA
]
```

| class | mean δ_c | fit-residual std (prior width) | residual max |
|---|---|---|---|
| clean | +0.0015 | 0.0093 | 0.1443 |
| LLS | -0.0044 | 0.0059 | 0.0732 |
| subDLA | -0.0127 | 0.0032 | 0.0329 |
| DLA | -0.0155 | 0.0016 | 0.0065 |

**Note on the clean `residual max` (0.14):** the large per-class max residuals (clean
0.14, LLS 0.07) are concentrated in **low-z, high-incidence** snap-blocks — exactly the
regime where the diagonal-Poisson `M₀` is weakest (strongest cross-class clustering) and
the counted clean fraction is noisiest. They are a small minority of the 1060 blocks; the
**std** (the prior width) is the robust summary and is dominated by the bulk, not the tail.
Future refinement (Phase 3): a robust/clipped fit or a weak cosmology-dependent `δ_c`
would shrink the tail; not needed for the ≤1% w_c prior width carried into the likelihood.
