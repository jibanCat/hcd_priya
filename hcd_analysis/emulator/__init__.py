"""Phase-2b JAX/Equinox HCD P1D + CDDF emulator.

See hcd_priya_notes/docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md and
hcd_priya_notes/docs/superpowers/plans/2026-06-01-phase2b-B-equinox-emulator.md.

The whole emulator pipeline runs in float64: the structural identities
(P_tier_p = Sum_c w_c*P_filt, telescoping w_c sum-to-1) are bit-level and break
under JAX's default float32. Enabling x64 here ensures every import of the
package (code and tests) uses double precision.
"""
import jax as _jax
_jax.config.update("jax_enable_x64", True)

CLASS_NAMES = ("clean", "LLS", "subDLA", "DLA")  # coarse 4-class order
N_CLASSES = 4
HCD_CLASSES = ("LLS", "subDLA", "DLA")  # the 3 with a non-zero Delta_c
