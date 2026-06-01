"""Phase-2b JAX/Equinox HCD P1D + CDDF emulator.

See docs/superpowers/specs/2026-05-29-phase2b-emulator-design.md and
docs/superpowers/plans/2026-06-01-phase2b-B-equinox-emulator.md.
"""
CLASS_NAMES = ("clean", "LLS", "subDLA", "DLA")  # coarse 4-class order
N_CLASSES = 4
HCD_CLASSES = ("LLS", "subDLA", "DLA")  # the 3 with a non-zero Delta_c
