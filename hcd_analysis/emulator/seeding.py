"""Deterministic seed derivation for the production NUTS drivers. ONE implementation, so the
real-fit and joint drivers cannot drift apart.

WHY THIS MODULE EXISTS. Python's builtin ``hash()`` on a str is SipHash-salted PER PROCESS
(PEP 456), so any key folded from ``hash(survey)`` draws a DIFFERENT chain in a fresh
interpreter. A recorded seed then does not reproduce the recorded chain, which silently voids
the reproducibility promise on every blind fit. ``zlib.crc32`` is a fixed function of the bytes:
stable across processes, machines and interpreter versions.

``scripts/run_joint_fit.py`` has always used crc32; ``scripts/run_real_fit.py`` carried the
``hash()`` idiom until the P0 fix (plan-to-unblind section 3A). Both now call ``nuts_fold_int``.

DELIBERATELY DEPENDENCY-FREE: pure stdlib, no jax, no numpy, no package-level import. That
keeps it importable by path in a test subprocess and keeps the derivation auditable in
isolation. Do NOT add imports here.

CHANGING THIS FUNCTION CHANGES EVERY FUTURE CHAIN STREAM. It is a forward-affecting
definition: alter it only as a new lock era.
"""
import zlib

# The 31-bit mask keeps the result inside jax.random.fold_in's non-negative int32 domain.
_FOLD_MASK = 0x7FFFFFFF

# Recorded verbatim in every driver's export meta, so a reader can re-derive the stream from the
# seed alone. Keep in sync with the implementation below; tests/test_seed_determinism.py pins it.
SEED_DERIVATION = ("fold_in(PRNGKey(seed), crc32(label) & 0x7fffffff) then fold_in(chain_id); "
                   "label = survey name (single-leg) or '+'.join(leg_names) (joint)")


def nuts_fold_int(label):
    """Deterministic per-leg fold value for the NUTS key.

    ``label`` is the survey name for a single-leg fit (``"eboss"``, ``"desi"``, ``"ks"``) or the
    joint driver's ``"+".join(leg_names)``. Returns ``crc32(label.encode()) & 0x7fffffff``:
    stable across processes and machines, unlike ``hash(label)``.
    """
    return zlib.crc32(str(label).encode()) & _FOLD_MASK
