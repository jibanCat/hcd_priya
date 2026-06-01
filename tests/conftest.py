"""Global pytest config: enable JAX float64 before any test creates arrays.

The Phase-2b emulator's structural identities are bit-level and require double
precision (see hcd_analysis/emulator/__init__.py, which also enables x64 on
import). Setting it here at collection time guarantees x64 even if a test
imports jax before importing the emulator package. No effect on the numpy-based
non-emulator tests.
"""
import jax
jax.config.update("jax_enable_x64", True)
