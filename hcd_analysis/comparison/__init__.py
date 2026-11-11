"""Cross-emulator comparison harness: one versioned record schema, explicit conventions, metrics and thin adapters.

``schema``      the ``hcd_priya.compare/1`` prediction record, its validator and the record-vs-record comparison
``conventions`` named parameters and the explicit unit / coordinate / mean-flux conversions (numpy only)
``metrics``     r = (P_test - P_ref) / P_ref summaries and the threshold verdict
``adapters``    hcd_priya (emu-jax), lyaemu PRIYA references (run in their own venv), LaCE (BLOCKED stub)

The core modules import only numpy and the standard library, so a reference's own interpreter can write records.
Comparison configurations and results are not part of this package; a run writes them to its own directory.
"""
