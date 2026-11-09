"""LaCE adapter: BLOCKED. It returns a schema-valid record whose observable is UNDEFINED and whose every case is
BLOCKED with the reason, so a LaCE comparison can never be counted as run, let alone passed, until the blockers are
resolved. Besides the configured reasons, no observable has been matched: hcd_priya's O1 is the all-sightline total
including high-column-density absorbers (after the tau > 1e6 trough fill), and how LaCE's training suites and its
cup1d use treat those absorbers has not been established, so neither the observable nor the hcd_priya representation
to compare against (O1, the clean class, or another combination) is defined."""
from __future__ import annotations

from ..schema import SCHEMA, UNDEFINED, validate_record

OBSERVABLE_BLOCKER = ("no matched observable: O1 includes HCD absorbers (1e6 trough fill only); LaCE's HCD content "
                      "and the hcd_priya representation to compare against are not established")


def blocked_record(cases, *, pins, n_modes, L_box_hmpc):
    reason = f"{pins['reason']}; {OBSERVABLE_BLOCKER}"
    return validate_record({
        "schema": SCHEMA, "observable": UNDEFINED, "L_box_hmpc": float(L_box_hmpc),
        "modes": list(range(1, n_modes + 1)),
        "comparator": {"id": "R3", "name": pins["name"], "repo": pins["repo"], "commit": pins["commit"],
                       "implementation": {}, "artifacts": [], "fidelity": "LF", "training": "not loaded", "env": {}},
        "cases": [{"case_id": c["case_id"], "family": c["family"], "z": c["z"], "state": c["state"],
                   "support": {"status": "BLOCKED", "reason": reason}} for c in cases]})
