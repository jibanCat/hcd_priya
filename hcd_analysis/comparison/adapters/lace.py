"""LaCE adapter: BLOCKED. It returns a schema-valid record whose every case is BLOCKED with the reason, so a LaCE
comparison can never be counted as run, let alone passed, until the blockers in the reason are resolved."""
from __future__ import annotations

from ..schema import SCHEMA, validate_record


def blocked_record(cases, *, pins, n_modes, L_box_hmpc):
    return validate_record({
        "schema": SCHEMA, "observable": "O1", "L_box_hmpc": float(L_box_hmpc), "modes": list(range(1, n_modes + 1)),
        "comparator": {"id": "R3", "name": pins["name"], "repo": pins["repo"], "commit": pins["commit"],
                       "implementation": {}, "artifacts": [], "fidelity": "LF", "training": "not loaded", "env": {}},
        "cases": [{"case_id": c["case_id"], "family": c["family"], "z": c["z"], "state": c["state"],
                   "support": {"status": "BLOCKED", "reason": pins["reason"]}} for c in cases]})
