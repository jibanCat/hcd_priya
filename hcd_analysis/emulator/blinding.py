"""Parameter-blinding for the REAL-data cosmology fit (A_p, n_s ONLY).

Blinding strategy (memory ``blinding-strategy`` + docs): the real-data fit is PARAMETER-blind
on the two cosmology parameters the measurement reports — the spectral index ``n_s`` and the
forest power amplitude ``A_p`` — via a HIDDEN ADDITIVE offset

    θ_shown = θ_inferred + δ ,     δ_{A_p, n_s} ~ Uniform(−3σ_prior, +3σ_prior),

with δ drawn from a SHA256(project-string + git-commit) seed. The seed string is committed to
``blind.lock``; the offset itself is derived deterministically from it and applied at VIEW time
(it is NOT baked into the chains). Sampler-health diagnostics (R̂, divergences, ESS) and the
NUISANCE parameters (τ₀, α_HCD, a_SiIII, …) stay fully visible — ONLY the A_p / n_s VALUES move.

Why an ADDITIVE offset (not a data-side cosmology shift): a τ₀/nuisance-degeneracy shift of the
DATA would be partly absorbed by the mean-flux/HCD nuisances and so would NOT cleanly hide the
cosmology (memory ``blinding-strategy``). A constant additive offset on the final inferred
posterior columns is a clean, exactly-invertible hide that the analysis CANNOT see through
(every downstream summary is computed on θ_shown), yet unblinds with a single subtraction once
the analysis is frozen.

σ_prior here is the prior STANDARD DEVIATION. The NUTS prior on A_p / n_s is UNIFORM over the
sampling box (``data.SAMPLING_LIMITS``), so σ_prior = (hi − lo)/√12 — the std of that uniform.
A ±3σ offset window is comfortably larger than the expected posterior width (the data constrain
A_p / n_s far better than the prior) so the blind genuinely hides the headline, while staying a
fixed, reproducible, well-defined transform.

This module is PURE / dependency-light (numpy + hashlib + json) so it is trivially unit-testable
and carries no JAX import cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np

from .data import SAMPLING_LIMITS, PARAM_LIMITS  # noqa: F401  (PARAM_LIMITS re-export convenience)
from .inference import PARAM_NAMES

# The two BLINDED cosmology parameters (and ONLY these). Names match inference.PARAM_NAMES /
# the packed-draw column names produced by closure_legb._packed_names_for.
BLIND_PARAMS = ("ns", "Ap")

# Map each blinded param to its index in PARAM_NAMES (= its column index in the θ9 block of the
# packed draws / GetDist sample matrix). Resolved once at import.
_PNAME_IDX = {nm: i for i, nm in enumerate(PARAM_NAMES)}
for _p in BLIND_PARAMS:
    if _p not in _PNAME_IDX:
        raise RuntimeError(f"blind param {_p!r} is not in PARAM_NAMES {PARAM_NAMES}")

# The ±N·σ_prior offset window (N=3 per the locked blinding strategy).
OFFSET_SIGMA_MULTIPLE = 3.0


def prior_sigma(param):
    """σ_prior (std of the UNIFORM NUTS prior) for a blinded param: (hi − lo)/√12 over the
    SAMPLING box (``data.SAMPLING_LIMITS``). A_p / n_s share PARAM_LIMITS == SAMPLING_LIMITS
    rows, so this is unambiguous."""
    i = _PNAME_IDX[param]
    lo, hi = float(SAMPLING_LIMITS[i, 0]), float(SAMPLING_LIMITS[i, 1])
    return (hi - lo) / np.sqrt(12.0)


def _hash_unit(seed_str, salt):
    """Deterministic uniform-[0,1) draw from SHA256(seed_str | salt). The salt makes the two
    params' offsets INDEPENDENT (so revealing/guessing one does not reveal the other). Uses the
    full 256-bit digest as a fixed-point fraction → no modulo bias at this scale."""
    h = hashlib.sha256(f"{seed_str}|{salt}".encode("utf-8")).hexdigest()
    return int(h, 16) / float(1 << 256)


def blind_offset(seed_str):
    """The hidden additive offset δ = {param: value} for the blinded params, derived
    DETERMINISTICALLY from ``seed_str`` (the committed ``blind.lock`` project-string).

    Per param: u = SHA256(seed_str|param) → [0,1); δ = (2u − 1)·(N·σ_prior) ∈ (−Nσ, +Nσ).
    Same seed ⇒ same δ (determinism); δ within ±N·σ_prior by construction; ONLY the blinded
    params get an entry (everything else is untouched downstream)."""
    out = {}
    for p in BLIND_PARAMS:
        u = _hash_unit(seed_str, p)
        half = OFFSET_SIGMA_MULTIPLE * prior_sigma(p)
        out[p] = float((2.0 * u - 1.0) * half)
    return out


def _resolve_cols(columns):
    """Map blinded-param name → its index in a user-supplied ``columns`` list (the GetDist
    .paramnames order, or PARAM_NAMES). Returns {param: col_index} for the blinded params that
    ARE present (a chain that, e.g., dropped a column simply isn't blinded on it — but for the
    production driver all 9 θ are always present)."""
    columns = list(columns)
    idx = {}
    for p in BLIND_PARAMS:
        if p in columns:
            idx[p] = columns.index(p)
    return idx


def _shift(samples, offset, columns, sign):
    """Add (sign=+1) or subtract (sign=−1) the offset on the blinded columns ONLY. ``samples``
    is an (N, P) array; ``columns`` names its P columns. Returns a COPY (never mutates input).
    Asserts the blinded columns are actually present (a silent no-op blind would be a privacy
    bug, not a convenience)."""
    out = np.array(samples, dtype=float, copy=True)
    if out.ndim != 2:
        raise ValueError(f"samples must be 2-D (N,P); got shape {out.shape}")
    if out.shape[1] != len(columns):
        raise ValueError(f"columns ({len(columns)}) != samples width ({out.shape[1]})")
    col_idx = _resolve_cols(columns)
    missing = [p for p in BLIND_PARAMS if p not in col_idx]
    if missing:
        raise ValueError(f"blinded params {missing} absent from columns {columns} — refusing "
                         f"to write a partially-blind artifact")
    for p, j in col_idx.items():
        out[:, j] = out[:, j] + sign * offset[p]
    return out


def apply_blind(samples, offset, columns=PARAM_NAMES):
    """θ_shown = θ_inferred + δ on the blinded columns only. ``columns`` defaults to the θ9
    PARAM_NAMES order (the leading 9 columns of the packed draws); pass the full packed-name
    list when the matrix carries τ₀/α columns too (the extra columns are passed through)."""
    return _shift(samples, offset, columns, +1.0)


def unblind(samples, offset, columns=PARAM_NAMES):
    """θ_inferred = θ_shown − δ on the blinded columns only (the exact inverse of
    ``apply_blind``)."""
    return _shift(samples, offset, columns, -1.0)


# --------------------------------------------------------------------------------------------- #
#  blind.lock  —  the committed SEED (project-string + git-commit). The OFFSET is NOT stored.
# --------------------------------------------------------------------------------------------- #
def git_commit(repo="/home/mfho/hcd_priya"):
    """The current git commit (short SHA) — part of the blind seed string so the offset is
    pinned to a frozen analysis state. Falls back to ``"nogit"`` if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def make_seed_str(project_string, commit=None, repo="/home/mfho/hcd_priya"):
    """The canonical blind seed string = ``"{project_string}@{git-commit}"``. ``commit=None``
    resolves the live HEAD; pass an explicit commit to reproduce a past blind."""
    c = commit if commit is not None else git_commit(repo)
    return f"{project_string}@{c}"


def write_blind_lock(path, project_string, commit=None, repo="/home/mfho/hcd_priya",
                     extra=None):
    """Write ``blind.lock`` (JSON): the project string + git commit + the derived SEED STRING,
    plus metadata. The OFFSET VALUES are DELIBERATELY NOT WRITTEN (the lock is committable; the
    offset is recomputed at unblind time from the seed). Refuses to overwrite an existing lock
    (a re-write would silently change the blind — the seed must be frozen)."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists — the blind seed is frozen; refusing to "
                              f"overwrite (delete it manually only if you intend to re-blind)")
    seed_str = make_seed_str(project_string, commit=commit, repo=repo)
    rec = dict(
        project_string=project_string,
        git_commit=(commit if commit is not None else git_commit(repo)),
        seed_str=seed_str,
        blind_params=list(BLIND_PARAMS),
        offset_sigma_multiple=OFFSET_SIGMA_MULTIPLE,
        prior_sigma={p: prior_sigma(p) for p in BLIND_PARAMS},
        created_utc=datetime.now(timezone.utc).isoformat(),
        note=("BLIND SEED ONLY. The offset is derived from seed_str at view time and is NOT "
              "stored here. Do not record the offset anywhere committable until UNBLINDING."),
    )
    if extra:
        rec.update(extra)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    return rec


def read_blind_lock(path):
    """Read ``blind.lock`` → its dict (incl. ``seed_str``)."""
    with open(path) as f:
        return json.load(f)


def offset_from_lock(path):
    """Convenience: read the lock and return ``blind_offset(seed_str)`` — the hidden δ. Calling
    this PRINTS NOTHING and is only meant for the (single) authorized unblind step / for applying
    the blind in the driver."""
    return blind_offset(read_blind_lock(path)["seed_str"])
