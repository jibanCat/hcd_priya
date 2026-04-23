"""
Shared helpers for the hcd_priya analysis scripts.

Primary use: keep large HDF5/CSV artifacts OUT of the git repository.
Scripts import `data_dir()` to get the directory where summary tables
and intermediate pickles live.  By default this points at scratch;
override with the `HCD_DATA_DIR` environment variable to relocate
(e.g. to Turbo for long-term storage).

Keeping one source of truth here means we can move the artifacts later
without touching every analysis script.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default: scratch storage scoped to the HCD analysis campaign.
# To relocate (e.g. Turbo), export HCD_DATA_DIR in the shell.
_DEFAULT_DATA_DIR = Path(
    "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/_hcd_analysis_data"
)


def data_dir() -> Path:
    """Return the data directory.  Creates it if missing.

    Resolution order:
      1. $HCD_DATA_DIR if set.
      2. The default scratch path above.

    The directory is NOT under version control — see `.gitignore`.
    Scripts should read and write all HDF5/CSV artifacts through this
    helper so relocating storage is a one-line change.
    """
    env = os.environ.get("HCD_DATA_DIR")
    p = Path(env) if env else _DEFAULT_DATA_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p
