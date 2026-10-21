"""Repository root of THIS checkout. Package code must build every in-repo path from here, never from an absolute
literal, so that a second checkout (e.g. a git worktree) loads its own code and products only.
See hcd_priya_notes docs/superpowers/emulator-paper-history/2026-10-05-INCIDENT-kgrid-representation-regression.md."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
