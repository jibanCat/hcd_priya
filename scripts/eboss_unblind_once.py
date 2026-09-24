#!/usr/bin/env python3
"""eboss_unblind_once.py -- the GOVERNED unblind-once step for a BLINDED real-data chain directory.

Authority: PI DECISIONS #24 (2026-09-24, notes repo) authorizes exactly ONE eBOSS real-data fit under the
frozen forward and its full unblind-once (Option 2 of the 2026-09-24 three-lanes response). This script is
the "minimal governed full-unblind path": every check below must pass, then the additive blind offset
(derived in memory from ``blind.lock`` via ``hcd_analysis.emulator.blinding``) is SUBTRACTED from the
blinded ``ns`` / ``Ap`` columns of every chain file and the result is written to NEW files next to the
blinded ones. The blinded files are never modified. A stamp file makes the step non-repeatable and lets
any wrapper refuse a second blind fit of the same leg.

Checks (all must pass; any failure REFUSES with exit 3 and writes nothing):
  1. ``--confirm`` equals the leg name recorded in ``<root>.health.json`` (typed confirmation).
  2. ``<root>.health.json`` says ``blinded: true`` and ``blind_params == ["ns", "Ap"]``.
  3. ``SHA256SUMS`` in the chain directory verifies every listed file (the chains are the committed ones).
  4. sha256(``--blind-lock``) equals ``analysis.lock["blinding"]["blind_lock_sha256"]`` (lock of record).
  5. ``--authorization`` file exists and its sha256 equals ``--authorization-sha256`` (the PI ruling of record).
  6. No ``UNBLINDED.stamp`` exists in the chain directory (unblind-once).
  7. No ``<root>.unblinded.*`` output exists.
``--dry-run`` performs checks 1 to 7 and exits 0 printing only ``DRY-RUN OK``; it never derives the offset.

Outputs (formal mode): ``<root>.unblinded.<c>.txt`` (same header/format, ns and Ap columns unblinded),
``<root>.unblinded.paramnames`` (copy), ``UNBLINDED.stamp`` (JSON: utc, leg, operator, code/notes HEADs,
chain sha256s, blind.lock sha256 and seed_str, the APPLIED OFFSET, authorization sha256), and one
append-only line in ``--events-log``. Prints the BLIND-vs-UNBLIND medians of ns / Ap and the offset.

Usage (emu-jax env):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu python3 scripts/eboss_unblind_once.py \
      --chain-dir <dir> --root real_eboss --blind-lock /home/mfho/hcd_priya/blind.lock \
      --analysis-lock /home/mfho/hcd_priya/analysis.lock --authorization <PI-24.md> \
      --authorization-sha256 <hex> --events-log <notes>/artifacts/unblind_events.log --confirm eBOSS [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse(msg):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(3)


def _git_head(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def verify_sha256sums(chain_dir):
    """Verify every entry of ``SHA256SUMS`` (sha256sum -c format). Returns the dict {file: sha}."""
    p = os.path.join(chain_dir, "SHA256SUMS")
    if not os.path.exists(p):
        refuse(f"SHA256SUMS missing in {chain_dir}")
    out = {}
    with open(p) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                refuse(f"malformed SHA256SUMS line: {line!r}")
            want, name = parts[0], parts[1].lstrip("*")
            fp = os.path.join(chain_dir, name)
            if not os.path.exists(fp):
                refuse(f"SHA256SUMS names a missing file: {name}")
            got = sha256_file(fp)
            if got != want:
                refuse(f"sha256 mismatch for {name}")
            out[name] = got
    if not out:
        refuse("SHA256SUMS is empty")
    return out


def read_chain(path):
    """Read a GetDist chain file written by run_real_fit.export_getdist -> (header_names, table)."""
    with open(path) as f:
        first = f.readline()
    if not first.startswith("#"):
        refuse(f"chain file without header: {path}")
    cols = first[1:].split()
    if cols[:2] != ["weight", "minusloglike"]:
        refuse(f"unexpected chain header in {path}: {cols[:2]}")
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] != len(cols):
        refuse(f"column count mismatch in {path}: header {len(cols)} vs data {table.shape[1]}")
    return cols[2:], table


def run_checks(a):
    health_p = os.path.join(a.chain_dir, f"{a.root}.health.json")
    if not os.path.exists(health_p):
        refuse(f"missing {health_p}")
    health = json.load(open(health_p))
    if a.confirm != health.get("leg"):
        refuse(f"--confirm {a.confirm!r} != leg of record {health.get('leg')!r}")
    if health.get("blinded") is not True:
        refuse("health.json does not say blinded: true")
    if list(health.get("blind_params", [])) != ["ns", "Ap"]:
        refuse(f"unexpected blind_params {health.get('blind_params')}")
    sums = verify_sha256sums(a.chain_dir)
    chain_files = sorted(n for n in sums if n.startswith(f"{a.root}.") and n.endswith(".txt")
                         and ".unblinded." not in n)
    if len(chain_files) != int(health.get("n_chains", -1)):
        refuse(f"chain file count {len(chain_files)} != health n_chains {health.get('n_chains')}")
    lock = json.load(open(a.analysis_lock))
    want = lock.get("blinding", {}).get("blind_lock_sha256")
    got = sha256_file(a.blind_lock)
    if not want or got != want:
        refuse("blind.lock sha256 != analysis.lock blinding.blind_lock_sha256 (not the lock of record)")
    if not os.path.exists(a.authorization):
        refuse(f"authorization record missing: {a.authorization}")
    auth_sha = sha256_file(a.authorization)
    if auth_sha != a.authorization_sha256:
        refuse("authorization record sha256 != pin")
    stamp = os.path.join(a.chain_dir, "UNBLINDED.stamp")
    if os.path.exists(stamp):
        refuse(f"{stamp} exists: this leg was already unblinded (unblind-once)")
    for n in chain_files:
        outp = os.path.join(a.chain_dir, n.replace(f"{a.root}.", f"{a.root}.unblinded.", 1))
        if os.path.exists(outp):
            refuse(f"output exists: {outp}")
    pn = os.path.join(a.chain_dir, f"{a.root}.paramnames")
    if not os.path.exists(pn):
        refuse(f"missing {pn}")
    return health, sums, chain_files, auth_sha, stamp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--root", default="real_eboss")
    ap.add_argument("--blind-lock", required=True)
    ap.add_argument("--analysis-lock", required=True)
    ap.add_argument("--authorization", required=True, help="the PI ruling file authorizing this unblind")
    ap.add_argument("--authorization-sha256", required=True)
    ap.add_argument("--events-log", required=True, help="append-only unblind events log (notes repo)")
    ap.add_argument("--confirm", required=True, help="type the leg name (e.g. eBOSS) to confirm the irreversible step")
    ap.add_argument("--operator", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--code-repo", default="/home/mfho/hcd_priya")
    ap.add_argument("--notes-repo", default="/home/mfho/hcd_priya_notes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    health, sums, chain_files, auth_sha, stamp = run_checks(a)
    if a.dry_run:
        print("DRY-RUN OK")
        return 0

    # ---- the irreversible step ----
    sys.path.insert(0, a.code_repo)
    import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
    from hcd_analysis.emulator import blinding as BL
    lockrec = BL.read_blind_lock(a.blind_lock)
    offset = BL.blind_offset(lockrec["seed_str"])

    written = []
    medians = {}
    for n in chain_files:
        names, table = read_chain(os.path.join(a.chain_dir, n))
        draws = table[:, 2:]
        unb = BL.unblind(draws, offset, columns=names)
        out = np.column_stack([table[:, :2], unb])
        outp = os.path.join(a.chain_dir, n.replace(f"{a.root}.", f"{a.root}.unblinded.", 1))
        np.savetxt(outp, out, fmt=["%.8g"] * out.shape[1], header="weight  minusloglike  " + "  ".join(names))
        written.append(outp)
        for p in ("ns", "Ap"):
            j = names.index(p)
            medians.setdefault(p, {"blind": [], "unblind": []})
            medians[p]["blind"].append(draws[:, j]); medians[p]["unblind"].append(unb[:, j])
    pn_src = os.path.join(a.chain_dir, f"{a.root}.paramnames")
    pn_dst = os.path.join(a.chain_dir, f"{a.root}.unblinded.paramnames")
    with open(pn_src) as f, open(pn_dst, "w") as g:
        g.write(f.read())
    written.append(pn_dst)

    utc = datetime.now(timezone.utc).isoformat()
    rec = dict(schema="UNBLINDED.stamp.v1", utc=utc, leg=health["leg"], survey=health.get("survey"), root=a.root,
               operator=a.operator, authorization=os.path.abspath(a.authorization), authorization_sha256=auth_sha,
               code_head=_git_head(a.code_repo), notes_head=_git_head(a.notes_repo),
               blind_lock=os.path.abspath(a.blind_lock), blind_lock_sha256=sha256_file(a.blind_lock),
               blind_lock_seed_str=lockrec["seed_str"], analysis_lock=os.path.abspath(a.analysis_lock),
               analysis_lock_sha256=sha256_file(a.analysis_lock),
               applied_offset={p: float(v) for p, v in offset.items()},
               blinded_chain_sha256={n: sums[n] for n in chain_files},
               unblinded_outputs={os.path.basename(p): sha256_file(p) for p in written},
               note="theta_inferred = theta_shown - offset on ns and Ap only; blinded files untouched; unblind-once")
    with open(stamp, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True); f.write("\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.events_log)), exist_ok=True)
    with open(a.events_log, "a") as f:
        f.write(json.dumps(dict(utc=utc, leg=health["leg"], root=a.root, operator=a.operator, chain_dir=os.path.abspath(a.chain_dir),
                                code_head=rec["code_head"], notes_head=rec["notes_head"], blind_lock_sha256=rec["blind_lock_sha256"],
                                seed_str=rec["blind_lock_seed_str"], authorization_sha256=auth_sha,
                                blinded_chain_sha256=rec["blinded_chain_sha256"]), sort_keys=True) + "\n")
    print(f"UNBLINDED {health['leg']} ({a.root}) at {utc}; stamp {stamp}")
    for p in ("ns", "Ap"):
        b = np.median(np.concatenate(medians[p]["blind"])); u = np.median(np.concatenate(medians[p]["unblind"]))
        print(f"  {p}: blind median {b:.6g} -> unblind median {u:.6g}   (offset {offset[p]:+.6g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
