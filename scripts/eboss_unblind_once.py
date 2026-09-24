#!/usr/bin/env python3
"""eboss_unblind_once.py -- the GOVERNED unblind-once step for a BLINDED real-data chain directory (v2, after reviews A/B).

Authority: PI DECISIONS #24 (2026-09-24, notes repo) authorizes exactly ONE eBOSS real-data fit under the frozen forward
and its full unblind-once. Preregistration v1.1 section 4. The additive blind offset is derived in memory from
``blind.lock`` (hcd_analysis.emulator.blinding) and SUBTRACTED from the ``ns`` / ``Ap`` columns of every chain file into
NEW files; the blinded files are never modified; a stamp makes the step non-repeatable.

Checks (ALL must pass; any failure REFUSES with exit 3 and writes nothing):
  1. ``--confirm`` equals the leg in ``<root>.health.json``; ``blinded: true``; ``blind_params == [ns, Ap]``.
  2. ``SHA256SUMS`` verifies every listed file.
  3. ``EXECUTION_RECORD.json`` exists with schema ``eboss_realfit_execution_record.v1``, ``exit_status == 0``,
     ``formal_execution_consumed`` true, and its ``outputs_sha256`` equals every SHA256SUMS entry (the wrapper wrote it):
     this excludes any directory that was not produced by the authorized wrapper run (e.g. the superseded June chains).
  4. The chain directory lies inside ``--notes-repo`` and every payload file is TRACKED and CLEAN there (committed first).
  5. sha256(``--blind-lock``) equals ``analysis.lock["blinding"]["blind_lock_sha256"]``.
  6. ``--authorization`` exists and its sha256 equals ``--authorization-sha256``.
  7. No ``UNBLINDED.stamp``; no ``<root>.unblinded.*`` output.
  8. Health gate (preregistration v1.1 section 3): GREEN or AMBER proceeds (label recorded); RED refuses.
  9. Every chain file parses and unblinds IN MEMORY (so a dry-run covers the whole write path).
``--dry-run`` performs 1 to 9 and prints only ``DRY-RUN OK <GATE>``; it never derives the offset.
Formal mode writes each output to ``*.tmp`` then renames, writes ``UNBLINDED.stamp`` LAST, appends one events-log line.
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

SCHEMA_RECORD = "eboss_realfit_execution_record.v1"
# ---- health gate thresholds (preregistration v1.1 section 3; Reviewer A M4) ----
GATE = dict(
    green=dict(rhat=1.01, ess_bulk=400, ess_tail=400, ebfmi=0.3, treedepth=0.02, rail=0.05),
    amber=dict(rhat=1.02, ess_bulk=200, ess_tail=200, ebfmi=0.3, treedepth=0.05),
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse(msg):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(3)


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def _git_head(repo):
    r = _git(repo, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else "unavailable"


def verify_sha256sums(chain_dir):
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
            if sha256_file(fp) != want:
                refuse(f"sha256 mismatch for {name}")
            out[name] = want
    if not out:
        refuse("SHA256SUMS is empty")
    return out


def read_chain(path):
    with open(path) as f:
        first = f.readline()
    if not first.startswith("#"):
        refuse(f"chain file without header: {path}")
    cols = first[1:].split()
    if cols[:2] != ["weight", "minusloglike"]:
        refuse(f"unexpected chain header in {path}: {cols[:2]}")
    try:
        table = np.loadtxt(path, ndmin=2)
    except Exception as e:  # noqa: BLE001
        refuse(f"cannot parse {path}: {type(e).__name__}")
    if table.shape[1] != len(cols):
        refuse(f"column count mismatch in {path}: header {len(cols)} vs data {table.shape[1]}")
    if not np.isfinite(table).all():
        refuse(f"non-finite value in {path}")
    return cols[2:], table


def nuisance_diagnostics(chain_dir, root):
    """Split-R-hat and ESS (numpyro.diagnostics, raw draws) per nuisance site from ``<root>.nuisance.npz`` plus the rail
    fractions the driver wrote in ``<root>.nuisance.json``. Returns (per_site dict, present flag)."""
    npz = os.path.join(chain_dir, f"{root}.nuisance.npz")
    js = os.path.join(chain_dir, f"{root}.nuisance.json")
    if not os.path.exists(npz):
        return {}, False
    import numpyro.diagnostics as npd
    d = np.load(npz)
    rails = {}
    if os.path.exists(js):
        rails = {k: v for k, v in json.load(open(js)).get("sites", {}).items()}
    out = {}
    for k in d.files:
        x = np.asarray(d[k], float)
        if x.ndim != 2:
            refuse(f"nuisance site {k} is not (C, N)")
        rh = float(npd.split_gelman_rubin(x)) if x.shape[0] >= 2 and x.shape[1] >= 4 else float("nan")
        ess = float(npd.effective_sample_size(x))
        r = rails.get(k, {})
        out[k] = dict(rhat=rh, ess=ess, frac_near_lo=r.get("frac_near_lo"), frac_near_hi=r.get("frac_near_hi"))
    return out, True


def classify_health(health, nuis):
    """GREEN / AMBER / RED per preregistration v1.1 section 3. ``nuis`` = nuisance_diagnostics() dict (may be empty)."""
    g, a = GATE["green"], GATE["amber"]
    rh = float(health.get("rhat_max", np.inf)); eb = float(health.get("ess_bulk_min", 0)); et = float(health.get("ess_tail_min", 0))
    div = int(health.get("n_divergent", 1)); bf = float(health.get("ebfmi_min", 0)); td = float(health.get("treedepth_sat_frac", 1))
    n_rh = max([v["rhat"] for v in nuis.values() if np.isfinite(v["rhat"])] + [0.0])
    n_ess = min([v["ess"] for v in nuis.values() if np.isfinite(v["ess"])] + [np.inf])
    rail_max = max([max(v["frac_near_lo"] or 0.0, v["frac_near_hi"] or 0.0) for v in nuis.values()] + [0.0])
    reasons = []
    def ok(level):
        t = GATE[level]
        conds = [("rhat", rh < t["rhat"]), ("ess_bulk", eb >= t["ess_bulk"]), ("ess_tail", et >= t["ess_tail"]), ("divergences", div == 0),
                 ("ebfmi", bf >= t["ebfmi"]), ("treedepth", td < t["treedepth"]),
                 ("nuisance_rhat", n_rh < t["rhat"]), ("nuisance_ess", n_ess >= t["ess_bulk"])]
        bad = [c for c, v in conds if not v]
        return bad
    bad_green = ok("green")
    rails_ok = rail_max < g["rail"]
    if not bad_green and rails_ok:
        label = "GREEN"
    else:
        bad_amber = ok("amber")
        if not bad_amber:
            label = "AMBER"; reasons = bad_green + ([] if rails_ok else ["rails"])
        else:
            label = "RED"; reasons = bad_amber
    return dict(label=label, reasons=reasons, rhat_max=rh, ess_bulk_min=eb, ess_tail_min=et, n_divergent=div, ebfmi_min=bf,
                treedepth_sat_frac=td, nuisance_rhat_max=n_rh, nuisance_ess_min=(None if np.isinf(n_ess) else n_ess),
                nuisance_rail_max=rail_max, thresholds=GATE)


def run_checks(a):
    chain_dir = os.path.abspath(a.chain_dir)
    health_p = os.path.join(chain_dir, f"{a.root}.health.json")
    if not os.path.exists(health_p):
        refuse(f"missing {health_p}")
    health = json.load(open(health_p))
    if a.confirm != health.get("leg"):
        refuse(f"--confirm {a.confirm!r} != leg of record {health.get('leg')!r}")
    if health.get("blinded") is not True:
        refuse("health.json does not say blinded: true")
    if list(health.get("blind_params", [])) != ["ns", "Ap"]:
        refuse(f"unexpected blind_params {health.get('blind_params')}")
    # 7 (first): unblind-once, checked before anything else so a second invocation reports it plainly
    stamp = os.path.join(chain_dir, "UNBLINDED.stamp")
    if os.path.exists(stamp):
        refuse(f"{stamp} exists: this leg was already unblinded (unblind-once)")
    sums = verify_sha256sums(chain_dir)
    chain_files = sorted(n for n in sums if n.startswith(f"{a.root}.") and n.endswith(".txt") and ".unblinded." not in n)
    if len(chain_files) != int(health.get("n_chains", -1)):
        refuse(f"chain file count {len(chain_files)} != health n_chains {health.get('n_chains')}")
    # 3. the wrapper's execution record (excludes any directory not produced by the authorized run)
    rec_p = os.path.join(chain_dir, "EXECUTION_RECORD.json")
    if not os.path.exists(rec_p):
        refuse("EXECUTION_RECORD.json missing: not a wrapper-produced chain directory")
    rec = json.load(open(rec_p))
    if rec.get("schema") != SCHEMA_RECORD:
        refuse(f"execution record schema {rec.get('schema')!r} != {SCHEMA_RECORD!r}")
    if int(rec.get("exit_status", 1)) != 0 or rec.get("formal_execution_consumed") is not True:
        refuse("execution record does not show a successful, consumed formal run")
    outs = rec.get("outputs_sha256") or {}
    for n, s in sums.items():
        if outs.get(n) != s:
            refuse(f"execution record outputs_sha256 disagrees with SHA256SUMS for {n}")
    # 4. tracked and clean in the notes repo
    notes = os.path.abspath(a.notes_repo)
    if not (chain_dir + os.sep).startswith(notes + os.sep):
        refuse("chain directory is not inside --notes-repo")
    rel = os.path.relpath(chain_dir, notes)
    for n in list(sums) + ["SHA256SUMS", "EXECUTION_RECORD.json"]:
        if _git(notes, "ls-files", "--error-unmatch", os.path.join(rel, n)).returncode != 0:
            refuse(f"not tracked in the notes repo: {n}")
    st = _git(notes, "status", "--porcelain", "--", rel)
    if st.returncode != 0 or st.stdout.strip():
        refuse("chain directory has uncommitted changes in the notes repo")
    # 5, 6
    lock = json.load(open(a.analysis_lock))
    want = lock.get("blinding", {}).get("blind_lock_sha256")
    if not want or sha256_file(a.blind_lock) != want:
        refuse("blind.lock sha256 != analysis.lock blinding.blind_lock_sha256 (not the lock of record)")
    if not os.path.exists(a.authorization):
        refuse(f"authorization record missing: {a.authorization}")
    auth_sha = sha256_file(a.authorization)
    if auth_sha != a.authorization_sha256:
        refuse("authorization record sha256 != pin")
    # 7 (continued)
    outputs = {}
    for n in chain_files:
        outp = os.path.join(chain_dir, n.replace(f"{a.root}.", f"{a.root}.unblinded.", 1))
        if os.path.exists(outp) or os.path.exists(outp + ".tmp"):
            refuse(f"output exists: {outp}")
        outputs[n] = outp
    pn = os.path.join(chain_dir, f"{a.root}.paramnames")
    if not os.path.exists(pn):
        refuse(f"missing {pn}")
    if os.path.exists(pn.replace(".paramnames", ".unblinded.paramnames")):
        refuse("unblinded paramnames output exists")
    # 8. health gate
    nuis, _present = nuisance_diagnostics(chain_dir, a.root)
    gate = classify_health(health, nuis)
    if gate["label"] == "RED":
        refuse(f"health gate RED: {gate['reasons']} (no unblind; report to the PI)")
    # 9. parse every chain in memory
    parsed = {}
    for n in chain_files:
        names, table = read_chain(os.path.join(chain_dir, n))
        if "ns" not in names or "Ap" not in names:
            refuse(f"{n} lacks the blinded columns")
        parsed[n] = (names, table)
    return dict(chain_dir=chain_dir, health=health, sums=sums, chain_files=chain_files, auth_sha=auth_sha, stamp=stamp,
                outputs=outputs, paramnames=pn, gate=gate, parsed=parsed, notes=notes, record=rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--root", default="real_eboss")
    ap.add_argument("--blind-lock", required=True)
    ap.add_argument("--analysis-lock", required=True)
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--authorization-sha256", required=True)
    ap.add_argument("--events-log", required=True)
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--operator", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--code-repo", default="/home/mfho/hcd_priya")
    ap.add_argument("--notes-repo", default="/home/mfho/hcd_priya_notes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    ck = run_checks(a)
    if a.dry_run:
        print(f"DRY-RUN OK {ck['gate']['label']}")
        return 0

    sys.path.insert(0, a.code_repo)
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator import blinding as BL
    lockrec = BL.read_blind_lock(a.blind_lock)
    offset = BL.blind_offset(lockrec["seed_str"])

    # unblind everything in memory first, then write atomically (tmp + rename), stamp LAST
    results = {}
    for n in ck["chain_files"]:
        names, table = ck["parsed"][n]
        unb = BL.unblind(table[:, 2:], offset, columns=names)
        results[n] = (names, np.column_stack([table[:, :2], unb]))
    written = []
    for n, (names, out) in results.items():
        outp = ck["outputs"][n]
        np.savetxt(outp + ".tmp", out, fmt=["%.8g"] * out.shape[1], header="weight  minusloglike  " + "  ".join(names))
        os.replace(outp + ".tmp", outp)
        written.append(outp)
    pn_dst = ck["paramnames"].replace(".paramnames", ".unblinded.paramnames")
    with open(ck["paramnames"]) as f, open(pn_dst + ".tmp", "w") as g:
        g.write(f.read())
    os.replace(pn_dst + ".tmp", pn_dst)
    written.append(pn_dst)

    utc = datetime.now(timezone.utc).isoformat()
    health = ck["health"]
    rec = dict(schema="UNBLINDED.stamp.v2", utc=utc, leg=health["leg"], survey=health.get("survey"), root=a.root, operator=a.operator,
               authorization=os.path.abspath(a.authorization), authorization_sha256=ck["auth_sha"],
               code_head=_git_head(a.code_repo), notes_head=_git_head(ck["notes"]),
               execution_record_code_head=ck["record"].get("code_head"), execution_record_wrapper_sha256=ck["record"].get("wrapper_sha256"),
               blind_lock=os.path.abspath(a.blind_lock), blind_lock_sha256=sha256_file(a.blind_lock), blind_lock_seed_str=lockrec["seed_str"],
               analysis_lock=os.path.abspath(a.analysis_lock), analysis_lock_sha256=sha256_file(a.analysis_lock),
               applied_offset={p: float(v) for p, v in offset.items()}, health_gate=ck["gate"],
               blinded_chain_sha256={n: ck["sums"][n] for n in ck["chain_files"]},
               unblinded_outputs={os.path.basename(p): sha256_file(p) for p in written},
               note="theta_inferred = theta_shown - offset on ns and Ap only; blinded files untouched; unblind-once. "
                    "The offset is shared by every leg under this blind.lock (salted by parameter only), so this unblind retires "
                    "the lock for any future DESI or KS blind fit (Reviewer B M4; preregistration v1.1 section 4).")
    with open(ck["stamp"] + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True); f.write("\n")
    os.replace(ck["stamp"] + ".tmp", ck["stamp"])
    os.makedirs(os.path.dirname(os.path.abspath(a.events_log)), exist_ok=True)
    with open(a.events_log, "a") as f:
        f.write(json.dumps(dict(utc=utc, leg=health["leg"], root=a.root, operator=a.operator, chain_dir=ck["chain_dir"], code_head=rec["code_head"],
                                notes_head=rec["notes_head"], blind_lock_sha256=rec["blind_lock_sha256"], seed_str=rec["blind_lock_seed_str"],
                                authorization_sha256=ck["auth_sha"], health_gate=ck["gate"]["label"], blinded_chain_sha256=rec["blinded_chain_sha256"]),
                           sort_keys=True) + "\n")
    print(f"UNBLINDED {health['leg']} ({a.root}) at {utc}; health gate {ck['gate']['label']}; stamp {ck['stamp']}")
    for p in ("ns", "Ap"):
        b = np.median(np.concatenate([ck["parsed"][n][1][:, 2 + ck["parsed"][n][0].index(p)] for n in ck["chain_files"]]))
        u = np.median(np.concatenate([results[n][1][:, 2 + results[n][0].index(p)] for n in ck["chain_files"]]))
        print(f"  {p}: blind median {b:.6g} -> unblind median {u:.6g}   (offset {offset[p]:+.6g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
