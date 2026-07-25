"""STAGE-V readout: sha-pinnable X-battery truth tables + pre-launch gate-power inputs.

Consumes the pooled npz written by scripts/build_xsel_truth_tables.py --pool and prints/writes:

  1. THE TRUTH TABLES (sha-pinnable json): per campaign, per KS z bin, the conditional/clean
     ratio on the KS leg k grid, pooled over the hires sims (mean + sim-to-sim sd):
       X1  = dilution-CORRECTED masked conditional DLA ratio (primary fork, PI annex OQ1)
       X1b = dilution-INCLUDED (deployed trough-fill bytes) ratio, readout OVERLAY (no fits)
       X1 wide = the 2x-wider-mask band ON THE CORRECTED FORK (round-2 revision 3)
       X2  = deployed-convention subDLA ratio; X3 = deployed-convention LLS ratio
  2. GATE-POWER INPUTS (round-2 MUST-FIX 4b, stated BEFORE launch): per arm, the truth-vs-K0
     data displacement chi = sqrt(delta^T C_ks^-1 delta) with delta = (R-1)*P_data on the KS
     leg; the expected sigma_pair PROXY; the paired SE at n; and P(Part-1 fail | zero bias)
     under the K0-pooled sigma_post convention (round-2 4a), by Monte Carlo. The proxy is
     EXPLICITLY heuristic and printed with its formula and with a full r = sigma_pair/
     sigma_post grid so pre-registration can override it:
       r_arm = clip(chi / sqrt(N_ks_rows), r_floor, sqrt(2))
     anchored at the coherent-pair limit (chi -> 0 => sigma_pair -> floor; R6 bit-identical
     precedent) and the independent-fit limit (per-mode displacement ~1 sigma_data =>
     pairing decoheres => sigma_pair -> sqrt(2) sigma_post).
  3. The subDLA MASKING-CONVENTION REVALIDATION summary (PI decision #7.4 + battery OQ9):
     per z, the fraction of subDLA rows altered by the trough-fill and the low-k power the
     fill removes (filled vs unfilled at the lowest KS k bin).
  4. The X1 estimator cross-check (round-2 cosmo MUST-FIX): transplanted-window division vs
     direct pixel-exclusion, max relative difference over the KS band per z.
  5. The at-least-one vs partition semantic difference (round-2 OQ10) on the KS band.

FAIL-LOUD: refuses missing/mismatched sidecar stamps, mixed conventions, a KS k grid that
does not match the deployed data_likelihood.load_ks_leg exactly, or missing suites.

ENV: emu-jax (imports the frozen data_likelihood READ-ONLY for the KS leg).

USAGE:
  python scripts/analyze_xsel_truth_tables.py --outdir <OUTDIR> \
      [--k0-shard-dir <dir with ks_selboost K0 pkls>] [--n-pairs 16] [--gate 0.30] \
      [--r-floor 0.05] [--json-out <path>]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_xsel_truth_tables import (  # noqa: E402  (pure helpers only)
    KS_Z_LO, KS_Z_HI, git_describe, ks_leg_kgrid, sha256_file, verify_sidecar)

GATE_DEFAULT = 0.30            # Part-1: |mean| + 2 SE < GATE * sigma_post (panel reform)
ARMS = ("X1_corr", "X1b_diluted", "X1_corr_wide", "X2_sub100", "X3_lls100")
ARM_NUM_KEY = {                # numerator curve key (KS-grid) per arm; denom = clean
    "X1_corr": "X1_corrected_ks",
    "X1b_diluted": "P_filt_dla_ks",
    "X1_corr_wide": "X1_corrected_wide_ks",
    "X2_sub100": "P_filt_sub_ks",
    "X3_lls100": "P_filt_lls_ks",
}


# ---------------------------------------------------------------------------
def load_pooled(outdir):
    """Verify + load the pooled npz and its suite stamps. Hard-fails on any drift."""
    pooled = Path(outdir) / "xsel_truth_pooled.npz"
    if not pooled.exists():
        raise FileNotFoundError(f"{pooled} missing -- run build_xsel_truth_tables --pool")
    stamp = verify_sidecar(pooled)
    if stamp.get("kind") != "xsel_truth_pooled":
        raise RuntimeError(f"{pooled}: wrong stamp kind {stamp.get('kind')!r}")
    for fid in ("lf", "hr"):
        if fid not in stamp.get("suites", {}):
            raise RuntimeError(f"{pooled}: suite {fid!r} missing from the pooled stamp")
        sp = Path(stamp["suite_files"][fid])
        st = verify_sidecar(sp)
        if st["npz_sha256"] != stamp["suites"][fid]["sha256"]:
            raise RuntimeError(f"suite {fid}: sha drift between suite npz and pooled stamp")
        conv = stamp["suites"][fid]["convention"]
        if conv.get("frame") != "deployed":
            raise RuntimeError(f"suite {fid}: convention frame {conv.get('frame')!r} "
                               "is not 'deployed' (round-2 MUST-FIX 1 violated)")
    d = np.load(pooled)
    return d, stamp


def ks_leg_or_die():
    """The deployed KS leg (frozen loader, READ-ONLY) + exact k-grid pin check."""
    from hcd_analysis.emulator import data_likelihood as DL
    leg = DL.load_ks_leg()
    ku_leg = np.unique(np.asarray(leg.k, float))
    ku_txt = ks_leg_kgrid()
    if ku_leg.shape != ku_txt.shape or not np.array_equal(ku_leg, ku_txt):
        raise RuntimeError("KS k-grid mismatch: deployed load_ks_leg vs the text-parsed "
                           f"grid used by the builder:\n{ku_leg}\nvs\n{ku_txt}")
    return leg, ku_leg


def _suite(d, fid, key):
    k = f"{fid}_{key}"
    if k not in d:
        raise KeyError(f"pooled npz missing {k}")
    return np.asarray(d[k])


def pooled_ratio(d, fid, num_key, z):
    """(mean, sim_sd, n_sims) of num/clean on the KS grid at one z over the suite sims."""
    zg = _suite(d, fid, "z_grid").astype(float)
    m = np.isclose(zg, z)
    if not m.any():
        return None
    num = _suite(d, fid, num_key)[m].astype(float)
    den = _suite(d, fid, "P_filt_clean_ks")[m].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den != 0, num / den, np.nan)
    return (np.nanmean(r, axis=0), np.nanstd(r, axis=0, ddof=1) if r.shape[0] > 1
            else np.zeros(r.shape[1]), int(r.shape[0]))


def truth_tables(d, k_ks, z_bins, fid="hr"):
    """The sha-pinnable per-campaign tables on the KS leg grid (hires suite)."""
    tables = {}
    for arm in ARMS:
        per_z = {}
        for z in z_bins:
            got = pooled_ratio(d, fid, ARM_NUM_KEY[arm], z)
            if got is None:
                raise RuntimeError(f"{arm}: no {fid} suite rows at z={z} -- incomplete "
                                   "suite; rerun the missing sims before pinning")
            mean, sd, n = got
            per_z[f"{z:.1f}"] = dict(ratio_mean=[round(float(x), 8) for x in mean],
                                     ratio_simsd=[round(float(x), 8) for x in sd],
                                     n_sims=n)
        tables[arm] = per_z
    tables["_k_grid"] = [float(x) for x in k_ks]
    tables["_denominator"] = "P_filt_clean (deployed global-mean-flux convention)"
    return tables


# ---------------------------------------------------------------------------
def k0_sigma_post(k0_dir, k0_arms=("K0_clean",)):
    """K0-pooled sigma_post per parameter (round-2 4a): RMS over mocks of the per-mock
    posterior sd, from the K0 (clean) shard pkls ONLY. Non-K0 arms in the same dir are
    skipped loudly; zero matching mocks is a hard failure."""
    paths = sorted(glob.glob(os.path.join(k0_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no pkls under {k0_dir}")
    per_param = {p: [] for p in ("ns", "Ap", "tau0_amp", "dtau0")}
    arms, skipped, n_mocks = set(), {}, 0
    for p in paths:
        with open(p, "rb") as f:
            dd = pickle.load(f)
        if dd.get("mode") == "smoke":
            continue
        if str(dd.get("arm")) not in k0_arms:
            skipped[str(dd.get("arm"))] = skipped.get(str(dd.get("arm")), 0) + 1
            continue
        arms.add(str(dd.get("arm")))
        for rec in dd["per_mock"]:
            n_mocks += 1
            names = list(rec["names"])
            for prm in per_param:
                if prm in names:
                    col = np.asarray(rec["draws"])[:, names.index(prm)]
                else:
                    col = np.asarray(rec["sites_extra"][prm]["draws"])
                per_param[prm].append(float(col.std()))
    if n_mocks == 0:
        raise RuntimeError(f"{k0_dir}: no non-smoke per-mock records for arms "
                           f"{k0_arms} (skipped: {skipped})")
    if skipped:
        print(f"[sigma_post] skipped non-K0 pkls in {k0_dir}: {skipped}")
    out = {prm: dict(pooled_rms=float(np.sqrt(np.mean(np.square(v)))),
                     mean=float(np.mean(v)), n=len(v))
           for prm, v in per_param.items() if v}
    out["_arms"] = sorted(arms)
    out["_n_mocks"] = n_mocks
    return out


def null_fail_prob(sigma_pair_over_post, n_pairs, gate, n_mc=200_000, seed=20260724):
    """P(Part-1 fail | zero bias) by MC: paired deltas iid N(0, sigma_pair^2), gate
    stat = |mean| + 2 * sd/sqrt(n) compared to gate * sigma_post. Exact under the
    null normal model (accounts for SE noise, unlike the Phi approximation)."""
    rng = np.random.default_rng(seed)
    r = float(sigma_pair_over_post)
    x = rng.normal(size=(n_mc, n_pairs)) * r
    mean = x.mean(axis=1)
    se = x.std(axis=1, ddof=1) / np.sqrt(n_pairs)
    return float(np.mean(np.abs(mean) + 2.0 * se > gate))


def gate_power(d, leg, k_ks, z_bins, sigma_post, *, n_pairs, gate, r_floor, n_mc):
    """Per-arm displacement chi over the KS covariance + the sigma_pair proxy and
    P(Part-1 fail | zero bias) per parameter. All formulas printed; proxy is heuristic."""
    zsel = np.isin(np.round(np.asarray(leg.z_row, float), 1),
                   np.round(np.asarray(z_bins, float), 1))
    if not zsel.any():
        raise RuntimeError(f"gate_power: no KS leg rows at z_bins={z_bins}")
    P_data = np.asarray(leg.P_data, float)[zsel]
    C = np.asarray(leg.C_data, float)[np.ix_(zsel, zsel)]
    z_row = np.asarray(leg.z_row, float)[zsel]
    k_row = np.asarray(leg.k, float)[zsel]
    L = np.linalg.cholesky(C)
    out = {}
    for arm in ARMS:
        R_by_z = {}
        for z in z_bins:
            got = pooled_ratio(d, "hr", ARM_NUM_KEY[arm], z)
            if got is None:
                raise RuntimeError(f"{arm}: missing hires z={z} for the gate-power input")
            R_by_z[round(z, 1)] = got[0]
        delta = np.empty_like(P_data)
        for i in range(P_data.size):
            R = R_by_z[round(float(z_row[i]), 1)]
            j = int(np.argmin(np.abs(k_ks - k_row[i])))
            if abs(k_ks[j] - k_row[i]) > 1e-9 * max(k_row[i], 1e-12):
                raise RuntimeError(f"KS row k={k_row[i]} not on the pinned 11-bin grid")
            if not np.isfinite(R[j]):
                raise RuntimeError(f"{arm}: non-finite truth ratio at z={z_row[i]} "
                                   f"k={k_row[i]} -- incomplete truth table")
            delta[i] = (R[j] - 1.0) * P_data[i]
        w = np.linalg.solve(L, delta)
        chi = float(np.sqrt(w @ w))
        r_arm = float(np.clip(chi / np.sqrt(P_data.size), r_floor, np.sqrt(2.0)))
        arm_out = dict(chi_over_ks_cov=round(chi, 3),
                       n_ks_rows=int(P_data.size),
                       r_proxy=round(r_arm, 4),
                       r_formula=f"clip(chi/sqrt(N), {r_floor}, sqrt(2)) [HEURISTIC "
                                 "proxy; override at pre-registration]")
        for prm in ("ns", "Ap"):
            sp = sigma_post[prm]["pooled_rms"]
            sig_pair = r_arm * sp
            arm_out[prm] = dict(
                sigma_post_pooled=round(sp, 5),
                sigma_pair_proxy=round(sig_pair, 5),
                SE_at_n=round(sig_pair / np.sqrt(n_pairs), 5),
                p_fail_null=round(null_fail_prob(r_arm, n_pairs, gate, n_mc), 5),
            )
        out[arm] = arm_out
    # model-independent r grid (the pre-registration decision table)
    grid = {}
    for r in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0, 1.414):
        grid[f"{r:.3f}"] = round(null_fail_prob(r, n_pairs, gate, n_mc), 5)
    out["_r_grid_p_fail_null"] = grid
    out["_gate"] = gate
    out["_n_pairs"] = n_pairs
    out["_note"] = ("sigma_pair proxy anchored at the coherent-pair limit (R6 "
                    "bit-identical precedent, chi->0) and the independent-fit limit "
                    "(sqrt(2) sigma_post); if p_fail_null is not small, revise n or "
                    "the gate form PRE-registration (round-2 MUST-FIX 4b)")
    return out


# ---------------------------------------------------------------------------
def oq9_summary(d, z_bins):
    """subDLA masking-convention revalidation (fill alters sub-DLA rows; OQ9 inputs)."""
    zg = _suite(d, "hr", "z_grid").astype(float)
    rows = {}
    for z in z_bins:
        m = np.isclose(zg, z)
        if not m.any():
            continue
        tf = float(np.nanmean(_suite(d, "hr", "trig_frac_sub")[m].astype(float)))
        fl = _suite(d, "hr", "P_filt_sub_ks")[m].astype(float)
        un = _suite(d, "hr", "P_unf_sub_ks")[m].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            loss0 = float(np.nanmean(1.0 - fl[:, 0] / un[:, 0]))
        rows[f"{z:.1f}"] = dict(subdla_rows_altered_frac=round(tf, 4),
                                lowk_power_lost_to_fill=round(loss0, 4))
    return rows


def estimator_crosscheck(d, z_bins):
    """X1 dilution: transplant-division vs pixel-exclusion, per z on the KS band."""
    zg = _suite(d, "hr", "z_grid").astype(float)
    out = {}
    for z in z_bins:
        m = np.isclose(zg, z)
        if not m.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.nanmean(_suite(d, "hr", "X1_corrected_ks")[m].astype(float)
                              / _suite(d, "hr", "P_filt_clean_ks")[m].astype(float), axis=0)
            seg = np.nanmean(_suite(d, "hr", "X1_seg_dla_ks")[m].astype(float)
                             / _suite(d, "hr", "X1_seg_ctrl_ks")[m].astype(float), axis=0)
            rel = np.abs(seg / corr - 1.0)
        out[f"{z:.1f}"] = dict(max_rel_diff_ks_band=(round(float(np.nanmax(rel)), 4)
                                                     if np.isfinite(rel).any() else None),
                               mean_rel_diff=(round(float(np.nanmean(rel)), 4)
                                              if np.isfinite(rel).any() else None))
    return out


def alo_summary(d, z_bins):
    """At-least-one vs partition (round-2 OQ10): KS-band-mean percent shift per z."""
    zg = _suite(d, "hr", "z_grid").astype(float)
    out = {}
    for z in z_bins:
        m = np.isclose(zg, z)
        if not m.any():
            continue
        row = {}
        for alo, part, lbl in (("P_alo_sub_filt_ks", "P_filt_sub_ks", "subDLA"),
                               ("P_alo_lls_filt_ks", "P_filt_lls_ks", "LLS")):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.nanmean(_suite(d, "hr", alo)[m].astype(float)
                               / _suite(d, "hr", part)[m].astype(float), axis=0) - 1.0
            row[lbl] = round(float(np.nanmean(r)) * 100.0, 3)
        out[f"{z:.1f}"] = row
    return out


# ---------------------------------------------------------------------------
#  Contract emission (scripts/xsel_truth_contract.md): the row-aligned truth-table npz
#  the X-battery consumes (ks_xsel_arms.load_truth_tables validates + sha-pins it).
# ---------------------------------------------------------------------------
def _rows_from_perz(R_by_z, leg_z, leg_k, k_ks, what):
    """Expand per-z 11-bin ratio curves onto the (N,) leg rows. Fail-loud on gaps."""
    rows = np.empty(leg_k.size)
    for i in range(leg_k.size):
        z = round(float(leg_z[i]), 1)
        if z not in R_by_z:
            raise RuntimeError(f"{what}: no truth curve at leg z={z}")
        j = int(np.argmin(np.abs(k_ks - leg_k[i])))
        if abs(k_ks[j] - leg_k[i]) > 1e-9 * leg_k[i]:
            raise RuntimeError(f"{what}: leg k={leg_k[i]} not on the 11-bin truth grid")
        rows[i] = R_by_z[z][j]
    if not (np.isfinite(rows).all() and (rows > 0).all()):
        raise RuntimeError(f"{what}: non-finite/non-positive contract rows -- incomplete "
                           "or broken truth table; refusing to emit")
    return rows


def emit_contract(d, leg, k_ks, z_bins, sigma_post, *, n_pairs, gate, r_floor, n_mc,
                  out_path=None):
    """Write the contract npz + run the CONSUMER'S OWN validator on it (fail-loud).
    Returns (path, sha256). D convention (documented in meta_json): for truth-curve arms
    D = sqrt(delta^T C_ks^-1 delta / N_rows), the RMS whitened per-row truth displacement
    from the stamped deployed-convention curve; for the in-manifold K8 arms D = 1.0 (one
    deployed dN/dX prior sigma, the displaced-arm S convention) and the coherent-pair
    floor r_floor is used for sigma_pair (in-manifold displacement, R6 precedent)."""
    import scripts.ks_xsel_arms as XA

    out_path = XA.XSEL_TRUTH_TABLE_PATH if out_path is None else str(out_path)
    leg_k = np.asarray(leg.k, float)
    leg_z = np.asarray(leg.z, float)[np.asarray(leg.z_idx, int)]
    C = np.asarray(leg.C_data, float)
    P_data = np.asarray(leg.P_data, float)
    L = np.linalg.cholesky(C)

    def perz(num_key):
        out = {}
        for z in z_bins:
            got = pooled_ratio(d, "hr", num_key, z)
            if got is None:
                raise RuntimeError(f"contract: missing hires suite z={z}")
            out[round(z, 1)] = got[0]
        return out

    rows = {}
    rows["ratio_rows_X1_dla100"] = _rows_from_perz(perz("X1_corrected_ks"), leg_z, leg_k,
                                                   k_ks, "X1_dla100")
    rows["ratio_rows_X1b_dla100_diluted"] = _rows_from_perz(perz("P_filt_dla_ks"), leg_z,
                                                            leg_k, k_ks, "X1b_diluted")
    rows["ratio_rows_X2_sub100"] = _rows_from_perz(perz("P_filt_sub_ks"), leg_z, leg_k,
                                                   k_ks, "X2_sub100")
    rows["ratio_rows_X3_lls100"] = _rows_from_perz(perz("P_filt_lls_ks"), leg_z, leg_k,
                                                   k_ks, "X3_lls100")
    wide = _rows_from_perz(perz("X1_corrected_wide_ks"), leg_z, leg_k, k_ks, "X1_wide")
    band_lo = np.minimum(rows["ratio_rows_X1_dla100"], wide)
    band_hi = np.maximum(rows["ratio_rows_X1_dla100"], wide)

    def chi_of(ratio_rows):
        delta = (np.asarray(ratio_rows) - 1.0) * P_data
        w = np.linalg.solve(L, delta)
        return float(np.sqrt(w @ w))

    def entry(ratio_rows=None, in_manifold=False):
        if in_manifold:
            D, r = 1.0, r_floor
        else:
            chi = chi_of(ratio_rows)
            D = chi / np.sqrt(P_data.size)
            r = float(np.clip(D, r_floor, np.sqrt(2.0)))
        return dict(D=float(D),
                    sigma_pair_expected=float(r),   # in sigma_post units (K0-pooled)
                    p_part1_fail_null=float(null_fail_prob(r, n_pairs, gate, n_mc)))

    gate_power = {
        "X1_dla100": entry(rows["ratio_rows_X1_dla100"]),
        "X1b_dla100_diluted": entry(rows["ratio_rows_X1b_dla100_diluted"]),
        "X2_sub100": entry(rows["ratio_rows_X2_sub100"]),
        "X3_lls100": entry(rows["ratio_rows_X3_lls100"]),
        "X4_prof": entry(XA.mixture_ratio(XA.f_sel(leg_z),
                                          rows["ratio_rows_X3_lls100"])),
        "K8a_eps_hi": entry(in_manifold=True),
        "K8b_eps_lo": entry(in_manifold=True),
        "K8c_kap_hi": entry(in_manifold=True),
    }
    missing = set(XA.GATE_POWER_ARMS) - set(gate_power)
    if missing:
        raise RuntimeError(f"contract gate_power missing arms {sorted(missing)}")

    convention = dict(XA.REQUIRED_CONVENTION)
    convention.update(dict(
        emitted_by="scripts/analyze_xsel_truth_tables.py (stage V)",
        sigma_post_ns=float(sigma_post["ns"]["pooled_rms"]),
        sigma_post_Ap=float(sigma_post["Ap"]["pooled_rms"]),
        d_convention="RMS whitened per-row truth displacement sqrt(chi^2/N) over the "
                     "deployed KS C_data; K8 arms D = 1.0 deployed dN/dX prior sigma",
        sigma_pair_convention=f"sigma_post units; proxy clip(D, {r_floor}, sqrt(2)); "
                              "K8 = coherent floor (in-manifold)",
        n_pairs=int(n_pairs), gate=float(gate),
    ))
    meta = dict(git_describe=git_describe(), n_rows=int(leg_k.size),
                z_bins=[float(z) for z in z_bins])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             leg_k=leg_k, leg_z=leg_z,
             band_lo_rows_X1_dla100=band_lo, band_hi_rows_X1_dla100=band_hi,
             convention_json=json.dumps(convention, sort_keys=True),
             gate_power_json=json.dumps(gate_power, sort_keys=True),
             meta_json=json.dumps(meta, sort_keys=True),
             **rows)
    sha = sha256_file(out_path)
    # the strongest ingest check available: the CONSUMER'S validator, at the fresh sha
    tt = XA.load_truth_tables(path=out_path, expect_sha=sha)
    assert tt["sha256"] == sha
    return out_path, sha


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", required=True,
                    help="dir containing xsel_truth_pooled.npz (+ suite npzs)")
    ap.add_argument("--k0-shard-dir", default=None,
                    help="dir of K0 (clean) shard pkls for the pooled sigma_post")
    ap.add_argument("--k0-arms", default="K0_clean",
                    help="comma-separated arm ids accepted as K0 (default K0_clean)")
    ap.add_argument("--sigma-post-ns", type=float, default=None,
                    help="override sigma_post(ns) if no K0 dir is given")
    ap.add_argument("--sigma-post-ap", type=float, default=None)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--gate", type=float, default=GATE_DEFAULT)
    ap.add_argument("--r-floor", type=float, default=0.05)
    ap.add_argument("--mc", type=int, default=200_000)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--contract-out", default=None,
                    help="override the contract npz path (default: "
                         "ks_xsel_arms.XSEL_TRUTH_TABLE_PATH)")
    ap.add_argument("--no-contract", action="store_true",
                    help="skip emitting the X-battery contract npz")
    ap.add_argument("--z-bins", default=None,
                    help="comma-separated z subset (default: full KS range 2.4-4.6; "
                         "a partial readout is stamped PARTIAL and must not be pinned)")
    args = ap.parse_args()

    d, stamp = load_pooled(args.outdir)
    leg, k_ks = ks_leg_or_die()
    if args.z_bins:
        z_bins = [round(float(z), 1) for z in args.z_bins.split(",")]
        print(f"[PARTIAL] z-bins restricted to {z_bins}: this readout is NOT pinnable")
    else:
        z_bins = [round(z, 1) for z in np.arange(KS_Z_LO, KS_Z_HI + 1e-9, 0.2)]
    print(f"[stamps] pooled OK: lf {stamp['suites']['lf']['n_members']} members, "
          f"hr {stamp['suites']['hr']['n_members']} members, "
          f"git {stamp['git_describe']}")
    print(f"[ks] deployed leg: {leg.k.size} rows, k grid pin EXACT ({k_ks.size} bins)")

    tables = truth_tables(d, k_ks, z_bins)
    print("\n=== TRUTH TABLES (hires suite, KS leg grid, deployed convention) ===")
    for arm in ARMS:
        print(f"\n-- {arm} (ratio to clean; mean over sims [sim sd] at "
              "k = 0.0055 / 0.0198 / 0.0627) --")
        for z in z_bins:
            e = tables[arm][f"{z:.1f}"]
            r, s = e["ratio_mean"], e["ratio_simsd"]
            print(f"  z={z:.1f}: {r[0]:+.4f} [{s[0]:.4f}]  {r[5]:+.4f} [{s[5]:.4f}]  "
                  f"{r[-1]:+.4f} [{s[-1]:.4f}]  (n={e['n_sims']})")

    if args.k0_shard_dir:
        sigma_post = k0_sigma_post(args.k0_shard_dir,
                                   k0_arms=tuple(args.k0_arms.split(",")))
        print(f"\n[sigma_post] K0-pooled from {args.k0_shard_dir}: "
              f"arms {sigma_post['_arms']}, {sigma_post['_n_mocks']} mocks")
    elif args.sigma_post_ns and args.sigma_post_ap:
        sigma_post = {"ns": dict(pooled_rms=args.sigma_post_ns, mean=args.sigma_post_ns,
                                 n=0),
                      "Ap": dict(pooled_rms=args.sigma_post_ap, mean=args.sigma_post_ap,
                                 n=0)}
        print("\n[sigma_post] MANUAL override values (no K0 pkls read)")
    else:
        raise SystemExit("need --k0-shard-dir OR both --sigma-post-ns/--sigma-post-ap "
                         "(the K0-pooled sigma_post convention is round-2 MUST-FIX 4a)")
    for prm in ("ns", "Ap", "tau0_amp", "dtau0"):
        if prm in sigma_post:
            e = sigma_post[prm]
            print(f"  sigma_post({prm}): pooled RMS {e['pooled_rms']:.5f} "
                  f"(mean {e['mean']:.5f}, n={e['n']})")

    gp = gate_power(d, leg, k_ks, z_bins, sigma_post, n_pairs=args.n_pairs,
                    gate=args.gate, r_floor=args.r_floor, n_mc=args.mc)
    print(f"\n=== GATE-POWER INPUTS (Part-1: |mean|+2SE < {args.gate} sigma_post, "
          f"n={args.n_pairs}) ===")
    for arm in ARMS:
        e = gp[arm]
        print(f"  {arm:14s} chi={e['chi_over_ks_cov']:9.2f}  r_proxy={e['r_proxy']:.3f}  "
              f"P(fail|0) ns={e['ns']['p_fail_null']:.4f} Ap={e['Ap']['p_fail_null']:.4f}"
              f"  (sigma_pair proxy ns={e['ns']['sigma_pair_proxy']:.4f}, "
              f"SE={e['ns']['SE_at_n']:.4f})")
    print("  r-grid P(fail|0):",
          "  ".join(f"r={k}:{v:.4f}" for k, v in gp["_r_grid_p_fail_null"].items()))
    print("  NOTE:", gp["_note"])

    oq9 = oq9_summary(d, z_bins)
    print("\n=== subDLA MASKING-CONVENTION REVALIDATION (OQ9; fill is NOT DLA-only) ===")
    for z, e in oq9.items():
        print(f"  z={z}: rows altered {e['subdla_rows_altered_frac']:.3f}, "
              f"low-k power lost to fill {e['lowk_power_lost_to_fill']:.3f}")

    xc = estimator_crosscheck(d, z_bins)
    print("\n=== X1 DILUTION ESTIMATOR CROSS-CHECK (division vs pixel-exclusion) ===")
    for z, e in xc.items():
        print(f"  z={z}: max rel diff {e['max_rel_diff_ks_band']}, "
              f"mean {e['mean_rel_diff']}")

    alo = alo_summary(d, z_bins)
    print("\n=== AT-LEAST-ONE vs PARTITION (OQ10; KS-band mean shift, %) ===")
    for z, e in alo.items():
        print(f"  z={z}: subDLA {e['subDLA']:+.2f}%  LLS {e['LLS']:+.2f}%")

    readout = dict(partial_z_bins=(z_bins if args.z_bins else None),
                   truth_tables=tables, gate_power=gp, sigma_post=sigma_post,
                   oq9_subdla_fill=oq9, x1_estimator_crosscheck=xc,
                   alo_vs_partition_pct=alo,
                   stamps=dict(pooled_sha256=stamp["npz_sha256"],
                               git_describe=stamp["git_describe"],
                               suites={f: stamp["suites"][f]["sha256"]
                                       for f in ("lf", "hr")}))
    json_out = args.json_out or str(Path(args.outdir) / "xsel_truth_readout.json")
    with open(json_out, "w") as f:
        json.dump(readout, f, indent=1, sort_keys=True)
    print(f"\n[readout] wrote {json_out}")
    print(f"[readout] sha256: {sha256_file(json_out)}")

    # X-battery contract emission (scripts/xsel_truth_contract.md): FULL z range only.
    if args.no_contract:
        print("[contract] SKIPPED (--no-contract)")
    elif args.z_bins:
        print("[contract] SKIPPED: partial --z-bins readout is never contract-emitted")
    else:
        cpath, csha = emit_contract(d, leg, k_ks, z_bins, sigma_post,
                                    n_pairs=args.n_pairs, gate=args.gate,
                                    r_floor=args.r_floor, n_mc=args.mc,
                                    out_path=args.contract_out)
        print(f"[contract] wrote {cpath} (consumer validator PASSED)")
        print(f"[contract] sha256 (PIN in ks_xsel_arms.XSEL_TRUTH_TABLE_SHA256 after the "
              f"delegated figure review): {csha}")


if __name__ == "__main__":
    main()
