"""X-battery-2 readout (PROPOSAL-extreme-battery-v2 as adopted by PI record #7 + execution
annex, 2026-07-24). Paired readout of the X/K8 arms against the REUSED R4/R5 corrected-
geometry K0 shards (annex OQ5).

GATES (REVISED pre-registration 2026-07-26, the one-shot gate-form revision under the
stage-V fallback; memo `2026-07-26-xsel-pilot-sigmapair-and-gate-revision`, 4-lens
panel ADOPTED -- the 4-pair pilot measured sigma_pair/sigma_post = 0.21-2.03 against
the r < 0.25 calibration requirement, so the original binary equivalence gate was
uncalibratable and was replaced BEFORE the full-battery launch):
  PART 1 (X1/X2/X3; ns AND Ap; n = 16 pairs): THREE-OUTCOME interval verdict per
    arm x channel with t* = t(0.975, n-1):
      PROTECTED       |mean| + t* SE < 0.30 sigma_post  (budget met, a fortiori)
      UNPROTECTED     |mean| - t* SE > 0.30 sigma_post  (bias detected above budget
                      -> corner-failure protocol, PI decision 8)
      UNDER-RESOLVED  otherwise (two-sided bound quoted; threshold study carries it;
                      PROTECTED is UNREACHABLE for X1/X2 at the measured sigma_pair,
                      stated pre-launch)
    Deltas are RAW-units paired biases (bias_X(m) - bias_K0(m)) normalized by
    sigma_post = the K0-POOLED posterior sd per parameter (round-2 revision 4a,
    unchanged). Paired t + Wilcoxon signed-rank and the measured sigma_pair are
    reported per channel; the legacy ub2 key is kept for continuity.
  PART 2 (DISCLOSURE-ONLY, never binding): S = (|mean| + 2 SE)/D per arm/param, D from
    the sha-pinned stage-V truth table's gate-power block, printed WITH the stage-V
    proxy P(binary-fail | zero bias) and the noise floor 2SE/D; proxy exceedances are
    printed as a 4b DISCLOSURE (resolved by the pre-launch revision), with
    expected-vs-measured sigma_pair.
  X4 and K8a/b/c: same lines, reported NON-BINDING (annex OQ4; K6-redesign Sec 7).
  tau0_amp + dtau0 ALWAYS reported alongside (standing convention). UNSIGNED n_s
  expectation notes printed per arm (round-2 revision 8). Rank lines are non-binding with
  pre-registered non-uniformity on every displaced-truth arm.

FAIL-LOUD INGEST (refusals, not soft skips):
  - own glob ks_xsel_*_shard_*.pkl (never sees legacy/K/R6 pkls); smoke filtered;
  - registry-signature homogeneity AND equality with the LIVE registry + PINNED truth table
    (the signature covers the table sha, so a swapped table refuses here);
  - per-pkl stamped truth-table sha == the loaded table's sha;
  - forward/prior signatures, prior_constants and core run_kw (seed INCLUDED) homogeneous;
  - K0 CROSS-SIGNATURE PAIRING ALLOWANCE (pre-registered, annex OQ5): K0 pkls may carry the
    frozen LEGACY registry signature (and ONLY that one, recomputed live from
    scripts/ks_selboost_arms.py); every other stamp (forward sig, prior sig, prior
    constants, run_kw core incl. the seed) must EQUAL the battery's; r6_override refused;
  - PAIR-IDENTITY asserts per (arm, mock): the arm's underlying clean draw == K0's truth
    (theta/tau0/site truths bit-identical; arm-specific alpha contract: corner values,
    f_sel rows, dN/dX-displaced rows recomputed through the frozen map, X1 full identity +
    swap-identity P_swap == P_clean x pinned ratio);
  - per-mock truth-admissibility tripwire (all-pass required).

CORNER-FAILURE PROTOCOL (PI decision 8): printed with the verdicts; a pure-corner FAIL
limits arbitrary-composition claims and triggers the selection-fraction/redshift-profile
threshold study; it is NOT a hard unblinding block.

Pure numpy at import (login-node safe); the K8 row recompute lazily imports jax + the frozen
dndx_wc map (selboost-analyzer precedent).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/analyze_xsel.py --shard-dir <xsel dir> [--k0-dir <ks_rerun dir>] \
     [--out-dir <readout dir>]
"""
import argparse
import functools
import glob
import os
import pickle

import numpy as np

from scripts.analyze_dla_selfdraw import _col_draws_truth
from scripts.analyze_ks_selboost import _hashable, stamp_era
import scripts.ks_xsel_arms as XA

print = functools.partial(print, flush=True)

GATE_PARAMS = ("ns", "Ap")
PAIRED_PARAMS = ("ns", "Ap", "tau0_amp", "dtau0")
_RUN_KW_CORE = ("n_warmup", "n_samples", "seed", "max_tree_depth", "dense_mass")
_K8_SITES = ("eps_lls", "kappa_lls", "m_sub", "t_sub", "dla_raw", "t_dla")
_P_FAIL_NULL_FLAG = 0.05


# ---------------------------------------------------------------- small stats (testable)

def bias_raw(rec, name):
    d, t = _col_draws_truth(rec, name)
    return float(d.mean()) - t


def sigma_post_pooled(k0_recs, name):
    """PRE-REGISTERED sigma_post: sqrt(mean over K0 mocks of the posterior variance)."""
    v = [float(np.asarray(_col_draws_truth(k0_recs[m], name)[0]).var(ddof=1))
         for m in sorted(k0_recs)]
    assert v and all(x > 0 for x in v), f"degenerate K0 posterior for {name}"
    return float(np.sqrt(np.mean(v)))


def gate_stats(deltas, gate=XA.GATE):
    """Part-1 arithmetic on paired deltas (already in sigma_post units).

    REVISED PRE-REGISTRATION (2026-07-26 memo `2026-07-26-xsel-pilot-sigmapair-and-
    gate-revision`, 4-lens panel ADOPTED; the one-shot gate-form revision under the
    stage-V fallback -- the pilot measured sigma_pair r = 0.21-2.03 vs the r < 0.25
    calibration requirement, so the binary equivalence verdict is uncalibratable):
    THREE-OUTCOME interval verdict with t(0.975, n-1) quantiles (sigma_pair is
    estimated from the same deltas):
      PROTECTED       |mean| + t* SE < gate   (original 0.30 budget met, a fortiori)
      UNPROTECTED     |mean| - t* SE > gate   (bias detected above budget)
      UNDER-RESOLVED  otherwise               (two-sided bound quoted; PI decision-8)
    The legacy ub2 = |mean| + 2 SE key is kept for continuity/npz compatibility.
    Also reports the measured sigma_pair, the paired t and Wilcoxon signed-rank
    statistics (robustness companion)."""
    d = np.asarray([x for x in deltas if x is not None], float)
    n = d.size
    assert n >= 2, f"gate_stats needs >= 2 paired deltas, got {n}"
    from scipy.stats import t as _t, wilcoxon as _wilcoxon
    sigma_pair = float(d.std(ddof=1))
    se = float(sigma_pair / np.sqrt(n))
    tq = float(_t.ppf(0.975, n - 1))
    mean = float(d.mean())
    ub_t = abs(mean) + tq * se
    lb_t = abs(mean) - tq * se
    if ub_t < gate:
        verdict = "PROTECTED"
    elif lb_t > gate:
        verdict = "UNPROTECTED"
    else:
        verdict = "UNDER-RESOLVED"
    t_stat = mean / se if se > 0 else float("inf")
    p_t = float(2.0 * _t.sf(abs(t_stat), n - 1)) if np.isfinite(t_stat) else 0.0
    try:
        w_p = float(_wilcoxon(d, zero_method="wilcox", mode="approx").pvalue)
    except ValueError:
        w_p = float("nan")
    return dict(n=n, mean=mean, se=se, median=float(np.median(d)),
                sigma_pair=sigma_pair,
                ub2=abs(mean) + 2.0 * se, ub_t=ub_t, lb_t=lb_t, t975=tq,
                t_stat=float(t_stat), p_t=p_t, p_wilcoxon=w_p,
                verdict=verdict)


def part2_disclosure(gs, gp):
    """Part-2 S = (|mean| + 2SE)/D with the stage-V gate-power inputs. DISCLOSURE ONLY."""
    D = float(gp["D"])
    return dict(D=D, S=gs["ub2"] / D, noise_floor_2se_over_D=2.0 * gs["se"] / D,
                p_part1_fail_null=float(gp["p_part1_fail_null"]),
                sigma_pair_expected=float(gp["sigma_pair_expected"]))


def rank_u(rec, name):
    d, t = _col_draws_truth(rec, name)
    return float((np.asarray(d) < t).mean())


# ---------------------------------------------------------------- pair-identity checks

def _theta_tau_block(names):
    """Indices of the theta9 + tau0 ladder block (everything before the alpha pivots)."""
    j_alpha = names.index("alpha_lls")
    return list(range(j_alpha)), [names.index(n) for n in
                                  ("alpha_lls", "alpha_subdla", "alpha_dla")]


def _sites_truths(rec):
    return {nm: float(rec["sites_extra"][nm]["truth"]) for nm in rec.get("sites_extra", {})}


def _softplus(x):
    return np.logaddexp(0.0, np.asarray(x, float))


def k8_rows_from_sites(sites, ref_z, xbar_z, zg):
    """Analyzer-side mirror of the frozen mapped forward for the K8 row contract: dN/dX from
    the (displaced) site truths through the frozen w_c_corrected (lazy read-only import) +
    the numpy tie-norm. MU0 is the fixed inference constant log(expm1(1))."""
    from hcd_analysis.emulator.dndx_wc import w_c_corrected   # lazy: jax import
    zg = np.asarray(zg, float)
    ratio = (1.0 + zg) / (1.0 + XA.Z_PIVOT)
    mu0 = float(np.log(np.expm1(1.0)))                        # == inference.KS_DNDX_DLA_RAW_MU0
    dla_amp = float(_softplus(sites["dla_raw"]) / _softplus(mu0))
    fac = np.stack([np.exp(sites["eps_lls"]) * ratio ** sites["kappa_lls"],
                    np.exp(sites["m_sub"]) * ratio ** sites["t_sub"],
                    dla_amp * ratio ** sites["t_dla"]], axis=-1)
    w = np.asarray(w_c_corrected(np.asarray(ref_z, float) * fac,
                                 np.asarray(xbar_z, float), zg), float)[..., 1:]
    s = w.sum(axis=-1, keepdims=True)
    return w / np.where(s > 1.0, s, 1.0)


def assert_pair_identity(arm_id, m, rec_x, rec_k0, meta_x, tt):
    """The per-pair identity contract (module docstring). AssertionError on any violation."""
    e = XA.ARMS[arm_id]
    names = list(rec_k0["names"])
    assert names == list(rec_x["names"]), f"{arm_id} mock {m}: packed names mismatch"
    tx = np.asarray(rec_x["truth_vec"], float)
    tk = np.asarray(rec_k0["truth_vec"], float)
    blk, alpha_idx = _theta_tau_block(names)
    assert np.array_equal(tx[blk], tk[blk]), \
        f"{arm_id} mock {m}: theta/tau0 truth differs from the K0 partner (pairing broken)"
    rows_x = np.asarray(rec_x["truth_alpha_hcd_z"], float)
    rows_k = np.asarray(rec_k0["truth_alpha_hcd_z"], float)
    zg = np.asarray(meta_x["arm_stamp"]["z_global"], float)
    assert rows_x.shape == rows_k.shape == (zg.size, 3), \
        f"{arm_id} mock {m}: truth_alpha_hcd_z shape {rows_x.shape}"
    st_x, st_k = _sites_truths(rec_x), _sites_truths(rec_k0)
    assert set(st_x) == set(st_k), f"{arm_id} mock {m}: sites_extra key sets differ"

    if e["kind"] == "data_swap":
        assert np.array_equal(tx, tk), \
            f"{arm_id} mock {m}: X1 truth_vec != K0 (must be the identical clean draw)"
        assert np.array_equal(rows_x, rows_k), f"{arm_id} mock {m}: X1 truth rows != K0"
        for nm in st_x:
            assert st_x[nm] == st_k[nm] or (np.isnan(st_x[nm]) and np.isnan(st_k[nm])), \
                f"{arm_id} mock {m}: X1 site truth {nm} differs from K0"
        sw = rec_x.get("xsel_swap")
        assert sw is not None and sw.get("fork") == "dilution_corrected", \
            f"{arm_id} mock {m}: missing/wrong xsel_swap stamp (fork must be the corrected one)"
        ratio = tt["ratios"][e["table_key"]]
        assert np.allclose(np.asarray(sw["P_swap_truth"], float),
                           np.asarray(sw["P_clean_truth"], float) * ratio,
                           rtol=1e-12, atol=0), \
            f"{arm_id} mock {m}: swap identity violated (P_swap != P_clean x pinned ratio)"
    else:
        for nm in st_x:
            if e["kind"] == "dndx_displaced" and nm == e["site"]:
                continue
            assert st_x[nm] == st_k[nm] or (np.isnan(st_x[nm]) and np.isnan(st_k[nm])), \
                f"{arm_id} mock {m}: site truth {nm} differs from K0 (clean draw not shared)"
        if e["kind"] == "mixture_corner":
            want_piv = np.zeros(3)
            want_piv[e["cls"]] = 1.0
            want_rows = XA.corner_alpha_rows(zg.size, e["cls"])
        elif e["kind"] == "mixture_profile":
            f_st = meta_x["arm_stamp"]["f_sel_z_global"]
            assert f_st is not None and np.allclose(np.asarray(f_st, float),
                                                    XA.f_sel(zg), rtol=1e-12, atol=0), \
                f"{arm_id}: stamped f_sel_z_global does not match the registry profile"
            want_piv = np.zeros(3)
            want_piv[e["cls"]] = float(XA.f_sel(float(XA.Z_PIVOT)))
            want_rows = XA.profile_alpha_rows(zg, e["cls"])
        else:                                        # dndx_displaced (K8)
            site, ns_ = e["site"], float(e["n_sigma"])
            pc = meta_x["prior_constants"]
            sig = float(pc["ks_dndx_sigma_eps"] if site == "eps_lls"
                        else pc["ks_dndx_sigma_kappa"])
            want_disp = st_k[site] + ns_ * sig
            assert abs(st_x[site] - want_disp) < 1e-12, (
                f"{arm_id} mock {m}: displaced {site} truth {st_x[site]} != "
                f"K0 {st_k[site]} + {ns_} x {sig}")
            sites = {nm: st_x[nm] for nm in _K8_SITES}
            want_rows = k8_rows_from_sites(sites, pc["ks_dndx_ref_z"], pc["ks_xbar_z"], zg)
            assert np.allclose(rows_x, want_rows, rtol=0, atol=1e-9), \
                f"{arm_id} mock {m}: K8 rows != frozen-map image of the displaced sites"
            want_piv = None                          # pivot: consistency check below
        if want_piv is not None:
            assert np.allclose(tx[alpha_idx], want_piv, rtol=1e-12, atol=1e-15), \
                f"{arm_id} mock {m}: alpha pivot truth {tx[alpha_idx]} != contract {want_piv}"
            assert np.allclose(rows_x, want_rows, rtol=1e-12, atol=1e-15), \
                f"{arm_id} mock {m}: truth rows violate the arm contract"
    assert XA.truth_admissible(rows_x, strict_interior=(e["kind"] == "dndx_displaced")), \
        f"{arm_id} mock {m}: truth rows fail the admissibility tripwire"


# ---------------------------------------------------------------- ingest

def _stamp_tuple(meta):
    fwd = meta["forward"]
    return (stamp_era(meta["prior_constants"]),
            fwd.get("forward_signature"), fwd.get("hcd_prior_signature"),
            tuple(sorted((k, _hashable(v)) for k, v in meta["prior_constants"].items())),
            tuple((k, meta["run_kw"].get(k)) for k in _RUN_KW_CORE))


def load_battery(shard_dir, k0_dir, table_path=None, expect_sha=None):
    """(k0 {mock: rec}, arms {arm: {mock: rec}}, meta {arm: meta}, k0_meta, tt, reg_sig).
    Every refusal in the module docstring fires here."""
    tt = XA.load_truth_tables(table_path, expect_sha)
    reg_sig = XA.registry_signature(table_path, expect_sha)

    paths = sorted(glob.glob(os.path.join(shard_dir, "ks_xsel_*_shard_*.pkl")))
    paths = [p for p in paths if ".smoke" not in os.path.basename(p)]
    assert paths, f"no ks_xsel shard pkls under {shard_dir}"
    arms, meta_by_arm, ref = {}, {}, None
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        assert d.get("survey") == "ks", f"{p}: survey {d.get('survey')!r} != 'ks'"
        arm = d["arm"]
        assert arm in XA.ARMS, f"{p}: unknown arm {arm!r} for this registry"
        meta = d["meta"]
        assert not (meta.get("prior_constants") or {}).get("r6_override"), \
            f"{p}: r6_override pkl in an X-battery dir (R6 pkls are not battery members)"
        st = _stamp_tuple(meta)
        if ref is None:
            ref = st
        assert st[0] == "mapped", f"{p}: era {st[0]} != mapped (X-battery is mapped-era only)"
        for i, what in ((1, "forward_signature"), (2, "hcd_prior_signature"),
                        (3, "prior_constants"), (4, "core run_kw")):
            assert st[i] == ref[i], f"{p}: {what} mismatch (mixed campaign must not pool)"
        a_st = meta["arm_stamp"]
        assert a_st["registry_signature"] == reg_sig, (
            f"{p}: stamped registry_signature != the LIVE scripts/ks_xsel_arms.py registry + "
            f"pinned truth table (drifted registry or swapped table; reconcile deliberately)")
        assert a_st["truth_table"]["sha256"] == tt["sha256"], \
            f"{p}: stamped truth-table sha != the loaded pinned table (mixed-table campaign)"
        assert a_st["arm_id"] == arm and a_st["campaign"] == XA.CAMPAIGN, f"{p}: arm stamp broken"
        tgt = arms.setdefault(arm, {})
        assert len(d["per_mock"]) == len(d["idxs"]), f"{p}: per_mock/idxs length mismatch"
        for m, rec in zip(d["idxs"], d["per_mock"]):
            assert int(m) not in tgt, f"{p}: duplicate mock idx {m} for arm {arm}"
            tgt[int(m)] = rec
        meta_by_arm[arm] = meta
    for arm, meta in meta_by_arm.items():
        want = set(range(int(XA.ARMS[arm]["n_mocks"])))
        assert int(meta["n_mocks"]) == len(want), \
            f"{arm}: stamped n_mocks {meta['n_mocks']} != registry {len(want)}"
        got = set(arms[arm])
        assert got == want, (f"campaign incomplete for {arm}: missing mocks "
                             f"{sorted(want - got)} (got {len(got)}/{len(want)})")

    # K0 (reused, cross-signature allowance)
    k0_paths = sorted(glob.glob(os.path.join(k0_dir, "ks_selboost_clean_shard_*.pkl")))
    k0_paths = [p for p in k0_paths if ".smoke" not in os.path.basename(p)]
    assert k0_paths, f"no reused K0 shards (ks_selboost_clean_shard_*.pkl) under {k0_dir}"
    import scripts.ks_selboost_arms as AR_LEGACY
    legacy_sig = AR_LEGACY.registry_signature()
    k0, k0_meta = {}, None
    for p in k0_paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        assert d.get("survey") == "ks" and d["arm"] == XA.K0_ARM_ID, \
            f"{p}: not a KS K0_clean shard"
        meta = d["meta"]
        assert not (meta.get("prior_constants") or {}).get("r6_override"), \
            f"{p}: r6_override K0 pkl refused (R6 arms are not the campaign baseline)"
        st = _stamp_tuple(meta)
        assert st[0] == "mapped", f"{p}: K0 era {st[0]} != mapped"
        # THE pre-registered cross-signature allowance (annex OQ5): the K0 stamp may differ
        # from the X registry ONLY by being exactly the frozen legacy registry signature.
        assert meta["arm_stamp"]["registry_signature"] == legacy_sig, (
            f"{p}: K0 registry signature is neither the X-battery's nor the frozen legacy "
            f"one; the pre-registered reuse allowance covers ONLY the R4/R5 "
            f"corrected-geometry K0 shards")
        for i, what in ((1, "forward_signature"), (2, "hcd_prior_signature"),
                        (3, "prior_constants"), (4, "core run_kw (seed included)")):
            assert st[i] == ref[i], (
                f"{p}: K0 {what} != the battery's (the cross-signature allowance requires "
                f"every non-registry stamp to be EQUAL; pairing would be broken)")
        for m, rec in zip(d["idxs"], d["per_mock"]):
            assert int(m) not in k0, f"{p}: duplicate K0 mock {m}"
            k0[int(m)] = rec
        k0_meta = meta
    assert set(k0) == set(range(XA.K0_N_MOCKS)), \
        f"K0 baseline incomplete: got {sorted(k0)} want 0..{XA.K0_N_MOCKS - 1}"

    # pair identity + admissibility, every (arm, mock)
    for arm, recs in arms.items():
        for m, rec in recs.items():
            assert m in k0, f"{arm} mock {m}: no K0 partner"
            assert_pair_identity(arm, m, rec, k0[m], meta_by_arm[arm], tt)
    return k0, arms, meta_by_arm, k0_meta, tt, reg_sig


# ---------------------------------------------------------------- summary + report

def summarize(shard_dir, k0_dir, table_path=None, expect_sha=None):
    k0, arms, meta, k0_meta, tt, reg_sig = load_battery(shard_dir, k0_dir, table_path,
                                                        expect_sha)
    sigma_post = {p: sigma_post_pooled(k0, p) for p in PAIRED_PARAMS}
    deltas, stats, part2 = {}, {}, {}
    for a, recs in arms.items():
        deltas[a] = {}
        stats[a] = {}
        part2[a] = {}
        for p in PAIRED_PARAMS:
            dl = {m: (bias_raw(recs[m], p) - bias_raw(k0[m], p)) / sigma_post[p]
                  for m in sorted(recs)}
            deltas[a][p] = dl
            gs = gate_stats(list(dl.values()))
            stats[a][p] = gs
            part2[a][p] = part2_disclosure(gs, tt["gate_power"][a])
    binding = {a: {p: stats[a][p] for p in GATE_PARAMS}
               for a in arms if XA.ARMS[a]["part1"]}
    # revised pre-registration (2026-07-26 memo): UNPROTECTED is the detected-above-
    # budget outcome that feeds the PI decision-8 protocol; UNDER-RESOLVED is quoted
    # as a bound. The legacy any_binding_fail name is kept for npz/test continuity.
    any_binding_fail = any(binding[a][p]["verdict"] == "UNPROTECTED"
                           for a in binding for p in GATE_PARAMS)
    any_under_resolved = any(binding[a][p]["verdict"] == "UNDER-RESOLVED"
                             for a in binding for p in GATE_PARAMS)
    ranks = {a: {p: [rank_u(arms[a][m], p) for m in sorted(arms[a])] for p in GATE_PARAMS}
             for a in arms}
    ranks["K0_clean"] = {p: [rank_u(k0[m], p) for m in sorted(k0)] for p in GATE_PARAMS}
    return dict(k0=k0, arms=arms, meta=meta, k0_meta=k0_meta, tt=tt, reg_sig=reg_sig,
                sigma_post=sigma_post, deltas=deltas, stats=stats, part2=part2,
                binding=binding, any_binding_fail=any_binding_fail,
                any_under_resolved=any_under_resolved, ranks=ranks)


def _write_arm_outputs(res, a, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    e = XA.ARMS[a]
    tt = res["tt"]
    mocks = sorted(res["arms"][a])
    npz = dict(arm=np.asarray(a), mocks=np.asarray(mocks),
               registry_signature=np.asarray(res["reg_sig"]),
               truth_table_sha256=np.asarray(tt["sha256"]),
               part1_binding=np.asarray(bool(e["part1"])),
               **{f"sigma_post_{p}": res["sigma_post"][p] for p in PAIRED_PARAMS})
    lines = [f"== X-battery arm {a} ({e['kind']}; "
             f"{'BINDING Part-1' if e['part1'] else 'NON-BINDING diagnostic'}) ==",
             f"purpose: {e['purpose']}",
             f"expected direction (pre-registered, UNSIGNED n_s): {XA.EXPECTED_DIRECTIONS[a]}"]
    for p in PAIRED_PARAMS:
        gs, p2 = res["stats"][a][p], res["part2"][a][p]
        npz[f"delta_{p}"] = np.asarray([res["deltas"][a][p][m] for m in mocks])
        for k in ("mean", "se", "ub2", "ub_t", "lb_t", "sigma_pair", "t_stat",
                  "p_t", "p_wilcoxon"):
            npz[f"{k}_{p}"] = gs[k]
        npz[f"verdict_{p}"] = np.asarray(gs["verdict"])
        npz[f"D_{p}"] = p2["D"]
        npz[f"S_{p}"] = p2["S"]
        gate_tag = (f"  Part-1 {gs['verdict']}" if (e["part1"] and p in GATE_PARAMS)
                    else "  (reported, not gated)")
        lines.append(
            f"{p:>9}: mean {gs['mean']:+.4f} +/- {gs['se']:.4f} sigma_post  "
            f"[t-lb {gs['lb_t']:.4f}, t-ub {gs['ub_t']:.4f}] vs gate {XA.GATE}; "
            f"sigma_pair {gs['sigma_pair']:.4f}; t {gs['t_stat']:+.2f} "
            f"(p {gs['p_t']:.3g}, Wilcoxon p {gs['p_wilcoxon']:.3g}){gate_tag}")
        lines.append(
            f"{'':>9}  Part-2 DISCLOSURE: S = {p2['S']:.3f} at D = {p2['D']:.3f} "
            f"(noise floor 2SE/D = {p2['noise_floor_2se_over_D']:.3f}; pre-registered "
            f"P(fail|null) = {p2['p_part1_fail_null']:.3f}, expected sigma_pair = "
            f"{p2['sigma_pair_expected']:.3f})")
    for p in GATE_PARAMS:
        u = res["ranks"][a][p]
        lines.append(f"rank u ({p}): mean {np.mean(u):.3f} over {len(u)} mocks "
                     f"(NON-BINDING; non-uniformity pre-registered under displaced truths)")
    txt = os.path.join(out_dir, f"xsel_{a}.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, f"xsel_{a}.npz"), **npz)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for j, p in enumerate(PAIRED_PARAMS):
        dl = npz[f"delta_{p}"]
        ax.scatter(np.full(dl.size, j) + np.linspace(-0.12, 0.12, dl.size), dl, s=12,
                   alpha=0.7, color=f"C{j}")
        gs = res["stats"][a][p]
        ax.errorbar([j], [gs["mean"]], yerr=[2 * gs["se"]], fmt="D", color="k", capsize=3)
    for g in (XA.GATE, -XA.GATE):
        ax.axhline(g, color="r", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(PAIRED_PARAMS)), PAIRED_PARAMS)
    ax.set_ylabel("paired delta (sigma_post units)")
    ax.set_title(f"{a}: paired deltas vs reused K0 "
                 f"({'binding' if e['part1'] else 'non-binding'}; gate {XA.GATE})")
    ax = axes[1]
    zrows = res["tt"]["leg_z"]
    if e["kind"] == "data_swap":
        for key, lab, styl in ((e["table_key"], "X1 corrected (fitted)", "-"),
                               (e["overlay_key"], "X1b diluted (OVERLAY, no fits)", "--")):
            r = res["tt"]["ratios"][key]
            for zz in np.unique(zrows):
                s = zrows == zz
                ax.plot(res["tt"]["leg_k"][s], r[s], styl, lw=0.8, alpha=0.6)
            ax.plot([], [], styl, color="k", label=lab)
        if res["tt"]["band"] is not None:
            lo, hi = res["tt"]["band"]
            ax.fill_between(res["tt"]["leg_k"], lo, hi, alpha=0.15, color="C3",
                            label="wide-mask band (overlay)")
        ax.legend(fontsize=7)
    elif e["kind"] == "mixture_corner":
        r = res["tt"]["ratios"][e["table_key"]]
        for zz in np.unique(zrows):
            s = zrows == zz
            ax.plot(res["tt"]["leg_k"][s], r[s], lw=0.8, alpha=0.7)
    elif e["kind"] == "mixture_profile":
        r3 = res["tt"]["ratios"]["ratio_rows_X3_lls100"]
        r4 = XA.mixture_ratio(XA.f_sel(zrows), r3)
        for zz in np.unique(zrows):
            s = zrows == zz
            ax.plot(res["tt"]["leg_k"][s], r4[s], lw=0.8, alpha=0.7)
        ax.set_title("derived X4 curve: 1 + f_sel(z) (R_X3 - 1)")
    else:                                                     # K8: truth alpha_LLS(z)
        zg = np.asarray(res["meta"][a]["arm_stamp"]["z_global"], float)
        rows = np.stack([np.asarray(res["arms"][a][m]["truth_alpha_hcd_z"], float)
                         for m in sorted(res["arms"][a])])
        k0rows = np.stack([np.asarray(res["k0"][m]["truth_alpha_hcd_z"], float)
                           for m in sorted(res["arms"][a])])
        ax.plot(zg, rows[:, :, 0].T, color="C0", lw=0.7, alpha=0.5)
        ax.plot(zg, k0rows[:, :, 0].T, color="C7", lw=0.7, alpha=0.4)
        ax.set_xlabel("z")
        ax.set_ylabel("truth alpha_LLS(z) (displaced blue vs K0 grey)")
    if e["kind"] != "dndx_displaced":
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("k (leg native)")
        ax.set_ylabel("truth P1D ratio to clean")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"xsel_{a}.png"), dpi=150)
    plt.close(fig)
    return txt


def main_report(shard_dir, k0_dir, out_dir, table_path=None, expect_sha=None):
    res = summarize(shard_dir, k0_dir, table_path, expect_sha)
    arms, meta = res["arms"], res["meta"]
    a0 = next(iter(meta))
    print(f"== X-battery-2 analyzer: {sum(len(r) for r in arms.values())} fits over "
          f"{len(arms)} arms vs the reused K0 baseline ({len(res['k0'])} mocks) ==")
    print(f"forward {meta[a0]['forward']['forward_signature'][:12]}...  "
          f"prior {meta[a0]['forward']['hcd_prior_signature'][:12]}...  "
          f"registry {res['reg_sig'][:12]}...  truth-table sha {res['tt']['sha256'][:12]}...")
    print(f"K0 reuse: PRE-REGISTERED cross-signature allowance (annex OQ5); pair identity + "
          f"stamp equality asserted per pair at seed "
          f"{meta[a0]['run_kw']['seed']} (PAIR_SEED {XA.PAIR_SEED})")
    print(f"sigma_post (K0-pooled posterior sd, PRE-REGISTERED): "
          + "  ".join(f"{p} {res['sigma_post'][p]:.4g}" for p in PAIRED_PARAMS))
    print("DISCLOSURES: X1 mock-noise keeps clean-composition C_emu weights (not repaired); "
          "selection unit = full 120 Mpc/h sightline (annex OQ6); X2 truth uses the DEPLOYED "
          "trough-fill convention (OQ9; window-mask sensitivity 17-57% of low-k template "
          "power documented in the proposal); highest-class partition (OQ10, at-least-one "
          "moves X2/X3 truths ~1-2.5% in band); span warning fires by design at corners.")
    for a in XA.arm_ids():
        if a not in arms:
            print(f"\n-- {a}: NOT PRESENT in this pool")
            continue
        txt = _write_arm_outputs(res, a, out_dir)
        with open(txt) as fh:
            print("\n" + fh.read().rstrip())
        div = sum(int(arms[a][m].get("n_div", 0) > 0) for m in arms[a])
        print(f"divergent fits: {div}/{len(arms[a])}")
    # X1b overlay line (no fits, annex OQ1)
    gp = res["tt"]["gate_power"]["X1b_dla100_diluted"]
    r1 = res["tt"]["ratios"]["ratio_rows_X1_dla100"]
    r1b = res["tt"]["ratios"]["ratio_rows_X1b_dla100_diluted"]
    print(f"\n-- X1b diluted-fork OVERLAY (no fits): band-mean ratio {np.mean(r1b):.3f} vs "
          f"corrected {np.mean(r1):.3f}; D = {gp['D']:.3f}, pre-registered P(fail|null) = "
          f"{gp['p_part1_fail_null']:.3f}. The diluted fork is the exact cache byte object "
          f"and the masking-pipeline-mismatch stress; the CORRECTED fork is the primary "
          f"(KS QMLE excludes masked pixels, annex OQ1).")
    # pre-registration hygiene disclosure (revision 4b, RESOLVED by the one-shot
    # gate-form revision of 2026-07-26: the pilot measured sigma_pair r = 0.21-2.03
    # vs the r < 0.25 calibration requirement, so the binary equivalence gate was
    # replaced pre-launch by the three-outcome interval verdict; memo
    # `2026-07-26-xsel-pilot-sigmapair-and-gate-revision`, 4-lens panel ADOPTED).
    for a in arms:
        if XA.ARMS[a]["part1"]:
            pf = res["tt"]["gate_power"][a]["p_part1_fail_null"]
            if pf > _P_FAIL_NULL_FLAG:
                sp_meas = {p: res["stats"][a][p]["sigma_pair"] for p in GATE_PARAMS}
                print(f"[4b disclosure] {a}: stage-V proxy P(binary-fail | null) = "
                      f"{pf:.3f} > {_P_FAIL_NULL_FLAG}; the pre-registered fallback "
                      f"fired and the gate form was revised BEFORE the full-battery "
                      f"launch (2026-07-26 memo). Expected-vs-measured sigma_pair: "
                      f"proxy {res['tt']['gate_power'][a]['sigma_pair_expected']:.3f} "
                      f"vs measured ns {sp_meas['ns']:.3f} / Ap {sp_meas['Ap']:.3f}.")
    print("\n" + XA.CORNER_FAILURE_PROTOCOL)
    nb = [a for a in arms if not XA.ARMS[a]["part1"]]
    missing_binding = [a for a in XA.part1_arm_ids() if a not in arms]
    if missing_binding:
        print(f"\n** PARTIAL BATTERY: binding arm(s) {missing_binding} absent from this "
              f"pool. The verdict below covers ONLY the present binding arms and is NOT a "
              f"certification readout. **")
    verdict_tbl = {a: {p: res["binding"][a][p]["verdict"] for p in GATE_PARAMS}
                   for a in res["binding"]}
    print(f"\n== PART-1 VERDICTS (three-outcome revised pre-registration, 2026-07-26; "
          f"budget {XA.GATE} sigma_post, t(0.975, n-1) intervals) ==")
    for a in sorted(verdict_tbl):
        print(f"   {a}: " + "  ".join(f"{p}={verdict_tbl[a][p]}" for p in GATE_PARAMS))
    if res["any_binding_fail"]:
        print("   >>> at least one UNPROTECTED (bias detected above budget): apply the "
              "corner-failure protocol (claims limited + threshold study; NOT an "
              "automatic unblinding block, PI decision 8).")
    elif res["any_under_resolved"]:
        print("   >>> no UNPROTECTED, but UNDER-RESOLVED verdicts present: quote the "
              "two-sided bounds; the threshold study carries those corners (PROTECTED "
              "is unreachable for X1/X2 at n=16 at the measured sigma_pair, "
              "pre-registered).")
    else:
        print("   >>> every binding arm PROTECTED within the 0.30 sigma_post budget.")
    print(f"   non-binding diagnostics reported alongside: {nb}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--k0-dir", default=XA.K0_DIR_DEFAULT,
                    help="the reused R4/R5 corrected-geometry K0 shards (annex OQ5)")
    ap.add_argument("--out-dir", default=None,
                    help="per-arm npz/txt/png output dir (default <shard-dir>/readout)")
    ap.add_argument("--truth-table", default=None)
    ap.add_argument("--expect-sha-test-override", default=None,
                    help="TEST-ONLY: bypass the registry sha pin with an explicit sha "
                         "(prints a loud banner; production readouts must use the pin)")
    a = ap.parse_args()
    if a.expect_sha_test_override:
        print("*** TEST-ONLY MODE: truth-table sha supplied on the command line, NOT the "
              "registry pin. This readout is NOT a certification artifact. ***")
    main_report(a.shard_dir, a.k0_dir, a.out_dir or os.path.join(a.shard_dir, "readout"),
                table_path=a.truth_table, expect_sha=a.expect_sha_test_override)


if __name__ == "__main__":
    main()
