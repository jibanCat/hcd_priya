"""Cross-leg r6x paired readout: per-leg LLS prior-centre sensitivity (PI decisions #7,
execution annex 2026-07-24; generalizes analyze_r6_pairs.py to the r6x arm pair).

Consumes ONLY r6x_<leg>_{deployed,dispprior}_shard_*.pkl written by
run_crossleg_r6_shard.py. Each pair = ONE mock (identical deployed-selfdraw truth + identical
fold_in(seed,m) noise key, seed 20260724) fit twice: under the DEPLOYED per-leg prior and
under the DISPLACED prior (LLS centre x1.287, fixed absolute width).

SIGNATURE-PAIR CONVENTION (panel revision 5, CS MUST-FIX): analyze_r6_pairs.py:86 asserts
prior-signature EQUALITY across arms; r6x arms DIFFER by design. This analyzer pins the
PRE-REGISTERED (deployed hex, displaced hex) PAIR per leg (crossleg_r6_common) and refuses
anything else -- including the two arms carrying EQUAL signatures (an override that never
happened).

SCOPE (PI amendment #6): every number here is "LLS-amplitude prior-centre sensitivity", one
displacement direction; never "the HCD degeneracy", never the legacy-vs-mapped migration test.

Outputs (pre-registered, proposal section C):
  1. per-pair delta table (delta = bias_dispprior - bias_deployed on the shared mock) for
     n_s, A_p, tau0_amp, dtau0 (mandatory, feedback-report-tau0-dtau0-bias), the alpha pivots,
     a_SiIII where sampled (DESI/eBOSS) and the DESI metal f/k node sites where present;
  2. full-sample paired mean +/- SE, theta AND physical units (n_s box 0.25, A_p box 1.4e-9);
  3. LOO means, LOO t-range, sign-consistency counts;
  4. deployed-arm absolute biases vs the pre-registered |mean| <= 2 SE tolerance (n=8);
  5. implementation health: pair identity (bit-identical truths + data sha), stamp
     homogeneity, divergence accounting, orphan check;
  6. the pre-registered under-resolution statement (panel revision 7) and the either-channel
     n=12 extension-rule verdict line;
  7. an npz + txt record.

Usage:
  python scripts/analyze_crossleg_r6.py --shard-dir DIR --leg eBOSS|DESI|KS
      [--npz-out out.npz] [--txt-out out.txt] [--smoke]
"""
import argparse
import glob
import io
import os
import pickle

import numpy as np

import scripts.crossleg_r6_common as CC

PAIR_PARAMS = ("ns", "Ap", "tau0_amp", "dtau0", "alpha_lls", "alpha_subdla", "alpha_dla")
OPTIONAL_PARAMS = ("a_SiIII",)                       # sampled on DESI/eBOSS only
METAL_NODE_PREFIXES = ("f_SiIII_", "f_SiII_", "k_SiIII_", "k_SiII_")
_RUN_KW_CORE = ("n_warmup", "n_samples", "seed", "max_tree_depth", "dense_mass")


def _col_draws_truth(rec, name):
    names = list(rec["names"])
    if name in names:
        j = names.index(name)
        return np.asarray(rec["draws"])[:, j], float(rec["truth_vec"][j])
    e = rec["sites_extra"][name]
    return np.asarray(e["draws"]), float(e["truth"])


def _has_param(rec, name):
    return name in list(rec["names"]) or name in rec.get("sites_extra", {})


def _bias(rec, name):
    d, t = _col_draws_truth(rec, name)
    return float(d.mean()) - t


def load_r6x(shard_dir, leg, *, smoke=False):
    """{arm: {mock: rec}} + per-arm meta. EVERY refusal fires here, fail-loud."""
    assert leg in CC.R6X_LEGS, f"unknown leg {leg!r} (expected one of {CC.R6X_LEGS})"
    by_arm, meta_by_arm = {a: {} for a in CC.R6X_ARMS}, {}
    for arm in CC.R6X_ARMS:
        pat = os.path.join(shard_dir, f"r6x_{leg.lower()}_{arm}_shard_*.pkl")
        paths = sorted(glob.glob(pat))
        # SMOKE convention (F6a): a real readout NEVER pools .smoke pkls; --smoke reads ONLY them.
        paths = [p for p in paths if (".smoke" in os.path.basename(p)) == bool(smoke)]
        assert paths, f"no {'smoke ' if smoke else ''}r6x_{leg.lower()}_{arm}_shard_*.pkl under {shard_dir}"
        for p in paths:
            with open(p, "rb") as f:
                d = pickle.load(f)
            meta = d["meta"]
            pc = meta["prior_constants"]
            # --- per-pkl stamp refusals -------------------------------------------------
            assert pc.get("r6x_override") is True, \
                f"{p}: r6x_override stamp missing/false -- not an r6x pkl (refusing)"
            assert pc.get("r6x_arm") == arm, f"{p}: stamped r6x_arm {pc.get('r6x_arm')!r} != {arm}"
            assert pc.get("r6x_leg") == leg, f"{p}: stamped r6x_leg {pc.get('r6x_leg')!r} != {leg}"
            assert int(pc.get("r6x_seed", -1)) == CC.R6X_SEED, \
                f"{p}: r6x_seed {pc.get('r6x_seed')!r} != the pre-registered fresh seed {CC.R6X_SEED}"
            assert pc.get("r6x_truth_source") == CC.R6X_TRUTH_SOURCE, \
                f"{p}: r6x_truth_source {pc.get('r6x_truth_source')!r} != {CC.R6X_TRUTH_SOURCE!r}"
            assert pc.get("hcd_parameterization") == CC.R6X_PARAMETERIZATION[leg], (
                f"{p}: parameterization {pc.get('hcd_parameterization')!r} != deployed "
                f"{CC.R6X_PARAMETERIZATION[leg]!r} for {leg} (BOTH r6x arms keep the deployed "
                f"parameterization; only the centre moves)")
            # --- PRE-REGISTERED SIGNATURE PAIR (never equality across arms) --------------
            want_hex = CC.expected_hex(leg, arm)
            for src, got in (("prior_constants", pc.get("hcd_prior_signature")),
                             ("forward", meta["forward"].get("hcd_prior_signature"))):
                assert got == want_hex, (
                    f"{p}: {src} hcd_prior_signature {str(got)[:12]}... != the pre-registered "
                    f"{arm} hex {want_hex[:12]}... for {leg} -- signature-pair refusal "
                    f"(panel revision 5)")
            assert int(meta["run_kw"].get("seed", -1)) == CC.R6X_SEED, \
                f"{p}: run_kw seed {meta['run_kw'].get('seed')!r} != {CC.R6X_SEED} (seed drift)"
            # --- within-arm homogeneity ---------------------------------------------------
            if arm in meta_by_arm:
                r = meta_by_arm[arm]
                for sig in ("forward_signature", "hcd_prior_signature"):
                    assert meta["forward"][sig] == r["forward"][sig], \
                        f"{p}: {sig} drift within the {arm} arm"
                for k in _RUN_KW_CORE:
                    assert meta["run_kw"].get(k) == r["run_kw"].get(k), \
                        f"{p}: run_kw[{k}] drift within the {arm} arm"
            else:
                meta_by_arm[arm] = meta
            for rec, mi in zip(d["per_mock"], d["idxs"]):
                assert int(mi) not in by_arm[arm], \
                    f"{p}: duplicate mock {mi} in the {arm} arm (two pkls claim it) -- refusing"
                by_arm[arm][int(mi)] = rec
    # --- cross-arm stamps ------------------------------------------------------------------
    mD, mX = meta_by_arm["deployed"], meta_by_arm["dispprior"]
    assert mD["forward"]["forward_signature"] == mX["forward"]["forward_signature"], \
        "cross-arm forward_signature mismatch (arms from different trees/forwards)"
    hd = mD["forward"]["hcd_prior_signature"]
    hx = mX["forward"]["hcd_prior_signature"]
    assert hd != hx, ("cross-arm hcd_prior_signature EQUALITY: the displaced arm carries the "
                      "deployed prior signature -- the override never happened (refusing; the "
                      "r6x convention is the pre-registered PAIR, not equality)")
    assert [hd, hx] == [CC.R6X_DEPLOYED_HEX, CC.R6X_DISPLACED_HEX[leg]], \
        "cross-arm signature pair != the pre-registered (deployed, displaced) pair"
    for k in _RUN_KW_CORE:
        assert mD["run_kw"].get(k) == mX["run_kw"].get(k), \
            f"cross-arm run_kw[{k}] mismatch (pairing needs identical sampler settings + seed)"
    return by_arm, meta_by_arm


def _pair_identity(by_arm, common):
    """Bit-identical truths + data-vector sha across every pair, else refuse."""
    for m in common:
        rD, rX = by_arm["deployed"][m], by_arm["dispprior"][m]
        assert np.array_equal(np.asarray(rD["truth_vec"]), np.asarray(rX["truth_vec"])), \
            f"pair {m}: truth_vec differs between arms -- NOT a shared-truth pair (refusing)"
        assert np.array_equal(np.asarray(rD["truth_alpha_hcd_z"]),
                              np.asarray(rX["truth_alpha_hcd_z"])), \
            f"pair {m}: truth_alpha_hcd_z differs between arms -- NOT a shared-truth pair"
        sD, sX = rD.get("r6x_data_sha256"), rX.get("r6x_data_sha256")
        assert sD is not None and sX is not None, \
            f"pair {m}: r6x_data_sha256 stamp missing -- cannot certify data bit-identity"
        assert sD == sX, (f"pair {m}: mock data sha256 differs between arms -- the pair's "
                          f"data vectors are NOT bit-identical (refusing)")


def _discover_params(by_arm, common):
    """The final param list: PAIR_PARAMS (hard-required in both arms, R6 convention) +
    OPTIONAL_PARAMS and metal node sites present in BOTH arms for EVERY common mock."""
    params = list(PAIR_PARAMS)
    for p in PAIR_PARAMS:
        for arm in CC.R6X_ARMS:
            for m in common:
                assert _has_param(by_arm[arm][m], p), \
                    f"param {p!r} absent in the {arm} arm, mock {m} -- broken/stale pkl"
    cands = list(OPTIONAL_PARAMS)
    # metal node sites: union of prefixed sites_extra keys, must then be homogeneous.
    node_names = set()
    for arm in CC.R6X_ARMS:
        for m in common:
            node_names |= {k for k in by_arm[arm][m].get("sites_extra", {})
                           if k.startswith(METAL_NODE_PREFIXES)}
    cands += sorted(node_names)
    for p in cands:
        present = [_has_param(by_arm[arm][m], p) for arm in CC.R6X_ARMS for m in common]
        if not any(present):
            continue
        assert all(present), (f"param {p!r} present in SOME but not all (arm, mock) cells -- "
                              f"inhomogeneous pkl population (refusing)")
        params.append(p)
    return params


def _loo_stats(dl):
    """(loo_means, loo_t_min, loo_t_max, sign_consistent_count)."""
    n = len(dl)
    loo_means, loo_t = [], []
    for i in range(n):
        rest = np.delete(dl, i)
        mu = rest.mean()
        se = rest.std(ddof=1) / np.sqrt(len(rest)) if len(rest) > 1 else np.nan
        loo_means.append(mu)
        loo_t.append(mu / se if se > 0 else np.nan)
    sign = int(np.sum(np.sign(dl) == np.sign(dl.mean()))) if dl.mean() != 0 else 0
    return np.asarray(loo_means), float(np.nanmin(loo_t)), float(np.nanmax(loo_t)), sign


def pair_report(shard_dir, leg, npz_out=None, txt_out=None, smoke=False):
    buf = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=buf)

    by_arm, meta = load_r6x(shard_dir, leg, smoke=smoke)
    common = sorted(set(by_arm["deployed"]) & set(by_arm["dispprior"]))
    assert common, "no common mock indices between the two arms"
    orphans = sorted(set(by_arm["deployed"]) ^ set(by_arm["dispprior"]))
    _pair_identity(by_arm, common)                 # refusals BEFORE any statistics gate
    params = _discover_params(by_arm, common)
    assert len(common) >= 2 or smoke, "need >= 2 pairs for a paired SE (non-smoke)"

    say(f"== r6x paired readout, leg {leg}: {len(common)} pairs "
        f"(orphan mocks skipped: {orphans}) ==")
    say(f"SCOPE: LLS-amplitude prior-centre sensitivity, ONE direction (x{CC.R6X_FACTOR}); "
        f"NOT 'the HCD degeneracy'; NOT the legacy-vs-mapped migration test (PI amendment #6).")
    say(f"truth source: {CC.R6X_TRUTH_SOURCE} (seed {CC.R6X_SEED}); displacement: "
        f"{CC.R6X_DISP_SPEC[leg]}")
    say(f"signature pair (pre-registered): deployed {CC.R6X_DEPLOYED_HEX[:12]}... / "
        f"displaced {CC.R6X_DISPLACED_HEX[leg][:12]}...; forward "
        f"{meta['deployed']['forward']['forward_signature'][:12]}... (equal across arms)")
    say("delta = bias_dispprior - bias_deployed per pair (posterior-mean bias vs the SHARED "
        "truth); ns/Ap in theta units (physical: boxes 0.25 / 1.4e-9).")

    # divergence accounting (0-divergence expectation; a divergence in one arm biases that
    # pair's delta -- the reader must see it).
    div_total = {}
    for arm in CC.R6X_ARMS:
        nd = {m: int(by_arm[arm][m].get("n_div", 0)) for m in common}
        bad = {m: n for m, n in nd.items() if n > 0}
        div_total[arm] = sum(nd.values())
        say(f"  [{arm}] divergences: total {div_total[arm]}, divergent fits "
            f"{len(bad)}/{len(common)}" + (f" (mocks {sorted(bad)})" if bad else ""))
    if any(div_total.values()):
        say("  WARNING: nonzero divergences -- flagged pairs' deltas may be biased "
            "(pre-registered health caveat; not silently dropped).")

    out = dict(mocks=np.asarray(common), leg=leg, orphans=np.asarray(orphans),
               signature_pair=[CC.R6X_DEPLOYED_HEX, CC.R6X_DISPLACED_HEX[leg]],
               n_div_deployed=div_total["deployed"], n_div_dispprior=div_total["dispprior"])

    say("\n-- per-pair deltas + paired means (theta units; physical in [] where defined) --")
    t_by_param = {}
    for p in params:
        dl = np.array([_bias(by_arm["dispprior"][m], p) - _bias(by_arm["deployed"][m], p)
                       for m in common])
        se = dl.std(ddof=1) / np.sqrt(len(dl)) if len(dl) > 1 else float("nan")
        t = dl.mean() / se if se > 0 else float("nan")
        t_by_param[p] = t
        phys = ""
        if p in CC.R6X_THETA_BOX:
            b = CC.R6X_THETA_BOX[p]
            phys = f"  [phys {dl.mean() * b:+.3e} +/- {se * b:.3e}]"
        loo_means, t_lo, t_hi, sign = _loo_stats(dl) if len(dl) > 2 else \
            (np.asarray([]), float("nan"), float("nan"), 0)
        say(f"  {p:>13}: mean {dl.mean():+.4f} +/- {se:.4f}  (t={t:+.2f}){phys}")
        say(f"  {'':>13}  per-pair {np.array2string(dl, precision=3)}")
        if len(dl) > 2:
            say(f"  {'':>13}  LOO mean [{loo_means.min():+.4f}, {loo_means.max():+.4f}]  "
                f"LOO t [{t_lo:+.2f}, {t_hi:+.2f}]  sign-consistent {sign}/{len(dl)}")
        out[f"delta_{p}"] = dl
        out[f"t_{p}"] = t
        out[f"loo_means_{p}"] = loo_means
        out[f"loo_t_range_{p}"] = np.asarray([t_lo, t_hi])
        out[f"sign_consistent_{p}"] = sign

    say("\n-- deployed-arm absolute biases (self-draw coherence check; pre-registered "
        "tolerance |mean| <= 2 SE at n=8) --")
    coh_fail = []
    for p in params:
        b = np.array([_bias(by_arm["deployed"][m], p) for m in common])
        se = b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else float("nan")
        ok = abs(b.mean()) <= 2.0 * se if np.isfinite(se) else True
        if not ok:
            coh_fail.append(p)
        say(f"  {p:>13}: mean {b.mean():+.4f} +/- {se:.4f}  "
            f"{'OK' if ok else 'EXCEEDS |mean|<=2SE (flag)'}")
        out[f"deployed_bias_{p}"] = b
    out["deployed_coherence_flags"] = np.asarray(coh_fail, dtype=object)
    if coh_fail:
        say(f"  FLAG: deployed-arm coherence tolerance exceeded for {coh_fail} -- the "
            f"deployed self-draw reference is NOT ~0 there; readout must discuss before use.")

    # under-resolution statement (panel revision 7, pre-registered wording).
    say("\n-- sensitivity vs the KS-R6-scale target --")
    for p in ("ns", "Ap"):
        dl = out[f"delta_{p}"]
        se = dl.std(ddof=1) / np.sqrt(len(dl)) if len(dl) > 1 else float("nan")
        tgt = CC.R6X_TARGET_SE_THETA[p]
        if not np.isfinite(se) or se > tgt:
            say(f"  {p}: paired SE {se:.4f} (theta) EXCEEDS the KS-scale target {tgt} -- this "
                f"leg UNDER-RESOLVES the target on {p}; the result must be quoted as "
                f"'under-resolves', never 'consistent with zero' alone (pre-registered).")
            out[f"under_resolves_{p}"] = True
        else:
            say(f"  {p}: paired SE {se:.4f} (theta) resolves the KS-scale target {tgt}.")
            out[f"under_resolves_{p}"] = False

    extend, line = CC.extension_verdict(t_by_param["ns"], t_by_param["Ap"])
    say("\n" + line)
    out["extend"] = extend
    out["t_ns"], out["t_Ap"] = t_by_param["ns"], t_by_param["Ap"]

    say(f"\ntau0_amp/dtau0 line (standing convention): delta tau0_amp "
        f"{out['delta_tau0_amp'].mean():+.4f} +/- "
        f"{out['delta_tau0_amp'].std(ddof=1) / np.sqrt(len(common)):.4f}, delta dtau0 "
        f"{out['delta_dtau0'].mean():+.4f} +/- "
        f"{out['delta_dtau0'].std(ddof=1) / np.sqrt(len(common)):.4f}")

    if npz_out:
        np.savez(npz_out, **{k: v for k, v in out.items() if not isinstance(v, str)})
        say(f"wrote {npz_out}")
    if txt_out:
        with open(txt_out, "w") as f:
            f.write(buf.getvalue())
        print(f"wrote {txt_out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--leg", required=True, choices=list(CC.R6X_LEGS))
    ap.add_argument("--npz-out", default=None)
    ap.add_argument("--txt-out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="read ONLY .smoke pkls (a real readout never pools them)")
    a = ap.parse_args()
    pair_report(a.shard_dir, a.leg, a.npz_out, a.txt_out, smoke=a.smoke)
