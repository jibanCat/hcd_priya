"""R6 paired readout: matched old-vs-new KS parameterization comparison (disposition row 6).

Consumes ONLY ks_r6_{legacy,mapped}_shard_*.pkl written by run_ks_selboost_shard.py --r6-arm
(both arms stamp r6_override=True; the campaign analyzer refuses them and its glob never
matches the ks_r6_ prefix). Each pair = ONE mock (identical truth from the stamped shared
truth source + identical fold_in(seed,m) noise key) fit twice: under the retired legacy
alpha-space KS prior and under the deployed mapped dN/dX prior.

THE MEASUREMENT (the only one of the recorded-as-UNMEASURED defect impact): per-pair
delta_bias(p) = bias_mapped(p) - bias_legacy(p) for p in ns/Ap/tau0_amp/dtau0 (tau0+dtau0
MANDATORY per feedback-report-tau0-dtau0-bias) + the alpha_lls/subdla/dla pivots, reported as
mean +/- paired SE with the per-pair scatter. NO gate threshold here — R6 is supporting
evidence for the certificate, read by the PI (do not oversell: it CANNOT adjudicate
count-vs-occupancy, panel edit 3).

PAIR-IDENTITY ASSERTS (Bayesian design review Q3): the two arms of every pair must carry
bit-identical truth (theta9/tau0/alpha rows) — a pair whose truths differ was NOT run under a
shared truth source (self-draw under each arm's own prior destroys the pairing) and the
readout REFUSES it.

Usage:
  python scripts/analyze_r6_pairs.py --shard-dir <dir> [--npz-out out.npz]
"""
import argparse
import glob
import os
import pickle

import numpy as np

PAIR_PARAMS = ("ns", "Ap", "tau0_amp", "dtau0", "alpha_lls", "alpha_subdla", "alpha_dla")
_RUN_KW_CORE = ("n_warmup", "n_samples", "seed", "max_tree_depth", "dense_mass")


def _col_draws_truth(rec, name):
    names = list(rec["names"])
    if name in names:
        j = names.index(name)
        return np.asarray(rec["draws"])[:, j], float(rec["truth_vec"][j])
    e = rec["sites_extra"][name]
    return np.asarray(e["draws"]), float(e["truth"])


def _bias(rec, name):
    d, t = _col_draws_truth(rec, name)
    return float(d.mean()) - t


def load_r6(shard_dir):
    """{arm: {mock: rec}}, plus the cross-checked meta of each arm. All refusals fire here."""
    by_arm, meta_by_arm = {"legacy": {}, "mapped": {}}, {}
    for arm in ("legacy", "mapped"):
        paths = sorted(glob.glob(os.path.join(shard_dir, f"ks_r6_{arm}_shard_*.pkl")))
        paths = [p for p in paths if ".smoke" not in os.path.basename(p)]
        assert paths, f"no ks_r6_{arm}_shard_*.pkl under {shard_dir}"
        for p in paths:
            with open(p, "rb") as f:
                d = pickle.load(f)
            meta = d["meta"]
            pc = meta["prior_constants"]
            assert pc.get("r6_override") is True, f"{p}: not an R6 pkl (r6_override missing)"
            assert pc.get("r6_arm") == arm, f"{p}: stamped r6_arm {pc.get('r6_arm')!r} != {arm}"
            _expect = {"legacy": "alpha_pivot_powerlaw_v1", "mapped": "dndx_mapped_v2"}[arm]
            assert pc.get("hcd_parameterization") == _expect, \
                f"{p}: parameterization {pc.get('hcd_parameterization')!r} != {_expect} for the {arm} arm"
            if arm in meta_by_arm:
                r = meta_by_arm[arm]
                assert pc.get("r6_truth_source") == r["prior_constants"]["r6_truth_source"], \
                    f"{p}: mixed r6_truth_source within the {arm} arm"
                for sig in ("forward_signature", "hcd_prior_signature"):
                    assert meta["forward"][sig] == r["forward"][sig], \
                        f"{p}: {sig} drift within the {arm} arm"
                for k in _RUN_KW_CORE:
                    assert meta["run_kw"].get(k) == r["run_kw"].get(k), \
                        f"{p}: run_kw[{k}] drift within the {arm} arm"
            else:
                meta_by_arm[arm] = meta
            for rec, mi in zip(d["per_mock"], d["idxs"]):
                assert int(mi) not in by_arm[arm], f"{p}: duplicate mock {mi} in the {arm} arm"
                by_arm[arm][int(mi)] = rec
    # CROSS-ARM stamps: same tree (both signatures — the constants payload covers BOTH eras, so
    # a pair generated astride a constant swap refuses), same truth source, same sampler core.
    mL, mM = meta_by_arm["legacy"], meta_by_arm["mapped"]
    assert mL["forward"]["forward_signature"] == mM["forward"]["forward_signature"], \
        "cross-arm forward_signature mismatch (arms from different trees)"
    assert mL["forward"]["hcd_prior_signature"] == mM["forward"]["hcd_prior_signature"], \
        "cross-arm hcd_prior_signature mismatch (arms astride a prior-constant swap)"
    assert mL["prior_constants"]["r6_truth_source"] == mM["prior_constants"]["r6_truth_source"], \
        "cross-arm r6_truth_source mismatch (pairs do not share a truth source)"
    for k in _RUN_KW_CORE:
        assert mL["run_kw"].get(k) == mM["run_kw"].get(k), \
            f"cross-arm run_kw[{k}] mismatch (pairing needs identical sampler settings + seed)"
    return by_arm, meta_by_arm


def pair_report(shard_dir, npz_out=None):
    by_arm, meta = load_r6(shard_dir)
    common = sorted(set(by_arm["legacy"]) & set(by_arm["mapped"]))
    assert common, "no common mock indices between the two arms"
    orphans = sorted(set(by_arm["legacy"]) ^ set(by_arm["mapped"]))
    # PAIR IDENTITY: bit-identical truth on every pair (theta9 + tau0 + alpha pivot via
    # truth_vec, and the z-resolved alpha rows).
    for m in common:
        rL, rM = by_arm["legacy"][m], by_arm["mapped"][m]
        assert np.array_equal(np.asarray(rL["truth_vec"]), np.asarray(rM["truth_vec"])), \
            f"pair {m}: truth_vec differs between arms — NOT a shared-truth pair (refusing)"
        assert np.array_equal(np.asarray(rL["truth_alpha_hcd_z"]),
                              np.asarray(rM["truth_alpha_hcd_z"])), \
            f"pair {m}: truth_alpha_hcd_z differs between arms — NOT a shared-truth pair"

    ts = meta["mapped"]["prior_constants"]["r6_truth_source"]
    print(f"== R6 paired readout: {len(common)} pairs (orphan mocks skipped: {orphans}) ==")
    print(f"truth source: {ts} (stamped, both arms); "
          f"forward {meta['mapped']['forward']['forward_signature'][:12]}...; "
          f"prior sig {meta['mapped']['forward']['hcd_prior_signature'][:12]}...")
    print("delta = bias_mapped - bias_legacy per pair (posterior-mean bias vs the SHARED truth);")
    print("mean +/- paired SE over pairs. ns/Ap are in theta_unit (unit-cube) coordinates, the")
    print("campaign-analyzer convention. SUPPORTING EVIDENCE, no gate; cannot adjudicate")
    print("count-vs-occupancy (needs external absorber counts).")
    # per-arm divergence accounting (domain review 2026-07-23: a divergence in one arm of a
    # pair biases that pair's delta — the reader must see it).
    for arm in ("legacy", "mapped"):
        nd = {m: int(by_arm[arm][m].get("n_div", 0)) for m in common}
        bad = {m: n for m, n in nd.items() if n > 0}
        print(f"  [{arm}] divergences: total {sum(nd.values())}, "
              f"divergent fits {len(bad)}/{len(common)}"
              + (f" (mocks {sorted(bad)})" if bad else ""))

    out = dict(mocks=np.asarray(common), truth_source=ts)
    for p in PAIR_PARAMS:
        # every PAIR_PARAM exists in BOTH eras (alpha pivots are re-emitted deterministics on
        # the mapped branch; tau0 sites in sites_extra on both) — absence means a broken/stale
        # pkl, never a parameterization difference. HARD refusal (domain review 2026-07-23).
        for arm in ("legacy", "mapped"):
            for m in common:
                rec = by_arm[arm][m]
                assert p in rec["names"] or p in rec.get("sites_extra", {}), \
                    f"param {p!r} absent in the {arm} arm, mock {m} — broken/stale pkl " \
                    f"(every PAIR_PARAM exists in both parameterizations)"
        dl = np.array([_bias(by_arm["mapped"][m], p) - _bias(by_arm["legacy"][m], p)
                       for m in common])
        se = dl.std(ddof=1) / np.sqrt(len(dl)) if len(dl) > 1 else float("nan")
        print(f"  {p:>13}: mean {dl.mean():+.4f} +/- {se:.4f}   per-pair "
              f"{np.array2string(dl, precision=3)}")
        out[f"delta_{p}"] = dl
    if npz_out:
        np.savez(npz_out, **out)
        print(f"wrote {npz_out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--npz-out", default=None)
    a = ap.parse_args()
    pair_report(a.shard_dir, a.npz_out)
