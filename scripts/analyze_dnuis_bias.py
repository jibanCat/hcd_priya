"""Analyze the data-nuisance injection-recovery BIAS shards → per-arm (A_p, n_s) PAIRED bias gate.

Each shard pkl stores a CLEAN run (inject_spec=None) and an INJECTED run at the SAME (seed, mock
index) — so the truth θ AND the cosmic noise ε are shared per mock and only the injected contaminant
differs. We form the per-mock PAIRED bias shift

    Δbias_z(m) = bias_z(injected, m) − bias_z(clean, m),   bias_z = (post_mean − truth) / post_sd

for A_p ("Ap") and n_s ("ns"), matching the clean and injected records by mock index/sim. The shared
noise cancels in the difference, collapsing the unpaired SE≈0.22 to the contaminant-only SE so the
gate can certify <0.3σ with N=8 paired mocks. We ALSO report the CLEAN-arm bias_z (a sanity check —
the Leg-A self-draw clean bias is ~0 by construction).

GATE (a confidence-bound gate, not just |mean|): PASS requires |mean Δbias_z| + 2·SE < 0.30σ for
BOTH params. For arm=lls_excess the threshold is 0.50σ and the cell is FLAGGED (not hard-failed)
between 0.30 and 0.50 (the documented irreducible HCD→n_s budget). The upper bound |mean|+2·SE is
printed explicitly.

Column indexing reuses run_legb's by-NAME positional map (θ9 → j; the α/τ₀ block offset by the
per-mock kept-z count), so it stays aligned with the metals/hierarchical appended columns.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/analyze_dnuis_bias.py --shard-dir <dir>
"""
import argparse
import functools
import glob
import os
import pickle

import numpy as np

print = functools.partial(print, flush=True)

from hcd_analysis.emulator.inference import PARAM_NAMES

N_THETA = 9
GATE = 0.30                       # |mean Δbias_z| + 2·SE pass threshold (σ)
LLS_BUDGET = 0.50                 # documented irreducible HCD→n_s budget (flag, not auto-fail)
PARAMS = ("Ap", "ns")            # the two blinded cosmology params we gate


def _col_of(rec, j):
    """run_legb's by-name positional column map (mirrors closure_legb._aggregate_legb._col_of):
    θ9 at j<9; the α/τ₀ block shifts by the per-mock kept-z count."""
    if j < N_THETA:
        return j
    n_kept = int(np.asarray(rec["kept_global"]).sum())
    return N_THETA + n_kept + (j - N_THETA)


def _param_index(name):
    return list(PARAM_NAMES).index(name)


def _bias_z(rec, j):
    """Signed bias_z = (post_mean − truth) / post_sd for column j of a single record. Returns
    None if the posterior sd is 0 (degenerate column)."""
    col = _col_of(rec, j)
    dc = np.asarray(rec["draws"])[:, col]
    sd = float(dc.std())
    if sd <= 0:
        return None
    return (float(dc.mean()) - float(rec["truth_vec"][col])) / sd


def _rec_key(rec):
    """A stable per-mock identity to pair clean↔injected: (sim, kept-z count). For Leg-A all
    mocks share sim='leg_a_prior', so the LIST ORDER (the mock index order run_legb iterates) is
    the true pairing — we pair positionally and only use this key to assert alignment."""
    return (rec.get("sim"), int(np.asarray(rec["kept_global"]).sum()))


def paired_delta_bias(clean_per_mock, inj_per_mock, param):
    """Per-mock paired Δbias_z = bias(injected) − bias(clean) for `param`, pooled over matched
    mocks (positional pairing — both runs iterate mock_indices in the SAME order at the SAME seed).
    Also returns the clean-arm bias_z array (the ~0 sanity check). Skips a pair if either record's
    posterior sd is 0."""
    j = _param_index(param)
    n = min(len(clean_per_mock), len(inj_per_mock))
    deltas, clean_zs = [], []
    for m in range(n):
        rc, ri = clean_per_mock[m], inj_per_mock[m]
        assert _rec_key(rc) == _rec_key(ri), \
            f"clean/injected mock {m} misaligned: {_rec_key(rc)} != {_rec_key(ri)}"
        zc, zi = _bias_z(rc, j), _bias_z(ri, j)
        if zc is None or zi is None:
            continue
        deltas.append(zi - zc)
        clean_zs.append(zc)
    return np.array(deltas), np.array(clean_zs)


def _bias_z_named(rec, name):
    """bias_z for a column addressed by its NAME in rec['names'] — covers the appended blocks
    PARAM_NAMES does not (tau0_z* mean-flux ladder, a_SiIII, alpha_*)."""
    names = list(rec["names"])
    if name not in names:
        return None
    col = names.index(name)
    dc = np.asarray(rec["draws"])[:, col]
    sd = float(dc.std())
    if sd <= 0:
        return None
    return (float(dc.mean()) - float(rec["truth_vec"][col])) / sd


def paired_delta_named(clean_per_mock, inj_per_mock, name):
    """Per-mock paired Δbias_z = bias(inj) − bias(clean) for a NAMED column (positional pairing,
    same convention as paired_delta_bias). Returns (deltas, clean_zs)."""
    n = min(len(clean_per_mock), len(inj_per_mock))
    deltas, clean_zs = [], []
    for m in range(n):
        zc = _bias_z_named(clean_per_mock[m], name)
        zi = _bias_z_named(inj_per_mock[m], name)
        if zc is None or zi is None:
            continue
        deltas.append(zi - zc)
        clean_zs.append(zc)
    return np.array(deltas), np.array(clean_zs)


def load_shards(shard_dir, arm=None, survey=None):
    """Load + group shard pkls by (arm, survey), concatenating clean/injected per-mock lists.
    Filter by arm/survey if given. Returns {(arm,survey): (clean_per_mock, inj_per_mock, ndiv)}."""
    pat = os.path.join(shard_dir, "*_shard_*.pkl")
    groups = {}
    for p in sorted(glob.glob(pat)):
        with open(p, "rb") as f:
            d = pickle.load(f)
        a, s = d.get("arm"), d.get("survey")
        if arm and a != arm:
            continue
        if survey and s != survey:
            continue
        if "clean_per_mock" not in d or "inj_per_mock" not in d:
            raise SystemExit(
                f"{p}: not a PAIRED shard (missing clean_per_mock/inj_per_mock). Re-run the shard "
                f"with the paired run_dnuis_bias_shard.py.")
        cl, inj, nd = groups.setdefault((a, s), ([], [], [0]))
        cl.extend(d["clean_per_mock"])
        inj.extend(d["inj_per_mock"])
        nd[0] += (sum(int(r.get("n_div", 0) > 0) for r in d["clean_per_mock"])
                  + sum(int(r.get("n_div", 0) > 0) for r in d["inj_per_mock"]))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--arm", default=None,
                    choices=[None, "metal_misspec", "resolution", "lls_excess", "metal_matched"])
    ap.add_argument("--survey", default=None, choices=[None, "desi", "ks", "eboss"])
    a = ap.parse_args()

    groups = load_shards(a.shard_dir, a.arm, a.survey)
    if not groups:
        raise SystemExit(f"no shard pkls in {a.shard_dir} (arm={a.arm} survey={a.survey})")

    hdr = (f"{'arm/survey':<22} {'param':<4} {'n':>4} {'ndiv':>4} "
           f"{'Δmean_z':>9} {'SE':>7} {'|Δ|+2SE':>8} {'clean_z':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    any_fail = False
    for (arm, survey), (clean_pm, inj_pm, nd) in sorted(groups.items()):
        n_div = nd[0]
        for param in PARAMS:
            deltas, clean_zs = paired_delta_bias(clean_pm, inj_pm, param)
            n = deltas.size
            mean = float(deltas.mean()) if n else np.nan
            se = float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else (
                float(abs(deltas[0])) if n == 1 else np.nan)
            ub = abs(mean) + 2.0 * se if n else np.nan       # the confidence-bound gate statistic
            clean_mean = float(clean_zs.mean()) if clean_zs.size else np.nan
            thr = LLS_BUDGET if arm == "lls_excess" else GATE
            ok = ub < GATE
            if arm == "lls_excess" and not ok and ub < LLS_BUDGET:
                verdict = f"FLAG (≤{LLS_BUDGET:.2f}σ HCD→n_s budget)"
            elif ub < thr:
                verdict = "PASS"
            else:
                verdict = "FAIL"
                any_fail = True
            print(f"{arm + '/' + survey:<22} {param:<4} {n:>4} {n_div:>4} "
                  f"{mean:>+9.3f} {se:>7.3f} {ub:>8.3f} {clean_mean:>+8.3f}  {verdict}")
    print("-" * len(hdr))
    print(f"GATE: |mean Δbias_z| + 2·SE < {GATE:.2f}σ for Ap & ns "
          f"(lls_excess flagged up to {LLS_BUDGET:.2f}σ, not auto-failed). "
          f"clean_z is the unpaired clean-arm bias (sanity ~0).")
    print("OVERALL:", "FAIL — at least one non-LLS arm exceeds the confidence-bound gate."
          if any_fail else "PASS — all non-LLS arms within the confidence-bound bias gate.")

    # --- standing-rule mean-flux + metal-nuisance report (NOT gated): the model samples the per-z
    # tau0_z ladder (no global tau0_amp/dtau0 sites), so we report the tau0_z Δbias averaged over z
    # (mean-flux amplitude proxy) + its z-range, plus a_SiIII (the metal nuisance the fit floats). The
    # mean flux is the suspected n_s channel — does a contaminant push it? ---
    print()
    print("MEAN-FLUX (tau0_z ladder, paired Δbias_z) + a_SiIII — report only, not gated:")
    rh = (f"{'arm/survey':<22} {'tau0z mean':>10} {'tau0z[min,max]':>18} "
          f"{'a_SiIII Δ':>10} {'a_SiIII clean':>13}")
    print(rh)
    print("-" * len(rh))
    for (arm, survey), (clean_pm, inj_pm, nd) in sorted(groups.items()):
        tau0_nms = [nm for nm in list(clean_pm[0]["names"]) if nm.startswith("tau0_z")]
        tz = [float(d[0].mean()) for d in (paired_delta_named(clean_pm, inj_pm, nm)
                                           for nm in tau0_nms) if d[0].size]
        da, da_clean = paired_delta_named(clean_pm, inj_pm, "a_SiIII")
        tz_mean = f"{np.mean(tz):>+10.3f}" if tz else f"{'n/a':>10}"
        rng = f"[{min(tz):+.2f},{max(tz):+.2f}]" if tz else ""
        a_s = f"{da.mean():>+10.3f}" if da.size else f"{'n/a':>10}"
        a_c = f"{da_clean.mean():>+13.3f}" if da_clean.size else f"{'n/a':>13}"
        print(f"{arm + '/' + survey:<22} {tz_mean} {rng:>18} {a_s} {a_c}")


if __name__ == "__main__":
    main()
