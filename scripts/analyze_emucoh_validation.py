#!/usr/bin/env python3
"""Phase-5a EMUCOH closure validation — read the referee triad from the EC0/EC1 arms.

The 60-sim LF-emulator k-coherent C_emu term ("emucoh") is built + unit-tested + 4-referee-reviewed
but has NO inference-level validation. The Bayesian + cosmology referees made that the BLOCKING gate
before production. This script reads it from the matched closure arms added by run_stepA.build_config:

  for each of 8 multi-fold mocks:  <base>_EC0  (mf_emucoh=0, OFF control)
                                   <base>_EC1  (mf_emucoh=1, ON, infl=1)
  BOTH run fresh with current code; same fold+sim+seed ⇒ byte-identical mock + noise; ONLY the
  likelihood covariance differs. (The OFF arm is run fresh, not the old D_f*/D_lmed* checkpoints, so
  code drift cannot confound emucoh.)

THE TRIAD (referee-mandated; emucoh-cemu.md §4 meta-synthesis action 1):
  (A) per-fold point-estimate SCATTER shrinks — RMS across folds of the point estimate's offset from
      truth, measured on a FIXED σ_OFF ruler (so it reflects the posterior MEAN moving, not σ growth).
  (B) σ_Ap / σ_ns WIDEN (PSD monotonicity) — σ_ON/σ_OFF ≥ 1 per mock. A SHRINK is the signature of an
      accidental θ-dependent covariance (the live-P pathology, shape-floor.md §4.3) → flagged loudly.
  (C) coverage → nominal — |bias z| = |post-mean − truth|/σ per mock, OFF vs ON, should move toward ≤1.

SIGN CONVENTION: we report bias_z = (post_mean − truth)/σ (the "report"/post−truth convention). NOTE
the run_stepA battery / health.json print the OPPOSITE sign (truth − post). Scatter/σ/coverage are all
magnitude-based, so the sign only affects the per-mock signed columns (labelled).

Usage (emu-3.9 has matplotlib; emu-jax also fine for the numbers):
  PYTHONPATH=/home/mfho/hcd_priya python3 scripts/analyze_emucoh_validation.py [--no-fig]
Robust to partial completion: a mock with <4 chains on either arm is skipped with a note.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/stepA"
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_multifidelity"
OUT_JSON = f"{CKPT}/emucoh_validation.json"

# the 8 baselines mirrored as EC0/EC1; family A = sim-mean center, family B = matched-lit σ0.15.
FAMILY = {
    "D_f3": "A", "D_f4": "A", "D_f6": "A", "D_f7": "A",
    "D_lmed3_15": "B", "D_lmed5_15": "B", "D_lmed7_15": "B", "D_llsmed": "B",
}
BASES = list(FAMILY)
PARAMS = ["Ap", "ns", "alpha_lls", "alpha_subdla", "alpha_dla"]
KEY = ["Ap", "ns"]            # the two the gate is about
SUMMARY_PARAMS = ["Ap", "ns", "alpha_subdla"]   # + alpha_subdla so its miscoverage is summarized


_CFG_CACHE = None


def _cfg():
    global _CFG_CACHE
    if _CFG_CACHE is None:
        sys.path.insert(0, f"{REPO}/scripts")
        import run_stepA  # noqa
        _CFG_CACHE = run_stepA.build_config()
    return _CFG_CACHE


def _chain_ids(mock_id):
    return [c["id"] for c in _cfg() if c["mock_id"] == mock_id]


def _pool(mock_id):
    """Pool the (≤4) chains of a mock → (names, pooled draws (C*N,P), truth_vec, n_chains, n_div)
    or None if no chains present yet."""
    ids = _chain_ids(mock_id)
    paths = [f"{CKPT}/{cid}.npz" for cid in ids]
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        return None
    packs, names, truth, ndiv = [], None, None, 0
    for p in have:
        z = np.load(p, allow_pickle=True)
        packs.append(np.asarray(z["packed"]))
        names = [str(x) for x in z["names"]] if names is None else names
        truth = np.asarray(z["truth_vec"]) if truth is None else truth
        ndiv += int(z["divergences"])
    nmin = min(p.shape[0] for p in packs)
    pooled = np.concatenate([p[:nmin] for p in packs], axis=0)
    return dict(names=names, pooled=pooled, truth=truth, n_chains=len(have),
                n_chains_expected=len(ids), n_div=ndiv)


def _stat(pool, pname):
    j = pool["names"].index(pname)
    col = pool["pooled"][:, j]
    return float(pool["truth"][j]), float(col.mean()), float(col.std())


def _battery(mock_id):
    """R-hat / ESS via run_stepA._battery_over_chains (None if not all 4 chains in)."""
    sys.path.insert(0, f"{REPO}/scripts")
    import run_stepA  # noqa
    ids = _chain_ids(mock_id)
    if not all(os.path.exists(f"{CKPT}/{cid}.npz") for cid in ids):
        return None
    try:
        return run_stepA._battery_over_chains(ids)
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def analyze():
    rows = []
    for base in BASES:
        off = _pool(f"{base}_EC0")
        on = _pool(f"{base}_EC1")
        if off is None or on is None:
            rows.append(dict(base=base, family=FAMILY[base], status="missing",
                             off_chains=(0 if off is None else off["n_chains"]),
                             on_chains=(0 if on is None else on["n_chains"])))
            continue
        rec = dict(base=base, family=FAMILY[base], status="ok",
                   off_chains=off["n_chains"], on_chains=on["n_chains"],
                   off_chains_exp=off["n_chains_expected"], on_chains_exp=on["n_chains_expected"],
                   off_div=off["n_div"], on_div=on["n_div"], params={})
        for p in PARAMS:
            if p not in off["names"]:
                continue
            t, m0, s0 = _stat(off, p)
            _, m1, s1 = _stat(on, p)
            rec["params"][p] = dict(
                truth=t,
                off_mean=m0, off_sd=s0, on_mean=m1, on_sd=s1,
                # bias_z = (post-mean − truth)/σ  (report convention)
                bias_z_off=(m0 - t) / s0 if s0 > 0 else None,
                bias_z_on=(m1 - t) / s1 if s1 > 0 else None,
                # FIXED-ruler point-estimate offset (same σ_OFF denominator → isolates the mean moving)
                pe_off_fixed=(m0 - t) / s0 if s0 > 0 else None,
                pe_on_fixed=(m1 - t) / s0 if s0 > 0 else None,
                sd_ratio=(s1 / s0) if s0 > 0 else None,
            )
        rec["battery_off"] = _battery(f"{base}_EC0")
        rec["battery_on"] = _battery(f"{base}_EC1")
        rows.append(rec)
    return rows


def _rms(xs):
    xs = [x for x in xs if x is not None]
    return float(np.sqrt(np.mean(np.square(xs)))) if xs else None


def summarize(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    out = {"n_mocks_ready": len(ok), "n_mocks_total": len(rows), "by_family": {}, "overall": {}}
    groups = {"A": [r for r in ok if r["family"] == "A"],
              "B": [r for r in ok if r["family"] == "B"],
              "ALL": ok}
    for g, rs in groups.items():
        d = {}
        for p in SUMMARY_PARAMS:
            pe_off = [r["params"].get(p, {}).get("pe_off_fixed") for r in rs if p in r["params"]]
            pe_on = [r["params"].get(p, {}).get("pe_on_fixed") for r in rs if p in r["params"]]
            bz_off = [r["params"].get(p, {}).get("bias_z_off") for r in rs if p in r["params"]]
            bz_on = [r["params"].get(p, {}).get("bias_z_on") for r in rs if p in r["params"]]
            sdr = [r["params"].get(p, {}).get("sd_ratio") for r in rs if p in r["params"]]
            sdr_ok = [x for x in sdr if x is not None]
            # (A) point-estimate SCATTER (σ_OFF ruler) — note: dominated by the lone large-bias
            # outlier; the leave-one-out below shows whether emucoh genuinely de-biases or just
            # reweights. (4-lens review 2026-06-12: the all-8 Ap "shrink" is a fold7 artifact.)
            base_off = [r["base"] for r in rs if p in r["params"]]
            pe_off_no7 = [v for b, v in zip(base_off, pe_off) if b != "D_lmed7_15"]
            pe_on_no7 = [v for b, v in zip(base_off, pe_on) if b != "D_lmed7_15"]
            d[p] = dict(
                n=len(pe_off),
                scatter_off_fixedruler=_rms(pe_off),
                scatter_on_fixedruler=_rms(pe_on),
                scatter_off_drop_fold7=_rms(pe_off_no7),
                scatter_on_drop_fold7=_rms(pe_on_no7),
                # (CALIBRATION — the legitimate GO basis, Bayesian/meta 2026-06-12): the ensemble
                # std of bias_z. ~1 = nominal; <1 over-covers (conservative); >1 under-covers. emucoh
                # should move this TOWARD/through 1 from below, NEVER push it >1.
                calib_z_std_off=(float(np.std([x for x in bz_off if x is not None], ddof=1)) if len([x for x in bz_off if x is not None]) > 1 else None),
                calib_z_std_on=(float(np.std([x for x in bz_on if x is not None], ddof=1)) if len([x for x in bz_on if x is not None]) > 1 else None),
                # mean signed point-estimate reweight (σ_OFF ruler) ON−OFF — a systematic same-sign
                # shift = GLS reweighting, not de-biasing.
                pe_shift_mean=(float(np.mean([a - b for a, b in zip(pe_on, pe_off)])) if pe_off else None),
                bias_z_rms_off=_rms(bz_off),            # (C) coverage z RMS
                bias_z_rms_on=_rms(bz_on),
                sd_ratio_median=(float(np.median(sdr_ok)) if sdr_ok else None),
                sd_ratio_min=(float(np.min(sdr_ok)) if sdr_ok else None),
                frac_within_1sig_off=(float(np.mean([abs(x) < 1 for x in bz_off])) if bz_off else None),
                frac_within_1sig_on=(float(np.mean([abs(x) < 1 for x in bz_on])) if bz_on else None),
            )
        if g == "ALL":
            out["overall"] = d
        else:
            out["by_family"][g] = d
    return out


def print_report(rows, summ):
    print("\n" + "=" * 96)
    print("PHASE-5a EMUCOH CLOSURE VALIDATION  (EC1=ON vs EC0=OFF, matched-pair, current code)")
    print(f"  ready {summ['n_mocks_ready']}/{summ['n_mocks_total']} mocks   "
          f"sign: bias_z = (post-mean − truth)/σ  [+ = recovered HIGH]")
    print("=" * 96)
    hdr = (f"{'mock':14s} {'fam':3s} {'param':12s} {'truth':>7s} "
           f"{'bz_OFF':>7s} {'bz_ON':>7s} {'pe_ON*':>7s} {'σ_ON/OFF':>8s}  conv")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['base']:14s} {r['family']:3s}  -- not ready (off {r.get('off_chains',0)}/4, "
                  f"on {r.get('on_chains',0)}/4)")
            continue
        bon = r.get("battery_on") or {}
        conv = (f"R̂{bon.get('rhat_max','?')} div{r['on_div']} ESSt{bon.get('ess_tail_min','?')}"
                if bon and "error" not in bon else f"div{r['on_div']}")
        for p in KEY:
            pp = r["params"].get(p)
            if not pp:
                continue
            print(f"{r['base']:14s} {r['family']:3s} {p:12s} {pp['truth']:>7.3f} "
                  f"{pp['bias_z_off']:>7.2f} {pp['bias_z_on']:>7.2f} {pp['pe_on_fixed']:>7.2f} "
                  f"{pp['sd_ratio']:>8.3f}  {conv if p=='Ap' else ''}")
    # per-mock θ-dependence guard (CS lens 2026-06-12): a SHRINK in σ is the live-P pathology
    # signature. Don't trust the median — flag the WORST per-mock sd_ratio against the MC-noise
    # floor (rel-MCSE on a σ-ratio ≈ sqrt(1/(2·ESS_eff)); with ESS≳500 pooled, 2·MCSE≈0.06, so a
    # sd_ratio below ~0.94 is below MC noise and a real concern).
    # Restrict to the KEY cosmology params (Ap/ns): those MUST widen under emucoh, so a σ-shrink there
    # is the θ-dep signature. (HCD-nuisance marginals like α_subDLA can legitimately SHRINK via the
    # degeneracy reweighting — that is not a θ-dependence bug, so they are excluded from this guard.)
    THETA_DEP_FLOOR = 0.94
    worst = min((pp["sd_ratio"] for r in rows if r["status"] == "ok"
                 for p, pp in r["params"].items() if p in KEY and pp.get("sd_ratio") is not None),
                default=None)
    print("\n--- VERDICT (referee gate, calibration basis) ---")
    print("  Sign reminder: the GO basis is CALIBRATION (ensemble std of bias_z → nominal, never >1),")
    print("  NOT de-biasing — the all-8 A_p scatter 'shrink' is the lone fold7 outlier (drop it → grows).")
    if worst is not None:
        ok = worst >= THETA_DEP_FLOOR
        print(f"  (B) θ-dep guard: worst per-mock σ_ON/σ_OFF = {worst:.3f} "
              f"({'≥' if ok else '<'} {THETA_DEP_FLOOR} floor) → "
              f"{'PSD-consistent, no θ-dep bug ✓' if ok else 'BELOW MC NOISE — POSSIBLE θ-DEP BUG ✗'}")
    for g in ("A", "B", "ALL"):
        d = summ["overall"] if g == "ALL" else summ["by_family"].get(g)
        if not d:
            continue
        gl = {"A": "Family A (sim-mean folds 3/4/6/7)", "B": "Family B (matched-lit σ0.15 folds 3/5/7/6)",
              "ALL": "ALL 8 mocks"}[g]
        print(f"\n  {gl}:")
        for p in KEY:
            x = d.get(p, {})
            if not x.get("n"):
                continue
            cof, con = x["calib_z_std_off"], x["calib_z_std_on"]
            cal = ("→nominal, never under-covers ✓" if (con is not None and con <= 1.02)
                   else "UNDER-COVERS (std z>1) ✗" if con is not None else "?")
            so7, sn7 = x["scatter_off_drop_fold7"], x["scatter_on_drop_fold7"]
            print(f"    {p:11s} CALIB ensemble std(z) {cof:.3f}→{con:.3f}  {cal}")
            print(f"    {'':11s} (A) scatter[σ_OFF ruler] all {x['scatter_off_fixedruler']:.3f}→"
                  f"{x['scatter_on_fixedruler']:.3f} | drop-fold7 {so7:.3f}→{sn7:.3f} "
                  f"(reweight mean {x['pe_shift_mean']:+.3f}σ)")
            print(f"    {'':11s} (B) σ_ON/OFF med {x['sd_ratio_median']:.3f} (min {x['sd_ratio_min']:.3f})   "
                  f"(C) |bias z| RMS {x['bias_z_rms_off']:.3f}→{x['bias_z_rms_on']:.3f}   "
                  f"frac|z|<1 {x['frac_within_1sig_off']:.2f}→{x['frac_within_1sig_on']:.2f}")
    # HCD-class miscoverage (Lyα lens 2026-06-12): the triad above is A_p/n_s only, but α_subDLA is
    # badly miscovered in these mocks (subDLA↔LLS↔DLA degeneracy) — orthogonal to emucoh (present OFF
    # too), and emucoh does NOT fix it (clean-class only). Surfaced so it can't be silently omitted.
    aall = summ["overall"].get("alpha_subdla", {})
    if aall.get("n"):
        print(f"\n  ⚠ HCD-class miscoverage (NOT fixed by emucoh — clean-class only; flags the HCD-class term):")
        print(f"    alpha_subdla  |bias z| RMS {aall['bias_z_rms_off']:.3f}→{aall['bias_z_rms_on']:.3f}   "
              f"σ_ON/OFF med {aall['sd_ratio_median']:.3f} (min {aall['sd_ratio_min']:.3f})")


def make_fig(rows, summ):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] matplotlib unavailable ({e}); skipping figure")
        return
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("[fig] no ready mocks; skipping figure")
        return
    labels = [r["base"] for r in ok]
    x = np.arange(len(ok))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for col, p in enumerate(KEY):
        a = ax[0, col]
        bz_off = [r["params"].get(p, {}).get("bias_z_off", np.nan) for r in ok]
        bz_on = [r["params"].get(p, {}).get("bias_z_on", np.nan) for r in ok]
        a.axhspan(-1, 1, color="green", alpha=0.07)
        a.axhline(0, color="k", lw=0.6)
        a.bar(x - 0.2, bz_off, 0.4, label="OFF (EC0)", color="0.6")
        a.bar(x + 0.2, bz_on, 0.4, label="ON (EC1)", color="C3")
        a.set_xticks(x); a.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        a.set_ylabel(f"bias z  (post−truth)/σ  [{p}]")
        a.set_title(f"{p}: coverage z, OFF vs ON")
        a.legend(fontsize=8)
        # σ ratio
        b = ax[1, col]
        sdr = [r["params"].get(p, {}).get("sd_ratio", np.nan) for r in ok]
        b.axhline(1.0, color="k", lw=0.8, ls="--", label="no change")
        b.bar(x, sdr, 0.6, color="C0")
        b.set_xticks(x); b.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        b.set_ylabel(f"σ_ON / σ_OFF  [{p}]")
        lo = min(0.95, np.nanmin(sdr) - 0.02) if len(sdr) else 0.95
        b.set_ylim(lo, max(1.4, np.nanmax(sdr) + 0.05) if len(sdr) else 1.4)
        d = summ["overall"].get(p, {})
        if d.get("n"):
            b.set_title(f"{p}: σ widen (med {d['sd_ratio_median']:.2f}); "
                        f"scatter {d['scatter_off_fixedruler']:.2f}→{d['scatter_on_fixedruler']:.2f}")
        b.legend(fontsize=8)
    fig.suptitle("Phase-5a emucoh closure validation — ON (EC1) vs OFF (EC0), matched pairs\n"
                 "gate: σ widens (PSD), per-fold scatter shrinks, |bias z| → nominal", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(NOTES_FIG, exist_ok=True)
    outp = f"{NOTES_FIG}/emucoh_validation.png"
    fig.savefig(outp, dpi=130)
    print(f"[fig] wrote {outp}")


def main():
    rows = analyze()
    summ = summarize(rows)
    print_report(rows, summ)
    with open(OUT_JSON, "w") as f:
        json.dump({"rows": rows, "summary": summ}, f, indent=2, default=float)
    print(f"\n[json] wrote {OUT_JSON}")
    if "--no-fig" not in sys.argv:
        make_fig(rows, summ)


if __name__ == "__main__":
    main()
