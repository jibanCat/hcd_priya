#!/usr/bin/env python3
"""Phase-5a MF n_s HIGH-K CERTIFICATION verdict — genuine HF-LOSO Test B, PER-SURVEY headline.

The real fits are SEPARATE per survey, so the cert is reported PER SURVEY:

  HEADLINE  (each with its own PASS/FAIL gate):
    DESI-only  (HFLOSO{tag})   — checkpoints/stepA/HFLOSO{tag}_c*.npz  (DESI leg, no MF floor)
    KS-only    (HFLOSO_KS{tag}) — checkpoints/stepA/HFLOSO_KS{tag}_c*.npz (KS leg, mf_floor_on=True)
  CONTEXT (secondary):
    joint DESI+KS (HFLOSO_DK{tag}) — the production combination, shown for reference only.

Each headline column is the per-fold n_s / A_p bias_z (5 HR sims × 4 chains) through the production
MF forward (LF emu × MF correction fit EXCLUDING that HR sim), truth = that HR sim's REAL measured
P1D.  GATE: per-fold n_s |bias_z| < 1 (ideally < 0.2σ).  bias_z = (post_mean - truth)/sigma_post.

If a survey's gate holds, the PRODUCTION MF controls its high-k n_s tilt; if it fails, a θ-resolved
res_corr / wider σ is needed before that survey's real fit. Restartable / partial-safe: any survey
whose checkpoints don't exist yet prints "(pending)" and does NOT crash.

Emits a per-survey figure (DESI-only vs KS-only headline panels; joint DESI+KS overlaid as context)
to the NOTES figure tree.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
       /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/analyze_mf_nscert_dk.py
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/stepA"
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"

SIMS = {  # mock-id stem (n_s*1000) -> physical n_s
    859: 0.859, 885: 0.885, 909: 0.909, 972: 0.972, 979: 0.979}


def load_bias(mock_id):
    """(post_mean - truth)/sigma_post for n_s & A_p, pooled over a mock's chains. Returns
    dict(ns, Ap, n_chains, n_draws, div) or None if no checkpoints exist (partial-safe)."""
    fs = sorted(glob.glob(f"{CKPT}/{mock_id}_c*.npz"))
    if not fs:
        return None
    Z = [np.load(f, allow_pickle=True) for f in fs]
    names = [str(x) for x in Z[0]["names"]]
    truth = Z[0]["truth_vec"]
    draws = np.concatenate([z["packed"] for z in Z], axis=0)
    out = {"n_chains": len(fs), "n_draws": int(draws.shape[0]),
           "div": int(sum(int(z["divergences"]) for z in Z))}
    for p in ("ns", "Ap"):
        i = names.index(p)
        m = float(draws[:, i].mean()); s = float(draws[:, i].std())
        out[p] = (m - truth[i]) / s if s > 0 else float("nan")
    return out


def _survey_verdict(rows, idx):
    """Per-survey gate over the folds that HAVE checkpoints. rows = list of (ns, tag, desi, ks, dk);
    idx selects the column (2=DESI-only, 3=KS-only, 4=joint DESI+KS). Returns
    (have_n, worst_ns, ns_pass, ap_pass) — verdict over the folds present (partial-safe)."""
    have = [r for r in rows if r[idx] is not None]
    if not have:
        return 0, 0.0, None, None
    worst = max(abs(r[idx]["ns"]) for r in have)
    ns_pass = all(abs(r[idx]["ns"]) < 1 for r in have)
    ap_pass = all(abs(r[idx]["Ap"]) < 1 for r in have)
    return len(have), worst, ns_pass, ap_pass


def main():
    rows = []
    for tag in sorted(SIMS):
        ns = SIMS[tag]
        desi = load_bias(f"HFLOSO{tag}")        # DESI-only (headline)
        ks = load_bias(f"HFLOSO_KS{tag}")       # KS-only (headline)
        dk = load_bias(f"HFLOSO_DK{tag}")       # joint DESI+KS (context)
        rows.append((ns, tag, desi, ks, dk))

    print("\n=== Phase-5a MF n_s HIGH-K CERT — genuine HF-LOSO Test B, PER-SURVEY ===")
    print("    bias_z = (post_mean - truth)/sigma_post   [+ => posterior OVERestimates]")
    print("    GATE (per survey): per-fold n_s |bias_z| < 1 (ideal < 0.2)")
    print("    HEADLINE = DESI-only + KS-only (the SEPARATE real fits); DESI+KS = context\n")
    hdr = (f"  {'HR n_s':>7s} | {'DESI n_s':>9s} {'Ap':>6s} | {'KS n_s':>9s} {'Ap':>6s} | "
           f"{'DK n_s':>7s} {'div(D/K/DK)':>12s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for ns, tag, desi, ks, dk in rows:
        def col(b, key, w):
            return f"{b[key]:>+{w}.2f}" if b is not None else f"{'(pend)':>{w}s}"
        d_flag = "" if (desi is None or abs(desi["ns"]) < 1) else " D!"
        k_flag = "" if (ks is None or abs(ks["ns"]) < 1) else " K!"
        divs = "/".join(str(b["div"]) if b is not None else "-" for b in (desi, ks, dk))
        print(f"  {ns:>7.3f} | {col(desi,'ns',9)} {col(desi,'Ap',6)} | "
              f"{col(ks,'ns',9)} {col(ks,'Ap',6)} | {col(dk,'ns',7)} {divs:>12s}{d_flag}{k_flag}")

    # --- per-survey verdicts (headline) ---
    d_n, d_worst, d_ns_ok, d_ap_ok = _survey_verdict(rows, 2)
    k_n, k_worst, k_ns_ok, k_ap_ok = _survey_verdict(rows, 3)
    dk_n, dk_worst, dk_ns_ok, dk_ap_ok = _survey_verdict(rows, 4)

    def report(name, n_have, worst, ns_ok, ap_ok, headline=True):
        tag = "HEADLINE" if headline else "context "
        if n_have == 0:
            print(f"\n  [{tag}] {name}: (pending) — no checkpoints yet.")
            return
        print(f"\n  [{tag}] {name}: worst |n_s bias_z| = {worst:.2f}σ over {n_have}/5 folds")
        print(f"           VERDICT n_s: {'PASS (|z|<1 all folds)' if ns_ok else 'FAIL (>=1 fold out of gate)'}"
              f"   |   A_p: {'PASS' if ap_ok else 'FAIL'}")

    report("DESI-only", d_n, d_worst, d_ns_ok, d_ap_ok, headline=True)
    report("KS-only", k_n, k_worst, k_ns_ok, k_ap_ok, headline=True)
    report("joint DESI+KS", dk_n, dk_worst, dk_ns_ok, dk_ap_ok, headline=False)

    # overall per-survey takeaway
    print("\n  --- per-survey takeaway ---")
    for name, n_have, ns_ok in (("DESI", d_n, d_ns_ok), ("KS", k_n, k_ns_ok)):
        if n_have == 0:
            print(f"    {name}: pending.")
        elif ns_ok:
            print(f"    {name}: PASS — the production MF controls the high-k n_s tilt; "
                  f"no θ-resolved res_corr / wider σ needed for the {name} real fit.")
        else:
            print(f"    {name}: FAIL — the production MF does NOT control the high-k n_s tilt; "
                  f"a θ-resolved res_corr or wider σ is needed before the {name} real fit.")

    # ---- figure: per-survey HEADLINE (DESI-only vs KS-only) + DESI+KS as context ----
    if d_n == 5 or k_n == 5:
        os.makedirs(NOTES_FIG, exist_ok=True)
        ns_x = np.array([r[0] for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        for ax, key, lab in zip(axes, ("ns", "Ap"), (r"$n_s$", r"$A_p$")):
            ax.axhspan(-1, 1, color="0.85", zorder=0, label="|z|<1 gate")
            ax.axhline(0, color="k", lw=0.8)
            # HEADLINE: DESI-only (C3) + KS-only (C0)
            desi_y = [r[2][key] if r[2] else np.nan for r in rows]
            ks_y = [r[3][key] if r[3] else np.nan for r in rows]
            ax.plot(ns_x, desi_y, "s-", color="C3", ms=8, lw=2, label="DESI-only (headline)")
            ax.plot(ns_x, ks_y, "o-", color="C0", ms=8, lw=2, label="KS-only (headline)")
            # CONTEXT: joint DESI+KS (grey dashed)
            dk_y = [r[4][key] if r[4] else np.nan for r in rows]
            ax.plot(ns_x, dk_y, "^--", color="0.5", ms=6, lw=1.2, label="DESI+KS (context)")
            for x, y in zip(ns_x, desi_y):
                if np.isfinite(y):
                    ax.annotate(f"{y:+.1f}", (x, y), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=7, color="C3")
            for x, y in zip(ns_x, ks_y):
                if np.isfinite(y):
                    ax.annotate(f"{y:+.1f}", (x, y), textcoords="offset points",
                                xytext=(0, -12), ha="center", fontsize=7, color="C0")
            ax.set_xlabel("HR-sim $n_s$")
            ax.set_ylabel(rf"$(\hat{{\theta}}-\theta^{{\rm true}})/\sigma$  [{lab}]")
            ax.set_ylim(-3.6, 3.6)
            ax.legend(fontsize=8, loc="upper left")
        d_v = ("PASS" if d_ns_ok else "FAIL") if d_n else "pend"
        k_v = ("PASS" if k_ns_ok else "FAIL") if k_n else "pend"
        axes[0].set_title(f"(a) $n_s$ closure — DESI {d_v} (worst {d_worst:.1f}σ) | "
                          f"KS {k_v} (worst {k_worst:.1f}σ)")
        axes[1].set_title("(b) $A_p$ closure")
        fig.suptitle("Phase-5a MF n_s HIGH-K cert — genuine HF-LOSO Test B through the production "
                     "MF forward, PER-SURVEY (DESI-only / KS-only headline)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = f"{NOTES_FIG}/mf_nscert_persurvey.png"
        fig.savefig(out, dpi=130)
        print(f"\n  wrote figure -> {out}")
    else:
        print("\n  (figure pending — needs a full 5-fold DESI-only OR KS-only set)")


if __name__ == "__main__":
    main()
