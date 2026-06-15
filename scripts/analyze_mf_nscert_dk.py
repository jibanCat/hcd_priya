#!/usr/bin/env python3
"""Phase-5a MF n_s HIGH-K CERTIFICATION verdict — genuine HF-LOSO Test B on the JOINT DESI+KS legs.

Reads the committed checkpoints/stepA/HFLOSO_DK*.npz (5 HR sims × 4 chains: the production MF
forward + KS small-scale leg) and prints the per-fold n_s / A_p bias_z plus the cert verdict:

  GATE: per-fold n_s |bias_z| < 1 (ideally < 0.2σ).  bias_z = (post_mean - truth)/sigma_post.

If the gate holds, the PRODUCTION MF controls the high-k n_s tilt with KS in the joint; if it
fails, a θ-resolved res_corr / wider σ is needed before the KS real fit. For context the script
also prints the DESI-only Test B (HFLOSO*) so the KS leg's effect on the tilt is visible.

Emits a figure (DESI+KS vs DESI-only n_s/A_p closure) to the NOTES figure tree.

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
    dict(ns, Ap, n_chains, n_draws, rhat_proxy) or None if no checkpoints exist."""
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


def main():
    rows = []
    for tag in sorted(SIMS):
        ns = SIMS[tag]
        dk = load_bias(f"HFLOSO_DK{tag}")
        desi = load_bias(f"HFLOSO{tag}")
        rows.append((ns, tag, dk, desi))

    have_dk = [r for r in rows if r[2] is not None]
    print("\n=== Phase-5a MF n_s HIGH-K CERT — genuine HF-LOSO Test B on DESI+KS ===")
    print("    bias_z = (post_mean - truth)/sigma_post   [+ => posterior OVERestimates]")
    print(f"    GATE: per-fold n_s |bias_z| < 1 (ideal < 0.2)\n")
    hdr = f"  {'HR n_s':>7s} | {'DESI+KS n_s':>11s} {'A_p':>7s} {'chains':>6s} {'div':>4s} | {'DESI-only n_s':>13s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    worst = 0.0
    for ns, tag, dk, desi in rows:
        if dk is None:
            print(f"  {ns:>7.3f} | {'(pending)':>11s}")
            continue
        flag = "" if abs(dk["ns"]) < 1 else "  <-- OUT-OF-GATE"
        worst = max(worst, abs(dk["ns"]))
        dstr = f"{desi['ns']:+.2f}" if desi else "n/a"
        print(f"  {ns:>7.3f} | {dk['ns']:>+11.2f} {dk['Ap']:>+7.2f} {dk['n_chains']:>6d} "
              f"{dk['div']:>4d} | {dstr:>13s}{flag}")

    n_have = len(have_dk)
    if n_have == 0:
        print("\n  No HFLOSO_DK checkpoints yet — run scripts/batch_mf_nscert_dk.sh first.")
        return
    all_in_gate = all(abs(r[2]["ns"]) < 1 for r in have_dk)
    ap_in_gate = all(abs(r[2]["Ap"]) < 1 for r in have_dk)
    print(f"\n  worst |n_s bias_z| (DESI+KS) = {worst:.2f}σ  over {n_have}/5 folds")
    print(f"  VERDICT n_s: {'PASS (|z|<1 all folds)' if all_in_gate else 'FAIL (>=1 fold out of gate)'}")
    print(f"  VERDICT A_p: {'PASS' if ap_in_gate else 'FAIL'}")
    if all_in_gate:
        print("  => the PRODUCTION MF controls the high-k n_s tilt on DESI+KS. No θ-resolved "
              "res_corr / wider σ needed.")
    else:
        print("  => the production MF does NOT control the high-k n_s tilt on DESI+KS; a "
              "θ-resolved res_corr or wider σ is needed before the KS real fit.")

    # ---- figure (only if all 5 folds are in) ----
    if n_have == 5:
        os.makedirs(NOTES_FIG, exist_ok=True)
        ns_x = np.array([r[0] for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, key, lab in zip(axes, ("ns", "Ap"), (r"$n_s$", r"$A_p$")):
            ax.axhspan(-1, 1, color="0.85", zorder=0, label="|z|<1 gate")
            ax.axhline(0, color="k", lw=0.8)
            dk_y = [r[2][key] for r in rows]
            ax.plot(ns_x, dk_y, "o-", color="C0", ms=8, lw=2, label="Test B DESI+KS (production MF)")
            desi_y = [r[3][key] if r[3] else np.nan for r in rows]
            ax.plot(ns_x, desi_y, "s--", color="C3", ms=7, lw=1.4, label="Test B DESI-only")
            for x, y in zip(ns_x, dk_y):
                ax.annotate(f"{y:+.1f}σ", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
            ax.set_xlabel("HR-sim $n_s$")
            ax.set_ylabel(rf"$(\hat{{\theta}}-\theta^{{\rm true}})/\sigma$  [{lab}]")
            ax.set_ylim(-3.4, 3.4)
            ax.legend(fontsize=8, loc="upper left")
        verdict = "PASS" if all_in_gate else "FAIL"
        axes[0].set_title(f"(a) $n_s$ closure — MF cert {verdict} (worst {worst:.1f}σ)")
        axes[1].set_title("(b) $A_p$ closure")
        fig.suptitle("Phase-5a MF n_s HIGH-K cert — genuine HF-LOSO Test B THROUGH the production "
                     "MF forward, JOINT DESI+KS (KS floor ON)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = f"{NOTES_FIG}/mf_nscert_dk_closure.png"
        fig.savefig(out, dpi=130)
        print(f"\n  wrote figure -> {out}")


if __name__ == "__main__":
    main()
