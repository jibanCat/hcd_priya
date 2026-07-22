#!/usr/bin/env python3
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# headroom-experiment aggregator; not the deployed emulator.
"""Aggregate retrain_headroom_pilot JSON outputs -> headroom verdict table + figure.

Reads all checkpoints/retrain/<tag>_<variant>_fold<f>_seed<s>.json, builds a per-class
held-out-error table (variant x class) RELATIVE to the seed-matched 'baseline' variant on
the same fold, and a per-param Fisher-sensitivity ratio. Emits a verdict: which lever (if
any) materially DROPS the held-out per-class error vs baseline, by how much.

Figure -> notes repo figures/analysis/04_emulator/retrain_headroom_<tag>.png
NPZ    -> same dir, retrain_headroom_<tag>.npz

ENV: PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np

CLASSES = ["clean", "LLS", "subDLA", "DLA"]
PARAMS = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]
NOTES_FIG = "/home/mfho/hcd_priya_notes/figures/analysis/04_emulator"


def load_all(tag, indir):
    recs = {}
    for fp in sorted(glob.glob(f"{indir}/{tag}_*.json")):
        d = json.loads(Path(fp).read_text())
        key = (d["variant"], d["fold"], d["seed"])
        recs[key] = d
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--indir", default="/home/mfho/hcd_priya/checkpoints/retrain")
    args = ap.parse_args()
    recs = load_all(args.tag, args.indir)
    if not recs:
        print("no records found"); return
    folds = sorted(set(k[1] for k in recs))
    variants = sorted(set(k[0] for k in recs))
    print(f"loaded {len(recs)} records: variants={variants} folds={folds}\n")

    # metric we headline: per-class held-out rms(k<0.06) and rms_all, averaged over folds
    # relative to the per-fold 'baseline' on the SAME fold/seed.
    metric = "rms_k06"
    print(f"# ===== HELD-OUT per-class {metric} (%), and DELTA vs seed-matched baseline =====")
    table = {}
    for v in variants:
        per_class = {c: [] for c in CLASSES}
        per_class_delta = {c: [] for c in CLASSES}
        n_ep = []; t_s = []
        for f in folds:
            for s in set(k[2] for k in recs if k[0] == v and k[1] == f):
                if (v, f, s) not in recs:
                    continue
                r = recs[(v, f, s)]
                base = recs.get(("baseline", f, s))
                n_ep.append(r.get("n_epochs", np.nan)); t_s.append(r.get("train_s", np.nan))
                for c in CLASSES:
                    val = r["held"][c][metric]
                    per_class[c].append(val)
                    if base is not None and v != "baseline":
                        per_class_delta[c].append(val - base["held"][c][metric])
        table[v] = dict(
            mean={c: float(np.mean(per_class[c])) if per_class[c] else np.nan for c in CLASSES},
            delta={c: float(np.mean(per_class_delta[c])) if per_class_delta[c] else np.nan for c in CLASSES},
            n_ep=float(np.mean(n_ep)) if n_ep else np.nan,
            t_s=float(np.mean(t_s)) if t_s else np.nan,
        )
    # print
    hdr = f"{'variant':14s} " + " ".join(f"{c:>9s}" for c in CLASSES) + "   epochs  s"
    print(hdr); print("-" * len(hdr))
    for v in ["baseline"] + [x for x in variants if x != "baseline"]:
        if v not in table:
            continue
        t = table[v]
        row = f"{v:14s} "
        for c in CLASSES:
            if v == "baseline":
                row += f"{100*t['mean'][c]:9.3f}"
            else:
                d = t["delta"][c]
                sign = "+" if d >= 0 else ""
                row += f"{sign}{100*d:8.3f}"   # delta in %, +=worse
        row += f"   {t['n_ep']:5.0f} {t['t_s']:5.0f}"
        print(row)
    print("\n(baseline row = absolute rms(k<0.06) %; other rows = DELTA vs baseline; "
          "NEGATIVE = better/headroom, POSITIVE = worse)")

    # ----- per-param Fisher-sensitivity ratio vs baseline -----
    print("\n# ===== per-param Fisher-sensitivity RATIO vs baseline (>1 = more responsive) =====")
    for v in [x for x in variants if x != "baseline"]:
        ratios = {p: [] for p in PARAMS}
        for f in folds:
            for s in set(k[2] for k in recs if k[0] == v and k[1] == f):
                r = recs.get((v, f, s)); base = recs.get(("baseline", f, s))
                if r is None or base is None or r.get("fisher") is None or base.get("fisher") is None:
                    continue
                for p in PARAMS:
                    b = base["fisher"].get(p, np.nan)
                    if b and np.isfinite(b) and b != 0:
                        ratios[p].append(r["fisher"][p] / b)
        rr = {p: (float(np.mean(ratios[p])) if ratios[p] else np.nan) for p in PARAMS}
        s = " ".join(f"{p}={rr[p]:.2f}" for p in PARAMS if np.isfinite(rr[p]))
        print(f"  {v:14s}: {s}")

    # ----- verdict -----
    print("\n# ===== VERDICT =====")
    THRESH = 0.001  # 0.1% absolute rms improvement to count as 'material'
    headroom = []
    for v in [x for x in variants if x != "baseline"]:
        t = table[v]
        best_c = min(CLASSES, key=lambda c: t["delta"][c] if np.isfinite(t["delta"][c]) else 0)
        best_d = t["delta"][best_c]
        improved = [c for c in CLASSES if np.isfinite(t["delta"][c]) and t["delta"][c] < -THRESH]
        worsened = [c for c in CLASSES if np.isfinite(t["delta"][c]) and t["delta"][c] > THRESH]
        verdict = "HEADROOM" if improved and not worsened else (
                  "MIXED" if improved else "no improvement")
        print(f"  {v:14s}: {verdict:14s} best Δ {100*best_d:+.3f}% ({best_c})  "
              f"improved={improved} worsened={worsened}")
        if improved and not worsened:
            headroom.append((v, best_d, best_c))
    if headroom:
        headroom.sort(key=lambda x: x[1])
        print(f"\n  => HEADROOM FOUND. Best lever: {headroom[0][0]} "
              f"({100*headroom[0][1]:+.3f}% on {headroom[0][2]})")
    else:
        print("\n  => NO MATERIAL HEADROOM: no lever improves a class without worsening another.")

    # ----- figure -----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vlist = [x for x in variants if x != "baseline"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(vlist)); w = 0.2
    for ic, c in enumerate(CLASSES):
        deltas = [100 * table[v]["delta"][c] for v in vlist]
        ax.bar(x + (ic - 1.5) * w, deltas, w, label=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-0.1, color="g", ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(vlist, rotation=30, ha="right")
    ax.set_ylabel("Δ held-out rms(k<0.06) vs baseline  [%]\n(negative = headroom)")
    ax.set_title(f"Emulator retraining headroom ({args.tag}): per-class held-out error change vs deployed recipe\n"
                 f"(LOSO folds {folds}; negative bar = a lever that REDUCES held-out error)")
    ax.legend(title="P1D class"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    Path(NOTES_FIG).mkdir(parents=True, exist_ok=True)
    out = f"{NOTES_FIG}/retrain_headroom_{args.tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n[fig] {out}")
    np.savez(f"{NOTES_FIG}/retrain_headroom_{args.tag}.npz",
             variants=np.array(vlist), classes=np.array(CLASSES), folds=np.array(folds),
             baseline_rms=np.array([[table["baseline"]["mean"][c]] for c in CLASSES]).ravel(),
             delta=np.array([[table[v]["delta"][c] for c in CLASSES] for v in vlist]))
    print(f"[npz] {NOTES_FIG}/retrain_headroom_{args.tag}.npz")


if __name__ == "__main__":
    main()
