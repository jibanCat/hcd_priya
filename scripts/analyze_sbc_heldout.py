#!/usr/bin/env python3
"""Analyze the HELD-OUT-sim SBC pilot (leg_a=False) — the HONEST n_s certifier.

Unlike the self-draw SBC (_sbc_prod_summary_analysis.py, leg_a=True, which CANCELS
emulator misspecification by construction), this reads the held-out run whose mocks
CONTAIN emulator error. It is intentionally light: text summary FIRST (the n_s gate
read), works incrementally on whatever has landed (no hard 48-mock assert), and does
NOT clobber the self-draw figure set.

Gate (per the handoff resume recipe):
  n_s pull-mean within +/-0.3 sigma  AND  pull-std <= 1.1  across the held-out mocks.
Standing rule (feedback-report-tau0-dtau0-bias): ALWAYS report tau0 amplitude + dtau0
bias alongside any cosmo-bias report.

Usage:
  PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
    scripts/analyze_sbc_heldout.py [SRC_DIR] [OUT_PREFIX]

  SRC_DIR     default /scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout
  OUT_PREFIX  default sbc_heldout  (figures -> notes 05_likelihood/<prefix>_*.png)
"""
import pickle, glob, json, os, sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/scratch/cavestru_root/cavestru1/mfho/prod_sbc_heldout"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "sbc_heldout"
OUT = "/home/mfho/hcd_priya_notes/figures/analysis/05_likelihood"
os.makedirs(OUT, exist_ok=True)

# tau0 ladder z-grid: closure_legb keeps rows with 2.2 <= z_grid <= 4.6 on the 0.2
# PRIYA/DESI spacing -> 13 rungs z = 2.2 + 0.2*i.  (NB the self-draw analyzer's
# 2.0..4.4 is WRONG for this ladder; we assert n_tau0==13 to catch a grid change.)
Z_TAU0 = np.array([2.2 + 0.2 * i for i in range(13)])

files = sorted(glob.glob(os.path.join(SRC, "mock_*.pkl")))
if not files:
    print(f"[held-out] nothing landed yet in {SRC}")
    sys.exit(0)

mocks, names = [], None
for f in files:
    try:
        d = pickle.load(open(f, "rb"))
    except (EOFError, pickle.UnpicklingError):
        print(f"[held-out] SKIP partial/corrupt {os.path.basename(f)} (still writing?)")
        continue
    if names is None:
        names = list(d["names"])
    mocks.append((os.path.basename(f), d))

M = len(mocks)
if M == 0:
    print("[held-out] all landed pkls are partial — re-run shortly")
    sys.exit(0)
P = len(names)
j_ns = names.index("ns")
j_ap = names.index("Ap")
TAU0_IDX = [names.index(f"tau0_z{z}") for z in range(13)]
assert len(TAU0_IDX) == 13, f"expected 13 tau0 rungs, got {len(TAU0_IDX)}"
lx = np.log(1.0 + Z_TAU0)
lx_c = lx - lx.mean()                       # centered -> intercept = ln(tau0) at z-bar


def tau0_amp_slope(tau0_ladder):
    """Regress ln(tau0(z)) on centered ln(1+z): return (amp=exp(intercept@zbar), slope=dtau0)."""
    y = np.log(np.clip(tau0_ladder, 1e-8, None))
    slope = np.sum(lx_c * (y - y.mean())) / np.sum(lx_c ** 2)
    intercept = y.mean()                     # value of ln(tau0) at z-bar (lx centered)
    return np.exp(intercept), slope


print(f"[held-out] SRC={SRC}")
print(f"[held-out] loaded {M} landed mocks ({P} params); tau0 ladder z={Z_TAU0[0]:.1f}..{Z_TAU0[-1]:.1f}\n")

# ---- per-mock pulls for the headline params + tau0 amp/dtau0 ----
rows = []
ns_pull, ap_pull = [], []
amp_pull, dtau_pull = [], []
ll_rank_frac = []
ndiv_all, L_all = [], []
subdla_pull, lls_pull = [], []

for fname, d in mocks:
    dr = np.asarray(d["draws"])              # (L, P)
    t = np.asarray(d["truth_vec"])           # (P,)
    L = dr.shape[0]
    L_all.append(L)
    ndiv_all.append(int(d.get("n_div", -1)))

    def pull(j):
        mu, sd = dr[:, j].mean(), dr[:, j].std(ddof=1)
        return (mu - t[j]) / sd if sd > 0 else np.nan

    pn, pa = pull(j_ns), pull(j_ap)
    ns_pull.append(pn); ap_pull.append(pa)
    subdla_pull.append(pull(names.index("alpha_subdla")))
    lls_pull.append(pull(names.index("alpha_lls")))

    # tau0 amplitude + dtau0 (regress each draw's ladder; truth ladder for the truth)
    amps = np.empty(L); slopes = np.empty(L)
    for i in range(L):
        amps[i], slopes[i] = tau0_amp_slope(dr[i, TAU0_IDX])
    amp_t, slope_t = tau0_amp_slope(t[TAU0_IDX])
    pamp = (amps.mean() - amp_t) / amps.std(ddof=1) if amps.std() > 0 else np.nan
    pdt = (slopes.mean() - slope_t) / slopes.std(ddof=1) if slopes.std() > 0 else np.nan
    amp_pull.append(pamp); dtau_pull.append(pdt)

    # loglik-rank (now correct z-resolved per the fixed _loglik_of_draws)
    if "ll_true" in d and "ll_draws" in d:
        lld = np.asarray(d["ll_draws"]); llt = float(d["ll_true"])
        ll_rank_frac.append(float(np.mean(lld < llt)))

    rows.append((fname, L, ndiv_all[-1], pn, pa, pamp, pdt))

ns_pull = np.array(ns_pull); ap_pull = np.array(ap_pull)
amp_pull = np.array(amp_pull); dtau_pull = np.array(dtau_pull)
subdla_pull = np.array(subdla_pull); lls_pull = np.array(lls_pull)


def summ(arr, name, gate_mean=0.3, gate_std=1.1):
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return f"{name:10s}  (no finite pulls yet)"
    m, s = a.mean(), (a.std(ddof=1) if len(a) > 1 else np.nan)
    sem = s / np.sqrt(len(a)) if len(a) > 1 else np.nan
    tag = ""
    if name == "ns":
        ok_m = abs(m) <= gate_mean if len(a) >= 4 else None
        ok_s = (s <= gate_std) if (len(a) > 1) else None
        tag = f"   GATE[|mean|<={gate_mean} & std<={gate_std}]: " + (
            "PASS" if (ok_m and ok_s) else ("FAIL" if (ok_m is not None and ok_s is not None) else "underpowered(N<4)"))
    return f"{name:10s}  mean={m:+.3f}  std={s:.3f}  sem={sem:.3f}  N={len(a)}{tag}"


print("=== per-mock pulls (held-out / HONEST instrument) ===")
print(f"{'mock':16s} {'L':>4s} {'div':>4s} {'ns':>8s} {'Ap':>8s} {'tau0amp':>8s} {'dtau0':>8s}")
for fname, L, nd, pn, pa, pamp, pdt in rows:
    print(f"{fname:16s} {L:4d} {nd:4d} {pn:+8.3f} {pa:+8.3f} {pamp:+8.3f} {pdt:+8.3f}")

print("\n=== headline pull summary ===")
print(summ(ns_pull, "ns"))
print(summ(ap_pull, "Ap"))
print(summ(amp_pull, "tau0amp"))
print(summ(dtau_pull, "dtau0"))
print(summ(subdla_pull, "alpha_subdla"))
print(summ(lls_pull, "alpha_lls"))

if ll_rank_frac:
    llr = np.array(ll_rank_frac)
    print(f"\nloglik-rank: mean={llr.mean():.3f} (ideal 0.5)  N={len(llr)}  [z-resolved, fixed path]")

print(f"\nrun health: divergences total={sum(max(0,x) for x in ndiv_all)}  "
      f"L median={int(np.median(L_all))}  range=[{min(L_all)},{max(L_all)}]")

# ---- concise JSON for the handoff/doc ----
def stat(arr):
    a = arr[np.isfinite(arr)]
    return {"mean": float(a.mean()) if len(a) else None,
            "std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "n": int(len(a))}

out_json = os.path.join(OUT, f"{PREFIX}_pull_summary.json")
json.dump({
    "src": SRC, "n_mocks": M, "n_params": P,
    "z_tau0": Z_TAU0.tolist(),
    "ns": stat(ns_pull), "Ap": stat(ap_pull),
    "tau0_amp": stat(amp_pull), "dtau0": stat(dtau_pull),
    "alpha_subdla": stat(subdla_pull), "alpha_lls": stat(lls_pull),
    "ll_rank_frac_mean": float(np.mean(ll_rank_frac)) if ll_rank_frac else None,
    "div_total": int(sum(max(0, x) for x in ndiv_all)),
    "per_mock": [{"mock": r[0], "L": r[1], "div": r[2], "ns": r[3],
                  "Ap": r[4], "tau0amp": r[5], "dtau0": r[6]} for r in rows],
}, open(out_json, "w"), indent=2)
print(f"\nwrote {out_json}")

# ---- figure (only if matplotlib available; headline n_s + tau0 amp/dtau0) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sst
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, arr, ttl in ((axes[0], ns_pull, r"$n_s$"),
                         (axes[1], amp_pull, r"$\tau_0$ amp"),
                         (axes[2], dtau_pull, r"$d\tau_0$")):
        a = arr[np.isfinite(arr)]
        ax.hist(a, bins=np.linspace(-4, 4, 17), density=True, color="#EE6677",
                alpha=0.7, edgecolor="white")
        xx = np.linspace(-4, 4, 200)
        ax.plot(xx, sst.norm.pdf(xx), "k-", lw=1.4, label=r"$\mathcal{N}(0,1)$")
        if len(a):
            ax.axvline(a.mean(), color="crimson", lw=1.8,
                       label=f"mean {a.mean():+.2f}, std {a.std(ddof=1):.2f}" if len(a) > 1 else f"mean {a.mean():+.2f}")
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_title(ttl); ax.set_xlabel("pull"); ax.legend(fontsize=8)
    fig.suptitle(f"Held-out SBC (HONEST, leg_a=False) — N={M} landed  |  gate: $n_s$ |mean|<=0.3 & std<=1.1",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figp = os.path.join(OUT, f"{PREFIX}_headline_pull.png")
    fig.savefig(figp, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figp}")
except Exception as e:
    print(f"(figure skipped: {e})")
