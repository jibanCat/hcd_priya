# %% [markdown]
# # Unblinding the eBOSS + KS real-data cosmology fits
#
# **Run this AFTER the blind real-data fits finish** (`scripts/run_real_fit.py` /
# `scripts/batch_real_fit.sh`). It loads the eBOSS and KS chains, reveals the hidden
# A_p / n_s offset, and reports the **UNBLIND** posteriors for these two *public* surveys.
#
# > **DESI stays blind.** DESI cosmology is private (`results_local/`, gitignored) and has its
# > own separate, local unblinding ritual. This notebook deliberately never touches DESI.
#
# ---
# ## 1. What "blinding" means here
#
# The real-data fit is **parameter-blind** on exactly the two cosmology numbers the measurement
# reports: the forest power amplitude **A_p** (column `Ap`) and the spectral index **n_s**
# (column `ns`). Blinding is a *hidden additive offset*:
#
# $$\theta_\text{shown} \;=\; \theta_\text{inferred} \;+\; \delta,\qquad
#   \delta_{A_p,\,n_s}\sim\text{Uniform}(-3\sigma_\text{prior},\,+3\sigma_\text{prior}).$$
#
# The offset $\delta$ is **derived deterministically** from a SHA256 seed that is committed to
# `blind.lock` (the offset *value* is never stored). Every downstream summary you have looked
# at so far was computed on $\theta_\text{shown}$, so the analysis literally could not see the
# real A_p / n_s. **Unblinding is the single subtraction** $\theta_\text{inferred}=\theta_\text{shown}-\delta$.
#
# Only `Ap` and `ns` move. All nuisances ($\tau_0$, $\alpha_\text{HCD}$, $a_\text{SiIII}$) and
# all sampler-health diagnostics (R-hat, ESS, divergences) were always fully visible.
#
# **This notebook does not run a fit.** It only loads existing chains and applies the offset.

# %%
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make the repo importable regardless of where Jupyter was launched.
REPO = "/home/mfho/hcd_priya"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from hcd_analysis.emulator import blinding as BL  # read_blind_lock / offset_from_lock / unblind

# ---- configuration (edit only if your paths differ) -----------------------------------------
RESULTS_DIR = os.path.join(REPO, "results", "real_fit")   # eBOSS / KS chains (committable)
BLIND_LOCK = os.path.join(REPO, "blind.lock")             # the committed blind SEED
SURVEYS = {                                               # display name -> chain root
    "eBOSS": "real_eboss",
    "KS":    "real_ks",
}
BLIND_COLS = list(BL.BLIND_PARAMS)                        # ('ns', 'Ap')

print("Repo            :", REPO)
print("Results dir     :", RESULTS_DIR, "(exists:", os.path.isdir(RESULTS_DIR), ")")
print("blind.lock      :", BLIND_LOCK, "(exists:", os.path.isfile(BLIND_LOCK), ")")
print("Blinded columns :", BLIND_COLS)


# %% [markdown]
# ## 2. Load the chains (graceful if a chain is missing)
#
# Each survey writes, in `results/real_fit/`:
#
# | file | contents |
# |---|---|
# | `<root>.1.txt` … `<root>.4.txt` | per-chain rows: `weight  minuslogpost  <physical θ9…>  <τ₀…>  α_lls  α_subDLA  α_dla  [a_SiIII]` — **A_p / n_s are BLINDED** |
# | `<root>.paramnames` | `name<TAB>latex` per column (matches the θ-block order, no weight/loglike) |
# | `<root>.yaml` / `<root>.health.json` | sampler health (R-hat, ESS, divergences) — **unblinded** |
#
# We prefer GetDist (`getdist.loadMCSamples`) for the triangle plot, but GetDist is **optional**:
# if it is not installed we fall back to a pure-numpy loader + a matplotlib corner plot, so the
# notebook still runs end-to-end. Either way the underlying samples are identical.

# %%
try:
    from getdist import MCSamples, loadMCSamples  # noqa: F401
    import getdist.plots as gdplt
    HAVE_GETDIST = True
except Exception as exc:  # pragma: no cover - environment dependent
    HAVE_GETDIST = False
    print("getdist not available (", type(exc).__name__, ") -> using the numpy fallback loader",
          "and a matplotlib corner plot. Samples/statistics are unaffected.")


def read_paramnames(root):
    """[(name, latex), ...] in column order from <root>.paramnames (excludes weight/loglike)."""
    path = root + ".paramnames"
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            name = parts[0].strip()
            latex = parts[1].strip() if len(parts) > 1 else name
            out.append((name, latex))
    return out


def load_chain_arrays(root, n_chains_max=8):
    """Load <root>.{c}.txt for c=1.. and return (weights, loglike, samples(N,P), names, labels).

    samples columns are in paramnames order (the θ-block); weight & minuslogpost are split off.
    Returns None if NO chain file is present (caller handles the 'not found yet' case).
    """
    pn = read_paramnames(root)
    names = [n for n, _ in pn]
    labels = [l for _, l in pn]
    tables = []
    for c in range(1, n_chains_max + 1):
        fn = f"{root}.{c}.txt"
        if os.path.isfile(fn):
            arr = np.loadtxt(fn)
            if arr.ndim == 1:
                arr = arr[None, :]
            tables.append(arr)
    if not tables:
        return None
    full = np.vstack(tables)                       # (N, 2 + P)
    weights = full[:, 0]
    loglike = full[:, 1]
    samples = full[:, 2:]
    if samples.shape[1] != len(names):
        raise ValueError(f"{root}: {samples.shape[1]} sample cols != {len(names)} paramnames")
    return dict(weights=weights, loglike=loglike, samples=samples,
                names=names, labels=labels, n_chains=len(tables))


def load_health(root):
    """Sampler health dict. Prefer <root>.health.json (pure JSON); else parse the YAML 'health:'
    block by hand (pyyaml may be absent). Returns {} if neither is found."""
    jpath = root + ".health.json"
    if os.path.isfile(jpath):
        with open(jpath) as f:
            return json.load(f)
    ypath = root + ".yaml"
    if os.path.isfile(ypath):
        rec, in_health = {}, False
        with open(ypath) as f:
            for line in f:
                if line.startswith("health:"):
                    in_health = True
                    continue
                if in_health:
                    if line.startswith(" ") and ":" in line:
                        k, v = line.strip().split(":", 1)
                        v = v.strip()
                        try:
                            rec[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                        except ValueError:
                            rec[k] = v
                    elif line.strip() and not line.startswith(" "):
                        break
        return rec
    return {}


# Load every configured survey; record which are present.
loaded = {}
for disp, root_name in SURVEYS.items():
    root = os.path.join(RESULTS_DIR, root_name)
    rec = None
    try:
        rec = load_chain_arrays(root)
    except FileNotFoundError:
        rec = None
    except Exception as exc:
        print(f"[{disp}] error while loading {root}: {type(exc).__name__}: {exc}")
        rec = None
    if rec is None:
        print(f"[{disp}] NOT FOUND yet at {root}.*.txt -- skipping "
              f"(re-run this notebook once the {disp} fit has finished).")
        continue
    rec["health"] = load_health(root)
    rec["root"] = root
    rec["disp"] = disp
    loaded[disp] = rec
    print(f"[{disp}] loaded {rec['n_chains']} chain(s), "
          f"{rec['samples'].shape[0]} draws x {rec['samples'].shape[1]} params from {root}")

if not loaded:
    print("\nNo eBOSS/KS chains found yet. The fits write to results/real_fit/ when they finish; "
          "re-run this notebook then. (Nothing below will error -- the cells guard on `loaded`.)")


# %% [markdown]
# ## 3. Unblind — reveal the hidden offset and subtract it
#
# We read the seed from `blind.lock`, derive the deterministic offset $\delta$, and apply
# `blinding.unblind` (= subtract $\delta$ on the `Ap` / `ns` columns only). The table below shows
# **BLIND vs UNBLIND** A_p / n_s side by side so the offset that was applied is fully transparent.

# %%
offset = None
if os.path.isfile(BLIND_LOCK):
    lock = BL.read_blind_lock(BLIND_LOCK)
    offset = BL.offset_from_lock(BLIND_LOCK)
    print("blind.lock seed_str :", lock.get("seed_str"))
    print("git_commit          :", lock.get("git_commit"))
    print("blind_params        :", lock.get("blind_params"))
    print("offset_sigma_mult   :", lock.get("offset_sigma_multiple"))
    print("\n>>> HIDDEN OFFSET delta (subtracted to unblind) <<<")
    for p in BLIND_COLS:
        print(f"    delta[{p:>3}] = {offset[p]:+.6g}")
else:
    print("blind.lock NOT found at", BLIND_LOCK, "-- cannot unblind. "
          "(If the chains were exported with --no-blind they are already unblinded; "
          "in that case set offset = {p: 0.0 for p in BLIND_COLS}.)")


def col(rec, name):
    """Index of a named param column in rec['samples']."""
    return rec["names"].index(name)


def mean_sigma(values, weights=None):
    """Weighted mean +/- std (weights default to 1)."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    mu = np.average(v, weights=w)
    var = np.average((v - mu) ** 2, weights=w)
    return mu, np.sqrt(var)


# Build an UNBLIND copy of each survey's samples (only Ap/ns change).
if offset is not None:
    print(f"\n{'survey':>7} | {'param':>3} | {'BLIND mean+/-sig':>26} | "
          f"{'delta':>12} | {'UNBLIND mean+/-sig':>26}")
    print("-" * 90)
    for disp, rec in loaded.items():
        rec["samples_unblind"] = BL.unblind(rec["samples"], offset, columns=rec["names"])
        for p in BLIND_COLS:
            j = col(rec, p)
            mb, sb = mean_sigma(rec["samples"][:, j], rec["weights"])
            mu, su = mean_sigma(rec["samples_unblind"][:, j], rec["weights"])
            print(f"{disp:>7} | {p:>3} | {mb:>+12.5g} +/- {sb:<10.4g} | "
                  f"{offset[p]:>+12.5g} | {mu:>+12.5g} +/- {su:<10.4g}")
else:
    # No offset available -> treat the loaded samples as already-unblinded (or refuse).
    for disp, rec in loaded.items():
        rec["samples_unblind"] = rec["samples"].copy()
    print("\n(No offset applied -- samples are shown as loaded.)")


# %% [markdown]
# ## 4. Report — unblind posteriors, corner plot, sampler health
#
# ### 4a. Headline: A_p and n_s (mean +/- sigma) per survey

# %%
if loaded:
    print(f"{'survey':>7} | {'A_p (unblind)':>26} | {'n_s (unblind)':>26}")
    print("-" * 66)
    for disp, rec in loaded.items():
        s = rec["samples_unblind"]
        mAp, sAp = mean_sigma(s[:, col(rec, "Ap")], rec["weights"])
        mns, sns = mean_sigma(s[:, col(rec, "ns")], rec["weights"])
        print(f"{disp:>7} | {mAp:>+14.5e} +/- {sAp:<9.3e} | {mns:>+12.5g} +/- {sns:<10.4g}")
else:
    print("(no surveys loaded -- nothing to report)")


# %% [markdown]
# ### 4b. Triangle / corner plot of (A_p, n_s) + key nuisances, eBOSS vs KS overlaid
#
# Uses GetDist `triangle_plot` if available, else a matplotlib fallback. Nuisances shown when
# present: $\tau_0$ amplitude (`tau0_z0` as a proxy), $\alpha_\text{LLS}$, and $a_\text{SiIII}$
# (eBOSS only). All on the **UNBLIND** samples.

# %%
# Choose the columns to display: the two cosmology params + a few key nuisances that exist.
PREFERRED = ["Ap", "ns", "tau0_z0", "alpha_lls", "a_SiIII"]


def plot_cols_for(recs):
    """Columns present in EVERY loaded survey (so the overlay shares axes), in PREFERRED order.
    Falls back to per-survey union for solo plots."""
    if not recs:
        return []
    common = set(recs[0]["names"])
    for r in recs[1:]:
        common &= set(r["names"])
    cols = [c for c in PREFERRED if c in common]
    return cols or [c for c in PREFERRED if c in recs[0]["names"]]


recs = list(loaded.values())
cols = plot_cols_for(recs)

if not recs:
    print("(no surveys loaded -- skipping the corner plot)")
elif HAVE_GETDIST:
    mcs = []
    for rec in recs:
        idx = [col(rec, c) for c in cols]
        labels = [rec["labels"][i] for i in idx]
        mcs.append(MCSamples(
            samples=rec["samples_unblind"][:, idx],
            weights=rec["weights"],
            names=cols, labels=labels, label=rec["disp"]))
    g = gdplt.get_subplot_plotter()
    g.triangle_plot(mcs, cols, filled=True, legend_labels=[r["disp"] for r in recs])
    plt.suptitle("UNBLIND posteriors (eBOSS vs KS)", y=1.02)
    plt.show()
else:
    # Pure-matplotlib corner: lower-triangle 2D hexbin/scatter + diagonal 1D histograms.
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    if n == 1:
        axes = np.array([[axes]])
    colors = ["C0", "C1", "C2", "C3"]
    for r_i, rec in enumerate(recs):
        s = rec["samples_unblind"]
        c = colors[r_i % len(colors)]
        for i in range(n):
            xi = col(rec, cols[i])
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis("off")
                    continue
                if i == j:
                    ax.hist(s[:, xi], bins=40, density=True, histtype="step",
                            color=c, label=rec["disp"])
                    if r_i == 0:
                        ax.set_title(cols[i], fontsize=9)
                else:
                    xj = col(rec, cols[j])
                    ax.scatter(s[:, xj], s[:, xi], s=2, alpha=0.15, color=c)
                if i == n - 1:
                    ax.set_xlabel(cols[j], fontsize=8)
                if j == 0 and i != 0:
                    ax.set_ylabel(cols[i], fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("UNBLIND posteriors (eBOSS vs KS) -- matplotlib fallback", y=1.0)
    fig.tight_layout()
    plt.show()
    print("Plotted columns:", cols)


# %% [markdown]
# ### 4c. Sampler health (R-hat, ESS, divergences) -- from `<root>.health.json` / `<root>.yaml`
#
# These were always **unblinded**. Targets: R-hat < 1.01, E-BFMI >= 0.3, divergences = 0,
# ESS comfortably large, low tree-depth saturation.

# %%
HEALTH_KEYS = ["rhat_max", "ess_bulk_min", "ess_tail_min", "ebfmi_min",
               "n_divergent", "treedepth_sat_frac"]
if loaded:
    hdr = f"{'survey':>7} | " + " | ".join(f"{k:>16}" for k in HEALTH_KEYS)
    print(hdr)
    print("-" * len(hdr))
    for disp, rec in loaded.items():
        h = rec.get("health", {})
        cells = []
        for k in HEALTH_KEYS:
            v = h.get(k, "n/a")
            cells.append(f"{v:>16.4g}" if isinstance(v, (int, float)) else f"{str(v):>16}")
        print(f"{disp:>7} | " + " | ".join(cells))
        nrows = h.get("n_real_rows")
        if nrows is not None:
            print(f"        (real data rows fit: {nrows}; chains={h.get('n_chains')}, "
                  f"draws/chain={h.get('n_draws')}, per-chain div={h.get('per_chain_div')})")

    def flag(disp, h):
        msgs = []
        if isinstance(h.get("rhat_max"), (int, float)) and h["rhat_max"] >= 1.01:
            msgs.append(f"R-hat {h['rhat_max']:.3f} >= 1.01")
        if isinstance(h.get("ebfmi_min"), (int, float)) and h["ebfmi_min"] < 0.3:
            msgs.append(f"E-BFMI {h['ebfmi_min']:.2f} < 0.3")
        if isinstance(h.get("n_divergent"), (int, float)) and h["n_divergent"] > 0:
            msgs.append(f"{int(h['n_divergent'])} divergence(s)")
        return msgs

    print()
    for disp, rec in loaded.items():
        msgs = flag(disp, rec.get("health", {}))
        print(f"[{disp}] HEALTH: " + ("OK" if not msgs else "WARN -- " + "; ".join(msgs)))
else:
    print("(no surveys loaded -- no health to report)")


# %% [markdown]
# ## 5. Caveats -- KS-specific open items (4-lens review)
#
# The eBOSS headline is clean. **The KS headline is PRELIMINARY** pending three items the
# 4-lens review flagged:
#
# 1. **KS z_lo: 2.4 vs 2.8 (PI decision).** Which low-z KODIAQ-SQUAD bins enter the fit is a PI
#    call; the unblind KS A_p / n_s above use whatever the run baseline set. Confirm the z_lo
#    choice matches the intended analysis before quoting KS.
# 2. **Mean-flux n_s high-k certification.** On PRIYA-on-KS, $\tau_0$ can absorb high-k HCD power
#    (it is co-varied, not Gaussian-pinned), which couples into n_s at high k. The MF/n_s high-k
#    behaviour needs its certification before the KS n_s is treated as final.
# 3. **~0.5sigma HCD-prior-center systematic.** The HCD->n_s coupling is a real (irreducible)
#    LLS-amplitude vs forest-amplitude degeneracy; the choice of HCD-prior *center* moves the
#    cosmology by ~0.5sigma (= prior_width x degeneracy). Treat ~0.5sigma as a systematic floor
#    on the HCD-sensitive direction, especially for KS.
#
# **Bottom line:** report eBOSS as the clean public cross-check; flag KS as preliminary with the
# above caveats until (1) the z_lo decision is confirmed, (2) the MF/n_s high-k certification is
# done, and (3) the ~0.5sigma HCD-center systematic is folded into the KS error budget.

# %% [markdown]
# ## 6. How to re-run / DESI
#
# **Re-run this notebook:** just execute all cells again. Every load is guarded -- if a chain is
# missing it is skipped with a message, so it is safe to run repeatedly while fits finish. The
# unblind offset is deterministic from `blind.lock`, so re-running always yields the same reveal.
#
# **To (re)produce the chains** (blind, on the cluster):
# ```bash
# sbatch --array=0-1 scripts/batch_real_fit.sh         # 0=eboss, 1=ks  (DESI is task 2 -> private)
# # or a single survey, locally:
# PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
#   /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/run_real_fit.py --survey eboss
# ```
#
# **DESI requires the separate, local unblinding ritual.** DESI cosmology is PRIVATE: its chains
# live in `results_local/desi_production/` (gitignored) and are intentionally NOT loaded here.
# Unblind DESI only via its own authorized local step -- never in this committable notebook, and
# never paste DESI A_p / n_s into anything committable.
#
# > Reminder: do **not** commit unblind A_p / n_s values, figures, or this notebook with outputs
# > saved, until the team has agreed to unblind. `blind.lock` (the seed) is committable; the
# > *revealed offset* and the unblind cosmology are not, until the unblinding is authorized.
