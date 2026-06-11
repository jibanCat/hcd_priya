"""Empirical recovery validation for the τ₀ refactor — STEP 2+3 (post-4-lens-review):
INTERIOR fiducial (fold6, n_s≈0.956-0.979, near the real cosmology) with the closure HCD prior
RE-CENTERED on the sim's own w_c (the referees' prerequisite: the lit/sim-centered prior was
pre-biasing the LLS↔subDLA split). This isolates the residual A_p bias (the genuine HCD-split
leak + the irreducible τ₀ floor) from the fixable artifacts (edge fiducial + prior mis-centering).
Dumps the slope-by-default sites too. Bypasses run_stepA's packed/names; saves to checkpoints/tau0val_interior/."""
import os, numpy as np, warnings; warnings.simplefilter("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import hcd_analysis.emulator  # noqa
import jax, jax.numpy as jnp
from numpyro.infer import init_to_sample
import numpyro.diagnostics as npd
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.inference import HCD_PRIOR_FRAC_SIGMA, HCD_DLA_RESIDUAL_FRAC, lit_over_sim_at_z

OUT = "checkpoints/tau0val_interior"; os.makedirs(OUT, exist_ok=True)
FOLD = 6; CKPT = f"{CL.REPO}/checkpoints/final_fold{FOLD}"
N_CHAINS, NW, NS, MTD = 3, 150, 200, 8

print(f"[tau0val] building DESI-only ctx, fold{FOLD} emulator (interior n_s)...", flush=True)
ctx, d = CL.build_legb_ctx(ckpt=CKPT)
ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
assert ctx.marginalize_zslope is True

sims, _ = CL.held_out_sims(d, fold=FOLD)
# pick the sim nearest n_s_unit≈0.66 (n_s≈0.965, ~Planck/eBOSS) among the held-out set
names = np.asarray(d["sim_name"]); pu = d["params_unit"]
nss = np.array([float(pu[np.where(names == s)[0][0], 0]) for s in sims])
sim = sims[int(np.argmin(np.abs(nss - 0.66)))]
truth = CL.make_truth_from_sim(d, sim, fold=FOLD)            # default "priya" central τ₀ anchor

# STEP 2 — re-center the CLOSURE HCD prior on the sim's own w_c (LLS, subDLA); keep DLA (it already
# recovers the 0.10·w_DLA masked residual cleanly). Removes the lit/sim-vs-truth pre-bias on the split.
wc = np.asarray(truth["w_c"]); fl, fs, fd = HCD_PRIOR_FRAC_SIGMA
new_mu = ctx.alpha_hcd_mu.at[0].set(float(wc[0])).at[1].set(float(wc[1]))
new_sig = ctx.alpha_hcd_sigma.at[0].set(fl * float(wc[0])).at[1].set(fs * float(wc[1]))
ctx = ctx._replace(alpha_hcd_mu=new_mu, alpha_hcd_sigma=new_sig)
print(f"[tau0val] sim={sim} n_s_unit={truth['params_unit'][0]:.3f} (phys {0.8+0.25*truth['params_unit'][0]:.3f}) "
      f"Ap_unit={truth['params_unit'][1]:.3f} tau0_amp={truth['tau0_amp']:.3f} dtau0={truth['dtau0']:.3f}", flush=True)
print(f"[tau0val] HCD prior re-centered on sim w_c: LLS={float(wc[0]):.3f} subDLA={float(wc[1]):.3f} (DLA prior kept)", flush=True)

k0 = jax.random.PRNGKey(20260610)
mock_legs, _tp, _info = CL.make_legb_mock(ctx, truth, k0)
core = CL._mock_core_per_leg(ctx, truth)

chains = []
for c in range(N_CHAINS):
    print(f"[tau0val] chain {c} ...", flush=True)
    s, ndiv = CL._run_nuts_legb(ctx, mock_legs, core, n_warmup=NW, n_samples=NS,
                                seed=jax.random.fold_in(k0, 100 + c), max_tree_depth=MTD,
                                dense_mass=True, init_strategy=init_to_sample)
    chains.append({k: np.asarray(v) for k, v in s.items()})
    print(f"[tau0val] chain {c} done, div={ndiv}", flush=True)

# slope-by-default sites now present (marginalize_zslope=True) — dump them too (referee ask)
keys = ["theta_unit", "tau0_amp", "dtau0", "tau0_vec", "alpha_lls", "alpha_subdla", "alpha_dla"]
keys += [k for k in ("s_lls", "s_subdla", "s_dla") if k in chains[0]]
def stack(key): return np.stack([ch[key] for ch in chains])
np.savez(f"{OUT}/recovery.npz", **{k: stack(k) for k in keys},
         truth_theta=truth["params_unit"], truth_tau0=truth["tau0"], truth_z=truth["z"],
         truth_tau0_amp=truth["tau0_amp"], truth_dtau0=truth["dtau0"], truth_wc=truth["w_c"],
         dla_truth=HCD_DLA_RESIDUAL_FRAC * float(np.asarray(lit_over_sim_at_z(3.0))[2]) * float(wc[2]))

th = stack("theta_unit"); thp = th.reshape(-1, th.shape[-1])
def rhat(a): return float(npd.split_gelman_rubin(a))
print(f"[tau0val] R-hat n_s={rhat(th[:,:,0]):.3f} A_p={rhat(th[:,:,1]):.3f} tau0_amp={rhat(stack('tau0_amp')):.3f}", flush=True)
# headline bias-z
for j, nm in [(0, "n_s"), (1, "A_p")]:
    c = thp[:, j]; t = float(truth["params_unit"][j])
    print(f"[tau0val] {nm} bias_z = {(t-c.mean())/c.std():+.2f}σ  (truth_unit {t:.3f}, post {c.mean():.3f})", flush=True)
for k in ("alpha_lls", "alpha_subdla"):
    c = stack(k).reshape(-1); i = 0 if k.endswith("lls") else 1; t = float(wc[i])
    print(f"[tau0val] {k} bias_z = {(t-c.mean())/c.std():+.2f}σ (re-centered truth {t:.3f}, post {c.mean():.3f})", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for j, nm, phys in [(0, "n_s", lambda x: 0.8+0.25*x), (1, "A_p (1e-9)", lambda x: 1.2+1.4*x)]:
    c = thp[:, j]; t = float(truth["params_unit"][j]); bz = (t-c.mean())/c.std()
    ax[0].hist(phys(c), bins=30, alpha=0.5, density=True, label=f"{nm}: bias {bz:+.2f}σ")
    ax[0].axvline(phys(t), color="k", ls="--", lw=1.2)
ax[0].set_title(f"n_s, A_p recovery — INTERIOR fold{FOLD} + re-centered HCD prior\nR-hat≈1, dashed=truth"); ax[0].set_xlabel("physical"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
tv = stack("tau0_vec").reshape(-1, len(truth["z"])); zt = truth["z"]
lo, mid, hi = np.percentile(tv, [16, 50, 84], 0)
ax[1].fill_between(zt, lo, hi, alpha=0.35, color="#2471a3", label="posterior 16–84%"); ax[1].plot(zt, mid, color="#2471a3", lw=1.5)
ax[1].plot(zt, truth["tau0"], "ko", ms=5, label="truth τ₀(z)"); ax[1].set_title("τ₀(z) posterior vs truth"); ax[1].set_xlabel("z"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
truth_a = [float(wc[0]), float(wc[1]), HCD_DLA_RESIDUAL_FRAC*float(np.asarray(lit_over_sim_at_z(3.0))[2])*float(wc[2])]
for i, (k, lab) in enumerate(zip(("alpha_lls", "alpha_subdla", "alpha_dla"), ["α_LLS", "α_subDLA", "α_DLA(resid)"])):
    c = stack(k).reshape(-1); bz = (truth_a[i]-c.mean())/c.std()
    ax[2].errorbar(i, c.mean(), yerr=c.std(), fmt="o", color="#2471a3", capsize=4)
    ax[2].plot(i, truth_a[i], "k_", ms=20, mew=2.5); ax[2].annotate(f"{bz:+.1f}σ", (i, c.mean()), fontsize=8, xytext=(6, 0), textcoords="offset points")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["α_LLS", "α_subDLA", "α_DLA(resid)"], fontsize=8); ax[2].set_title("HCD α vs re-centered truth (black _)"); ax[2].grid(alpha=0.3)
plt.suptitle(f"τ₀-refactor recovery STEP 2+3 — interior fold{FOLD} (n_s≈{0.8+0.25*float(truth['params_unit'][0]):.3f}), HCD prior re-centered on sim w_c", y=1.02)
plt.tight_layout(); out = "figures/analysis/05_likelihood/tau0_refactor_recovery_interior.png"
plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"[tau0val] saved {out}\n[tau0val] DONE", flush=True)
