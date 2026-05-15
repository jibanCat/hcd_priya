# Phase 2 emulator design memo — 2026-05-15

> Background-agent output (`aed314412039a2950`) responding to the
> 2026-05-15 PM scope pivot: mean-flux sampling moved into Phase 2.
> Read alongside
> [`2026-05-15-priya-coarse-grid-dive.md`](2026-05-15-priya-coarse-grid-dive.md)
> (PRIYA-side mechanics) and
> [`../SESSION_HANDOVER_2026_05_15.md`](../SESSION_HANDOVER_2026_05_15.md) §9.

## 1. Architecture under the new scope

**Recommendation: τ₀-sweep happens *outside* the model — the cache holds ~10 rescaled rows per (sim, snap), and Head B treats `τ₀` as an extra input dim exactly as the locked decision says.** This keeps the locked Head A / Head B structure intact, while making the τ₀ axis explicit in the *training distribution* rather than in the *forward pass*. The model has no idea where the data came from; it just sees (params, z, τ₀) → observables.

The PRIYA-side τ₀ generator (the other agent's deliverable) produces, for each base row, a small `α`-grid that rescales `τ → α·τ`, recomputes per-class P1D, per-class mean_F, and the CDDF (CDDF actually does *not* change with α — see §2). The cache builder ingests these. The model never sees `α`; it sees `τ₀ = -ln⟨F⟩` for the rescaled state.

Data flow, with shapes for a single training step (batch `B`):

```
params         (B, 9)
z              (B, 1)
tau0           (B, 1)         # the per-row mean optical depth after rescale
   |
   v   concat → encoder MLP
x_enc          (B, 11)        # 9 + z + (optionally) tau0 — see §4 Q1
latent         (B, 64)
   |
   +----------> Head A:  latent           → (B, 30) f_nhi + (B, 3) dN/dX
   |
   +----------> Head B:  concat(latent, tau0) → (B, 4*50) → four (B, 50) P1Ds
```

Note that `τ₀` appears in **two places** if z is concatenated at the encoder: at the encoder (to condition Head A's CDDF on z, because CDDF redshift-evolves) and at Head B (to condition the four P1D templates on mean flux). This is intentional and not double-counting: Head A still does not need τ₀ because the CDDF is τ₀-invariant (population statistic). Head B needs both z (via latent) and τ₀ (explicit) because P1D varies with both.

## 2. CDDF coupling — interpretation

**Recommendation: option (b) lite — a *consistency* coupling enforced through the data and the loss, not a hard physical constraint in the architecture.**

The strict reading of "P_class_only is the P1D of the subset where only that class is present, and the CDDF tells you the population mix" is correct but does **not** give a closed-form algebraic relation that the network must satisfy. The four `P_class_only` templates are statistics computed on *disjoint* sightline subsets selected by their HCD content, and the CDDF is the column-density distribution of *all* absorbers; combining them into the joint P1D requires the per-sightline class fractions, not the CDDF itself.

What we *can* enforce, and what "tightly coupled" should mean operationally:

1. **Shared latent.** Head A and Head B both branch off the same encoder, so the CDDF prediction and the per-class P1D predictions are driven by a common representation of (params, z). This is already the locked design; no change.
2. **Mean-flux consistency for `P_clean`.** Under the τ→α·τ post-processing, `P_clean` rescales in a known way (clean sightlines just transform under α). Add an *auxiliary* loss term: predicted `mean_F_clean` from the model (or the cache row) must equal the implied `exp(-τ₀)` to within tolerance. This is cheap, and it forces Head B to learn the τ₀ dependence smoothly.
3. **CDDF is τ₀-invariant.** During training, the same CDDF target is reused across all ~10 τ₀ variants of a given (sim, snap). This is a real coupling: it tells Head A that the τ₀ axis is a nuisance dim it should ignore, and provides ~10× redundancy on CDDF targets without 10× cost.
4. **Joint loss weights.** Keep the joint scalar loss but weight Head A's CDDF term by 1/(10) when iterating over τ₀-replicated rows, otherwise Head A's gradient signal is 10× inflated and Head B starves.

Net architectural change: **none**. Net loss change: add the mean-flux-consistency auxiliary term, and re-weight the joint loss to account for τ₀-replication.

## 3. Training-set sizing & shape

- **Split.** Keep "hold out 6 of 64 sims entirely", but the held-out sims *include all their τ₀ variants*. Don't let τ₀-siblings of a train row leak into val/test. Effective sizes: train ≈ 58 sims × ~17 snaps × 10 τ₀ ≈ 9,860; val/test ≈ 6 sims × ~17 × 10 ≈ 1,020. Use random val *within the 58 train sims* (e.g. 10% of snaps).
- **Batch size / steps.** With ~10k rows and a small MLP (encoder 256→128→64, heads of comparable size), the model is parameter-poor by modern standards (<1M params). Batch size 256, ~200 epochs ≈ 8,000 steps, single GPU (A100 or even L4): wall time **5–15 min**. Don't over-engineer.
- **Cache schema.** Keep the existing `observables.h5` schema and **add a *second* on-disk cache** (e.g. `observables_tau0.h5`) keyed by (sim, snap, α_idx) with the same per-row columns plus a `tau0` scalar and an `alpha` scalar. Reasons: (i) preserves Phase 1 as the immutable natural-τ₀ reference; (ii) the τ₀ cache can be regenerated independently when the PRIYA-side recipe changes; (iii) the data loader stacks the two transparently. CDDF rows are duplicated across α (cheap, 30 floats per row).
- **NaN masking.** τ₀-rescaling is uniform across pixels and doesn't change the velocity grid, so the Nyquist mask is unchanged. The existing per-row NaN pattern for P1D bins above each snap's native Nyquist carries over verbatim to the rescaled rows.

## 4. Revisiting the 5 open design questions

1. **Where does `z` enter?** Unchanged: input #10 to the shared encoder. The new scope adds `τ₀` as input #11 to Head B (and only Head B). Single model covers all snaps and all mean-flux states.
2. **Train/val/test split.** Unchanged in principle (hold out 6 sims), but with the explicit rule that all τ₀-siblings of a held-out sim are held out too. No τ₀ leakage.
3. **Loss structure.** Joint scalar, NaN-safe, plus (new) the τ₀-replication re-weighting in §2 and the optional mean-flux-consistency auxiliary term. Equal weights per remaining term to start; sweep later.
4. **Architecture sizes.** Unchanged starting defaults. Encoder `[10 → 256 → 128 → 64]`, Head A `[64 → 128 → 33]`, Head B `[65 → 256 → 200]`. The 10k-row dataset is too small to justify going bigger before a sweep.
5. **Output transforms.** Unchanged: log space for `f_nhi`, `dN/dX`, and all four P1Ds. The dynamic range argument is even stronger across τ₀ variants.

## 5. Concrete next actions (priority-ordered)

1. **Write `docs/superpowers/plans/2026-05-15-phase2-jax-emulator.md`** — the implementation plan, structured around the four code modules below, with TDD checkpoints. This is the right next artifact, not code.
2. **Implement first: the data loader** (`hcd_analysis/emulator/data.py`) that ingests both caches (Phase-1 natural-τ₀ + Phase-2 τ₀-extended) and yields batched (params, z, τ₀, targets, masks). Loader can be written **before** the PRIYA-side cache is finalised, as long as we agree on the schema (one extra `tau0` column, otherwise identical).
3. **In parallel, the JAX model** (`hcd_analysis/emulator/model.py`) — encoder + Head A + Head B as Flax/Equinox modules, plus the joint loss. Independent of the PRIYA dive.
4. **Blocked on PRIYA-code-dive sub-agent**: the actual τ₀-extended cache builder (`scripts/build_emulator_cache_tau0.py`). Don't start coding it until their findings land — but **agree the schema now** so the loader and model can be tested against synthetic τ₀-extended rows.
5. **Can start without their findings**: loader, model, training loop skeleton (`hcd_analysis/emulator/train.py`), unit tests on a synthetic 100-row cache, NaN-mask handling tests. All of this exercises the architecture without needing real rescaled data.
