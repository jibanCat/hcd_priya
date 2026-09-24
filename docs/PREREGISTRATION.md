# Pre-registered real-data leg order and unblind protocol (documentation only)

Recorded per `hcd_priya_notes/docs/superpowers/2026-07-20-plan-to-unblind.md` step 3 and PI DECISIONS #24 (2026-09-24).

- Leg order, decided while blinded (2026-07-20): **eBOSS -> DESI -> KS**. One leg at a time; a leg's cosmology is viewed at most once (unblind-once); no forward, prior, covariance or likelihood change on a leg after it is unblinded within the same lock era.
- The sole unblind certificate per leg is that leg's deployed-geometry SBC arm (Wave-2 re-SBC). Status at 2026-09-24: eBOSS A1c PROMOTED (certificate); DESI A2c UNCERTIFIED (no fit licensed); KS A3c FAIL closed-final (no fit licensed).
- Blinding: parameter-blind on (ns, Ap) by the additive offset derived at view time from the sealed `blind.lock` (seed `hcd_priya_real_fit_v1@aefaf51`, sha256 pinned in `analysis.lock`). `blind.lock` is never modified or regenerated.
- eBOSS execution of record: PI #24 authorizes ONE blind fit under the frozen forward `68f71a3d` at 4 chains x 2000 draws (sampler-length-only extension), via the pinned wrapper in the notes repo (`docs/superpowers/eboss-realdata-2026-09/`), then the governed unblind-once (`scripts/eboss_unblind_once.py`) and the preregistered PRIYA consistency readout (`scripts/eboss_priya_consistency.py`). Preregistration: `hcd_priya_notes/docs/superpowers/eboss-realdata-2026-09/2026-09-24-EBOSS-REALDATA-PREREGISTRATION-v1.md`.
- DESI real data additionally requires the P0-C private-routing guard and a clean certificate; KS additionally requires a new PI ruling and the parked S < 0.50 authorization.
