# cup1d cross-reference -> see ~/cup1d_ref/

The DESI DR1 reference P1D pipeline (`cup1d`) + its emulator (`LaCE`) live OUTSIDE this repo, in
**`~/cup1d_ref/`**, kept separate so they cannot perturb this analysis or the `emu-jax` env.

**For any cup1d cross-check, consistency question, or cross-code SBC test, read `~/cup1d_ref/CLAUDE.md`
first.** It has the env setup, the run API, the cosmology mapping (`A_p/n_s` vs cup1d's compressed
`Delta2_star/n_star`), the audited consistency findings, and the cross-SBC plan.

Audit write-ups (in the notes repo):
- `~/hcd_priya_notes/docs/superpowers/2026-06-30-cup1d-likelihood-crosscheck.md` (full-likelihood audit)
- `~/hcd_priya_notes/docs/superpowers/2026-06-30-desi-metal-model-vs-modelc.md` (metal Eq-4.9 vs our Model C)

Headline: our likelihood is broadly consistent with cup1d (documentable differences); the metal model needs
extending to "Model C+" (float the SiIII/SiII decorrelation scales + add the SiIII-SiII cross term), and the
emulator covariance must run emucoh-ON. The HCD tight-prior and thermal-IGM slaving are deliberate,
documented divergences.
