# Production emulator ensemble (pinned, freeze decision 6)

The deployed production emulator is an N=5 ensemble: `final_prod_seed{0..4}.eqx` with paired
normalizers `final_prod_seed{0..4}.norm.pkl`, trained on ALL sims (no holdout) by
`scripts/train_production_emulator.py` (one run per `--seed`). The deployed forward is the
ensemble MEAN over members of the reconstructed post-exp `P_filt`
(`hcd_analysis/emulator/ensemble.py`). The binaries are gitignored (they live only at
`/home/mfho/hcd_priya/checkpoints/`); their identity of record is the committed
machine-readable manifest `checkpoints/production_ensemble_manifest.json`, which this file
mirrors for humans. The manifest, this README, the on-disk files and the (regenerated)
`analysis.lock` must always agree.

## Identity: digest table

| index | member | eqx sha256 | norm sha256 | meta sha256 |
|---|---|---|---|---|
| 0 | final_prod_seed0 | `718bd9306f9864e907d766f1765ed802e3581c39a53c7679151960fceb018057` | `b9d224fc9ca52558169660d195424ed75d1e3096881fb1ac49f151035b7a0454` | `61d15e7659f316f2125880e8ec7a278ea76b10e36af7a134bef0df35dd71d4e3` |
| 1 | final_prod_seed1 | `d1097fff6880cde3616979014977bfd8107383c05579cb6334f411efc6908047` | `b9d224fc9ca52558169660d195424ed75d1e3096881fb1ac49f151035b7a0454` | `7d598fc91990b91d03b9c86dda23d2c2df88dc1b0cb9cf5f3ac58ea92bb3be9c` |
| 2 | final_prod_seed2 | `c99b59eb45298810ef3011121333147a7206f32c6957fac825899bd3d00a6a46` | `b9d224fc9ca52558169660d195424ed75d1e3096881fb1ac49f151035b7a0454` | `62a5c215253bbc7cbbbb6c1d1bb38f708d6a3b51f7d8e6ad01a670aed6d04387` |
| 3 | final_prod_seed3 | `3a811ccb63e5bf90d3215122b3fe91a8a9c737668d1cb5c2d01b73ba4b1bbdf3` | `b9d224fc9ca52558169660d195424ed75d1e3096881fb1ac49f151035b7a0454` | `56dd59efc905a9d3a75fbafcfa95ca30b4943fcfc89aec6875a6718ca1cb0df2` |
| 4 | final_prod_seed4 | `b4100eb26928f2cb3f68e7292e06e9f6533de7100c05c5fb57bccfa8685f924d` | `b9d224fc9ca52558169660d195424ed75d1e3096881fb1ac49f151035b7a0454` | `5fae44cac6327a33a17cd8b4c1f70cb5f2f8ab0b8f71908a23baeb1510a1d752` |

The five `.norm.pkl` are byte-identical BY CONSTRUCTION (one sha256 for all five): every member
is trained on the same data with the same recipe, only the network init seed differs, so they
share one normalization. Checkpoint-to-normalizer pairing is therefore pinned STRUCTURALLY
(`final_prod_seed<i>.eqx` pairs with `final_prod_seed<i>.norm.pkl`, index order 0..4), not by
digest: a digest cannot distinguish a norm mis-pair.

## How it was produced

`scripts/train_production_emulator.py --seed S` for S = 0..4 (all-sims training, no LOSO
holdout; LOSO folds are closure/C_emu validation only). The exact training config and history
per member are in `final_prod_seed<i>.meta.json` (digest-pinned in the manifest) and the
git-tracked `final_prod_seed<i>.hist.json`.

## How it is verified

* Load time: every deployed driver obtains the member list via
  `hcd_analysis.emulator.prod_ensemble` (`load_production_ensemble` /
  `production_member_paths`), which verifies against the manifest on EVERY call: sha256 of each
  `.eqx`/`.norm.pkl`/`.meta.json`, exact member count (5), exact index order (0..4), exact
  structural pairing, plus a fail-loud tripwire that raises if ANY on-disk file matches
  `final_prod_seed*.eqx` without being pinned (a stray `final_prod_seed5.eqx` is an error,
  never a silent sixth member). There is deliberately no glob/prefix argument.
* Standalone: `scripts/gen_ensemble_manifest.py --check` runs the same battery and exits
  nonzero on any mismatch.
* Unit tests: `tests/test_prod_ensemble_manifest.py` (modified checkpoint/normalizer, missing
  member, extra member, wrong count, wrong pairing, digest mismatch, ordering mismatch,
  alternate-prefix rejection, --check green/red, real-checkpoints happy path).
* `scripts/run_real_fit.py --ensemble-glob` remains as a DIAGNOSTIC escape hatch only: it
  prints a prominent warning and stamps the run meta `ensemble_pinned: false` plus the glob
  used. `--single-member` runs are likewise stamped unpinned. No production/blind artifact may
  carry `ensemble_pinned: false`.

## Intentional replacement procedure

Never edit the manifest (or this table) by hand. To replace the production ensemble:

1. Retrain: `scripts/train_production_emulator.py --seed S` per member.
2. Regenerate the manifest: `scripts/gen_ensemble_manifest.py` (refuses to build over stray
   prefix files), then update this README's digest table to match.
3. PI sign-off on the new ensemble identity.
4. Regenerate `analysis.lock` so the lock of record carries the new member set/digests.
