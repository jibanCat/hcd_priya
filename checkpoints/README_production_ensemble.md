# Production emulator ensemble (pinned; manifest schema v2, gate E, PI ruling E5 of 2026-10-06)

The deployed production emulator is an N=5 ensemble: `prod_repaired_seed{0..4}.eqx` with paired normalizers
`prod_repaired_seed{0..4}.norm.pkl`, trained on ALL simulations (no holdout) on the S4-repaired LF cache (sha256
`d9c3783892872f8739f6c1a4039ee84cae5ab84803235a61d7d06ac119336cf2`), schema 2.0 (canonical k coordinate). The
deployed forward is the ensemble MEAN over members of the reconstructed post-exp `P_filt`
(`hcd_analysis/emulator/ensemble.py`).

The binaries live on Turbo in the gate C directory, `/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/checkpoints/gateC`
(not in git). Their identity of record is the committed machine-readable manifest
`checkpoints/production_ensemble_manifest.json`, which names that directory, the training cache and the member schema,
and which this file mirrors for humans. The manifest, this README, the on-disk files, the directory's `SHA256SUMS` and
the (regenerated) `analysis.lock` must always agree.

The pre-2026-10 manifest (schema 1, members `final_prod_seed0-4` trained on the historical cache with the pre-debug k
grid) is kept, superseded, as `checkpoints/production_ensemble_manifest_v1_pre2026-10.json`; the loader refuses
schema-1 manifests.

## Identity: digest table

| index | member | eqx sha256 | norm sha256 | meta sha256 |
|---|---|---|---|---|
| 0 | prod_repaired_seed0 | `200f60ca54fd2714b5f1e42d705b24426514494381e9042660b4c7022dc39a3f` | `810a078041fb3eb86a280b0529b1f7115589532c053a8cc3366fac7aa9a5abb8` | `cfe6426242c3e616adf9957a4c4c027b7bb43f922266a69909b30c31a6943671` |
| 1 | prod_repaired_seed1 | `315ed27cbe14cf67bba175b881c0b00f70037f67105d57cb09a05ca550716bb8` | `810a078041fb3eb86a280b0529b1f7115589532c053a8cc3366fac7aa9a5abb8` | `0bbde0456756dcfc98497a96c553103a071d5755cd0e356d9ca4de94f7267dae` |
| 2 | prod_repaired_seed2 | `b922caa8bd59a07b71b329d159fcf85c06a4f1a794b45ed10b8683a8eb07089c` | `810a078041fb3eb86a280b0529b1f7115589532c053a8cc3366fac7aa9a5abb8` | `cbf07af66c14a44996b7fa499d618b3a7e05d3f688a13a8d5dc8b46341d41cc1` |
| 3 | prod_repaired_seed3 | `5feddd80ac5f6f782becb5d6c3447a2a96a7c9ed77f62db670ca44d1ab887c03` | `810a078041fb3eb86a280b0529b1f7115589532c053a8cc3366fac7aa9a5abb8` | `97aa7ef088814ae1f44f7027b7832f88f0de4af4573c3fe8570c393655ba2b81` |
| 4 | prod_repaired_seed4 | `45c9d893fca57135a41d68f025d6b1e911e9d70481d710aabaaaefe8ba6482cd` | `810a078041fb3eb86a280b0529b1f7115589532c053a8cc3366fac7aa9a5abb8` | `cf59eea652d2c33bc84390e918958d512496fd7e4834c502681d163e5c99210a` |

The five `.norm.pkl` are byte-identical BY CONSTRUCTION (one sha256 for all five): every member is trained on the same
data with the same recipe, only the network init seed differs, so they share one normalization. Checkpoint-to-normalizer
pairing is therefore pinned STRUCTURALLY (`prod_repaired_seed<i>.eqx` pairs with `prod_repaired_seed<i>.norm.pkl`,
index order 0..4), not by digest: a digest cannot distinguish a norm mis-pair.

## How it was produced

Gate C of the emulator-debug campaign (PU-0056/PU-0057): `scripts/train_production_emulator.py --seed S` for
S = 0..4 (all-simulations training, no leave-one-out holdout; the leave-one-out runs are validation and emulator-error
sources only), run by the notes batch `gateC/batch_gateC_train_01ad2ff.sbatch` from a git-archive export of commit
01ad2ff (so each meta carries `git_sha` null; the gate C reviewer verified the export; see the directory's
`MANIFEST_README.txt`). The training config per member is in `prod_repaired_seed<i>.meta.json` (digest-pinned in the
manifest; schema 2.0, cache sha256, k_com) and the history in `prod_repaired_seed<i>.hist.json` beside it.

## How it is verified

* Load time: every deployed driver obtains the member list via `hcd_analysis.emulator.prod_ensemble`
  (`load_production_ensemble` / `production_member_paths`), which verifies against the manifest on EVERY call:
  manifest schema 2; each member meta of schema 2.0 on the manifest's cache sha256 with identical k_com; sha256 of each
  `.eqx`/`.norm.pkl`/`.meta.json`; agreement with the directory's `SHA256SUMS`; exact member count (5), exact index
  order (0..4), exact structural pairing; plus a fail-loud tripwire that raises if ANY on-disk file matches
  `prod_repaired_seed*.eqx` without being pinned (a stray `prod_repaired_seed5.eqx` is an error, never a silent sixth
  member). There is deliberately no glob/prefix argument.
* Standalone: `scripts/gen_ensemble_manifest.py --check` runs the same battery and exits nonzero on any mismatch.
* Unit tests: `tests/test_prod_ensemble_manifest.py` (modified checkpoint/normalizer, missing member, extra member,
  wrong count, wrong pairing, digest mismatch, ordering mismatch, alternate-prefix rejection, schema-1 refusal, member
  schema and cache checks, SHA256SUMS cross-check, --check green/red, real-checkpoints happy path).
* `scripts/run_real_fit.py --ensemble-glob` remains as a DIAGNOSTIC escape hatch only: it prints a prominent warning
  and stamps the run meta `ensemble_pinned: false` plus the glob used. `--single-member` runs are likewise stamped
  unpinned. No production/blind artifact may carry `ensemble_pinned: false`.

## Intentional replacement procedure

Never edit the manifest (or this table) by hand. To replace the production ensemble:

1. Retrain: `scripts/train_production_emulator.py --seed S` per member.
2. Regenerate the manifest: `scripts/gen_ensemble_manifest.py` (refuses to build over stray prefix files, members of
   another schema or cache, or digests that disagree with the directory's `SHA256SUMS`), then update this README's
   digest table to match.
3. PI sign-off on the new ensemble identity.
4. Regenerate `analysis.lock` so the lock of record carries the new member set/digests.
