# emu-jax environment provenance (Phase-2b)

**Date:** 2026-06-01
**Env:** `/home/mfho/.conda/envs/emu-jax` (python 3.11), created via
`mamba create -n emu-jax python=3.11`, JAX stack pip-installed.

## Installed versions
```
equinox==0.13.8  h5py==3.16.0  jax==0.10.1  jaxlib==0.10.1
jax-cuda12-pjrt==0.10.1  jax-cuda12-plugin==0.10.1  jaxtyping==0.3.10
numpy==2.4.6  optax==0.2.8  typing_extensions==4.15.0
```

## ⚠️ MANDATORY: always set `PYTHONNOUSERSITE=1`

`~/.local/lib/python3.11/site-packages` is populated (numpy<2, h5py, typing_extensions,
etc.). Without `PYTHONNOUSERSITE=1`, the emu-jax interpreter **silently borrows**
those user-site packages (e.g. a stale numpy<2), and `pip install` reports them
"already satisfied" and skips installing into the env. The env was made
self-contained by reinstalling under `PYTHONNOUSERSITE=1`. Every run/test/install
command MUST set it:

```bash
export EMUJAX=/home/mfho/.conda/envs/emu-jax/bin/python3
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya $EMUJAX -m pytest <test> -v
```

Do **not** `conda activate` (matches the emu-3.9 convention).

## GPU / CUDA
- `jax[cuda12]` wheels are self-contained (bundle CUDA via `jax-cuda12-*`).
- On a **login node** there is no GPU → `cuInit` error 303 → JAX falls back to CPU.
  This is expected and fine for the TDD tasks (Tasks 0–12, 16, 17).
- **Training (Task 13) + validation (Task 15) run on a GPU node** under SLURM
  account **cavestru0**, partition `gpu` (`srun -A cavestru0 -p gpu --gres=gpu:1`).
- ⚠️ The cavestru0 budget is tight (~4000 CPU-h equiv) — profile one fold before
  launching the k-fold LOSO sweep. See memory `cavestru0-compute-budget`.

## This env never imports fake_spectra
The emulator code is pure-Python/JAX; it does NOT need the emu-3.9 build env or
the GSL `LD_LIBRARY_PATH` shim. Keep the two envs separate.
