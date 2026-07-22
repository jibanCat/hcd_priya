#!/bin/bash
# === ATTIC (2026-07-22 freeze triage execution; dispositions PI-approved 2026-07-20, hcd_priya_notes docs/superpowers/2026-07-20-freeze-script-triage.md). NOT IN THE FROZEN FORWARD. ===
# retrain wave launcher; not the deployed emulator.
# Run a sequence of retrain_headroom_pilot variants on a fold, one at a time (controls core use).
# Usage: _retrain_runwave.sh <fold> <tag> <variant1> <variant2> ...
set -e
cd /home/mfho/hcd_priya
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# cap XLA intra-op threads so we don't saturate the shared 24-core node
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=3"
FOLD=$1; TAG=$2; shift 2
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
for V in "$@"; do
  echo "=========== WAVE: variant=$V fold=$FOLD tag=$TAG  $(date +%H:%M:%S) ==========="
  $PY scripts/retrain_headroom_pilot.py --fold "$FOLD" --variant "$V" --tag "$TAG" --seed 0 \
    2>&1 | grep -vE "WARNING|XLA|tcmalloc|^I[0-9]|^W[0-9]|external/" || echo "VARIANT_FAILED:$V"
done
echo "=========== WAVE DONE $(date +%H:%M:%S) ==========="
