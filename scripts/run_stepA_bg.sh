#!/usr/bin/env bash
# Thin background wrapper for the STEP-A launcher. Launches the 14-worker pool DETACHED
# (nohup + setsid) so it survives the shell/SSH session, with all output to a log. Idempotent
# + RESTARTABLE: re-running it just resumes (finished chains are skipped by their checkpoints).
#
#   bash scripts/run_stepA_bg.sh            # default: 14 workers, full 44-chain run
#   WORKERS=10 bash scripts/run_stepA_bg.sh # override the pool size
#
# Watch:   tail -f checkpoints/stepA/run_stepA.out
#          watch -n5 cat checkpoints/stepA/health.txt
#          python3 -c "import json;print(json.load(open('checkpoints/stepA/health.json'))['counts'])"
set -euo pipefail

REPO=/home/mfho/hcd_priya
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
CKPT=$REPO/checkpoints/stepA
WORKERS=${WORKERS:-14}

mkdir -p "$CKPT"
LOG="$CKPT/run_stepA.out"

# refuse to double-launch (a running pool holds a pidfile).
PIDFILE="$CKPT/run_stepA.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "STEP-A pool already running (pid $(cat "$PIDFILE")). Tail: $LOG" >&2
  exit 1
fi

echo "[bg] launching STEP-A pool: $WORKERS workers; log -> $LOG"
setsid env \
  PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 \
  "$PY" "$REPO/scripts/run_stepA.py" --run --workers "$WORKERS" \
  >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "[bg] pool pid $(cat "$PIDFILE"). Watch: tail -f $LOG ; cat $CKPT/health.txt"
