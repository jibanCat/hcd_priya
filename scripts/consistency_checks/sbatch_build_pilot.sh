#!/bin/bash
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tau0_pilot
#SBATCH --output=/home/mfho/hcd_priya/scripts/consistency_checks/sbatch_logs/%x_%j.out
#SBATCH --error=/home/mfho/hcd_priya/scripts/consistency_checks/sbatch_logs/%x_%j.err

# Production-build PILOT: 2 full-resolution (sim, snap) pairs x 20 alpha,
# Tier P + Tier C, to measure real per-pair wallclock and validate the
# full-resolution cache output before committing the full 1076-pair campaign.

set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

OUT=/tmp/tau0_pilot_${SLURM_JOB_ID}.h5

echo "=== pilot start $(date) ==="
/usr/bin/time -v $PY /home/mfho/hcd_priya/scripts/build_emulator_cache_tau0.py \
    --limit 2 --n-alpha 20 --output "$OUT" --spot-check 2>&1
echo "=== pilot end $(date) ==="
echo "output size:"; ls -la "$OUT"
$PY - "$OUT" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    print("datasets:", sorted(f.keys()))
    print("n_rows:", f.attrs["n_rows"], "n_snaps:", f.attrs["n_snaps"],
          "cache_version:", f.attrs["cache_version"])
    print("P_tier_p shape:", f["P_tier_p"].shape)
    print("alpha_slope[:5]:", f["alpha_slope"][:5])
    print("z_grid:", np.unique(f["z_grid"][...]))
    print("params[0]:", f["params"][0])
    finite = np.isfinite(f["P_tier_p"][...]).mean()
    print("P_tier_p finite frac:", finite)
PYEOF