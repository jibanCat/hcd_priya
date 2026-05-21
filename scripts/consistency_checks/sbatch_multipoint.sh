#!/bin/bash
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=priya_pcheck
#SBATCH --output=/home/mfho/hcd_priya/scripts/consistency_checks/sbatch_logs/%x_%A_%a.out
#SBATCH --error=/home/mfho/hcd_priya/scripts/consistency_checks/sbatch_logs/%x_%A_%a.err
#SBATCH --array=0-11

# Fan out 12 (sim_idx, snap, z_idx, sim_folder) tuples to one task each.
# 3 sims × 4 redshifts. Same parameter set as the serial multipoint script.

# All in shell arrays (bash >= 4):
SIM_IDX=(    44 44 44 44     0  0  0  0    29 29 29 29 )
SIM_FOLDER=(
    "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"
    "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"
    "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"
    "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"
    "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"
    "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"
)
SNAP=(       8 11 17 21      8 11 17 21      8 11 17 21 )
ZIDX=(       0  3  8 11      0  3  8 11      0  3  8 11 )

i=$SLURM_ARRAY_TASK_ID
echo "Job array task $i: sim_idx=${SIM_IDX[$i]} snap=${SNAP[$i]} z_idx=${ZIDX[$i]}"

set -euo pipefail

# Bypass conda activate / mamba module reloading entirely; use the env's
# python binary directly and inject GSL on LD_LIBRARY_PATH so that
# fake_spectra._spectra_priv (C extension linking libgsl.so.25) loads.
# Order matters: GSL first (for libgsl.so.25), then conda env's libstdc++ for
# scipy's _highs C extension (needs GLIBCXX_3.4.29, which the system gcc 10.3
# libstdc++ does NOT have).
export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:${LD_LIBRARY_PATH:-}
PY=/home/mfho/.conda/envs/emu-3.9/bin/python3

# NumPy/MKL/OpenBLAS auto-threading - match cpus-per-task we requested
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

OUT_DIR=/home/mfho/hcd_priya/docs/superpowers/figs/multipoint
mkdir -p "$OUT_DIR"

$PY /home/mfho/hcd_priya/scripts/consistency_checks/priya_p1d_consistency_one_snap.py \
    --sim-idx ${SIM_IDX[$i]} \
    --sim-folder "${SIM_FOLDER[$i]}" \
    --snap ${SNAP[$i]} \
    --z-idx ${ZIDX[$i]} \
    --out "$OUT_DIR/sim${SIM_IDX[$i]}_snap${SNAP[$i]}.npz"
