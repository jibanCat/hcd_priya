#!/bin/bash
# Task-1 (KS f_res 0.15 / k_max 0.065 wiring) end-to-end GOLDEN + WIRED smoke, on cavestru0.
# (1) heavy forward-regression pytest that is too big for the interactive node;
# (2) a short SBC self-draw on the DEPLOYED forward for BOTH the golden DESI path (must stay 0-div,
#     unchanged) and the newly-wired KS path (first real end-to-end build of the deployed KS echelle
#     f_res forward + diag cov surgery via build_legb_ctx(ks_kwargs=...)). We assert 0 divergences and
#     that the KS pkl stamps sample_res=True / ks_kmax=0.065.
#   sbatch scripts/batch_smoke_task1.sh
#SBATCH --job-name=smoke_task1
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/smoke_task1_%A.out
#SBATCH --error=/home/mfho/hcd_priya/logs/smoke_task1_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -uo pipefail

NCPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
OUT=/scratch/cavestru_root/cavestru0/mfho
mkdir -p "$OUT/smoke_task1_ks" "$OUT/smoke_task1_desi" /home/mfho/hcd_priya/logs

echo "=== [1] heavy forward-regression pytest  start: $(date) ==="
"$PY" -m pytest tests/test_legb_resolution.py tests/test_norc_forward.py -q
RC_REG=$?
echo "=== regression exit=$RC_REG ==="

echo "=== [2a] KS-WIRED SBC smoke (deployed echelle f_res + diag surgery)  start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py --leg KS --shard 0 --n-shards 1 --n-mocks 1 \
    --n-warmup 100 --n-samples 150 --max-tree-depth 8 --out-dir "$OUT/smoke_task1_ks"
RC_KS=$?

echo "=== [2b] DESI-GOLDEN SBC smoke (unchanged path)  start: $(date) ==="
"$PY" -u scripts/run_prod_sbc_shard.py --leg DESI --shard 0 --n-shards 1 --n-mocks 1 \
    --n-warmup 100 --n-samples 150 --max-tree-depth 8 --out-dir "$OUT/smoke_task1_desi"
RC_DESI=$?

echo "=== [3] verdict  $(date) ==="
"$PY" - "$OUT/smoke_task1_ks" "$OUT/smoke_task1_desi" <<'PYEOF'
import sys, glob, pickle
def load(d):
    fs = sorted(glob.glob(f"{d}/mock_*.pkl"))
    if not fs: return None
    return pickle.load(open(fs[0], "rb"))
ks, desi = load(sys.argv[1]), load(sys.argv[2])
ok = True
for name, rec in (("KS", ks), ("DESI", desi)):
    if rec is None:
        print(f"[{name}] NO PKL WRITTEN -> FAIL"); ok = False; continue
    ndiv = rec.get("n_div", "?"); cfg = rec.get("run_cfg", {})
    print(f"[{name}] n_div={ndiv}  sample_res={cfg.get('sample_res')}  "
          f"f_res_amp_sigma={cfg.get('f_res_amp_sigma')}  ks_kmax={cfg.get('ks_kmax')}  "
          f"metal_prior={cfg.get('metal_prior')}")
    if ndiv not in (0,): print(f"[{name}] n_div != 0 -> INVESTIGATE"); ok = False
if ks is not None:
    c = ks.get("run_cfg", {})
    if not (c.get("sample_res") is True and c.get("ks_kmax") == 0.065):
        print("[KS] WIRED STAMP WRONG (expected sample_res=True, ks_kmax=0.065) -> FAIL"); ok = False
print("SMOKE_VERDICT:", "PASS" if ok else "FAIL")
PYEOF
echo "=== regression=$RC_REG ks=$RC_KS desi=$RC_DESI  done: $(date) ==="
