#!/bin/bash
# Task-2 Phase-1 (OOS instrument-resolution arm) end-to-end GOLDEN + OOS smoke, on cavestru0.
# (1) the resolution/OOS pytest regression suite (too heavy for the interactive node) -> golden path intact;
# (2) a tiny end-to-end SCALAR resolution injection (golden, arm=resolution) AND a tiny OOS bstar injection
#     (new, arm=resolution_oos) via the deployed float-f_res DESI forward, --smoke (1 mock, tiny NUTS). Confirms
#     the whole chain (basis npz -> _resolve_res_instr_inject -> per-z injection -> run_legb -> pkl) works, the
#     tags are correct, and 0 divergences.
#   sbatch scripts/batch_smoke_task2_oos.sh
#SBATCH --job-name=smoke_task2oos
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/mfho/hcd_priya
#SBATCH --output=/home/mfho/hcd_priya/logs/smoke_task2oos_%A.out
#SBATCH --error=/home/mfho/hcd_priya/logs/smoke_task2oos_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mfho@umich.edu
set -uo pipefail

NCPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NCPU"
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
PY=/home/mfho/.conda/envs/emu-jax/bin/python3
OUT=/scratch/cavestru_root/cavestru0/mfho
mkdir -p "$OUT/smoke_t2_scalar" "$OUT/smoke_t2_oos" /home/mfho/hcd_priya/logs

echo "=== [1] resolution/OOS regression pytest  start: $(date) ==="
"$PY" -m pytest tests/test_legb_resolution.py tests/test_dnuis_inject.py tests/test_resinj_instr.py \
    tests/test_resinj_oos_spec.py tests/test_analyze_dnuis_resolution_oos.py tests/test_res_instr_basis.py \
    tests/test_dnuis_arm_inject_spec.py -q
RC_REG=$?
echo "=== regression exit=$RC_REG ==="

echo "=== [2a] GOLDEN scalar resolution injection (arm=resolution, DESI float-f_res)  $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py --arm resolution --survey desi --float-res --b-res 0.02 \
    --shard 0 --n-shards 1 --smoke --out-dir "$OUT/smoke_t2_scalar"
RC_SCAL=$?

echo "=== [2b] OOS bstar injection (arm=resolution_oos, DESI float-f_res)  $(date) ==="
"$PY" -u scripts/run_dnuis_bias_shard.py --arm resolution --survey desi --float-res \
    --b-res-oos-member bstar --b-res-oos-strength 1.0 \
    --shard 0 --n-shards 1 --smoke --out-dir "$OUT/smoke_t2_oos"
RC_OOS=$?

echo "=== [3] verdict  $(date) ==="
"$PY" - "$OUT/smoke_t2_scalar" "$OUT/smoke_t2_oos" <<'PYEOF'
import sys, glob, pickle
def load(d):
    fs = sorted(glob.glob(f"{d}/*_shard_*.pkl"))
    return pickle.load(open(fs[0], "rb")) if fs else None
scal, oos = load(sys.argv[1]), load(sys.argv[2])
ok = True
for name, rec, want_arm in (("scalar", scal, "resolution"), ("oos", oos, "resolution_oos")):
    if rec is None:
        print(f"[{name}] NO PKL -> FAIL"); ok = False; continue
    arm = rec.get("arm"); meta = rec.get("meta", {})
    ndiv = sum(int(r.get("n_div",0)>0) for r in rec.get("clean_per_mock",[]) + rec.get("inj_per_mock",[]))
    inj = meta.get("inject_spec")
    print(f"[{name}] arm={arm}  oos_member={meta.get('b_res_oos_member')}  n_div_fits={ndiv}  inject={inj}")
    if arm != want_arm: print(f"[{name}] arm != {want_arm} -> FAIL"); ok = False
    if ndiv != 0: print(f"[{name}] divergences -> INVESTIGATE"); ok = False
if oos is not None:
    m = oos.get("meta", {})
    isp = m.get("inject_spec") or {}
    res = isp.get("resolution") if isinstance(isp, dict) else None
    if not (isinstance(res, dict) and res.get("member") == "bstar"):
        print(f"[oos] inject_spec.resolution.member != bstar (got {res}) -> FAIL"); ok = False
print("SMOKE_VERDICT:", "PASS" if ok else "FAIL")
PYEOF
echo "=== regression=$RC_REG scalar=$RC_SCAL oos=$RC_OOS  done: $(date) ==="
