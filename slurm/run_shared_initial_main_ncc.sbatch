#!/bin/bash
#SBATCH --job-name=lfhe-main-shared-init
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=logs/shared_init_%A_%a.out
#SBATCH --error=logs/shared_init_%A_%a.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
mkdir -p logs

MANIFEST="${MANIFEST:-$PWD/manifests/workshop_main_shared_static_init_n50_500.csv}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"

"$PYTHON_BIN" -u run_manifest_row.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --main main.py \
  --data-root "${LFHE_DATA_ROOT:-./data}"
status=$?
if [[ $status -eq 99 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
