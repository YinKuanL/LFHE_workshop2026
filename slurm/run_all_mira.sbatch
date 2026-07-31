#!/bin/bash
#SBATCH --job-name=lfhe-mira-all
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --array=0-412%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python -u run_manifest_row.py \
  --manifest manifests/mira_all.csv \
  --index "$SLURM_ARRAY_TASK_ID" \
  --main main.py \
  --data-root "${LFHE_DATA_ROOT:-./data}"
status=$?
if [[ $status -eq 99 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
