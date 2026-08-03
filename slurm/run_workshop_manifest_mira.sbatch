#!/bin/bash
#SBATCH --job-name=lfhe-workshop
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=logs/workshop_%A_%a.out
#SBATCH --error=logs/workshop_%A_%a.err

set -uo pipefail
cd "$HOME/LFHE_workshop"
mkdir -p logs
: "${MANIFEST:?set MANIFEST to a workshop CSV path}"
"$HOME/LFHE_workshop/.venv/bin/python" -u run_manifest_row.py \
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
