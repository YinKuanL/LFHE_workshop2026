#!/bin/bash
#SBATCH --job-name=lfhe1000
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue

set -uo pipefail
mkdir -p logs outputs

# Adjust these two lines to your Mira environment.
source .venv/bin/activate
export LFHE_DATA_ROOT="${LFHE_DATA_ROOT:-$PWD/data}"

METHOD="${METHOD:-lfhe}"
SEED="${SEED:-42}"
NCLIENTS="${NCLIENTS:-1000}"
OUT="outputs/${METHOD}_n${NCLIENTS}_seed${SEED}"

python -u lfhe_scalable_experiment.py \
  --method "$METHOD" \
  --num-clients "$NCLIENTS" \
  --seed "$SEED" \
  --rounds 300 \
  --alpha 0.1 \
  --min-samples 10 \
  --participation-rate 0.10 \
  --local-steps 1 \
  --batch-size 16 \
  --max-degree 4 \
  --topology-interval 10 \
  --eval-interval 10 \
  --eval-clients 50 \
  --checkpoint-interval 25 \
  --output-dir "$OUT" \
  --resume
STATUS=$?

# The Python process returns 99 after saving a signal-triggered checkpoint.
# Requeue the same job so --resume continues from checkpoint.pt.
if [[ "$STATUS" -eq 99 ]]; then
  echo "Checkpoint saved after scheduler signal; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi

exit "$STATUS"
