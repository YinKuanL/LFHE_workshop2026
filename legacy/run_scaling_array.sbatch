#!/bin/bash
#SBATCH --job-name=lfhe-scale
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-35%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue

set -uo pipefail
mkdir -p logs outputs
source .venv/bin/activate
export LFHE_DATA_ROOT="${LFHE_DATA_ROOT:-$PWD/data}"

METHODS=(ring static_random lfhe)
SCALES=(100 300 500 1000)
SEEDS=(42 43 44)

TASK_ID="$SLURM_ARRAY_TASK_ID"
SEED_INDEX=$(( TASK_ID % 3 ))
SCALE_INDEX=$(( (TASK_ID / 3) % 4 ))
METHOD_INDEX=$(( TASK_ID / 12 ))

METHOD="${METHODS[$METHOD_INDEX]}"
NCLIENTS="${SCALES[$SCALE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
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

if [[ "$STATUS" -eq 99 ]]; then
  echo "Checkpoint saved after scheduler signal; requeueing array task $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$STATUS"
