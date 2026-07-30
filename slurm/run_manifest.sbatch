#!/bin/bash
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --array=1-25%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
MANIFEST=${MANIFEST:?submit with --export=ALL,MANIFEST=manifests/<file>.csv}
mkdir -p logs
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
line=${line%$'\r'}
IFS=, read -r method n seed rounds protocol alpha dmax ti ei graph rate epochs steps batch checkpoint eval_clients final_all update output <<< "$line"
args=(--method "$method" --num-clients "$n" --seed "$seed" --rounds "$rounds" --protocol "$protocol" --alpha "$alpha" --dmax "$dmax" --topology-interval "$ti" --eval-interval "$ei" --initial-graph "$graph" --participation-rate "$rate" --batch-size "$batch" --checkpoint-interval "$checkpoint" --eval-clients "$eval_clients" --update-mode "$update" --output-dir "$output" --resume --final-eval-all)
if [[ -n "$epochs" ]]; then args+=(--local-epochs "$epochs"); else args+=(--local-steps "$steps"); fi
python -u main.py "${args[@]}"
status=$?
if [[ $status -eq 99 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
if [[ $status -ne 0 ]]; then echo "FAILED status=$status task=$SLURM_ARRAY_TASK_ID" >&2; exit "$status"; fi
