#!/bin/bash
set -uo pipefail
mkdir -p logs
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST"); line=${line%$'\r'}
IFS=, read -r method n seed rounds protocol alpha dmax ti ei graph rate epochs steps batch checkpoint eval_clients final_all update regime samples minimum link stale output <<< "$line"
args=(--method "$method" --num-clients "$n" --seed "$seed" --rounds "$rounds" --protocol "$protocol" --alpha "$alpha" --dmax "$dmax" --topology-interval "$ti" --eval-interval "$ei" --initial-graph "$graph" --participation-rate "$rate" --batch-size "$batch" --checkpoint-interval "$checkpoint" --eval-clients "$eval_clients" --update-mode "$update" --data-regime "$regime" --min-samples-per-client "$minimum" --link-failure-rate "$link" --stale-view-rounds "$stale" --output-dir "$output" --resume --final-eval-all)
if [[ -n "$samples" ]]; then args+=(--samples-per-client "$samples"); fi
if [[ -n "$epochs" ]]; then args+=(--local-epochs "$epochs"); else args+=(--local-steps "$steps"); fi
python -u main.py "${args[@]}"; status=$?
if [[ $status -eq 99 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
if [[ $status -ne 0 ]]; then echo "FAILED status=$status task=$SLURM_ARRAY_TASK_ID" >&2; exit "$status"; fi
