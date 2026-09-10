#!/bin/bash
#SBATCH --job-name=paper-exact
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --array=0-53%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+2))p" manifests/paper_exact.csv)
line=${line%$'\r'}
IFS=, read -r method n seed output <<< "$line"
"$HOME/LFHE_workshop/.venv/bin/python" -u main_paper_exact.py --method "$method" --num-clients "$n" --seed "$seed" --output-dir "$output" --data-root "${LFHE_DATA_ROOT:-./data}"
