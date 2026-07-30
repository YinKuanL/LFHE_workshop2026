#!/bin/bash
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --array=1-12%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
export MANIFEST=manifests/stage02_feasibility_5round.csv
source slurm/manifest_worker.sh
