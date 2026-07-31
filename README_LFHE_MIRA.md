# LFHE on Mira

See `README.md` for the authoritative workflow. The only official Python entry point is `main.py`.

Submit either suite with:

```bash
sbatch slurm/run_core_mira.sbatch
sbatch slurm/run_all_mira.sbatch
```

Core has 194 runs. All has 413 runs and reuses Core's output directories for the shared rows, so the two arrays must not run concurrently. A `SUCCESS` marker means completion; `checkpoint.pt` without `SUCCESS` means the task can resume. Dataset files, outputs, checkpoints, and scheduler logs are local experiment artifacts and must not be committed.

Superseded single-run and scaling-array scripts are retained under `legacy/` for historical reference. They are not supported execution entry points.
