# LFHE Mira experiment suite

The single official experiment entry point is `main.py`. It supports the canonical LFHE protocol and scalable Mira runs, and drives the real Morph implementation through `MorphNode` from `morph.py`. Do not substitute Random-FoF or another topology method for Morph.

## Repository layout

- `main.py`: official experiment runner.
- `morph.py`, `lfhe.py`, `dissdl.py`, `epidemic.py`: topology implementations. Morph uses the node protocol `begin_round`, `update_wanted_senders`, payload exchange, and `aggregate`.
- `generate_mira_manifest.py`: deterministic Core/All manifest generator.
- `run_manifest_row.py`: executes one zero-based CSV data row.
- `manifests/mira_core.csv`: 194 experiment rows.
- `manifests/mira_all.csv`: 413 experiment rows; its first 194 rows reuse Core output directories.
- `slurm/run_core_mira.sbatch`: array indices `0-193`.
- `slurm/run_all_mira.sbatch`: array indices `0-412`.
- `validate_stage.py`, `tests/`, and the stage manifests/scripts: staged validation workflow retained for reproducibility.
- `legacy/`: superseded runners and submission scripts retained for historical reference only.

## Environment

Use a cluster-supported Python environment with PyTorch, torchvision, NumPy, NetworkX, SciPy, psutil, and pytest. Stage CIFAR-10 before compute-node execution if compute nodes have no internet access. Set `LFHE_DATA_ROOT` to the staged dataset directory when needed.

Generate the checked-in manifests reproducibly:

```bash
python generate_mira_manifest.py --level core --output manifests/mira_core.csv
python generate_mira_manifest.py --level all --output manifests/mira_all.csv
```

## Submit on Mira

```bash
sbatch slurm/run_core_mira.sbatch
sbatch slurm/run_all_mira.sbatch
```

Core contains 194 runs. All contains 413 runs and deliberately reuses the same output directories for its Core subset. Do not run Core and All concurrently, because two tasks must never write to the same output directory.

Each array task passes its zero-based `SLURM_ARRAY_TASK_ID` to `run_manifest_row.py`. `csv.DictReader` removes the header, so Core indices `0-193` select all 194 data rows and All indices `0-412` select all 413 data rows without skipping the first experiment or reading the header.

## Completion and resume

- `SUCCESS` means the run completed.
- `checkpoint.pt` without `SUCCESS` means the run is incomplete and can resume.
- Existing `SUCCESS` directories are skipped.
- Existing incomplete checkpoint directories receive `--resume`.
- Never allow two jobs to write to one output directory.

Dataset, outputs, results, logs, checkpoints, caches, job IDs, model artifacts, result arrays, and local archives are excluded from Git. Keep these on project or scratch storage, not in commits.

## Validation

Before submission or publication, run:

```bash
python -m py_compile main.py lfhe.py morph.py dissdl.py epidemic.py run_manifest_row.py generate_mira_manifest.py validate_stage.py
pytest -q
bash -n slurm/run_core_mira.sbatch
bash -n slurm/run_all_mira.sbatch
```

The older stage manifests and SLURM scripts remain in place because `validate_stage.py`, tests, and `EXPERIMENT_PLAN.md` still use that staged workflow. They are reproducibility assets, not alternative official entry points.
