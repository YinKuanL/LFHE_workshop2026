# LFHE Workshop 2026 Scaling Experiments

This repository studies when bounded local topology evolution remains effective as decentralized learning systems scale.

## Method Overview

LFHE uses local friends-of-friends discovery and bounded topology transactions to evolve a communication graph without a central optimizer. The workshop suite evaluates how this mechanism behaves as client count, data regime, degree budget, graph mixing, participation, and link reliability change.

![LFHE topology evolution overview](results/figures/lfhe_topology_evolution_overview.png)

![Morph and LFHE topology update comparison](results/figures/lfhe_morph_comparison.png)

## Key Results

The repository now tracks compact CSV summaries aggregated from complete frozen `summary.json` outputs in the local workshop snapshot. These are public, lightweight summaries only; raw datasets, checkpoints, logs, and bulk generated outputs remain untracked.

| Table | Scope |
|---|---|
| [results/tables/scaling_fixed_total.csv](results/tables/scaling_fixed_total.csv) | Fixed-total N={50,100,200,500} complete summaries where available |
| [results/tables/scaling_fixed_per_client.csv](results/tables/scaling_fixed_per_client.csv) | Fixed-samples-per-client summaries for verified completed runs |
| [results/tables/degree_scaling.csv](results/tables/degree_scaling.csv) | Static-random degree-sweep controls found in the frozen snapshot |
| [results/tables/graph_diagnostics.csv](results/tables/graph_diagnostics.csv) | Final graph degree, clustering, spectral-gap, and LFHE transaction diagnostics |
| [results/tables/large_n_controls.csv](results/tables/large_n_controls.csv) | Large-N static-random control summaries found in the frozen snapshot |

## Main Result Table

Selected fixed-total summaries from `scaling_fixed_total.csv`:

| Method | N | Degree budget | Completed seeds | Final accuracy mean | Normalized AUC mean |
|---|---:|---:|---|---:|---:|
| DissDL | 50 | 4 | 42;43;44;45;46 | 0.740376 | 0.638655 |
| Epidemic | 50 | 4 | 42;43;44;45;46 | 0.754318 | 0.649799 |
| Static Random | 50 | 4 | 46 | 0.720476 | 0.616024 |
| DissDL | 100 | 4 | 42;43;44;45;46 | 0.691472 | 0.572236 |
| Epidemic | 100 | 4 | 42;43;44;45;46 | 0.696443 | 0.578910 |
| LFHE | 100 | 4 | 42;43;44;45 | 0.675151 | 0.557260 |
| Morph | 100 | 4 | 42;43;44;45;46 | 0.705163 | 0.594443 |
| Random-FoF | 100 | 4 | 42;43;44;45;46 | 0.646677 | 0.538270 |
| Static Random | 100 | 4 | 42;43;44;45;46 | 0.671432 | 0.555069 |

Seed coverage is explicit because the frozen snapshot contains partial coverage for some methods. Use the CSV files, not this excerpt, as the complete tracked summary.

## Main Figures

The tracked figures are method/design figures:

- [results/figures/lfhe_topology_evolution_overview.png](results/figures/lfhe_topology_evolution_overview.png)
- [results/figures/lfhe_morph_comparison.png](results/figures/lfhe_morph_comparison.png)

No generated scaling plot is promoted unless its source and finalized status can be verified. The current numerical evidence is published as CSV tables instead.

## Key Findings

- The fixed-total snapshot contains complete five-seed summaries for DissDL, Epidemic, and Static Random at N=100/200/500, plus complete or partial coverage for LFHE, Morph, and Random-FoF at N=100.
- The fixed-per-client snapshot contains verified completed Static Random and Random-FoF controls for N=100/200/500, but not a complete LFHE fixed-per-client headline suite.
- The degree-sweep table currently represents verified Static Random controls, not a full LFHE degree-ablation claim.
- Graph diagnostics expose final graph structure and LFHE transaction counters where those fields are present in the frozen summaries.

## Reproduction

Use a Python environment with PyTorch, torchvision, NumPy, NetworkX, SciPy, psutil, and pytest. Stage CIFAR-10 before compute-node execution if compute nodes have no internet access. Set `LFHE_DATA_ROOT` to the staged dataset directory when needed.

```bash
python -m py_compile main.py lfhe.py morph.py dissdl.py epidemic.py run_manifest_row.py generate_mira_manifest.py validate_stage.py
python -m pytest -q
```

Generate the checked-in Core and All manifests reproducibly:

```bash
python generate_mira_manifest.py --level core --output manifests/mira_core.csv
python generate_mira_manifest.py --level all --output manifests/mira_all.csv
```

Submit the main Mira suites:

```bash
sbatch slurm/run_core_mira.sbatch
sbatch slurm/run_all_mira.sbatch
```

Core and All must not run concurrently because they intentionally share output directories for the Core subset.

## Repository Structure

- `main.py`: official experiment runner.
- `morph.py`, `lfhe.py`, `dissdl.py`, `epidemic.py`, `lfhe_pac.py`: topology implementations and baselines.
- `generate_mira_manifest.py`, `run_manifest_row.py`: deterministic manifest generation and row execution.
- `manifests/`: Core, All, workshop, and staged experiment manifests.
- `slurm/`: Mira/NCC submission scripts and manifest workers.
- `scripts/`: manifest generation, validation, summarization, and topology-animation utilities.
- `tests/`: regression tests for manifests, validation contracts, and topology utilities.
- `legacy/`: superseded runners and submission scripts retained for historical reference only.
- `results/tables/`: curated lightweight result summaries.
- `results/figures/`: curated README method/design figures.

## Result Provenance

See [docs/results_provenance.md](docs/results_provenance.md). Generated datasets, checkpoints, result arrays, logs, plots, scheduler outputs, and local archives are intentionally excluded from Git.

## Citation and License

No author-identifying citation or publication-status statement is included here while review constraints are uncertain. Add citation and license information only when compatible with the submission policy for this repository.
