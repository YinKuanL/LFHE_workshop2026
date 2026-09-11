# Results Provenance

This page records the evidence chain for the public README, curated figures, and lightweight result tables.

## Curated README Figures

| Figure | Source | Use |
|---|---|---|
| `results/figures/lfhe_topology_evolution_overview.png` | Copied unchanged from `figures/lfhe_topology_evolution_overview.png`, originally copied from `images/overview.png` | LFHE topology-evolution method overview |
| `results/figures/lfhe_morph_comparison.png` | Copied unchanged from `figures/lfhe_morph_comparison.png`, originally copied from `images/morph_lfhe.png` | Morph/LFHE topology-update comparison |

Both figures are method/design figures. They are not used as numeric scaling-result evidence.

## Curated Result Tables

All CSV tables under `results/tables/` were generated from complete `summary.json` files in the local frozen workshop snapshot `tmp/ncc_snapshot_20260812_082235/lfhe_results_snapshot_20260812_082235/outputs/`. The snapshot itself remains untracked because it contains bulk generated experiment outputs.

| Table | Snapshot suites | Notes |
|---|---|---|
| `results/tables/scaling_fixed_total.csv` | `workshop_headline_d4_initheadroomv2` | Fixed-total summaries for complete runs in the frozen snapshot. The table includes completed seed IDs and completion counts. |
| `results/tables/scaling_fixed_per_client.csv` | `workshop_fixed_per_client` | Fixed-samples-per-client summaries for verified completed runs. The snapshot contains Static Random and Random-FoF controls, not a complete LFHE headline suite. |
| `results/tables/degree_scaling.csv` | `workshop_degree_sweep` | Degree-sweep controls found in the frozen snapshot. The verified rows are Static Random controls, so the README does not present them as an LFHE degree ablation. |
| `results/tables/graph_diagnostics.csv` | `workshop_headline_d4_initheadroomv2`, `workshop_degree_sweep`, `workshop_large_scale` | Final graph degree, clustering, spectral-gap, candidate-packet, and transaction summaries where present. |
| `results/tables/large_n_controls.csv` | `workshop_large_scale` | Large-N static-random control summaries found in the frozen snapshot. |

For each run, the aggregation uses only summaries with `status == complete`. Numeric means are computed over the completed seeds listed in each row and written with six decimal places. No draft manuscript TODO placeholders are used as result evidence.

## Verified Tracked Provenance

- `manifests/mira_core.csv`: 194 experiment rows.
- `manifests/mira_all.csv`: 413 experiment rows; first 194 rows reuse Core output directories.
- `manifests/workshop_main_shared_static_init_n50_500.csv`: 120-row N={50,100,200,500}, seeds 42-46 shared-initial-topology comparison.
- `EXPERIMENT_PLAN.md`: staged approval gates, stopping thresholds, promotion rules, and optional feasibility studies.
- `validate_stage.py` and `tests/`: manifest, validation-contract, topology-invariant, and checkpoint/resume checks.

## Result Boundary

Generated datasets, checkpoints, result arrays, logs, plots, scheduler outputs, and local archives are intentionally excluded from Git. The current public evidence includes curated method/design figures and compact CSV summaries only. Any future promoted scaling figure should record the exact raw-output suite, plotting command, manifest rows, seed coverage, and completion markers used to create it.
