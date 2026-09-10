# Results Provenance

This page records what the README can claim from tracked repository evidence.

## Curated README Figures

| Figure | Source | Use |
|---|---|---|
| `figures/lfhe_topology_evolution_overview.png` | Copied from `images/overview.png` | LFHE topology-evolution method overview |
| `figures/lfhe_morph_comparison.png` | Copied from `images/morph_lfhe.png` | Morph/LFHE topology-update comparison |

Both figures are method/design figures. They are not used as numeric scaling-result evidence.

## Verified Tracked Provenance

- `manifests/mira_core.csv`: 194 experiment rows.
- `manifests/mira_all.csv`: 413 experiment rows; first 194 rows reuse Core output directories.
- `manifests/workshop_main_shared_static_init_n50_500.csv`: 120-row N={50,100,200,500}, seeds 42-46 shared-initial-topology comparison.
- `EXPERIMENT_PLAN.md`: staged approval gates, stopping thresholds, promotion rules, and optional feasibility studies.
- `validate_stage.py` and `tests/`: manifest, validation-contract, topology-invariant, and checkpoint/resume checks.

## Result Boundary

The repository does not currently track final numeric scaling-result summaries or submission-ready result plots. Draft-manuscript TODO placeholders must not be treated as final results.

Generated datasets, checkpoints, result arrays, logs, plots, scheduler outputs, and local archives are intentionally excluded from Git. When final workshop results are ready for publication, add only curated lightweight figures and tables whose source outputs can be mapped back to the exact manifest rows, seeds, command lines, environment, and completion markers.
