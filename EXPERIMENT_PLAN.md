# Approval-gated LFHE workshop experiment plan

No stage is submitted automatically. N={100,200,500,1000} defines the workshop scaling claim; N={1500,2000} is optional feasibility evidence only.

## Matrix and gates

| Stage | Experiment | Matrix | Runs | Gate |
|---|---|---:|---:|---|
| 01 | Canonical alignment | N=30; Ring, Static Random, Epidemic, DissDL, LFHE; seeds 42–46 | 25 | Every run complete; canonical config/representation verified; final accuracy inside the recorded historical method range |
| 02 | Five-round feasibility | N=100,500,1000; Static Random/LFHE; participation 1.0/0.1; seed 42 | 12 | Resource and repair thresholds pass before Stage 03 |
| 03 | Twenty-round feasibility | same matrix as Stage 02 | 12 | Resource projection passes before any 300-round scaling array |
| 04 | Primary fixed-degree | N=100,200,500,1000; Static Random/random-FoF/LFHE; D_max=4; seeds 42–44 | 36 | Review three-seed accuracy, variance, runtime, communication, and graph invariants |
| 05 | Core degree sweep (secondary) | N=500,1000; three primary methods; D_max=2,4,8,log2; seeds 42–44 | 72 | Run only after Stage 04 and remaining-compute review |
| 06 | Concurrent/stale views (secondary) | N=100,500; LFHE; bounded/clustered; sequential, snapshot-concurrent, concurrent+5-round stale view; seeds 42–44 | 36 | Select a compact subset before the full matrix |
| 07 | FoF disconnected limit (secondary) | N=100; random-FoF/LFHE; two disconnected clusters; seeds 42–44 | 6 | Components remain disconnected and cross-component FoF proposals remain zero |
| 08 | Link-failure stress (secondary) | N=500; three primary methods; failure 0.1/0.3; seeds 42–44 | 18 | Compare with Stage-04 zero-failure controls; report effective connectivity and recovery |
| 09 | Fixed samples/client (secondary) | N=100,500,1000; three primary methods; 25 samples/client; seeds 42–44 | 27 | Interpret separately from fixed-total data |

Submission-critical total: **73 runs** when Stages 01–04 are all completed, or **61 runs** when Stage 03 is restricted to only configurations needing a longer pilot. Stages 05–09 are secondary, not workshop-submission prerequisites. Prefer one compact concurrency or failure experiment over the full 72-run degree sweep.

Optional, explicitly approved only after prior gates:

| Experiment | Runs |
|---|---:|
| N=1500/2000 five- and twenty-round feasibility | 16 |
| Headline seeds 45–46 for scales selected after Stage 04 | up to 24 |
| N=2000 degree sweep | 36 |

Optional maximum: **76 runs**. Grand maximum: **320 runs**.

## Promotion and stopping thresholds

- Canonical: no downstream array until all 25 outputs pass `validate_stage.py`; classifier weight shape must be `[10,256]`, flattened dimension 2560, and each method's final accuracy must lie inside `historical_canonical_ranges.json`.
- Projected 300-round runtime: at most 48 hours.
- Peak CPU RSS: at most 22.4 GiB under the 28 GiB request.
- Peak GPU reserved memory: at most 90% of allocated device memory.
- Checkpoint: at most 10 GiB and at most 600 seconds.
- Final full-client evaluation: at most three hours.
- Partition repair: at most 5% of selected samples; otherwise mark partition distortion and stop promotion.
- Correctness: no NaN/Inf, negative aggregation weight, degree-cap violation, unexpected disconnection, checkpoint mismatch, duplicate output path, or missing completion artifact.
- N=1500/2000: any threshold failure is reported as a system limit; it does not invalidate the N<=1000 workshop claim.
- Snapshot-concurrent: reject promotion if commits violate D_max or metrics do not account for every proposed change as committed, shared-endpoint conflict, stale rejection, or degree-safety rejection.
- Disconnected FoF: if an edge crosses the original components, treat it as an implementation error; the purpose is to demonstrate the limitation, not repair it silently.

## Approval order

1. Unit tests and N=10 CPU/GPU smoke.
2. Stage 01 only.
3. Review canonical gate output.
4. Stage 02 only, then review.
5. Stage 03 only, then review.
6. Stage 04 seeds 42–44, then review before selecting headline scales/seeds.
7. Review remaining time and select a compact Stage 06, 07, 08, or 09 study.
8. Do not submit Stages 05–09 without explicit post-Stage-04 approval; a compact concurrency or failure study has priority over the full degree sweep.

## First three Mira commands

```bash
mkdir -p logs outputs && python generate_manifests.py && python -m pytest -q
sbatch slurm/stage01_canonical_alignment.sbatch
python validate_stage.py --kind canonical --manifest manifests/stage01_canonical_alignment.csv
```

Run command three only after the Stage-01 array has completed. Do not submit Stage 02 unless its JSON report says `"passed": true` and the scientific curves have been reviewed.
