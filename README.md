# LFHE workshop scalability experiments

This runner separates the reported CIFAR-10 experiment from new large-scale deployment experiments. `--protocol canonical` is the alignment path; `--protocol scalable` exposes execution changes. Results from partial participation or local steps are **not** directly comparable to the original 30-client experiment.

## Protocols

Canonical defaults are CIFAR-10, fixed total data, Dirichlet alpha 0.1, 300 rounds, one complete local epoch, batch 32, SGD at 0.05, topology/evaluation every 5 rounds, D_max=4, LFHE weights (1, 1, 0.1), all-client evaluation, and full participation. LFHE uses the historical post-round-20 update window, canonical exponential beta annealing, sequential FoF add/swap procedure, degree-aware Metropolis aggregation, and flattened final-classifier weights. The model directly logs shape `[10,256]` and dimension 2560; this is checked against the original source and historical results.

Scalable mode keeps the model and LFHE objective but stores client state dictionaries on CPU and reuses one GPU model. Intermediate evaluation may use a fixed subset; the final evaluation defaults to all clients. `--participation-rate < 1` is opt-in and is labeled `partial_participation` in configuration and summary output. Inactive clients neither train nor aggregate in that round; only active-to-active graph neighbors transmit models. D_max remains 4 unless explicitly set to 2, 8, or `log2`.

Initial graph choices are `canonical_er`, `bounded_connected`, hard connected `clustered`, and `disconnected_clusters`. The last option explicitly demonstrates that FoF discovery cannot create an edge between disconnected components. Static Random and LFHE deterministically receive the same graph for the same seed/configuration.

Workshop extensions are separate experiments: `--update-mode snapshot_concurrent` proposes against one immutable snapshot and deterministically commits conflict-free changes; it records shared-endpoint conflicts, stale-topology rejection, degree-safety rejection, and committed proposals. `--stale-view-rounds` selects an older classifier representation view. `--link-failure-rate` drops communication links from the aggregation view without silently changing the stored topology. `--data-regime fixed_per_client --samples-per-client 25` provides a second regime where local sample count does not shrink with N.

The DissDL baseline deliberately records an all-client known-peer directory of size N-1 per client. This is global metadata, not local control, and `--dissdl-max-n` can disable it at large N. Fully Connected is not offered. FedAvg is only a centralized reference.

## Mira / NCC setup

Use a virtual environment with PyTorch 2.4.1+cu121 (or a cluster-supported compatible CUDA build), torchvision, NumPy, NetworkX, SciPy, and psutil. Compute nodes may not have internet access, so stage CIFAR-10 under `--data-root` first. The supplied SLURM script uses NCC's required typed request:

```text
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
```

Generate/check manifests with `python generate_manifests.py`. Ready-to-submit array scripts carry the correct bounds, for example:

```bash
mkdir -p logs
sbatch slurm/stage01_canonical_alignment.sbatch
```

The submission-critical manifests are approval-gated:

- `stage01_canonical_alignment.csv`: 25 N=30 canonical runs.
- `stage02_feasibility_5round.csv`: 12 five-round N=100/500/1000 resource pilots.
- `stage03_feasibility_20round.csv`: up to 12 twenty-round pilots, limited to configurations needing more evidence after Stage 02.
- `stage04_primary_fixed_d4_seeds42_44.csv`: 36 primary N=100/200/500/1000 runs.

These total 73 runs when all Stage-03 rows are needed, or 61 when Stage 03 is restricted. Stage 04 is the primary workshop scaling claim.

Secondary manifests, selected only after Stage 04 and remaining-compute review:
- `stage05_degree_sweep_500_1000.csv`: 72 degree-budget runs.
- `stage06_concurrent_and_stale_views.csv`: 36 sequential/concurrent, clustered, and stale-view runs.
- `stage07_fof_disconnected_limit.csv`: 6 explicit disconnected-component runs.
- `stage08_link_failure_stress.csv`: 18 link-failure runs; zero-failure controls reuse Stage 4.
- `stage09_fixed_samples_per_client.csv`: 27 constant-local-data runs.

Prioritize one compact concurrency or failure experiment over the full 72-run degree sweep. Stages 05–09 are scientifically valuable but are not submission prerequisites.

Optional manifests cover N=1500/2000 feasibility, N=2000 degree scaling, and headline seeds 45–46. They are not part of the minimum workshop claim and require explicit approval.

Each manifest row has an independent output directory. No script submits a later stage. After Stage 1, run `python validate_stage.py --kind canonical --manifest manifests/stage01_canonical_alignment.csv`; after feasibility stages use `--kind feasibility`. Review the report before explicitly submitting the next array.

Feasibility promotion requires: projected 300-round runtime no more than 48 hours; peak CPU RSS no more than 22.4 GiB; GPU reserved memory below 90% of device capacity; checkpoint no more than 10 GiB and 600 seconds; full evaluation no more than three hours; partition repair fraction no more than 5%; and no NaN, degree-safety, checkpoint, or completion failure. N=1500/2000 failures are reported as system limits rather than weakening the N≤1000 headline.

## Smoke tests

Tiny CPU runs (CIFAR-10 must already be available, or allow its download):

```bash
python main.py --protocol scalable --method static_random --num-clients 10 --seed 42 --rounds 2 --dmax 4 --initial-graph bounded_connected --participation-rate 1 --local-epochs 1 --batch-size 32 --topology-interval 5 --eval-interval 1 --eval-clients 10 --output-dir outputs/smoke_static --data-root ./data
python main.py --protocol scalable --method lfhe --num-clients 10 --seed 42 --rounds 2 --dmax 4 --initial-graph bounded_connected --participation-rate 1 --local-epochs 1 --batch-size 32 --topology-interval 1 --eval-interval 1 --eval-clients 10 --output-dir outputs/smoke_lfhe --data-root ./data
```

GPU smoke command (submit on a GPU node; this is not a full experiment):

```bash
srun --partition=ug-gpu-small --gres=gpu:ampere:1 --cpus-per-task=4 --mem=28G python main.py --protocol scalable --method lfhe --num-clients 10 --seed 42 --rounds 2 --dmax 4 --initial-graph bounded_connected --participation-rate 1 --local-epochs 1 --batch-size 32 --topology-interval 1 --eval-interval 1 --eval-clients 10 --output-dir outputs/gpu_smoke
```

## Checkpoints, signals, and outputs

Round-level checkpoints are atomically replaced and include client states, exact split, graph, metrics, RNG states, config/hash, sampler state, and method state. `SIGUSR1` or `SIGTERM` produces a safe checkpoint and exit code 99; the SLURM wrapper requeues and passes `--resume`. A `SUCCESS` run is skipped unless `--force` is supplied. Changing scientific configuration causes a resume hash mismatch.

Every completed directory contains `config.json`, `checkpoint.pt`, `metrics.jsonl`, `summary.json`, `graph_initial.edgelist`, `graph_final.edgelist`, and `SUCCESS`.

Metrics include complete evaluation curves and client accuracy dispersion; normalized AUC and rounds/bytes/time to target; model and control communication; phase and round timings; sparse graph structure/spectral metrics; LFHE proposal outcomes; consensus/representation variance; and CPU/GPU memory. Large graphs use sampled path length and SciPy sparse eigensolvers, never dense eigendecomposition.

Resource use is dominated by N CPU model states and checkpoints (roughly N times one serialized CNN state), while GPU model memory stays approximately constant. Full-client CIFAR-10 evaluation and full participation are computationally expensive at N=500–1000. Exact runtime, checkpoint size, and RSS must be measured by the pilot; no performance results are claimed here.

Known limitations: canonical mode uses the historical splitting algorithm but deterministically resamples if it produces an empty client; `--min-samples-per-client` can request a larger minimum. Increasing N under fixed-total data also reduces samples/client, so it is not interpreted as topology scaling alone; Stage 9 supplies the control regime. Partial participation is a separate deployment protocol. DissDL has O(N²) peer-directory metadata. N=1500/2000 remain optional until pilots pass. The local Windows execution environments found during preparation lacked a working complete PyTorch stack, so model smoke tests must be run in the documented Mira/project environment.

## Representative commands

Canonical N=30 LFHE validation:

```bash
python main.py --protocol canonical --method lfhe --num-clients 30 --seed 42 --output-dir outputs/canonical/lfhe_n30_seed42
```

N=100 scaling smoke:

```bash
python main.py --protocol scalable --method lfhe --num-clients 100 --seed 42 --rounds 5 --dmax 4 --initial-graph bounded_connected --participation-rate 1 --local-epochs 1 --batch-size 32 --topology-interval 5 --eval-interval 5 --eval-clients 20 --output-dir outputs/smoke_n100
```

N=1000 feasibility deployment:

```bash
python main.py --protocol scalable --method lfhe --num-clients 1000 --seed 42 --rounds 300 --dmax 4 --initial-graph bounded_connected --participation-rate 0.1 --local-epochs 1 --batch-size 32 --topology-interval 5 --eval-interval 5 --eval-clients 50 --final-eval-all --checkpoint-interval 25 --output-dir outputs/n1000/lfhe_n1000_seed42 --resume
```
