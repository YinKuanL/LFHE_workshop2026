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


## Running and monitoring experiments on NCC / Mira

### Before submission

```bash
source .venv/bin/activate
mkdir -p logs outputs
python generate_manifests.py
```

A login node may report that CUDA is unavailable. Confirm GPU access inside an allocation:

```bash
srun --partition=ug-gpu-small --gres=gpu:ampere:1 \
  --cpus-per-task=4 --mem=28G --time=00:10:00 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run a short GPU smoke test first:

```bash
srun --partition=ug-gpu-small --gres=gpu:ampere:1 \
  --cpus-per-task=4 --mem=28G --time=00:20:00 \
  python main.py \
    --protocol scalable \
    --method lfhe \
    --num-clients 10 \
    --seed 42 \
    --rounds 2 \
    --dmax 4 \
    --initial-graph bounded_connected \
    --participation-rate 1 \
    --local-epochs 1 \
    --batch-size 32 \
    --topology-interval 1 \
    --eval-interval 1 \
    --eval-clients 10 \
    --output-dir outputs/gpu_smoke \
    --data-root ./data
```

### Recommended staged execution

Submit and validate stages in order:

```bash
sbatch slurm/stage01_canonical_alignment.sbatch

python validate_stage.py \
  --kind canonical \
  --manifest manifests/stage01_canonical_alignment.csv

sbatch slurm/stage02_feasibility_5round.sbatch

python validate_stage.py \
  --kind feasibility \
  --manifest manifests/stage02_feasibility_5round.csv

# Only when additional evidence is needed:
sbatch slurm/stage03_feasibility_20round.sbatch

# Submit after feasibility passes:
sbatch slurm/stage04_primary_fixed_d4_seeds42_44.sbatch
```

Do not submit every expensive stage at once. Use an array concurrency limit such as `%4` or `%6`.

### Run the complete configured suite

Preview first:

```bash
python main_all_experiments.py \
  --run-all \
  --suite all \
  --results-root "$SCRATCH/LFHE_workshop_results" \
  --data-root "$SCRATCH/LFHE_data" \
  --dry-run
```

Run:

```bash
python main_all_experiments.py \
  --run-all \
  --suite all \
  --results-root "$SCRATCH/LFHE_workshop_results" \
  --data-root "$SCRATCH/LFHE_data"
```

Rerun behavior:

- completed experiments with `SUCCESS` are skipped;
- incomplete experiments with `checkpoint.pt` are resumed;
- new experiments start from round 0;
- `--force` restarts an experiment and removes its previous artifacts.

Never run two processes against the same output directory.

The current suite only includes methods implemented in the repository. Morph must be connected through its actual implementation before being included; Random-FoF must not be used as a substitute.

### Monitor SLURM jobs

```bash
squeue -u "$USER"
watch -n 5 'squeue -u "$USER"'
```

Show useful columns:

```bash
squeue -u "$USER" \
  -o "%.18i %.20j %.2t %.10M %.10l %.12R"
```

Common states:

- `PD`: pending;
- `R`: running;
- `CG`: completing;
- `CD`: completed;
- `F`: failed;
- `CA`: cancelled;
- `TO`: time limit;
- `OOM`: out of memory.

`PD (Priority)` or `PD (Resources)` normally means waiting, not failure.

Inspect one job:

```bash
scontrol show job <JOB_ID>
```

Inspect one array task:

```bash
scontrol show job <ARRAY_JOB_ID>_<TASK_ID>
```

### Follow logs

```bash
ls -lhtr logs | tail
tail -f logs/<log-file>.out
tail -n 100 logs/<log-file>.out
```

Search for failures:

```bash
grep -RniE "error|exception|traceback|out of memory|oom|nan|killed" logs
```

Find the newest logs:

```bash
find logs -type f -printf '%T@ %p\n' | sort -n | tail
```

A quiet stdout file does not always mean the job is stuck because Python output may be buffered. Also check whether `metrics.jsonl` and `checkpoint.pt` timestamps are changing.

### Monitor output progress

Count completed experiments:

```bash
find "$SCRATCH/LFHE_workshop_results" -name SUCCESS | wc -l
```

Count checkpoints:

```bash
find "$SCRATCH/LFHE_workshop_results" -name checkpoint.pt | wc -l
```

List checkpointed but incomplete runs:

```bash
find "$SCRATCH/LFHE_workshop_results" -name checkpoint.pt -print0 |
while IFS= read -r -d '' cp; do
  dir="$(dirname "$cp")"
  test -f "$dir/SUCCESS" || echo "$dir"
done
```

Inspect the latest round:

```bash
tail -n 1 <experiment-output-dir>/metrics.jsonl | python -m json.tool
```

Inspect the final summary:

```bash
python -m json.tool <experiment-output-dir>/summary.json
```

Check storage:

```bash
du -sh "$SCRATCH/LFHE_workshop_results"
du -h --max-depth=2 "$SCRATCH/LFHE_workshop_results" | sort -h | tail
quota -s
```

### Inspect resource usage

After a job finishes or leaves the queue:

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES,MaxRSS,ExitCode
```

For an array:

```bash
sacct -j <ARRAY_JOB_ID> \
  --format=JobID%20,State,Elapsed,MaxRSS,ExitCode
```

While running, where supported:

```bash
sstat -j <JOB_ID>.batch \
  --format=JobID,AveCPU,MaxRSS,AveRSS
```

On an interactive GPU node:

```bash
watch -n 2 nvidia-smi
```

### Safely stop and resume

A running process keeps using the code loaded when it started. Editing `main.py` does not update that running process.

Ask the job to checkpoint:

```bash
scancel --signal=USR1 <JOB_ID>
```

Verify the checkpoint timestamp:

```bash
ls -lh --time-style=long-iso <experiment-output-dir>/checkpoint.pt
```

If it does not exit or requeue automatically:

```bash
scancel <JOB_ID>
```

Resume a single run with exactly the same scientific arguments and output directory:

```bash
python main.py <same arguments as before> \
  --output-dir <same-output-dir> \
  --resume
```

Resume the whole suite by rerunning the same `main_all_experiments.py` command.

Do not resume an old checkpoint if training, data partitioning, aggregation, LFHE fitness, representation, or topology behavior changed. Use a new output root or restart that experiment with `--force`.

### Fast diagnosis checklist

1. Check whether the job is pending or running with `squeue`.
2. Read the scheduler reason with `scontrol show job`.
3. Check whether the log file is growing.
4. Check timestamps of `metrics.jsonl` and `checkpoint.pt`.
5. Inspect `sacct` or `sstat` for memory and exit-code problems.
6. Check quota and checkpoint size.
7. Stop or resubmit only after these checks.

An SSH disconnection does not terminate an `sbatch` job.


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
