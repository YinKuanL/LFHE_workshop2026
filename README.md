# LFHE workshop scalability experiments

This runner separates the reported CIFAR-10 experiment from new large-scale deployment experiments. `--protocol canonical` is the alignment path; `--protocol scalable` exposes execution changes. Results from partial participation or local steps are **not** directly comparable to the original 30-client experiment.

## Protocols

Canonical defaults are CIFAR-10, fixed total data, Dirichlet alpha 0.1, 300 rounds, one complete local epoch, batch 32, SGD at 0.05, topology/evaluation every 5 rounds, D_max=4, LFHE weights (1, 1, 0.1), all-client evaluation, and full participation. LFHE uses the canonical exponential beta annealing, sequential FoF add/swap procedure, degree-aware Metropolis aggregation, and flattened final-classifier weights. Seeds 42–46 are in `manifests/canonical_alignment.csv`.

Scalable mode keeps the model and LFHE objective but stores client state dictionaries on CPU and reuses one GPU model. Intermediate evaluation may use a fixed subset; the final evaluation defaults to all clients. `--participation-rate < 1` is opt-in and is labeled `partial_participation` in configuration and summary output. Inactive clients neither train nor aggregate in that round; only active-to-active graph neighbors transmit models. D_max remains 4 unless explicitly set to 2, 8, or `log2`.

Initial graph choices are `canonical_er` (connected Erdos–Rényi, expected degree four; over-cap nodes are logged) and `bounded_connected` (randomized ring plus bounded chords, connected with max degree at most D_max). Static Random and LFHE deterministically receive the same graph for the same seed/configuration.

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
sbatch slurm/canonical_alignment.sbatch
```

The manifests are:

- `canonical_alignment.csv`: N=30, five methods, seeds 42–46.
- `fixed_degree_scaling_pilot.csv`: N=25,50,100,200,500, six methods, seeds 42–44.
- `n1000_feasibility.csv`: Static Random, random FoF, LFHE, seed 42, explicitly 10% participation.
- `degree_sweep.csv`: N=100,200,500 and D_max 2,4,8,log2 for three methods and seeds 42–44.

Each manifest row has an independent output directory. Change the SLURM array upper bound to `data rows`, never run two tasks against one output directory.

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

Known limitations: canonical mode uses the historical splitting algorithm but deterministically resamples if it produces an empty client; `--min-samples-per-client` can request a larger minimum. Partial-participation aggregation is a separate deployment protocol. DissDL has O(N²) peer-directory metadata. Checkpoint files at N=1000 can be large. The local Windows validation environment used to prepare this repository did not contain PyTorch/NetworkX, so model smoke tests must be run in the documented project environment.

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
