# LFHE 1000-client Mira experiment

## Files

- `lfhe_scalable_experiment.py`: scalable CPU-state / single-GPU runner.
- `run_lfhe_1000.sbatch`: one LFHE 1000-client run.
- `run_scaling_array.sbatch`: 36 runs = 3 methods × 4 scales × 3 seeds.

## Why this version is different

The old script creates one CUDA model per client. This version stores all client
state dictionaries in CPU RAM and uses one reusable CUDA model. It also avoids
full evaluation of all 1000 clients every five rounds.

The default 1000-client protocol is:

- CIFAR-10, Dirichlet alpha = 0.1 with minimum-size repair;
- 10% participation (100 active clients per round);
- one local minibatch per active client;
- maximum degree 4;
- topology update every 10 rounds;
- evaluation on a fixed 50-client subset every 10 rounds;
- checkpoint every 25 rounds.

This is a new large-scale partial-participation protocol. It should not be mixed
with the original 30-client, 100%-participation numbers without clearly stating
the protocol difference.

## First smoke test

```bash
python lfhe_scalable_experiment.py \
  --method lfhe \
  --num-clients 1000 \
  --rounds 5 \
  --seed 42 \
  --participation-rate 0.1 \
  --eval-interval 5 \
  --checkpoint-interval 2 \
  --output-dir outputs/smoke_lfhe_n1000_seed42 \
  --resume
```

Check GPU visibility before submitting:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## Submit one run

```bash
sbatch run_lfhe_1000.sbatch
```

Override settings without editing the file:

```bash
METHOD=static_random SEED=43 NCLIENTS=1000 sbatch run_lfhe_1000.sbatch
```

## Submit the full scaling array

```bash
sbatch run_scaling_array.sbatch
```

The array contains:

- methods: `ring`, `static_random`, `lfhe`;
- clients: `100`, `300`, `500`, `1000`;
- seeds: `42`, `43`, `44`;
- maximum six simultaneous jobs.

## Checkpoint and resume behavior

Each output directory contains:

```text
checkpoint.pt
metrics.json
SUCCESS
```

`checkpoint.pt` stores:

- next communication round;
- every client model state;
- topology edges;
- client data split;
- metrics;
- Python, NumPy, CPU Torch, and CUDA RNG states.

Saving is atomic: a temporary file is written and then renamed. `--resume`
loads `checkpoint.pt` automatically. The SLURM files request `SIGUSR1` three
minutes before termination. Python saves a checkpoint and returns code 99; the
wrapper then requeues the same job.

A normal internet/SSH disconnection does not stop a submitted SLURM job. The
checkpoint mechanism is primarily for time limits, preemption, node failure, or
manual cancellation after a scheduler signal.

## Important storage note

A 1000-client checkpoint is large because it contains 1000 model states. Use a
project/scratch filesystem with sufficient quota. Do not write checkpoints to a
small home directory. Set `--output-dir` to the appropriate Mira storage path.


## Submit, monitor, stop, and resume

### Submit

```bash
sbatch run_lfhe_1000.sbatch
METHOD=lfhe SEED=42 NCLIENTS=1000 sbatch run_lfhe_1000.sbatch
mkdir -p logs
sbatch run_scaling_array.sbatch
```

Record the returned job ID.

### Monitor

```bash
squeue -u "$USER"
watch -n 5 'squeue -u "$USER"'
squeue -u "$USER" -o "%.18i %.20j %.2t %.10M %.10l %.12R"
scontrol show job <JOB_ID>
```

`PD (Priority)` or `PD (Resources)` means waiting, not failure.

### Follow logs

```bash
ls -lhtr logs | tail
tail -f logs/<log-file>.out
tail -n 100 logs/<log-file>.out
grep -RniE "error|exception|traceback|out of memory|oom|nan|killed" logs
```

### Check progress and storage

```bash
find outputs -name SUCCESS | wc -l
find outputs -name checkpoint.pt | wc -l
du -sh outputs
du -h --max-depth=2 outputs | sort -h | tail
quota -s
```

For the newer runner:

```bash
tail -n 1 <output-dir>/metrics.jsonl | python -m json.tool
python -m json.tool <output-dir>/summary.json
```

### Resource accounting

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES,MaxRSS,ExitCode
```

For arrays:

```bash
sacct -j <ARRAY_JOB_ID> \
  --format=JobID%20,State,Elapsed,MaxRSS,ExitCode
```

### Safely pause and resume

A running process continues using the code loaded when it started.

```bash
scancel --signal=USR1 <JOB_ID>
ls -lh --time-style=long-iso <output-dir>/checkpoint.pt
```

If needed:

```bash
scancel <JOB_ID>
```

Resume with identical arguments and the same output directory:

```bash
python lfhe_scalable_experiment.py <same arguments> \
  --output-dir <same-output-dir> \
  --resume
```

Do not start two jobs that write to the same output directory.

A run is complete only when `SUCCESS` exists. A directory with `checkpoint.pt` but no `SUCCESS` is incomplete and should be resumed.


## Recommended order

1. 5-round smoke test at N=1000.
2. 20-round pilot for `static_random` and `lfhe`, seed 42.
3. One complete LFHE seed.
4. Full array only after runtime and storage are confirmed.
