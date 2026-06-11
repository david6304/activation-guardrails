# ICF / MLP Cluster Guide

Operational guidance for this repository. Cluster availability and hardware
change: verify live state before relying on dated observations.

For this account, `ssh icf` and `ssh mlp` have both reached the `hastings` head
node. Select resources by Slurm partition, not SSH alias.

## Access Baseline

Last checked 2026-06-02:

- accessible: `Teaching`, `Interactive`, and `Wintermute`;
- unavailable to this account: `ICF-Free`, `Open-Research`, and
  `ICF-Research`;
- direct tiny `sbatch` checks were more reliable than inferred account/QOS
  listings.

Recheck before a consequential run:

```bash
sinfo -N -p Teaching,Interactive,Wintermute \
  -O partition,nodelist:20,statecompact,gres:80,cpus:10,memory:12
squeue -p Teaching,Interactive,Wintermute
```

Use a one-minute job to confirm uncertain access. Do not infer current access
from this document alone.

## Connect And Activate

```bash
ssh s2296274@mlp -J s2296274@student.ssh.inf.ed.ac.uk
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
```

The recorded environment used Python 3.12 with cluster-provided NumPy and
PyTorch wheels. Verify that the toolchain path and wheels still exist before
recreating it. Install packages on the head node; compute nodes may not have
internet access.

Keep the cluster checkout at an explicit commit:

```bash
cd ~/activation-guardrails
git fetch
git checkout <COMMIT>
git status --short
```

Do not run reportable work from an unidentified or dirty checkout.

## Partition Selection

Last observed 2026-06-02:

| Partition | Appropriate use | Observed limit/hardware |
| --- | --- | --- |
| `Interactive` | short debugging and CUDA checks | 4 hours; 2080 Ti 11 GB |
| `Teaching` | normal batch work when an appropriate GPU is requested | 2 days; mixed GPUs, including A6000 and smaller cards |
| `Wintermute` | memory-intensive single-GPU LLM work | 5 days; A100 80 GB observed |

Examples:

```bash
# Short interactive validation
srun -p Interactive --gres=gpu:1 --time=00:30:00 --pty bash

# Memory-intensive single-GPU job
sbatch -p Wintermute --gres=gpu:1 --time=1-00:00:00 run.sh

# Teaching A6000, if still advertised
sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 \
  --nodelist=landonia11 --time=1-00:00:00 run.sh
```

Never request bare `--gres=gpu:1` on a heterogeneous partition when the job has
a known VRAM minimum. Inspect live GRES names and request a capable device.
Request multiple GPUs only when the code is explicitly multi-GPU.

## Storage And Model Caches

| Location | Use |
| --- | --- |
| `~/` | persistent checkout, environment, small metadata |
| `~/models` | persistent Hugging Face cache |
| `/disk/scratch/$USER/` | fast, ephemeral data and intermediates |

- Keep irreplaceable metadata on persistent storage.
- Treat scratch as disposable.
- Avoid moving thousands of small files when one archive or shard is suitable.
- Do not commit model weights, gated datasets, activation caches, or generated
  harmful content.

Prefetch every required model and dataset on a networked node before submitting
an offline job. Set offline flags in compute jobs:

```bash
export HF_HOME=~/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

An offline smoke test should load the exact pinned model/tokenizer/data from
cache before a long run is submitted.

## Job Script Contract

A cluster job should:

- use `set -euo pipefail`;
- activate the intended toolchain and environment;
- change to the repository checkout explicitly;
- print hostname, timestamp, commit, dirty status, command, and environment
  versions;
- accept config, output path, and resource-sensitive values as arguments;
- write to a unique run directory;
- checkpoint or append atomically when practical;
- fail early if required caches or inputs are absent.

Minimal shape:

```bash
#!/usr/bin/env bash
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HOME=~/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

date --iso-8601=seconds
hostname
git rev-parse HEAD
git status --short

python <entrypoint> --config <config> --output-dir <unique-run-dir>
```

Prefer repository submission wrappers with `--dry-run` or print-only behaviour.
Do not edit a script body for each run when the value belongs in a config or
command-line argument.

## Preflight Before Submission

Validate locally or in a short interactive allocation:

1. targeted tests and Ruff checks when the milestone contains Python;
2. entrypoint `--help` when an entrypoint exists;
3. config parsing and path validation;
4. offline model/tokenizer/data loading;
5. a tiny end-to-end or dry run;
6. output metadata and resume behaviour;
7. the exact rendered `sbatch` command.

The cluster is not the first-pass debugger. A GPU-only step may remain, but
imports, arguments, config, paths, and failure messages should be checked first.

## Monitoring

```bash
squeue -u "$USER"
tail -f slurm-<JOB_ID>.out
scancel <JOB_ID>
```

For GPU utilization inside an allocation:

```bash
nvidia-smi
srun --jobid=<JOB_ID> --overlap nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 5
```

Tune batch size from a small measured pilot for the exact model, sequence
length, precision, and feature extraction mode. Historical measurements from a
different experiment are not defaults.

## Results And Recovery

- Write exploratory and reportable runs to distinct run directories.
- Copy required outputs from scratch before job expiry.
- Preserve run metadata even when large artifacts remain local.
- Do not push generated outputs indiscriminately; commit only compact,
  explicitly selected report artifacts.
- Long jobs must checkpoint often enough to recover within the partition time
  limit.

## Common Failures

- **No CUDA:** confirm the toolchain is active and the shell is inside a GPU
  allocation.
- **Pending job:** inspect `squeue` reason and current partition/node state.
- **Out of memory:** reduce batch/sequence size or request a GPU meeting the
  measured VRAM requirement.
- **Offline hang:** confirm all dependencies are cached and offline environment
  variables are set.
- **Timeout:** resume from a validated checkpoint rather than overwriting the
  prior run.
