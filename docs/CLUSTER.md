# ICF / MLP GPU Cluster - Quick Guide

For this account, `ssh icf` and `ssh mlp` both land on the `hastings` head
node. Treat this as one Slurm environment. The important choice is the
partition, not the SSH alias.

Live access check on 2026-06-02:

- Accessible: `Teaching`, `Interactive`, `Wintermute`.
- Not currently accessible: `ICF-Free`, `Open-Research`, `ICF-Research`
  (`Invalid account or account/partition combination specified` on `sbatch`).
- `sacctmgr show assoc user=$USER ...` shows only the teaching account/QOS, but
  direct `sbatch` tests confirm access to `Teaching`, `Interactive`, and
  `Wintermute`.

## FIRST TIME SETUP

```bash
# 1. Connect to cluster
ssh s2296274@mlp -J s2296274@student.ssh.inf.ed.ac.uk
# Equivalent for this account:
# ssh s2296274@icf -J s2296274@student.ssh.inf.ed.ac.uk
# Enter DICE password twice

# 2. Setup CUDA + Python environment
source /home/htang2/toolchain-20251006/toolchain.rc
python3 -m venv ~/venvs/ml
source ~/venvs/ml/bin/activate

# Install cluster-optimised numpy and PyTorch (faster than pip install from PyPI)
pip install /home/htang2/toolchain-20251006/whl/numpy-2.2.3-cp312-cp312-linux_x86_64.whl
pip install /home/htang2/toolchain-20251006/whl/torch-2.8.0a0+gitunknown-cp312-cp312-linux_x86_64.whl
# Install other packages normally
pip install matplotlib pandas scikit-learn

# 3. Clone your code
git clone https://github.com/yourusername/your-project.git
cd your-project
```

---

## EVERY TIME YOU USE THE CLUSTER

### Step 1: Push code from laptop

```bash
# On your laptop
cd ~/your-project
git add .
git commit -m "update"
git push
```

### Step 2: Connect and pull

```bash
# SSH to cluster
ssh s2296274@mlp -J s2296274@student.ssh.inf.ed.ac.uk

# Get latest code
cd ~/your-project
git pull

# Activate environment
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
```

### Step 3: Create job script

```bash
nano run.sh
```

Paste this template:

```bash
#!/bin/bash
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate

# Optional: use fast disk for large data
# mkdir -p /disk/scratch/s2296274
# cp ~/your-project/data.zip /disk/scratch/s2296274/
# cd /disk/scratch/s2296274 && unzip data.zip

cd ~/your-project
python train.py --epochs 50 --lr 0.001

# Optional: copy results from scratch
# cp /disk/scratch/s2296274/*.pt ~/your-project/results/
# rm -rf /disk/scratch/s2296274/*
```

Save: Ctrl+O, Enter, Ctrl+X

```bash
chmod +x run.sh
```

### Step 4: Test interactively

```bash
# Get GPU for testing (Interactive is short/debug-only)
srun -p Interactive --gres=gpu:1 --time=00:30:00 --pty bash

# Check GPU works
nvidia-smi

# Activate environment
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate

# Test with 1 epoch
cd ~/your-project
python train.py --epochs 1

# Exit when done testing
exit
```

### Step 5: Submit real job

```bash
# Submit job on the Teaching A6000 node (2 days max)
sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 --nodelist=landonia11 run.sh

# Output: Submitted batch job 12345
```

### Step 6: Monitor

```bash
# Check status
squeue -u s2296274

# Watch output
tail -f slurm-12345.out

# Cancel if needed
scancel 12345
```

### Step 7: Get results

```bash
# When job done, check results
ls ~/your-project/results/

# Push to git
cd ~/your-project
git add results/
git commit -m "training done"
git push

exit
```

### Step 8: Pull results to laptop

```bash
# On your laptop
cd ~/your-project
git pull

# Or copy directly
scp -J s2296274@student.ssh.inf.ed.ac.uk s2296274@mlp:~/your-project/results/model.pt ./
```

---

## ESSENTIAL COMMANDS

```bash
# Connect
ssh s2296274@mlp -J s2296274@student.ssh.inf.ed.ac.uk
# or: ssh s2296274@icf -J s2296274@student.ssh.inf.ed.ac.uk

# Activate environment (do this every session)
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate

# Interactive GPU (short debugging)
srun -p Interactive --gres=gpu:1 --time=00:30:00 --pty bash

# Submit job
sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 --nodelist=landonia11 run.sh

# Check jobs
squeue -u s2296274

# Cancel job
scancel 12345

# See all partitions and nodes
sinfo
```

---

## FILE SYSTEM

| Location | What | Persists? | Speed |
|---|---|---|---|
| `~/` | Your code, venv, results | Yes | Slow (NFS) |
| `/disk/scratch/s2296274/` | Training data (temp) | No | Fast (local disk) |
| `/afs/inf.ed.ac.uk/user/.../s2296274/` | Backups only | Yes + backed up daily | Slow |

Rule: Keep code in `~/`, copy big datasets to `/disk/scratch/` for training. Zip datasets before copying — moving one big file is much faster than many small files.

## MODEL WEIGHTS

Compute nodes have no internet access. HuggingFace models must be downloaded on the head node (`hastings`) before submitting jobs.

```bash
# Download a model on the head node (has internet)
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
HF_HOME=~/models python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto')
"
```

- `~/models` is the persistent model cache (`HF_HOME`) — weights survive across jobs
- `/disk/scratch/` is for activations and intermediate data only — do not cache models there (ephemeral)
- Pipeline scripts set `HF_HOME=~/models` unconditionally; override with `--model-cache` if needed
- **Always set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in job scripts.** Compute nodes have no internet; without these, `AutoTokenizer.from_pretrained` will hang indefinitely trying to reach HuggingFace Hub after the model weights have already loaded. Symptoms: model loads fine, GPU shows 0% utilisation, no further output. Pipeline scripts set both automatically.

---

## PARTITION CHOICE

Use this decision rule:

| Partition | Use for | Avoid for | Max time | Current hardware seen |
|---|---|---|---|---|
| `Wintermute` | Serious LLM jobs: Heretic, Gemma generation, activation extraction, large batches | Casual smoke tests or jobs that fit cheaply elsewhere | 5 days | `wintermute`, 2x A100 80GB PCIe reported by Slurm, 112 CPUs, 755 GiB RAM, 3.5T scratch |
| `Teaching` | Normal batch jobs and fallback GPU work, especially A6000 on `landonia11` | Bare `--gres=gpu:1` for 4B+ LLMs | 2 days | mostly 2080 Ti 11GB, plus A6000 on `landonia11`, A100 MIG on `saxa` |
| `Interactive` | Short interactive debugging and CUDA sanity checks | Long jobs, large models | 4 hours | `landonia[01-02]`, 2080 Ti 11GB |

Recommended commands:

```bash
# Best current route for serious single-GPU LLM work.
srun -p Wintermute --gres=gpu:1 --time=04:00:00 --pty bash
sbatch -p Wintermute --gres=gpu:1 --time=1-00:00:00 run.sh

# Only request both A100s if the code genuinely uses multi-GPU.
srun -p Wintermute --gres=gpu:2 --time=01:00:00 --pty bash

# Teaching A6000 route. Avoid bare gpu:1 for large models.
srun -p Teaching --gres=gpu:nvidia_rtx_a6000:1 --nodelist=landonia11 --time=02:00:00 --pty bash
sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 --nodelist=landonia11 --time=1-00:00:00 run.sh

# Cheap short debugging; these are 11GB GPUs.
srun -p Interactive --gres=gpu:1 --time=00:30:00 --pty bash
```

Check live availability before submitting:

```bash
sinfo -N -p Teaching,Interactive,Wintermute \
  -O partition,nodelist:20,statecompact,gres:80,cpus:10,memory:12
squeue -p Teaching,Interactive,Wintermute
```

Confirm access with tiny jobs if anything changes:

```bash
for p in Teaching Interactive Wintermute; do
  echo "== $p =="
  sbatch -p "$p" --time=00:01:00 --wrap='hostname && nvidia-smi -L || true'
done
```

## NODES AND GPUS

As of 2026-06-02 (`sinfo -N -p Teaching,Interactive,Wintermute ...`):

| Partition | Nodes | GPU | Count | VRAM | Suitable for 4B+/7B LLM work? |
|---|---|---|---|---|---|
| `Wintermute` | `wintermute` | A100 80GB PCIe | 2 reported by Slurm; `--gres=gpu:1` exposes one GPU | 80 GB each | **Yes — preferred for serious LLM jobs** |
| `Teaching` | `landonia11` | RTX A6000 | 8 | 48 GB | Yes |
| `Teaching` | `saxa` | A100 MIG (`1g.18gb`, `3g.71gb`) | 49 / 2 | 18 / 71 GB | 3g.71gb only, complex to request |
| `Teaching` | `damnii[07-12]`, `landonia[03,05,08,23,25]` | RTX 2080 Ti | 8 per node | 11 GB | **No for 4B+/7B unless tiny/quantized** |
| `Interactive` | `landonia[01-02]` | RTX 2080 Ti | 8 per node | 11 GB | Debugging only |

Max job times:

- `Wintermute`: **5 days** (`MaxTime=5-00:00:00`)
- `Teaching`: **2 days** (`MaxTime=2-00:00:00`)
- `Interactive`: **4 hours** (`MaxTime=04:00:00`)

**GPU selection in submit scripts:**
- `--constraint` does **not** work on this cluster (returns "Invalid feature specification")
- Use GRES type names directly: `--gres=gpu:nvidia_rtx_a6000:1`,
  `--gres=gpu:nvidia_a100_80gb_pcie:1`, or the MIG names shown by live `sinfo`
- For Wintermute, `--gres=gpu:1` worked interactively and exposed one A100 80GB
- For Teaching A6000: `--gres=gpu:nvidia_rtx_a6000:1 --nodelist=landonia11`
- Submit scripts handle this via `--gpu-type wintermute|a6000|any`

---

## CRITICAL RULES

1. NEVER run training on the head node (`hastings`) — use `srun`/`sbatch`
2. Install packages on the head node — compute nodes have no internet
3. Copy large datasets to `/disk/scratch/` for fast training
4. Zip datasets before copying — way faster than many small files
5. No backups on NFS — push important stuff to GitHub (or AFS)
6. `Interactive` partition jobs = 4hrs max — for testing/debugging only
7. `Teaching` batch jobs = 2 days max; `Wintermute` jobs = 5 days max — save checkpoints to resume
8. One GPU per job — jobs queue faster than multi-GPU requests
9. Never submit to `Teaching` with `--gres=gpu:1` alone for 4B+/7B models — it will usually land on an 11GB 2080 Ti and OOM

---

## SCRIPT DESIGN GUIDELINES

When writing cluster-facing scripts, prefer scripts that are reusable and configurable rather than hard-coded for one exact run.

Default design preferences:
- default serious single-GPU LLM runs to `Wintermute` where access and queue
  pressure allow
- otherwise default capable Teaching jobs to `landonia11` with
  `--gres=gpu:nvidia_rtx_a6000:1`
- allow command-line overrides for GPU type and similar cluster/job parameters
- expose switches to run only part of a pipeline when useful, rather than forcing every stage every time

## CC++ GEMMA 3 OFFLINE GENERATION

Compute nodes have no internet, so before any offline GPU job, cache everything
on the head node (`hastings`):

```bash
python scripts/cluster/prefetch_hf.py
```

This caches `google/gemma-3-4b-it` plus Heretic's evaluation datasets
(`mlabonne/harmless_alpaca`, `mlabonne/harmful_behaviors`) into `~/models`, so
Heretic can run offline on a compute node. Abliteration is interactive (it ends
with a save menu), so run it under `srun --pty`, not `sbatch`:

```bash
srun -p Wintermute --gres=gpu:1 --time=04:00:00 --pty bash
export HF_HOME=~/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
heretic google/gemma-3-4b-it   # save to ~/models/gemma-3-4b-it-heretic
```

Completion generation is a batch job; submit it with the configurable wrapper
(`--dry-run` to preview, `--gpu-type`, `--stage`, `--limit` for smoke tests):

```bash
scripts/cluster/submit_generate_completions.sh --dry-run
scripts/cluster/submit_generate_completions.sh --stage both
```

## PROJECT-SPECIFIC TUNING NOTES

- Do not default to very conservative batch sizes for Gemma jobs on `A100 80GB`
  / `A6000`. Check utilisation early and tune upward unless there is clear
  memory pressure.
- Preferred monitoring command for an active job:

```bash
srun --jobid=<JOB_ID> --overlap nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 5
```

- Historical A40 notes from `crannog01` may still be useful as rough
  utilisation guidance, but `crannog` is not currently in the accessible
  partitions for this account.
- Historical Gemma 2 9B jobs used batch size `8` as a reasonable first smoke
  test on A100 80GB / A6000-class GPUs. Do not transfer that default to the
  current Gemma 3 4B pipeline; use the measured guidance below.
- Measured 2026-06-02, `gemma-3-4b-it-heretic` completion generation on
  `Wintermute` A100 80GB, `max_new_tokens=512`, left-padded batched HF
  `generate`: **batch 64 used ~18 GiB and only ~52% GPU-util** — heavily
  under-utilised. Weights ~8 GiB, so KV cache costs ~0.16 GiB/prompt at this
  decode length. Rule of thumb for a 4B model at 512 tokens on 80 GB: cache
  budget ≈ (0.85·80 − 8) ≈ 62 GiB ⇒ ~390 prompts, so **start at batch 256**
  (margin for the longest-prompt batch) and only back off on OOM. The submit
  script default is now `256`. Scale this estimate with model weights and
  `max_new_tokens` for other jobs.
- If VRAM usage is still comfortably below capacity and there is no OOM, prefer increasing batch size before changing other aspects of the experiment. If GPU utilisation is already high, expect only moderate speedups; if you need a larger runtime reduction, shortening decode length (`max_new_tokens`) matters more than small batch-size increases.
- expose options to target only a subset of models / layers / datasets / stages when the script naturally supports that
- keep sensible defaults for the common path, but make partial reruns easy

Examples of useful CLI controls:
- GPU type selection such as `--gpu-type wintermute|a6000|any` with overrides
  for other supported types
- stage controls such as `--train-only`, `--eval-only`, or `--stage train`
- subset controls such as `--models qwen`, `--layers 8 16 24`, or similar
- dry-run / print-only options for checking the exact command or `sbatch` submission before launching
- output / artifact path options so runs do not overwrite each other accidentally

Practical rule:
- optimise for scripts that make the default path easy
- but avoid designs where changing one small thing requires editing the script body

This matters on the cluster because queue times, GPU availability, and rerun costs are real constraints. Partial reruns and resource overrides should usually be CLI arguments, not manual code edits.

---

## EXAMPLE: COMPLETE SESSION

```bash
# LAPTOP
cd ~/my-project
git push

# CLUSTER
ssh s2296274@mlp -J s2296274@student.ssh.inf.ed.ac.uk
cd ~/my-project && git pull
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate

# Test
srun -p Interactive --gres=gpu:1 --time=00:30:00 --pty bash
python train.py --epochs 1
exit

# Submit
sbatch -p Wintermute --gres=gpu:1 run.sh
# Job 12345 submitted

# Monitor
squeue -u s2296274
tail -f slurm-12345.out

# Get results
git add results/ && git commit -m "done" && git push
exit

# LAPTOP
git pull
```

---

## TROUBLESHOOTING

**"No CUDA available"**
```bash
# Did you activate toolchain?
source /home/htang2/toolchain-20251006/toolchain.rc
# Did you request GPU?
srun -p Interactive --gres=gpu:1 --time=00:30:00 --pty bash
```

**"Import torch is slow"**
- Normal on NFS (10–30 seconds), only happens once per script

**"Job pending forever"**
```bash
squeue -u s2296274
# If PD (pending), wait or cancel and retry
```

**"Out of memory"**
- Reduce batch size

**"Need to resume training after timeout"**
- Save checkpoints every epoch and load last checkpoint at start of script

---

## QUICK START CHECKLIST

- [ ] Get cluster access (computing support form)
- [ ] SSH and setup venv (first time only)
- [ ] Clone your repo to `~/`
- [ ] Create `run.sh` job script
- [ ] Test with `srun` interactively
- [ ] Submit with `sbatch`
- [ ] Monitor with `squeue` and `tail`
- [ ] Pull results back to laptop
