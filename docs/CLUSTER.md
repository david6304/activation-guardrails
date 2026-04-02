# MLP GPU Cluster - Quick Guide

## FIRST TIME SETUP

```bash
# 1. Connect to cluster
ssh s2296274@mlp -J s2296274@student.ssh.inf.ed.ac.uk
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
# Get GPU for testing (2hr max)
srun -p Teaching --gres=gpu:1 --pty bash

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
# Submit job (3 days 8hrs max)
sbatch -p Teaching --gres=gpu:1 run.sh

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

# Activate environment (do this every session)
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate

# Interactive GPU (testing)
srun -p Teaching --gres=gpu:1 --pty bash

# Submit job
sbatch -p Teaching --gres=gpu:1 run.sh

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

## NODES AND GPUS

As of 2026-03-29 (`sinfo -p Teaching`):

| Nodes | GPU | Count | VRAM | Suitable for 7B? |
|---|---|---|---|---|
| `crannog[01-02]` | A40 | 4 per node | 48 GB | Yes — preferred |
| `landonia11` | RTX A6000 | 8 | 48 GB | Yes |
| `saxa` | A100 MIG (1g.18gb / 3g.71gb) | 49 / 2 | 18 / 71 GB | 3g.71gb only, complex to request |
| `damnii[07-12]`, `landonia[03,05,08,23,25]` | RTX 2080 Ti | 8 per node | 11 GB | **No — will OOM on 7B models** |

Max job time: **2 days** (`MaxTime=2-00:00:00`).

**GPU selection in submit scripts:**
- `--constraint` does **not** work on this cluster (returns "Invalid feature specification")
- Use GRES type names directly: `--gres=gpu:a40:1`, `--gres=gpu:nvidia_rtx_a6000:1`
- To get any capable GPU (A40 or A6000): `--gres=gpu:1 --nodelist=crannog[01-02],landonia11`
- Submit scripts handle this via `--gpu-type a40|a6000|any|unrestricted`

---

## CRITICAL RULES

1. NEVER run training on the head node (`hastings`) — use `srun`/`sbatch`
2. Install packages on the head node — compute nodes have no internet
3. Copy large datasets to `/disk/scratch/` for fast training
4. Zip datasets before copying — way faster than many small files
5. No backups on NFS — push important stuff to GitHub (or AFS)
6. Interactive jobs = 2hrs max — for testing/debugging only
7. Batch jobs = 2 days max — save checkpoints to resume
8. One GPU per job — jobs queue faster than multi-GPU requests
9. Never submit with `--gres=gpu:1` alone — will land on a 2080 Ti and OOM for 7B+ models

---

## SCRIPT DESIGN GUIDELINES

When writing cluster-facing scripts, prefer scripts that are reusable and configurable rather than hard-coded for one exact run.

Default design preferences:
- default to queueing for `A40` where GPU type selection is supported
- allow command-line overrides for GPU type and similar cluster/job parameters
- expose switches to run only part of a pipeline when useful, rather than forcing every stage every time

## PROJECT-SPECIFIC TUNING NOTES

- Do not default to very conservative batch sizes for Gemma 2 9B jobs on `A40` / `A6000`. Check utilisation early and tune upward unless there is clear memory pressure.
- Preferred monitoring command for an active job:

```bash
srun --jobid=<JOB_ID> --overlap nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 5
```

- For the refusal-probing `generate_responses.py` stage on `crannog01` (`NVIDIA A40`, 48 GB), an observed run on 2026-04-02 reached `91-98%` GPU utilisation but used only about `19 / 46 GiB` VRAM at batch size `4`. This means the initial setting was too conservative on memory and future runs should start from a larger batch size.
- Default assumption going forward for Gemma 2 9B response generation: try batch size `8` first on `A40` / `A6000`, not `4`, unless a job-specific prompt length or decoding setting suggests otherwise.
- If VRAM usage is still comfortably below capacity and there is no OOM, prefer increasing batch size before changing other aspects of the experiment. If GPU utilisation is already high, expect only moderate speedups; if you need a larger runtime reduction, shortening decode length (`max_new_tokens`) matters more than small batch-size increases.
- expose options to target only a subset of models / layers / datasets / stages when the script naturally supports that
- keep sensible defaults for the common path, but make partial reruns easy

Examples of useful CLI controls:
- GPU type selection such as `--gpu-type a40` with overrides for other supported types
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
srun -p Teaching --gres=gpu:1 --pty bash
python train.py --epochs 1
exit

# Submit
sbatch -p Teaching --gres=gpu:1 run.sh
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
srun -p Teaching --gres=gpu:1 --pty bash
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
