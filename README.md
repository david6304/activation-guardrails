# Activation Guardrails

MSc dissertation research on whether internal language-model activations can
support guardrails that are more robust and interpretable than text-only
methods. The project studies three broad questions:

- **Robustness:** how activation-based detectors behave under jailbreaks and
  distribution shift.
- **Harmfulness and refusal:** whether these are distinct internal signals and
  what a detector is actually using.
- **SAE interpretability:** whether sparse autoencoder features help explain
  safety-related detection.

## Setup

1. Clone this repository.
2. Restore the local research-material folders (`papers/`, `course-docs/`,
   `docs/references/`, and `archive/`). These are intentionally not tracked.
3. Clone the separate `msc-writeup` repository into `msc-writeup/`.
4. Use the `msc-diss` conda environment. Python dependencies and deliberate
   version pins are recorded in `requirements.txt`; the cluster-provided PyTorch
   and NumPy builds must not be overwritten.
5. See `docs/CLUSTER.md` before running or changing any cluster job.

Quick local checks:

```bash
conda run -n msc-diss python --version
conda run -n msc-diss ruff check <changed-python-files>
conda run -n msc-diss python -m py_compile <changed-python-files>
conda run -n msc-diss python <entrypoint>.py --help
```

There is no single full test suite. Validate a change at the smallest real
boundary it affects, then use a tiny real-model smoke before a reportable run.

## Structure

```text
README.md          project overview and setup
RESEARCH_LOG.md    important research decisions and findings
docs/CLUSTER.md    cluster guidance
requirements.txt   pinned Python dependencies (excluding cluster PyTorch/NumPy)
phase1/            frozen Phase 1 scripts, launchers, and detailed results
*.py               shared helpers and active later-phase experiment scripts
run_*.sh           launchers for shared or active later-phase work
data/              local inputs, outputs, manifests, and compact result artifacts
papers/            local papers and literature notes (not tracked)
course-docs/       local course guidance (not tracked)
docs/references/   supplementary reference material (not tracked)
archive/           historical project material (not tracked)
msc-writeup/       separate dissertation write-up repository
```

### papers/

- `raw/` — PDFs named `NN_author_year_shortname.pdf` (numbered entries) or
  `arxivid-author-shortname.pdf` (later additions).
- `notes/` — Markdown notes mirroring `raw/` naming. `lit-review-index.md` is
  the master index. `sae-safety.md` and
  `sae4safety-zhao-cross-representational-safety-mechanisms.md` are standalone
  deep-dive notes on key references. Notes contain strategic summaries (claims,
  results, dissertation relevance); read the PDF when you need exact formulas or
  hyperparameters.

### course-docs/

- `diss/` and `ipp/` — university guidance for the dissertation and IPP.

### Experiment scripts

The code intentionally remains a small collection of scripts rather than a
general experiment framework. Frozen phase-specific entrypoints and launchers
live in their phase folder; shared helpers stay at the repository root. Use the
newest entries in `RESEARCH_LOG.md` to identify the active experiment. Treat the
matching launcher as the source for its reportable command and output paths;
inspect the Python entrypoint's `--help` before changing arguments.

The definitive Phase 1 analysis and write-up record is
[`phase1/RESULTS.md`](phase1/RESULTS.md). Run its entrypoints from the repository
root, for example:

```bash
conda run -n msc-diss python -m phase1.analyse_phase1 --help
```
