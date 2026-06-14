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
4. See `docs/CLUSTER.md` when cluster compute is needed.

## Structure

```text
README.md          project overview and setup
RESEARCH_LOG.md    important research decisions and findings
docs/CLUSTER.md    cluster guidance
papers/            local papers and literature notes
course-docs/       local course guidance
archive/           historical project material
msc-writeup/       separate dissertation write-up repository
```

Code directories and dependencies will be added only when the current research
task requires them.
