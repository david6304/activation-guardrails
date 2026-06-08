# WildJailbreak CC++ Protocol

Checkpoint C0 freezes machine-checkable choices before source auditing,
generation, labelling, activation extraction, or training. The active source is
`docs/WILDJAILBREAK_CCPP_IMPLEMENTATION_PLAN.md`; runtime evidence in
`protocol.v1.json` overrides assumptions.

## Frozen Contracts

- Dataset: `allenai/wildjailbreak` at immutable revision
  `5ddc12a7894f842b0619b8e1c7ee496b198af009`.
- Loader: `datasets.load_dataset` config/split `train`, with
  `delimiter="\t"` and `keep_default_na=false`. C0 verifies the pinned local
  cache, exact columns, and fingerprint; C1 owns row counts and pairing.
- Duplicate detection: SHA-256 exact matching, token 5-gram Jaccard `>= 0.80`,
  and cosine similarity `>= 0.95` from
  `sentence-transformers/all-MiniLM-L6-v2` revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Primary label: complete-exchange harmful enablement under
  `exchange_harmfulness_rubric.v1`. Source intent and refusal are diagnostics,
  not replacements for the primary label.
- Exchange classifier: single-token `YES`/`NO` log odds only after tokenizer
  verification; otherwise normalized full-label-sequence likelihood. Frozen
  prompt-logit classification is the declared lower-faithfulness PEFT fallback.

The fixed splits, accepted-label minima, seeds, metrics, duplicate policy,
fallbacks, and component classifications are in
`configs/ccpp/wildjailbreak_gemma3_4b_exchange.yaml`.

## Required Evidence

The refusal-ablated protected artifact must provide an exact model id, immutable
revision, deterministic weight hash, runtime model class, hidden size, layer
count, hidden-state count and ordering, maximum sequence length, chat-template
hash, and verified assistant-token boundaries.

The independent judge must have a distinct identity and immutable revision plus
positive access evidence. A credential variable is recorded only as a
credential indicator. It is never accepted as service access. If judge access
is unavailable, the frozen path is to stop before C4 and manually adjudicate
primary labels.

## Validator Semantics

Run:

```bash
python scripts/ccpp/validate_protocol.py \
  --config configs/ccpp/wildjailbreak_gemma3_4b_exchange.yaml
```

The validator always writes:

- `artifacts/ccpp/<run-id>/protocol/protocol.v1.json`
- `artifacts/ccpp/<run-id>/protocol/environment.v1.json`

Exit code `0` means every mandatory check passed. Exit code `2` means the
artifacts were written but at least one mandatory fact is blocked or failed.
Blocked output is the expected result while protected-model or judge evidence
is unavailable.

The environment artifact records the command, timestamp, Git commit and dirty
state, Python/OS/hardware details, package versions, `pip check`, Torch/CUDA/
cuDNN/MPS details, GPU/driver evidence where available, and credential
indicators with an explicit non-access warning. The protocol artifact freezes
raw and canonical config/rubric hashes, the judge-prompt hash, environment hash,
checks, evidence, and protocol-component classifications.

`--evidence` accepts a small JSON metadata object for externally collected
preflight evidence. The validator may hash explicitly listed local weight files
but does not download model weights.

## C0 Boundary

C0 does not audit source pair counts, create duplicate graphs, build splits or
manifests, generate exchanges, run a judge, extract activations, train probes or
classifiers, calibrate thresholds, or evaluate systems.
