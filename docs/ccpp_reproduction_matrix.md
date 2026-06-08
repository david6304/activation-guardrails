# CC++ WildJailbreak Reproduction Matrix

This matrix is the human-readable view of the C0 classifications frozen in
`configs/ccpp/wildjailbreak_gemma3_4b_exchange.yaml`.

| Component | Local contract | Classification | C0 status |
| --- | --- | --- | --- |
| Protected model | Refusal-ablated Gemma 3 4B artifact used for generation and activations | `pending_evidence` | Exact identity, immutable revision, weights, and runtime contract required |
| WildJailbreak source | Pinned public paired harmful/benign source | `substitute` | Revision and loader contract frozen; pairing/count audit belongs to C1 |
| Exchange harmfulness target | Relevant, valid, meaningful harmful enablement in the complete exchange | `faithful` | Rubric and judge-prompt hashes frozen |
| Independent judge | Judge differs from protected model and trainable classifier | `pending_evidence` | Identity, immutable revision, and positive access evidence required |
| Activation probe | Streaming hidden-state probe | `faithful` | Runtime hidden-state evidence pending protected-model access |
| Layer set | All runtime transformer hidden layers | `faithful` | Layer count pending; reduced-layer fallback must freeze before dev |
| SWiM window | Complete sliding windows of length 16 | `faithful` | Declared for C6, not implemented in C0 |
| EMA gamma | Fixed `0.1` | `our_default` | Primary-source value remains unverified |
| Exchange classifier | Gemma YES/NO classifier with PEFT preference | `substitute` | Token strategy and lower-faithfulness fallback frozen |
| Ensemble weights | Include `0.55` probe and `0.45` classifier analogue | `faithful` | Declared for development selection in C10 |
| Static evaluation | Grouped TPR at 1% FPR and ROC-AUC | `substitute` | Production VDR is unavailable |

Every row must use exactly one of `faithful`, `substitute`, `our_default`, or
`pending_evidence`. `scripts/ccpp/validate_protocol.py` rejects missing or
unknown classifications.
