# Consolidated results (living table)

Update as runs land. All on protected `gemma-3-it`. Raw data paths noted per block.
Freeze target ~end July 2026. See `RESEARCH_LOG.md` for method/decisions.

## 1. Capability — decode retention on WebQuestions (acc / plain-acc), N=150

| transform     | 12B ret | 27B ret | mechanism |
|---------------|--------:|--------:|-----------|
| plain (base)  | 100% (acc 50.7%) | 100% (acc 54.0%) | — |
| french        | 96% | 100% | NLLB translation |
| swahili       | 87% | 99% | NLLB translation |
| hindi         | 85% | 88% | NLLB translation |
| zulu          | 72% | 89% | NLLB translation |
| reverse       | 36% | 51% | char reversal + decode scaffold |
| vowel-removal | —   | 61% | fallback cipher |
| piglatin      | —   | 2%  | fallback (model-blind → control) |
| nato          | 0%  | 1%  | serial-CoT decode → predicted-fail control |
| morse         | 0%  | 0%  | serial-CoT decode → predicted-fail control |

27B ≥ 12B on every transform (left-shift, as claim 2 predicts). Caveat: retention is a
WebQuestions trivia proxy; `capability_qa.py:42` substring-matches short golds — hand-validation
with paired CIs still pending. Raw: Eddie `data/cap_qa_langs_{12b,27b}.jsonl`;
MLP `data/cap_qa_{27b,12b}.jsonl` (jobs 3543515/3543526) + `data/cap_qa_27b.jsonl` (3557511, fallbacks).

## 2. Public-guard recall / FPR — WJ-harmful vs XSTest-safe

Languages n=150 (job 3562722); ciphers n=50 (jobs 3541577, 3557500). WildGuard = dedicated
prompt-harm classifier; ShieldGemma-9b = composite-guideline P(Yes).

| transform | WildGuard recall | WG FPR | ShieldGemma recall | SG FPR |
|-----------|-----------------:|-------:|-------------------:|-------:|
| plain     | 96.7% | 0.7% | 62.0% | 12.7% |
| french    | 88.0% | 0.0% | 56.7% | 11.3% |
| hindi     | 16.0% | 1.3% | 50.7% | 12.0% |
| swahili   |  0.0% | 0.0% | 38.7% |  6.7% |
| zulu      |  0.0% | 0.7% | 13.3% |  5.3% |
| reverse   |  0.0% | 0.0% |  0.0% |  0.0% |
| nato      |  0.0% | 0.0% |  0.0% |  0.0% |
| morse     |  0.0% | 0.0% |  0.0% |  0.0% |
| vowel     |  4.0% | 2.0% | 22.0% |  2.0% |
| piglatin  |  4.0% | 0.0% | 10.0% |  4.0% |

WildGuard shows graded blindness by language resource level (plain → french → hindi → swahili/zulu);
both guards ≈0% on ciphers. **Swahili = capability/detectability dissociation**: 99% model decode
(27B) but 0% WildGuard recall. Raw: MLP `data/guard_screen_langs.jsonl`, `data/guard_screen.jsonl`.

## 3. Probe — plain-trained all-layer concat, transfer to test conditions

Threshold = 0.1% WildChat alert rate. **Probe currently scores plain / reverse / zulu only**
(`probe_prompt.py` test conditions); french/hindi/swahili probe points for the language coupling
axis are NOT yet implemented.

| transform | 12B AUROC | 12B TPR | 27B AUROC | 27B TPR |
|-----------|-----------|---------|-----------|---------|
| plain     | _pending_ | | _pending_ | |
| reverse   | _pending_ | | _pending_ | |
| zulu      | _pending_ | | _pending_ | |

Jobs: Eddie 56990400 (12B), Wintermute 3563857 (27B). Reverse pilot (per-layer, n=300, raw WJ
labels): plain-val AUROC ~0.997; plain→reverse transfer peaks **0.955 @ layer 34/62**.
Raw (when done): `data/probe_prompt_{model}_report.json` + `_scores.npz`.
