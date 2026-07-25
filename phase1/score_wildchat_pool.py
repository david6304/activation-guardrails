"""Score the frozen C3 WildChat background pool with one detector.

One detector per invocation so each is a short, resumable job. The probe and
centroid reuse the frozen Phase 1 parameters; the guards reuse the same runners
as the Phase 1 matrix. WildChat's length tail is far heavier than
WildJailbreak's, so every detector sees prompts left-truncated to the same
2048-token budget the probe already applies (`iter_position_batches`).

WildChat is unlabelled: the resulting quantity is a background alert rate, not
an FPR.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from guard_screen import run_shieldgemma
from phase1.extend_multilingual_guards import resolve_cached_snapshot
from phase1.phase1_activation import score_detectors
from phase1.score_modern_guards import run_qwen3guard
from probe_prompt import load_model, strings_sha256, truncate_left_tokens

GEMMA_MODEL = "google/gemma-3-27b-it"
GEMMA_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
GUARDS = {
    "shieldgemma": ("google/shieldgemma-9b", run_shieldgemma),
    "qwen3guard": ("Qwen/Qwen3Guard-Gen-8B", run_qwen3guard),
}
MAX_TOKENS = 2048


def load_prompts(path, condition, swahili_path, limit):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if condition == "swahili":
        translations = {
            row["id"]: row["translation"]
            for line in open(swahili_path)
            if line.strip()
            for row in [json.loads(line)]
        }
        missing = [row["id"] for row in rows if row["id"] not in translations]
        if missing:
            raise ValueError(f"{swahili_path} is missing {len(missing)} translations")
        for row in rows:
            row["text"] = translations[row["id"]]
    else:
        for row in rows:
            row["text"] = row["prompt"]
    if limit:
        rows = rows[:limit]
    return rows


def score_probe(rows, source_path, batch_size, seed):
    with np.load(source_path, allow_pickle=False) as saved:
        weight = saved["logistic_weight"]
        intercept = float(saved["logistic_intercept"])
        harmful = saved["harmful_centroid"]
        harmless = saved["harmless_centroid"]
    model, tokeniser, num_layers, hidden = load_model(GEMMA_MODEL, seed)
    revision = str(getattr(model.config, "_commit_hash", "") or "")
    if revision != GEMMA_REVISION:
        raise RuntimeError(f"loaded Gemma revision {revision!r} != {GEMMA_REVISION!r}")
    if weight.shape != ((num_layers + 1) * hidden,):
        raise ValueError("frozen logistic weight does not match the model")
    logistic, centroid = score_detectors(
        [row["text"] for row in rows],
        model,
        tokeniser,
        batch_size,
        weight,
        intercept,
        harmful,
        harmless,
        "t_inst",
    )
    return {"logistic": logistic, "centroid": centroid}, revision


def score_guard(rows, detector, batch_size):
    model_id, runner = GUARDS[detector]
    snapshot, revision = resolve_cached_snapshot(model_id)
    from transformers import AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(snapshot)
    truncated = 0
    guard_rows = []
    for text in truncate_left_tokens(
        [row["text"] for row in rows], tokeniser, MAX_TOKENS
    ):
        guard_rows.append({"text": text})
    for row, guard_row in zip(rows, guard_rows):
        truncated += int(guard_row["text"] != row["text"])
    runner(guard_rows, str(snapshot), batch_size)
    scores = {detector: np.asarray([row["score"] for row in guard_rows], dtype=np.float32)}
    if detector == "qwen3guard":
        scores["qwen3guard_controversial"] = np.asarray(
            [row["controversial"] for row in guard_rows], dtype=np.float32
        )
        scores["qwen3guard_verdict"] = np.asarray(
            [row["verdict"] for row in guard_rows], dtype=np.int32
        )
    else:
        scores[f"{detector}_flag"] = np.asarray(
            [bool(row["flag"]) for row in guard_rows], dtype=np.bool_
        )
    return scores, revision, truncated, [row["text"] for row in guard_rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="data/c3_wildchat_prompts.jsonl")
    parser.add_argument("--swahili", default="data/c3_wildchat_swahili.jsonl")
    parser.add_argument("--condition", choices=("plain", "swahili"), default="plain")
    parser.add_argument(
        "--detector",
        choices=("probe", *GUARDS),
        required=True,
    )
    parser.add_argument("--source", default="data/phase1_activation_27b.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    rows = load_prompts(args.prompts, args.condition, args.swahili, args.limit)
    print(f"[input] {len(rows)} prompts, condition={args.condition}", flush=True)

    if args.detector == "probe":
        scores, revision = score_probe(rows, args.source, args.batch_size, args.seed)
        truncated = None
        scored_texts = [row["text"] for row in rows]
    else:
        scores, revision, truncated, scored_texts = score_guard(
            rows, args.detector, args.batch_size
        )
    for name, values in scores.items():
        if len(values) != len(rows) or not np.isfinite(
            values.astype(np.float64)
        ).all():
            raise ValueError(f"invalid {name} scores")

    metadata = {
        "prompts": args.prompts,
        "condition": args.condition,
        "detector": args.detector,
        "model_revision": revision,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "max_tokens": MAX_TOKENS,
        "truncated_to_max_tokens": truncated,
        "n": len(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray([row["id"] for row in rows]),
        **{f"{args.condition}_{name}": values for name, values in scores.items()},
        metadata_json=np.asarray(json.dumps(metadata)),
        inputs_sha256=np.asarray(strings_sha256(scored_texts)),
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
