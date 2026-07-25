"""Score the frozen Phase 1 split's six conditions with Qwen3Guard-Gen-8B.

Qwen3Guard-Gen emits a three-way verdict whose first line is
`Safety: Safe|Unsafe|Controversial`. Teacher-forcing the prefix `Safety:` puts
the decision at one position, so a single forward pass gives the continuous
label-token probability the Phase 1 operating-point protocol needs; the three
branches start with distinct first tokens, so the softmax over them is exact.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from phase1.extend_multilingual_guards import CONDITIONS, resolve_cached_snapshot
from phase1.extend_multilingual_tfidf import load_translations
from phase1.phase1_baselines import build_conditions, guard_rows, unpack_guard
from probe_prompt import load_judged_rows, split_rows, strings_sha256


QWEN_MODEL = "Qwen/Qwen3Guard-Gen-8B"
DECISION_PREFIX = "Safety:"
LABELS = (" Safe", " Unsafe", " Controversial")
AUDIT_INDICES = (0, 254, 508, 763, 1017, 1272, 1526, 1780)
AUDIT_CONDITIONS = ("plain", "swahili", "reverse")
AUDIT_TOLERANCE = 1e-3
EXPECTED_SPLIT = (5341, 1781, 1781)


def label_token_ids(tokeniser):
    ids = []
    for label in LABELS:
        encoded = tokeniser(label, add_special_tokens=False)["input_ids"]
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"verdict branches share a first token: {ids}")
    return ids


def run_qwen3guard(rows, model_path, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(model_path)
    tokeniser.padding_side = "left"
    ids = label_token_ids(tokeniser)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to("cuda")
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            tokeniser.apply_chat_template(
                [{"role": "user", "content": row["text"]}], tokenize=False
            )
            + DECISION_PREFIX
            for row in batch
        ]
        encoded = tokeniser(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, ids]
        probabilities = torch.softmax(logits.float(), dim=-1)
        for row, probability in zip(batch, probabilities.tolist()):
            row["score"] = probability[1]
            row["controversial"] = probability[2]
            row["verdict"] = int(np.argmax(probability))
        if start % (batch_size * 50) == 0 or start + batch_size >= len(rows):
            print(
                f"  qwen3guard {min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    del model
    torch.cuda.empty_cache()
    return tokeniser


def generation_check(rows, model_path, batch_size):
    """Confirm the forced-prefix verdict is what the model would freely generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(model_path)
    tokeniser.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to("cuda")
    model.eval()
    agree = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            tokeniser.apply_chat_template(
                [{"role": "user", "content": row["text"]}], tokenize=False
            )
            for row in batch
        ]
        encoded = tokeniser(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokeniser.pad_token_id,
            )
        new = generated[:, encoded["input_ids"].shape[1] :]
        for row, ids in zip(batch, new):
            text = tokeniser.decode(ids, skip_special_tokens=True)
            forced = LABELS[row["verdict"]].strip()
            match = f"Safety: {forced}" in text
            agree += int(match)
            print(f"  gen={text.splitlines()[0]!r} forced={forced} match={match}")
    print(f"[generation check] agreement={agree}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()


def audit_rows(rows):
    """The 48 frozen audit cells, as independent single-example rescoring rows."""
    lookup = {(row["split"], row["condition"], row["id"]): row for row in rows}
    selected = []
    for split in ("tune", "test"):
        for condition in AUDIT_CONDITIONS:
            for index in AUDIT_INDICES:
                selected.append(dict(lookup[(split, condition, index)]))
    return selected


def batch_audit(rows, rescored):
    lookup = {(row["split"], row["condition"], row["id"]): row for row in rows}
    differences = {}
    for row in rescored:
        key = (row["split"], row["condition"], row["id"])
        differences[f"{row['split']}:{row['condition']}:{row['id']}"] = abs(
            float(row["score"]) - float(lookup[key]["score"])
        )
    maximum = max(differences.values())
    return {
        "comparison": "batched versus batch-size-1 rescore of the same rows",
        "tolerance": AUDIT_TOLERANCE,
        "maximum_difference": maximum,
        "differences": differences,
        "passed": maximum <= AUDIT_TOLERANCE,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--out", default="data/c4_modern_guards.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N prompts per condition as a smoke test",
    )
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("the frozen Phase 1 split requires --seed 0")
    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    rows, _, _, _, _ = load_judged_rows(
        Path(args.inp), 0, args.seed, keep_protected_group=False
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    split_sizes = (len(train_rows), len(tune_rows), len(test_rows))
    if split_sizes != EXPECTED_SPLIT:
        raise ValueError(f"unexpected frozen split sizes: {split_sizes}")

    translations, translation_hashes = load_translations(
        Path(args.translations_dir), tune_rows, test_rows
    )
    conditions_by_split = {}
    for split, split_rows_, offset in (
        ("tune", tune_rows, 0),
        ("test", test_rows, len(tune_rows)),
    ):
        window = slice(offset, offset + len(split_rows_))
        built = build_conditions(split_rows_, translations["swahili"][window])
        built.update(
            {name: translations[name][window] for name in ("french", "hindi", "zulu")}
        )
        conditions_by_split[split] = {name: built[name] for name in CONDITIONS}

    if args.limit:
        conditions_by_split = {
            split: {name: texts[: args.limit] for name, texts in conditions.items()}
            for split, conditions in conditions_by_split.items()
        }

    snapshot, revision = resolve_cached_snapshot(QWEN_MODEL)
    print(f"[models] Qwen3Guard={revision}", flush=True)

    scoring_rows = guard_rows("tune", conditions_by_split["tune"]) + guard_rows(
        "test", conditions_by_split["test"]
    )
    print(f"[Qwen3Guard] rows={len(scoring_rows)}", flush=True)
    run_qwen3guard(scoring_rows, str(snapshot), args.batch_size)

    if args.limit:
        audit = {"skipped": "smoke run"}
        generation_check(scoring_rows, str(snapshot), args.batch_size)
    else:
        rescored = audit_rows(scoring_rows)
        print(f"[audit] rows={len(rescored)} batch_size=1", flush=True)
        run_qwen3guard(rescored, str(snapshot), 1)
        audit = batch_audit(scoring_rows, rescored)
        print(
            f"[audit] passed={audit['passed']} "
            f"max_difference={audit['maximum_difference']:.6g}",
            flush=True,
        )

    expected = args.limit or len(tune_rows)
    output = {
        "tune_ids": np.asarray([str(row["id"]) for row in tune_rows]),
        "test_ids": np.asarray([str(row["id"]) for row in test_rows]),
        "tune_labels": np.asarray(
            [int(bool(row["harmful"])) for row in tune_rows], dtype=np.int64
        ),
        "test_labels": np.asarray(
            [int(bool(row["harmful"])) for row in test_rows], dtype=np.int64
        ),
    }
    for split in ("tune", "test"):
        for condition in CONDITIONS:
            scores = unpack_guard(scoring_rows, split, condition, "score").astype(
                np.float32
            )
            if scores.shape != (expected,) or not np.isfinite(scores).all():
                raise ValueError(f"invalid Qwen3Guard scores for {split} {condition}")
            output[f"{split}_{condition}_qwen3guard"] = scores
            output[f"{split}_{condition}_qwen3guard_controversial"] = unpack_guard(
                scoring_rows, split, condition, "controversial"
            ).astype(np.float32)
            output[f"{split}_{condition}_qwen3guard_verdict"] = unpack_guard(
                scoring_rows, split, condition, "verdict"
            ).astype(np.int8)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    import torch

    metadata = {
        "model": QWEN_MODEL,
        "revision": revision,
        "decision_prefix": DECISION_PREFIX,
        "labels": list(LABELS),
        "score": "softmax P(' Unsafe') over the three verdict branches",
        "verdict_codes": {"0": "Safe", "1": "Unsafe", "2": "Controversial"},
        "batch_size": args.batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "source_commit": commit,
        "conditions_scored": list(CONDITIONS),
        "translation_hashes": translation_hashes,
        "audit": audit,
        "audit_indices": list(AUDIT_INDICES),
        "audit_conditions": list(AUDIT_CONDITIONS),
        "cuda_device": torch.cuda.get_device_name(),
        "model_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "training": "none",
    }
    output["modern_guard_json"] = np.asarray(json.dumps(metadata))
    output["modern_guard_inputs_sha256"] = np.asarray(
        strings_sha256(
            [
                text
                for split in ("tune", "test")
                for condition in CONDITIONS
                for text in conditions_by_split[split][condition]
            ]
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
