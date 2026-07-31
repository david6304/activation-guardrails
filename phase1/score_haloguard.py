"""Score HaloGuard 1.0-4B on the frozen Phase 1 six-condition matrix.

HaloGuard (July 2026) is the only 2026-era guard covering Swahili and isiZulu, so it
answers the "current guards" objection on the axis this project claims. Score is the
unsafe-label probability at the teacher-forced `Safety: ` position, the same
construction validated for Qwen3Guard in C4.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phase1.extend_multilingual_guards import CONDITIONS, resolve_cached_snapshot
from phase1.extend_multilingual_tfidf import load_translations
from phase1.phase1_baselines import build_conditions, guard_rows, unpack_guard
from phase1.score_modern_guards import EXPECTED_SPLIT, audit_rows, batch_audit
from probe_prompt import file_sha256, load_judged_rows, split_rows, strings_sha256

HALO_MODEL = "astroware/HaloGuard1-Gen-4B"
POLICY_PATH = Path(__file__).with_name("haloguard_policy.txt")
DECISION_PREFIX = "Safety: "
LABELS = ("safe", "unsafe")


def label_token_ids(tokeniser):
    ids = [
        int(tokeniser(label, add_special_tokens=False)["input_ids"][0])
        for label in LABELS
    ]
    if len(set(ids)) != len(ids):
        raise ValueError(f"verdict branches share a first token: {ids}")
    return ids


def run_haloguard(rows, model_path, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    policy = POLICY_PATH.read_text()
    tokeniser = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokeniser.padding_side = "left"
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
    ids = label_token_ids(tokeniser)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True
    ).to("cuda")
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            tokeniser.apply_chat_template(
                [
                    {"role": "system", "content": policy},
                    {"role": "user", "content": row["text"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
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
            row["verdict"] = int(np.argmax(probability))
        if start % (batch_size * 50) == 0 or start + batch_size >= len(rows):
            print(f"  haloguard {min(start + batch_size, len(rows))}/{len(rows)}",
                  flush=True)
    del model
    torch.cuda.empty_cache()


def generation_check(rows, model_path, batch_size):
    """Free-generate the verdict and confirm it matches the teacher-forced argmax."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    policy = POLICY_PATH.read_text()
    tokeniser = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokeniser.padding_side = "left"
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True
    ).to("cuda")
    model.eval()
    agree = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            tokeniser.apply_chat_template(
                [
                    {"role": "system", "content": policy},
                    {"role": "user", "content": row["text"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + DECISION_PREFIX
            for row in batch
        ]
        encoded = tokeniser(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=4, do_sample=False,
                pad_token_id=tokeniser.pad_token_id,
            )
        new = generated[:, encoded["input_ids"].shape[1] :]
        for row, sequence in zip(batch, new):
            text = tokeniser.decode(sequence, skip_special_tokens=True).strip().lower()
            spoken = 1 if text.startswith("unsafe") else 0
            agree += int(spoken == row["verdict"])
    del model
    torch.cuda.empty_cache()
    return agree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--out", default="data/c4_haloguard.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N prompts per condition")
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
    if (len(train_rows), len(tune_rows), len(test_rows)) != EXPECTED_SPLIT:
        raise ValueError("unexpected frozen split sizes")

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

    snapshot, revision = resolve_cached_snapshot(HALO_MODEL)
    print(f"[models] haloguard={revision}", flush=True)

    scored = []
    for split, conditions in conditions_by_split.items():
        batch = guard_rows(split, conditions)
        print(f"[score] {split} n={len(batch)}", flush=True)
        run_haloguard(batch, str(snapshot), args.batch_size)
        scored.extend(batch)

    smoke = scored[: min(96, len(scored))]
    agree = generation_check(smoke, str(snapshot), args.batch_size)
    print(f"[generation-check] {agree}/{len(smoke)} agree with the argmax", flush=True)

    audit = None
    if not args.limit:
        selected = audit_rows(scored)
        rescored = [dict(row) for row in selected]
        run_haloguard(rescored, str(snapshot), 1)
        audit = batch_audit(selected, rescored)
        print(
            f"[audit] max |batched - single| = {audit['maximum_difference']:.4g} "
            f"passed={audit['passed']}",
            flush=True,
        )

    output = {}
    for split, conditions in conditions_by_split.items():
        for condition in conditions:
            output[f"{split}_{condition}_haloguard"] = unpack_guard(
                scored, split, condition, "score"
            ).astype(np.float32)
        rows_for_split = tune_rows if split == "tune" else test_rows
        if args.limit:
            rows_for_split = rows_for_split[: args.limit]
        output[f"{split}_labels"] = np.asarray(
            [int(bool(row["harmful"])) for row in rows_for_split]
        )
        output[f"{split}_ids"] = np.asarray([str(row["id"]) for row in rows_for_split])

    metadata = {
        "model": HALO_MODEL,
        "model_revision": revision,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "policy_sha256": file_sha256(POLICY_PATH),
        "decision_prefix": DECISION_PREFIX,
        "labels": list(LABELS),
        "generation_check": f"{agree}/{len(smoke)}",
        "batch_audit": audit,
        "translation_hashes": translation_hashes,
    }
    output["metadata_json"] = np.asarray(json.dumps(metadata))
    output["inputs_sha256"] = np.asarray(
        strings_sha256(
            [
                text
                for conditions in conditions_by_split.values()
                for condition in CONDITIONS
                for text in conditions[condition]
            ]
        )
    )
    np.savez_compressed(output_path, **output)
    print(f"[done] {output_path} sha256={file_sha256(output_path)}", flush=True)


if __name__ == "__main__":
    main()
