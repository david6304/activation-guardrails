"""Score the frozen prompt set with activation detectors at one token position.

One Gemma forward pass supplies both the all-layer logistic probe and the
Zhao-style layer-averaged cosine-centroid score. Only scalar tune/test scores
and fitted detector parameters are persisted. The Phase 1 default remains
``t_inst``; Phase 2 uses the same code at ``t_post_inst``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from capability_check import build_sent
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    load_model,
    load_or_translate,
    split_rows,
    strings_sha256,
    train_probe,
    truncate_left_tokens,
)


NLLB_CODE = "swh_Latn"


def iter_position_batches(texts, model, tok, batch_size, position_name):
    """Yield original row indices and [batch, layer, hidden] position states."""
    import torch

    end_of_turn_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id == tok.unk_token_id:
        raise ValueError("Gemma tokenizer has no <end_of_turn> token")

    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    started = time.time()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch_texts = truncate_left_tokens(
            [texts[index] for index in indices], tok, 2048
        )
        messages = [[{"role": "user", "content": text}] for text in batch_texts]
        encoded = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        positions = []
        for input_ids, attention_mask in zip(
            encoded["input_ids"], encoded["attention_mask"]
        ):
            matches = (input_ids == end_of_turn_id).nonzero(as_tuple=False).flatten()
            if len(matches) == 0:
                raise ValueError("rendered prompt contains no <end_of_turn> token")
            instruction_position = int(matches[-1]) - 1
            if position_name == "t_inst":
                position = instruction_position
            else:
                attended = attention_mask.nonzero(as_tuple=False).flatten()
                position = int(attended[-1])
                if position <= instruction_position:
                    raise ValueError("prompt-final position does not follow instruction")
            if position < 0 or not bool(attention_mask[position]):
                raise ValueError(f"invalid {position_name} position")
            positions.append(position)

        with torch.no_grad():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )

        batch_features = np.empty(
            (len(indices), len(output.hidden_states), output.hidden_states[0].shape[-1]),
            dtype=np.float32,
        )
        for layer, states in enumerate(output.hidden_states):
            rows = torch.arange(len(indices), device=states.device)
            layer_positions = torch.as_tensor(positions, device=states.device)
            batch_features[:, layer, :] = (
                states[rows, layer_positions, :].float().cpu().numpy()
            )

        done = min(start + batch_size, len(order))
        print(
            f"  forwarded {done}/{len(order)}  {done / (time.time() - started):.2f}/s",
            flush=True,
        )
        yield indices, batch_features


def extract_features(
    texts, model, tok, batch_size, layer_count, hidden_size, position_name
):
    features = np.empty(
        (len(texts), layer_count, hidden_size), dtype=np.float32
    )
    for indices, batch in iter_position_batches(
        texts, model, tok, batch_size, position_name
    ):
        features[indices] = batch
    return features


def fit_centroids(features, labels):
    """Class centroids for hidden layers 1..L, matching Zhao et al. Eq. 3."""
    hidden_states = features[:, 1:, :]
    harmful = hidden_states[labels == 1].mean(axis=0)
    harmless = hidden_states[labels == 0].mean(axis=0)
    return harmful.astype(np.float32), harmless.astype(np.float32)


def centroid_scores(features, harmful_centroid, harmless_centroid):
    hidden_states = features[:, 1:, :]
    state_norm = np.linalg.norm(hidden_states, axis=2)
    harmful_norm = np.linalg.norm(harmful_centroid, axis=1)
    harmless_norm = np.linalg.norm(harmless_centroid, axis=1)
    harmful_cosine = np.einsum(
        "nlh,lh->nl", hidden_states, harmful_centroid
    ) / np.maximum(state_norm * harmful_norm, 1e-12)
    harmless_cosine = np.einsum(
        "nlh,lh->nl", hidden_states, harmless_centroid
    ) / np.maximum(state_norm * harmless_norm, 1e-12)
    return (harmful_cosine - harmless_cosine).mean(axis=1).astype(np.float32)


def score_detectors(
    texts,
    model,
    tok,
    batch_size,
    logistic_weight,
    logistic_intercept,
    harmful_centroid,
    harmless_centroid,
    position_name,
):
    logistic = np.empty(len(texts), dtype=np.float32)
    centroid = np.empty(len(texts), dtype=np.float32)
    for indices, features in iter_position_batches(
        texts, model, tok, batch_size, position_name
    ):
        logistic[indices] = (
            features.reshape(len(indices), -1) @ logistic_weight
            + logistic_intercept
        )
        centroid[indices] = centroid_scores(
            features, harmful_centroid, harmless_centroid
        )
    return logistic, centroid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--prepare-translations", action="store_true")
    parser.add_argument("--position", choices=("t_inst", "t_post_inst"), default="t_inst")
    parser.add_argument("--out")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output_path = Path(
        args.out
        or (
            "data/phase1_activation_27b.npz"
            if args.position == "t_inst"
            else "data/phase2_activation_t_post_inst_27b.npz"
        )
    )
    if args.smoke:
        model, tok, num_layers, hidden_size = load_model(args.model, args.seed)
        _, features = next(
            iter_position_batches(
                ["Explain photosynthesis.", "What is two plus two?"],
                model,
                tok,
                2,
                args.position,
            )
        )
        expected = (2, num_layers + 1, hidden_size)
        if features.shape != expected or not np.isfinite(features).all():
            raise ValueError(f"invalid smoke features: {features.shape}, expected {expected}")
        print(f"[smoke] position={args.position} shape={features.shape} finite=true")
        return

    rows, _, _, _, _ = load_judged_rows(
        Path(args.inp), args.limit, args.seed, keep_protected_group=False
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    print(
        f"[split] train={len(train_rows)} tune={len(tune_rows)} test={len(test_rows)}",
        flush=True,
    )

    tune_plain = [row["prompt"] for row in tune_rows]
    test_plain = [row["prompt"] for row in test_rows]
    translated_prompts = tune_plain + test_plain
    translations_dir = Path(
        f"{args.translations_dir}_limit{args.limit}" if args.limit else args.translations_dir
    )
    translation_path = translations_dir / "swahili.jsonl"
    swahili, truncated = load_or_translate(
        translated_prompts,
        translation_path,
        args.nllb,
        NLLB_CODE,
        args.prepare_translations,
    )
    metadata_path = translations_dir / "metadata.json"
    if args.prepare_translations:
        from transformers import AutoConfig

        revision = getattr(AutoConfig.from_pretrained(args.nllb), "_commit_hash", None)
        if not revision:
            raise RuntimeError("could not resolve the cached NLLB revision")
        metadata_path.write_text(
            json.dumps(
                {
                    "nllb_model": args.nllb,
                    "nllb_revision": revision,
                    "swahili_sha256": file_sha256(translation_path),
                    "inputs_exceeding_256_tokens": len(truncated),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"[done] translations -> {translations_dir}", flush=True)
        return

    metadata = json.loads(metadata_path.read_text())
    if metadata["nllb_model"] != args.nllb:
        raise ValueError("translation model mismatch")
    if metadata["swahili_sha256"] != file_sha256(translation_path):
        raise ValueError("Swahili translation checksum mismatch")
    tune_swahili = swahili[: len(tune_rows)]
    test_swahili = swahili[len(tune_rows) :]
    tune_conditions = {
        "plain": tune_plain,
        "swahili": tune_swahili,
        "reverse": [
            build_sent(text, "reverse", in_obf=True, out_obf=False)
            for text in tune_plain
        ],
    }
    test_conditions = {
        "plain": test_plain,
        "swahili": test_swahili,
        "reverse": [
            build_sent(text, "reverse", in_obf=True, out_obf=False)
            for text in test_plain
        ],
    }

    model, tok, num_layers, hidden_size = load_model(args.model, args.seed)
    layer_count = num_layers + 1
    print(
        f"[features] position={args.position} layers={layer_count} hidden={hidden_size}",
        flush=True,
    )
    print("[extract] plain train", flush=True)
    train_features = extract_features(
        [row["prompt"] for row in train_rows],
        model,
        tok,
        args.batch_size,
        layer_count,
        hidden_size,
        args.position,
    )
    print("[extract] plain tune", flush=True)
    tune_features = extract_features(
        tune_plain,
        model,
        tok,
        args.batch_size,
        layer_count,
        hidden_size,
        args.position,
    )
    train_labels = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    tune_labels = np.asarray([int(bool(row["harmful"])) for row in tune_rows])
    test_labels = np.asarray([int(bool(row["harmful"])) for row in test_rows])
    harmful_centroid, harmless_centroid = fit_centroids(
        train_features, train_labels
    )
    weight, intercept, selected_c, selected_loss, selected_se, tuning = train_probe(
        train_features.reshape(len(train_rows), -1).copy(),
        train_labels,
        tune_features.reshape(len(tune_rows), -1).copy(),
        tune_labels,
        args.seed,
    )
    del train_features

    scores = {}
    scores["tune_plain_logistic"] = (
        tune_features.reshape(len(tune_rows), -1) @ weight + intercept
    ).astype(np.float32)
    scores["tune_plain_centroid"] = centroid_scores(
        tune_features, harmful_centroid, harmless_centroid
    )
    del tune_features
    for split, conditions in (("tune", tune_conditions), ("test", test_conditions)):
        for condition, texts in conditions.items():
            if split == "tune" and condition == "plain":
                continue
            print(f"[score] {split} {condition}", flush=True)
            logistic, centroid = score_detectors(
                texts,
                model,
                tok,
                args.batch_size,
                weight,
                intercept,
                harmful_centroid,
                harmless_centroid,
                args.position,
            )
            scores[f"{split}_{condition}_logistic"] = logistic
            scores[f"{split}_{condition}_centroid"] = centroid

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        model=np.asarray(args.model),
        model_revision=np.asarray(getattr(model.config, "_commit_hash", None) or ""),
        seed=np.asarray(args.seed),
        position=np.asarray(args.position),
        train_ids=np.asarray([str(row["id"]) for row in train_rows]),
        tune_ids=np.asarray([str(row["id"]) for row in tune_rows]),
        test_ids=np.asarray([str(row["id"]) for row in test_rows]),
        tune_labels=tune_labels,
        test_labels=test_labels,
        test_categories=np.asarray([str(row.get("category")) for row in test_rows]),
        inputs_sha256=np.asarray(
            strings_sha256(
                [text for values in tune_conditions.values() for text in values]
                + [text for values in test_conditions.values() for text in values]
            )
        ),
        logistic_weight=weight,
        logistic_intercept=np.asarray(intercept),
        selected_c=np.asarray(selected_c),
        selected_tune_log_loss=np.asarray(selected_loss),
        selected_tune_log_loss_se=np.asarray(selected_se),
        tuning_json=np.asarray(json.dumps(tuning)),
        harmful_centroid=harmful_centroid,
        harmless_centroid=harmless_centroid,
        translation_metadata=np.asarray(json.dumps(metadata)),
        **scores,
    )
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
