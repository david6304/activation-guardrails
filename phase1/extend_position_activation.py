"""P1 scoring job: the frozen detectors on the cipher conditions at two read positions.

`t_inst` sits seven tokens *after* the ciphertext, because the decoder-instruction
wrapper appends "Respond normally in plain English." to the payload (measured by
`phase1/prepare_p1_conditions.py`). So "reverse is not represented" may be
"reverse is not represented at the position we happened to read". This job scores
both positions:

  t_inst    the frozen Phase 1 position, the token before <end_of_turn>
  t_cipher  the last token of the encoded payload itself

Nothing is refitted. The all-layer probe and centroid come from the frozen
multilingual artefact, the per-layer probes and centroids from the frozen
layerwise artefact; the full layerwise curve is free in the same extraction and
separates "resolved early in depth" from "decode consumes depth". The `plain`
condition is scored at `t_inst` as an audit of the whole path against the frozen
numbers, and to give extra 9 the plain class-conditional means it needs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from phase1.phase1_activation import centroid_scores
from phase1.phase1_layerwise import per_layer_centroid_scores
from phase1.prepare_p1_conditions import (
    CONDITIONS as CIPHER_CONDITIONS,
    build_p1_conditions,
    payload_end_char,
)
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    load_model,
    split_rows,
    strings_sha256,
)


CONDITIONS = ("plain", *CIPHER_CONDITIONS)
EXPECTED_MODEL = "google/gemma-3-27b-it"
EXPECTED_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
EXPECTED_SPLIT = (5341, 1781, 1781)
# The plain condition has no ciphertext, so t_cipher is undefined there.
AUDIT_AUROC_TOLERANCE = 0.01


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def encode_batch(rendered, tokenizer):
    """One tokenisation path, with offsets aligned to the left-padded rows."""
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    return encoded, encoded.pop("offset_mapping")


def resolve_positions(
    rendered, encoded, offsets, spans, position_name, end_of_turn_id
):
    """Read position per row, as an index into the left-padded sequence."""
    positions = []
    for row, rendered_text in enumerate(rendered):
        ids = encoded["input_ids"][row]
        matches = (ids == end_of_turn_id).nonzero(as_tuple=False).flatten()
        if len(matches) == 0:
            raise ValueError("rendered prompt contains no <end_of_turn> token")
        instruction_position = int(matches[-1]) - 1
        if position_name == "t_inst":
            position = instruction_position
        else:
            end_char = payload_end_char(rendered_text, spans[row])
            # Offsets are aligned to the padded row; left padding is (0, 0).
            candidates = [
                token
                for token, (first, last) in enumerate(offsets[row].tolist())
                if last > first and last <= end_char
            ]
            if not candidates:
                raise ValueError("no token ends inside the payload")
            position = max(candidates)
            if position >= instruction_position:
                raise ValueError("payload position does not precede the instruction")
        if position < 0 or not bool(encoded["attention_mask"][row, position]):
            raise ValueError(f"invalid {position_name} position")
        positions.append(position)
    return positions


def iter_batches(texts, payloads, model, tokenizer, batch_size, position_name):
    """Yield original row indices and [batch, layer, hidden] states at one position."""
    import torch

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id == tokenizer.unk_token_id:
        raise ValueError("Gemma tokenizer has no <end_of_turn> token")

    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    started = time.time()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": texts[index]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for index in indices
        ]
        encoded, offsets = encode_batch(rendered, tokenizer)
        positions = resolve_positions(
            rendered,
            encoded,
            offsets,
            [payloads[index] for index in indices],
            position_name,
            end_of_turn_id,
        )
        encoded = encoded.to(model.device)

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


def score_cell(
    texts, payloads, labels, model, tokenizer, batch_size, position_name, frozen
):
    """All-layer and per-layer scores, plus the standing rider's class statistics.

    The class-conditional mean and variance diagonal per layer cost nothing here
    and cost a whole job to recover later, so they are accumulated in the same
    pass (finish plan, standing rider; they unblock extra 9).
    """
    layer_count = frozen["layer_weights"].shape[0]
    scores = {
        "logistic": np.empty(len(texts), dtype=np.float32),
        "centroid": np.empty(len(texts), dtype=np.float32),
        "layer_logistic": np.empty((len(texts), layer_count), dtype=np.float32),
        "layer_centroid": np.empty((len(texts), layer_count), dtype=np.float32),
    }
    totals = {}
    counts = {"harmful": 0, "benign": 0}
    for indices, features in iter_batches(
        texts, payloads, model, tokenizer, batch_size, position_name
    ):
        flat = features.reshape(len(indices), -1)
        scores["logistic"][indices] = flat @ frozen["weight"] + frozen["intercept"]
        scores["centroid"][indices] = centroid_scores(
            features, frozen["harmful_centroid"], frozen["harmless_centroid"]
        )
        states = features[:, 1:, :]
        scores["layer_logistic"][indices] = (
            np.einsum("nlh,lh->nl", states, frozen["layer_weights"])
            + frozen["layer_intercepts"]
        ).astype(np.float32)
        scores["layer_centroid"][indices] = per_layer_centroid_scores(
            features, frozen["layer_harmful_centroid"], frozen["layer_harmless_centroid"]
        )
        wide = features.astype(np.float64)
        batch_labels = labels[indices]
        for name, value in (("harmful", 1), ("benign", 0)):
            selected = wide[batch_labels == value]
            if not len(selected):
                continue
            if name not in totals:
                totals[name] = [
                    np.zeros(features.shape[1:], dtype=np.float64),
                    np.zeros(features.shape[1:], dtype=np.float64),
                ]
            totals[name][0] += selected.sum(axis=0)
            totals[name][1] += np.square(selected).sum(axis=0)
            counts[name] += len(selected)

    statistics = {}
    for name, (total, total_square) in totals.items():
        mean = total / counts[name]
        statistics[f"{name}_mean"] = mean.astype(np.float32)
        statistics[f"{name}_var"] = np.maximum(
            total_square / counts[name] - np.square(mean), 0
        ).astype(np.float32)
        statistics[f"{name}_n"] = np.asarray(counts[name])
    return scores, statistics


def collect_layer_features(texts, payloads, model, tokenizer, batch_size, position_name):
    """Per-layer features for a whole cell, held in float32.

    Not float16: Gemma-3's residual stream carries massive activations well past
    float16's 65504 ceiling, so the cast overflows to inf and the fit dies (job
    57303314). 5341 train rows x 63 layers x 5376 dims is 7.2 GB at float32 and
    the node has already peaked at 119 GB, so the memory is there. Probes are fit
    per layer rather than on the all-layer concatenation, which liblinear will not
    take at this width -- and per-layer is the more informative answer anyway: it
    says *where* in depth base64 harm becomes linearly readable, if it ever does.
    """
    features = None
    for indices, batch in iter_batches(
        texts, payloads, model, tokenizer, batch_size, position_name
    ):
        if features is None:
            features = np.empty(
                (len(texts), batch.shape[1], batch.shape[2]), dtype=np.float32
            )
        features[indices] = batch
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite {position_name} features")
    return features


def fit_condition_probe(train_features, train_labels, cells, seed):
    """One logistic probe per layer, trained on this condition's own activations.

    The frozen probe reads the plain-English harmfulness direction. If it is at
    chance under base64 the content may still be represented, just not on that
    direction -- a probe trained on base64 activations separates "not represented"
    from "represented elsewhere". Nothing frozen is touched.
    """
    from sklearn.linear_model import LogisticRegression

    layer_count = train_features.shape[1]
    scores = {
        name: np.empty((len(f), layer_count), dtype=np.float32)
        for name, f in cells.items()
    }
    for layer in range(layer_count):
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=seed,
            solver="liblinear",
            max_iter=2000,
        ).fit(train_features[:, layer, :], train_labels)
        for name, features in cells.items():
            scores[name][:, layer] = classifier.decision_function(
                features[:, layer, :]
            ).astype(np.float32)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument(
        "--activation", default="data/phase1_activation_multilingual_27b.npz"
    )
    parser.add_argument("--layerwise", default="data/phase1_layerwise_27b.npz")
    parser.add_argument("--manifest", default="data/p1_conditions_manifest.json")
    parser.add_argument("--out", default="data/p1_position_scores.npz")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N tune/test rows per cell (real-model smoke)",
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

    plaintexts = [row["prompt"] for row in tune_rows + test_rows]
    conditions, payloads = build_p1_conditions(plaintexts, args.seed)
    conditions["plain"] = plaintexts
    payloads["plain"] = [None] * len(plaintexts)

    manifest = json.loads(Path(args.manifest).read_text())
    for condition in CIPHER_CONDITIONS:
        if strings_sha256(conditions[condition]) != manifest["strings_sha256"][condition]:
            raise ValueError(f"reconstructed {condition} does not match the manifest")
    print(f"[manifest] {len(CIPHER_CONDITIONS)} condition hashes match", flush=True)

    activation = load_npz(Path(args.activation))
    layerwise = load_npz(Path(args.layerwise))
    if str(activation["model_revision"]) != EXPECTED_REVISION:
        raise ValueError("frozen activation revision mismatch")
    if str(layerwise["model_revision"]) != EXPECTED_REVISION:
        raise ValueError("frozen layerwise revision mismatch")
    if not np.array_equal(activation["test_ids"], layerwise["test_ids"]):
        raise ValueError("frozen artefacts disagree on the test split")
    frozen = {
        "weight": activation["logistic_weight"],
        "intercept": float(activation["logistic_intercept"]),
        "harmful_centroid": activation["harmful_centroid"],
        "harmless_centroid": activation["harmless_centroid"],
        "layer_weights": layerwise["weights"],
        "layer_intercepts": layerwise["intercepts"],
        "layer_harmful_centroid": layerwise["harmful_centroid"],
        "layer_harmless_centroid": layerwise["harmless_centroid"],
    }

    tune_labels = np.asarray([int(bool(row["harmful"])) for row in tune_rows])
    test_labels = np.asarray([int(bool(row["harmful"])) for row in test_rows])
    if not np.array_equal(activation["tune_labels"], tune_labels):
        raise ValueError("frozen tune labels differ")
    if not np.array_equal(activation["test_labels"], test_labels):
        raise ValueError("frozen test labels differ")
    if args.limit:
        tune_labels = tune_labels[: args.limit]
        test_labels = test_labels[: args.limit]

    model, tokenizer, num_layers, hidden_size = load_model(EXPECTED_MODEL, args.seed)
    loaded_revision = str(getattr(model.config, "_commit_hash", "") or "")
    if loaded_revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"loaded Gemma revision {loaded_revision!r} != {EXPECTED_REVISION!r}"
        )
    if frozen["weight"].shape != ((num_layers + 1) * hidden_size,):
        raise ValueError("frozen logistic weight dimension does not match the model")
    if frozen["layer_weights"].shape != (num_layers, hidden_size):
        raise ValueError("frozen layerwise weight shape does not match the model")

    output = {}
    audits = {}
    for condition in CONDITIONS:
        positions = ("t_inst",) if condition == "plain" else ("t_inst", "t_cipher")
        for position in positions:
            for split, offset, count in (
                ("tune", 0, len(tune_rows)),
                ("test", len(tune_rows), len(test_rows)),
            ):
                take = min(args.limit, count) if args.limit else count
                texts = conditions[condition][offset : offset + take]
                spans = payloads[condition][offset : offset + take]
                labels = tune_labels if split == "tune" else test_labels
                print(f"[score] {split} {condition} {position} n={take}", flush=True)
                scores, statistics = score_cell(
                    texts,
                    spans,
                    labels,
                    model,
                    tokenizer,
                    args.batch_size,
                    position,
                    frozen,
                )
                prefix = f"{split}_{condition}_{position}"
                for name in ("logistic", "centroid", "layer_logistic", "layer_centroid"):
                    values = scores[name]
                    if not np.isfinite(values).all():
                        raise ValueError(f"non-finite {name} for {prefix}")
                    output[f"{prefix}_{name}"] = values
                for name, values in statistics.items():
                    output[f"{prefix}_{name}"] = values

    # Both `plain` and `reverse` at t_inst already exist in the frozen artefact, so
    # rescoring them audits the whole new path -- tokenisation, padding, position
    # resolution and the reused detector parameters -- against known numbers.
    for condition in ("plain", "reverse"):
        for split, labels in (("tune", tune_labels), ("test", test_labels)):
            if len(np.unique(labels)) < 2:
                print(f"[audit] {split} {condition}: skipped, one class under --limit")
                continue
            frozen_scores = activation[f"{split}_{condition}_logistic"][: len(labels)]
            reproduced = output[f"{split}_{condition}_t_inst_logistic"]
            difference = float(np.abs(frozen_scores - reproduced).max())
            frozen_auroc = float(roc_auc_score(labels, frozen_scores))
            reproduced_auroc = float(roc_auc_score(labels, reproduced))
            audits[f"{condition}_{split}"] = {
                "max_abs_score_difference": difference,
                "frozen_auroc": frozen_auroc,
                "reproduced_auroc": reproduced_auroc,
            }
            print(
                f"[audit] {split} {condition} t_inst: max|delta|={difference:.4f} "
                f"AUROC {frozen_auroc:.4f} -> {reproduced_auroc:.4f}",
                flush=True,
            )
            if abs(frozen_auroc - reproduced_auroc) > AUDIT_AUROC_TOLERANCE:
                raise ValueError(
                    f"{split} {condition} t_inst does not reproduce the frozen artefact"
                )

    # Does base64 harm live anywhere linearly readable, or nowhere? Train on this
    # condition's own train-split activations and score its own tune/test rows.
    train_plaintexts = [row["prompt"] for row in train_rows]
    train_conditions, train_payloads = build_p1_conditions(train_plaintexts, args.seed)
    train_labels = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    train_take = min(args.limit, len(train_rows)) if args.limit else len(train_rows)
    for position in ("t_inst", "t_cipher"):
        print(f"[selftrain] base64 {position} train n={train_take}", flush=True)
        train_features = collect_layer_features(
            train_conditions["base64"][:train_take],
            train_payloads["base64"][:train_take],
            model,
            tokenizer,
            args.batch_size,
            position,
        )
        cells = {}
        for split, offset, count in (
            ("tune", 0, len(tune_labels)),
            ("test", len(tune_rows), len(test_labels)),
        ):
            print(f"[selftrain] base64 {position} {split} n={count}", flush=True)
            cells[split] = collect_layer_features(
                conditions["base64"][offset : offset + count],
                payloads["base64"][offset : offset + count],
                model,
                tokenizer,
                args.batch_size,
                position,
            )
        if len(np.unique(train_labels[:train_take])) < 2:
            print("[selftrain] skipped, one class under --limit", flush=True)
            continue
        scores = fit_condition_probe(
            train_features, train_labels[:train_take], cells, args.seed
        )
        for split, values in scores.items():
            output[f"{split}_base64_{position}_selftrained"] = values

    metadata = {
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "conditions": list(CONDITIONS),
        "positions": {
            "t_inst": "token before <end_of_turn>, the frozen Phase 1 position",
            "t_cipher": "last token of the encoded payload",
        },
        "frozen_activation": str(Path(args.activation)),
        "frozen_activation_sha256": file_sha256(Path(args.activation)),
        "frozen_layerwise": str(Path(args.layerwise)),
        "frozen_layerwise_sha256": file_sha256(Path(args.layerwise)),
        "conditions_manifest_sha256": file_sha256(Path(args.manifest)),
        "training": (
            "frozen detectors reused unchanged; additionally one per-layer "
            "logistic probe per read position trained on base64 train-split "
            "activations at C=1.0, saved as *_selftrained"
        ),
        "frozen_audit": audits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tune_ids=np.asarray([str(row["id"]) for row in tune_rows][: len(tune_labels)]),
        test_ids=np.asarray([str(row["id"]) for row in test_rows][: len(test_labels)]),
        tune_labels=tune_labels,
        test_labels=test_labels,
        layer_indices=layerwise["layer_indices"],
        position_metadata_json=np.asarray(json.dumps(metadata)),
        **output,
    )
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
