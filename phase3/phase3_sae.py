"""Run the frozen two-layer dense-versus-SAE Phase 3 comparison.

Both probe types use Gemma 3 27B residual outputs at the final instruction
token. Probes are trained on plain prompts only and evaluated on the frozen
plain, Swahili, and reverse tune/test conditions.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from capability_check import build_sent
from probe_prompt import (
    C_GRID,
    file_sha256,
    load_judged_rows,
    load_model,
    load_or_translate,
    split_rows,
    strings_sha256,
    train_probe,
    truncate_left_tokens,
)


MODEL = "google/gemma-3-27b-it"
MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
SAE_REPO = "google/gemma-scope-2-27b-it"
SAE_REVISION = "5c58dd4cddd52cef653059d85e12a86bf6222a28"
BLOCKS = (31, 40)
HIDDEN_STATE_INDICES = {31: 32, 40: 41}
SAE_IDS = {
    block: f"resid_post/layer_{block}_width_65k_l0_medium" for block in BLOCKS
}
NLLB_CODE = "swh_Latn"
FEATURE_WIDTH = 65536


class JumpReLUSAE:
    """Minimal loader for the frozen Gemma Scope 2 JumpReLU checkpoints."""

    def __init__(self, block):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        self.block = block
        self.sae_id = SAE_IDS[block]
        config_path = hf_hub_download(
            SAE_REPO,
            f"{self.sae_id}/config.json",
            revision=SAE_REVISION,
            local_files_only=True,
        )
        params_path = hf_hub_download(
            SAE_REPO,
            f"{self.sae_id}/params.safetensors",
            revision=SAE_REVISION,
            local_files_only=True,
        )
        self.config = json.loads(Path(config_path).read_text())
        expected = {
            "hf_hook_point_in": f"model.layers.{block}.output",
            "width": FEATURE_WIDTH,
            "model_name": MODEL,
            "architecture": "jump_relu",
            "affine_connection": False,
            "type": "sae",
        }
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(
                    f"{self.sae_id} has unexpected {key}: {self.config.get(key)!r}"
                )

        with safe_open(params_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected_keys = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
            if keys != expected_keys:
                raise ValueError(f"{self.sae_id} parameter keys differ: {sorted(keys)}")
            self.w_enc = handle.get_tensor("w_enc")
            self.w_dec = handle.get_tensor("w_dec")
            self.b_enc = handle.get_tensor("b_enc")
            self.b_dec = handle.get_tensor("b_dec")
            self.threshold = handle.get_tensor("threshold")

        hidden = self.w_enc.shape[0]
        shapes = {
            "w_enc": (hidden, FEATURE_WIDTH),
            "w_dec": (FEATURE_WIDTH, hidden),
            "b_enc": (FEATURE_WIDTH,),
            "b_dec": (hidden,),
            "threshold": (FEATURE_WIDTH,),
        }
        for name, shape in shapes.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
                raise ValueError(
                    f"{self.sae_id} {name} is {tuple(tensor.shape)} {tensor.dtype}, "
                    f"expected {shape} float32"
                )
        if not bool((self.threshold >= 0).all()):
            raise ValueError(f"{self.sae_id} has a negative JumpReLU threshold")
        self.device = torch.device("cpu")
        self.validation_metrics = None

    def to(self, device):
        import torch

        device = torch.device(device)
        if device == self.device:
            return
        for name in ("w_enc", "w_dec", "b_enc", "b_dec", "threshold"):
            setattr(self, name, getattr(self, name).to(device))
        self.device = device

    def encode(self, activations):
        import torch

        if activations.device != self.device:
            self.to(activations.device)
        pre = activations.float() @ self.w_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)

    def validate(self, activations, features):
        import torch

        reconstruction = features @ self.w_dec + self.b_dec
        residual_mse = torch.mean((reconstruction - activations.float()) ** 2)
        centred = activations.float() - activations.float().mean(dim=0)
        baseline_mse = torch.mean(centred**2)
        values = {
            "relative_mse": float((residual_mse / baseline_mse).cpu()),
            "explained_variance": float((1 - residual_mse / baseline_mse).cpu()),
            "mean_l0": float((features > 0).sum(dim=1).float().mean().cpu()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"{self.sae_id} validation is non-finite")
        if values["mean_l0"] <= 0:
            raise ValueError(f"{self.sae_id} encoded no active features")
        self.validation_metrics = values


def subset_by_class(rows, per_class):
    selected = []
    for label in (0, 1):
        matching = [
            row for row in rows if int(bool(row["harmful"])) == label
        ]
        selected.extend(matching[:per_class])
    return selected


def select_texts(rows, full_rows, full_texts):
    by_id = {
        str(row["id"]): text for row, text in zip(full_rows, full_texts, strict=True)
    }
    return [by_id[str(row["id"])] for row in rows]


def load_inputs(args):
    rows, parse_errors, malformed, excluded_pilot, excluded_categories = (
        load_judged_rows(
            Path(args.inp), 0, args.seed, keep_protected_group=False
        )
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    tune_plain = [row["prompt"] for row in tune_rows]
    test_plain = [row["prompt"] for row in test_rows]
    translated, truncated = load_or_translate(
        tune_plain + test_plain,
        Path(args.translations_dir) / "swahili.jsonl",
        args.nllb,
        NLLB_CODE,
        allow_translate=False,
    )
    metadata_path = Path(args.translations_dir) / "metadata.json"
    translation_metadata = json.loads(metadata_path.read_text())
    translation_path = Path(args.translations_dir) / "swahili.jsonl"
    if translation_metadata["swahili_sha256"] != file_sha256(translation_path):
        raise ValueError("Swahili translation checksum mismatch")
    if truncated:
        raise ValueError("frozen Swahili inputs unexpectedly include truncation")

    tune_swahili = translated[: len(tune_rows)]
    test_swahili = translated[len(tune_rows) :]
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

    if args.smoke:
        full_tune_rows, full_test_rows = tune_rows, test_rows
        train_rows = subset_by_class(train_rows, args.smoke_per_class)
        tune_rows = subset_by_class(tune_rows, args.smoke_per_class)
        test_rows = subset_by_class(test_rows, args.smoke_per_class)
        tune_conditions = {
            condition: select_texts(tune_rows, full_tune_rows, texts)
            for condition, texts in tune_conditions.items()
        }
        test_conditions = {
            condition: select_texts(test_rows, full_test_rows, texts)
            for condition, texts in test_conditions.items()
        }

    provenance = {
        "input": args.inp,
        "parse_errors_dropped": parse_errors,
        "malformed_lines_skipped": malformed,
        "pilot_overlap_excluded": excluded_pilot,
        "non_operational_positive_categories_excluded": excluded_categories,
        "translation_metadata": translation_metadata,
    }
    return train_rows, tune_rows, test_rows, tune_conditions, test_conditions, provenance


def instruction_positions(encoded, tok):
    end_of_turn_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id == tok.unk_token_id:
        raise ValueError("Gemma tokenizer has no <end_of_turn> token")
    positions = []
    for input_ids, attention_mask in zip(
        encoded["input_ids"], encoded["attention_mask"], strict=True
    ):
        matches = (input_ids == end_of_turn_id).nonzero(as_tuple=False).flatten()
        if len(matches) == 0:
            raise ValueError("rendered prompt contains no <end_of_turn> token")
        position = int(matches[-1]) - 1
        if position < 0 or not bool(attention_mask[position]):
            raise ValueError("invalid t_inst position")
        positions.append(position)
    return positions


def extract(texts, model, tok, saes, batch_size, validate_saes=False):
    import torch

    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    hidden = saes[BLOCKS[0]].w_enc.shape[0]
    dense = {
        block: np.empty((len(texts), hidden), dtype=np.float32) for block in BLOCKS
    }
    sparse_parts = {
        block: {"rows": [], "columns": [], "values": []} for block in BLOCKS
    }
    started = time.time()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch_texts = truncate_left_tokens(
            [texts[index] for index in indices], tok, 2048
        )
        encoded = tok.apply_chat_template(
            [[{"role": "user", "content": text}] for text in batch_texts],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        positions = instruction_positions(encoded, tok)
        with torch.no_grad():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
            for block in BLOCKS:
                states = output.hidden_states[HIDDEN_STATE_INDICES[block]]
                rows = torch.arange(len(indices), device=states.device)
                positions_tensor = torch.as_tensor(positions, device=states.device)
                activations = states[rows, positions_tensor, :].float()
                features = saes[block].encode(activations)
                if validate_saes and saes[block].validation_metrics is None:
                    saes[block].validate(activations, features)
                dense[block][indices] = activations.cpu().numpy()
                nonzero = features.nonzero(as_tuple=False)
                if len(nonzero):
                    sparse_parts[block]["rows"].append(
                        np.asarray(indices, dtype=np.int32)[
                            nonzero[:, 0].cpu().numpy()
                        ]
                    )
                    sparse_parts[block]["columns"].append(
                        nonzero[:, 1].cpu().numpy().astype(np.int32)
                    )
                    sparse_parts[block]["values"].append(
                        features[nonzero[:, 0], nonzero[:, 1]]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
        done = min(start + batch_size, len(order))
        print(
            f"  forwarded+encoded {done}/{len(order)} "
            f"{done / (time.time() - started):.2f}/s",
            flush=True,
        )

    codes = {}
    for block in BLOCKS:
        parts = sparse_parts[block]
        if not parts["values"]:
            raise ValueError(f"block {block} produced no active SAE features")
        codes[block] = sparse.csr_matrix(
            (
                np.concatenate(parts["values"]),
                (
                    np.concatenate(parts["rows"]),
                    np.concatenate(parts["columns"]),
                ),
            ),
            shape=(len(texts), FEATURE_WIDTH),
            dtype=np.float32,
        )
        if not np.isfinite(codes[block].data).all():
            raise ValueError(f"block {block} SAE codes are non-finite")
    return dense, codes


def train_sparse_probe(x_train, y_train, x_tune, y_tune, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False).fit(x_train)
    keep = scaler.var_ > 0
    x_train = scaler.transform(x_train)[:, keep]
    x_tune = scaler.transform(x_tune)[:, keep]
    print(
        f"[SAE probe] dimensions={len(keep)} kept={int(keep.sum())} "
        f"zero_variance={int((~keep).sum())}",
        flush=True,
    )
    candidates = []
    for c_value in C_GRID:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
            max_iter=2000,
        ).fit(x_train, y_train)
        if int(classifier.n_iter_[0]) >= classifier.max_iter:
            raise RuntimeError(f"SAE probe did not converge for C={c_value:g}")
        logits = classifier.decision_function(x_tune)
        losses = np.logaddexp(0.0, logits) - y_tune * logits
        candidates.append(
            (
                c_value,
                float(losses.mean()),
                float(losses.std(ddof=1) / np.sqrt(len(losses))),
                classifier,
            )
        )
        print(
            f"  C={c_value:g} tune_log_loss={candidates[-1][1]:.6f} "
            f"SE={candidates[-1][2]:.6f}",
            flush=True,
        )
    best = min(candidates, key=lambda candidate: candidate[1])
    cutoff = best[1] + best[2]
    selected = min(
        (candidate for candidate in candidates if candidate[1] <= cutoff),
        key=lambda candidate: candidate[0],
    )
    c_value, mean_loss, se_loss, classifier = selected
    weight = np.zeros(len(keep), dtype=np.float32)
    weight[keep] = (
        classifier.coef_[0] / scaler.scale_[keep]
    ).astype(np.float32)
    tuning = [
        {"C": c, "log_loss": loss, "standard_error": se}
        for c, loss, se, _ in candidates
    ]
    print(
        f"[SAE probe] best_C={best[0]:g} selected_C={c_value:g}",
        flush=True,
    )
    return (
        weight,
        float(classifier.intercept_[0]),
        c_value,
        mean_loss,
        se_loss,
        tuning,
    )


def fit_probes(dense_train, codes_train, y_train, dense_tune, codes_tune, y_tune, seed):
    probes = {}
    for block in BLOCKS:
        print(f"[fit] block {block} dense", flush=True)
        dense_result = train_probe(
            dense_train[block].copy(),
            y_train,
            dense_tune[block].copy(),
            y_tune,
            seed,
        )
        print(f"[fit] block {block} SAE", flush=True)
        sae_result = train_sparse_probe(
            codes_train[block], y_train, codes_tune[block], y_tune, seed
        )
        probes[(block, "dense")] = dense_result
        probes[(block, "sae")] = sae_result
    return probes


def score_features(dense, codes, probes):
    scores = {}
    for block in BLOCKS:
        dense_weight, dense_intercept = probes[(block, "dense")][:2]
        sae_weight, sae_intercept = probes[(block, "sae")][:2]
        scores[f"dense_block{block}"] = (
            dense[block] @ dense_weight + dense_intercept
        ).astype(np.float32)
        scores[f"sae_block{block}"] = np.asarray(
            codes[block] @ sae_weight + sae_intercept,
            dtype=np.float32,
        )
    return scores


def feature_statistics(codes_by_condition, labels, probes, test_ids):
    positive = labels == 1
    negative = labels == 0
    contributions = {}
    examples = {}
    for block in BLOCKS:
        weight = probes[(block, "sae")][0]
        examples[str(block)] = {}
        for condition in ("plain", "swahili"):
            codes = codes_by_condition[condition][block]
            difference = np.asarray(
                codes[positive].mean(axis=0) - codes[negative].mean(axis=0)
            ).ravel()
            contributions[(block, condition)] = (weight * difference).astype(
                np.float32
            )
        top_plain = np.argsort(
            np.abs(contributions[(block, "plain")])
        )[-10:][::-1]
        for feature in top_plain:
            feature_record = {}
            for condition in ("plain", "swahili"):
                values = codes_by_condition[condition][block][
                    :, int(feature)
                ].toarray().ravel()
                top_rows = np.argsort(values)[-3:][::-1]
                feature_record[condition] = [
                    {
                        "id": str(test_ids[index]),
                        "activation": float(values[index]),
                    }
                    for index in top_rows
                    if values[index] > 0
                ]
            examples[str(block)][str(int(feature))] = feature_record
    return contributions, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--out", default="data/phase3_scores.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-class", type=int, default=8)
    args = parser.parse_args()

    if args.model != MODEL:
        raise ValueError(f"Phase 3 backbone is frozen to {MODEL}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    (
        train_rows,
        tune_rows,
        test_rows,
        tune_conditions,
        test_conditions,
        provenance,
    ) = load_inputs(args)
    print(
        f"[split] train={len(train_rows)} tune={len(tune_rows)} "
        f"test={len(test_rows)} smoke={args.smoke}",
        flush=True,
    )
    y_train = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    y_tune = np.asarray([int(bool(row["harmful"])) for row in tune_rows])
    y_test = np.asarray([int(bool(row["harmful"])) for row in test_rows])
    train_ids = np.asarray([str(row["id"]) for row in train_rows])
    tune_ids = np.asarray([str(row["id"]) for row in tune_rows])
    test_ids = np.asarray([str(row["id"]) for row in test_rows])

    print("[load] language model", flush=True)
    model, tok, _, hidden = load_model(args.model, args.seed)
    model_revision = getattr(model.config, "_commit_hash", None)
    if model_revision != MODEL_REVISION:
        raise ValueError(
            f"model revision {model_revision!r} differs from frozen {MODEL_REVISION}"
        )
    print("[load] two frozen SAE checkpoints", flush=True)
    saes = {block: JumpReLUSAE(block) for block in BLOCKS}
    if any(sae.w_enc.shape[0] != hidden for sae in saes.values()):
        raise ValueError("SAE input width differs from model hidden size")

    print("[extract] plain train", flush=True)
    dense_train, codes_train = extract(
        [row["prompt"] for row in train_rows],
        model,
        tok,
        saes,
        args.batch_size,
        validate_saes=True,
    )
    print("[extract] plain tune", flush=True)
    dense_tune, codes_tune = extract(
        tune_conditions["plain"], model, tok, saes, args.batch_size
    )
    probes = fit_probes(
        dense_train,
        codes_train,
        y_train,
        dense_tune,
        codes_tune,
        y_tune,
        args.seed,
    )

    scores = {}
    for name, values in score_features(dense_tune, codes_tune, probes).items():
        scores[f"tune_plain_{name}"] = values
    del dense_train, codes_train, dense_tune, codes_tune

    codes_for_stability = {}
    l0 = {}
    for split, conditions in (("tune", tune_conditions), ("test", test_conditions)):
        for condition, texts in conditions.items():
            if split == "tune" and condition == "plain":
                continue
            print(f"[extract+score] {split} {condition}", flush=True)
            dense, codes = extract(texts, model, tok, saes, args.batch_size)
            for name, values in score_features(dense, codes, probes).items():
                scores[f"{split}_{condition}_{name}"] = values
            for block in BLOCKS:
                l0[f"{split}_{condition}_block{block}"] = float(
                    codes[block].getnnz(axis=1).mean()
                )
            if split == "test" and condition in {"plain", "swahili"}:
                codes_for_stability[condition] = codes

    contributions, examples = feature_statistics(
        codes_for_stability, y_test, probes, test_ids
    )
    arrays = {
        "model": np.asarray(args.model),
        "model_revision": np.asarray(model_revision),
        "sae_repo": np.asarray(SAE_REPO),
        "sae_revision": np.asarray(SAE_REVISION),
        "sae_ids_json": np.asarray(json.dumps(SAE_IDS)),
        "seed": np.asarray(args.seed),
        "position": np.asarray("t_inst"),
        "smoke": np.asarray(args.smoke),
        "train_ids": train_ids,
        "tune_ids": tune_ids,
        "test_ids": test_ids,
        "tune_labels": y_tune,
        "test_labels": y_test,
        "test_categories": np.asarray(
            [str(row.get("category")) for row in test_rows]
        ),
        "inputs_sha256": np.asarray(
            strings_sha256(
                [text for values in tune_conditions.values() for text in values]
                + [text for values in test_conditions.values() for text in values]
            )
        ),
        "provenance_json": np.asarray(json.dumps(provenance)),
        "sae_validation_json": np.asarray(
            json.dumps(
                {block: saes[block].validation_metrics for block in BLOCKS}
            )
        ),
        "l0_json": np.asarray(json.dumps(l0)),
        "top_examples_json": np.asarray(json.dumps(examples)),
        **scores,
    }
    for block in BLOCKS:
        for representation in ("dense", "sae"):
            result = probes[(block, representation)]
            arrays[f"{representation}_weight_block{block}"] = result[0]
            arrays[f"{representation}_intercept_block{block}"] = np.asarray(result[1])
            arrays[f"{representation}_selected_c_block{block}"] = np.asarray(result[2])
            arrays[f"{representation}_selected_loss_block{block}"] = np.asarray(result[3])
            arrays[f"{representation}_selected_se_block{block}"] = np.asarray(result[4])
            arrays[f"{representation}_tuning_block{block}"] = np.asarray(
                json.dumps(result[5])
            )
        for condition in ("plain", "swahili"):
            arrays[f"contribution_{condition}_block{block}"] = contributions[
                (block, condition)
            ]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
