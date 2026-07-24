"""Train and score the two frozen Step-3 text baselines on the Phase 1 split."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from capability_check import build_sent
from capability_qa import LANGS
from probe_prompt import (
    C_GRID,
    file_sha256,
    load_judged_rows,
    load_or_translate,
    split_rows,
    strings_sha256,
    train_probe,
)


CONDITIONS = ("plain", "french", "hindi", "swahili", "zulu", "reverse")
EXPECTED_SPLIT = (5341, 1781, 1781)
MAX_LENGTH = 512
SMALL_GUARD_MODEL = "microsoft/deberta-v3-small"
SMALL_GUARD_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"
E5_MODEL = "intfloat/multilingual-e5-base"
E5_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def verify_cached_config(model_id, revision):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    resolved = str(getattr(config, "_commit_hash", "") or "")
    if resolved != revision:
        raise RuntimeError(
            f"cached revision mismatch for {model_id}: {resolved!r} != {revision!r}"
        )
    return config


def select_per_class(rows, n):
    if not n:
        return rows, np.arange(len(rows))
    labels = np.asarray([int(bool(row["harmful"])) for row in rows])
    indices = np.concatenate(
        [np.flatnonzero(labels == label)[:n] for label in (0, 1)]
    )
    indices.sort()
    if len(indices) != 2 * n:
        raise ValueError(f"cannot select {n} rows from each class")
    return [rows[int(index)] for index in indices], indices


def load_frozen_translations(translations_dir, tune_rows, test_rows):
    metadata_path = translations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("nllb_model") != NLLB_MODEL:
        raise ValueError("translation model mismatch")
    if metadata.get("nllb_revision") != NLLB_REVISION:
        raise ValueError("translation revision mismatch")
    verify_cached_config(NLLB_MODEL, NLLB_REVISION)

    prompts = [row["prompt"] for row in tune_rows + test_rows]
    translations = {}
    hashes = {}
    for language in ("french", "hindi", "swahili", "zulu"):
        path = translations_dir / f"{language}.jsonl"
        translated, _ = load_or_translate(
            prompts,
            path,
            NLLB_MODEL,
            LANGS[language],
            allow_translate=False,
        )
        digest = file_sha256(path)
        expected = metadata.get("manifests", {}).get(language)
        if isinstance(expected, dict):
            expected = expected.get("sha256")
        if language == "swahili" and expected is None:
            expected = metadata.get("swahili_sha256")
        if digest != expected:
            raise ValueError(f"checksum mismatch for frozen {language} manifest")
        translations[language] = translated
        hashes[language] = digest
    return translations, hashes


def build_conditions(rows, translations, translated_offset, selected_indices):
    plain = [row["prompt"] for row in rows]
    conditions = {"plain": plain}
    for language in ("french", "hindi", "swahili", "zulu"):
        all_translated = translations[language]
        conditions[language] = [
            all_translated[translated_offset + int(index)] for index in selected_indices
        ]
    conditions["reverse"] = [
        build_sent(text, "reverse", in_obf=True, out_obf=False) for text in plain
    ]
    return conditions


def token_truncation_count(texts, tokenizer, prefix):
    count = 0
    for start in range(0, len(texts), 256):
        batch = [prefix + text for text in texts[start : start + 256]]
        lengths = tokenizer(
            batch, add_special_tokens=True, padding=False, truncation=False
        )["input_ids"]
        count += sum(len(ids) > MAX_LENGTH for ids in lengths)
    return count


def truncation_report(tokenizer, train_rows, tune_conditions, test_conditions, prefix):
    report = {
        "train_plain": token_truncation_count(
            [row["prompt"] for row in train_rows], tokenizer, prefix
        )
    }
    for split, conditions in (("tune", tune_conditions), ("test", test_conditions)):
        for condition, texts in conditions.items():
            report[f"{split}_{condition}"] = token_truncation_count(
                texts, tokenizer, prefix
            )
    return report


def encoded_batch(tokenizer, texts, device, prefix):
    return tokenizer(
        [prefix + text for text in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)


def score_small_guard(texts, model, tokenizer, batch_size, device):
    import torch

    scores = np.empty(len(texts), dtype=np.float32)
    model.eval()
    started = time.time()
    for start in range(0, len(texts), batch_size):
        encoded = encoded_batch(
            tokenizer, texts[start : start + batch_size], device, prefix=""
        )
        with torch.no_grad():
            logits = model(**encoded).logits.float()
        if logits.shape[1] != 2:
            raise ValueError(f"expected two classifier logits, got {tuple(logits.shape)}")
        scores[start : start + len(logits)] = (
            logits[:, 1] - logits[:, 0]
        ).cpu().numpy()
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (20 * batch_size) == 0:
            print(
                f"  scored {done}/{len(texts)} "
                f"{done / max(time.time() - started, 1e-9):.1f}/s",
                flush=True,
            )
    return scores


def train_small_guard(
    train_rows,
    tune_rows,
    model,
    tokenizer,
    batch_size,
    device,
    checkpoint_path,
    seed,
):
    import torch
    import torch.nn.functional as functional

    train_texts = [row["prompt"] for row in train_rows]
    train_labels = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    tune_texts = [row["prompt"] for row in tune_rows]
    tune_labels = np.asarray([int(bool(row["harmful"])) for row in tune_rows])
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.as_tensor(
        len(train_labels) / (2.0 * counts), dtype=torch.float32, device=device
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )
    generator = np.random.default_rng(seed)
    tuning = []
    best_loss = float("inf")
    selected_epoch = None
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 4):
        model.train()
        order = generator.permutation(len(train_rows))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            texts = [train_texts[int(index)] for index in indices]
            labels = torch.as_tensor(
                train_labels[indices], dtype=torch.long, device=device
            )
            optimiser.zero_grad(set_to_none=True)
            logits = model(
                **encoded_batch(tokenizer, texts, device, prefix="")
            ).logits.float()
            loss = functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(indices)

        tune_scores = score_small_guard(
            tune_texts, model, tokenizer, batch_size, device
        )
        losses = np.logaddexp(0.0, tune_scores) - tune_labels * tune_scores
        tune_loss = float(losses.mean())
        tuning.append(
            {
                "epoch": epoch,
                "train_weighted_cross_entropy": total_loss / len(train_rows),
                "plain_tune_log_loss": tune_loss,
            }
        )
        print(
            f"[epoch {epoch}] train_weighted_loss={total_loss / len(train_rows):.6f} "
            f"plain_tune_log_loss={tune_loss:.6f}",
            flush=True,
        )
        if tune_loss < best_loss:
            best_loss = tune_loss
            selected_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    if selected_epoch is None:
        raise RuntimeError("no DeBERTa checkpoint was selected")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(
        f"[checkpoint] restored epoch={selected_epoch} "
        f"plain_tune_log_loss={best_loss:.6f} from {checkpoint_path}",
        flush=True,
    )
    return selected_epoch, tuning, counts.tolist()


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    pooled = (last_hidden_state * mask).sum(dim=1) / denominator
    normalised = torch.nn.functional.normalize(pooled, p=2, dim=1)
    if not torch.isfinite(normalised).all():
        raise ValueError("non-finite E5 embedding")
    return normalised


def extract_e5(texts, model, tokenizer, batch_size, device):
    import torch

    embeddings = np.empty((len(texts), 768), dtype=np.float32)
    model.eval()
    started = time.time()
    for start in range(0, len(texts), batch_size):
        encoded = encoded_batch(
            tokenizer, texts[start : start + batch_size], device, prefix="query: "
        )
        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        if pooled.shape[1] != 768:
            raise ValueError(f"expected 768 E5 dimensions, got {pooled.shape[1]}")
        embeddings[start : start + len(pooled)] = pooled.float().cpu().numpy()
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (20 * batch_size) == 0:
            print(
                f"  embedded {done}/{len(texts)} "
                f"{done / max(time.time() - started, 1e-9):.1f}/s",
                flush=True,
            )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("E5 embeddings are not L2-normalised")
    return embeddings


def load_encoder(baseline, device):
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if baseline == "small_guard":
        model_id, revision = SMALL_GUARD_MODEL, SMALL_GUARD_REVISION
        verify_cached_config(model_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            num_labels=2,
        )
    else:
        model_id, revision = E5_MODEL, E5_REVISION
        verify_cached_config(model_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
    resolved = str(getattr(model.config, "_commit_hash", "") or "")
    if resolved != revision:
        raise RuntimeError(f"loaded model revision {resolved!r} != {revision!r}")
    model.to(device)
    return model, tokenizer, model_id, revision


def save_scores(
    output_path,
    baseline,
    model_id,
    revision,
    seed,
    smoke_per_class,
    train_rows,
    tune_rows,
    test_rows,
    tune_scores,
    test_scores,
    translation_hashes,
    truncation,
    training,
    extra_arrays,
):
    tune_labels = np.asarray([int(bool(row["harmful"])) for row in tune_rows])
    test_labels = np.asarray([int(bool(row["harmful"])) for row in test_rows])
    arrays = {
        "baseline": np.asarray(baseline),
        "model": np.asarray(model_id),
        "model_revision": np.asarray(revision),
        "seed": np.asarray(seed, dtype=np.int64),
        "smoke_per_class": np.asarray(smoke_per_class, dtype=np.int64),
        "train_ids": np.asarray([str(row["id"]) for row in train_rows]),
        "tune_ids": np.asarray([str(row["id"]) for row in tune_rows]),
        "test_ids": np.asarray([str(row["id"]) for row in test_rows]),
        "train_labels": np.asarray(
            [int(bool(row["harmful"])) for row in train_rows]
        ),
        "tune_labels": tune_labels,
        "test_labels": test_labels,
        "train_categories": np.asarray(
            [str(row.get("category") or "none") for row in train_rows]
        ),
        "tune_categories": np.asarray(
            [str(row.get("category") or "none") for row in tune_rows]
        ),
        "test_categories": np.asarray(
            [str(row.get("category") or "none") for row in test_rows]
        ),
        "translation_hashes_json": np.asarray(json.dumps(translation_hashes)),
        "truncation_counts_json": np.asarray(json.dumps(truncation)),
        "training_json": np.asarray(json.dumps(training)),
        **extra_arrays,
    }
    for split, scores in (("tune", tune_scores), ("test", test_scores)):
        for condition in CONDITIONS:
            values = np.asarray(scores[condition], dtype=np.float32)
            expected = len(tune_rows) if split == "tune" else len(test_rows)
            if values.shape != (expected,) or not np.isfinite(values).all():
                raise ValueError(
                    f"invalid {split} {condition} scores: {values.shape}"
                )
            arrays[f"{split}_{condition}_scores"] = values
    arrays["scores_sha256"] = np.asarray(
        hashlib.sha256(
            b"".join(
                arrays[f"{split}_{condition}_scores"].tobytes()
                for split in ("tune", "test")
                for condition in CONDITIONS
            )
        ).hexdigest()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"[done] {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", required=True, choices=("small_guard", "multilingual_e5")
    )
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--out")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-per-class", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("the frozen experiment requires --seed 0")
    if args.smoke_per_class < 0:
        raise ValueError("--smoke-per-class must be non-negative")
    if args.smoke_per_class and not args.out:
        raise ValueError("smoke runs require an explicit --out")
    if args.baseline == "small_guard" and args.smoke_per_class and not args.checkpoint_out:
        raise ValueError("small-guard smoke runs require an explicit --checkpoint-out")

    defaults = {
        "small_guard": ("data/phase1_small_guard.npz", 16),
        "multilingual_e5": ("data/phase1_multilingual_e5.npz", 32),
    }
    output_path = Path(args.out or defaults[args.baseline][0])
    batch_size = args.batch_size or defaults[args.baseline][1]
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint_path = Path(
        args.checkpoint_out or "data/phase1_small_guard_checkpoint.pt"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    from transformers import set_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_seed(args.seed)

    rows, _, _, _, _ = load_judged_rows(
        Path(args.inp), 0, args.seed, keep_protected_group=False
    )
    full_train, full_tune, full_test = split_rows(rows, args.seed)
    split_sizes = (len(full_train), len(full_tune), len(full_test))
    if split_sizes != EXPECTED_SPLIT:
        raise ValueError(f"unexpected frozen split sizes: {split_sizes}")
    translations, translation_hashes = load_frozen_translations(
        Path(args.translations_dir), full_tune, full_test
    )
    train_rows, _ = select_per_class(full_train, args.smoke_per_class)
    tune_rows, tune_indices = select_per_class(full_tune, args.smoke_per_class)
    test_rows, test_indices = select_per_class(full_test, args.smoke_per_class)
    tune_conditions = build_conditions(
        tune_rows, translations, translated_offset=0, selected_indices=tune_indices
    )
    test_conditions = build_conditions(
        test_rows,
        translations,
        translated_offset=len(full_tune),
        selected_indices=test_indices,
    )
    print(
        f"[split] train={len(train_rows)} tune={len(tune_rows)} "
        f"test={len(test_rows)} smoke_per_class={args.smoke_per_class}",
        flush=True,
    )

    model, tokenizer, model_id, revision = load_encoder(args.baseline, device)
    prefix = "" if args.baseline == "small_guard" else "query: "
    truncation = truncation_report(
        tokenizer, train_rows, tune_conditions, test_conditions, prefix
    )
    print(
        f"[tokenisation] max_length={MAX_LENGTH} "
        f"truncated_total={sum(truncation.values())}",
        flush=True,
    )
    labels_train = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    labels_tune = np.asarray([int(bool(row["harmful"])) for row in tune_rows])

    if args.baseline == "small_guard":
        selected_epoch, tuning, class_counts = train_small_guard(
            train_rows,
            tune_rows,
            model,
            tokenizer,
            batch_size,
            device,
            checkpoint_path,
            args.seed,
        )
        tune_scores = {}
        test_scores = {}
        for split, conditions, target in (
            ("tune", tune_conditions, tune_scores),
            ("test", test_conditions, test_scores),
        ):
            for condition, texts in conditions.items():
                print(f"[score] {split} {condition}", flush=True)
                target[condition] = score_small_guard(
                    texts, model, tokenizer, batch_size, device
                )
        training = {
            "condition": "plain train only",
            "max_length": MAX_LENGTH,
            "optimiser": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "loss": "balanced two-class cross-entropy",
            "class_counts": class_counts,
            "epochs": 3,
            "selection": "minimum plain-tune unweighted log loss; earliest exact tie",
            "selected_epoch": selected_epoch,
            "tuning": tuning,
            "checkpoint": str(checkpoint_path),
        }
        extra_arrays = {
            "selected_epoch": np.asarray(selected_epoch, dtype=np.int64),
            "checkpoint_path": np.asarray(str(checkpoint_path)),
        }
    else:
        print("[embed] plain train", flush=True)
        train_embeddings = extract_e5(
            [row["prompt"] for row in train_rows],
            model,
            tokenizer,
            batch_size,
            device,
        )
        print("[embed] plain tune", flush=True)
        tune_embeddings = extract_e5(
            tune_conditions["plain"], model, tokenizer, batch_size, device
        )
        weight, intercept, selected_c, selected_loss, selected_se, tuning = train_probe(
            train_embeddings.copy(),
            labels_train,
            tune_embeddings.copy(),
            labels_tune,
            args.seed,
        )
        del train_embeddings
        tune_scores = {
            "plain": (tune_embeddings @ weight + intercept).astype(np.float32)
        }
        del tune_embeddings
        test_scores = {}
        for split, conditions, target in (
            ("tune", tune_conditions, tune_scores),
            ("test", test_conditions, test_scores),
        ):
            for condition, texts in conditions.items():
                if split == "tune" and condition == "plain":
                    continue
                print(f"[embed] {split} {condition}", flush=True)
                embeddings = extract_e5(
                    texts, model, tokenizer, batch_size, device
                )
                target[condition] = (
                    embeddings @ weight + intercept
                ).astype(np.float32)
        training = {
            "condition": "plain train only",
            "input_prefix": "query: ",
            "max_length": MAX_LENGTH,
            "pooling": "attention-mask mean pooling then L2 normalisation",
            "embedding_dimension": 768,
            "classifier": "balanced logistic regression",
            "C_grid": list(C_GRID),
            "selection": "existing train_probe one-standard-error procedure",
            "selected_C": selected_c,
            "selected_plain_tune_log_loss": selected_loss,
            "selected_plain_tune_log_loss_standard_error": selected_se,
            "tuning": tuning,
        }
        extra_arrays = {
            "probe_weight": weight,
            "probe_intercept": np.asarray(intercept, dtype=np.float64),
            "selected_c": np.asarray(selected_c, dtype=np.float64),
        }

    save_scores(
        output_path,
        args.baseline,
        model_id,
        revision,
        args.seed,
        args.smoke_per_class,
        train_rows,
        tune_rows,
        test_rows,
        tune_scores,
        test_scores,
        translation_hashes,
        truncation,
        training,
        {
            "inputs_sha256": np.asarray(
                strings_sha256(
                    [row["prompt"] for row in train_rows]
                    + [
                        text
                        for conditions in (tune_conditions, test_conditions)
                        for condition in CONDITIONS
                        for text in conditions[condition]
                    ]
                )
            ),
            **extra_arrays,
        },
    )


if __name__ == "__main__":
    main()
