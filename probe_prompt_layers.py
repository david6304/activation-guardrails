"""Diagnose per-layer transfer for the operational-harm prompt probe.

All layer probes are trained independently on the same plain train split used by
``probe_prompt.py`` and scored descriptively on the fixed plain/transform test split.
Only pre-specified hidden-state indices 32, 34, and 41 receive WildChat calibration,
preventing selection of an operational detector on transformed-test performance. Indices
32 and 41 are residual outputs from zero-based blocks 31 and 40; index 34 is fixed by the
earlier reverse-mechanism pilot's own hidden-state indexing.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from capability_check import build_sent
from probe_prompt import (
    LANGS,
    MAX_INPUT_TOKENS,
    condition_metrics,
    file_sha256,
    load_judged_rows,
    load_model,
    load_or_translate,
    load_wildchat_prompts,
    split_rows,
    strings_sha256,
    train_probe,
    truncate_left_tokens,
)


CALIBRATION_LAYERS = (32, 34, 41)


def extract_last_token_layers(texts, model, tok, batch_size, layer_count, hidden):
    """Extract [row, hidden-state index, dimension] safely from sharded models."""
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    features = np.empty((len(texts), layer_count, hidden), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch_texts = truncate_left_tokens(
            [texts[i] for i in idx], tok, MAX_INPUT_TOKENS
        )
        msgs = [[{"role": "user", "content": text}] for text in batch_texts]
        enc = tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
        if len(out.hidden_states) != layer_count:
            raise ValueError("model returned an unexpected hidden-state count")
        # A device-mapped model can return different layers on different GPUs. Moving
        # each small last-token slice to CPU before assignment avoids cross-device stack.
        for layer, states in enumerate(out.hidden_states):
            features[idx, layer, :] = states[:, -1, :].float().cpu().numpy()
        done = min(start + batch_size, len(order))
        print(
            f"  extracted {done}/{len(order)}  {done / (time.time() - t0):.2f}/s",
            flush=True,
        )
    return features


def score_layer_probes(
    texts,
    model,
    tok,
    batch_size,
    weights,
    intercepts,
    layer_indices,
    checkpoint_path=None,
    item_ids=None,
    resume=False,
):
    """Score one independently trained probe at each requested hidden-state index."""
    import torch

    layer_indices = np.asarray(layer_indices, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[0] != len(layer_indices):
        raise ValueError("one weight vector is required per layer index")
    if intercepts.shape != (len(layer_indices),):
        raise ValueError("one intercept is required per layer index")

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores = np.full((len(texts), len(layer_indices)), np.nan, dtype=np.float32)
    completed = 0
    ids_hash = strings_sha256(item_ids) if item_ids is not None else None
    probe_hash = hashlib.sha256(
        weights.tobytes() + intercepts.tobytes() + layer_indices.tobytes()
    ).hexdigest()
    if resume and checkpoint_path and checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            if str(saved["ids_sha256"]) != ids_hash:
                raise ValueError(f"item mismatch in {checkpoint_path}")
            if str(saved["probe_sha256"]) != probe_hash:
                raise ValueError(f"probe mismatch in {checkpoint_path}")
            scores = saved["scores"]
            completed = int(saved["completed"])
        if scores.shape != (len(texts), len(layer_indices)):
            raise ValueError(f"score shape mismatch in {checkpoint_path}")
        if not 0 <= completed <= len(order):
            raise ValueError(f"invalid progress in {checkpoint_path}")
        if np.isnan(scores[order[:completed]]).any():
            raise ValueError(f"missing completed scores in {checkpoint_path}")
        print(f"  resumed {completed}/{len(order)} from {checkpoint_path}", flush=True)

    def save_progress(done):
        if not checkpoint_path:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(".tmp.npz")
        np.savez(
            tmp_path,
            scores=scores,
            completed=np.asarray(done, dtype=np.int64),
            ids_sha256=np.asarray(ids_hash),
            probe_sha256=np.asarray(probe_hash),
        )
        tmp_path.replace(checkpoint_path)

    selected_layers = layer_indices.tolist()
    t0 = time.time()
    for start in range(completed, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch_texts = truncate_left_tokens(
            [texts[i] for i in idx], tok, MAX_INPUT_TOKENS
        )
        msgs = [[{"role": "user", "content": text}] for text in batch_texts]
        enc = tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
        batch_scores = np.empty((len(idx), len(selected_layers)), dtype=np.float32)
        # As in extraction, score each layer after moving only its last-token slice to
        # CPU. This is a small transfer and works whether adjacent layers share a GPU.
        for column, layer in enumerate(selected_layers):
            last = out.hidden_states[layer][:, -1, :].float().cpu().numpy()
            batch_scores[:, column] = last @ weights[column] + intercepts[column]
        scores[idx] = batch_scores
        done = min(start + batch_size, len(order))
        if done == len(order) or done // 500 > start // 500:
            save_progress(done)
            print(
                f"  scored {done}/{len(order)}  "
                f"{(done - completed) / (time.time() - t0):.2f}/s",
                flush=True,
            )
    return scores


def layer_aurocs(labels, condition_scores, trained_mask):
    from sklearn.metrics import roc_auc_score

    return {
        condition: [
            float(roc_auc_score(labels, scores[:, layer]))
            if trained_mask[layer]
            else None
            for layer in range(scores.shape[1])
        ]
        for condition, scores in condition_scores.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--translations-dir", default="data/probe_prompt_translations"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume-diagnostic", action="store_true")
    parser.add_argument("--scores-out")
    parser.add_argument("--report-out")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    model_slug = args.model.rstrip("/").split("/")[-1]
    scores_path = Path(
        args.scores_out or f"data/probe_prompt_layers_{model_slug}_scores.npz"
    )
    report_path = Path(
        args.report_out or f"data/probe_prompt_layers_{model_slug}_report.json"
    )
    diagnostic_checkpoint = scores_path.with_name(
        f"{scores_path.stem}_checkpoint.npz"
    )
    wildchat_checkpoint = scores_path.with_name(
        f"{scores_path.stem}_wildchat_checkpoint.npz"
    )

    rows, parse_errors, malformed, excluded_pilot, excluded_categories = load_judged_rows(
        Path(args.inp), args.limit, args.seed, keep_protected_group=False
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    labels = np.asarray([int(bool(row["harmful"])) for row in test_rows])
    categories = np.asarray([str(row.get("category") or "none") for row in test_rows])
    test_ids = np.asarray([str(row["id"]) for row in test_rows])
    print(
        f"[split] train={len(train_rows)} tune={len(tune_rows)} test={len(test_rows)}",
        flush=True,
    )

    translations_dir = Path(args.translations_dir)
    metadata_path = translations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing frozen translation metadata: {metadata_path}")
    translation_metadata = json.loads(metadata_path.read_text())
    if translation_metadata["nllb_model"] != args.nllb:
        raise ValueError("frozen translations use a different NLLB model")

    test_plain = [row["prompt"] for row in test_rows]
    test_texts = {
        "plain": test_plain,
        "reverse": [
            build_sent(prompt, "reverse", in_obf=True, out_obf=False)
            for prompt in test_plain
        ],
    }
    translation_reports = {}
    for language, code in LANGS.items():
        path = translations_dir / f"{language}.jsonl"
        translated, truncated = load_or_translate(
            test_plain, path, args.nllb, code, allow_translate=False
        )
        digest = file_sha256(path)
        if digest != translation_metadata["manifests"].get(language):
            raise ValueError(f"checksum mismatch for frozen manifest {path}")
        test_texts[language] = translated
        translation_reports[language] = {
            "code": code,
            "manifest": str(path),
            "sha256": digest,
            "inputs_exceeding_256_tokens": len(truncated),
        }

    test_inputs_hash = strings_sha256(
        [text for condition in ("plain", "reverse", *LANGS) for text in test_texts[condition]]
    )
    model, tok, n_layers, hidden = load_model(args.model, args.seed)
    model_revision = getattr(model.config, "_commit_hash", None)
    layer_count = n_layers + 1
    if max(CALIBRATION_LAYERS) >= layer_count:
        raise ValueError(
            f"pre-specified calibration layer exceeds available index {layer_count - 1}"
        )
    print(
        f"[features] hidden-state indices=0..{n_layers} hidden={hidden}", flush=True
    )

    if args.resume_diagnostic:
        with np.load(diagnostic_checkpoint, allow_pickle=False) as saved:
            if str(saved["model"]) != args.model:
                raise ValueError("checkpoint model mismatch")
            if str(saved["model_revision"]) != str(model_revision or ""):
                raise ValueError("checkpoint model revision mismatch")
            if int(saved["seed"]) != args.seed:
                raise ValueError("checkpoint seed mismatch")
            if not np.array_equal(saved["ids"], test_ids):
                raise ValueError("checkpoint test split mismatch")
            if str(saved["test_inputs_sha256"]) != test_inputs_hash:
                raise ValueError("checkpoint test inputs mismatch")
            weights = saved["weights"]
            intercepts = saved["intercepts"]
            selected_cs = saved["selected_cs"]
            selected_losses = saved["selected_losses"]
            selected_ses = saved["selected_ses"]
            trained_mask = saved["trained_mask"]
            tuning_json = saved["tuning_json"]
            condition_scores = {
                condition: saved[f"{condition}_scores"]
                for condition in ("plain", "reverse", *LANGS)
            }
        print(f"[resume] diagnostic <- {diagnostic_checkpoint}", flush=True)
    else:
        print("[extract] plain train", flush=True)
        x_train = extract_last_token_layers(
            [row["prompt"] for row in train_rows],
            model,
            tok,
            args.batch_size,
            layer_count,
            hidden,
        )
        print("[extract] plain tune", flush=True)
        x_tune = extract_last_token_layers(
            [row["prompt"] for row in tune_rows],
            model,
            tok,
            args.batch_size,
            layer_count,
            hidden,
        )
        y_train = np.asarray([int(bool(row["harmful"])) for row in train_rows])
        y_tune = np.asarray([int(bool(row["harmful"])) for row in tune_rows])

        weights = np.empty((layer_count, hidden), dtype=np.float32)
        intercepts = np.empty(layer_count, dtype=np.float64)
        selected_cs = np.empty(layer_count, dtype=np.float64)
        selected_losses = np.empty(layer_count, dtype=np.float64)
        selected_ses = np.empty(layer_count, dtype=np.float64)
        trained_mask = np.ones(layer_count, dtype=bool)
        tuning = []
        for layer in range(layer_count):
            print(f"[probe] hidden-state index {layer}/{n_layers}", flush=True)
            if not np.any(np.var(x_train[:, layer, :], axis=0) > 0):
                print("  skipped: every dimension has zero train variance", flush=True)
                weights[layer] = 0
                intercepts[layer] = 0
                selected_cs[layer] = np.nan
                selected_losses[layer] = np.nan
                selected_ses[layer] = np.nan
                trained_mask[layer] = False
                tuning.append([])
                continue
            result = train_probe(
                x_train[:, layer, :].copy(),
                y_train,
                x_tune[:, layer, :].copy(),
                y_tune,
                args.seed,
            )
            (
                weights[layer],
                intercepts[layer],
                selected_cs[layer],
                selected_losses[layer],
                selected_ses[layer],
                layer_tuning,
            ) = result
            tuning.append(layer_tuning)
        del x_train, x_tune
        tuning_json = np.asarray([json.dumps(value) for value in tuning])

        condition_scores = {}
        all_layers = np.arange(layer_count)
        for condition, texts in test_texts.items():
            print(f"[score] test {condition}", flush=True)
            condition_scores[condition] = score_layer_probes(
                texts,
                model,
                tok,
                args.batch_size,
                weights,
                intercepts,
                all_layers,
            )
        diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            diagnostic_checkpoint,
            model=np.asarray(args.model),
            model_revision=np.asarray(model_revision or ""),
            seed=np.asarray(args.seed, dtype=np.int64),
            ids=test_ids,
            test_inputs_sha256=np.asarray(test_inputs_hash),
            weights=weights,
            intercepts=intercepts,
            selected_cs=selected_cs,
            selected_losses=selected_losses,
            selected_ses=selected_ses,
            trained_mask=trained_mask,
            tuning_json=tuning_json,
            **{
                f"{condition}_scores": scores
                for condition, scores in condition_scores.items()
            },
        )
        print(f"[checkpoint] diagnostic -> {diagnostic_checkpoint}", flush=True)

    aurocs = layer_aurocs(labels, condition_scores, trained_mask)
    wildchat_n = min(100000, args.limit) if args.limit else 100000
    wildchat_ids, wildchat_prompts = load_wildchat_prompts(wildchat_n, args.seed)
    candidate_layers = np.asarray(CALIBRATION_LAYERS, dtype=np.int64)
    print(
        f"[score] WildChat n={wildchat_n} layers={candidate_layers.tolist()}", flush=True
    )
    wildchat_scores = score_layer_probes(
        wildchat_prompts,
        model,
        tok,
        args.batch_size,
        weights[candidate_layers],
        intercepts[candidate_layers],
        candidate_layers,
        checkpoint_path=wildchat_checkpoint,
        item_ids=wildchat_ids,
        resume=args.resume_diagnostic,
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(wildchat_n)
    calibration_index, audit_index = np.array_split(order, 2)

    candidate_reports = {}
    for column, layer in enumerate(candidate_layers):
        calibration_scores = wildchat_scores[calibration_index, column]
        threshold = float(np.quantile(calibration_scores, 0.999, method="higher"))
        plain_tpr = float(
            (condition_scores["plain"][:, layer][labels == 1] > threshold).mean()
        )
        candidate_reports[str(layer)] = {
            "threshold": threshold,
            "calibration_alert_rate": float((calibration_scores > threshold).mean()),
            "audit_alert_rate": float(
                (wildchat_scores[audit_index, column] > threshold).mean()
            ),
            "metrics": {
                condition: condition_metrics(
                    labels,
                    categories,
                    scores[:, layer],
                    threshold,
                    plain_tpr,
                )
                for condition, scores in condition_scores.items()
            },
        }

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores_path,
        ids=test_ids,
        labels=labels,
        categories=categories,
        layer_indices=np.arange(layer_count),
        weights=weights,
        intercepts=intercepts,
        selected_cs=selected_cs,
        selected_losses=selected_losses,
        selected_ses=selected_ses,
        trained_mask=trained_mask,
        wildchat_ids=np.asarray(wildchat_ids),
        wildchat_candidate_layers=candidate_layers,
        wildchat_scores=wildchat_scores,
        wildchat_calibration_index=calibration_index,
        wildchat_audit_index=audit_index,
        **{
            f"{condition}_scores": scores
            for condition, scores in condition_scores.items()
        },
    )
    report = {
        "diagnostic_status": "descriptive; all-layer probe remains the primary detector",
        "model": args.model,
        "model_revision": model_revision,
        "input": args.inp,
        "seed": args.seed,
        "split": {
            "train": len(train_rows),
            "tune": len(tune_rows),
            "test": len(test_rows),
            "parse_errors_dropped": parse_errors,
            "malformed_lines_skipped": malformed,
            "pilot_overlap_excluded": excluded_pilot,
            "non_operational_positive_categories_excluded": excluded_categories,
        },
        "features": {
            "pooling": "prompt-final token",
            "hidden_state_index": "0=embedding; 1..L=successive block outputs",
            "count": layer_count,
            "hidden_size": hidden,
            "dtype": "float32",
        },
        "probe": {
            "training_condition": "plain train only; one independent probe per layer",
            "class_weight": "balanced",
            "selection": "C tuned on plain tune log-loss by the same one-SE rule as primary",
            "untrained_hidden_state_indices": np.flatnonzero(~trained_mask).tolist(),
            "selected_C": [
                float(value) if np.isfinite(value) else None for value in selected_cs
            ],
            "selected_tune_log_loss": [
                float(value) if np.isfinite(value) else None
                for value in selected_losses
            ],
            "selected_tune_log_loss_standard_error": [
                float(value) if np.isfinite(value) else None for value in selected_ses
            ],
        },
        "translations": translation_reports,
        "descriptive_aurocs": aurocs,
        "wildchat": {
            "source": "WildChat-1M first user turn",
            "calibration_n": len(calibration_index),
            "audit_n": len(audit_index),
            "target_alert_rate": 0.001,
            "quantile_method": "higher",
            "comparison": "score > threshold",
            "pre_specified_layers": list(CALIBRATION_LAYERS),
            "rationale": {
                "32": "residual output from Gemma Scope/SAE block 31",
                "34": "fixed by the earlier reverse-mechanism pilot",
                "41": "residual output from Gemma Scope/SAE block 40",
            },
            "layers": candidate_reports,
        },
        "scores": str(scores_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\n[report]", flush=True)
    for layer, result in candidate_reports.items():
        print(
            f"  layer {layer}: threshold={result['threshold']:.6g} "
            f"audit={result['audit_alert_rate']:.3%}",
            flush=True,
        )
        for condition, metrics in result["metrics"].items():
            print(
                f"    {condition:7s} AUROC={metrics['auroc']:.4f} "
                f"TPR={metrics['tpr']:.3%} retention={metrics['retention_ratio']}",
                flush=True,
            )
    print(f"[done] scores -> {scores_path}", flush=True)
    print(f"[done] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
