"""P2 stage 3: score every response position with the frozen probe, in one pass.

Teacher-forces prompt+response through the **protected** 27B and applies the
frozen prompt-trained probe in memory at every response position. One forward
pass per example gives the whole latency curve; no activation is ever persisted,
only scalars.

Two things this file is careful about.

*The probe is never rebuilt.* The all-layer logistic probe is a single weight
over the concatenation of every layer's hidden state at one position, so the
score at response position k is just a sum of per-layer dot products. Reshaping
the frozen weight to [layer, hidden] and accumulating over layers computes it
without materialising a [positions, layers, hidden] tensor.

*k is defined in the reader's tokenisation.* `generate_responses.py` saves
decoded text, so re-tokenising on the reader side can move prefix boundaries.
Every horizon k here counts protected-reader tokens, and the reader's response
token ids are saved so the text comparators construct their prefixes by decoding
from exactly these boundaries rather than tokenising the text themselves.

k=0 is the prompt-only score at `t_inst` -- the baseline the latency claim has to
beat, saved explicitly.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from probe_prompt import file_sha256, load_model


EXPECTED_MODEL = "google/gemma-3-27b-it"
EXPECTED_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
# Pre-declared horizon: the full generation budget. A shorter horizon would
# selectively drop late harm, which the length sweep showed is real.
K_MAX = 512


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def tokenise_example(row, tokenizer, k_max):
    """Prompt ids exactly as generated, plus the reader's response ids."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row.get("prompt_sent", row["prompt"])}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    return prompt_ids, response_ids[:k_max]


def instruction_position(prompt_ids, end_of_turn_id):
    matches = [index for index, token in enumerate(prompt_ids) if token == end_of_turn_id]
    if not matches:
        raise ValueError("rendered prompt contains no <end_of_turn> token")
    return matches[-1] - 1


def score_positions(hidden_states, rows, positions, weight, intercept,
                    harmful_centroid, harmless_centroid):
    """Probe and centroid scores at arbitrary (row, position) pairs, on device.

    ``weight`` is [layer, hidden] and ``*_centroid`` are [layer-1, hidden], matching
    the frozen artefacts. Accumulating layer by layer keeps peak memory at one
    layer's slice instead of the whole [positions, layers, hidden] block.
    """
    import torch

    count = len(rows)
    # With device_map="auto" over two cards the model shards, so hidden states for
    # different layers live on different devices (docs/EDDIE.md, multi-GPU caveat).
    # Accumulate on one device and move each layer's slice and parameters to it.
    device = weight.device
    logistic = torch.full((count,), float(intercept), dtype=torch.float32, device=device)
    harmful_cosine = torch.zeros(count, dtype=torch.float32, device=device)
    harmless_cosine = torch.zeros_like(harmful_cosine)
    for layer, states in enumerate(hidden_states):
        layer_rows = rows.to(states.device)
        layer_positions = positions.to(states.device)
        slice_ = states[layer_rows, layer_positions, :].float().to(device)
        logistic += slice_ @ weight[layer]
        if layer == 0:
            continue  # the centroid is fitted on transformer outputs 1..L
        centroid_index = layer - 1
        norm = slice_.norm(dim=1).clamp_min(1e-12)
        harmful_cosine += (slice_ @ harmful_centroid[centroid_index]) / (
            norm * harmful_centroid[centroid_index].norm().clamp_min(1e-12)
        )
        harmless_cosine += (slice_ @ harmless_centroid[centroid_index]) / (
            norm * harmless_centroid[centroid_index].norm().clamp_min(1e-12)
        )
    layer_count = len(hidden_states) - 1
    centroid = (harmful_cosine - harmless_cosine) / layer_count
    return logistic, centroid


def run(rows, model, tokenizer, frozen, batch_size, k_max):
    import torch

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    pad_id = tokenizer.pad_token_id
    device = model.device
    weight = torch.as_tensor(frozen["weight"], device=device)
    harmful_centroid = torch.as_tensor(frozen["harmful_centroid"], device=device)
    harmless_centroid = torch.as_tensor(frozen["harmless_centroid"], device=device)
    intercept = frozen["intercept"]

    count = len(rows)
    output = {
        "prompt_logistic": np.zeros(count, dtype=np.float32),
        "prompt_centroid": np.zeros(count, dtype=np.float32),
        "response_logistic": np.full((count, k_max), np.nan, dtype=np.float32),
        "response_centroid": np.full((count, k_max), np.nan, dtype=np.float32),
        "response_token_ids": np.full((count, k_max), -1, dtype=np.int32),
        "response_length": np.zeros(count, dtype=np.int32),
    }

    encoded = []
    for index, row in enumerate(rows):
        prompt_ids, response_ids = tokenise_example(row, tokenizer, k_max)
        encoded.append((index, prompt_ids, response_ids))
        output["response_length"][index] = len(response_ids)
        if response_ids:
            output["response_token_ids"][index, : len(response_ids)] = response_ids

    # Length-sort so each batch pads to a similar length.
    encoded.sort(key=lambda item: len(item[1]) + len(item[2]))
    started = time.time()
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        sequences = [prompt + response for _, prompt, response in batch]
        width = max(len(sequence) for sequence in sequences)
        # Right padding: causal attention means trailing pads cannot affect any
        # earlier position, and it keeps every k index absolute from the start.
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.as_tensor(sequence)
            attention_mask[row, : len(sequence)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            forward = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )

        rows_index = []
        positions = []
        targets = []
        for row, (index, prompt_ids, response_ids) in enumerate(batch):
            rows_index.append(row)
            positions.append(instruction_position(prompt_ids, end_of_turn_id))
            targets.append((index, -1))
            for offset in range(len(response_ids)):
                rows_index.append(row)
                positions.append(len(prompt_ids) + offset)
                targets.append((index, offset))
        logistic, centroid = score_positions(
            forward.hidden_states,
            torch.as_tensor(rows_index, device=device),
            torch.as_tensor(positions, device=device),
            weight,
            intercept,
            harmful_centroid,
            harmless_centroid,
        )
        logistic = logistic.cpu().numpy()
        centroid = centroid.cpu().numpy()
        for slot, (index, offset) in enumerate(targets):
            if offset < 0:
                output["prompt_logistic"][index] = logistic[slot]
                output["prompt_centroid"][index] = centroid[slot]
            else:
                output["response_logistic"][index, offset] = logistic[slot]
                output["response_centroid"][index, offset] = centroid[slot]

        done = min(start + batch_size, len(encoded))
        print(
            f"  scored {done}/{len(encoded)}  {done / (time.time() - started):.2f}/s",
            flush=True,
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/p2_judged.jsonl")
    parser.add_argument(
        "--activation", default="data/phase1_activation_multilingual_27b.npz"
    )
    parser.add_argument("--out", default="data/p2_latency_scores.npz")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N rows (smoke)")
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    rows = [json.loads(line) for line in Path(args.inp).open() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    missing = [row["id"] for row in rows if not row.get("response")]
    if missing:
        raise ValueError(f"{len(missing)} rows have no response, first={missing[0]}")
    print(f"[rows] {len(rows)} responses from {args.inp}", flush=True)

    activation = load_npz(Path(args.activation))
    if str(activation["model_revision"]) != EXPECTED_REVISION:
        raise ValueError("frozen activation revision mismatch")

    model, tokenizer, num_layers, hidden_size = load_model(EXPECTED_MODEL, args.seed)
    loaded_revision = str(getattr(model.config, "_commit_hash", "") or "")
    if loaded_revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"loaded Gemma revision {loaded_revision!r} != {EXPECTED_REVISION!r}"
        )
    weight = activation["logistic_weight"]
    if weight.shape != ((num_layers + 1) * hidden_size,):
        raise ValueError("frozen logistic weight dimension does not match the model")
    frozen = {
        "weight": weight.reshape(num_layers + 1, hidden_size),
        "intercept": float(activation["logistic_intercept"]),
        "harmful_centroid": activation["harmful_centroid"],
        "harmless_centroid": activation["harmless_centroid"],
    }

    scores = run(rows, model, tokenizer, frozen, args.batch_size, args.k_max)

    metadata = {
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "reader": "protected gemma-3-27b-it, teacher-forced, one pass per example",
        "generator": sorted({str(row.get("generator", "")) for row in rows}),
        "k_max": args.k_max,
        "k_definition": "protected-reader response tokens; k=0 is the prompt-only score",
        "sequence_rule": "max score so far",
        "input": str(Path(args.inp)),
        "input_sha256": file_sha256(Path(args.inp)),
        "frozen_activation_sha256": file_sha256(Path(args.activation)),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "training": "none; frozen prompt-trained probe applied off-distribution",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray([str(row["id"]) for row in rows]),
        split=np.asarray([str(row.get("split", "")) for row in rows]),
        label=np.asarray([str(row.get("label", "")) for row in rows]),
        condition=np.asarray([str(row.get("condition", "")) for row in rows]),
        judged_harmful=np.asarray(
            [-1 if row.get("harmful") is None else int(bool(row["harmful"])) for row in rows]
        ),
        judged_refusal=np.asarray(
            [-1 if row.get("refusal") is None else int(bool(row["refusal"])) for row in rows]
        ),
        truncated=np.asarray([int(bool(row.get("truncated"))) for row in rows]),
        latency_metadata_json=np.asarray(json.dumps(metadata)),
        **scores,
    )
    lengths = scores["response_length"]
    print(
        f"[lengths] median={np.median(lengths):.0f} p95={np.quantile(lengths, 0.95):.0f} "
        f"at_k_max={(lengths >= args.k_max).mean():.1%}"
    )
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
