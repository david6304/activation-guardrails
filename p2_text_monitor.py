"""P2 text comparator: response TF-IDF over the reader's prefixes (CPU).

The prefixes come from `response_token_ids` in the latency artefact, decoded with
the protected reader's tokeniser, so every horizon k means the same thing for the
text monitor as it does for the probe. Tokenising the response text here instead
would move the prefix boundaries and quietly compare monitors on different
streams.

The classifier is the frozen response baseline from `text_baseline.py` -- word
1-2-grams fitted on the judged train-split responses, a different generation run
from anything P2 scores. At k=0 the prefix is empty, so the text monitor is
undefined there; the probe's k=0 prompt-only score is reported separately as the
baseline the latency claim has to beat.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from probe_prompt import file_sha256


# Includes 0 so the curve starts where the text monitor has nothing, and 512 so it
# ends at the generation budget. Dense early, where a latency difference lives.
K_GRID = (0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
READER_MODEL = "google/gemma-3-27b-it"


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def fit_response_tfidf(train_path, seed):
    """The frozen text_baseline.py response classifier, refitted deterministically."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rows = [
        row
        for row in (json.loads(line) for line in Path(train_path).open() if line.strip())
        if row.get("harmful") in (True, False)
    ]
    labels = np.asarray([bool(row["harmful"]) for row in rows])
    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )
    classifier.fit(vectoriser.fit_transform([row["response"] for row in rows]), labels)
    print(
        f"[train] response TF-IDF n={len(rows)} positives={int(labels.sum())} "
        f"vocabulary={len(vectoriser.vocabulary_)}",
        flush=True,
    )
    return vectoriser, classifier, len(rows)


def prefix_texts(token_ids, lengths, k, tokenizer):
    texts = []
    for row, length in enumerate(lengths):
        take = min(k, int(length))
        ids = token_ids[row, :take]
        texts.append(tokenizer.decode(ids.tolist()) if take else "")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/p2_latency_scores.npz")
    parser.add_argument("--train", default="data/judged_train.jsonl")
    parser.add_argument("--tokenizer", default=READER_MODEL)
    parser.add_argument("--out", default="data/p2_text_monitor.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    scores = load_npz(Path(args.scores))
    token_ids = scores["response_token_ids"]
    lengths = scores["response_length"]
    grid = [k for k in K_GRID if k <= token_ids.shape[1]]

    vectoriser, classifier, train_size = fit_response_tfidf(args.train, args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tfidf = np.empty((len(lengths), len(grid)), dtype=np.float32)
    for column, k in enumerate(grid):
        texts = prefix_texts(token_ids, lengths, k, tokenizer)
        tfidf[:, column] = classifier.decision_function(
            vectoriser.transform(texts)
        ).astype(np.float32)
        print(f"  scored k={k}", flush=True)

    metadata = {
        "monitor": "response word 1-2-gram TF-IDF, frozen text_baseline.py settings",
        "train": str(Path(args.train)),
        "train_rows": train_size,
        "prefixes": "decoded from the protected reader's response token ids",
        "tokenizer": args.tokenizer,
        "k_grid": grid,
        "k_zero": "empty prefix; the text monitor is undefined before any token",
        "scores_sha256": file_sha256(Path(args.scores)),
        "seed": args.seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=scores["ids"],
        k_grid=np.asarray(grid),
        tfidf=tfidf,
        text_monitor_metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
