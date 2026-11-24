"""Build a locked split, run the zero-shot baseline, and train/evaluate TakeMeter."""
import csv
import json
import platform
import subprocess
from tempfile import TemporaryDirectory
from collections import Counter

import numpy as np

from takemeter import FLOOR, LABELS, ROOT, SEMANTIC_DIMS, features, prepare, softmax, tokens

SEED = 201
BATCH = 16
EPOCHS = 36
RATES = (0.005, 0.015)
PROTOTYPES = {
    "asserted": ("I think the answer is a title or opinion without support",
                 "a short personal guess with no evidence", "this is the movie with that actor"),
    "reasoned": ("the conclusion follows because of the character's actions and plot details",
                 "explains by matching a scene and making a causal inference",
                 "uses observable details to explain why this happened"),
    "sourced": ("according to an interview, transcript or official screenplay",
                "a named source provides evidence with a direct quotation",
                "traceable evidence cited from a book, report or clip"),
}


def split_data(rows):
    rng = np.random.default_rng(SEED)
    splits = {name: [] for name in ("train", "validation", "test")}
    for label in LABELS:
        group = sorted((row for row in rows if row["label"] == label), key=lambda row: int(row["answer_id"]))
        shuffled = [group[i] for i in rng.permutation(len(group))]
        test_size = round(len(group) * 0.15)
        validation_size = round(len(group) * 0.15)
        cut = test_size + validation_size
        splits["test"].extend(shuffled[:test_size])
        splits["validation"].extend(shuffled[test_size:cut])
        splits["train"].extend(shuffled[cut:])
    for name in splits:
        splits[name].sort(key=lambda row: int(row["answer_id"]))
    return splits


def metrics(gold, predicted):
    gold = np.asarray(gold)
    predicted = np.asarray(predicted)
    matrix = np.array([[(np.logical_and(gold == i, predicted == j)).sum()
                        for j in range(len(LABELS))] for i in range(len(LABELS))])
    result = {}
    for i, label in enumerate(LABELS):
        correct = int(matrix[i, i])
        precision = correct / max(int(matrix[:, i].sum()), 1)
        recall = correct / max(int((gold == i).sum()), 1)
        result[label] = {"precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / max(precision + recall, FLOOR),
                         "support": int((gold == i).sum())}
    return {"accuracy": float(np.mean(gold == predicted)),
            "macro_f1": float(np.mean([record["f1"] for record in result.values()])),
            "per_class": result, "confusion_matrix": matrix.tolist(),
            "invalid_responses": int((predicted < 0).sum())}


def labels(rows):
    return np.array([LABELS.index(row["label"]) for row in rows], dtype=np.int64)


def export_vectors(words):
    with TemporaryDirectory(prefix="takemeter-swift-") as cache:
        completed = subprocess.run(
            ["swift", "-module-cache-path", cache, str(ROOT / "ExportEmbeddings.swift")],
            input=json.dumps(sorted(words)), text=True, capture_output=True, check=True,
        )
    return json.loads(completed.stdout)


def baseline(rows, vectors):
    """Zero-shot retrieval using the class prompts as fixed semantic prototypes."""
    def unit_mean(text):
        known = [vectors[word] for word in tokens(text) if word in vectors]
        mean = np.mean(known, axis=0) if known else np.zeros(SEMANTIC_DIMS)
        return mean / max(np.linalg.norm(mean), FLOOR)

    prototypes = np.stack([unit_mean(prompt) for label in LABELS for prompt in PROTOTYPES[label]])
    output = []
    for row in rows:
        cosine = (prototypes @ unit_mean(row["text"])).reshape(len(LABELS), -1)
        scores = cosine.mean(axis=1)
        output.append({"answer_id": row["answer_id"], "label": LABELS[int(scores.argmax())],
                       "cosine_scores": dict(zip(LABELS, map(float, scores)))})
    return output


class Adam:
    def __init__(self, initial):
        self.mean = [np.zeros_like(parameter) for parameter in initial]
        self.variance = [np.zeros_like(parameter) for parameter in initial]
        self.step = 0

    def update(self, parameters, gradients, rates):
        self.step += 1
        for parameter, gradient, mean, variance, rate in zip(
            parameters, gradients, self.mean, self.variance, rates
        ):
            mean *= 0.9
            mean += 0.1 * gradient
            variance *= 0.999
            variance += 0.001 * (gradient * gradient)
            correction = (mean / (1 - 0.9 ** self.step)) / (
                np.sqrt(variance / (1 - 0.999 ** self.step)) + 1e-7
            )
            parameter -= rate * correction


def train_split(train, valid, gold_train, gold_valid, initial, center, scale, rate):
    rng = np.random.default_rng(SEED)
    embedding = initial.copy()
    feature_count = features(train[:1], embedding, center, scale).shape[1]
    weights = rng.normal(0.0, 0.01, size=(feature_count, len(LABELS))).astype(np.float32)
    bias = np.zeros(len(LABELS), dtype=np.float32)
    optimizer = Adam([embedding, weights, bias])
    class_weights = (len(train) / (len(LABELS) * np.bincount(gold_train))) ** 0.5
    best = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        permutation = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            indices = permutation[start:start + BATCH]
            batch = [train[i] for i in indices]
            expected = gold_train[indices]
            x = features(batch, embedding, center, scale)
            error = softmax(x @ weights + bias)
            error[np.arange(len(batch)), expected] -= 1
            error *= class_weights[expected, None] / len(batch)
            embedding_gradient = np.zeros_like(embedding)
            semantic_gradient = (error @ weights[:SEMANTIC_DIMS].T) * 4
            for (ids, _, _), gradient in zip(batch, semantic_gradient):
                if len(ids):
                    np.add.at(embedding_gradient, ids, gradient / len(ids))
            embedding_gradient += 0.004 * (embedding - initial)
            optimizer.update([embedding, weights, bias],
                             [embedding_gradient, x.T @ error + 0.01 * weights, error.sum(axis=0)],
                             [rate * 0.15, rate, rate])
        validation_logits = features(valid, embedding, center, scale) @ weights + bias
        assessment = metrics(gold_valid, validation_logits.argmax(axis=1))
        validation_nll = nll(gold_valid, softmax(validation_logits))
        score = assessment["macro_f1"], -validation_nll
        history.append({"epoch": epoch, "macro_f1": score[0], "nll": validation_nll})
        if best is None or score > best["score"]:
            best = dict(score=score, epoch=epoch, rate=rate, embedding=embedding.copy(),
                        weights=weights.copy(), bias=bias.copy(), validation_logits=validation_logits.copy())
    return best, history


def nll(gold, probabilities):
    return float(-np.log(np.clip(probabilities[np.arange(len(gold)), gold], FLOOR, 1)).mean())


def calibration(gold, probabilities):
    confidence = probabilities.max(axis=1)
    correct = gold == probabilities.argmax(axis=1)
    bins = []
    for low, high in [(0, .5), (.5, .65), (.65, .8), (.8, .9), (.9, 1.01)]:
        group = (low <= confidence) & (confidence < high)
        count = int(group.sum())
        bins.append({"min": low, "max": min(high, 1), "count": count,
                     "mean_confidence": float(confidence[group].mean()) if count else None,
                     "accuracy": float(correct[group].mean()) if count else None})
    ece = sum(row["count"] * abs(row["mean_confidence"] - row["accuracy"])
              for row in bins if row["count"]) / len(gold)
    brier = np.square(probabilities - np.eye(len(LABELS))[gold]).sum(axis=1).mean()
    return {"ece": ece, "brier": float(brier), "nll": nll(gold, probabilities), "bins": bins}


def word_set(rows):
    return {word for row in rows for word in tokens(row["text"])}


def main():
    with (ROOT / "labeled_posts.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len({row["question_id"] for row in rows}) == len(rows)
    splits = split_data(rows)
    train, valid, test = (splits[name] for name in ("train", "validation", "test"))
    prompts = {word for values in PROTOTYPES.values() for phrase in values for word in tokens(phrase)}
    vectors = export_vectors(word_set(train) | word_set(test) | prompts)

    # This baseline consumes test text and fixed class descriptions. It never sees labels.
    baseline_predictions = baseline(test, vectors)
    words = sorted(word_set(train) & vectors.keys())
    vocabulary = {word: index for index, word in enumerate(words)}
    initial = np.asarray([vectors[word] for word in words], dtype=np.float32)
    train_input, valid_input, test_input = ([prepare(row["text"], vocabulary) for row in partition]
                                          for partition in (train, valid, test))
    surfaces = np.stack([row[2] for row in train_input])
    center, scale = surfaces.mean(axis=0), np.maximum(surfaces.std(axis=0), 0.15)
    gold_train, gold_valid, gold_test = map(labels, (train, valid, test))
    trials = [train_split(train_input, valid_input, gold_train, gold_valid, initial,
                          center, scale, rate) for rate in RATES]
    best = max((trial[0] for trial in trials), key=lambda trial: trial["score"])
    temperatures = np.linspace(0.5, 3.0, 51)
    temperature = min(temperatures, key=lambda T: nll(gold_valid, softmax(best["validation_logits"] / T)))

    logits = features(test_input, best["embedding"], center, scale) @ best["weights"] + best["bias"]
    raw = softmax(logits)
    probabilities = softmax(logits / temperature)
    predicted = probabilities.argmax(axis=1)
    result = {
        "labels": LABELS, "seed": SEED, "platform": platform.platform(),
        "base_model": "Apple NaturalLanguage English 300-dimensional pretrained word embedding",
        "split": {name: {"count": len(part), "labels": dict(Counter(row["label"] for row in part)),
                         "answer_ids": [row["answer_id"] for row in part]}
                  for name, part in splits.items()},
        "training": {"batch_size": BATCH, "epoch_limit": EPOCHS, "learning_rate": best["rate"],
                     "embedding_learning_rate": best["rate"] * 0.15, "selected_epoch": best["epoch"],
                     "validation_macro_f1": best["score"][0], "vocabulary": len(words),
                     "updated_embedding_rows": int(np.count_nonzero(np.any(best["embedding"] != initial, axis=1))),
                     "trials": [{"learning_rate": trial["rate"], "best_epoch": trial["epoch"],
                                 "validation_macro_f1": trial["score"][0], "history": history}
                                for trial, history in trials],
                     "validation_temperature": float(temperature)},
        "fine_tuned": metrics(gold_test, predicted),
        "zero_shot": {"method": "Pretrained Apple word-vector cosine similarity to fixed verbal class prompts",
                      "prototypes": PROTOTYPES,
                      **metrics(gold_test, [LABELS.index(row["label"]) for row in baseline_predictions])},
        "calibration": {"raw": calibration(gold_test, raw), "temperature_scaled": calibration(gold_test, probabilities)},
        "predictions": [{"answer_id": row["answer_id"], "gold": row["label"],
                         "prediction": LABELS[int(predicted[i])], "confidence": float(probabilities[i].max()),
                         "probabilities": dict(zip(LABELS, map(float, probabilities[i]))),
                         "word_count": len(tokens(row["text"])), "baseline": baseline_predictions[i]}
                        for i, row in enumerate(test)],
    }
    (ROOT / "model").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "model" / "takemeter.npz", words=np.asarray(words),
                        embedding=best["embedding"], coefficients=best["weights"], bias=best["bias"],
                        center=center, scale=scale, temperature=temperature)
    (ROOT / "evaluation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("fine_tuned", "zero_shot", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
