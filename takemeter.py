"""Portable TakeMeter model and text preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ("asserted", "reasoned", "sourced")
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
HASH_SIZE = 768
SEMANTIC_DIMS = 300
MAX_WORDS = 220
FLOOR = 1e-9


def tokens(text):
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [word.replace("’", "'") for word in WORD.findall(normalized)[:MAX_WORDS]]


def ngram_vector(words):
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    vector = np.zeros(HASH_SIZE, dtype=np.float32)
    for gram in set(grams):
        value = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "little")
        vector[value % HASH_SIZE] += 1 if value & (1 << 31) else -1
    return vector / max(np.linalg.norm(vector), 1.0)


def surface_features(text, words):
    counts = (
        len(words), len(re.findall(r"[.!?](?:\s|$)", text)), len(re.findall(r"\d+", text)),
        text.count('"') + text.count("“") + text.count("”"), text.count(":"),
        text.count("("), len(re.findall(r"\b[A-Z]{2,}\b", text)),
        len(re.findall(r"\b\d{1,2}:\d\d(?::\d\d)?\b", text)),
        len(re.findall(r"\b(?:wikipedia|imdb|interview|script|screenplay|quoted|transcript|noveli[sz]ation|commentary|credits|according to)\b", text, re.I)),
        len(re.findall(r"\b(?:because|therefore|thus|which means|which is why|so that|as a result)\b", text, re.I)),
    )
    return np.log1p(np.asarray(counts, dtype=np.float32))


def prepare(text, vocabulary):
    words = tokens(text)
    identifiers = np.asarray([vocabulary[word] for word in words if word in vocabulary], dtype=np.int64)
    return identifiers, ngram_vector(words), surface_features(text, words)


def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def features(batch, embedding, center, scale):
    semantic = np.stack([embedding[ids].mean(axis=0) if len(ids) else
                         np.zeros(SEMANTIC_DIMS, dtype=np.float32) for ids, _, _ in batch])
    lexical = np.stack([row[1] for row in batch])
    surface = (np.stack([row[2] for row in batch]) - center) / scale
    return np.concatenate((semantic * 4, lexical * 1.7, np.clip(surface, -5, 5)), axis=1)


class TakeMeter:
    def __init__(self, checkpoint=ROOT / "model" / "takemeter.npz"):
        with np.load(checkpoint, allow_pickle=False) as state:
            self.vocabulary = {word: index for index, word in enumerate(state["words"].tolist())}
            self.embedding = state["embedding"]
            self.coefficients = state["coefficients"]
            self.bias = state["bias"]
            self.center = state["center"]
            self.scale = state["scale"]
            self.temperature = float(state["temperature"])

    def classify(self, texts):
        batch = [prepare(text, self.vocabulary) for text in texts]
        values = features(batch, self.embedding, self.center, self.scale)
        scores = softmax((values @ self.coefficients + self.bias) / self.temperature)
        return [
            {"label": LABELS[int(row.argmax())], "confidence": float(row.max()),
             "probabilities": dict(zip(LABELS, map(float, row)))}
            for row in scores
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a Movies & TV answer")
    parser.add_argument("text", help="Paste the complete text of an answer")
    args = parser.parse_args()
    if not tokens(args.text):
        parser.error("provide at least one word")
    print(json.dumps(TakeMeter().classify([args.text])[0], indent=2))
