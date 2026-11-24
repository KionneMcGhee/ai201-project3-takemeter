"""Run an optional instruction-model baseline on the frozen test IDs."""
import argparse
import csv
import json
import os
import time
from urllib import error, request

from train import labels, metrics
from takemeter import LABELS, ROOT

DEFAULT_MODEL = "openai/gpt-oss-120b"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def request_label(key, model, text, instructions):
    data = json.dumps({"model": model, "temperature": 0, "max_completion_tokens": 512,
                       "messages": [{"role": "system", "content": instructions},
                                    {"role": "user", "content": text}]}).encode()
    call = request.Request(ENDPOINT, data, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    for attempt in range(5):
        try:
            with request.urlopen(call, timeout=45) as response:
                return json.load(response)["choices"][0]["message"]["content"].strip()
        except error.HTTPError as exc:
            transient = exc.code == 429 or exc.code in {500, 502, 503, 504}
            if not transient or attempt == 4:
                raise RuntimeError(f"Groq rejected the request with HTTP {exc.code}") from None
            retry_after = exc.headers.get("Retry-After", "").strip()
            time.sleep(min(60, float(retry_after) if retry_after.isdecimal() else 2 ** (attempt + 1)))
    raise RuntimeError("Exhausted requests")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        parser.error("set GROQ_API_KEY in the environment")
    results = json.loads((ROOT / "evaluation_results.json").read_text())
    with (ROOT / "labeled_posts.csv").open(newline="", encoding="utf-8") as handle:
        dataset = {row["answer_id"]: row for row in csv.DictReader(handle)}
    rows = [dataset[answer_id] for answer_id in results["split"]["test"]["answer_ids"]]
    prompt = (ROOT / "baseline_prompt.txt").read_text()
    predictions = []
    for row in rows:
        raw = request_label(key, args.model, row["text"], prompt)
        normalized = raw.strip().lower()
        predictions.append({"answer_id": row["answer_id"], "raw_response": raw,
                            "label": normalized if normalized in LABELS else None})
    scores = metrics(labels(rows), [LABELS.index(p["label"]) if p["label"] else -1 for p in predictions])
    output = {"model": args.model, "prompt": prompt, "metrics": scores, "predictions": predictions}
    (ROOT / "groq_baseline_results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    run()
