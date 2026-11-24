"""Build a self-contained browser interface from the saved checkpoint."""
import base64
import csv
import json
import numpy as np
from takemeter import LABELS, ROOT

SAMPLE_IDS = ("123483", "124151", "124144", "124137")


def examples():
    with (ROOT / "labeled_posts.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["answer_id"]: row for row in csv.DictReader(handle)}
    return [{key: rows[answer_id][key] for key in ("answer_id", "text", "label", "author", "source_url", "license")}
            for answer_id in SAMPLE_IDS]


def payload():
    with np.load(ROOT / "model" / "takemeter.npz", allow_pickle=False) as state:
        data = {name: state[name].tolist() for name in ("words", "bias", "center", "scale", "temperature")}
        for name in ("embedding", "coefficients"):
            data[name] = base64.b64encode(np.asarray(state[name], dtype="<f4").tobytes()).decode()
    return dict(data, labels=LABELS, examples=examples())


def main():
    template = (ROOT / "interface_template.html").read_text()
    model = json.dumps(payload(), ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    inference = (ROOT / "inference.js").read_text()
    output = template.replace("__MODEL_DATA__", model).replace("/* __INFERENCE_JS__ */", inference)
    (ROOT / "TakeMeter.html").write_text(output)


if __name__ == "__main__":
    main()
