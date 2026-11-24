"""Run four held-out answers through the saved TakeMeter classifier."""
from build_interface import examples
from takemeter import TakeMeter


def main():
    sample = examples()
    model = TakeMeter()
    for row, result in zip(sample, model.classify([item["text"] for item in sample])):
        print(row["source_url"])
        print(row["text"])
        print(f"Prediction: {result['label']} ({result['confidence']:.1%})")
        print(f"Reference: {row['label']}\n")


if __name__ == "__main__":
    main()
