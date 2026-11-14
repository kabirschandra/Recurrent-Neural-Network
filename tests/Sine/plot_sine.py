#!/usr/bin/env python3
"""Plot the sine RNN training curve and target-versus-prediction results."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Open plots interactively")
    args = parser.parse_args()
    loss = read_csv(HERE / "loss.csv")
    predictions = read_csv(HERE / "predictions.csv")
    if not loss or not predictions:
        raise SystemExit("CSV files are empty. Run the C++ sine test first.")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([int(r["epoch"]) for r in loss], [float(r["mse"]) for r in loss])
    ax.set(xlabel="Epoch", ylabel="Mean squared error", title="RNN sine training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "loss.png", dpi=160)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    times = [float(r["time"]) for r in predictions]
    ax2.plot(times, [float(r["target"]) for r in predictions], "o-", label="Target")
    ax2.plot(times, [float(r["prediction"]) for r in predictions], "o-", label="Prediction")
    ax2.set(xlabel="Next-sample time", ylabel="Amplitude", title="Sine next-step prediction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(HERE / "predictions.png", dpi=160)
    print("Saved tests/Sine/loss.png and tests/Sine/predictions.png")
    if args.show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
