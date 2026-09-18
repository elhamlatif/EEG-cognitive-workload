"""
Final, one-shot evaluation on the held-out test set
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import EEGWindowDataset, load_processed
from model import EEGNet
from splits import make_splits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"


def load_frozen_choice():
    """
    Read the selected run's metadata: checkpoint, seed, and threshold.

    All of this was decided earlier by train.py on validation data and is
    only being used here to produce the final report.
    """
    with open(OUT / "chosen_run.json") as f:
        chosen = json.load(f)

    with open(OUT / f"meta_seed{chosen['seed']}.json") as f:
        meta = json.load(f)

    return chosen, meta


def run_inference(model, loader, device):
    """One full pass over the loader; returns labels and class-1 probabilities."""
    model.eval()
    all_y, all_prob = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            prob1 = torch.softmax(model(xb), dim=1)[:, 1]
            all_y.append(yb.numpy())
            all_prob.append(prob1.cpu().numpy())

    return np.concatenate(all_y), np.concatenate(all_prob)


def compute_metrics(y_true, y_pred, y_prob, threshold):
    """Bundle every metric we want to report into a single dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_high": precision_score(y_true, y_pred, pos_label=1),
        "recall_high": recall_score(y_true, y_pred, pos_label=1),
        "f1_high": f1_score(y_true, y_pred, pos_label=1),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "threshold_used": threshold,
        "n_test": int(len(y_true)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chosen, meta = load_frozen_choice()
    threshold = meta["tuned_threshold"]
    ckpt_path = OUT / chosen["checkpoint"]

    print(
        f"loading {ckpt_path} (seed {chosen['seed']}), "
        f"using threshold={threshold:.3f} (tuned on validation)"
    )

    X, y, groups = load_processed(DATA)
    _, _, test_idx = make_splits(X, y, groups, random_state=42)
    print(f"test set: {len(test_idx)} windows")

    test_ds = EEGWindowDataset(X[test_idx], y[test_idx])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_channels=n_ch, n_samples=n_t).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_prob = run_inference(model, test_loader, device)
    y_pred = (y_prob >= threshold).astype(int)

    results = compute_metrics(y_true, y_pred, y_prob, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n=== FINAL TEST RESULTS (single run, not to be re-tuned) ===")
    for k, v in results.items():
        print(f"{k:16s}: {v}")

    print("\nconfusion matrix [rows=actual, cols=predicted], order [Low, High]:")
    print(cm)

    out_path = OUT / "final_test_metrics.json"
    with open(out_path, "w") as f:
        json.dump({**results, "confusion_matrix": cm.tolist()}, f, indent=2)

    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
