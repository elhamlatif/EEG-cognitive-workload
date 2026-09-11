# evaluate.py
# Loads the trained EEGNet and checks how it does on the validation subjects
# Written by Elham Latif

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from dataset import EEGWindowDataset, load_processed
from model import EEGNet


# paths - same layout as the other scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, groups = load_processed(PROCESSED_DIR)

    # use the exact same split as train.py
    # so we evaluate on the same validation subjects
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    _, val_idx = next(splitter.split(X, y, groups))

    val_ds = EEGWindowDataset(X[val_idx], y[val_idx])
    val_loader = DataLoader(val_ds, batch_size=64)

    n_channels = X.shape[1]
    n_samples = X.shape[2]

    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
    ).to(device)

    # load the weights we saved during training
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()  # turn off dropout / batchnorm updates

    preds = []
    probs = []
    labels = []

    # no grads needed here, we're just doing inference
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)

            out = model(X_batch)

            # probability of class 1 (high workload)
            p = torch.softmax(out, dim=1)[:, 1]

            pred = out.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(y_batch.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    print("Accuracy: {:.4f}".format(acc))
    print("F1-score: {:.4f}".format(f1))
    print("ROC-AUC:  {:.4f}".format(auc))

    save_confusion_matrix(labels, preds)


def save_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(cm)

    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Low workload", "High workload"],
        yticklabels=["Low workload", "High workload"],
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )

    # write the counts on top of each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.tight_layout()

    out_path = RESULTS_DIR / "confusion_matrix.png"

    fig.savefig(out_path, dpi=150)

    plt.close(fig)

    print("Saved confusion matrix to:", out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EEGNet model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(RESULTS_DIR / "best_model.pt"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model)