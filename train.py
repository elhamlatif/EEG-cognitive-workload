"""
Train EEGNet on the preprocessed EEG windows.

Uses a subject-independent split (GroupShuffleSplit) so that windows
from the same person never appear in both train and validation sets.
"""

from pathlib import Path
import argparse

import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EEGWindowDataset, load_processed
from model import EEGNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, groups = load_processed(PROCESSED_DIR)

    # Subject-wise split. random_state is fixed so evaluate.py can reproduce the same split.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    train_ds = EEGWindowDataset(X[train_idx], y[train_idx])
    val_ds = EEGWindowDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    n_channels, n_samples = X.shape[1], X.shape[2]
    model = EEGNet(n_channels=n_channels, n_samples=n_samples).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- training ---
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for X_batch, y_batch in progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_ds)

        # --- validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet for mental workload classification")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())