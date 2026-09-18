# dataset.py
# For EEG classification project
# Written by Elham Latif

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGWindowDataset(Dataset):
    # Dataset for EEG windows.
    # EEGNet expects each sample as (1, channels, time).

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index].unsqueeze(0)
        label = self.y[index]
        return x, label


def load_processed(processed_dir):
    # Load preprocessed EEG data and labels.
    processed_dir = Path(processed_dir)

    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")

    # Subject IDs are needed for subject-wise train/validation splits.
    try:
        groups = np.load(processed_dir / "groups.npy")
    except FileNotFoundError:
        groups = None

    return X, y, groups


if __name__ == "__main__":
    X, y, groups = load_processed("data/processed")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    dataset = EEGWindowDataset(X, y)
    x, label = dataset[0]

    print("Sample shape:", x.shape)
    print("Label:", label.item())
