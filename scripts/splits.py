"""
Subject-wise train/val/test split.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def make_splits(X, y, groups, val_size=0.2, test_size=0.2, random_state=42):
    groups = np.asarray(groups)

    # stage 1: hold out the test subjects
    test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    trainval_idx, test_idx = next(test_splitter.split(X, y, groups))

    # stage 2: split the remainder into train and val
    # val_size is expressed as a fraction of the ORIGINAL data, so it has
    # to be rescaled to a fraction of trainval_idx before the second split
    rel_val_size = val_size / (1.0 - test_size)
    val_splitter = GroupShuffleSplit(
        n_splits=1, test_size=rel_val_size, random_state=random_state
    )
    tr_rel_idx, va_rel_idx = next(
        val_splitter.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx])
    )

    train_idx = trainval_idx[tr_rel_idx]
    val_idx = trainval_idx[va_rel_idx]

    # sanity check: no subject should appear in more than one split
    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])
    test_subjects = set(groups[test_idx])

    assert not (train_subjects & val_subjects), "subject leakage between train/val"
    assert not (train_subjects & test_subjects), "subject leakage between train/test"
    assert not (val_subjects & test_subjects), "subject leakage between val/test"

    return train_idx, val_idx, test_idx