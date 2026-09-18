"""
EEGNet training for mental workload classification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EEGWindowDataset, load_processed
from model import EEGNet
from splits import make_splits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"


def set_seed(seed):
    """Pin every RNG we care about so runs are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, device):
    """
    One full pass over the loader. Returns:
      y_true : ground-truth labels
      y_pred : hard predictions (argmax)
      y_prob : probability of class 1 (needed for threshold tuning)
    """
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        all_y.append(yb.numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob1.cpu().numpy())

    return (
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def best_threshold_from_pr_curve(y_true, y_prob):
    """
    Find the threshold that maximizes F1 on the given data.

    Only ever call this with validation data. Calling it with test data
    means you're fitting the threshold on test and the numbers become
    meaningless.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision and recall have one extra element vs thresholds; drop the last
    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / (denom + 1e-12), 0.0)

    best_i = int(np.argmax(f1))
    return float(thresholds[best_i]), float(f1[best_i])


def train_one_run(args, X, y, groups, train_idx, val_idx, device, seed):
    """One full training run with a given seed. Returns run metadata."""
    set_seed(seed)

    train_ds = EEGWindowDataset(X[train_idx], y[train_idx])
    val_ds = EEGWindowDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    # shuffle=False for val so run_eval returns labels in a stable order
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_channels=n_ch, n_samples=n_t).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Inverse-frequency class weights so the minority class isn't ignored
    counts = np.bincount(y[train_idx], minlength=2)
    class_weights = counts.sum() / (2.0 * counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_path = OUT / f"best_model_seed{seed}.pt"

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[seed {seed}] ep {epoch}/{args.epochs}",
            leave=False,
        )
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / len(train_ds)

        y_true, y_pred, _ = run_eval(model, val_loader, device)
        val_acc = (y_true == y_pred).mean()
        val_f1_macro = f1_score(y_true, y_pred, average="macro")
        val_f1_high = f1_score(y_true, y_pred, pos_label=1, average="binary")

        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"val_acc {val_acc:.4f} | val_f1_macro {val_f1_macro:.4f} | "
            f"val_f1_high {val_f1_high:.4f}"
        )

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(
                    f"[seed {seed}] early stopping at epoch {epoch} "
                    f"(no val F1 improvement for {args.patience} epochs)"
                )
                break

    # Roll back to the best epoch before the final eval and threshold tuning
    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)

    y_true, y_pred, y_prob = run_eval(model, val_loader, device)
    threshold, thr_f1 = best_threshold_from_pr_curve(y_true, y_prob)

    meta = {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_f1_macro_at_0.5": best_f1,
        "val_f1_at_tuned_threshold": thr_f1,
        "tuned_threshold": threshold,
        "checkpoint": ckpt_path.name,
    }

    with open(OUT / f"meta_seed{seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[seed {seed}] best val F1(macro)={best_f1:.4f} @ epoch {best_epoch} | "
        f"tuned threshold={threshold:.3f} (val F1 there: {thr_f1:.4f})"
    )

    return meta


def pick_representative_run(runs):
    """
    From several runs, pick the one whose F1 is closest to the mean.

    Why not just pick the best? Picking the best is cherry-picking and
    makes the final evaluation look better than it really is.
    """
    if len(runs) == 1:
        return runs[0]

    f1s = np.array([r["val_f1_macro_at_0.5"] for r in runs])
    closest = int(np.argmin(np.abs(f1s - f1s.mean())))
    return runs[closest]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    X, y, groups = load_processed(DATA)

    # test_idx is not used past this point; only evaluate.py touches it,
    # once, at the very end
    train_idx, val_idx, test_idx = make_splits(X, y, groups, random_state=42)
    print(
        f"train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)} "
        f"(test held out, not used here)"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds else [42]
    runs = [
        train_one_run(args, X, y, groups, train_idx, val_idx, device, s)
        for s in seeds
    ]

    if len(runs) > 1:
        f1s = np.array([r["val_f1_macro_at_0.5"] for r in runs])
        print(
            f"\nacross {len(runs)} seeds: "
            f"val F1(macro) = {f1s.mean():.4f} +/- {f1s.std():.4f}"
        )

    chosen = pick_representative_run(runs)

    with open(OUT / "chosen_run.json", "w") as f:
        json.dump(chosen, f, indent=2)

    print(
        f"\nchosen run for final evaluation: seed {chosen['seed']} "
        f"-> {chosen['checkpoint']}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--patience",
        type=int,
        default=12,
        help="early stopping patience, in epochs without val F1 improvement",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="e.g. --seeds 1 2 3 4 5   (default: single run with seed 42)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())