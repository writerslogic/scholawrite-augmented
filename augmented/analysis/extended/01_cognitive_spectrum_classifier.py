"""Classifies datasets by task type using process signals.

Tests whether {mean_iki, std_iki, entropy, lag1_autocorr, revision_density} form a
detectable cognitive spectrum (composition > transcription > password). Uses logistic
regression with 5-fold cross-validation.

Paper relevance: empirically validates cognitive spectrum claim.

Usage:
    uv run python analysis/extended/01_cognitive_spectrum_classifier.py \\
        --data-dir data/ -o results/cognitive_spectrum.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scholawrite.datasets import (
    REGISTRY,
    TaskType,
    entropy_bits,
    lag1_autocorr,
    load_all,
)


def _out(*args, **kwargs):
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _softmax(logits):
    import math
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _normalize_features(X):
    """Standardize each feature column to zero mean, unit variance."""
    n = len(X)
    if n == 0:
        return X, [], []
    n_feat = len(X[0])
    means = [_mean([row[j] for row in X]) for j in range(n_feat)]
    stds = [_std([row[j] for row in X]) for j in range(n_feat)]
    X_norm = []
    for row in X:
        normed = []
        for j in range(n_feat):
            s = stds[j] if stds[j] > 1e-12 else 1.0
            normed.append((row[j] - means[j]) / s)
        X_norm.append(normed)
    return X_norm, means, stds


def _apply_normalization(X, means, stds):
    X_norm = []
    for row in X:
        normed = []
        for j in range(len(row)):
            s = stds[j] if stds[j] > 1e-12 else 1.0
            normed.append((row[j] - means[j]) / s)
        X_norm.append(normed)
    return X_norm


def _cross_val_logistic(X, y, classes, cv=5, lr=0.1, max_iter=300, reg=0.01):
    """Logistic regression with softmax, SGD, L2 reg, k-fold cross-validation.

    Returns mean accuracy, std accuracy, and final weights trained on full data.
    """
    import math
    import random

    n = len(X)
    n_feat = len(X[0])
    n_cls = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Shuffle indices with fixed seed for reproducibility
    rng = random.Random(42)
    indices = list(range(n))
    rng.shuffle(indices)

    fold_size = n // cv
    accs = []

    for fold in range(cv):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < cv - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        # Normalize on train, apply to val
        X_tr_norm, means, stds = _normalize_features(X_train)
        X_v_norm = _apply_normalization(X_val, means, stds)

        # Weights: shape (n_cls, n_feat)
        W = [[0.0] * n_feat for _ in range(n_cls)]
        b = [0.0] * n_cls

        for epoch in range(max_iter):
            epoch_lr = lr / (1.0 + 0.01 * epoch)
            perm = list(range(len(X_tr_norm)))
            rng.shuffle(perm)
            for idx in perm:
                xrow = X_tr_norm[idx]
                true_cls = cls_to_idx[y_train[idx]]
                logits = [_dot(W[c], xrow) + b[c] for c in range(n_cls)]
                probs = _softmax(logits)
                for c in range(n_cls):
                    delta = probs[c] - (1.0 if c == true_cls else 0.0)
                    for j in range(n_feat):
                        W[c][j] -= epoch_lr * (delta * xrow[j] + reg * W[c][j])
                    b[c] -= epoch_lr * delta

        # Evaluate
        correct = 0
        for xrow, true_label in zip(X_v_norm, y_val):
            logits = [_dot(W[c], xrow) + b[c] for c in range(n_cls)]
            pred = classes[logits.index(max(logits))]
            if pred == true_label:
                correct += 1
        accs.append(correct / len(y_val) if y_val else 0.0)

    # Train final model on all data
    X_all_norm, means_all, stds_all = _normalize_features(X)
    W_final = [[0.0] * n_feat for _ in range(n_cls)]
    b_final = [0.0] * n_cls
    for epoch in range(max_iter):
        epoch_lr = lr / (1.0 + 0.01 * epoch)
        perm = list(range(len(X_all_norm)))
        rng.shuffle(perm)
        for idx in perm:
            xrow = X_all_norm[idx]
            true_cls = cls_to_idx[y[idx]]
            logits = [_dot(W_final[c], xrow) + b_final[c] for c in range(n_cls)]
            probs = _softmax(logits)
            for c in range(n_cls):
                delta = probs[c] - (1.0 if c == true_cls else 0.0)
                for j in range(n_feat):
                    W_final[c][j] -= epoch_lr * (delta * xrow[j] + reg * W_final[c][j])
                b_final[c] -= epoch_lr * delta

    return accs, W_final, means_all, stds_all


def _confusion_and_f1(X, y, classes, W, means, stds):
    """Compute confusion matrix and per-class F1 from fitted weights."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    X_norm = _apply_normalization(X, means, stds)

    cm = [[0] * n_cls for _ in range(n_cls)]
    for xrow, true_label in zip(X_norm, y):
        logits = [_dot(W[c], xrow) for c in range(n_cls)]
        pred_idx = logits.index(max(logits))
        true_idx = cls_to_idx[true_label]
        cm[true_idx][pred_idx] += 1

    per_class_f1 = {}
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n_cls) if r != i)
        fn = sum(cm[i][c] for c in range(n_cls) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[cls] = round(f1, 4)

    return cm, per_class_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("-o", "--output", default="results/cognitive_spectrum.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _out("Loading all datasets...")
    all_datasets = load_all(data_dir)
    _out(f"  Loaded {len(all_datasets)} datasets with data")

    EXCLUDE_TYPES = {TaskType.REFERENCE.value}

    features = []
    labels = []
    n_per_class: dict[str, int] = {}

    for ds_name, records in all_datasets.items():
        loader = REGISTRY.get(ds_name)
        if loader is None:
            continue
        task_type_val = loader.METADATA.task_type.value
        if task_type_val in EXCLUDE_TYPES:
            continue

        for rec in records:
            ac = rec.lag1_autocorrelation if rec.lag1_autocorrelation is not None else 0.0
            feat = [
                rec.mean_iki_ms,
                rec.std_iki_ms,
                rec.iki_entropy_bits,
                ac,
                rec.revision_density,
            ]
            features.append(feat)
            labels.append(task_type_val)
            n_per_class[task_type_val] = n_per_class.get(task_type_val, 0) + 1

    _out(f"  Total samples: {len(features)}")
    for cls, cnt in sorted(n_per_class.items()):
        _out(f"    {cls}: {cnt}")

    if len(features) < 10:
        _out("ERROR: insufficient data to run classification.")
        sys.exit(1)

    classes = sorted(set(labels))
    _out(f"\nClasses: {classes}")

    # Only run CV if we have enough samples per class
    min_per_class = min(n_per_class.values())
    cv_folds = min(5, min_per_class)
    if cv_folds < 2:
        _out("WARNING: fewer than 2 samples in smallest class; skipping CV.")
        accs = [0.0]
        W, means_all, stds_all = [[0.0] * 5 for _ in classes], [0.0] * 5, [1.0] * 5
    else:
        _out(f"\nRunning {cv_folds}-fold cross-validation...")
        accs, W, means_all, stds_all = _cross_val_logistic(
            features, labels, classes, cv=cv_folds
        )

    acc_mean = _mean(accs)
    acc_std = _std(accs)
    _out(f"  CV accuracy: {acc_mean:.3f} +/- {acc_std:.3f}")

    cm, per_class_f1 = _confusion_and_f1(features, labels, classes, W, means_all, stds_all)
    _out("\nPer-class F1:")
    for cls, f1 in per_class_f1.items():
        _out(f"  {cls}: {f1:.4f}")

    # Feature importance: mean absolute weight across classes
    feature_names = ["mean_iki_ms", "std_iki_ms", "iki_entropy_bits", "lag1_autocorr", "revision_density"]
    importance = {}
    for j, fname in enumerate(feature_names):
        mean_abs = _mean([abs(W[c][j]) for c in range(len(classes))])
        importance[fname] = round(mean_abs, 6)

    _out("\nFeature importance (mean |weight|):")
    for fname, imp in sorted(importance.items(), key=lambda x: -x[1]):
        _out(f"  {fname}: {imp:.6f}")

    result = {
        "accuracy_mean": round(acc_mean, 4),
        "accuracy_std": round(acc_std, 4),
        "cv_folds": cv_folds,
        "per_class_f1": per_class_f1,
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm,
        },
        "feature_importance": importance,
        "n_samples_per_class": n_per_class,
        "n_total": len(features),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _out(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
