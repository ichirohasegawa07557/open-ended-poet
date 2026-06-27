from __future__ import annotations

import numpy as np


def graph_scores(pred: np.ndarray, true: np.ndarray):
    mask = ~np.eye(true.shape[0], dtype=bool)
    p = pred[mask].astype(int)
    t = true[mask].astype(int)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def message_entropy(symbols):
    if len(symbols) == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)
