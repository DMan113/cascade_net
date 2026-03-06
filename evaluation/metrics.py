"""
Evaluation metrics.

Classification: AUROC, AUPRC, Brier score, F1, Precision@k.
Cascade:        Spearman ρ, MAE (raw + log1p).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
)
from scipy.stats import spearmanr
from typing import Dict


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: float = 0.10,
) -> Dict[str, float]:
    """
    Compute classification metrics for default prediction.

    Args:
        y_true: binary labels (N,)
        y_prob: predicted probabilities (N,)
        k: fraction for Precision@k

    Returns:
        Dict with auroc, auprc, brier, f1, precision_at_k
    """
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.0, "auprc": 0.0, "brier": 1.0,
            "f1": 0.0, "precision_at_k": 0.0,
        }

    y_pred = (y_prob > 0.5).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_k": float(_precision_at_k(y_true, y_prob, k)),
    }


def cascade_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute cascade size prediction metrics.

    Args:
        y_true: true cascade sizes (N,)
        y_pred: predicted cascade sizes (N,)

    Returns:
        Dict with spearman_rho, mae_log, mae_raw
    """
    if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return {"spearman_rho": 0.0, "mae_log": float("inf"), "mae_raw": float("inf")}

    rho, _ = spearmanr(y_true, y_pred)
    mae_log = mean_absolute_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))
    mae_raw = mean_absolute_error(y_true, y_pred)

    return {
        "spearman_rho": float(rho),
        "mae_log": float(mae_log),
        "mae_raw": float(mae_raw),
    }


def _precision_at_k(y_true, scores, k=0.10):
    """Precision among top-k% scored nodes."""
    n = max(1, int(len(y_true) * k))
    top_idx = np.argsort(scores)[-n:]
    return float(y_true[top_idx].mean())


