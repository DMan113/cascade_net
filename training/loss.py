"""
Multi-task loss for CascadeNet.

L = L_pd + cascade_weight × (L_gate + L_cascade)

  L_pd:      weighted BCE for default classification
  L_gate:    BCE for cascade occurrence (cascade > 0?)
  L_cascade: Huber loss on log1p(cascade_size) for nonzero cascades
"""

import numpy as np
import torch
import torch.nn.functional as F


def cascadenet_loss(
            pd_logit: torch.Tensor,
            cascade_gate: torch.Tensor,
            cascade_size: torch.Tensor,
            y_true: torch.Tensor,
            cascade_true: torch.Tensor,
            pos_weight: float,
            cascade_weight: float = 0.3,
    ) -> torch.Tensor:
    """
    Compute combined loss for default prediction and cascade estimation.

    Args:
        pd_logit:      (N,) default logits
        cascade_gate:  (N,) cascade occurrence logits
        cascade_size:  (N,) predicted cascade magnitude
        y_true:        (N,) binary default labels
        cascade_true:  (N,) true cascade sizes
        pos_weight:    class weight for positive (default) class
        cascade_weight: weight of cascade loss terms (staged)

    Returns:
        Scalar loss
    """
    dev = y_true.device

    # ФІКС: Обмежуємо pos_weight максимумом 3.0.
    # Великі значення штучно завищують ймовірності і руйнують F1 (поріг 0.5)
    capped_pw = min(pos_weight, 3.0)
    pw = torch.tensor([capped_pw], device=dev)

    # Head A: default prediction
    L_pd = F.binary_cross_entropy_with_logits(pd_logit, y_true, pos_weight=pw)

    # Head B: cascade estimation
    cascade_occurred = (cascade_true > 0).float()
    L_gate = F.binary_cross_entropy_with_logits(cascade_gate, cascade_occurred)

    mask = cascade_true > 0
    if mask.sum() > 0:
        L_cascade = F.huber_loss(
            cascade_size[mask],
            torch.log1p(cascade_true[mask]),
            delta=2.0,
        )
        # ФІКС: Балансуємо регресію, щоб її градієнти не домінували над BCE
        L_cascade = L_cascade * 0.5
    else:
        L_cascade = torch.tensor(0.0, device=dev)

    return L_pd + cascade_weight * (L_gate + L_cascade)


def compute_pos_weight(labels: np.ndarray, lo: float = 1.0, hi: float = 20.0) -> float:

    """
    Compute class weight for imbalanced binary classification.
    Clipped to [lo, hi] for stability.
    """
    p = labels.mean()
    if 0 < p < 1:
        return float(np.clip((1 - p) / p, lo, hi))
    return 1.0