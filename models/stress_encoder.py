"""
FiLM Stress Encoder — generates γ (scale) and β (shift) vectors
from a macro-stress input for conditioning GNN message passing.

Fix: γ range is (gamma_min, gamma_max) via scaled sigmoid, allowing
both attenuation AND amplification of signals under stress.
"""

import torch
import torch.nn as nn


class StressEncoder(nn.Module):
    """
    Maps a stress vector s ∈ R^stress_dim to FiLM parameters (γ, β).

    γ ∈ (gamma_min, gamma_max) — multiplicative modulation
    β ∈ (-1, 1)                — additive shift
    """

    def __init__(
        self,
        stress_dim: int = 8,
        emb_dim: int = 32,
        gamma_min: float = 0.5,
        gamma_max: float = 2.0,
    ):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_range = gamma_max - gamma_min

        self.shared = nn.Sequential(
            nn.Linear(stress_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.gamma_head = nn.Sequential(
            nn.Linear(64, emb_dim),
            nn.Sigmoid(),  # output ∈ (0, 1), rescaled in forward
        )
        self.beta_head = nn.Sequential(
            nn.Linear(64, emb_dim),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor):
        """
        Args:
            s: stress vector (1, stress_dim) or (stress_dim,)

        Returns:
            gamma: (1, emb_dim) — scale factors in (gamma_min, gamma_max)
            beta:  (1, emb_dim) — shift in (-1, 1)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)

        h = self.shared(s)
        gamma = self.gamma_min + self.gamma_range * self.gamma_head(h)
        beta = self.beta_head(h)
        return gamma, beta
