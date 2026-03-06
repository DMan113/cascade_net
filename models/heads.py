"""
Prediction heads for CascadeNet.

DualHead:
  - Head A: default probability (binary classification logit)
  - Head B: cascade size = gate × magnitude
    - gate:      P(cascade > 0)
    - magnitude: expected cascade size (Softplus)

Fix: stress input is [γ, β] concatenated (not averaged), preserving
both scale and shift information with their distinct semantics.
"""

import torch
import torch.nn as nn


class DualHead(nn.Module):
    """
    Two-headed prediction from node embeddings + stress parameters.

    Inputs:
        h:     (N, h_dim)          — node embeddings from GNN
        gamma: (1, stress_emb_dim) — FiLM scale parameter
        beta:  (1, stress_emb_dim) — FiLM shift parameter

    Outputs:
        pd_logit:      (N,) — default probability logit
        cascade_gate:  (N,) — cascade occurrence logit
        cascade_size:  (N,) — expected cascade magnitude (>0)
    """

    def __init__(
            self,
            h_dim: int = 64,
            stress_emb_dim: int = 32,
            head_hidden: int = 128,
            input_dim: int = 70,
    ):
        super().__init__()
        stress_input = stress_emb_dim * 2

        self.stress_compress = nn.Sequential(
            nn.Linear(stress_input, stress_emb_dim),
            nn.GELU(),
        )

        inp_compressed = h_dim + stress_emb_dim + input_dim

        # ФІКС: Окремі проекції для ізоляції градієнтів
        self.pd_proj = nn.Sequential(
            nn.Linear(inp_compressed, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
        )

        self.cas_proj = nn.Sequential(
            nn.Linear(inp_compressed, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
        )

        self.pd_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.cascade_gate = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.cascade_magnitude = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, h, gamma, beta, x):
        stress_cat = torch.cat([gamma, beta], dim=-1)
        stress_repr = self.stress_compress(stress_cat)
        stress_repr = stress_repr.expand(h.size(0), -1)

        inp = torch.cat([h, stress_repr, x], dim=-1)

        # ФІКС: Розділяємо ознаки перед тим, як віддавати їх у голови
        pd_feat = self.pd_proj(inp)
        cas_feat = self.cas_proj(inp)

        pd = self.pd_head(pd_feat).squeeze(-1)
        gate = self.cascade_gate(cas_feat).squeeze(-1)
        mag = self.cascade_magnitude(cas_feat).squeeze(-1) + 1e-6

        return pd, gate, mag
