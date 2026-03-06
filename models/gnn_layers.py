"""
GNN message-passing layers.

StressCondGNNLayer — FiLM-conditioned (for CascadeNet)
VanillaGNNLayer   — no stress conditioning (for B3 baseline)
StressSkipGNNLayer — stress concatenated at every layer (for B4 baseline)

Fix: StressCondGNNLayer no longer passes both raw `agg` and `sa` to the
update MLP — only [h, sa] — removing the "escape hatch" from FiLM.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class StressCondGNNLayer(MessagePassing):
    """
    GNN layer with FiLM stress conditioning on aggregated messages.

    Update: h' = MLP([h, γ·agg + β]) + h
    """

    def __init__(self, h_dim: int, stress_emb_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
        )
        self.stress_proj = (
            nn.Linear(stress_emb_dim, h_dim)
            if stress_emb_dim != h_dim
            else nn.Identity()
        )
        # Only [h, stress_modulated_agg] — no raw agg
        self.upd_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h, edge_index, edge_weight, gamma, beta):
        agg = self.propagate(edge_index, x=h, edge_weight=edge_weight)

        g = self.stress_proj(gamma)
        b = self.stress_proj(beta)
        if g.dim() == 2 and g.size(0) == 1:
            g = g.expand(h.size(0), -1)
            b = b.expand(h.size(0), -1)

        # FiLM modulation
        sa = g * agg + b

        return self.upd_mlp(torch.cat([h, sa], dim=-1)) + h

    def message(self, x_j, edge_weight):
        m = self.msg_mlp(x_j)
        if edge_weight is not None:
            return m * edge_weight.unsqueeze(-1)
        return m


class VanillaGNNLayer(MessagePassing):
    """Standard GNN layer — no stress conditioning at all."""

    def __init__(self, h_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        return self.upd_mlp(torch.cat([h, agg], dim=-1)) + h

    def message(self, x_j, edge_weight):
        m = self.msg_mlp(x_j)
        if edge_weight is not None:
            return m * edge_weight.unsqueeze(-1)
        return m


class StressSkipGNNLayer(MessagePassing):
    """
    GNN layer with stress vector injected via skip connection.

    B4 baseline: stress is concatenated (not FiLM-multiplied) at
    every layer, giving it repeated access without multiplicative
    modulation. This isolates FiLM's contribution vs simple availability.
    """

    def __init__(self, h_dim: int, stress_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
        )
        # h + agg + stress → h
        self.upd_mlp = nn.Sequential(
            nn.Linear(h_dim * 2 + stress_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h, edge_index, edge_weight, stress):
        agg = self.propagate(edge_index, x=h, edge_weight=edge_weight)

        # Expand stress to all nodes
        s = stress.expand(h.size(0), -1) if stress.dim() == 2 else stress.unsqueeze(0).expand(h.size(0), -1)

        return self.upd_mlp(torch.cat([h, agg, s], dim=-1)) + h

    def message(self, x_j, edge_weight):
        m = self.msg_mlp(x_j)
        if edge_weight is not None:
            return m * edge_weight.unsqueeze(-1)
        return m