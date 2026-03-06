"""
GNN baselines for ablation study.

B3: VanillaGNN — stress concatenated to input, no conditioning after that.
B4: StressSkipGNN — stress injected at every layer via skip connections.

Both use the same depth/width as CascadeNet for fair comparison.
"""

import torch
import torch.nn as nn

from config import Config
from .gnn_layers import VanillaGNNLayer, StressSkipGNNLayer


class VanillaGNN(nn.Module):
    """
    B3 baseline: GNN with stress concatenated to node features at input.
    No FiLM, no per-layer stress injection.

    Isolates: does the GNN need stress at all beyond the input?
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim + cfg.stress_dim, cfg.hidden_dim)

        self.gnn_layers = nn.ModuleList([
            VanillaGNNLayer(cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.num_gnn_layers)
        ])

        self.pd_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.head_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_weight, stress):
        if stress.dim() == 1:
            stress = stress.unsqueeze(0)
        s = stress.expand(x.size(0), -1)

        h = self.input_proj(torch.cat([x, s], dim=-1))
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_weight)

        return self.pd_head(h).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StressSkipGNN(nn.Module):
    """
    B4 baseline: GNN with stress concatenated at input AND every layer.
    Stress has full access throughout, but via concatenation not FiLM.

    Isolates: is FiLM (multiplicative) better than additive skip access?
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim + cfg.stress_dim, cfg.hidden_dim)

        self.gnn_layers = nn.ModuleList([
            StressSkipGNNLayer(cfg.hidden_dim, cfg.stress_dim, cfg.dropout)
            for _ in range(cfg.num_gnn_layers)
        ])

        self.pd_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.head_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_weight, stress):
        if stress.dim() == 1:
            stress = stress.unsqueeze(0)
        s = stress.expand(x.size(0), -1)

        h = self.input_proj(torch.cat([x, s], dim=-1))
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_weight, stress)

        return self.pd_head(h).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)