"""
CascadeNet — Stress-Conditioned Cascade Risk Predictor.

Full model: StressEncoder → FiLM-conditioned GNN → DualHead.
"""

import torch
import torch.nn as nn
from typing import Tuple

from config import Config
from .stress_encoder import StressEncoder
from .gnn_layers import StressCondGNNLayer
from .heads import DualHead


class CascadeNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # ФІКС: Окремий енкодер для кожного шару GNN
        self.stress_encoders = nn.ModuleList([
            StressEncoder(
                stress_dim=cfg.stress_dim,
                emb_dim=cfg.stress_emb_dim,
                gamma_min=cfg.gamma_min,
                gamma_max=cfg.gamma_max,
            )
            for _ in range(cfg.num_gnn_layers)
        ])

        self.gnn_layers = nn.ModuleList([
            StressCondGNNLayer(cfg.hidden_dim, cfg.stress_emb_dim, cfg.dropout)
            for _ in range(cfg.num_gnn_layers)
        ])

        self.dual_head = DualHead(
            h_dim=cfg.hidden_dim,
            stress_emb_dim=cfg.stress_emb_dim,
            head_hidden=cfg.head_hidden,
            input_dim=cfg.input_dim,
        )

    def forward(
            self, x, edge_index, edge_weight, stress,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h = self.input_proj(x)
        last_gamma, last_beta = None, None

        for layer, enc in zip(self.gnn_layers, self.stress_encoders):
            last_gamma, last_beta = enc(stress)
            h = layer(h, edge_index, edge_weight, last_gamma, last_beta)

        # Передаємо сирий x у голову
        return self.dual_head(h, last_gamma, last_beta, x)

    def get_embeddings(self, x, edge_index, edge_weight, stress) -> torch.Tensor:
        h = self.input_proj(x)
        for layer, enc in zip(self.gnn_layers, self.stress_encoders):
            gamma, beta = enc(stress)
            h = layer(h, edge_index, edge_weight, gamma, beta)
        return h.detach()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

