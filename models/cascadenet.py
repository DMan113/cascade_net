
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

        self.input_proj = nn.Linear(cfg.input_dim + cfg.stress_dim, cfg.hidden_dim)

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

        # ── FIX: stop-gradient прапорець ─────────────────────────
        self.cascade_stop_grad = cfg.cascade_stop_grad

    def forward(
            self, x, edge_index, edge_weight, stress,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Розширюємо стрес-вектор на всі вузли графа
        if stress.dim() == 1:
            stress_expanded = stress.unsqueeze(0).expand(x.size(0), -1)
        else:
            stress_expanded = stress.expand(x.size(0), -1)

        # КОНКАТЕНУЄМО стрес із сирими фічами перед проекцією
        h = self.input_proj(torch.cat([x, stress_expanded], dim=-1))

        last_gamma, last_beta = None, None
        for layer, enc in zip(self.gnn_layers, self.stress_encoders):
            last_gamma, last_beta = enc(stress)
            h = layer(h, edge_index, edge_weight, last_gamma, last_beta)

        # Далі без змін, як було...
        if self.cascade_stop_grad:
            pd_logit, cascade_gate, cascade_size = self.dual_head(
                h, last_gamma, last_beta, x,
                h_detached=h.detach(),
            )
        else:
            pd_logit, cascade_gate, cascade_size = self.dual_head(
                h, last_gamma, last_beta, x,
            )

        return pd_logit, cascade_gate, cascade_size

    def get_embeddings(self, x, edge_index, edge_weight, stress) -> torch.Tensor:
        # 1. Розширюємо стрес-вектор
        if stress.dim() == 1:
            stress_expanded = stress.unsqueeze(0).expand(x.size(0), -1)
        else:
            stress_expanded = stress.expand(x.size(0), -1)

        # 2. ОСЬ ФІКС: Конкатенуємо так само, як у forward!
        h = self.input_proj(torch.cat([x, stress_expanded], dim=-1))

        # 3. Пропускаємо через шари
        last_gamma, last_beta = None, None
        for layer, enc in zip(self.gnn_layers, self.stress_encoders):
            last_gamma, last_beta = enc(stress)
            h = layer(h, edge_index, edge_weight, last_gamma, last_beta)

        return h.detach()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

