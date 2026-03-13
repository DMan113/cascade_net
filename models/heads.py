
"""
Prediction heads for CascadeNet.

DualHead:
  - Head A: default probability (binary classification logit)
  - Head B: cascade size = gate × magnitude

Fix: додано h_detached для ізоляції каскадних градієнтів від backbone.
"""

import torch
import torch.nn as nn


class DualHead(nn.Module):
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

        # ДВІ НЕЗАЛЕЖНІ гілки для обробки сирих фіч (вирішує конфлікт градієнтів)
        self.x_proj_pd = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
        )

        self.x_proj_cas = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
        )

        # FIX: Нормалізація перед злиттям (LayerNorm для кожного потоку)
        self.norm_h = nn.LayerNorm(h_dim)
        self.norm_s = nn.LayerNorm(stress_emb_dim)
        self.norm_x_pd = nn.LayerNorm(head_hidden)
        self.norm_x_cas = nn.LayerNorm(head_hidden)

        inp_compressed = h_dim + stress_emb_dim + head_hidden

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

        # FIX: Dropout перенесено перед останнім шаром для кращої регуляризації
        self.pd_head = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self.cascade_gate = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self.cascade_magnitude = nn.Sequential(
            nn.Linear(head_hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, h, gamma, beta, x, h_detached=None):
        stress_cat = torch.cat([gamma, beta], dim=-1)
        stress_repr = self.stress_compress(stress_cat)
        stress_repr = stress_repr.expand(h.size(0), -1)

        # Нормалізуємо стрес
        s_norm = self.norm_s(stress_repr)

        # PD-голова отримує свою версію табличних фіч
        x_repr_pd = self.x_proj_pd(x)

        # Зливаємо НОРМАЛІЗОВАНІ тензори
        inp_pd = torch.cat([self.norm_h(h), s_norm, self.norm_x_pd(x_repr_pd)], dim=-1)
        pd_feat = self.pd_proj(inp_pd)
        pd = self.pd_head(pd_feat).squeeze(-1)

        # Каскадна голова отримує свою версію
        x_repr_cas = self.x_proj_cas(x)
        h_cas = h_detached if h_detached is not None else h

        # Зливаємо НОРМАЛІЗОВАНІ тензори
        inp_cas = torch.cat([self.norm_h(h_cas), s_norm, self.norm_x_cas(x_repr_cas)], dim=-1)
        cas_feat = self.cas_proj(inp_cas)

        gate = self.cascade_gate(cas_feat).squeeze(-1)
        mag = self.cascade_magnitude(cas_feat).squeeze(-1) + 1e-6

        return pd, gate, mag
