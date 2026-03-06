"""
Trainer for GNN baselines (B3: VanillaGNN, B4: StressSkipGNN).

Simplified: classification only (no cascade head).
Same LR schedule and early stopping as CascadeNet for fairness.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict

from config import Config
from .loss import compute_pos_weight


class BaselineGNNTrainer:
    def __init__(self, model: nn.Module, cfg: Config, name: str = "Baseline"):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.name = name
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.baseline_lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.baseline_epochs, eta_min=1e-5,
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state = None

    def train_epoch(self, data, scenarios: List[Dict]) -> float:
        self.model.train()
        dev = self.cfg.device
        total, n = 0.0, 0

        for sc in scenarios:
            x = data.x.to(dev)
            ei = data.edge_index.to(dev)
            ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
            s = torch.tensor(sc["stress"], device=dev)
            y = torch.tensor(sc["labels"], device=dev)
            pw = compute_pos_weight(sc["labels"])

            self.optimizer.zero_grad()
            logit = self.model(x, ei, ew, s)
            loss = F.binary_cross_entropy_with_logits(
                logit, y, pos_weight=torch.tensor([pw], device=dev),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total += loss.item()
            n += 1

        self.scheduler.step()
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, data, scenarios: List[Dict]) -> float:
        self.model.eval()
        dev = self.cfg.device
        total, n = 0.0, 0

        for sc in scenarios:
            x = data.x.to(dev)
            ei = data.edge_index.to(dev)
            ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
            s = torch.tensor(sc["stress"], device=dev)
            y = torch.tensor(sc["labels"], device=dev)
            pw = compute_pos_weight(sc["labels"])

            logit = self.model(x, ei, ew, s)
            loss = F.binary_cross_entropy_with_logits(
                logit, y, pos_weight=torch.tensor([pw], device=dev),
            )
            total += loss.item()
            n += 1

        return total / max(n, 1)

    def fit(self, data, sc_train: List[Dict], sc_val: List[Dict]) -> nn.Module:
        cfg = self.cfg
        print(f"\n  [{self.name}] Training ({self.model.count_params():,} params)...")

        for ep in range(1, cfg.baseline_epochs + 1):
            np.random.shuffle(sc_train)
            tl = self.train_epoch(data, sc_train)
            vl = self.validate(data, sc_val)

            if ep % 10 == 0:
                print(f"    {self.name} Epoch {ep:3d}/{cfg.baseline_epochs} │ "
                      f"Train: {tl:.4f} │ Val: {vl:.4f}")

            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.baseline_patience:
                    print(f"    {self.name} early stopping at epoch {ep}")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
            self.model.to(cfg.device)

        print(f"  [{self.name}] Best val loss: {self.best_val_loss:.4f}")
        return self.model

