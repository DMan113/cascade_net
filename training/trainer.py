"""
CascadeNet trainer with:
  - Gradient accumulation over multiple scenarios
  - Staged cascade weight schedule
  - Cosine annealing LR
  - Training history for visualization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple

from config import Config
from .loss import cascadenet_loss, compute_pos_weight


class CascadeNetTrainer:
    def __init__(self, model: nn.Module, cfg: Config):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=1e-5,
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "cascade_weight": [],
        }

    def _cascade_weight(self, epoch: int) -> float:
        """Linear ramp from cascade_weight_start → cascade_weight_end."""
        t = min(1.0, epoch / self.cfg.epochs)
        s, e = self.cfg.cascade_weight_start, self.cfg.cascade_weight_end
        return s + (e - s) * t

    def train_epoch(self, data, scenarios: List[Dict], epoch: int) -> float:
        self.model.train()
        dev = self.cfg.device
        cw = self._cascade_weight(epoch)
        accum = self.cfg.grad_accum_steps

        self.optimizer.zero_grad()
        total_loss, n_steps, n_accum = 0.0, 0, 0

        for i, sc in enumerate(scenarios):
            x = data.x.to(dev)
            ei = data.edge_index.to(dev)
            ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
            s = torch.tensor(sc["stress"], device=dev)
            y = torch.tensor(sc["labels"], device=dev)
            cas = torch.tensor(sc["cascade_sizes"], device=dev)
            pw = compute_pos_weight(sc["labels"])

            pl, cg, cs = self.model(x, ei, ew, s)
            loss = cascadenet_loss(pl, cg, cs, y, cas, pw, cw)

            # Scale loss for accumulation
            (loss / accum).backward()
            total_loss += loss.item()
            n_steps += 1
            n_accum += 1

            if n_accum >= accum or i == len(scenarios) - 1:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                n_accum = 0

        self.scheduler.step()
        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def validate(self, data, scenarios: List[Dict], epoch: int) -> float:
        self.model.eval()
        dev = self.cfg.device
        cw = self._cascade_weight(epoch)
        total, n = 0.0, 0

        for sc in scenarios:
            x = data.x.to(dev)
            ei = data.edge_index.to(dev)
            ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
            s = torch.tensor(sc["stress"], device=dev)
            y = torch.tensor(sc["labels"], device=dev)
            cas = torch.tensor(sc["cascade_sizes"], device=dev)
            pw = compute_pos_weight(sc["labels"])

            pl, cg, cs = self.model(x, ei, ew, s)
            loss = cascadenet_loss(pl, cg, cs, y, cas, pw, cw)
            total += loss.item()
            n += 1

        return total / max(n, 1)

    def fit(
        self, data, sc_train: List[Dict], sc_val: List[Dict],
    ) -> nn.Module:
        cfg = self.cfg
        print(f"\n[CascadeNet] Training")
        print(f"  {len(sc_train)} train / {len(sc_val)} val scenarios")
        print(f"  Params: {self.model.count_params():,}")
        print(f"  Device: {cfg.device}")
        print(f"  Grad accum: {cfg.grad_accum_steps} steps")
        print(f"  Cascade weight: {cfg.cascade_weight_start} → "
              f"{cfg.cascade_weight_end}\n")

        for ep in range(1, cfg.epochs + 1):
            np.random.shuffle(sc_train)
            tl = self.train_epoch(data, sc_train, ep)
            vl = self.validate(data, sc_val, ep)
            cw = self._cascade_weight(ep)

            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["cascade_weight"].append(cw)

            if ep % 5 == 0 or ep == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {ep:3d}/{cfg.epochs} │ "
                      f"Train: {tl:.4f} │ Val: {vl:.4f} │ "
                      f"LR: {lr:.6f} │ CW: {cw:.3f}")

            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.patience:
                    print(f"\n  Early stopping at epoch {ep}")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
            self.model.to(cfg.device)

        print(f"  Best val loss: {self.best_val_loss:.4f}")
        return self.model