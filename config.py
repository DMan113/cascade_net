"""
CascadeNet — Configuration
All hyperparameters in one place. Override via CLI or constructor.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
import numpy as np


@dataclass
class Config:
    # ── Data ─────────────────────────────────────────────────────────
    data_dir: str = "./data/interbank/datasets"
    quarter: str = "2022Q4"
    synthetic: bool = False          # fallback: generate BA graph

    # AI4Risk dataset
    node_feature_dim: int = 70       # 70 financial features from AI4Risk
    stress_dim: int = 8              # macro-stress vector dimensionality

    # Synthetic fallback only
    num_nodes: int = 5_000
    ba_edges_per_node: int = 3
    synth_feature_dim: int = 15

    # ── Stress scenarios ─────────────────────────────────────────────
    num_scenarios: int = 50
    base_default_rate: float = 0.05  # realistic: 1-5% baseline
    stress_max_default_rate: float = 0.25  # up to 25% under severe stress
    lgd: float = 0.45               # loss given default (Basel standard)
    stress_contagion_mult: float = 2.0
    cascade_workers: int = -1        # joblib: -1 = all cores

    # ── Train / val / test split ─────────────────────────────────────
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Model ────────────────────────────────────────────────────────
    hidden_dim: int = 64
    stress_emb_dim: int = 32
    num_gnn_layers: int = 3
    dropout: float = 0.1
    head_hidden: int = 128

    # StressEncoder γ range — allows amplification (0.5, 2.0)
    gamma_min: float = 0.5
    gamma_max: float = 2.0

    # ── Training ─────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 15
    grad_accum_steps: int = 4        # accumulate gradients over N scenarios
    max_grad_norm: float = 1.0

    # Staged cascade weight: ramps linearly from start → end
    cascade_weight_start: float = 0.05
    cascade_weight_end: float = 0.3

    # ── Baseline training ────────────────────────────────────────────
    baseline_epochs: int = 80
    baseline_lr: float = 1e-3
    baseline_patience: int = 15

    # XGBoost
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_lr: float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────
    num_seeds: int = 1               # multi-seed for confidence intervals
    precision_at_k: float = 0.10     # top-k% for precision metric

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "./outputs"
    save_model: bool = True
    plot_results: bool = True

    # ── System ───────────────────────────────────────────────────────
    seed: int = 42
    device: str = ""                 # auto-detect if empty

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_seed(self.seed)

        # In AI4Risk mode, feature dim is always 70
        # In synthetic mode, it's synth_feature_dim
        if self.synthetic:
            self.node_feature_dim = self.synth_feature_dim

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def input_dim(self) -> int:
        """Actual input dimension for models."""
        return self.node_feature_dim

    def for_quick_test(self) -> "Config":
        """Return a copy configured for fast debugging."""
        self.num_nodes = 500
        self.num_scenarios = 10
        self.epochs = 5
        self.baseline_epochs = 5
        self.patience = 3
        self.baseline_patience = 3
        self.grad_accum_steps = 2
        return self