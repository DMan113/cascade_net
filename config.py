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
    synthetic: bool = False

    node_feature_dim: int = 70
    stress_dim: int = 8

    num_nodes: int = 5_000
    ba_edges_per_node: int = 3
    synth_feature_dim: int = 15

    # ── Stress scenarios ─────────────────────────────────────────────
    num_scenarios: int = 200
    base_default_rate: float = 0.05
    stress_max_default_rate: float = 0.25
    lgd: float = 0.45
    stress_contagion_mult: float = 2.0
    cascade_workers: int = -1

    # ── FIX 1: Stress shock parameters (зменшені) ───────────────────
    asset_shock_max: float = 0.20         # було неявно 0.25
    lgd_stress_max: float = 0.20         # було неявно 0.30
    lgd_cap: float = 0.70               # було неявно 0.90

    # ── Train / val / test split ─────────────────────────────────────
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Model ────────────────────────────────────────────────────────
    hidden_dim: int = 128
    stress_emb_dim: int = 32
    num_gnn_layers: int = 4
    dropout: float = 0.1
    head_hidden: int = 128

    gamma_min: float = 0.5
    gamma_max: float = 2.0

    # ── Training ─────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    patience: int = 25
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    cascade_weight_start: float = 0.05
    cascade_weight_end: float = 0.3

    # ── FIX 2: Нові прапорці для абляцій ────────────────────────────
    cascade_stop_grad: bool = False      # stop-gradient перед каскадною головою
    pd_only_ablation: bool = False       # вимкнути каскадний лосс повністю

    # ── Baseline training ────────────────────────────────────────────
    baseline_epochs: int = 150
    baseline_lr: float = 1e-3
    baseline_patience: int = 25

    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_lr: float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────
    num_seeds: int = 1
    precision_at_k: float = 0.10

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "./outputs"
    save_model: bool = True
    plot_results: bool = True

    # ── System ───────────────────────────────────────────────────────
    seed: int = 42
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_seed(self.seed)
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
        return self.node_feature_dim

    def for_quick_test(self) -> "Config":
        self.num_nodes = 500
        self.num_scenarios = 10
        self.epochs = 5
        self.baseline_epochs = 5
        self.patience = 3
        self.baseline_patience = 3
        self.grad_accum_steps = 2
        return self

    patience: int = 15
    grad_accum_steps: int = 8        # accumulate gradients over N scenarios
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
        self.num_nodes = 500
        self.num_scenarios = 10
        self.epochs = 5
        self.baseline_epochs = 5
        self.patience = 3
        self.baseline_patience = 3
        self.grad_accum_steps = 2
        return self