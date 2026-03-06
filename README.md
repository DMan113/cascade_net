# CascadeNet — Stress-Conditioned Cascade Risk Predictor

Predicting default cascades in financial interbank networks using
FiLM-conditioned Graph Neural Networks with Eisenberg-Noe contagion dynamics.

## Key Ideas

- **FiLM stress conditioning** — macro-stress vectors modulate GNN message passing
  via learned γ/β (scale/shift), not just concatenation
- **Dual-head prediction** — Head A predicts node-level default probability;
  Head B predicts cascade size (gate + magnitude)
- **Eisenberg-Noe cascades** — ground-truth labels from full fixed-point
  contagion simulation on every node (no imputation)
- **Real data backbone** — node features and network topology from
  [AI4Risk/interbank](https://github.com/AI4Risk/interbank) dataset
  (4 548 banks, 70 financial features, 32 quarters)

## Project Structure

```
cascadenet/
│
├── config.py                  # All hyperparameters in one place
│
├── data/
│   ├── __init__.py
│   ├── loader.py              # AI4Risk dataset loader + synthetic fallback
│   ├── cascade.py             # Eisenberg-Noe simulation (parallelized, full)
│   └── scenarios.py           # Stress scenario generation & data splits
│
├── models/
│   ├── __init__.py
│   ├── stress_encoder.py      # FiLM stress encoder (γ/β generation)
│   ├── gnn_layers.py          # StressCondGNNLayer, VanillaGNNLayer
│   ├── heads.py               # DualHead (PD + cascade gate/magnitude)
│   ├── cascadenet.py          # Full CascadeNet assembly
│   └── baselines.py           # VanillaGNN, StressSkipGNN
│
├── training/
│   ├── __init__.py
│   ├── loss.py                # Multi-task loss with staged cascade weight
│   ├── trainer.py             # CascadeNet trainer (gradient accumulation)
│   └── baseline_trainer.py    # Simplified trainer for baseline GNNs
│
├── evaluation/
│   ├── __init__.py
│   ├── runner.py              # Run all baselines (B1–B6), collect metrics
│   ├── metrics.py             # AUROC, AUPRC, Brier score, Spearman ρ
│   └── visualization.py       # 4-panel results figure
│
├── main.py                    # Full pipeline entry point
└── README.md                  # ← you are here
```

## Data

### AI4Risk/interbank (primary)

Download from: https://github.com/AI4Risk/interbank

Expected layout after cloning:

```
data/interbank/
├── datasets/
│   ├── {YYYY}Q{Q}_edge.csv    # Edge tables (source, target, weight)
│   ├── {YYYY}Q{Q}_feature.csv # 70 financial features + 3 label columns
│   └── ...                     # 32 quarters: 2016Q1 → 2023Q4
```

The loader reads a chosen quarter (default: 2022Q4), extracts:
- **Node features**: 70 normalized financial indicators (first 70 columns)
- **Edge index + weights**: directed interbank exposure network
- **Labels**: credit rating (A/B/C/D → ordinal 0–3), SRISK ratio, SRISK value

### Synthetic fallback

If AI4Risk data is unavailable, `loader.py` falls back to generating a
Barabási-Albert graph with synthetic features. This mode is clearly
flagged in all outputs and should only be used for debugging.

## Baselines

| ID | Model                        | Purpose                                  |
|----|------------------------------|------------------------------------------|
| B1 | XGBoost (node features)      | No graph, no stress structure             |
| B2 | XGBoost + network features   | Static topology value                     |
| B3 | VanillaGNN (concat stress)   | GNN without FiLM — isolates conditioning |
| B4 | StressSkipGNN                | Stress on every layer (skip connections)  |
| B5 | **CascadeNet** (full)        | FiLM conditioning + dual head             |
| B6 | GNN Embeddings + XGBoost     | Hybrid: learned features + tree model     |

### Ablation logic

- **Graph value**: B2 − B1 (do static topology features help?)
- **GNN value**: B5 − B2 (does message passing help beyond features?)
- **FiLM value**: B5 − B3 (does multiplicative conditioning beat concat?)
- **Skip value**: B5 − B4 (is FiLM better than skip-connected stress?)
- **Hybrid value**: B6 − B5 (does XGBoost extract more from embeddings?)

## Key Fixes over Original Prototype

### Data integrity
- Full EN cascade computation for **all** nodes (parallelized via joblib)
- No linear imputation of cascade sizes
- Directed graph with asymmetric exposures (not symmetric `to_directed()`)

### Evaluation
- Consistent train/val/test splits across all models
- XGBoost trains on train scenarios, evaluates on test (same as GNN)
- Added AUPRC, Brier score alongside AUROC
- Confidence intervals via multi-seed runs

### Architecture
- StressEncoder γ range: (0.5, 2.0) via scaled sigmoid — allows amplification
- DualHead receives `[γ, β]` concatenated, not averaged
- Gradient accumulation over N scenarios before optimizer step
- StressSkipGNN baseline for fair FiLM comparison
- Per-node edge weight normalization (degree-scaled)

### Calibration
- Default rates calibrated to realistic ranges (1–15% baseline, up to 30% stressed)
- Stress scenarios can be calibrated from EBA adverse scenario parameters

## Usage

```bash
# Full pipeline with AI4Risk data
python main.py --data-dir ./data/interbank/datasets --quarter 2022Q4

# Quick test with synthetic data
python main.py --synthetic --quick

# Multi-seed evaluation (5 seeds)
python main.py --data-dir ./data/interbank/datasets --seeds 5
```

## Requirements

```
torch >= 2.0
torch-geometric >= 2.4
networkx
numpy
scipy
scikit-learn
xgboost
matplotlib
joblib
tqdm
pandas
```

## Citation

If you use the AI4Risk dataset, please cite their work:

```bibtex
@article{li2024hftcrnet,
  title   = {HFTCRNet: Hierarchical Fusion Transformer for Interbank
             Credit Rating and Risk Assessment},
  author  = {Li, Jiangtong and Zhou, Ziyuan and Zhang, Jingkai
             and Cheng, Dawei and Jiang, Changjun},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2024}
}
```