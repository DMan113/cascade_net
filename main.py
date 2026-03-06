"""
CascadeNet — Full Pipeline

Usage:
    python main.py --data-dir ./data/interbank/datasets --quarter 2022Q4
    python main.py --synthetic --quick
    python main.py --data-dir ./data/interbank/datasets --seeds 5
"""

import os
import argparse
import warnings
import json

import numpy as np
import torch

warnings.filterwarnings("ignore")

from config import Config
from datapipe import load_data, generate_scenarios, split_scenarios
from models import CascadeNet, VanillaGNN, StressSkipGNN
from training import CascadeNetTrainer, BaselineGNNTrainer
from evaluation import EvaluationRunner, plot_results


def run_pipeline(cfg: Config) -> dict:
    """Execute the full CascadeNet pipeline for a single seed."""

    # ── Step 1: Load data ────────────────────────────────────────
    print("=" * 72)
    print("  STEP 1: Loading Data")
    print("=" * 72)
    dataset = load_data(cfg)
    data = dataset["pyg_data"]
    meta = dataset["metadata"]

    print(f"\n  Source: {meta['data_source']}")
    print(f"  Nodes: {meta['num_nodes']}, Edges: {meta['num_edges']}")
    print(f"  Avg degree: {meta['avg_degree']:.1f}")

    # ── Step 2: Generate stress scenarios ────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 2: Generating Stress Scenarios")
    print("=" * 72)
    scenarios = generate_scenarios(
        cfg,
        node_features=dataset["node_features_raw"],
        adj_sparse=dataset["adj_sparse"],
        labels=dataset["labels"],
    )
    sc_train, sc_val, sc_test = split_scenarios(scenarios, cfg)

    # ── Step 3: Train CascadeNet ─────────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 3: Training CascadeNet")
    print("=" * 72)
    model = CascadeNet(cfg)
    trainer = CascadeNetTrainer(model, cfg)
    trained_model = trainer.fit(data, sc_train, sc_val)

    # ── Step 4: Train B3 (VanillaGNN) ────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 4: Training Baselines")
    print("=" * 72)

    b3 = VanillaGNN(cfg)
    b3_trainer = BaselineGNNTrainer(b3, cfg, name="B3-VanillaGNN")
    trained_b3 = b3_trainer.fit(data, sc_train, sc_val)

    # ── Step 5: Train B4 (StressSkipGNN) ─────────────────────────
    b4 = StressSkipGNN(cfg)
    b4_trainer = BaselineGNNTrainer(b4, cfg, name="B4-StressSkipGNN")
    trained_b4 = b4_trainer.fit(data, sc_train, sc_val)

    # ── Step 6: Evaluate all models ──────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 5: Evaluation (B1-B6)")
    print("=" * 72)
    runner = EvaluationRunner(cfg)
    results, cas_met, cascade_preds = runner.run_all(
        node_features=dataset["node_features_raw"],
        network_features=dataset["network_features"],
        cascadenet_model=trained_model,
        b3_model=trained_b3,
        b4_model=trained_b4,
        data=data,
        sc_train=sc_train,
        sc_test=sc_test,
    )

    # ── Step 7: Visualize ────────────────────────────────────────
    if cfg.plot_results:
        print("\n" + "=" * 72)
        print("  STEP 6: Visualization")
        print("=" * 72)
        os.makedirs(cfg.output_dir, exist_ok=True)
        fig_path = os.path.join(cfg.output_dir, "cascadenet_results.png")
        plot_results(
            results, cas_met, trainer.history,
            sc_test, cascade_preds, fig_path,
        )

    # ── Step 8: Save model ───────────────────────────────────────
    if cfg.save_model:
        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, "cascadenet.pt")
        torch.save({
            "model_state": trained_model.state_dict(),
            "config": cfg,
            "metadata": meta,
        }, save_path)
        print(f"\n  Model saved to {save_path}")

    return {
        "results": results,
        "cascade_metrics": cas_met,
        "metadata": meta,
    }


def main():
    parser = argparse.ArgumentParser(description="CascadeNet Pipeline")
    parser.add_argument("--data-dir", type=str, default="./data/interbank/datasets")
    parser.add_argument("--quarter", type=str, default="2022Q4")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        quarter=args.quarter,
        synthetic=args.synthetic,
        output_dir=args.output_dir,
        num_seeds=args.seeds,
    )
    if args.device:
        cfg.device = args.device
    if args.quick:
        cfg.for_quick_test()
        print("[Quick mode]\n")

    if cfg.num_seeds == 1:
        run_pipeline(cfg)
    else:
        # Multi-seed evaluation for confidence intervals
        all_results = []
        for seed in range(cfg.num_seeds):
            print(f"\n{'#' * 72}")
            print(f"  SEED {seed + 1}/{cfg.num_seeds}")
            print(f"{'#' * 72}")

            cfg.set_seed(cfg.seed + seed)
            out = run_pipeline(cfg)
            all_results.append(out["results"])

        _print_multi_seed_summary(all_results)


def _print_multi_seed_summary(all_results):
    """Print mean ± std across seeds."""
    print("\n" + "=" * 72)
    print("  Multi-Seed Summary (mean ± std)")
    print("=" * 72)

    # Collect per-model AUROC across seeds
    model_names = [r["name"] for r in all_results[0]]
    for name in model_names:
        aurocs = []
        auprcs = []
        for seed_results in all_results:
            r = next(x for x in seed_results if x["name"] == name)
            aurocs.append(r["auroc"])
            auprcs.append(r.get("auprc", 0))

        aurocs = np.array(aurocs)
        auprcs = np.array(auprcs)
        print(f"  {name:<30s}  AUROC: {aurocs.mean():.4f} ± {aurocs.std():.4f}  "
              f"AUPRC: {auprcs.mean():.4f} ± {auprcs.std():.4f}")


if __name__ == "__main__":
    main()

