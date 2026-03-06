"""
Stress scenario generation and train/val/test splitting.
Each scenario:
  1. Sample stress vector s in [0,1]^stress_dim
  2. Compute stress-conditioned STOCHASTIC default per node
  3. Run multi-trigger EN cascade for contagion defaults
  4. Compute per-node cascade sizes (full, parallelized)
"""

import numpy as np
from scipy import sparse as sp
from tqdm import tqdm
from typing import Dict, List, Tuple

from config import Config
from .cascade import multi_trigger_cascade, compute_all_cascades


def generate_scenarios(
    cfg: Config,
    node_features: np.ndarray,
    adj_sparse: sp.csr_matrix,
    labels: np.ndarray,
) -> List[Dict]:
    rng = np.random.RandomState(cfg.seed)
    N = node_features.shape[0]

    adj_T = adj_sparse.T.tocsr()
    liabilities = np.array(adj_sparse.sum(axis=1)).flatten()

    # FIX 2: Compute assets taking into account incoming payments
    base_assets = _compute_base_assets(labels, rng, N, adj_sparse)

    stress_sens = _compute_stress_sensitivity(node_features)
    fund_pd = _compute_fundamental_pd(node_features, labels)

    scenarios = []
    print(f"\n[Scenarios] Generating {cfg.num_scenarios} stress scenarios...")
    print(f"  Workers: {cfg.cascade_workers}, LGD: {cfg.lgd}")
    print(f"  Target default rate: {cfg.base_default_rate:.1%} – "
          f"{cfg.stress_max_default_rate:.1%}")

    for i in tqdm(range(cfg.num_scenarios), desc="Scenarios"):
        stress = rng.uniform(0, 1, cfg.stress_dim).astype(np.float32)
        sl = stress.mean()

        # FIX 1: Probabilistic default sampling instead of deterministic thresholds
        initial_defaults = _sample_stochastic_defaults(
            fund_pd, stress_sens, sl,
            cfg.base_default_rate, cfg.stress_max_default_rate, rng
        )

        # Stress affects:
        #   1. External assets decrease (economic downturn)
        #   2. LGD increases (recovery rates drop)
        #   3. Interbank obligations (liabilities) stay UNCHANGED
        asset_shock = 1.0 - 0.25 * sl       # up to -25% asset reduction
        stressed_lgd = min(0.9, cfg.lgd + 0.3 * sl)  # LGD: 0.45 → up to 0.75
        s_assets = base_assets * asset_shock

        all_defaults, n_contagion = multi_trigger_cascade(
            adj_T, liabilities, s_assets, initial_defaults, stressed_lgd,
        )

        cascade_sizes = compute_all_cascades(
            adj_T, liabilities, s_assets, stressed_lgd,
            n_jobs=cfg.cascade_workers,
        )

        sc = {
            "stress": stress,
            "labels": all_defaults.astype(np.float32),
            "cascade_sizes": cascade_sizes,
            "n_initial_defaults": int(initial_defaults.sum()),
            "n_contagion_defaults": int(n_contagion),
            "contagion_mask": (all_defaults & ~initial_defaults),
            "default_rate": float(all_defaults.mean()),
        }
        scenarios.append(sc)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}] default_rate={sc['default_rate']:.2%}, "
                  f"contagion={n_contagion}, stress={sl:.2f}")

    _print_summary(scenarios)
    return scenarios


def split_scenarios(
    scenarios: List[Dict], cfg: Config,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    n = len(scenarios)
    n_test = max(2, int(n * cfg.test_ratio))
    n_val = max(1, int(n * cfg.val_ratio))
    n_train = n - n_test - n_val

    sc_train = scenarios[:n_train]
    sc_val = scenarios[n_train:n_train + n_val]
    sc_test = scenarios[n_train + n_val:]

    print(f"\n[Split] {len(sc_train)} train / {len(sc_val)} val / "
          f"{len(sc_test)} test")
    return sc_train, sc_val, sc_test


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _compute_fundamental_pd(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """PD from credit ratings + feature-based perturbation."""
    N = features.shape[0]
    n_feat = features.shape[1]

    if n_feat >= 70:
        base_pd = np.array([0.01, 0.05, 0.15, 0.40])[np.clip(labels, 0, 3)]
        equity = features[:, 2]
        liquidity = features[:, 3]
        impaired = features[:, 12] if n_feat > 12 else 0.5
        cet1 = features[:, 17] if n_feat > 17 else 0.5

        health = 0.3 * equity + 0.25 * liquidity + 0.25 * cet1 + 0.2 * (1 - impaired)
        multiplier = 1.5 - health
        pd = base_pd * multiplier
        return np.clip(pd, 0.001, 0.95).astype(np.float32)
    else:
        w = np.zeros(n_feat)
        w[0] = 3.0; w[1] = -2.0
        if n_feat > 5: w[5] = -1.5
        z = features @ w - 1.0
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _compute_stress_sensitivity(features: np.ndarray) -> np.ndarray:
    """How sensitive is each node to macro stress. in [0, 1]."""
    n_feat = features.shape[1]
    if n_feat >= 70:
        wholesale = features[:, 10] if n_feat > 10 else 0.5
        liquidity = features[:, 3]
        interbank = features[:, 7] if n_feat > 7 else 0.5
        z = 2.0 * wholesale - 1.5 * liquidity + 0.8 * interbank - 0.5
    else:
        z = 2.0 * features[:, 0] - 1.5 * features[:, 1] - 0.3
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _sample_stochastic_defaults(fund_pd, stress_sens, stress_level, base_rate, max_rate, rng):
    """
    STOCHASTIC FIX: Compute stressed PDs and sample defaults probabilistically
    so ML models cannot simply reverse-engineer a deterministic threshold.
    """
    # 1. Determine target macro default rate for this stress level
    target_rate = base_rate + (max_rate - base_rate) * stress_level

    # 2. Shift the odds ratio based on bank's specific stress sensitivity
    odds = fund_pd / (1.001 - fund_pd)
    stress_multiplier = np.exp(stress_sens * stress_level * 3.0)
    new_odds = odds * stress_multiplier

    pd_stressed = new_odds / (1.0 + new_odds)

    # 3. Calibrate globally so the expected default rate perfectly matches the macro target
    current_rate = pd_stressed.mean()
    if current_rate > 0:
        pd_stressed *= (target_rate / current_rate)

    pd_stressed = np.clip(pd_stressed, 0.001, 0.95)

    # 4. Stochastic sampling
    return rng.rand(len(fund_pd)) < pd_stressed


def _compute_base_assets(labels, rng, N, adj_sparse=None):
    """
    ASSET FIX: External assets must account for BOTH liabilities (money owed)
    AND incoming payments (money expected). Otherwise, nodes with high liabilities
    but no incoming edges will structurally default at step 0.
    """
    rating_buffer = np.array([1.50, 1.30, 1.15, 1.05])[np.clip(labels, 0, 3)]

    if adj_sparse is not None:
        liabilities = np.array(adj_sparse.sum(axis=1)).flatten()
        incoming = np.array(adj_sparse.sum(axis=0)).flatten()  # Money coming in

        liabilities = np.maximum(liabilities, 1.0)

        # Bank is solvent if: external_assets + incoming >= buffer * liabilities
        # Therefore: external_assets = max(buffer * liabilities - incoming, safe_minimum)
        assets = (rating_buffer * liabilities) - incoming

        # Ensure a safe minimum external asset floor (e.g. 15% of liabilities)
        # to prevent instant defaults on minor asset shocks
        min_assets = 0.15 * liabilities
        assets = np.maximum(assets, min_assets)

        noise = rng.uniform(0.9, 1.1, N).astype(np.float32)
        assets = assets * noise
    else:
        assets = rating_buffer * 100.0 + rng.exponential(10.0, N).astype(np.float32)

    return assets.astype(np.float32)


def _print_summary(scenarios):
    rates = [s["default_rate"] for s in scenarios]
    cont = [s["n_contagion_defaults"] / max(1, s["labels"].sum())
            for s in scenarios]
    print(f"\n[Scenarios] Summary:")
    print(f"  Default rate: {min(rates):.2%} – {max(rates):.2%}")
    print(f"  Contagion fraction: {np.mean(cont):.1%} avg")
    print(f"  Contagion defaults: "
          f"{np.mean([s['n_contagion_defaults'] for s in scenarios]):.0f} avg/scenario")