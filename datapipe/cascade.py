"""
Eisenberg-Noe cascade simulation — full computation, parallelized.

No imputation: cascade size is computed for every node via fixed-point
iteration on the interbank payment system.
"""

import numpy as np
from scipy import sparse as sp
from joblib import Parallel, delayed
from typing import Tuple


def eisenberg_noe_cascade(
    adj_T: sp.csr_matrix,
    liabilities: np.ndarray,
    assets: np.ndarray,
    trigger_node: int,
    lgd: float,
    max_rounds: int = 50,
) -> int:
    """
    Compute cascade size when a single node defaults.

    Uses Eisenberg-Noe fixed-point clearing: defaulted nodes pay
    (1 - LGD) fraction of their obligations. Iterate until no new
    defaults occur.

    Args:
        adj_T: transposed weighted adjacency (N×N sparse).
                adj_T[i,j] = exposure of i to j.
        liabilities: total liabilities per node (N,).
        assets: external assets per node (N,).
        trigger_node: index of initially defaulting node.
        lgd: loss given default ∈ (0, 1).
        max_rounds: max contagion iterations.

    Returns:
        Number of additional defaults caused (excluding trigger).
    """
    N = len(assets)
    solvent = np.ones(N, dtype=bool)
    solvent[trigger_node] = False

    for _ in range(max_rounds):
        # Payment rates: solvent nodes pay 100%, defaulted pay (1-LGD)
        pay_rate = np.where(solvent, 1.0, 1.0 - lgd)

        # Income = external assets + payments received from counterparties
        income = assets + adj_T @ pay_rate

        # Check solvency
        new_solvent = solvent.copy()
        new_solvent[solvent & (income < liabilities)] = False

        if np.array_equal(new_solvent, solvent):
            break
        solvent = new_solvent

    # Cascade size = total defaults minus the trigger
    return max(0, int((~solvent).sum()) - 1)


def multi_trigger_cascade(
    adj_T: sp.csr_matrix,
    liabilities: np.ndarray,
    assets: np.ndarray,
    initial_defaults: np.ndarray,
    lgd: float,
    max_rounds: int = 50,
) -> Tuple[np.ndarray, int]:
    """
    Cascade from multiple simultaneous initial defaults.

    Args:
        initial_defaults: boolean mask (N,) of fundamentally defaulting nodes.

    Returns:
        (all_defaults_mask, n_contagion_defaults)
    """
    N = len(assets)
    solvent = ~initial_defaults.copy()

    for _ in range(max_rounds):
        pay_rate = np.where(solvent, 1.0, 1.0 - lgd)
        income = assets + adj_T @ pay_rate
        new_solvent = solvent.copy()
        new_solvent[solvent & (income < liabilities)] = False

        if np.array_equal(new_solvent, solvent):
            break
        solvent = new_solvent

    all_defaults = ~solvent
    n_contagion = max(0, int(all_defaults.sum() - initial_defaults.sum()))
    return all_defaults, n_contagion


def compute_all_cascades(
    adj_T: sp.csr_matrix,
    liabilities: np.ndarray,
    assets: np.ndarray,
    lgd: float,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute single-trigger cascade size for EVERY node. Parallelized.

    This is the key fix over the original: no linear imputation.
    Full EN fixed-point for each of N nodes.

    Args:
        n_jobs: number of parallel workers. -1 = all cores.

    Returns:
        cascade_sizes: (N,) array of cascade sizes per node.
    """
    N = len(assets)

    def _single(i):
        return eisenberg_noe_cascade(adj_T, liabilities, assets, i, lgd)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single)(i) for i in range(N)
    )

    return np.array(results, dtype=np.float32)