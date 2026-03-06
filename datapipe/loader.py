"""
Data loading: AI4Risk/interbank dataset or synthetic fallback.

AI4Risk layout (discovered via inspect_data.py):
    datasets/
    ├── edges/edge_2022Q4.csv     # Sourceid, Targetid, Weights
    └── nodes/2022Q4.csv          # index, 70 features, rank_next_quarter, srisk_ratio, srisk_value

Node columns (74 total):
    [0]     index                        — bank sequential ID
    [1:71]  Total_assets ... Net_interest_margin_(interest_earning_assets)  — 70 financial features
    [71]    rank_next_quarter            — credit rating 1-4 (mapped to 0-3)
    [72]    srisk_ratio                  — SRISK ratio (may be NaN)
    [73]    srisk_value                  — SRISK value (may be NaN)
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy import sparse as sp

from config import Config


def load_data(cfg: Config) -> Dict:
    if cfg.synthetic:
        print("[Data] Synthetic mode — generating directed BA graph")
        return _load_synthetic(cfg)
    return _load_ai4risk(cfg)


# ═══════════════════════════════════════════════════════════════════════
# AI4Risk loader
# ═══════════════════════════════════════════════════════════════════════

def _load_ai4risk(cfg: Config) -> Dict:
    root = Path(cfg.data_dir)
    q = cfg.quarter

    if not root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root}\n"
            f"  git clone https://github.com/AI4Risk/interbank.git\n"
            f"  Copy datasets/ folder into data/interbank/"
        )

    # ── Locate files ─────────────────────────────────────────────
    node_path = _find_node_file(root, q)
    edge_path = _find_edge_file(root, q)

    if node_path is None or edge_path is None:
        _print_available(root, q, edge_path, node_path)
        raise FileNotFoundError(f"Missing files for quarter {q}")

    print(f"[Data] Loading AI4Risk — quarter {q}")
    print(f"  Nodes: {node_path.relative_to(root)}")
    print(f"  Edges: {edge_path.relative_to(root)}")

    # ── Load node file ───────────────────────────────────────────
    node_df = pd.read_csv(node_path)
    N = len(node_df)

    # Column layout: index | 70 features | rank_next_quarter | srisk_ratio | srisk_value
    # Drop 'index' column, extract features and labels
    if "index" in node_df.columns or node_df.columns[0].lower() == "index":
        node_df = node_df.drop(columns=[node_df.columns[0]])

    # Last 3 columns are labels
    feature_cols = node_df.columns[:-3]
    X = node_df[feature_cols].values.astype(np.float32)

    # Labels
    rank_col = node_df.iloc[:, -3]  # rank_next_quarter (1-4)
    srisk_ratio = pd.to_numeric(node_df.iloc[:, -2], errors="coerce").fillna(0.0).values.astype(np.float32)
    srisk_value = pd.to_numeric(node_df.iloc[:, -1], errors="coerce").fillna(0.0).values.astype(np.float32)

    # Convert rank 1-4 → 0-3  (1=A=best, 4=D=worst)
    ratings = pd.to_numeric(rank_col, errors="coerce").fillna(2).values.astype(int)
    ratings = np.clip(ratings - 1, 0, 3)  # 1-based → 0-based

    # Handle NaN in features
    X = _normalize(X)
    n_feat = X.shape[1]

    print(f"  N={N}, features={n_feat}")
    print(f"  Ratings (0=A,1=B,2=C,3=D): {dict(zip(*np.unique(ratings, return_counts=True)))}")
    print(f"  SRISK ratio: {np.nanmean(srisk_ratio):.2f} avg, "
          f"{np.isnan(node_df.iloc[:, -2].values.astype(float)).sum()} NaN")

    # ── Load edges ───────────────────────────────────────────────
    edge_df = pd.read_csv(edge_path)
    # Columns: Sourceid, Targetid, Weights
    src = edge_df.iloc[:, 0].values.astype(int)
    tgt = edge_df.iloc[:, 1].values.astype(int)
    wt = edge_df.iloc[:, 2].values.astype(np.float32)

    # Remap if node IDs exceed N
    src, tgt, wt = _remap_edges(src, tgt, wt, N)
    wt = np.nan_to_num(wt, nan=1.0)
    wt_norm = wt / (wt.max() + 1e-8)

    num_edges = len(src)
    print(f"  Edges={num_edges}, avg_degree={num_edges / N:.1f}")

    # ── Symmetric Graph Normalization ────────────────────────────
    src_t = torch.tensor(src, dtype=torch.long)
    tgt_t = torch.tensor(tgt, dtype=torch.long)
    wt_t = torch.tensor(wt_norm, dtype=torch.float32)

    # Compute degree
    deg = torch.bincount(tgt_t, minlength=N).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # Apply symmetric normalization: edge_weight * (1 / sqrt(deg_src * deg_tgt))
    norm_wt = deg_inv_sqrt[src_t] * wt_t * deg_inv_sqrt[tgt_t]

    # ── Build PyG + sparse ───────────────────────────────────────
    pyg = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.stack([src_t, tgt_t]),
        edge_weight=norm_wt,
        num_nodes=N,
    )
    # Use raw weights for cascade simulation (not normalized)
    adj = sp.csr_matrix((wt, (src, tgt)), shape=(N, N))

    # ── Network features ─────────────────────────────────────────
    print(f"  Computing network centralities...")
    net_feats = _compute_network_features(src, tgt, wt_norm, N)

    cfg.node_feature_dim = n_feat

    return {
        "pyg_data": pyg,
        "adj_sparse": adj,
        "labels": ratings,
        "srisk_ratio": srisk_ratio,
        "srisk_value": srisk_value,
        "node_features_raw": X,
        "network_features": net_feats,
        "metadata": {
            "num_nodes": N, "num_edges": num_edges,
            "avg_degree": num_edges / N,
            "data_source": "ai4risk", "quarter": q,
            "rating_distribution": {
                int(k): int(v)
                for k, v in zip(*np.unique(ratings, return_counts=True))
            },
        },
    }


# ─── File finders (exact AI4Risk layout) ────────────────────────

def _find_node_file(root: Path, quarter: str) -> Optional[Path]:
    """Find node file: nodes/{quarter}.csv or nodes/node_{quarter}.csv"""
    candidates = [
        root / "nodes" / f"{quarter}.csv",
        root / "nodes" / f"node_{quarter}.csv",
        root / f"nodes_{quarter}.csv",
        root / f"{quarter}_node.csv",
        root / f"{quarter}_feature.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Glob fallback
    for pat in [f"nodes/*{quarter}*", f"*{quarter}*node*", f"*{quarter}*feature*"]:
        m = list(root.glob(pat))
        if m:
            return sorted(m)[0]
    return None


def _find_edge_file(root: Path, quarter: str) -> Optional[Path]:
    """Find edge file: edges/edge_{quarter}.csv"""
    candidates = [
        root / "edges" / f"edge_{quarter}.csv",
        root / "edges" / f"{quarter}_edge.csv",
        root / "edges" / f"{quarter}.csv",
        root / f"edge_{quarter}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    for pat in [f"edges/*{quarter}*", f"*{quarter}*edge*"]:
        m = list(root.glob(pat))
        if m:
            return sorted(m)[0]
    return None


def _print_available(root, quarter, edge_path, node_path):
    print(f"\n  Files not found for quarter {quarter}:")
    print(f"    edge: {'OK' if edge_path else 'MISSING'}")
    print(f"    node: {'OK' if node_path else 'MISSING'}")
    print(f"\n  Available:")
    for f in sorted(root.rglob("*.csv"))[:20]:
        print(f"    {f.relative_to(root)}")


# ─── Edge remapping ─────────────────────────────────────────────

def _remap_edges(src, tgt, wt, N):
    max_id = max(src.max(), tgt.max())
    if max_id < N and src.min() >= 0:
        return src, tgt, wt

    all_ids = np.unique(np.concatenate([src, tgt]))
    id_map = {old: new for new, old in enumerate(sorted(all_ids))}

    valid = np.array([
        s in id_map and t in id_map and id_map[s] < N and id_map[t] < N
        for s, t in zip(src, tgt)
    ])
    new_src = np.array([id_map[s] for s, v in zip(src, valid) if v])
    new_tgt = np.array([id_map[t] for t, v in zip(tgt, valid) if v])
    new_wt = wt[valid]
    print(f"  Remapped {len(all_ids)} IDs → [0,{N-1}] ({valid.sum()} edges)")
    return new_src, new_tgt, new_wt


# ─── Normalization ───────────────────────────────────────────────

def _normalize(X: np.ndarray) -> np.ndarray:
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = np.isnan(col)
        if nans.any():
            med = np.nanmedian(col)
            X[nans, j] = med if not np.isnan(med) else 0.0

    lo = X.min(axis=0, keepdims=True)
    hi = X.max(axis=0, keepdims=True)
    return ((X - lo) / (hi - lo + 1e-8)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Synthetic fallback
# ═══════════════════════════════════════════════════════════════════════

def _load_synthetic(cfg: Config) -> Dict:
    rng = np.random.RandomState(cfg.seed)
    N = cfg.num_nodes

    G = nx.barabasi_albert_graph(N, cfg.ba_edges_per_node, seed=cfg.seed)
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        w = float(rng.lognormal(2.0, 1.0))
        r = rng.random()
        if r < 0.5:
            rows.append(u); cols.append(v); vals.append(w)
        elif r < 0.8:
            rows.append(v); cols.append(u); vals.append(w)
        else:
            rows.append(u); cols.append(v); vals.append(w)
            rows.append(v); cols.append(u); vals.append(w * rng.uniform(0.3, 0.7))

    rows, cols = np.array(rows), np.array(cols)
    vals = np.array(vals, dtype=np.float32)
    vals_norm = vals / (vals.max() + 1e-8)

    X = rng.randn(N, cfg.synth_feature_dim).astype(np.float32)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    z = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
    ratings = np.digitize(z, bins=np.percentile(z, [25, 50, 75]))

    # ── Symmetric Graph Normalization ────────────────────────────
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    vals_t = torch.tensor(vals_norm, dtype=torch.float32)

    # Compute degree based on incoming edges (cols)
    deg = torch.bincount(cols_t, minlength=N).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # Apply symmetric normalization
    norm_vals = deg_inv_sqrt[rows_t] * vals_t * deg_inv_sqrt[cols_t]

    # ── Build PyG + sparse ───────────────────────────────────────
    pyg = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.stack([rows_t, cols_t]),
        edge_weight=norm_vals,
        num_nodes=N,
    )

    adj = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    net_feats = _compute_network_features(rows, cols, vals_norm, N)
    cfg.node_feature_dim = cfg.synth_feature_dim

    return {
        "pyg_data": pyg, "adj_sparse": adj,
        "labels": ratings,
        "srisk_ratio": np.zeros(N, dtype=np.float32),
        "srisk_value": np.zeros(N, dtype=np.float32),
        "node_features_raw": X,
        "network_features": net_feats,
        "metadata": {
            "num_nodes": N, "num_edges": len(rows),
            "avg_degree": len(rows) / N,
            "data_source": "synthetic", "quarter": "N/A",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Network features
# ═══════════════════════════════════════════════════════════════════════

def _compute_network_features(src, tgt, weights, N: int) -> np.ndarray:
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for s, t, w in zip(src, tgt, weights):
        G.add_edge(int(s), int(t), weight=float(w))

    # PageRank: increase iterations, handle convergence failure
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=500, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        print("  PageRank did not converge, using degree-based fallback")
        dg_sum = sum(dict(G.degree()).values()) or 1
        pr = {i: G.degree(i) / dg_sum for i in range(N)}

    Gu = G.to_undirected()
    bc = nx.betweenness_centrality(Gu, k=min(200, N))
    cc = nx.clustering(Gu)
    dg = dict(G.degree())

    feats = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        feats[i] = [pr.get(i, 0), bc.get(i, 0), cc.get(i, 0), dg.get(i, 0)]

    for j in range(4):
        col = feats[:, j]
        lo, hi = col.min(), col.max()
        feats[:, j] = (col - lo) / (hi - lo + 1e-8)

    return feats