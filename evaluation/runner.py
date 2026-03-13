
"""
Evaluation runner — executes all baselines (B1–B6) + ablations on test scenarios.
"""

import numpy as np
import torch
import xgboost as xgb
from typing import Dict, List, Tuple, Optional

from config import Config
from .metrics import classification_metrics, cascade_metrics


class EvaluationRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_all(
        self,
        node_features: np.ndarray,
        network_features: np.ndarray,
        cascadenet_model,
        b3_model,
        b4_model,
        data,
        sc_train: List[Dict],
        sc_test: List[Dict],
        # ── FIX: Додаткові моделі для абляцій ───────────────────
        cascadenet_pd_only: Optional[object] = None,
        cascadenet_stop_grad: Optional[object] = None,
    ) -> Tuple[List[Dict], Dict]:
        dev = self.cfg.device
        N = node_features.shape[0]
        results = []

        X_train, y_train, X_test, y_test = self._prepare_xgb_data(
            node_features, network_features, sc_train, sc_test,
        )

        # ── B1: XGBoost (node features only) ────────────────────
        results.append(self._run_xgb(
            X_train["nodes"], y_train,
            X_test["nodes"], y_test,
            "B1: XGBoost (nodes)",
        ))

        # ── B2: XGBoost + network features ──────────────────────
        results.append(self._run_xgb(
            X_train["nodes_net"], y_train,
            X_test["nodes_net"], y_test,
            "B2: XGBoost + NetFeats",
        ))

        # ── B3: VanillaGNN ──────────────────────────────────────
        b3_probs = self._predict_gnn(b3_model, data, sc_test, dev)
        results.append(self._eval_probs(b3_probs, y_test, "B3: VanillaGNN"))

        # ── B4: StressSkipGNN ───────────────────────────────────
        b4_probs = self._predict_gnn(b4_model, data, sc_test, dev)
        results.append(self._eval_probs(b4_probs, y_test, "B4: StressSkipGNN"))

        # ── B5: CascadeNet (full) ───────────────────────────────
        b5_probs, cascade_preds = self._predict_cascadenet(
            cascadenet_model, data, sc_test, dev,
        )
        results.append(self._eval_probs(b5_probs, y_test, "B5: CascadeNet"))

        # ── B5a: CascadeNet PD-only (абляція) ───────────────────
        if cascadenet_pd_only is not None:
            b5a_probs, _ = self._predict_cascadenet(
                cascadenet_pd_only, data, sc_test, dev,
            )
            results.append(self._eval_probs(
                b5a_probs, y_test, "B5a: CascadeNet PD-only"
            ))

        # ── B5b: CascadeNet stop-grad (абляція) ─────────────────
        if cascadenet_stop_grad is not None:
            b5b_probs, cascade_preds_sg = self._predict_cascadenet(
                cascadenet_stop_grad, data, sc_test, dev,
            )
            results.append(self._eval_probs(
                b5b_probs, y_test, "B5b: CascadeNet StopGrad"
            ))

        # ── B6: GNN Embeddings + XGBoost ────────────────────────
        embs_train = self._extract_embeddings(cascadenet_model, data, sc_train, dev)
        embs_test = self._extract_embeddings(cascadenet_model, data, sc_test, dev)

        X_b6_train = np.hstack([X_train["nodes_net"], embs_train])
        X_b6_test = np.hstack([X_test["nodes_net"], embs_test])
        results.append(self._run_xgb(
            X_b6_train, y_train, X_b6_test, y_test,
            "B6: GNN Emb + XGBoost",
        ))

        # ── Cascade metrics (B5 only) ───────────────────────────
        cas_true = np.concatenate([s["cascade_sizes"] for s in sc_test])
        cas_met = cascade_metrics(cas_true, cascade_preds)

        self._print_results(results, cas_met)
        return results, cas_met, cascade_preds

    # ═══════════════════════════════════════════════════════════════
    # XGBoost helpers
    # ═══════════════════════════════════════════════════════════════

    def _prepare_xgb_data(self, node_features, net_features, sc_train, sc_test):
        N = node_features.shape[0]

        def _tile(scenarios):
            n_sc = len(scenarios)
            X_nodes = np.tile(node_features, (n_sc, 1))
            X_net = np.tile(net_features, (n_sc, 1))
            X_stress = np.concatenate([
                np.tile(s["stress"], (N, 1)) for s in scenarios
            ])
            y = np.concatenate([s["labels"] for s in scenarios])
            return {
                "nodes": np.hstack([X_nodes, X_stress]),
                "nodes_net": np.hstack([X_nodes, X_net, X_stress]),
            }, y

        X_train, y_train = _tile(sc_train)
        X_test, y_test = _tile(sc_test)
        return X_train, y_train, X_test, y_test

    def _run_xgb(self, X_tr, y_tr, X_te, y_te, name) -> Dict:
        cfg = self.cfg
        scale = max(1, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        clf = xgb.XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_lr,
            scale_pos_weight=float(scale),
            eval_metric="logloss",
            random_state=cfg.seed,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        metrics = classification_metrics(y_te, probs, cfg.precision_at_k)
        metrics["name"] = name
        return metrics

    # ═══════════════════════════════════════════════════════════════
    # GNN prediction helpers
    # ═══════════════════════════════════════════════════════════════

    def _predict_gnn(self, model, data, scenarios, dev) -> np.ndarray:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for sc in scenarios:
                x = data.x.to(dev)
                ei = data.edge_index.to(dev)
                ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
                s = torch.tensor(sc["stress"], device=dev)
                logit = model(x, ei, ew, s)
                all_probs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(all_probs)

    def _predict_cascadenet(self, model, data, scenarios, dev):
        model.eval()
        all_pd, all_cas = [], []
        with torch.no_grad():
            for sc in scenarios:
                x = data.x.to(dev)
                ei = data.edge_index.to(dev)
                ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
                s = torch.tensor(sc["stress"], device=dev)

                pl, cg, cs = model(x, ei, ew, s)
                all_pd.append(torch.sigmoid(pl).cpu().numpy())

                gate_prob = torch.sigmoid(cg)
                raw_size = torch.expm1(cs).clamp(min=0)
                all_cas.append((gate_prob * raw_size).cpu().numpy())

        return np.concatenate(all_pd), np.concatenate(all_cas)

    def _extract_embeddings(self, model, data, scenarios, dev) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for sc in scenarios:
                x = data.x.to(dev)
                ei = data.edge_index.to(dev)
                ew = data.edge_weight.to(dev) if data.edge_weight is not None else None
                s = torch.tensor(sc["stress"], device=dev)
                emb = model.get_embeddings(x, ei, ew, s)
                all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs)

    def _eval_probs(self, probs, y_true, name) -> Dict:
        metrics = classification_metrics(y_true, probs, self.cfg.precision_at_k)
        metrics["name"] = name
        return metrics

    # ═══════════════════════════════════════════════════════════════
    # Reporting — оновлено для нових абляцій
    # ═══════════════════════════════════════════════════════════════

    def _print_results(self, results, cas_met):
        print("\n" + "=" * 76)
        print("  CascadeNet — Evaluation Results")
        print("=" * 76)

        print(f"\n  {'Model':<35s} {'AUROC':>8s} {'AUPRC':>8s} "
              f"{'Brier':>8s} {'F1':>8s}")
        print(f"  {'─' * 71}")

        for r in results:
            tag = " ◀" if r["name"] == "B5: CascadeNet" else ""
            print(f"  {r['name']:<35s} {r['auroc']:>7.4f} "
                  f"{r['auprc']:>7.4f} {r['brier']:>7.4f} "
                  f"{r['f1']:>7.4f}{tag}")

        print(f"\n  Cascade prediction:")
        print(f"    Spearman ρ:  {cas_met['spearman_rho']:.4f}")
        print(f"    MAE log1p:   {cas_met['mae_log']:.4f}")
        print(f"    MAE raw:     {cas_met['mae_raw']:.1f}")

        b = {r["name"][:3].strip(":"): r for r in results}
        pairs = [
            ("B2", "B1", "Graph value"),
            ("B5", "B2", "GNN value"),
            ("B5", "B3", "FiLM value"),
            ("B5", "B4", "Skip value"),
            ("B6", "B5", "Hybrid value"),
        ]

        # ── FIX: Нові абляції ────────────────────────────────────
        # Шукаємо за повною назвою
        named = {r["name"]: r for r in results}
        ablation_pairs = [
            ("B5a: CascadeNet PD-only", "B3: VanillaGNN",
             "FiLM vs Vanilla (no cascade)"),
            ("B5a: CascadeNet PD-only", "B5: CascadeNet",
             "PD-only vs full (cascade cost)"),
            ("B5b: CascadeNet StopGrad", "B5: CascadeNet",
             "StopGrad vs full"),
        ]

        print(f"\n  Ablation (standard):")
        for hi, lo, label in pairs:
            if hi in b and lo in b:
                d = b[hi]["auroc"] - b[lo]["auroc"]
                verdict = "✓" if d > 0.005 else ("~" if d > 0 else "✗")
                print(f"    {label:<30s} ({hi}-{lo}): {d:>+.4f} AUROC  {verdict}")

        print(f"\n  Ablation (new):")
        for hi_name, lo_name, label in ablation_pairs:
            if hi_name in named and lo_name in named:
                d = named[hi_name]["auroc"] - named[lo_name]["auroc"]
                verdict = "✓" if d > 0.005 else ("~" if d > 0 else "✗")
                print(f"    {label:<35s}: {d:>+.4f} AUROC  {verdict}")

