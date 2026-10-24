"""
Explanation phase (paper Sec. III-D3): asynchronous SHAP attributions for
flows classified as malicious, so that the fast-path alert is never delayed.

- Same explainer configuration evaluated offline (training/xai_eval.py):
  shap.TreeExplainer, interventional, background of 500 training flows
  (artifacts/shap_background.csv, exported by training/export_shap_background.py).
- Jobs run on a dedicated thread pool; results are kept in a bounded store and
  served by GET /explain/{alert_id}; each explanation is appended to a JSONL log
  with its timestamps (prediction, start, end) for latency analysis.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("disc-api.xai")


class AsyncExplainer:
    def __init__(
        self,
        model,
        features_order: List[str],
        class_index: Dict[str, int],
        background_path: str,
        top_k: int = 5,
        workers: int = 1,
        max_stored: int = 2000,
        log_path: Optional[str] = None,
    ):
        import shap  # imported here so that the API still starts if XAI is disabled

        self.features_order = features_order
        self.class_index = class_index
        self.top_k = top_k
        self.max_stored = max_stored
        self.log_path = Path(log_path) if log_path else None

        if Path(background_path).exists():
            bg = pd.read_csv(background_path)[features_order]
            self.explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
            self.mode = f"interventional (background={len(bg)})"
        else:
            logger.warning("SHAP background %s not found: falling back to tree_path_dependent", background_path)
            self.explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            self.mode = "tree_path_dependent"

        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="xai")
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------
    def warmup(self, x_df: pd.DataFrame) -> float:
        t0 = time.perf_counter()
        self._shap_row(x_df)
        return (time.perf_counter() - t0) * 1000.0

    def _shap_row(self, x_df: pd.DataFrame) -> np.ndarray:
        """SHAP values of a single row, shape (n_features, n_classes)."""
        sv = self.explainer.shap_values(x_df, check_additivity=False)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        return np.asarray(sv)[0]

    def explain(self, x_df: pd.DataFrame, pred_class: str) -> Dict[str, object]:
        """Synchronous explanation of the predicted class (canonical label "0"/"1"/"2")."""
        t0 = time.perf_counter()
        phi = self._shap_row(x_df)[:, self.class_index[pred_class]]
        compute_ms = (time.perf_counter() - t0) * 1000.0

        base = np.asarray(self.explainer.expected_value).ravel()
        base_value = float(base[self.class_index[pred_class]]) if base.size > 1 else float(base[0])
        values = x_df.iloc[0].to_numpy(dtype=float)
        order = np.argsort(-np.abs(phi))
        return {
            "class": pred_class,
            "base_value": base_value,
            "top": [
                {"feature": self.features_order[i], "value": float(values[i]), "shap": float(phi[i])}
                for i in order[: self.top_k]
            ],
            "shap": {f: float(v) for f, v in zip(self.features_order, phi)},
            "compute_ms": round(compute_ms, 2),
        }

    # ------------------------------------------------------------------
    def submit(self, alert_id: str, x_df: pd.DataFrame, pred_class: str,
               t_pred: float, meta: Dict[str, object]) -> None:
        self._put(alert_id, {"alert_id": alert_id, "status": "pending", "t_pred": t_pred})
        self._pool.submit(self._run, alert_id, x_df.copy(), pred_class, t_pred, meta)

    def _run(self, alert_id: str, x_df: pd.DataFrame, pred_class: str,
             t_pred: float, meta: Dict[str, object]) -> None:
        t_start = time.time()
        try:
            chi = self.explain(x_df, pred_class)
            t_end = time.time()
            res = {
                "alert_id": alert_id,
                "status": "done",
                "flow_id": meta.get("flow_id"),
                "chi": chi,
                "timestamps": {"t_pred": t_pred, "t_xai_start": t_start, "t_xai_end": t_end},
                "latency_ms": {
                    "queue": round((t_start - t_pred) * 1000.0, 2),
                    "compute": chi["compute_ms"],
                    "pred_to_explanation": round((t_end - t_pred) * 1000.0, 2),
                },
            }
        except Exception as e:  # never break the API because of the explainer
            logger.exception("SHAP explanation failed for %s", alert_id)
            res = {"alert_id": alert_id, "status": "error", "error": str(e)}
        self._put(alert_id, res)
        self._log(res)

    def _put(self, alert_id: str, item: Dict[str, object]) -> None:
        with self._lock:
            self._store[alert_id] = item
            self._store.move_to_end(alert_id)
            while len(self._store) > self.max_stored:
                self._store.popitem(last=False)

    def get(self, alert_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            return self._store.get(alert_id)

    def _log(self, res: Dict[str, object]) -> None:
        if not self.log_path:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append explanation to %s", self.log_path)
