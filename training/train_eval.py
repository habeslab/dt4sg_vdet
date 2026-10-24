#!/usr/bin/env python3
"""
Reproducible model comparison for the IEC 60870-5-104 detector (Table I).

Same data protocol as MainProject.ipynb (class filter, 3 macro-classes,
identifier columns dropped, stratified 80/20 split), extended with:
  - grid search with stratified k-fold CV on the training split only
    (scaler fitted inside each fold via Pipeline), scoring = macro F1;
  - repetition over several random seeds (split, CV folds and models);
  - the deployed detector (RF on the 15 SHAP-selected raw features).

Outputs (in --out):
  runs.csv           one row per (seed, model): test metrics + best params
  summary.csv        mean/std over seeds
  table_models.tex   LaTeX table (mean ± std)
  best_params.json   selected configuration per seed and model
  confusion.json     test confusion matrices per seed and model

Environment: see requirements.txt (Python 3.10, XGBoost on CUDA).

Usage:
  .venv/bin/python train_eval.py --seeds 0 1 2 3 4 5 6 7 8 9
  .venv/bin/python train_eval.py --xgb-device cpu      # without GPU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# --- data protocol (identical to MainProject.ipynb) ---------------------------
LABEL_COL = "Label"
ALLOWED = [
    "NORMAL",
    "c_rp_na_1", "c_sc_na_1", "c_se_na_1", "c_rd_na_1",
    "c_ci_na_1_DoS", "c_sc_na_1_DoS", "c_se_na_1_DoS",
]
DROP_IF_PRESENT = {
    "timestamp", "time", "date", "datetime", "ts", "flow_start", "flow_end",
    "src_ip", "dst_ip", "source_ip", "destination_ip",
    "src_port", "dst port", "source_port", "destination_port",
    "sport", "dport", "proto", "protocol", "flow_id", "id", "uid", "src port",
}
CLASS_NAMES = ["Normal", "Availability", "Integrity"]
TEST_SIZE = 0.2


def to_numeric_label(val: str) -> int:
    """0 = Normal, 1 = Availability (DoS), 2 = Integrity."""
    v = val.lower()
    if v.endswith("dos"):
        return 1
    if v == "normal":
        return 0
    return 2


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False).dropna(axis=1, how="all")
    df = df[df[LABEL_COL].isin(ALLOWED)]
    y = df[LABEL_COL].apply(to_numeric_label).astype(np.int32)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in numeric_cols if c != LABEL_COL and c.lower() not in DROP_IF_PRESENT]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X, y


# --- models and search spaces (original configurations are included) ---------
def model_specs(seed: int, deployed_feats: list[str], xgb_device: str) -> dict[str, dict]:
    return {
        "SVM": {
            "pipe": Pipeline([("scaler", StandardScaler()),
                              ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
            "grid": {"clf__C": [0.1, 1, 10, 100],
                     "clf__gamma": ["scale", 0.01, 0.1]},
        },
        "XGBoost": {
            "pipe": Pipeline([("scaler", StandardScaler()),
                              ("clf", XGBClassifier(objective="multi:softprob", num_class=3,
                                                    eval_metric="mlogloss", tree_method="hist",
                                                    device=xgb_device,
                                                    subsample=0.9, colsample_bytree=0.9,
                                                    random_state=seed, n_jobs=1))]),
            "grid": {"clf__n_estimators": [100, 300],
                     "clf__learning_rate": [0.05, 0.1, 0.3],
                     "clf__max_depth": [6, 12, 24]},
            # few concurrent fits on a single (shared) GPU
            "n_jobs": 4 if xgb_device == "cuda" else None,
        },
        "Decision Tree": {
            "pipe": Pipeline([("scaler", StandardScaler()),
                              ("clf", DecisionTreeClassifier(random_state=seed))]),
            "grid": {"clf__criterion": ["gini", "entropy"],
                     "clf__max_depth": [2, 4, 8, 16, None],
                     "clf__class_weight": [None, "balanced"]},
        },
        "Random Forest": {
            "pipe": Pipeline([("scaler", StandardScaler()),
                              ("clf", RandomForestClassifier(random_state=seed, n_jobs=1))]),
            "grid": {"clf__n_estimators": [100, 300, 500],
                     "clf__max_depth": [12, 24, None],
                     "clf__min_samples_leaf": [1, 2, 4]},
        },
        # deployed detector: RF on the 15 SHAP-selected raw (unscaled) features
        "Random Forest (top-15, deployed)": {
            "pipe": Pipeline([("clf", RandomForestClassifier(random_state=seed, n_jobs=1))]),
            "grid": {"clf__n_estimators": [100, 300, 500],
                     "clf__max_depth": [12, 24, None],
                     "clf__min_samples_leaf": [1, 2, 4]},
            "features": deployed_feats,
        },
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here / "merged_network_flow_leayer.csv")
    ap.add_argument("--features", type=Path, default=here / "artifacts" / "features_three.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--xgb-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out", type=Path, default=here / "results_gridsearch")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    deployed_feats = json.loads(args.features.read_text(encoding="utf-8"))
    print(f"dataset: {len(X)} samples, {X.shape[1]} features, classes={np.bincount(y).tolist()}")

    rows, best_params, confusions = [], {}, {}
    for seed in args.seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=seed)
        for name, spec in model_specs(seed, deployed_feats, args.xgb_device).items():
            if args.models and name not in args.models:
                continue
            cols = spec.get("features", list(X.columns))
            gs = GridSearchCV(spec["pipe"], spec["grid"], scoring="f1_macro", cv=cv,
                              n_jobs=spec.get("n_jobs") or args.n_jobs, refit=True)
            t0 = time.time()
            gs.fit(Xtr[cols], ytr)
            elapsed = time.time() - t0
            y_pred = gs.predict(Xte[cols])
            params = {k.removeprefix("clf__"): v for k, v in gs.best_params_.items()}
            m = evaluate(yte, y_pred)
            rows.append({"seed": seed, "model": name, "n_features": len(cols),
                         "cv_f1": gs.best_score_, **m, "search_s": round(elapsed, 1),
                         "best_params": json.dumps(params)})
            best_params.setdefault(name, {})[seed] = params
            confusions.setdefault(name, {})[seed] = confusion_matrix(yte, y_pred, labels=[0, 1, 2]).tolist()
            print(f"[seed {seed}] {name:34s} acc={m['accuracy']:.3f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f} cvF1={gs.best_score_:.3f} "
                  f"({elapsed:.0f}s) {params}", flush=True)
            # checkpoint after every model so partial runs are usable
            pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)

    runs = pd.DataFrame(rows)
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = runs.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary.to_csv(args.out / "summary.csv")
    (args.out / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    (args.out / "confusion.json").write_text(json.dumps(confusions, indent=2))

    n = runs["seed"].nunique()
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\",
             r"\midrule"]
    for model, r in summary.iterrows():
        cells = [f"{r[(k, 'mean')]:.3f} $\\pm$ {r[(k, 'std')]:.3f}" for k in metrics]
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% mean $\\pm$ std over {n} seeds, {args.cv}-fold CV grid search, macro-averaged metrics"]
    (args.out / "table_models.tex").write_text("\n".join(lines) + "\n")
    print("\n" + summary.round(3).to_string())


if __name__ == "__main__":
    main()
