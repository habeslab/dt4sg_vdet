#!/usr/bin/env python3
"""
One-vs-rest ROC curves of the deployed detector (paper Fig. 7).

Model:  artifacts/rf_model_three.pkl (RF on the 15 SHAP-selected raw features)
Data:   seed-42 stratified 80/20 split (same as Fig. 6 and xai_eval.py)
Labels: 0 = Normal, 1 = Availability, 2 = Integrity

Outputs (in --out): roc_curves.{png,pdf}, roc_auc.json

Usage:
  .venv/bin/python roc_eval.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from train_eval import CLASS_NAMES, TEST_SIZE, load_dataset

SPLIT_SEED = 42


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here / "merged_network_flow_leayer.csv")
    ap.add_argument("--model", type=Path, default=here / "artifacts" / "rf_model_three.pkl")
    ap.add_argument("--features", type=Path, default=here / "artifacts" / "features_three.json")
    ap.add_argument("--out", type=Path, default=here / "results_roc")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    feats = json.loads(args.features.read_text(encoding="utf-8"))
    X, y = load_dataset(args.data)
    _, Xte, _, yte = train_test_split(X[feats], y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    model = joblib.load(args.model)
    proba = model.predict_proba(Xte)

    aucs = {}
    plt.figure(figsize=(6, 4.5))
    for c, name in enumerate(CLASS_NAMES):
        col = list(model.classes_).index(c)
        fpr, tpr, _ = roc_curve(yte == c, proba[:, col])
        aucs[name] = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} ({c}), AUC = {aucs[name]:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(args.out / f"roc_curves.{ext}", dpi=300)
    (args.out / "roc_auc.json").write_text(json.dumps(aucs, indent=2))
    print(json.dumps(aucs, indent=2))


if __name__ == "__main__":
    main()
