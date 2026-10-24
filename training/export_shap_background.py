#!/usr/bin/env python3
"""
Export the SHAP background set used by the online explainer in disc-api.

Same background as xai_eval.py: 500 flows sampled (random_state=1) from the
training part of the seed-42 split, restricted to the 15 deployed features.

Output: ../gan-ics/out_pipeline/artifacts/shap_background.csv

Usage:
  .venv/bin/python export_shap_background.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from train_eval import TEST_SIZE, load_dataset
from xai_eval import BG_SIZE, SPLIT_SEED


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=here / "merged_network_flow_leayer.csv")
    ap.add_argument("--features", type=Path, default=here / "artifacts" / "features_three.json")
    ap.add_argument("--out", type=Path,
                    default=here.parent / "gan-ics" / "out_pipeline" / "artifacts" / "shap_background.csv")
    args = ap.parse_args()

    feats = json.loads(args.features.read_text(encoding="utf-8"))
    X, y = load_dataset(args.data)
    Xtr, _, _, _ = train_test_split(X[feats], y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    bg = Xtr.sample(n=BG_SIZE, random_state=1)
    bg.to_csv(args.out, index=False)
    print(f"background: {bg.shape} -> {args.out}")


if __name__ == "__main__":
    main()
