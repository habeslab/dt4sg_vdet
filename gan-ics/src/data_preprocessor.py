# data_preprocessor.py
from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import project_dirs, autodetect_label_column, save_feature_names

@dataclass
class DataPreprocessor:
    """
    Carica un CSV "merged", fa:
    - individuazione colonna label (heuristica)
    - oversampling della/e classe/i minoritarie (tutte le classi reali presenti)
    - split stratificato train/val
    - fit scaler su train e salvataggio scaler.pkl e features.json
    - salva gli split RAW (non scalati) per training/eval successivi
    """
    csv_path: str
    val_size: float = 0.2
    random_state: int = 42
    save_artifacts: bool = True
    save_raw_splits: bool = True

    def __post_init__(self):
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.root, self.art, self.proc = project_dirs()

    # --------- utils interni ----------
    def _detect_label_and_features(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        label_col = autodetect_label_column(df.columns.tolist())
        feat_cols = [c for c in df.columns if c != label_col]
        # Porta la label a int32
        df[label_col] = df[label_col].astype(int)
        return label_col, feat_cols

    def _oversample_minority(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Random oversampling fino a eguagliare la classe più numerosa."""
        counts = df[label_col].value_counts().to_dict()
        max_n = max(counts.values())
        parts = []
        rng = np.random.default_rng(self.random_state)
        for cls, n in counts.items():
            df_c = df[df[label_col] == cls]
            if n == max_n:
                parts.append(df_c)
            else:
                idx = rng.choice(df_c.index.values, size=max_n - n, replace=True)
                parts.append(pd.concat([df_c, df.loc[idx]], ignore_index=False))
        out = pd.concat(parts).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return out

    def _train_val_split(self, df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified split semplice senza sklearn.model_selection."""
        # per classe, split a mano
        rng = np.random.default_rng(self.random_state)
        train_parts, val_parts = [], []
        for cls, grp in df.groupby(label_col):
            idx = grp.index.values
            rng.shuffle(idx)
            n_val = int(np.floor(len(idx) * self.val_size))
            val_idx = idx[:n_val]
            tr_idx  = idx[n_val:]
            val_parts.append(df.loc[val_idx])
            train_parts.append(df.loc[tr_idx])
        tr = pd.concat(train_parts).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        va = pd.concat(val_parts).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return tr, va

    # --------- pipeline pubblica ----------
    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        df = pd.read_csv(self.csv_path)
        label_col, feat_cols = self._detect_label_and_features(df)

        # Oversampling
        df_bal = self._oversample_minority(df, label_col)

        # Split
        df_tr, df_va = self._train_val_split(df_bal, label_col)

        # Salva RAW split se richiesto
        if self.save_raw_splits:
            (self.proc / "gan_train_raw.csv").parent.mkdir(parents=True, exist_ok=True)
            df_tr.to_csv(self.proc / "gan_train_raw.csv", index=False)
            df_va.to_csv(self.proc / "gan_val_raw.csv", index=False)

        X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr[label_col].to_numpy(dtype=np.int32)
        X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
        y_va = df_va[label_col].to_numpy(dtype=np.int32)

        # Scaler su TRAIN
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)

        if self.save_artifacts:
            with open(self.art / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            save_feature_names(feat_cols, self.art / "features.json")

        return X_tr_s, y_tr, X_va_s, y_va, feat_cols
