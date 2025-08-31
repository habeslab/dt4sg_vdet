from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Tuple, Optional, List, Union
import numpy as np
import tensorflow as tf

# =============== PATHS ===============
def project_dirs() -> Tuple[Path, Path, Path]:
    """Ritorna (ROOT, ARTIFACTS, PROCESSED) partendo da questo file (src/)."""
    here = Path(__file__).resolve().parent
    root = here.parent
    art  = root / "artifacts"
    proc = root / "processed"
    art.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    return root, art, proc

# =============== MODELLI ===============
def robust_load_model(path: Union[str, Path],
                      custom_objects: Optional[dict]=None,
                      compile: bool=False) -> tf.keras.Model:
    """Carica un Keras model gestendo custom_objects e compile flag."""
    path = str(path)
    custom_objects = custom_objects or {}
    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=compile)

def logits_to_proba(logits: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
    """Converte logit -> probabilità (softmax)."""
    if isinstance(logits, tf.Tensor):
        logits = logits.numpy()
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

# =============== CSV HELPERS ===============
def autodetect_label_column(columns: List[str]) -> str:
    """Heuristica robusta per trovare la colonna label."""
    candidates = ["label", "y", "class", "target", "is_malicious", "malicious"]
    for c in candidates:
        if c in columns:
            return c
    # fallback: ultima colonna
    return columns[-1]

def autodetect_csv(proc: Path) -> Optional[Path]:
    """Preferisci val, poi train."""
    for name in ("gan_val_raw.csv", "gan_train_raw.csv"):
        p = proc / name
        if p.exists():
            return p
    return None

# =============== SERIALIZZAZIONE FEATURE LIST ===============
def save_feature_names(names: List[str], out_path: Path):
    out_path.write_text(json.dumps({"features": names}, indent=2), encoding="utf-8")

def load_feature_names(in_path: Path) -> List[str]:
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    return data.get("features", [])
