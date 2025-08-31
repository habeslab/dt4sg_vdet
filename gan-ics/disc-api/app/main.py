from __future__ import annotations
"""
Discriminatore come servizio (disc-api)

Endpoint:
  - GET  /healthz   → stato + soglie + ordine feature
  - POST /predict3  → predizione 3-classi con policy operativa

Policy:
  1) Softmax → p0 (benign), p1 (malicious), p2 (synthetic/OOD)
  2) Se p2 ≥ τ_fake ⇒ origin=synthetic, label=synthetic
  3) Altrimenti origin=real, label=malicious se p_mal_2c ≥ τ_mal, altrimenti benign

Ritorna anche: model_label (argmax puro del modello), features_order, thresholds.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# TF/Keras + caricamento modello
import tensorflow as tf

# --- compat: decorator per registrare custom layer su diverse versioni TF/Keras ---
try:
    # TF2.x classico
    from tensorflow.keras.utils import register_keras_serializable # type: ignore
except Exception:
    try:
        # Alcune build Keras standalone
        from keras.utils import register_keras_serializable
    except Exception:
        # Fallback no-op (se il modello non usa davvero il layer custom)
        def register_keras_serializable(**kwargs):
            def deco(cls):
                return cls
            return deco

# scaler: verrà caricato con joblib/pickle; serve scikit-learn installato
try:
    import joblib
    _USE_JOBLIB = True
except Exception:
    import pickle
    _USE_JOBLIB = False

# opzionale per auto-tune (caricato on-demand)
try:
    import pandas as pd  
except Exception:
    pd = None

# =========================
# Config da ENV
# =========================
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/artifacts")
MODEL_PATH    = os.getenv("MODEL_PATH", "")           # se vuoto → glob in ARTIFACTS_DIR
SCALER_PATH   = os.getenv("SCALER_PATH", f"{ARTIFACTS_DIR}/scaler.pkl")
FEATS_PATH    = os.getenv("FEATS_PATH",  f"{ARTIFACTS_DIR}/features.json")
VAL_CSV       = os.getenv("VAL_CSV", "/out_pipeline/gan_val_raw.csv")

TAU_FAKE_ENV = float(os.getenv("TAU_FAKE", "0.9"))
TAU_MAL_ENV  = float(os.getenv("TAU_MAL",  "0.7"))
AUTO_TUNE    = os.getenv("AUTO_TUNE", "0") not in ("0", "false", "False", "")

# =========================
# Keras custom layer (se modello la usa)
# =========================
@register_keras_serializable(package="discriminator")
class MinibatchStdDev(tf.keras.layers.Layer):
    """Aggiunge 1 feature: std media del batch (inference-safe)."""
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)       # [B, F]
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=0, keepdims=True)
        std  = tf.sqrt(var + self.epsilon)    # [1, F]
        s    = tf.reduce_mean(std, axis=1, keepdims=True)  # [1,1]
        s    = tf.tile(s, [tf.shape(x)[0], 1])             # [B,1]
        return tf.concat([x, s], axis=1)      # [B, F+1]
    def get_config(self):
        return {"epsilon": self.epsilon, **super().get_config()}

def _robust_load_model(path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        return tf.keras.models.load_model(path, compile=False,
                                          custom_objects={"MinibatchStdDev": MinibatchStdDev})

# =========================
# Util: features.json loader (supporta due formati)
# =========================
def _load_features_order(p: str) -> List[str]:
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    feats = data.get("features", [])
    if not isinstance(feats, list) or not feats:
        raise RuntimeError("features.json invalido o vuoto")
    return feats

# =========================
# Util: probabilità robuste
# =========================
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / (s if s > 0 else 1.0)

def _ensure_probs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size != 3:
        raise RuntimeError(f"Output modello non a 3 classi: shape={y.shape}")
    # se non somma ~1 o ha valori fuori [0,1] → softmax
    if (y < 0).any() or (y > 1).any() or not (0.99 <= y.sum() <= 1.01):
        y = _softmax(y)
    y = np.clip(y, 0.0, 1.0)
    y = y / max(1e-12, y.sum())
    return y

# =========================
# Stato globale
# =========================
STATE: Dict[str, object] = {}

def _glob_first_model(dirpath: str) -> str:
    for pat in ("discriminator_*.keras", "*.keras", "*.h5"):
        found = sorted(Path(dirpath).glob(pat))
        if found:
            return str(found[-1])
    raise RuntimeError("Modello non trovato: specifica MODEL_PATH o deposita un .keras/.h5 in /artifacts")

def _load_state() -> None:
    model_path = MODEL_PATH or _glob_first_model(ARTIFACTS_DIR)
    feats_order = _load_features_order(FEATS_PATH)
    # scaler
    if not Path(SCALER_PATH).exists():
        raise RuntimeError(f"Scaler mancante: {SCALER_PATH}")
    if _USE_JOBLIB:
        scaler = joblib.load(SCALER_PATH)
    else:
        import pickle
        scaler = pickle.load(open(SCALER_PATH, "rb"))
    # modello
    model = _robust_load_model(model_path)

    # soglie iniziali
    tau_fake, tau_mal, source = TAU_FAKE_ENV, TAU_MAL_ENV, "env"
    if AUTO_TUNE and Path(VAL_CSV).exists() and pd is not None:
        try:
            tau_fake, tau_mal = _auto_tune_thresholds(model, scaler, feats_order, VAL_CSV)
            source = "auto_tune"
        except Exception as e:
            print(f"[disc-api][WARN] auto_tune fallito ({e}); uso ENV.", flush=True)

    STATE.update({
        "model": model,
        "scaler": scaler,
        "features_order": feats_order,
        "thresholds": {"tau_fake": float(tau_fake), "tau_mal": float(tau_mal), "source": source},
        "model_version": Path(model_path).name
    })

def _auto_tune_thresholds(model: tf.keras.Model, scaler, feats: List[str], val_csv: str) -> Tuple[float, float]:
    """Calcola soglie su probabilità normalizzate."""
    df = pd.read_csv(val_csv)  # type: ignore[arg-type]
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy() if "label" in df.columns else None

    Xs = scaler.transform(X)
    raw = model.predict(Xs, verbose=0)
    raw = np.asarray(raw)
    # normalizza riga per riga
    probs = np.apply_along_axis(_ensure_probs, 1, raw)
    p2 = probs[:, 2]
    p0, p1 = probs[:, 0], probs[:, 1]
    denom = (p0 + p1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mal_2c = np.where(denom > 0, p1 / denom, 0.0)

    # quantili prudenziali
    q_fake = float(np.quantile(p2, 0.95))
    q_mal  = float(np.quantile(p_mal_2c, 0.95 if y is None else 0.90))
    q_fake = float(min(1.0, max(0.5, q_fake)))
    q_mal  = float(min(1.0, max(0.5, q_mal)))
    return q_fake, q_mal

# =========================
# FastAPI
# =========================
app = FastAPI(title="disc-api", version="1.0.0")

@app.on_event("startup")
def _startup():
    _load_state()
    print(f"[disc-api] model={STATE['model_version']} feats={STATE['features_order']} "
          f"thr={STATE['thresholds']}", flush=True)

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "ts": time.time(),
        "features_order": STATE["features_order"],
        "thresholds": STATE["thresholds"],
        "model_version": STATE["model_version"],
    }

@app.post("/predict3")
def predict3(payload: Dict[str, object]):
    feats: Dict[str, float] = (payload or {}).get("features") or {}  # type: ignore[assignment]
    meta: Dict[str, object]  = (payload or {}).get("meta") or {}      

    feats_order: List[str] = STATE["features_order"]  # type: ignore[assignment]
    missing = [f for f in feats_order if f not in feats]
    if missing:
        raise HTTPException(status_code=422, detail={
            "error": "missing_features", "missing": missing, "required_order": feats_order
        })

    x = np.array([[feats[f] for f in feats_order]], dtype=np.float32)
    try:
        Xs = STATE["scaler"].transform(x)  # type: ignore[attr-defined]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore scaler.transform: {e}")

    try:
        raw = STATE["model"].predict(Xs, verbose=0)[0]  # type: ignore[attr-defined]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore MODEL.predict: {e}")

    probs = _ensure_probs(raw)
    p0, p1, p2 = map(float, probs.tolist())
    denom = (p0 + p1)
    p_mal_2c = float(p1 / denom) if denom > 0 else 0.0

    model_idx = int(np.argmax(probs))
    model_label = ["benign", "malicious", "synthetic"][model_idx]

    tau_fake = float(STATE["thresholds"]["tau_fake"])  # type: ignore[index]
    tau_mal  = float(STATE["thresholds"]["tau_mal"])   # type: ignore[index]

    if p2 >= tau_fake:
        origin = "synthetic"
        label  = "synthetic"
    else:
        origin = "real"
        label  = "malicious" if p_mal_2c >= tau_mal else "benign"

    return JSONResponse({
        "ts": time.time(),
        "features_order": feats_order,
        "p0": p0, "p1": p1, "p2": p2,
        "p_mal_2c": p_mal_2c,
        "thresholds": {"tau_fake": tau_fake, "tau_mal": tau_mal, "source": STATE["thresholds"]["source"]},  # type: ignore[index]
        "origin": origin,
        "label": label,             
        "model_label": model_label,   
        "model_version": STATE["model_version"],
        "meta": meta,
    })
