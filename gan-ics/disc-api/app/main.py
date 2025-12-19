from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# --- loader modello/scaler ---
try:
    import joblib
    _USE_JOBLIB = True
except Exception:
    import pickle
    _USE_JOBLIB = False

# =========================
# Logging
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("disc-api")

# =========================
# Config da ENV
# =========================
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/artifacts")

# RF model path
MODEL_PATH  = os.getenv("MODEL_PATH", f"{ARTIFACTS_DIR}/rf_model_three.pkl")

# features.json: lista ordinata come in training
FEATS_PATH  = os.getenv("FEATS_PATH", f"{ARTIFACTS_DIR}/features_three.json")

# scaler opzionale (se non esiste, viene ignorato)

# mapping class labels (se nel training hai usato stringhe diverse)
# es: "benign,malicious,synthetic"
CLASS_ORDER_ENV = os.getenv("CLASS_ORDER", "0,1,2")

# =========================
# Util: features.json loader
# =========================
def _load_features_order(p: str) -> List[str]:
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    feats = data.get("features", [])
    if not isinstance(feats, list) or not feats:
        raise RuntimeError("features.json invalido o vuoto")
    return feats

def _load_pickle(path: str):
    if _USE_JOBLIB:
        return joblib.load(path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def _as_float(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0

# =========================
# Stato globale
# =========================
STATE: Dict[str, object] = {}

def _startup_load() -> None:
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"MODEL_PATH non trovato: {MODEL_PATH}")
    if not Path(FEATS_PATH).exists():
        raise RuntimeError(f"FEATS_PATH non trovato: {FEATS_PATH}")

    feats_order = _load_features_order(FEATS_PATH)
    model = _load_pickle(MODEL_PATH)


    # classi del modello (sklearn): model.classes_
    # possono essere int o str; noi vogliamo mappare in {benign, malicious, synthetic}
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise RuntimeError("Il modello RF non espone classes_. È un modello sklearn?")

    model_classes = [str(c) for c in list(model_classes)]
    desired = [c.strip() for c in CLASS_ORDER_ENV.split(",") if c.strip()]

    # Creiamo una mappa "label canonica" -> indice nella predict_proba
    # Assunzione: nel training le classi del modello corrispondono ai label (stringhe) o a indici 0/1/2.
    # Se sono 0/1/2 come stringhe, le riconosciamo.
    idx_map: Dict[str, int] = {}
    for canon in desired:
        # match diretto
        if canon in model_classes:
            idx_map[canon] = model_classes.index(canon)
            continue
        # match numerico "0/1/2" se canon è benign/malicious/synthetic
        # (se nel training y era 0/1/2 e classes_ = ["0","1","2"])
        if canon == "benign" and "0" in model_classes:
            idx_map[canon] = model_classes.index("0")
        elif canon == "integrity" and "1" in model_classes:
            idx_map[canon] = model_classes.index("1")
        elif canon == "denial_of_service" and "2" in model_classes:
            idx_map[canon] = model_classes.index("2")

    if set(idx_map.keys()) != set(desired):
        raise RuntimeError(
            f"Impossibile mappare le classi. classes_={model_classes}, CLASS_ORDER={desired}. "
            "Imposta CLASS_ORDER in modo coerente (o allinea le etichette del training)."
        )

    STATE.update({
        "model": model,
        "features_order": feats_order,
        "model_version": Path(MODEL_PATH).name,
        "classes_": model_classes,
        "class_index": idx_map,  # canon -> idx proba
        "class_order": desired,
    })

# =========================
# FastAPI
# =========================
app = FastAPI(title="disc-api (RF)", version="2.0.0")

@app.on_event("startup")
def _startup():
    _startup_load()
    logger.info(
        "Loaded model=%s | feats=%d | classes_=%s | map=%s",
        STATE["model_version"],
        len(STATE["features_order"]),
        STATE["classes_"],
        STATE["class_index"]
    )

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "ts": time.time(),
        "features_order": STATE["features_order"],
        "model_version": STATE["model_version"],
        "classes_": STATE["classes_"],
        "class_index": STATE["class_index"]
    }

@app.post("/predict3")
def predict3(payload: Dict[str, object]):
    feats: Dict[str, object] = (payload or {}).get("features") or {}  # type: ignore[assignment]
    meta: Dict[str, object]  = (payload or {}).get("meta") or {}

    feats_order: List[str] = STATE["features_order"]  # type: ignore[assignment]

    # --- missing features check ---
    missing = [f for f in feats_order if f not in feats]
    if missing:
        raise HTTPException(status_code=422, detail={
            "error": "missing_features",
            "missing": missing[:50],
            "missing_count": len(missing),
            "required_order": feats_order
        })

    # --- build X row ---
    x_df = pd.DataFrame([{f: _as_float(feats[f]) for f in feats_order}], columns=feats_order)
    
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    probs_raw = STATE["model"].predict_proba(x_df)[0]
    


    probs_raw = np.asarray(probs_raw, dtype=np.float64).ravel()

    # --- map proba to canonical p0/p1/p2 ---
    idx = STATE["class_index"]  # type: ignore[assignment]
    p0 = float(probs_raw[idx["0"]])
    p1 = float(probs_raw[idx["1"]])
    p2 = float(probs_raw[idx["2"]])

    if p0 < p1 or p0 < p2:
        # If p1 greater than p0 then we need to check the exact class for the 
        print({
            "ts": time.time(),
            "features_order": feats_order,
            "p0": p0, "p1": p1, "p2": p2,
            "model_version": STATE["model_version"],
            "meta": meta,
        })
    return JSONResponse({
        "ts": time.time(),
        "features_order": feats_order,
        "p0": p0, "p1": p1, "p2": p2,
        "model_version": STATE["model_version"],
        "meta": meta,
    })
