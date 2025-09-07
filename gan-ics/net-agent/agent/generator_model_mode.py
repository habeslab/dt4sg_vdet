from __future__ import annotations
"""
Generator FEATURE-MODE per net-agent.
Espone GeneratorWorker(start/stop).
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List
import json
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None  # permette di avviare l'agent anche senza TF se GEN_ENABLED=0

# scaler
try:
    import pickle
    from sklearn.preprocessing import StandardScaler  # noqa: F401
except Exception:
    pickle = None

# ----------------- ENV -----------------
GEN_ENABLED = os.getenv("GEN_ENABLED", "1") not in ("0","false","False","")
GEN_MODE = os.getenv("GEN_MODE", "feature")

GEN_MODEL_PATH = os.getenv("GEN_MODEL_PATH", "/artifacts/generator.keras")
GEN_LATENT_DIM = int(os.getenv("GEN_LATENT_DIM", "64"))
GEN_BATCH      = int(os.getenv("GEN_BATCH", "32"))
GEN_RATE_PER_MIN = int(os.getenv("GEN_RATE_PER_MIN", "60"))  # quanti esempi/min
GEN_BURST      = int(os.getenv("GEN_BURST", "10"))
GEN_SEED       = os.getenv("GEN_SEED", "")

FEATS_PATH  = os.getenv("FEATS_PATH", "/artifacts/features.json")
SCALER_PATH = os.getenv("SCALER_PATH", "/artifacts/scaler.pkl")

GEN_FORCE_PROTO_TCP = os.getenv("GEN_FORCE_PROTO_TCP", "1") not in ("0","false","False","")
GEN_FORCE_SPORT = os.getenv("GEN_FORCE_SPORT", "2404")  # "" per non forzare
GEN_DPORT_MIN = int(os.getenv("GEN_DPORT_MIN", "49152"))
GEN_DPORT_MAX = int(os.getenv("GEN_DPORT_MAX", "65535"))
GEN_PKT_MIN   = int(os.getenv("GEN_PKT_MIN", "1"))
GEN_BYTES_MIN = int(os.getenv("GEN_BYTES_MIN", "64"))

# ----------------- utils -----------------
def _load_features_order(p: str) -> List[str]:
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    feats = data.get("features", [])
    if not isinstance(feats, list) or not feats:
        raise RuntimeError("features.json invalido o vuoto")
    return feats

def _postprocess_rows(rows: np.ndarray, feats: List[str]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for r in rows:
        d = {name: float(val) for name, val in zip(feats, r.tolist())}
        d["proto"] = 6 if GEN_FORCE_PROTO_TCP else int(max(1, min(255, round(d.get("proto", 6)))))
        d["sport"] = int(GEN_FORCE_SPORT) if GEN_FORCE_SPORT else int(max(1, min(65535, round(d.get("sport", 2404)))))
        d["dport"] = int(max(GEN_DPORT_MIN, min(GEN_DPORT_MAX, round(d.get("dport", GEN_DPORT_MIN)))))
        d["pkt_count"] = int(max(GEN_PKT_MIN, round(d.get("pkt_count", 1))))
        d["bytes_total"] = int(max(GEN_BYTES_MIN, round(d.get("bytes_total", 64))))
        out.append(d)
    return out

# ----------------- core -----------------
class _FeatureGenerator:
    def __init__(self, outq):
        self.outq = outq
        self._thread: threading.Thread | None = None
        self._stop = False

        if GEN_SEED:
            try:
                seed = int(GEN_SEED)
                np.random.seed(seed)
                if tf is not None:
                    tf.random.set_seed(seed)
            except Exception:
                pass

        self.features_order = _load_features_order(FEATS_PATH)

        # scaler opzionale: solo per inverse_transform se il tuo generatore lavora in spazio standardizzato
        self.scaler = None
        if pickle is not None and Path(SCALER_PATH).exists():
            try:
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = None

        # carica generatore se disponibile
        self.G = None
        if tf is not None and Path(GEN_MODEL_PATH).exists():
            try:
                self.G = tf.keras.models.load_model(GEN_MODEL_PATH, compile=False)
            except Exception:
                self.G = None

    def start(self) -> None:
        if not GEN_ENABLED or GEN_MODE != "feature":
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="generator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = 60.0 / max(GEN_RATE_PER_MIN, 1)
        while not self._stop:
            batch = min(GEN_BATCH, GEN_BURST)
            # se non c'è G, genera rumore “dummy” realistico
            if self.G is None:
                X_raw = self._dummy_batch(batch)
            else:
                z = np.random.standard_normal(size=(batch, GEN_LATENT_DIM)).astype(np.float32)
                X_std = self.G.predict(z, verbose=0)
                if self.scaler is not None:
                    X_raw = self.scaler.inverse_transform(X_std).astype(np.float32)
                else:
                    X_raw = X_std

            rows = _postprocess_rows(X_raw, self.features_order)
            ts = time.time()
            for i, f in enumerate(rows):
                flow_id = f"gen|feature|t{int(ts)}|b{batch}|i{i}"
                self.outq.put({
                    "features": f,
                    "meta": {
                        "flow_id": flow_id,
                        "synthetic": True,
                        "mode": "feature",
                        "window_ts": ts,
                        "ts": ts, 
                    }
                    })

            time.sleep(interval)

    def _dummy_batch(self, batch: int) -> np.ndarray:
        # genera valori in range plausibili
        cols = len(self.features_order)
        X = np.zeros((batch, cols), dtype=np.float32)
        names = self.features_order
        for i in range(batch):
            row = []
            for n in names:
                if n == "sport":
                    row.append(float(int(GEN_FORCE_SPORT) if GEN_FORCE_SPORT else np.random.randint(2400, 65000)))
                elif n == "dport":
                    row.append(float(np.random.randint(GEN_DPORT_MIN, GEN_DPORT_MAX)))
                elif n == "proto":
                    row.append(6.0 if GEN_FORCE_PROTO_TCP else float(np.random.randint(1, 255)))
                elif n == "pkt_count":
                    row.append(float(np.random.randint(GEN_PKT_MIN, GEN_PKT_MIN + 50)))
                elif n == "bytes_total":
                    row.append(float(np.random.randint(GEN_BYTES_MIN, GEN_BYTES_MIN + 20000)))
                else:
                    row.append(float(np.random.random()*10))
            X[i, :] = row
        return X

# ----------------- wrapper compatibile con main.py -----------------
class GeneratorWorker:
    """Semplice wrapper che espone start/stop come si aspetta main.py."""
    def __init__(self, outq):
        self._gen = _FeatureGenerator(outq)
    def start(self) -> None:
        self._gen.start()
    def stop(self) -> None:
        self._gen.stop()
