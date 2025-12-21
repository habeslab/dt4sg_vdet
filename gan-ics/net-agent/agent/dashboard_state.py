from __future__ import annotations
import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()
_last: Optional[Dict[str, Any]] = None

def set_last(payload: Dict[str, Any]) -> None:
    global _last
    with _lock:
        _last = payload

def get_last() -> Dict[str, Any]:
    with _lock:
        if _last is None:
            return {
                "ts": time.time(),
                "features_order": [],
                "p0": 1.0, "p1": 0.0, "p2": 0.0,
                "model_version": "unknown",
                "meta": {"note": "No data yet"},
            }
        return _last
