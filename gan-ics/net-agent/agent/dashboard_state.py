from __future__ import annotations
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_last: Optional[Dict[str, Any]] = None
# spiegazioni (stadio diagnostico) per alert_id, le più recenti in coda
_explanations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_MAX_EXPLANATIONS = 200

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

def set_explanation(alert_id: str, report: Dict[str, Any]) -> None:
    with _lock:
        _explanations[alert_id] = {"alert_id": alert_id, "ts": time.time(), **report}
        _explanations.move_to_end(alert_id)
        while len(_explanations) > _MAX_EXPLANATIONS:
            _explanations.popitem(last=False)

def get_explanations(limit: int = 20) -> List[Dict[str, Any]]:
    with _lock:
        return list(reversed(list(_explanations.values())))[:limit]
