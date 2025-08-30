# /net-agent/agent/dispatcher.py
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import queue as _queue
import requests

DEFAULT_DISC_URL = os.getenv("DISC_URL", "http://disc-api:8000/predict3")
OUT_JSONL_PATH = Path(os.getenv("OUT_JSONL_PATH", "/data/flows.jsonl"))
TIMEOUT = float(os.getenv("DISC_TIMEOUT_SEC", "3.0"))
RETRY = int(os.getenv("DISC_RETRY", "2"))
RETRY_SLEEP = float(os.getenv("DISC_RETRY_SLEEP", "0.3"))

OOD_MESSAGE = "p2 sopra soglia (fuori distribuzione → possibile synthetic/malicious)"

class Dispatcher:
    def __init__(
        self,
        in_queue: "_queue.Queue[Optional[Dict[str, Any]]]" | None = None,
        disc_url: str = DEFAULT_DISC_URL,
        out_jsonl: Path | str = OUT_JSONL_PATH,
        workers: int | None = 2,
    ):
        self.queue: "_queue.Queue[Optional[Dict[str, Any]]]" = (
            in_queue if in_queue is not None else _queue.Queue(maxsize=1000)
        )
        if workers is None:
            self.worker_count = 2
        elif isinstance(workers, int):
            self.worker_count = max(1, workers)
        else:
            try:
                self.worker_count = max(1, int(workers))
            except Exception:
                self.worker_count = 2

        self.disc_url = disc_url
        self.out_jsonl = out_jsonl if isinstance(out_jsonl, Path) else Path(out_jsonl)
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_jsonl, "a", buffering=1, encoding="utf-8")

        self._stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []

    # lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        for i in range(self.worker_count):
            t = threading.Thread(target=self._worker_loop, name=f"dispatch-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_evt.set()
        for _ in self._threads:
            self.queue.put(None)
        for t in self._threads:
            t.join(timeout=1.0)
        try:
            self._fh.close()
        except Exception:
            pass
        self._threads.clear()

    # api ---------------------------------------------------------------
    def submit(self, features: Dict[str, Any], meta: Dict[str, Any]) -> None:
        # garantisce ts
        meta = dict(meta or {})
        meta.setdefault("ts", time.time())
        try:
            self.queue.put({"features": features, "meta": meta}, timeout=0.5)
        except Exception:
            print("[dispatcher] queue piena, scarto record", flush=True)

    # http --------------------------------------------------------------
    def _post_predict(self, features: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"features": features, "meta": meta}
        last_exc: Optional[Exception] = None
        for _ in range(RETRY + 1):
            try:
                r = requests.post(self.disc_url, json=payload, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(RETRY_SLEEP)
        raise RuntimeError(f"disc-api unreachable after retries: {last_exc}")

    # decision policy ---------------------------------------------------
    @staticmethod
    def _make_decision(resp: Dict[str, Any]) -> tuple[str, str, str]:
        p2 = float(resp.get("p2", 0.0))
        thr = resp.get("thresholds", {}) or {}
        tau_fake = float(thr.get("tau_fake", 0.9))
        tau_mal = float(thr.get("tau_mal", 0.7))
        p_mal_2c = float(resp.get("p_mal_2c", 0.0))

        # sanity: dovrebbero essere in [0,1]
        if p2 > 1.0 or tau_fake > 1.0 or tau_mal > 1.0 or p_mal_2c > 1.0:
            print(f"[dispatcher][WARN] valori non normalizzati: "
                  f"p2={p2:.3f}, p_mal_2c={p_mal_2c:.3f}, "
                  f"tau_fake={tau_fake:.3f}, tau_mal={tau_mal:.3f}", flush=True)

        if p2 >= tau_fake:
            decision = "alert"
            explanation = f"{OOD_MESSAGE} (p2={p2:.3f} ≥ τ_fake={tau_fake:.2f})."
            risk = "high"
        elif p_mal_2c >= tau_mal:
            decision = "malicious"
            explanation = f"p_mal_2c={p_mal_2c:.3f} ≥ τ_mal={tau_mal:.2f}"
            risk = "high"
        else:
            decision = "benign"
            explanation = f"p_mal_2c={p_mal_2c:.3f} < τ_mal={tau_mal:.2f} e p2={p2:.3f} < τ_fake={tau_fake:.2f}"
            risk = "low"
        return decision, explanation, risk

    # log ---------------------------------------------------------------
    def send_and_log(self, features: Dict[str, Any], meta: Dict[str, Any]) -> None:
        resp = self._post_predict(features, meta)
        decision, explanation, risk = self._make_decision(resp)
        record = {
            "ts": meta.get("ts"),
            "flow_id": meta.get("flow_id"),
            "features": features,
            "synthetic": bool(meta.get("synthetic", False)),
            "mode": meta.get("mode", "sniff"),
            "latency_ms": meta.get("latency_ms"),
            "response": resp,
            "decision": decision,
            "explanation": explanation,
            "risk": risk,
        }
        if decision == "alert":
            print(f"[ALERT][{record.get('flow_id')}] {explanation}", flush=True)
        try:
            line = json.dumps(record, ensure_ascii=False)
            self._fh.write(line + "\n")
        except Exception as e:
            print(f"[dispatcher] errore scrittura JSONL: {e}", flush=True)

    # worker loop -------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except Exception:
                continue
            if item is None:
                break
            try:
                features = item.get("features") or {}
                meta = item.get("meta") or {}
                self.send_and_log(features, meta)
            except Exception as e:
                print(f"[dispatcher] errore worker: {e}", flush=True)
