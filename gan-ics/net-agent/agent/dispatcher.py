from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import queue as _queue
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime, timezone, timedelta
from agent.dashboard_state import set_last, set_explanation

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ_LOCAL = ZoneInfo("Europe/Rome")
except Exception:
    # Fallback (senza cambio automatico ora legale): forza +02:00
    TZ_LOCAL = timezone(timedelta(hours=2))
"""
Dispatcher per inoltrare feature al servizio disc-api e loggare le risposte in JSONL.

- Coda thread-safe + pool di worker per invii concorrenti a `DEFAULT_DISC_URL` (/predict3).
- Ritenti HTTP configurabili (TIMEOUT, RETRY, RETRY_SLEEP) con backoff fisso.
- Decision policy: classe predetta = argmax(p0, p1, p2) (0 Normal, 1 Availability/DoS, 2 Integrity).
- Alerting a due stadi (paper Sec. III-E): record "fast_path" subito dopo la classificazione;
  per i flussi malevoli, record "diagnostic" con la spiegazione SHAP (chi) letta da
  GET /explain/{alert_id} quando disc-api l'ha calcolata in modo asincrono.
- Ogni record salvato su `OUT_JSONL_PATH` include meta (ts, flow_id), risposta, decisione e latenze.
- Metodi principali:
  * submit(features, meta): enqueue non bloccante (auto-ts).
  * start()/stop(): lifecycle dei worker e gestione file di output.
  * send_and_log(features, meta): invio sincrono + decisione + persistenza.
- Variabili d’ambiente utili: DISC_URL, OUT_JSONL_PATH, DISC_TIMEOUT_SEC, DISC_RETRY, DISC_RETRY_SLEEP,
  XAI_URL (default: base di DISC_URL + /explain), XAI_POLL_SEC, XAI_TIMEOUT_SEC.
"""

DEFAULT_DISC_URL = os.getenv("DISC_URL", "http://disc-api:8000/predict3")
OUT_JSONL_PATH = Path(os.getenv("OUT_JSONL_PATH", "/data/flows.jsonl"))
TIMEOUT = float(os.getenv("DISC_TIMEOUT_SEC", "3.0"))
RETRY = int(os.getenv("DISC_RETRY", "2"))
RETRY_SLEEP = float(os.getenv("DISC_RETRY_SLEEP", "0.3"))

XAI_POLL_SEC = float(os.getenv("XAI_POLL_SEC", "0.02"))
XAI_TIMEOUT_SEC = float(os.getenv("XAI_TIMEOUT_SEC", "10.0"))

CLASS_LABELS = {0: "normal", 1: "availability", 2: "integrity"}


def _explain_base_url(disc_url: str) -> str:
    """http://host:8000/predict3 -> http://host:8000/explain"""
    u = urlsplit(disc_url)
    return urlunsplit((u.scheme, u.netloc, "/explain", "", ""))


DEFAULT_XAI_URL = os.getenv("XAI_URL", _explain_base_url(DEFAULT_DISC_URL))

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
        self.xai_url = DEFAULT_XAI_URL if disc_url == DEFAULT_DISC_URL else _explain_base_url(disc_url)
        self.out_jsonl = out_jsonl if isinstance(out_jsonl, Path) else Path(out_jsonl)
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_jsonl, "a", buffering=1, encoding="utf-8")

        self._fh_lock = threading.Lock()

        self._stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []
        # stadio diagnostico: attesa delle spiegazioni senza bloccare i worker del fast path
        self._xai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xai-wait")

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
        self._xai_pool.shutdown(wait=False, cancel_futures=True)
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
    def _make_decision(resp: Dict[str, Any]) -> tuple[int, str, str, str]:
        p = [float(resp.get(k, 0.0)) for k in ("p0", "p1", "p2")]
        cls = int(resp.get("pred_class", max(range(3), key=lambda i: p[i])))
        label = CLASS_LABELS.get(cls, str(cls))
        if cls == 0:
            return cls, label, f"normal traffic (p0={p[0]:.3f})", "low"
        return cls, label, f"{label} attack (p{cls}={p[cls]:.3f}, p0={p[0]:.3f})", "high"

    # log ---------------------------------------------------------------
    def send_and_log(self, features: Dict[str, Any], meta: Dict[str, Any]) -> None:
        # --- Timestamps robusti lato client ---
        t_client = float(meta.get("ts") or meta.get("window_ts") or time.time())

        # --- Chiamata al disc-api + rtt http ---
        t0 = time.time()
        resp = self._post_predict(features, meta)
        

        set_last(resp)
        t1 = time.time()
        rtt_ms = int(max(0, round((t1 - t0) * 1000)))

        # --- Timestamp lato server (dal disc-api) ---
        t_server = float(resp.get("ts") or t1)

        # --- Decisione e rischio ---
        pred_class, decision, explanation, risk = self._make_decision(resp)

        # --- Metriche di latenza (spiegate) ---
        e2e_sec = max(0.0, round(t_server - t_client, 3))  # end-to-end: cattura -> decisione
        latency_expl = f"end_to_end = server_ts - ts = {t_server:.3f} - {t_client:.3f} ≈ {e2e_sec:.3f}s"
        obs = _observation_latency(meta, t_server, "alert")

        # --- Selezione campi chiave dalla risposta per non 'rumoreggiare' il JSON ---
        decision_block = {
            "pred_class": pred_class,
            "label": decision,
            "model": resp.get("model_version"),
            "p0": resp.get("p0"),
            "p1": resp.get("p1"),
            "p2": resp.get("p2"),
            "alert_id": resp.get("alert_id"),
            "xai": resp.get("xai"),
        }

        # --- Record compatto ma esplicativo ---
        record = {
            "stage": "fast_path",
            "flow_id": meta.get("flow_id"),
            "mode": meta.get("mode", "sniff"),
            "synthetic": bool(meta.get("synthetic", False)),

            "timestamps": {
                "client": {
                    "epoch": t_client,
                    "utc": iso_utc(t_client),
                    "local": iso_local(t_client),
                },
                "server": {
                    "epoch": t_server,
                    "utc": iso_utc(t_server),
                    "local": iso_local(t_server),
                },
            },

            "latency": {
                "http_rtt_ms": rtt_ms,
                "end_to_end_s": e2e_sec,
                "explanation": latency_expl,
                **obs,
            },

            "features": features,
            "decision": decision_block,
            "risk": risk,
            "explanation": explanation,
        }

        # --- Log console 'accattivante' e leggibile a colpo d'occhio ---
        pretty_line = (
        f"[{record['timestamps']['server']['local']}] "
        f"{record['mode']} flow={record['flow_id']} "
        f"→ {decision} "
        f"(e2e≈{e2e_sec:.2f}s, http={rtt_ms}ms, p0={_fmt(decision_block['p0'])}, p1={_fmt(decision_block['p1'])}, "
        f"p2={_fmt(decision_block['p2'])}, xai={decision_block['xai']})"
        )
        print(pretty_line, flush=True)

        # --- Scrittura JSONL (1 riga compatta e pulita) ---
        self._write(record)

        # --- Stadio diagnostico (asincrono) ---
        if resp.get("xai") == "pending" and resp.get("alert_id"):
            self._xai_pool.submit(self._await_explanation, str(resp["alert_id"]), meta, t_client, t_server)

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False)
            with self._fh_lock:
                self._fh.write(line + "\n")
        except Exception as e:
            print(f"[dispatcher] errore scrittura JSONL: {e}", flush=True)

    def _await_explanation(self, alert_id: str, meta: Dict[str, Any], t_client: float, t_fast: float) -> None:
        url = f"{self.xai_url}/{alert_id}"
        deadline = time.time() + XAI_TIMEOUT_SEC
        res: Optional[Dict[str, Any]] = None
        while time.time() < deadline and not self._stop_evt.is_set():
            try:
                r = requests.get(url, timeout=TIMEOUT)
                if r.status_code == 200:
                    res = r.json()
                    break
                if r.status_code != 202:
                    print(f"[dispatcher] explain {alert_id}: HTTP {r.status_code}", flush=True)
                    return
            except Exception as e:
                print(f"[dispatcher] explain {alert_id}: {e}", flush=True)
            time.sleep(XAI_POLL_SEC)
        if res is None or res.get("status") != "done":
            print(f"[dispatcher] explain {alert_id}: not available ({(res or {}).get('status', 'timeout')})", flush=True)
            return

        t_recv = time.time()
        chi = res.get("chi") or {}
        t_xai_end = float((res.get("timestamps") or {}).get("t_xai_end") or t_recv)
        latency = {
            **(res.get("latency_ms") or {}),
            "capture_to_fast_path_s": round(max(0.0, t_fast - t_client), 3),
            "capture_to_explanation_s": round(max(0.0, t_xai_end - t_client), 3),
            "capture_to_diagnostic_report_s": round(max(0.0, t_recv - t_client), 3),
            **_observation_latency(meta, t_xai_end, "explanation"),
            **_observation_latency(meta, t_recv, "diagnostic_report"),
        }
        set_explanation(alert_id, {"flow_id": meta.get("flow_id"), "chi": chi, "latency": latency})
        self._write({
            "stage": "diagnostic",
            "alert_id": alert_id,
            "flow_id": meta.get("flow_id"),
            "chi": chi,
            "latency": latency,
        })
        top = ", ".join(f"{t['feature']}={t['shap']:+.3f}" for t in chi.get("top", [])[:3])
        print(f"[diagnostic] flow={meta.get('flow_id')} class={chi.get('class')} "
              f"(+{latency.get('pred_to_explanation', 0):.0f}ms) {top}", flush=True)



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

def _observation_latency(meta: Dict[str, Any], t: float, what: str) -> Dict[str, float]:
    """Latency of event `what` (at time t) from the packet observation and window boundaries of the flow."""
    out: Dict[str, float] = {}
    for key, name in (("first_pkt_ts", "first_pkt"), ("last_pkt_ts", "last_pkt"),
                      ("window_end_ts", "window_end"), ("timestamp", "window_emit")):
        v = meta.get(key)
        if isinstance(v, (int, float)):
            out[f"{name}_to_{what}_s"] = round(t - float(v), 4)
    return out


def iso_utc(ts: float) -> str:
    """2025-09-07T13:48:17Z"""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def iso_local(ts: float) -> str:
    """2025-09-07T15:48:17+02:00 (Europe/Rome, con DST se ZoneInfo disponibile)"""
    return datetime.fromtimestamp(ts, tz=TZ_LOCAL).isoformat(timespec="seconds")
def _fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "n/a"
