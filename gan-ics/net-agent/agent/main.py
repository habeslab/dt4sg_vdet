from __future__ import annotations
"""
Entry point del net-agent:
  - Avvia Dispatcher (N worker)
  - Avvia Sniffer (se SNIFF_ENABLED=1)
  - Avvia Generatore in FEATURE-MODE (se GEN_ENABLED=1 e GEN_MODE=feature)

Termina pulito su SIGINT/SIGTERM.
"""
import os
import signal
import queue
import time
import threading
import uvicorn

from agent.sniffer import Sniffer
from agent.dispatcher import Dispatcher

def _run_web():
    port = int(os.getenv("DASH_PORT", "8080"))
    uvicorn.run("agent.webapp:app", host="0.0.0.0", port=port, log_level="info")

def main() -> None:
    q: "queue.Queue[dict]" = queue.Queue(maxsize=5000)

    # Dispatcher
    workers = int(os.getenv("DISPATCH_WORKERS", "2"))
    disp = Dispatcher(in_queue=q, workers=workers)
    disp.start()
    
    t_web = threading.Thread(target=_run_web, name="web", daemon=True)
    t_web.start()

    # Sniffer (opzionale)
    sniff_enabled = os.getenv("SNIFF_ENABLED", "1") not in ("0", "false", "False", "")
    sniffer = None
    if sniff_enabled:
        sniffer = Sniffer(q)
        sniffer.start()
    
    # Gestione segnali
    stop = False
    def _sig(_signum, _frame):
        nonlocal stop
        stop = True
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _sig)

    try:
        while not stop:
            time.sleep(0.5)
    finally:
        if sniffer:
            sniffer.stop()
        disp.stop()

if __name__ == "__main__":
    main()
