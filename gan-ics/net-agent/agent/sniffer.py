from __future__ import annotations
"""
Sniffer: cattura pacchetti ICS (IEC-104 su TCP/2404 per default), aggrega su finestre temporali
e manda al dispatcher un record con 5 feature:
    - sport, dport, proto, pkt_count, bytes_total

Env:
  IFACE_ICS   (default: eth1)   Interfaccia su cui sniffare
  BPF         (default: 'tcp port 2404') Filtro BPF
  WINDOW_SEC  (default: 60)     Durata finestra in secondi
  GRACE_SEC   (default: 5)      Ritardo prima di emettere una finestra chiusa
"""
import os
import time
import threading
import queue
from typing import Dict, Tuple, Any

from scapy.all import sniff, IP, TCP  # type: ignore

IFACE_ICS  = os.getenv("IFACE_ICS", "eth1")
BPF        = os.getenv("BPF", "tcp port 2404")
WINDOW_SEC = int(os.getenv("WINDOW_SEC", "60"))
GRACE_SEC  = int(os.getenv("GRACE_SEC",  "5"))

Key = Tuple[str, str, int, int, int]   # (src, dst, sport, dport, proto)

class _WindowAgg:
    __slots__ = ("pkt_count", "bytes_total")
    def __init__(self) -> None:
        self.pkt_count = 0
        self.bytes_total = 0

class Sniffer:
    def __init__(self, outq: "queue.Queue[Dict[str,Any]]") -> None:
        self.outq = outq
        self._stop = threading.Event()
        self._windows: Dict[Tuple[Key, int], _WindowAgg] = {}
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._flusher = threading.Thread(target=self._flush_loop, name="win-flusher", daemon=True)

    def start(self) -> None:
        """Start flusher and sniffer threads."""
        self._flusher.start()
        t = threading.Thread(target=self._sniff_loop, name="sniffer", daemon=True)
        t.start()

    def stop(self) -> None:
        """Signal stop and flush any remaining windows."""
        self._stop.set()
        try:
            self._flush_all()
        except Exception:
            pass

    # --- internals -----------------------------------------------------

    def _now_win(self, ts: float) -> int:
        return int((ts - self._t0) // WINDOW_SEC)

    def _sniff_loop(self) -> None:
        def onpkt(pkt) -> None:
            try:
                ts = float(getattr(pkt, "time", time.time()))
                # accetta solo IP/TCP
                if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
                    return
                ip = pkt[IP]
                tcp = pkt[TCP]
                src = ip.src
                dst = ip.dst
                sport = int(tcp.sport)
                dport = int(tcp.dport)
                proto = 6  # TCP
                if sport <= 0 or dport <= 0:
                    return

                win = self._now_win(ts)
                key: Key = (src, dst, sport, dport, proto)
                k = (key, win)
                ln = 0
                try:
                    ln = len(bytes(pkt))
                except Exception:
                    # fallback prudente
                    ln = int(getattr(pkt, "len", 0)) or 0

                with self._lock:
                    agg = self._windows.get(k)
                    if agg is None:
                        agg = _WindowAgg()
                        self._windows[k] = agg
                    agg.pkt_count += 1
                    agg.bytes_total += ln
            except Exception:
                # non interrompere lo sniff in caso di anomalie su singoli pacchetti
                pass

        try:
            sniff(
                iface=IFACE_ICS,
                filter=BPF,
                store=False,
                prn=onpkt,
                stop_filter=lambda _: self._stop.is_set(),
            )
        except Exception:
            # Se scapy fallisce (es. interfaccia non presente), fermiamo comunque il flusher
            self._stop.set()

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                now = time.time()
                to_emit: list[tuple[Tuple[Key, int], _WindowAgg]] = []
                with self._lock:
                    # emetti finestre chiuse da oltre GRACE_SEC
                    for (key, w), agg in list(self._windows.items()):
                        win_end_ts = self._t0 + (w + 1) * WINDOW_SEC
                        if (now - win_end_ts) >= GRACE_SEC:
                            to_emit.append(((key, w), agg))
                            del self._windows[(key, w)]
                for ((src, dst, sport, dport, proto), w), agg in to_emit:
                    if agg.pkt_count <= 0:
                        continue
                    features = {
                        "sport": sport,
                        "dport": dport,
                        "proto": proto,
                        "pkt_count": int(agg.pkt_count),
                        "bytes_total": int(agg.bytes_total),
                    }
                    flow_id = f"{src}|{dst}|{sport}|{dport}|{proto}|w{w}"
                    self.outq.put({
                        "features": features,
                        "meta": {
                            "flow_id": flow_id,
                            "window_ts": self._t0 + w * WINDOW_SEC,
                            "synthetic": False,
                            "mode": "sniff",
                        }
                    })
            except Exception:
                pass
            time.sleep(1.0)

    def _flush_all(self) -> None:
        """Emette tutte le finestre ancora in memoria (usato allo stop)."""
        to_emit: list[tuple[Tuple[Key, int], _WindowAgg]] = []
        with self._lock:
            for (key, w), agg in list(self._windows.items()):
                to_emit.append(((key, w), agg))
            self._windows.clear()
        for ((src, dst, sport, dport, proto), w), agg in to_emit:
            if agg.pkt_count <= 0:
                continue
            features = {
                "sport": sport,
                "dport": dport,
                "proto": proto,
                "pkt_count": int(agg.pkt_count),
                "bytes_total": int(agg.bytes_total),
            }
            flow_id = f"{src}|{dst}|{sport}|{dport}|{proto}|w{w}"
            self.outq.put({
                "features": features,
                "meta": {
                    "flow_id": flow_id,
                    "window_ts": self._t0 + w * WINDOW_SEC,
                    "synthetic": False,
                    "mode": "sniff",
                }
            })
