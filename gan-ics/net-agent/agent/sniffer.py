from __future__ import annotations
import os
import time
import threading
import queue
from typing import Dict, Tuple, Any, Optional

from scapy.all import sniff, IP, TCP  # type: ignore
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
logger = logging.getLogger("Sniffer")

IFACE_ICS  = os.getenv("ICS_IFACE", "eth1")
BPF        = os.getenv("BPF", "tcp port 2404")
WINDOW_SEC = int(os.getenv("WINDOW_SEC", "20"))
GRACE_SEC  = int(os.getenv("GRACE_SEC",  "5"))
EXCLUDE_IP = os.getenv("EXCLUDE_IP", "10.0.0.20")

# Flow canonico (bidirezionale): endpoint A < endpoint B
Endpoint = Tuple[str, int]  # (ip, port)
FlowKey = Tuple[Endpoint, Endpoint, int]  # ((ipA,portA),(ipB,portB), proto)

def _canon_flow(src: str, sport: int, dst: str, dport: int, proto: int) -> Tuple[FlowKey, bool]:
    """Ritorna (flow_key_canonico, is_forward) dove forward = A->B."""
    a: Endpoint = (src, sport)
    b: Endpoint = (dst, dport)
    if a <= b:
        return ((a, b, proto), True)
    else:
        return ((b, a, proto), False)

def _iec104_msg_type(tcp_payload: bytes) -> Optional[str]:
    """
    Classifica IEC 60870-5-104 APCI format: I/S/U.
    IEC-104 over TCP: starts with 0x68, then length, then 4-byte control field.
    Control field:
      - I-format: bit0 == 0
      - S-format: bit0 == 1 and bit1 == 0
      - U-format: bit0 == 1 and bit1 == 1
    """
    if len(tcp_payload) < 6:
        return None
    if tcp_payload[0] != 0x68:
        return None
    # byte[1] = length (non sempre serve validarlo qui)
    c0 = tcp_payload[2]
    # c1 = tcp_payload[3]  # non necessario per il formato
    if (c0 & 0x01) == 0:
        return "I"
    # bit0=1
    if (c0 & 0x02) == 0:
        return "S"
    return "U"

def _iec104_apdu_len(tcp_payload: bytes) -> Optional[int]:
    """
    IEC-104 APDU: 0x68 | L | ... (L bytes after length field)
    Lunghezza totale frame = L + 2.
    """
    if len(tcp_payload) < 2:
        return None
    if tcp_payload[0] != 0x68:
        return None
    L = int(tcp_payload[1])
    apdu_len = L + 2
    # sanity: non superare payload reale
    if apdu_len <= 0 or apdu_len > len(tcp_payload):
        # se il payload TCP contiene più APDU concatenati o frammentazione,
        # qui manteniamo una stima conservativa
        apdu_len = min(len(tcp_payload), max(0, apdu_len))
    return apdu_len


class _WindowAgg:
    __slots__ = (
        "total_flow_packets",
        "bytes_total",
        "ack_flag_count",
        "psh_flag_count",
        "init_fw_window_bytes",
        "init_bw_window_bytes",
        "fw_subflow_packets",
        "fw_subflow_bytes",
        "bw_subflow_packets",
        "bw_subflow_bytes",
        "iec104_s_total",
        "iec104_u_total",
        "iec104_u_bw_total",
        # --- NEW: fw IAT stats ---
        "last_fw_ts",
        "fw_iat_count",
        "fw_iat_sum",
        "fw_iat_sumsq",
        # --- NEW: IEC-104 U forward ---
        "iec104_u_fw_total",
        # --- NEW: APDU length stats (flow-level) ---
        "apdu_count",
        "apdu_sum",
        "apdu_sumsq",
        # --- NEW: APDU total length forward ---
        "fw_apdu_sum",

    )

    def __init__(self) -> None:
        self.total_flow_packets = 0
        self.bytes_total = 0

        self.ack_flag_count = 0
        self.psh_flag_count = 0

        self.init_fw_window_bytes: Optional[int] = None
        self.init_bw_window_bytes: Optional[int] = None

        self.fw_subflow_packets = 0
        self.fw_subflow_bytes = 0
        self.bw_subflow_packets = 0
        self.bw_subflow_bytes = 0

        self.iec104_s_total = 0
        self.iec104_u_total = 0
        self.iec104_u_bw_total = 0
        # --- NEW: fw IAT stats ---
        self.last_fw_ts: Optional[float] = None
        self.fw_iat_count = 0
        self.fw_iat_sum = 0.0
        self.fw_iat_sumsq = 0.0
        # --- NEW: IEC-104 U forward ---
        self.iec104_u_fw_total = 0
        # --- NEW: APDU stats (flow-level) ---
        self.apdu_count = 0
        self.apdu_sum = 0.0
        self.apdu_sumsq = 0.0
        # --- NEW: APDU total length forward ---
        self.fw_apdu_sum = 0


class Sniffer:
    def __init__(self, outq: "queue.Queue[Dict[str,Any]]") -> None:
        self.outq = outq
        self._stop = threading.Event()
        self._windows: Dict[Tuple[FlowKey, int], _WindowAgg] = {}
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._flusher = threading.Thread(target=self._flush_loop, name="win-flusher", daemon=True)

    def start(self) -> None:
        self._flusher.start()
        t = threading.Thread(target=self._sniff_loop, name="sniffer", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._flush_all()
        except Exception:
            pass

    def _now_win(self, ts: float) -> int:
        return int((ts - self._t0) // WINDOW_SEC)

    def _sniff_loop(self) -> None:
        def onpkt(pkt) -> None:
            try:
                logger.debug("Packet received")

                ts = float(getattr(pkt, "time", time.time()))
                if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
                    return

                ip = pkt[IP]
                tcp = pkt[TCP]
                src = ip.src
                dst = ip.dst
                if src == EXCLUDE_IP or dst == EXCLUDE_IP:
                    return

                sport = int(tcp.sport)
                dport = int(tcp.dport)
                proto = 6  # TCP
                
                if sport <= 0 or dport <= 0:
                    return

                win = self._now_win(ts)

                flow_key, is_fw = _canon_flow(src, sport, dst, dport, proto)
                k = (flow_key, win)

                try:
                    pkt_len = len(bytes(pkt))
                except Exception:
                    pkt_len = int(getattr(pkt, "len", 0)) or 0

                # TCP flags (Scapy: tcp.flags è un FlagValue; possiamo usare bitmask)
                flags = int(tcp.flags)
                has_ack = (flags & 0x10) != 0
                has_psh = (flags & 0x08) != 0

                # TCP window advertised (non-scalata): ok come feature "init * window bytes"
                tcp_win = int(getattr(tcp, "window", 0) or 0)

                # IEC-104 type (I/S/U) dal payload TCP
                payload = bytes(tcp.payload) if tcp.payload is not None else b""
                iec_type = _iec104_msg_type(payload)
                apdu_len = _iec104_apdu_len(payload)
                
                if iec_type:
                    logger.debug(
                        "IEC-104 detected: type=%s direction=%s",
                        iec_type, "FW" if is_fw else "BW"
                    )

                logger.debug(
                    "Pkt %s:%d -> %s:%d flags=%s win=%d len=%d",
                    src, sport, dst, dport, tcp.flags, tcp_win, pkt_len
                )
                with self._lock:
                    agg = self._windows.get(k)
                    if agg is None:
                        agg = _WindowAgg()
                        self._windows[k] = agg

                    # Totali
                    agg.total_flow_packets += 1
                    agg.bytes_total += pkt_len

                    # Flag counts
                    if has_ack:
                        agg.ack_flag_count += 1
                    if has_psh:
                        agg.psh_flag_count += 1

                    # Init window bytes fw/bw (prima osservazione nella finestra)
                    if is_fw:
                        if agg.init_fw_window_bytes is None and tcp_win > 0:
                            agg.init_fw_window_bytes = tcp_win
                        agg.fw_subflow_packets += 1
                        agg.fw_subflow_bytes += pkt_len
                        if agg.last_fw_ts is not None:
                            iat = ts - agg.last_fw_ts
                            if iat >= 0:
                                agg.fw_iat_count += 1
                                agg.fw_iat_sum += iat
                                agg.fw_iat_sumsq += iat * iat
                        agg.last_fw_ts = ts
                    else:
                        if agg.init_bw_window_bytes is None and tcp_win > 0:
                            agg.init_bw_window_bytes = tcp_win
                        agg.bw_subflow_packets += 1
                        agg.bw_subflow_bytes += pkt_len
                        

                    # IEC-104 counts
                    if iec_type == "S":
                        agg.iec104_s_total += 1
                    elif iec_type == "U":
                        agg.iec104_u_total += 1
                        if is_fw:
                            agg.iec104_u_fw_total += 1
                        else:
                            agg.iec104_u_bw_total += 1

                    if apdu_len is not None and apdu_len > 0:
                        # flow-level stats (tutte le direzioni)
                        agg.apdu_count += 1
                        agg.apdu_sum += float(apdu_len)
                        agg.apdu_sumsq += float(apdu_len) * float(apdu_len)

                        # fw total APDU length
                        if is_fw:
                            agg.fw_apdu_sum += int(apdu_len)

                    
                    
                    logger.debug(
                        "Agg update | flow_packets=%d fw_pkts=%d bw_pkts=%d "
                        "S=%d U=%d U_bw=%d",
                        agg.total_flow_packets,
                        agg.fw_subflow_packets,
                        agg.bw_subflow_packets,
                        agg.iec104_s_total,
                        agg.iec104_u_total,
                        agg.iec104_u_bw_total,
                    )

            except Exception:
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
            self._stop.set()

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                now = time.time()
                to_emit: list[tuple[Tuple[FlowKey, int], _WindowAgg]] = []
                with self._lock:
                    for (fk, w), agg in list(self._windows.items()):
                        win_end_ts = self._t0 + (w + 1) * WINDOW_SEC
                        if (now - win_end_ts) >= GRACE_SEC:
                            to_emit.append(((fk, w), agg))
                            del self._windows[(fk, w)]

                for ((a, b, proto), w), agg in to_emit:
                    if agg.total_flow_packets <= 0:
                        continue

                    (ipA, portA) = a
                    (ipB, portB) = b
                    
                    # fw IAT std
                    if agg.fw_iat_count > 0:
                        mean = agg.fw_iat_sum / agg.fw_iat_count
                        var = max(0.0, (agg.fw_iat_sumsq / agg.fw_iat_count) - (mean * mean))
                        fw_iat_std = float(var ** 0.5)
                    else:
                        fw_iat_std = 0.0

                    # flow packet APDU length var
                    if agg.apdu_count > 0:
                        mean_apdu = agg.apdu_sum / agg.apdu_count
                        var_apdu = max(0.0, (agg.apdu_sumsq / agg.apdu_count) - (mean_apdu * mean_apdu))
                    else:
                        var_apdu = 0.0

                    

                    features = {
                        # --- matcha le feature importance che mi hai mostrato ---
                        "total flow packets": int(agg.total_flow_packets),
                        "flow ACK flag count": int(agg.ack_flag_count),
                        "flow PSH flag count": int(agg.psh_flag_count),

                        "init fw window bytes": int(agg.init_fw_window_bytes or 0),
                        "init bw window bytes": int(agg.init_bw_window_bytes or 0),

                        "fw_subflow_packets": int(agg.fw_subflow_packets),
                        "fw_subflow_bytes": int(agg.fw_subflow_bytes),

                        # se il tuo modello non le usa, puoi anche non inviarle
                        "bw_subflow_packets": int(agg.bw_subflow_packets),
                        "bw_subflow_bytes": int(agg.bw_subflow_bytes),

                        "flow total IEC104_S_Message packets": int(agg.iec104_s_total),
                        "flow total IEC104_U_Message packets": int(agg.iec104_u_total),
                        "bw total IEC104_U_Message packets": int(agg.iec104_u_bw_total),

                        # utile se vuoi mantenere compatibilità / debug
                        "proto": int(proto),
                        "bytes_total": int(agg.bytes_total),
                    }

                    features.update({
                        "flow packet APDU length var": float(var_apdu),
                        "fw IAT std": float(fw_iat_std),
                        "fw total IEC104_U_Message packets": int(agg.iec104_u_fw_total),
                        "total fw packets": int(agg.fw_subflow_packets),
                        "fw packets APDU total length": int(agg.fw_apdu_sum),
                    })


                    flow_id = f"{ipA}:{portA}|{ipB}:{portB}|{proto}|w{w}"
                    self.outq.put({
                        "features": features,
                        "meta": {
                            "flow_id": flow_id,
                            "window_ts": self._t0 + w * WINDOW_SEC,
                            "synthetic": False,
                            "mode": "sniff",
                            "timestamp": time.time(),
                        }
                    })

            except Exception:
                pass
            time.sleep(1.0)

    def _flush_all(self) -> None:
        to_emit: list[tuple[Tuple[FlowKey, int], _WindowAgg]] = []
        with self._lock:
            for (fk, w), agg in list(self._windows.items()):
                to_emit.append(((fk, w), agg))
            self._windows.clear()

        for ((a, b, proto), w), agg in to_emit:
            if agg.total_flow_packets <= 0:
                continue
            (ipA, portA) = a
            (ipB, portB) = b
            # fw IAT std
            if agg.fw_iat_count > 0:
                mean = agg.fw_iat_sum / agg.fw_iat_count
                var = max(0.0, (agg.fw_iat_sumsq / agg.fw_iat_count) - (mean * mean))
                fw_iat_std = float(var ** 0.5)
            else:
                fw_iat_std = 0.0

            # flow packet APDU length var
            if agg.apdu_count > 0:
                mean_apdu = agg.apdu_sum / agg.apdu_count
                var_apdu = max(0.0, (agg.apdu_sumsq / agg.apdu_count) - (mean_apdu * mean_apdu))
            else:
                var_apdu = 0.0

            features = {
                "total flow packets": int(agg.total_flow_packets),
                "flow ACK flag count": int(agg.ack_flag_count),
                "flow PSH flag count": int(agg.psh_flag_count),
                "init fw window bytes": int(agg.init_fw_window_bytes or 0),
                "init bw window bytes": int(agg.init_bw_window_bytes or 0),
                "fw_subflow_packets": int(agg.fw_subflow_packets),
                "fw_subflow_bytes": int(agg.fw_subflow_bytes),
                "bw_subflow_packets": int(agg.bw_subflow_packets),
                "bw_subflow_bytes": int(agg.bw_subflow_bytes),
                "flow total IEC104_S_Message packets": int(agg.iec104_s_total),
                "flow total IEC104_U_Message packets": int(agg.iec104_u_total),
                "bw total IEC104_U_Message packets": int(agg.iec104_u_bw_total),
                "proto": int(proto),
                "bytes_total": int(agg.bytes_total),
            }
            features.update({
                "flow packet APDU length var": float(var_apdu),
                "fw IAT std": float(fw_iat_std),
                "fw total IEC104_U_Message packets": int(agg.iec104_u_fw_total),
                "total fw packets": int(agg.fw_subflow_packets),
                "fw packets APDU total length": int(agg.fw_apdu_sum),
            })

            flow_id = f"{ipA}:{portA}|{ipB}:{portB}|{proto}|w{w}"
            self.outq.put({
                "features": features,
                "meta": {
                    "flow_id": flow_id,
                    "window_ts": self._t0 + w * WINDOW_SEC,
                    "synthetic": False,
                    "mode": "sniff",
                    "timestamp": time.time(),
                }
            })