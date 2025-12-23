#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
import base64
import logging
import random
from typing import Dict, Any, Tuple

import paho.mqtt.client as mqtt
from scapy.all import Ether, IP, TCP, Raw  # type: ignore

LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mqtt-multi-phy-pub")

# MQTT
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "sg/packets/iec104")

# Traffic model
MODE = os.getenv("MODE", "I_DUMMY").upper()
RATE_HZ = float(os.getenv("RATE_HZ", "10.0"))
COUNT = int(os.getenv("COUNT", "0"))  # 0=infinite

# Multi-node settings
N_NODES = int(os.getenv("N_NODES", "10"))
NODE_SCHED = os.getenv("NODE_SCHED", "RR").upper()  # RR | RAND
SEED = int(os.getenv("SEED", "42"))

# Addressing plan
# Example: 10.0.0.(BASE_IP_LAST + node_id)
SRC_IP_BASE = os.getenv("SRC_IP_BASE", "10.0.0.")
SRC_IP_LAST0 = int(os.getenv("SRC_IP_LAST0", "200"))  # node0 -> .200
SRC_PORT0 = int(os.getenv("SRC_PORT0", "40000"))

# MAC base: 02:00:00:00:00:XX (XX = node_id)
SRC_MAC_PREFIX = os.getenv("SRC_MAC_PREFIX", "02:00:00:00:00")

# Destination (fixed)
DST_IP = os.getenv("DST_IP", "10.0.0.10")
DST_PORT = int(os.getenv("DST_PORT", "2404"))
DST_MAC = os.getenv("DST_MAC", "ff:ff:ff:ff:ff:ff")  # ok in lab

def iec104_apdu_u_startdt() -> bytes:
    return bytes([0x68, 0x04, 0x07, 0x00, 0x00, 0x00])

def iec104_apdu_u_testfr() -> bytes:
    return bytes([0x68, 0x04, 0x43, 0x00, 0x00, 0x00])

def iec104_apdu_s_ack(rx_seq: int = 0) -> bytes:
    ack = (rx_seq & 0x7FFF) << 1
    return bytes([0x68, 0x04, 0x01, 0x00, ack & 0xFF, (ack >> 8) & 0xFF])

def iec104_apdu_i_dummy(tx_seq: int = 0, rx_seq: int = 0, asdu_payload: bytes = b"\x64\x01\x06\x00\x01\x00") -> bytes:
    ns = (tx_seq & 0x7FFF) << 1
    nr = (rx_seq & 0x7FFF) << 1
    c0, c1 = ns & 0xFF, (ns >> 8) & 0xFF
    c2, c3 = nr & 0xFF, (nr >> 8) & 0xFF
    L = 4 + len(asdu_payload)
    return bytes([0x68, L, c0, c1, c2, c3]) + asdu_payload

def build_apdu(mode: str, seq: int) -> bytes:
    if mode == "U_STARTDT":
        return iec104_apdu_u_startdt()
    if mode == "U_TESTFR":
        return iec104_apdu_u_testfr()
    if mode == "S_ACK":
        return iec104_apdu_s_ack(rx_seq=seq)
    # default: I_DUMMY
    return iec104_apdu_i_dummy(tx_seq=seq, rx_seq=seq)

def node_addrs(node_id: int) -> Tuple[str, int, str]:
    # IP
    last = SRC_IP_LAST0 + node_id
    if not (0 <= last <= 254):
        last = 220
        #raise ValueError("SRC_IP_LAST0 + node_id must be within [0..254]")
        
    src_ip = f"{SRC_IP_BASE}{last}"

    # Port
    src_port = SRC_PORT0 + node_id

    # MAC
    mac_last = f"{node_id & 0xFF:02x}"
    src_mac = f"{SRC_MAC_PREFIX}:{mac_last}"
    return src_ip, src_port, src_mac

def build_frame(node_id: int, apdu: bytes, seq: int) -> bytes:
    src_ip, src_port, src_mac = node_addrs(node_id)
    pkt = (
        Ether(src=src_mac, dst=DST_MAC)
        / IP(src=src_ip, dst=DST_IP)
        / TCP(sport=src_port, dport=DST_PORT, flags="PA", seq=1000 + seq, ack=1, window=8192)
        / Raw(apdu)
    )
    return bytes(pkt)

def main() -> None:
    random.seed(SEED)

    client = mqtt.Client(client_id=f"multi-phy-pub-{int(time.time())}", clean_session=True)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()

    if RATE_HZ <= 0:
        raise ValueError("RATE_HZ must be > 0")
    period = 1.0 / RATE_HZ

    log.info("MQTT=%s:%d topic=%s mode=%s rate=%.2fHz nodes=%d sched=%s",
             MQTT_HOST, MQTT_PORT, MQTT_TOPIC, MODE, RATE_HZ, N_NODES, NODE_SCHED)

    sent = 0
    global_seq = 0
    rr = 0
    next_t = time.time()

    try:
        while True:
            global_seq += 1

            if NODE_SCHED == "RAND":
                node_id = random.randrange(0, N_NODES)
            else:
                node_id = rr
                rr = (rr + 1) % N_NODES

            apdu = build_apdu(MODE, global_seq)
            frame = build_frame(node_id, apdu, global_seq)

            now = time.time()
            msg: Dict[str, Any] = {
                "pkt_b64": base64.b64encode(frame).decode("ascii"),
                "ts": now,
                "meta": {
                    "mode": "mqtt_multi_phy_pub",
                    "node_id": int(node_id),
                    "seq": int(global_seq),
                    "iec104_mode": MODE,
                    "src": f"{DST_IP}:{DST_PORT}",
                    "len": int(len(frame)),
                },
            }

            client.publish(MQTT_TOPIC, json.dumps(msg, separators=(",", ":")), qos=0, retain=False)
            sent += 1

            if sent % max(1, int(RATE_HZ)) == 0:
                log.debug("Sent=%d last_node=%d last_len=%d", sent, node_id, len(frame))

            if COUNT > 0 and sent >= COUNT:
                break

            # sleep without drift
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))

    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()
        log.info("Done. Sent=%d", sent)

if __name__ == "__main__":
    main()
