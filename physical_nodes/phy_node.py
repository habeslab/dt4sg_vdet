#!/usr/bin/env python3
"""
MQTT publisher: genera pacchetti IEC-104 arbitrari e pubblica frame Ethernet completi.

Pubblica JSON:
  {
    "pkt_b64": "<base64(Ethernet frame bytes)>",
    "ts": <float epoch>,
    "meta": {...}
  }

Env:
  MQTT_HOST (default 127.0.0.1)
  MQTT_PORT (default 1883)
  MQTT_TOPIC (default sg/packets/iec104)

  SRC_IP    (default 10.0.0.200)
  DST_IP    (default 10.0.0.10)
  SRC_PORT  (default 40000)
  DST_PORT  (default 2404)

  SRC_MAC   (default 02:00:00:00:00:aa)
  DST_MAC   (default ff:ff:ff:ff:ff:ff)  # broadcast ok in lab

  MODE      (default U_STARTDT)  # U_STARTDT | U_TESTFR | S_ACK | I_DUMMY
  RATE_HZ   (default 1.0)        # pacchetti al secondo
  COUNT     (default 0)          # 0=infinito

Requires:
  pip install paho-mqtt scapy
"""

from __future__ import annotations

import os
import time
import json
import base64
import logging
from typing import Dict, Any

import paho.mqtt.client as mqtt
from scapy.all import Ether, IP, TCP, Raw  # type: ignore

LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mqtt-iec104-pub")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "sg/packets/iec104")

SRC_IP = os.getenv("SRC_IP", "10.0.0.200")
DST_IP = os.getenv("DST_IP", "10.0.0.10")
SRC_PORT = int(os.getenv("SRC_PORT", "40000"))
DST_PORT = int(os.getenv("DST_PORT", "2404"))

SRC_MAC = os.getenv("SRC_MAC", "02:00:00:00:00:aa")
DST_MAC = os.getenv("DST_MAC", "ff:ff:ff:ff:ff:ff")

MODE = os.getenv("MODE", "U_STARTDT")
RATE_HZ = float(os.getenv("RATE_HZ", "1.0"))
COUNT = int(os.getenv("COUNT", "0"))

def iec104_apdu_u_startdt() -> bytes:
    # U-frame: STARTDT act = 0x07 0x00 0x00 0x00 (control field)
    # APCI: 0x68 L C0 C1 C2 C3   (L = 4 for control field only)
    return bytes([0x68, 0x04, 0x07, 0x00, 0x00, 0x00])

def iec104_apdu_u_testfr() -> bytes:
    # U-frame: TESTFR act = 0x43 0x00 0x00 0x00
    return bytes([0x68, 0x04, 0x43, 0x00, 0x00, 0x00])

def iec104_apdu_s_ack(rx_seq: int = 0) -> bytes:
    # S-frame: bit0=1 bit1=0 => 0x01, then ack = N(R)<<1
    # Control field: 0x01 0x00  (send seq not used), ack in last 2 bytes
    ack = (rx_seq & 0x7FFF) << 1
    c2 = ack & 0xFF
    c3 = (ack >> 8) & 0xFF
    return bytes([0x68, 0x04, 0x01, 0x00, c2, c3])

def iec104_apdu_i_dummy(tx_seq: int = 0, rx_seq: int = 0, asdu_payload: bytes = b"\x00") -> bytes:
    """
    I-frame dummy: control field + minimal ASDU bytes.
    Non garantisce semantica ASDU, ma produce un frame I formalmente valido:
      - C0/C1 encode N(S)<<1 with bit0=0
      - C2/C3 encode N(R)<<1
    L = 4 + len(ASDU)
    """
    ns = (tx_seq & 0x7FFF) << 1
    nr = (rx_seq & 0x7FFF) << 1
    c0 = ns & 0xFF
    c1 = (ns >> 8) & 0xFF
    c2 = nr & 0xFF
    c3 = (nr >> 8) & 0xFF
    L = 4 + len(asdu_payload)
    return bytes([0x68, L, c0, c1, c2, c3]) + asdu_payload

def build_apdu(mode: str, i: int) -> bytes:
    mode = mode.upper()
    if mode == "U_STARTDT":
        return iec104_apdu_u_startdt()
    if mode == "U_TESTFR":
        return iec104_apdu_u_testfr()
    if mode == "S_ACK":
        return iec104_apdu_s_ack(rx_seq=i)
    if mode == "I_DUMMY":
        # dummy ASDU: 6 bytes arbitrary (you can change)
        asdu = b"\x64\x01\x06\x00\x01\x00"  # totally arbitrary bytes
        return iec104_apdu_i_dummy(tx_seq=i, rx_seq=i, asdu_payload=asdu)
    # fallback
    return iec104_apdu_u_startdt()

def build_frame(apdu: bytes, seq: int) -> bytes:
    # Nota: TCP senza handshake reale, ma ok per replay/IDS pipeline.
    pkt = (
        Ether(src=SRC_MAC, dst=DST_MAC)
        / IP(src=SRC_IP, dst=DST_IP)
        / TCP(sport=SRC_PORT, dport=DST_PORT, flags="PA", seq=1000 + seq, ack=1, window=8192)
        / Raw(apdu)
    )
    return bytes(pkt)

def main() -> None:
    client = mqtt.Client(client_id=f"iec104-pub-{int(time.time())}", clean_session=True)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()

    if RATE_HZ <= 0:
        raise ValueError("RATE_HZ must be > 0")
    period = 1.0 / RATE_HZ

    log.info("Publishing IEC-104 frames to mqtt://%s:%d topic=%s mode=%s rate=%.3fHz count=%s",
             MQTT_HOST, MQTT_PORT, MQTT_TOPIC, MODE, RATE_HZ, COUNT if COUNT > 0 else "∞")

    sent = 0
    i = 0
    try:
        while True:

            log.debug("Start")
            i += 1
            apdu = build_apdu(MODE, i)
            frame = build_frame(apdu, i)
            
            msg: Dict[str, Any] = {
                "pkt_b64": base64.b64encode(frame).decode("ascii"),
                "ts": time.time(),
                "meta": {
                    "mode": "mqtt_pub",
                    "iec104_mode": MODE,
                    "src": f"{SRC_IP}:{SRC_PORT}",
                    "dst": f"{DST_IP}:{DST_PORT}",
                    "len": len(frame),
                },
            }
            log.debug("Start the sending with packet %s", msg)
            client.publish(MQTT_TOPIC, json.dumps(msg, separators=(",", ":")), qos=0, retain=False)
            sent += 1

            if sent % max(1, int(RATE_HZ)) == 0:
                log.debug("Sent=%d last_len=%d apdu_len=%d", sent, len(frame), len(apdu))

            if COUNT > 0 and sent >= COUNT:
                break

            time.sleep(period)

    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()
        log.info("Done. Sent=%d", sent)

if __name__ == "__main__":
    main()
