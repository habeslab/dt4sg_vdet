#!/usr/bin/env python3
"""
MQTT-based packet replay.

Replica esattamente il comportamento del replay da PCAP:
- riceve pacchetti completi via MQTT (base64)
- reinietta i pacchetti su IFACE usando sendp()

Env:
  MQTT_HOST      (default: 127.0.0.1)
  MQTT_PORT      (default: 1883)
  MQTT_TOPIC     (default: sg/packets/#)

  IFACE          (default: eth0)
  DELAY_FACTOR   (default: 1.0)

  LOGLEVEL       (default: INFO)

Requires:
  pip install paho-mqtt scapy
"""

from __future__ import annotations

import os
import sys
import time
import json
import base64
import logging
import csv
import queue
import threading

from scapy.all import Ether, sendp  # type: ignore
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MQTT_HOST  = os.getenv("MQTT_HOST", "192.168.122.49")
MQTT_PORT  = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "sg/packets/#")

IFACE = os.getenv("IFACE", "eth0")
REPLICATION_IFACE = os.getenv("REPLICATION_IFACE", "eth1")
EVALUATION = os.getenv("EVALUATION", "")
DELAY = float(os.getenv("DELAY_FACTOR", "1.0"))

LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mqtt-replay")
CSV_PATH_QUEUE = os.getenv("CSV_PATH", "/out/queue_length_" + EVALUATION + ".csv")
CSV_PATH = os.getenv("CSV_PATH", "/out/replication_latency_" + EVALUATION + ".csv")
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

q = queue.Queue(maxsize=5000)
csv_file_queue = open(CSV_PATH_QUEUE, "a", newline="")
csv_file = open(CSV_PATH, "a", newline="")
writer_queue = csv.writer(csv_file_queue)
writer = csv.writer(csv_file)

# scrivi header una sola volta
if csv_file.tell() == 0:
    writer.writerow([
        "exp_id", "scenario", "flow_id",
        "ts_capture", "ts_receive", "lat_rep_ms"
    ])
if csv_file_queue.tell() == 0:
    writer_queue.writerow([
        "exp_id", "scenario", "flow_id",
        "ts_capture", "ts_receive", "lat_rep_ms"
    ])
# ---------------------------------------------------------------------------
# State (per timing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("Connected to MQTT %s:%d – subscribing to %s",
                 MQTT_HOST, MQTT_PORT, MQTT_TOPIC)
        client.subscribe(MQTT_TOPIC, qos=0)
    else:
        log.error("MQTT connection failed rc=%s", rc)
import queue
import threading
import time

q = queue.Queue(maxsize=5000)  # limita backlog (scegli tu)

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        raw, ts_pub, t_enq, flow_id = item

        # qui misuri due cose utili:
        # 1) quanto backlog hai accumulato nel receiver
        t_deq = time.time()
        queue_ms = (t_deq - t_enq) * 1000

        try:
            pkt = Ether(raw)
            sendp(pkt, iface=REPLICATION_IFACE, verbose=False)
            t_sent = time.time()
        except Exception as e:
            log.error("sendp failed: %s", e)
            q.task_done()
            continue

        # 2) "end-to-end" dal publisher al momento in cui riesci a processare davvero
        e2e_ms = (t_sent - ts_pub) * 1000

        writer_queue.writerow([1, "three_nodes", flow_id, ts_pub, t_sent, e2e_ms])
        csv_file_queue.flush()

        q.task_done()

threading.Thread(target=worker, daemon=True).start()

def on_message(client, userdata, msg):
    t_enq = time.time()
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        b64 = data.get("pkt_b64")
        ts_pub = data.get("ts")
        mqtt_ms = (t_enq - ts_pub) * 1000
        flow_id = data.get("meta", {}).get("src", "unknown")
        writer.writerow([1, "three_nodes", flow_id, ts_pub, t_enq, mqtt_ms])
        csv_file.flush()
        if not isinstance(b64, str) or not isinstance(ts_pub, (int, float)):
            return
        raw = base64.b64decode(b64)
    except Exception:
        return

    # se la queue è piena, droppi (meglio droppare che accumulare 10s di backlog)
    try:
        q.put_nowait((raw, ts_pub, t_enq, flow_id))
    except queue.Full:
        log.warning("Queue full -> drop")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client = mqtt.Client(
        client_id=f"mqtt-replay-{int(time.time())}",
        clean_session=True
    )
    client.on_connect = on_connect
    client.on_message = on_message

    log.info("Starting MQTT replay module on IFACE=%s (delay=%s) with MQTT Server %s on port %d",
             IFACE, DELAY, MQTT_HOST, MQTT_PORT)

    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        log.info("Stopped by user")

if __name__ == "__main__":
    main()
