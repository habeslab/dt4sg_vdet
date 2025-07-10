#!/usr/bin/env python3
"""
RTU IEC-104 minimale con libreria `c104` (iec104-python).
"""

import os, json, time, random, c104

DATASET = os.getenv("DATASET", "/data/dataset.json")
HOST_IP = os.getenv("HOST_IP", "0.0.0.0")
PORT    = int(os.getenv("PORT", "2404"))
CA      = int(os.getenv("CA",   "1"))

print(f"[RTU] dataset={DATASET}  ip={HOST_IP}:{PORT}  CA={CA}")

# ─── carica dataset ─────────────────────────────────────────────────────────
with open(DATASET) as fh:
    data = json.load(fh)                 # es. [{"ioa":11,"value":0}, …]

srv     = c104.Server(ip=HOST_IP, port=PORT)
station = srv.add_station(common_address=CA)

points = {}
for obj in data:
    p = station.add_point(io_address=obj["ioa"], type=c104.Type.M_SP_NA_1)
    p.value = bool(obj["value"])         # ← cast a bool evita ValueError
    points[obj["ioa"]] = p
    print(f"[RTU] init ioa={obj['ioa']} → {int(p.value)}")

# ─── variazioni spontanee ───────────────────────────────────────────────────
def spontaneous():
    while True:
        ioa, p = random.choice(list(points.items()))
        p.value = not p.value            # toggle booleano
        print(f"[RTU] toggle ioa={ioa} → {int(p.value)}")
        time.sleep(random.randint(5, 10))

srv.start()
print("[RTU] IEC-104 server in ascolto")
spontaneous()
