#!/usr/bin/env python3
"""
substation_simulator.py – Simulatore RTU IEC-104 professionale e retro-compatibile

Due modalità:
• Compatibility Mode (FACTORY=0): replica il vecchio RTU solo SP.
• Factory Mode (FACTORY=1): aggiunge realisticità con AN (TypeID 13) e CT (TypeID 15).
"""

import os, sys, json, asyncio, logging, math, random, signal
from dataclasses import dataclass
import c104

# ─── Config da env-var ─────────────────────────
DATASET = os.getenv("DATASET", "/data/dataset.json")
HOST_IP = os.getenv("HOST_IP", "0.0.0.0")
PORT    = int(os.getenv("PORT", "2404"))
CA      = int(os.getenv("CA", "1"))
FACTORY = os.getenv("FACTORY", "0") == "1"

# ─── Logger ─────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, stream=sys.stdout
)
log = logging.getLogger("SubstationRTU")
mode = "Factory Mode" if FACTORY else "Compatibility Mode"
log.info("Avviato in modalità %s (CA=%d) su %s:%d", mode, CA, HOST_IP, PORT)

# ─── Data model ─────────────────────────────────
@dataclass
class SubstationPointConfig:
    ioa: int
    ptype: str
    init: float|int
    toggle_sec: int = 10
    ampl: float = 1.0
    period_sec: int = 60
    inc: int = 1

    @staticmethod
    def from_dict(d: dict) -> "SubstationPointConfig":
        return SubstationPointConfig(
            ioa=d["ioa"],
            ptype=d.get("type", "SP").upper(),
            init=d.get("init", 0),
            toggle_sec=d.get("toggle_sec", 10),
            ampl=d.get("ampl", 1.0),
            period_sec=d.get("period_sec", 60),
            inc=d.get("inc", 1),
        )

# ─── Carica dataset ──────────────────────────────
try:
    with open(DATASET) as f:
        raw_cfg = json.load(f)
except Exception as e:
    log.error("Impossibile leggere %s: %s", DATASET, e)
    sys.exit(1)
points_cfg = [SubstationPointConfig.from_dict(r) for r in raw_cfg]

# ─── Mappa TypeID IEC-104 ───────────────────────
TYPE_MAP = {
    "SP": c104.Type.M_SP_NA_1,
    "AN": c104.Type.M_ME_NC_1,
    "CT": c104.Type.M_IT_NA_1,
}

# ─── Server & station (globale) ─────────────────
srv     = c104.Server(ip=HOST_IP, port=PORT)
station = srv.add_station(common_address=CA)

# ─── Simulator tasks ────────────────────────────
async def simulate_sp(pt, sec):
    jitter = 0.05
    while True:
        await asyncio.sleep(sec * random.uniform(1-jitter, 1+jitter))
        pt.value = not pt.value
        log.info("[SP] ioa=%d → %d", pt.io_address, int(pt.value))

async def simulate_analog(pt, base, ampl, period):
    t = 0
    while True:
        await asyncio.sleep(1)
        val = base + ampl * math.sin(2*math.pi*t/period) + random.gauss(0, 0.2)
        pt.value = val
        if t % 5 == 0:
            log.info("[AN] ioa=%d → %.2f", pt.io_address, val)
        t += 1

async def simulate_counter(pt, inc, period):
    jitter = 0.05
    while True:
        await asyncio.sleep(period * random.uniform(1-jitter, 1+jitter))
        pt.value += inc
        log.info("[CT] ioa=%d → %d", pt.io_address, pt.value)

async def legacy_spontaneous(points):
    while True:
        ioa, pt = random.choice(list(points.items()))
        pt.value = not pt.value
        log.info("[LEG] ioa=%d → %d", ioa, int(pt.value))
        await asyncio.sleep(random.randint(5, 10))

# ─── Main ────────────────────────────────────────
async def main():
    tasks: list[asyncio.Task] = []
    points: dict[int, c104.Point] = {}

    for cfg in points_cfg:
        t = cfg.ptype
        if t not in TYPE_MAP:
            log.warning("Tipo %s non supportato, skip IOA %d", t, cfg.ioa)
            continue

        pt = station.add_point(io_address=cfg.ioa, type=TYPE_MAP[t])
        if t == "SP":
            pt.value = bool(cfg.init)
        else:
            pt.value = cfg.init
        points[cfg.ioa] = pt
        log.info("init ioa=%d → %s", cfg.ioa, cfg.init)

        if FACTORY:
            if t == "SP":
                tasks.append(asyncio.create_task(simulate_sp(pt, cfg.toggle_sec)))
            elif t == "AN":
                tasks.append(asyncio.create_task(simulate_analog(pt, cfg.init, cfg.ampl, cfg.period_sec)))
            elif t == "CT":
                tasks.append(asyncio.create_task(simulate_counter(pt, cfg.inc, cfg.period_sec)))

    srv.start()
    log.info("Server IEC-104 in ascolto su %s:%d (CA=%d)", HOST_IP, PORT, CA)

    if FACTORY:
        await asyncio.gather(*tasks)
    else:
        await legacy_spontaneous(points)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(main())
    finally:
        srv.stop()
        log.info("Arresto completato")
