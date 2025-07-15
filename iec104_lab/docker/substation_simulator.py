#!/usr/bin/env python3
"""
RTU IEC-104 – compat. c104-python 2.2.1
• invio spontaneo solo se il valore cambia oltre la soglia ±2 %
• nessun Test-Frame con time-tag (Type 107)
"""
import os, sys, json, asyncio, logging, math, random, signal
from dataclasses import dataclass
import c104

# ── configurazione --------------------------------------------------------
DATASET  = os.getenv("DATASET", "/data/dataset.json")
HOST_IP  = os.getenv("HOST_IP", "0.0.0.0")
PORT     = int(os.getenv("PORT", "2404"))
CA       = int(os.getenv("CA", "1"))
THR      = float(os.getenv("THRESHOLD", "0.02"))
FACTORY  = os.getenv("FACTORY", "0") == "1"

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("RTU")
log.info("RTU CA=%d – %s mode", CA, "Factory" if FACTORY else "Compat")

# ── dataset ---------------------------------------------------------------
@dataclass
class PointCfg:
    ioa: int; ptype: str; init: float | int
    toggle: int = 10; ampl: float = 1.0; period: int = 60; inc: int = 1

    @staticmethod
    def from_dict(d: dict) -> "PointCfg":
        return PointCfg(
            ioa=d["ioa"],
            ptype=d.get("type", "SP").upper(),
            init=d.get("init", 0),
            toggle=d.get("toggle_sec", 10),
            ampl=d.get("ampl", 1.0),
            period=d.get("period_sec", 60),
            inc=d.get("inc", 1),
        )

with open(DATASET) as f:
    cfg_list = [PointCfg.from_dict(r) for r in json.load(f)]

TYPE = {"SP": c104.Type.M_SP_NA_1,
        "AN": c104.Type.M_ME_NC_1,
        "CT": c104.Type.M_IT_NA_1}

# ── server/stazione -------------------------------------------------------
srv = c104.Server(ip=HOST_IP, port=PORT)
stn = srv.add_station(common_address=CA)

def push(pt: c104.Point):
    pt.transmit(cause=c104.Cot.SPONTANEOUS)

def changed(old, new):
    if isinstance(old.value, (int, float)):
        return abs(old.value - new) > THR * max(abs(old.value), 1)
    return old.value != new

# generatori ---------------------------------------------------------------
async def sim_sp(pt, sec):
    while True:
        await asyncio.sleep(sec)
        new = not pt.value
        if changed(pt, new):
            pt.value = new
            push(pt)

async def sim_an(pt, base, ampl, period):
    t = 0
    while True:
        await asyncio.sleep(1)
        new = base + ampl * math.sin(2 * math.pi * t / period)
        if changed(pt, new):
            pt.value = new + random.gauss(0, ampl * 0.01)
            push(pt)
        t += 1

async def sim_ct(pt, inc, period):
    while True:
        await asyncio.sleep(period)
        pt.value += inc
        push(pt)

async def legacy_toggle(points):
    while True:
        ioa, pt = random.choice(list(points.items()))
        pt.value = not pt.value
        push(pt)
        await asyncio.sleep(random.randint(5, 10))

# main ---------------------------------------------------------------------
async def main():
    tasks, points = [], {}
    for cfg in cfg_list:
        if cfg.ptype not in TYPE:
            continue
        pt = stn.add_point(io_address=cfg.ioa, type=TYPE[cfg.ptype])
        pt.value = bool(cfg.init) if cfg.ptype == "SP" else cfg.init
        points[cfg.ioa] = pt

        if FACTORY:
            if cfg.ptype == "SP":
                tasks.append(asyncio.create_task(sim_sp(pt, cfg.toggle)))
            elif cfg.ptype == "AN":
                tasks.append(asyncio.create_task(sim_an(
                    pt, cfg.init, cfg.ampl, cfg.period)))
            elif cfg.ptype == "CT":
                tasks.append(asyncio.create_task(sim_ct(
                    pt, cfg.inc, cfg.period)))

    srv.start()
    log.info("Listening %s:%d", HOST_IP, PORT)
    if FACTORY:
        await asyncio.gather(*tasks)
    else:
        await legacy_toggle(points)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(main())
    finally:
        srv.stop(); log.info("Stopped")
