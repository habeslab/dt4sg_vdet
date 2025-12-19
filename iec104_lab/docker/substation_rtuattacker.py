#!/usr/bin/env python3
"""
IEC-104 RTU "fault-injection" simulator (application-level integrity faults)
Updated dataset: SP (401–408) + CT (501–506)

- Simulates SP points and CT (integrated totals) points using c104-python style API.
- Sends SPONTANEOUS transmissions.
- Integrity faults are applied BEFORE transmit (no MITM/sniffing).

Point types:
  - SP -> M_SP_NA_1 (single point)
  - CT -> M_IT_NA_1 (integrated totals / counter)

Fault types supported:
  - freeze (SP/CT): hold last "sent" value during window
  - flip   (SP): invert boolean (or force to 0/1)
  - bias   (CT): value += bias (useful to simulate counter tampering)
  - clamp  (CT): clamp to [min, max]
  - drift  (CT): value += slope * (t - t_start) (slow manipulation)

ENV:
  HOST_IP=0.0.0.0
  PORT=2404
  CA=1
  AUDIT_LOG=/data/rtu_audit.jsonl         # optional, leave empty to disable
  ATTACKS_JSON=/data/attacks.json         # optional path
  ATTACKS='[...]'                         # optional inline JSON (overrides file if present)

Attacks JSON format (list of rules):
[
  {"name":"flip_sp_402","type":"flip","targets":[402],"start_sec":0,"end_sec":999999,"params":{"mode":"invert"}},
  {"name":"freeze_ct_503","type":"freeze","targets":[503],"start_sec":120,"end_sec":240},
  {"name":"bias_ct_501","type":"bias","targets":[501],"start_sec":200,"end_sec":400,"params":{"bias":-500}},
  {"name":"clamp_ct_502","type":"clamp","targets":[502],"start_sec":0,"end_sec":999999,"params":{"min":7800,"max":9000}},
  {"name":"drift_ct_506","type":"drift","targets":[506],"start_sec":100,"end_sec":500,"params":{"slope":2.0}}
]
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union

import c104  # type: ignore


# ── logging ────────────────────────────────────────────────────────────────
LOGLEVEL = os.getenv("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("RTU-FI")


# ── env config ────────────────────────────────────────────────────────────
HOST_IP = os.getenv("HOST_IP", "0.0.0.0")
PORT = int(os.getenv("PORT", "2404"))
CA = int(os.getenv("CA", "1"))
AUDIT_LOG = os.getenv("AUDIT_LOG", "")  # e.g. /data/rtu_audit.jsonl, leave empty to disable

ATTACKS_JSON = os.getenv("ATTACKS_JSON", "")
ATTACKS_INLINE = os.getenv("ATTACKS", "")


# ── dataset (hardcoded per tua richiesta) ─────────────────────────────────
SP_POINTS = [
    {"ioa": 401, "type": "SP", "init": 0, "toggle_sec": 20},
    {"ioa": 402, "type": "SP", "init": 1, "toggle_sec": 25},
    {"ioa": 403, "type": "SP", "init": 0, "toggle_sec": 30},
    {"ioa": 404, "type": "SP", "init": 1, "toggle_sec": 35},
    {"ioa": 405, "type": "SP", "init": 0, "toggle_sec": 40},
    {"ioa": 406, "type": "SP", "init": 1, "toggle_sec": 45},
    {"ioa": 407, "type": "SP", "init": 0, "toggle_sec": 50},
    {"ioa": 408, "type": "SP", "init": 1, "toggle_sec": 55},
]
CT_POINTS = [
    {"ioa": 501, "type": "CT", "init": 10000, "inc": 100, "period_sec": 60},
    {"ioa": 502, "type": "CT", "init": 8000, "inc": 80, "period_sec": 45},
    {"ioa": 503, "type": "CT", "init": 6000, "inc": 60, "period_sec": 30},
    {"ioa": 504, "type": "CT", "init": 4000, "inc": 40, "period_sec": 90},
    {"ioa": 505, "type": "CT", "init": 2000, "inc": 20, "period_sec": 120},
    {"ioa": 506, "type": "CT", "init": 500, "inc": 10, "period_sec": 150},
]


# ── IEC-104 type mapping ──────────────────────────────────────────────────
TYPE_MAP = {
    "SP": c104.Type.M_SP_NA_1,
    "CT": c104.Type.M_IT_NA_1,
}


# ── utility ───────────────────────────────────────────────────────────────
def push(pt: c104.Point) -> None:
    pt.transmit(cause=c104.Cot.SPONTANEOUS)


def _now_sec(t0: float) -> float:
    return time.time() - t0


# ── attacks model ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AttackRule:
    name: str
    atype: str                    # bias|clamp|freeze|flip|drift
    targets: List[int]            # list of IOA
    start_sec: float
    end_sec: float
    params: Dict[str, Any]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AttackRule":
        return AttackRule(
            name=str(d.get("name", d.get("type", "attack"))),
            atype=str(d["type"]).strip().lower(),
            targets=list(d.get("targets", [])),
            start_sec=float(d.get("start_sec", 0)),
            end_sec=float(d.get("end_sec", 1e18)),
            params=dict(d.get("params", {})),
        )

    def active(self, t_sec: float, ioa: int) -> bool:
        return (ioa in self.targets) and (self.start_sec <= t_sec <= self.end_sec)


class FaultInjector:
    """
    Applies application-level "integrity faults" to outgoing values.
    For freeze we keep per-IOA frozen values.
    Rules are applied in listed order.
    """
    def __init__(self, rules: List[AttackRule]) -> None:
        self.rules = rules
        self._frozen: Dict[int, Any] = {}

    def _active_rules(self, ioa: int, t_sec: float) -> List[AttackRule]:
        return [r for r in self.rules if r.active(t_sec, ioa)]

    def _clear_freeze_if_inactive(self, ioa: int, t_sec: float) -> None:
        if ioa not in self._frozen:
            return
        if not any((r.atype == "freeze" and r.active(t_sec, ioa)) for r in self.rules):
            self._frozen.pop(ioa, None)

    def apply_sp(self, ioa: int, true_value: bool, t_sec: float) -> Tuple[bool, List[str]]:
        labels: List[str] = []
        v: bool = bool(true_value)

        for r in self._active_rules(ioa, t_sec):
            labels.append(r.name)

            if r.atype == "freeze":
                if ioa not in self._frozen:
                    self._frozen[ioa] = v
                v = bool(self._frozen[ioa])

            elif r.atype == "flip":
                mode = str(r.params.get("mode", "invert")).lower()
                if mode == "force0":
                    v = False
                elif mode == "force1":
                    v = True
                else:
                    v = not v
            elif r.atype == "burst_flip":
                # durante la finestra attiva, flippa spesso: l'effetto burst lo fai nel task
                # qui puoi trattarlo come "invert sempre" (il task si occuperà di chiamare push più volte)
                v = not v


            # bias/clamp/drift not meaningful for SP; ignored.

        self._clear_freeze_if_inactive(ioa, t_sec)
        return v, labels

    def apply_ct(self, ioa: int, true_value: int, t_sec: float) -> Tuple[int, List[str]]:
        labels: List[str] = []
        v: float = float(true_value)

        for r in self._active_rules(ioa, t_sec):
            labels.append(r.name)

            if r.atype == "freeze":
                if ioa not in self._frozen:
                    self._frozen[ioa] = v
                v = float(self._frozen[ioa])

            elif r.atype == "bias":
                bias = float(r.params.get("bias", 0.0))
                v = v + bias

            elif r.atype == "clamp":
                vmin = float(r.params.get("min", -1e18))
                vmax = float(r.params.get("max", +1e18))
                if v < vmin:
                    v = vmin
                elif v > vmax:
                    v = vmax

            elif r.atype == "drift":
                slope = float(r.params.get("slope", 0.0))  # units per second
                dt = max(0.0, t_sec - r.start_sec)
                v = v + slope * dt

            elif r.atype == "rollback":
                # sottrae una quantità (non monotono)
                amount = float(r.params.get("amount", 0.0))
                v = v - abs(amount)

            elif r.atype == "reset":
                # reset a un valore (default 0)
                to = float(r.params.get("to", 0.0))
                v = to

            elif r.atype == "spike":
                # aggiunge un picco temporaneo
                delta = float(r.params.get("delta", 0.0))
                v = v + delta

            # flip not meaningful for CT; ignored.

        self._clear_freeze_if_inactive(ioa, t_sec)
        # CT is integer integrated total
        return int(round(v)), labels


# ── load attack rules ─────────────────────────────────────────────────────
def load_attacks() -> List[AttackRule]:
    raw: Optional[List[Dict[str, Any]]] = None

    if ATTACKS_INLINE.strip():
        raw = json.loads(ATTACKS_INLINE)
    elif ATTACKS_JSON.strip():
        with open(ATTACKS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)

    if not raw:
        return []

    rules: List[AttackRule] = []
    for item in raw:
        try:
            rules.append(AttackRule.from_dict(item))
        except Exception as e:
            log.warning("Skipping invalid attack rule %s: %s", item, e)
    return rules


# ── audit log ─────────────────────────────────────────────────────────────
class Audit:
    def __init__(self, path: str) -> None:
        self.path = path.strip()
        self._fp = None
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._fp = open(self.path, "a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        if not self._fp:
            return
        self._fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            if self._fp:
                self._fp.close()
        except Exception:
            pass


# ── server setup ──────────────────────────────────────────────────────────
srv = c104.Server(ip=HOST_IP, port=PORT)
stn = srv.add_station(common_address=CA)

async def sim_sp_task(
    pt: c104.Point,
    ioa: int,
    toggle_sec: int,
    injector: FaultInjector,
    audit: Audit,
    t0: float,
) -> None:
    last_sent: bool = bool(pt.value)

    # parametri burst default (se non hai regole burst, non succede nulla)
    burst_period_sec = 0.2   # flip ogni 200 ms
    burst_count = 10         # 10 flip in un burst

    while True:
        await asyncio.sleep(toggle_sec)
        t_sec = _now_sec(t0)

        true_value = not bool(pt.value)

        # applica fault standard
        attacked, labels = injector.apply_sp(ioa, true_value, t_sec)

        # se c'è una regola burst_flip attiva, genera una raffica
        is_burst = any(
            (r.atype == "burst_flip" and r.active(t_sec, ioa))
            for r in injector.rules
        )

        if is_burst:
            for i in range(burst_count):
                attacked = not last_sent
                pt.value = attacked
                push(pt)
                last_sent = attacked
                audit.write({
                    "ts": time.time(),
                    "t_sec": _now_sec(t0),
                    "ioa": ioa,
                    "ptype": "SP",
                    "true": None,
                    "sent": bool(attacked),
                    "attack_active": True,
                    "attack_labels": labels or ["burst_flip"],
                    "cause": "SPONTANEOUS",
                })
                await asyncio.sleep(burst_period_sec)
            continue

        if attacked != last_sent:
            pt.value = attacked
            push(pt)
            last_sent = attacked
            audit.write({
                "ts": time.time(),
                "t_sec": t_sec,
                "ioa": ioa,
                "ptype": "SP",
                "true": bool(true_value),
                "sent": bool(attacked),
                "attack_active": bool(labels),
                "attack_labels": labels,
                "cause": "SPONTANEOUS",
            })

async def sim_ct_task(
    pt: c104.Point,
    ioa: int,
    inc: int,
    period_sec: int,
    injector: FaultInjector,
    audit: Audit,
    t0: float,
) -> None:
    """
    True process: increments by inc every period_sec.
    Transmission: sends every increment tick (spontaneous), but value can be attacked.
    """
    true_counter: int = int(pt.value)
    last_sent: int = int(pt.value)

    while True:
        await asyncio.sleep(period_sec)
        t_sec = _now_sec(t0)

        true_counter += int(inc)
        attacked, labels = injector.apply_ct(ioa, true_counter, t_sec)

        # Send every tick (integrated totals normally update periodically)
        pt.value = int(attacked)
        push(pt)
        last_sent = int(attacked)

        audit.write({
            "ts": time.time(),
            "t_sec": t_sec,
            "ioa": ioa,
            "ptype": "CT",
            "true": int(true_counter),
            "sent": int(attacked),
            "attack_active": bool(labels),
            "attack_labels": labels,
            "cause": "SPONTANEOUS",
        })


async def main() -> None:
    t0 = time.time()
    rules = load_attacks()
    injector = FaultInjector(rules=rules)
    audit = Audit(AUDIT_LOG)

    # Create points
    points: Dict[int, c104.Point] = {}

    for d in SP_POINTS:
        ioa = int(d["ioa"])
        pt = stn.add_point(io_address=ioa, type=TYPE_MAP["SP"])
        pt.value = bool(int(d.get("init", 0)))
        points[ioa] = pt

    for d in CT_POINTS:
        ioa = int(d["ioa"])
        pt = stn.add_point(io_address=ioa, type=TYPE_MAP["CT"])
        pt.value = int(d.get("init", 0))
        points[ioa] = pt

    # Start server
    srv.start()
    log.info("IEC-104 RTU-FI listening on %s:%d (CA=%d)", HOST_IP, PORT, CA)
    if rules:
        log.info("Loaded %d attack rule(s): %s", len(rules), [r.name for r in rules])
    else:
        log.info("No attack rules loaded (benign simulation).")

    # Schedule tasks
    tasks: List[asyncio.Task] = []

    for d in SP_POINTS:
        ioa = int(d["ioa"])
        tasks.append(asyncio.create_task(
            sim_sp_task(
                pt=points[ioa],
                ioa=ioa,
                toggle_sec=int(d["toggle_sec"]),
                injector=injector,
                audit=audit,
                t0=t0,
            )
        ))

    for d in CT_POINTS:
        ioa = int(d["ioa"])
        tasks.append(asyncio.create_task(
            sim_ct_task(
                pt=points[ioa],
                ioa=ioa,
                inc=int(d["inc"]),
                period_sec=int(d["period_sec"]),
                injector=injector,
                audit=audit,
                t0=t0,
            )
        ))

    try:
        await asyncio.gather(*tasks)
    finally:
        audit.close()
        srv.stop()
        log.info("Stopped.")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(main())
    finally:
        try:
            loop.close()
        except Exception:
            pass
