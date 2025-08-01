#!/usr/bin/env python3
"""
IEC-104 Master – compat. c104-python 2.2.1
• General Interrogation all’apertura
• TESTFR ogni 25 s (senza time-tag, Type 104)
"""
import os, time, threading, logging, c104

# ── configurazione ───────────────────────────────────────────
PEERS = os.getenv(
    "PEERS",
    "10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.18"
).split()
if not PEERS:
    raise SystemExit("PEERS is empty")

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
# opzionale: silenzia eventuali warn interni della libreria
logging.getLogger("c104.IncomingMessage").setLevel(logging.ERROR)

log = logging.getLogger("MASTER")

CA_MAP = {ip: i + 1 for i, ip in enumerate(PEERS)}     # CA 1-7
client  = c104.Client(tick_rate_ms=100)

# ── utilità --------------------------------------------------
def pretty(pt: c104.Point):
    if pt.type.name.startswith("M_SP"):
        return "ON" if pt.value else "OFF"
    if pt.type.name.startswith("M_ME"):
        return f"{pt.value:.2f}"
    return pt.value

def on_point(point:        c104.Point,
             previous_info: c104.Information,
             message:       c104.IncomingMessage) -> c104.ResponseState:
    ca   = point.station.common_address
    conn = point.station.connection
    log.info("CA=%d IOA=%d %s → %s",
             ca, point.io_address,
             conn.ip if conn else "-", pretty(point))
    return c104.ResponseState.SUCCESS

# ── keep-alive TESTFR (senza time-tag) ----------------------
def keepalive(con: c104.Connection, ca: int):
    while con.state == c104.ConnectionState.OPEN:
        time.sleep(25)
        try:
            con.test(common_address=ca, with_time=False,  
                     wait_for_response=False)
        except Exception as exc:
            log.warning("TESTFR %s: %s", con.ip, exc)

# ── stato connessione --------------------------------------
def handle_state(connection: c104.Connection,
                 state:      c104.ConnectionState) -> None:
    if state == c104.ConnectionState.OPEN:
        ca = CA_MAP.get(connection.ip, 0)
        log.info("%s OPEN – GI CA=%d", connection.ip, ca)
        connection.interrogation(common_address=ca)
        threading.Thread(target=keepalive,
                         args=(connection, ca),
                         daemon=True).start()

# ── init stazione & punti dinamici -------------------------
def on_station_initialized(client:  c104.Client,
                           station: c104.Station,
                           cause:   c104.Coi) -> None:
    for p in station.points.values():
        p.on_receive(callable=on_point)
client.on_station_initialized(on_station_initialized)

def on_new_point(client:    c104.Client,
                 station:   c104.Station,
                 io_address:int,
                 point_type:c104.Type) -> None:
    p = station.add_point(io_address=io_address, type=point_type)
    p.on_receive(callable=on_point)
client.on_new_point(on_new_point)

# ── apertura connessioni -----------------------------------
for ip in PEERS:
    con = client.add_connection(ip=ip, port=2404, init=c104.Init.NONE)
    con.on_state_change(handle_state)
    con.connect()

client.start()

try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    client.stop()
    log.info("Master stopped")
