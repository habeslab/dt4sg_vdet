#!/usr/bin/env python3
"""
Master IEC-104 con libreria «c104».
• Si collega agli RTU indicati in $PEERS
• Quando la connessione diventa OPEN invia la GI
• Stampa ogni valore ricevuto
"""
import os, time, threading, c104
from c104 import (Client, ResponseState, Station, Coi,
                  ConnectionState, Type, Connection, IncomingMessage)

PEERS = os.getenv("PEERS", "").split()          
POLL_SEC = int(os.getenv("POLL_SEC", "30"))
if not PEERS:
    raise SystemExit("PEERS vuoto (es: PEERS='10.0.0.12 10.0.0.13')")

print(f"[MASTER] avviato – PEERS={PEERS}")

CA_MAP = {          
    "10.0.0.12": 1,   
    "10.0.0.13": 2   
}

client = Client(tick_rate_ms=100)

# ─── Callback corretto, usando point.station.connection.ip ───
def _print_point(point: c104.Point,
                 previous_info: c104.Information,
                 message: IncomingMessage) -> ResponseState:
    """Callback invocato su ogni informazione in arrivo."""
    conn = point.station.connection
    source_ip = conn.ip if conn else "unknown"
    print(f"[MASTER] {source_ip} ioa={point.io_address} → {int(point.value)}")
    return ResponseState.SUCCESS  


def on_station_initialized(client: Client, station: Station, cause: Coi) -> None:
    for p in station.points.values():
        p.on_receive(callable=_print_point)

client.on_station_initialized(on_station_initialized)

def on_new_point(client: Client, station: Station,
                 io_address: int, point_type: Type) -> None:
    p = station.add_point(io_address=io_address, type=point_type)
    p.on_receive(callable=_print_point)

client.on_new_point(on_new_point)

# ─── connessione + polling ───
def connect_and_poll(ip: str):
    ca = CA_MAP.get(ip, 1)                         
    con = client.add_connection(ip=ip, port=2404, init=c104.Init.NONE)

    def on_state_change(connection: Connection, state: ConnectionState) -> None:
        if state == ConnectionState.OPEN:
            print(f"[MASTER] {ip} OPEN – GI CA={ca}")
            connection.interrogation(common_address=ca)

    con.on_state_change(callable=on_state_change)
    con.connect()

    while True:
        con.interrogation(common_address=ca)
        time.sleep(POLL_SEC)

client.start()
for ip in PEERS:
    threading.Thread(target=connect_and_poll, args=(ip,), daemon=True).start()

while True:
    time.sleep(3600)
