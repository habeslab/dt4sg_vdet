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

# ─── Configurazione RTU ─────────────────────────────────────────
# Lista di peer predefinita (modifica includendo nuovi RTU)
PEERS = os.getenv(
    "PEERS",
    "10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.18"
).split()
POLL_SEC = int(os.getenv("POLL_SEC", "30"))
if not PEERS:
    raise SystemExit("PEERS vuoto (es: PEERS='10.0.0.12 10.0.0.13 ...')")

print(f"[MASTER] avviato – PEERS={PEERS}")

# Mappa Common Address (CA) per ogni RTU IP
CA_MAP = {
    "10.0.0.12": 1,   # RTU_Power
    "10.0.0.13": 2,   # RTU_Factory
    "10.0.0.14": 3,   # RTU_Suburb
    "10.0.0.15": 4,   # RTU_Solar
    "10.0.0.16": 5,   # RTU_Wind
    "10.0.0.17": 6,   # RTU_Industry
    "10.0.0.18": 7    # RTU_EV
}

# ─── Inizializza client IEC-104 ─────────────────────────────────
client = Client(tick_rate_ms=100)

# Callback per ogni punto ricevuto
def _print_point(point: c104.Point,
                 previous_info: c104.Information,
                 message: IncomingMessage) -> ResponseState:
    conn = point.station.connection
    source_ip = conn.ip if conn else "unknown"
    print(
        f"[MASTER] {source_ip} ioa={point.io_address} "
        f"→ {int(point.value)}"
    )
    return ResponseState.SUCCESS

# Registrazione callback all'avvio di ogni station
def on_station_initialized(client: Client, station: Station, cause: Coi) -> None:
    for p in station.points.values():
        p.on_receive(callable=_print_point)
client.on_station_initialized(on_station_initialized)

# Callback per nuovi punti dinamici
def on_new_point(client: Client, station: Station,
                 io_address: int, point_type: Type) -> None:
    p = station.add_point(io_address=io_address, type=point_type)
    p.on_receive(callable=_print_point)
client.on_new_point(on_new_point)

# ─── Connessione e polling ciclico ───────────────────────────────
def connect_and_poll(ip: str):
    ca = CA_MAP.get(ip, 1)
    con = client.add_connection(ip=ip, port=2404, init=c104.Init.NONE)

    # Invia General Interrogation all'apertura della connessione
    def on_state_change(connection: Connection, state: ConnectionState) -> None:
        if state == ConnectionState.OPEN:
            print(f"[MASTER] {ip} OPEN – GI CA={ca}")
            connection.interrogation(common_address=ca)
    con.on_state_change(callable=on_state_change)

    con.connect()
    # Polling ciclico
    while True:
        con.interrogation(common_address=ca)
        time.sleep(POLL_SEC)

# Avvia client e thread di connessione per ciascun RTU
client.start()
for ip in PEERS:
    threading.Thread(
        target=connect_and_poll,
        args=(ip,),
        daemon=True
    ).start()

# Mantieni il processo attivo
while True:
    time.sleep(3600)
