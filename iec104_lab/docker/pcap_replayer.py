#!/usr/bin/env python3
"""
Replay automatico di tutti i file .pcap presenti (ricerca ricorsiva)
nella directory /data.  Mantiene gli intervalli temporali originali
(adattabili con la variabile d’ambiente DELAY_FACTOR).

• Ordine di riproduzione  : alfabetico per percorso completo
• Interfaccia di uscita   : IFACE            (default: eth0)
• Fattore di dilatazione  : DELAY_FACTOR     (default: 1.0 – tempi reali)

Richiede Scapy ≥ 2.6 (gestisce i timestamp come EDecimal).
"""

import os
import sys
import time
from pathlib import Path

from scapy.all import PcapReader, sendp  # type: ignore

# ---------------------------------------------------------------------------
# Parametri runtime
# ---------------------------------------------------------------------------
DATA_DIR = Path("/data")                             # volume montato
DELAY     = float(os.getenv("DELAY_FACTOR", "1.0"))  # scala dei ritardi
IFACE     = os.getenv("IFACE", "eth0")               # output interface

# ---------------------------------------------------------------------------
# Funzioni
# ---------------------------------------------------------------------------
def replay_file(pcap_path: Path) -> None:
    """Riproduce un singolo file pcap rispettandone i timing."""
    with PcapReader(str(pcap_path)) as reader:
        print(f"[+] {pcap_path}  –  replay in corso…")
        prev_ts: float | None = None

        for pkt in reader:
            ts = float(getattr(pkt, "time", 0.0))  # cast da EDecimal → float
            if prev_ts is not None:
                pause = max(0.0, (ts - prev_ts) * DELAY)
                time.sleep(pause)
            prev_ts = ts
            sendp(pkt, iface=IFACE, verbose=False)

    print(f"[✔] Replay completato ({pcap_path})\n")


def main() -> None:
    pcap_files = sorted(DATA_DIR.rglob("*.pcap"))

    if not pcap_files:
        print("[!] Nessun file .pcap trovato in /data – uscita.")
        sys.exit(1)

    print(f"[★] Avvio replay automatico di {len(pcap_files)} file  (delay {DELAY})\n")

    for pcap in pcap_files:
        replay_file(pcap)

    print("[🏁] Tutti i dataset sono stati riprodotti con successo.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[⏹] Interrotto dall’utente")
        sys.exit(130)
