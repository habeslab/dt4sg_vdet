#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

from scapy.all import (
    PcapReader, sendp,
    Ether, IP, TCP, UDP,
    get_if_hwaddr, get_if_addr,
    ARP, srp1
)  # type: ignore

DATA_DIR = Path("/data")
DELAY    = float(os.getenv("DELAY_FACTOR", "1.0"))
IFACE    = os.getenv("IFACE", "eth0")

TARGET_IP  = os.getenv("TARGET_IP", "10.0.0.50").strip()
TARGET_MAC = os.getenv("TARGET_MAC", "ff:ff:ff:ff:ff:ff").strip()

SRC_IP  = os.getenv("SRC_IP", "10.0.0.99").strip()
SRC_MAC = os.getenv("SRC_MAC", "").strip()

KEEP_PORTS = os.getenv("KEEP_PORTS", "1").strip() != "0"

def _arp_resolve(ip: str) -> str | None:
    """Risolvi MAC con ARP sulla IFACE (richiede L2 reachability)."""
    try:
        req = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip)
        ans = srp1(req, iface=IFACE, timeout=2, verbose=False)
        if ans and ans.haslayer(ARP):
            return ans[ARP].hwsrc
    except Exception:
        pass
    return None

def _ensure_addrs() -> tuple[str, str, str]:
    """Ritorna (src_ip, src_mac, dst_mac)."""
    if not TARGET_IP:
        print("[!] Devi impostare TARGET_IP (es. 10.0.0.10).")
        sys.exit(2)

    src_ip = SRC_IP or get_if_addr(IFACE)
    src_mac = SRC_MAC or get_if_hwaddr(IFACE)

    dst_mac = TARGET_MAC or _arp_resolve(TARGET_IP)
    if not dst_mac:
        print("[!] TARGET_MAC non fornito e ARP fallito. Imposta TARGET_MAC esplicitamente.")
        sys.exit(3)

    return src_ip, src_mac, dst_mac

def _rewrite_packet(pkt, src_ip: str, src_mac: str, dst_ip: str, dst_mac: str):
    """Riscrive L2/L3 e ricalcola checksum/len."""
    if not pkt.haslayer(Ether) or not pkt.haslayer(IP):
        return None

    p = pkt.copy()

    # L2
    p[Ether].src = src_mac
    p[Ether].dst = dst_mac

    # L3
    p[IP].src = src_ip
    p[IP].dst = dst_ip

    # Opzionale: mantieni/riscrivi porte
    if not KEEP_PORTS:
        if p.haslayer(TCP):
            # esempio: forza dport IEC-104
            p[TCP].dport = 2404
        elif p.haslayer(UDP):
            # se serve
            pass

    # Forza ricalcolo checksum/len (Scapy li rigenera se cancellati)
    try:
        del p[IP].chksum
    except Exception:
        pass
    try:
        del p[IP].len
    except Exception:
        pass
    if p.haslayer(TCP):
        try:
            del p[TCP].chksum
        except Exception:
            pass
    if p.haslayer(UDP):
        try:
            del p[UDP].chksum
        except Exception:
            pass

    return p

def replay_file(pcap_path: Path, src_ip: str, src_mac: str, dst_ip: str, dst_mac: str) -> None:
    with PcapReader(str(pcap_path)) as reader:
        print(f"[+] {pcap_path}  –  replay verso {dst_ip} (src={src_ip}) …")
        prev_ts: float | None = None

        for pkt in reader:
            ts = float(getattr(pkt, "time", 0.0))
            if prev_ts is not None:
                pause = max(0.0, (ts - prev_ts) * DELAY)
                time.sleep(pause)
            prev_ts = ts

            newp = _rewrite_packet(pkt, src_ip, src_mac, dst_ip, dst_mac)
            if newp is None:
                continue

            sendp(newp, iface=IFACE, verbose=False)

    print(f"[✔] Replay completato ({pcap_path})\n")

def main() -> None:
    src_ip, src_mac, dst_mac = _ensure_addrs()

    pcap_files = sorted(DATA_DIR.rglob("*.pcap"))
    if not pcap_files:
        print("[!] Nessun file .pcap trovato in /data – uscita.")
        sys.exit(1)

    print(f"[★] Replay di {len(pcap_files)} file (delay {DELAY})")
    print(f"    IFACE={IFACE}")
    print(f"    SRC={src_ip} / {src_mac}")
    print(f"    DST={TARGET_IP} / {dst_mac}")
    print(f"    KEEP_PORTS={int(KEEP_PORTS)}\n")

    for pcap in pcap_files:
        replay_file(pcap, src_ip, src_mac, TARGET_IP, dst_mac)

    print("[🏁] Tutti i dataset sono stati riprodotti con successo.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[⏹] Interrotto dall’utente")
        sys.exit(130)
