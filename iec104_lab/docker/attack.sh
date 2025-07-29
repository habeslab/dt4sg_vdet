#!/usr/bin/env bash
set -euo pipefail

RTU_LIST=(10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.18)
PORT=2404

SCADA_LATERAL_SRC="$(ip -4 -o addr show eth0 | awk '{print $4}' | cut -d/ -f1)"   # es: 10.0.0.99
EXTERNAL_SRC="1.2.3.4"          # spoof: NON nella /24 SCADA

echo "[*] Starting IDS test suite …"

################### SID 1200101  –  lateral movement ###################
echo "[*] 1200101  ▶ Lateral movement (1 SYN per RTU)"
for rtu in "${RTU_LIST[@]}"; do
  nc -z -w1 "$rtu" $PORT || true
done

################### SID 1200201  –  ext. intrusion #####################
echo "[*] 1200201  ▶ External intrusion (1 SYN spoof da $EXTERNAL_SRC)"
for rtu in "${RTU_LIST[@]}"; do
  hping3 -c 1 -S -p $PORT -a "$EXTERNAL_SRC" "$rtu" >/dev/null 2>&1 || true
done

################### SID 1000101  –  SYN-flood ##########################
echo "[*] 1000101  ▶ SYN-flood (≥20 SYN in 1 s → ${RTU_LIST[0]})"
hping3 -c 25 -i u30000 -S -p $PORT "${RTU_LIST[0]}" >/dev/null 2>&1 || true
# 25 pkt • 30 000 µs ≈ 0,75 s  → rate ≈33 SYN/s (sopra soglia 20)

################### SID 1100101  –  ASDU 100 craft #####################
echo "[*] 1100101  ▶ Crafting ASDU 100 senza handshake"
python3 - <<'PY'
from scapy.all import *
RTU="10.0.0.12"; PORT=2404
SRC="10.0.0.99"
SEQ=12345
syn  = IP(src=SRC, dst=RTU)/TCP(sport=RandShort(), dport=PORT, flags="S", seq=SEQ)
synack = sr1(syn, timeout=1, verbose=0)
if synack:
    ack = IP(src=SRC, dst=RTU)/TCP(sport=syn[TCP].sport, dport=PORT,
                                   flags="A", seq=SEQ+1, ack=synack.seq+1)
    send(ack/Raw(b"\x64\x00"), verbose=0)
else:
    print("[!] SYN bloccato: 1100101 non può essere testata (1200101 attiva?)")
PY

echo "[✔] Tests completed – check Suricata logs."
