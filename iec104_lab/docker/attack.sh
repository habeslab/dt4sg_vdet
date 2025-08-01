#!/usr/bin/env bash
set -euo pipefail

# ====== parametri per la IDS test-suite ======
RTU_LIST=(10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.18)
PORT=2404
SCADA_SRC="$(ip -4 -o addr show eth0 | awk '{print $4}' | cut -d/ -f1)"
EXT_SRC="1.2.3.4"   # spoof

# ====== menu ======
echo "========================================================="
echo " 🚀  ATTACK MENU "
echo "========================================================="
echo "1) IDS test-suite  (lateral, ext-intrusion, SYN-flood)"
echo "2) Replay AUTOMATICO di tutti i PCAP in /data"
echo "Q) Solo esci"
echo "========================================================="
read -rp "Scelta (1/2/Q): " CHOICE
echo

case "$CHOICE" in
  1)
    echo "[*] Avvio IDS test-suite…"

    echo "[1200101] Lateral movement"
    for r in "${RTU_LIST[@]}"; do nc -z -w1 "$r" $PORT || true; done

    echo "[1200201] External intrusion (spoof $EXT_SRC)"
    for r in "${RTU_LIST[@]}"; do hping3 -c1 -S -p $PORT -a "$EXT_SRC" "$r" >/dev/null 2>&1 || true; done

    echo "[1000101] SYN-flood"
    hping3 -c25 -i u30000 -S -p $PORT "${RTU_LIST[0]}" >/dev/null 2>&1 || true

    echo "[✔] IDS test-suite terminata."
    ;;
  2)
    echo "[*] Replay di tutti i dataset in /data…"
    python3 /usr/local/bin/pcap_replayer.py
    ;;
  [Qq])
    echo "[*] Nessuna azione eseguita."
    ;;
  *)
    echo "[!] Scelta non valida."
    ;;
esac
