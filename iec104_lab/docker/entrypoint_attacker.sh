#!/usr/bin/env bash
set -euo pipefail

echo "[*] Configurazione rete…"
ip link set eth0 up
ip link set eth1 up 2>/dev/null || true
ATTACKER_IP="${ATTACKER_IP:-10.0.0.99/24}"
ip addr add "$ATTACKER_IP" dev eth0
echo "[*] Network configured:"
echo "[✔] Rete pronta (eth0 → $ATTACKER_IP). Usa /attack.sh per avviare gli attacchi."

exec /bin/bash       


