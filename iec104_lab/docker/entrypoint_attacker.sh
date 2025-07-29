#!/usr/bin/env bash
set -euo pipefail

echo "[*] Configuring network interfaces..."

# alza sempre eth0; ignora eth1 se non c'è
ip link set eth0 up
ip link set eth1 up 2>/dev/null || true

# IP da variabile o default 10.0.0.99/24
ATTACKER_IP="${ATTACKER_IP:-10.0.0.99/24}"
ip addr add "$ATTACKER_IP" dev eth0

echo "[*] Network configured:"
ip addr show eth0

# lancia l'attacco e poi rimani in shell per ispezione
/attack.sh || true
exec /bin/bash
