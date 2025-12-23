#!/usr/bin/env bash
set -euo pipefail

echo "[*] Configurazione rete…"
ip link set eth0 up
IP_NODE="${IP_NODE:-192.168.122.40/24}"
ip addr add "$IP_NODE" dev eth0

ip link set dev eth0 promisc on

echo "[*] Network configured:"
echo "[✔] Rete pronta (eth0 → $IP_NODE)."

python3 multi_phy_node.py