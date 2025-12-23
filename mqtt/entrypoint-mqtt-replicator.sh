#!/usr/bin/env bash
set -euo pipefail

echo "[*] Configurazione rete…"
ip link set eth0 up
ip link set eth1 up
IP_IFACE="${IP_IFACE:-192.168.122.48/24}"
IP_REPLICATION_IFACE="${IP_REPLICATION_IFACE:-10.0.0.88/24}"
ip addr add "$IP_IFACE" dev eth0
ip addr add "$IP_REPLICATION_IFACE" dev eth1

ip link set dev eth1 promisc on

echo "[*] Network configured:"
echo "[✔] Rete pronta (eth0 → $IP_IFACE)."
echo "[✔] Rete pronta (eth1 → $IP_REPLICATION_IFACE)."

python3 mqtt_replay.py