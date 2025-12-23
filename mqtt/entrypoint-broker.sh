#!/bin/sh
set -e pipefail

echo "[MQTT] Starting Mosquitto broker"
echo "[MQTT] Config: /mosquitto/config/mosquitto.conf"

echo "[*] Configurazione rete…"
ip link set eth0 up
IP_BROKER="${IP_BROKER:-192.168.122.49/24}"
ip addr add "$IP_BROKER" dev eth0
echo "[*] Network configured:"
echo "[✔] Rete pronta (eth0 → $IP_BROKER)."

exec mosquitto -c /mosquitto/config/mosquitto.conf
