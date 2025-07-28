#!/bin/sh
set -e

# Alza le interfacce (necessario in GNS3)
ip link set eth0 up
ip link set eth1 up

echo "✔ Bringing up eth0 and eth1..."

echo "✔ Starting Suricata in live af-packet mode..."
exec suricata -c /etc/suricata/suricata.yaml \
             --af-packet=eth0 \
             --af-packet=eth1 \
             ${SURICATA_OPTIONS:-}