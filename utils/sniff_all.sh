#!/usr/bin/env bash
#
# Cattura traffico IEC-104 (porta 2404) da tutti i container
# la cui *immagine* è iec104_lab_master:latest oppure iec104_lab_rtu_*:latest.
# Non serve conoscere i nomi runtime (clever_snyder, …).

set -euo pipefail

# ── trova i container da monitorare ───────────────────────────────────────
readarray -t containers < <(
  docker ps --format '{{.Names}} {{.Image}}' |
  awk '$2 ~ /^iec104_lab-(master|rtu_)/ {print $1}'
)

if [[ ${#containers[@]} -eq 0 ]]; then
  echo "❌ Nessun container IEC-104 in esecuzione."; exit 1
fi

mkdir -p pcaps
timestamp=$(date +%Y%m%d_%H%M%S)

echo "▶️  Avvio tcpdump su ${#containers[@]} container…"
for c in "${containers[@]}"; do
  docker exec -d "$c" \
    tcpdump -i eth0 -w "/tmp/${c}_${timestamp}.pcap" tcp port 2404
done

echo
read -rp "🟢 Cattura in corso: premi INVIO per interrompere…"

echo "⏹️  Interrompo tcpdump…"
for c in "${containers[@]}"; do
  docker exec "$c" pkill tcpdump || true
done

echo "⬇️  Copio i file .pcap sull'host…"
for c in "${containers[@]}"; do
  docker cp "$c":"/tmp/${c}_${timestamp}.pcap" pcaps/ 2>/dev/null || true
done

echo "✅ PCAP salvati in ./pcaps"