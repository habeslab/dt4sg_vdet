#!/bin/sh
set -e

echo "[ENTRYPOINT] Avvio entrypoint.sh"
IFACE="${IFACE:-eth0}"
echo "[ENTRYPOINT] Interfaccia: $IFACE"
echo "[ENTRYPOINT] IP da assegnare: $HOST_IP"

if [ -n "$HOST_IP" ]; then
  if ! ip addr show "$IFACE" | grep -q "$HOST_IP"; then
    ip addr add "$HOST_IP/24" dev "$IFACE"
  fi
  ip link set "$IFACE" up 2>/dev/null || true
fi

echo "[ENTRYPOINT] Eseguo comando finale: $@"
exec "$@"
