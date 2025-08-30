#!/usr/bin/env bash
set -euo pipefail

# Ordine di precedenza per assegnare IP:
# 1) Se c’è HOST_IP → usa quello (come nel tuo esempio)
# 2) Altrimenti se c’è MGMT_IP_CIDR → usa quello
# 3) Altrimenti prova DHCP → fallback statico

MGMT_IFACE="${MGMT_IFACE:-${IFACE:-eth0}}"
MGMT_DHCP="${MGMT_DHCP:-1}"
MGMT_IP_CIDR="${MGMT_IP_CIDR:-}"
HOST_IP="${HOST_IP:-}"
HOST_NETMASK="${HOST_NETMASK:-24}"
ADD_DEFAULT_ROUTE="${ADD_DEFAULT_ROUTE:-0}"
MGMT_GW="${MGMT_GW:-}"
SET_MTU="${SET_MTU:-}"

wait_if() { local i="$1"; for _ in {1..60}; do ip link show "$i" &>/dev/null && return 0; sleep 0.5; done; return 1; }

echo "[disc-api] Setup rete..."
wait_if "$MGMT_IFACE" || echo "[disc-api] WARN: interfaccia $MGMT_IFACE non trovata (continuo)"
ip link set "$MGMT_IFACE" up 2>/dev/null || true
[[ -n "$SET_MTU" ]] && ip link set dev "$MGMT_IFACE" mtu "$SET_MTU" 2>/dev/null || true

# 1) Come il TUO esempio: se c'è HOST_IP lo assegno
if [[ -n "$HOST_IP" ]]; then
  echo "[disc-api] Using HOST_IP=${HOST_IP}/${HOST_NETMASK} on ${MGMT_IFACE}"
  ip addr add "${HOST_IP}/${HOST_NETMASK}" dev "$MGMT_IFACE" 2>/dev/null || true
  MGMT_DHCP=0   # salta DHCP
  MGMT_IP_CIDR=""  # non serve
else
  # 2) Se non c'è HOST_IP ma c'è MGMT_IP_CIDR, usa quello
  if [[ -n "$MGMT_IP_CIDR" ]]; then
    echo "[disc-api] Using MGMT_IP_CIDR=${MGMT_IP_CIDR} on ${MGMT_IFACE}"
    ip addr add "$MGMT_IP_CIDR" dev "$MGMT_IFACE" 2>/dev/null || true
    MGMT_DHCP=0
  else
    # 3) DHCP → fallback statico senza default route
    if [[ "$MGMT_DHCP" = "1" ]]; then
      if command -v udhcpc >/dev/null; then
        timeout 12s udhcpc -i "$MGMT_IFACE" -q -s /usr/share/udhcpc/default.script || {
          echo "[disc-api] DHCP KO → fallback 10.0.0.20/24 (NO default route)"
          ip addr add "10.0.0.20/24" dev "$MGMT_IFACE" 2>/dev/null || true
        }
      else
        echo "[disc-api] udhcpc assente → fallback 10.0.0.20/24"
        ip addr add "10.0.0.20/24" dev "$MGMT_IFACE" 2>/dev/null || true
      fi
    fi
  fi
fi

# Default route SOLO se esplicitamente richiesta e con GW reale
if [[ "$ADD_DEFAULT_ROUTE" = "1" && -n "$MGMT_GW" ]]; then
  ip route add default via "$MGMT_GW" dev "$MGMT_IFACE" 2>/dev/null || true
fi

echo "[disc-api] Stato interfacce:"
ip -br addr || true
ip route || true

echo "[disc-api] Avvio uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  --workers "${UVICORN_WORKERS:-1}" --log-level info
