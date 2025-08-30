#!/usr/bin/env bash
set -euo pipefail

# Precedenza IP come sopra:
# 1) HOST_IP + HOST_NETMASK  2) MGMT_IP_CIDR  3) DHCP→fallback

MGMT_IFACE="${MGMT_IFACE:-${IFACE:-eth0}}"
ICS_IFACE="${ICS_IFACE:-eth1}"

MGMT_DHCP="${MGMT_DHCP:-1}"
MGMT_IP_CIDR="${MGMT_IP_CIDR:-}"

HOST_IP="${HOST_IP:-}"
HOST_NETMASK="${HOST_NETMASK:-24}"

ADD_DEFAULT_ROUTE="${ADD_DEFAULT_ROUTE:-0}"
MGMT_GW="${MGMT_GW:-}"

ICS_DHCP="${ICS_DHCP:-0}"
ICS_IP_CIDR="${ICS_IP_CIDR:-}"
ICS_PROMISC="${ICS_PROMISC:-1}"
SET_MTU="${SET_MTU:-}"

wait_if() { local i="$1"; for _ in {1..60}; do ip link show "$i" &>/dev/null && return 0; sleep 0.5; done; return 1; }

echo "[net-agent] Preparazione interfacce…"
wait_if "$MGMT_IFACE" || echo "[net-agent] Avviso: interfaccia gestione '$MGMT_IFACE' non trovata (continuo)"
wait_if "$ICS_IFACE"  || echo "[net-agent] Avviso: interfaccia ICS '$ICS_IFACE' non trovata (continuo)"

ip link set "$MGMT_IFACE" up 2>/dev/null || true
ip link set "$ICS_IFACE"  up 2>/dev/null || true
[[ -n "$SET_MTU" ]] && ip link set dev "$MGMT_IFACE" mtu "$SET_MTU" 2>/dev/null || true
[[ "$ICS_PROMISC" = "1" ]] && ip link set dev "$ICS_IFACE" promisc on 2>/dev/null || true

# --- Gestione (MGMT) ---
if [[ -n "$HOST_IP" ]]; then
  echo "[net-agent] Using HOST_IP=${HOST_IP}/${HOST_NETMASK} on ${MGMT_IFACE}"
  ip addr add "${HOST_IP}/${HOST_NETMASK}" dev "$MGMT_IFACE" 2>/dev/null || true
  MGMT_DHCP=0
  MGMT_IP_CIDR=""
elif [[ -n "$MGMT_IP_CIDR" ]]; then
  echo "[net-agent] Using MGMT_IP_CIDR=${MGMT_IP_CIDR} on ${MGMT_IFACE}"
  ip addr add "$MGMT_IP_CIDR" dev "$MGMT_IFACE" 2>/dev/null || true
  MGMT_DHCP=0
else
  if [[ "$MGMT_DHCP" = "1" ]]; then
    if command -v udhcpc >/dev/null; then
      timeout 12s udhcpc -i "$MGMT_IFACE" -q -s /usr/share/udhcpc/default.script || {
        echo "[net-agent] DHCP KO → fallback 10.0.0.21/24 (NO default route)"
        ip addr add "10.0.0.21/24" dev "$MGMT_IFACE" 2>/dev/null || true
      }
    else
      echo "[net-agent] udhcpc assente → fallback 10.0.0.21/24"
      ip addr add "10.0.0.21/24" dev "$MGMT_IFACE" 2>/dev/null || true
    fi
  fi
fi

# Default route SOLO se richiesta
if [[ "$ADD_DEFAULT_ROUTE" = "1" && -n "$MGMT_GW" ]]; then
  ip route add default via "$MGMT_GW" dev "$MGMT_IFACE" 2>/dev/null || true
fi

# --- ICS (mai default route) ---
if [[ "$ICS_DHCP" = "1" ]]; then
  command -v udhcpc >/dev/null && timeout 12s udhcpc -i "$ICS_IFACE" -q -s /usr/share/udhcpc/default.script || true
elif [[ -n "$ICS_IP_CIDR" ]]; then
  ip addr add "$ICS_IP_CIDR" dev "$ICS_IFACE" 2>/dev/null || true
fi

# --- DISC_URL & app ---
PREDICT_PATH="${PREDICT_PATH:-/predict3}"

# Compat: LOG_PATH → OUT_JSONL_PATH
if [[ -z "${OUT_JSONL_PATH:-}" && -n "${LOG_PATH:-}" ]]; then
  OUT_JSONL_PATH="$LOG_PATH"
fi
export OUT_JSONL_PATH="${OUT_JSONL_PATH:-/data/flows.jsonl}"

export IFACE_ICS="${IFACE_ICS:-$ICS_IFACE}"
export BPF="${BPF:-tcp port 2404}"
export FEATS_PATH="${FEATS_PATH:-/artifacts/features.json}"
export SCALER_PATH="${SCALER_PATH:-/artifacts/scaler.pkl}"

# Se DISC_URL è vuoto e sei in GNS3, fallback su disc-api a 10.0.0.20
if [[ "${GNS3_MODE:-0}" = "1" && -z "${DISC_URL:-}" ]]; then
  DISC_URL="http://10.0.0.20:8000"
fi

case "${DISC_URL:-}" in
  *"$PREDICT_PATH") ;;               # già contiene il path
  */) DISC_URL="${DISC_URL%/}${PREDICT_PATH}" ;;
  "") echo "[net-agent] ERRORE: DISC_URL non impostato"; exit 1 ;;
  *)  DISC_URL="${DISC_URL}${PREDICT_PATH}" ;;
esac
export DISC_URL

echo "[net-agent] IFACE_ICS=$IFACE_ICS BPF='$BPF' DISC_URL=$DISC_URL"
echo "[net-agent] Stato interfacce:"
ip -br addr || true
ip route || true

echo "[net-agent] Avvio servizi…"
exec python -m agent.main
