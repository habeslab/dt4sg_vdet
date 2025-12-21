#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# net-agent entrypoint
#
# Goals:
# - Keep current ICS behavior (never set default route via ICS).
# - Make MGMT configuration robust for GNS3 NAT/Internet node scenarios.
# - Preserve existing env knobs (HOST_IP, HOST_NETMASK, MGMT_IP_CIDR, MGMT_DHCP,
#   ADD_DEFAULT_ROUTE, MGMT_GW, etc.) and only adjust MGMT defaults/fallbacks.
#
# MGMT priority order:
#   1) HOST_IP/HOST_NETMASK  (legacy)
#   2) MGMT_IP_CIDR          (explicit CIDR)
#   3) DHCP (udhcpc if present)
#   4) NAT static fallback (if MGMT_NAT=1): MGMT_NAT_IP_CIDR + MGMT_NAT_GW
#   5) "no config" fallback: keep link up only (no bogus 10.0.0.21 on mgmt)
#
# =============================================================================

MGMT_IFACE="${MGMT_IFACE:-eth0}"
ICS_IFACE="${ICS_IFACE:-eth1}"

# Existing knobs
MGMT_DHCP="${MGMT_DHCP:-1}"

HOST_IP="${HOST_IP:-}"
HOST_NETMASK="${HOST_NETMASK:-24}"

ADD_DEFAULT_ROUTE="${ADD_DEFAULT_ROUTE:-0}"
MGMT_GW="${MGMT_GW:-}"

ICS_DHCP="${ICS_DHCP:-0}"
ICS_IP_CIDR="${ICS_IP_CIDR:-}"
ICS_PROMISC="${ICS_PROMISC:-1}"
SET_MTU="${SET_MTU:-}"

# NAT mgmt knobs (new, safe defaults for GNS3 NAT node)
GNS3_MODE="${GNS3_MODE:-0}"
MGMT_NAT="${MGMT_NAT:-}"
if [[ -z "$MGMT_NAT" ]]; then
  # default: if you are in GNS3_MODE, assume mgmt via NAT node unless overridden
  MGMT_NAT="$([[ "$GNS3_MODE" = "1" ]] && echo "1" || echo "0")"
fi
MGMT_NAT_IP_CIDR="${MGMT_NAT_IP_CIDR:-192.168.122.50/24}"
MGMT_NAT_GW="${MGMT_NAT_GW:-192.168.122.1}"

# Helpers
wait_if() {
  local i="$1"
  for _ in {1..60}; do
    ip link show "$i" &>/dev/null && return 0
    sleep 0.5
  done
  return 1
}

safe_del_default_on_iface() {
  local dev="$1"
  # remove any default route using dev (ignore errors)
  ip route show default 2>/dev/null | awk -v d="$dev" '$0 ~ (" dev " d " ") {print $0}' | while read -r line; do
    ip route del $line 2>/dev/null || true
  done
}

add_default_once() {
  local gw="$1" dev="$2"
  # add/replace default route in a safe way
  if ip route show default 2>/dev/null | grep -q .; then
    # Replace only if requested; do not destroy other setups unless explicitly asked.
    ip route replace default via "$gw" dev "$dev" 2>/dev/null || true
  else
    ip route add default via "$gw" dev "$dev" 2>/dev/null || true
  fi
}

echo "[net-agent] Preparazione interfacce…"
wait_if "$MGMT_IFACE" || echo "[net-agent] Avviso: interfaccia gestione '$MGMT_IFACE' non trovata (continuo)"
wait_if "$ICS_IFACE"  || echo "[net-agent] Avviso: interfaccia ICS '$ICS_IFACE' non trovata (continuo)"

# Bring links up
ip link set "$MGMT_IFACE" up 2>/dev/null || true
ip link set "$ICS_IFACE"  up 2>/dev/null || true

# MTU + promisc
[[ -n "$SET_MTU" ]] && ip link set dev "$MGMT_IFACE" mtu "$SET_MTU" 2>/dev/null || true
[[ "$ICS_PROMISC" = "1" ]] && ip link set dev "$ICS_IFACE" promisc on 2>/dev/null || true

# =============================================================================
# MGMT configuration
# =============================================================================
echo "[net-agent] Config MGMT su ${MGMT_IFACE} (MGMT_NAT=${MGMT_NAT}, MGMT_DHCP=${MGMT_DHCP})"

# Always start clean on MGMT (avoid leftover from previous runs)
# (We do NOT touch ICS.)
ip addr flush dev "$MGMT_IFACE" 2>/dev/null || true
safe_del_default_on_iface "$MGMT_IFACE"

mgmt_configured="0"

echo "[net-agent] MGMT: Using MGMT_IP_CIDR=${MGMT_IP_CIDR} on ${MGMT_IFACE}"
ip addr add "$MGMT_IP_CIDR" dev "$MGMT_IFACE" 2>/dev/null || true
ip addr add "$HOST_IP" dev "$ICS_IFACE" 2>/dev/null || true

ip route add default via 192.168.122.1 2>/dev/null || true
ip route add 10.0.0.0/24 dev "$ICS_IFACE" 2>/dev/null || true

mgmt_configured="1"

# 4) NAT fallback (recommended for GNS3 NAT/Internet node)
if [[ "$mgmt_configured" = "0" && "$MGMT_NAT" = "1" ]]; then
  echo "[net-agent] MGMT: NAT fallback → ${MGMT_NAT_IP_CIDR} (gw ${MGMT_NAT_GW})"
  ip addr add "$MGMT_NAT_IP_CIDR" dev "$MGMT_IFACE" 2>/dev/null || true
  # add default route via NAT gateway (safe)
  add_default_once "$MGMT_NAT_GW" "$MGMT_IFACE"
  mgmt_configured="1"
fi

# 5) Optional default route (only if explicitly requested)
# NOTE: For NAT fallback we already add default route. This is for other setups.
if [[ "$ADD_DEFAULT_ROUTE" = "1" && -n "$MGMT_GW" ]]; then
  echo "[net-agent] MGMT: ADD_DEFAULT_ROUTE=1 → default via ${MGMT_GW} dev ${MGMT_IFACE}"
  add_default_once "$MGMT_GW" "$MGMT_IFACE"
fi

# =============================================================================
# DISC_URL & app env
# =============================================================================
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

# If DISC_URL empty and in GNS3, fallback
if [[ "$GNS3_MODE" = "1" && -z "${DISC_URL:-}" ]]; then
  DISC_URL="http://10.0.0.20:8000"
fi

case "${DISC_URL:-}" in
  *"$PREDICT_PATH") ;;                         # already has path
  */) DISC_URL="${DISC_URL%/}${PREDICT_PATH}" ;;
  "") echo "[net-agent] ERRORE: DISC_URL non impostato"; exit 1 ;;
  *)  DISC_URL="${DISC_URL}${PREDICT_PATH}" ;;
esac
export DISC_URL

echo "[net-agent] IFACE_ICS=$IFACE_ICS BPF='$BPF' DISC_URL=$DISC_URL"
echo "[net-agent] Stato interfacce:"
ip -br addr || true
echo "[net-agent] Rotte:"
ip route || true

echo "[net-agent] Avvio servizi..."
exec python -m agent.main
