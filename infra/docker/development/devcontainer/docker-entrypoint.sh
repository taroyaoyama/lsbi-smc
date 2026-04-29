#!/bin/bash
set -e

echo "[entrypoint] Starting rsyslog (for iptables LOG capture)..."
rsyslogd || echo "[entrypoint] WARNING: rsyslogd failed to start" >&2

if [ "${FIREWALL_ENABLED:-true}" = "true" ]; then
    echo "[entrypoint] Initializing firewall..."
    if /usr/local/bin/init-firewall.sh; then
        echo "[entrypoint] Firewall ready."
    else
        echo "[entrypoint] WARNING: Firewall initialization failed. Starting without firewall." >&2
    fi
else
    echo "[entrypoint] Firewall disabled (FIREWALL_ENABLED=false)."
fi

exec "$@"
